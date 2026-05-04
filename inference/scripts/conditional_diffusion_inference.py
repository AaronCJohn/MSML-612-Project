#!/usr/bin/env python3
"""
Batch inference for the **type-boost** conditional Pokémon diffusion model
(`pokemon_diffusion_type_boost_updated.ipynb`): 128×128 RGB, COND_EMBED_DIM=256,
attention at (32, 16, 8), and Conv2d + scaled_dot_product_attention blocks.

How to run (pick folder + epoch, or a single file)
-------------------------------------------------
  From the **repository root** (paths below are relative to that root):

  # Folder that contains ckpt_epoch_<N>.pt — quote paths with & or spaces
  python inference/scripts/conditional_diffusion_inference.py \\
    --ckpt-dir "inference/scripts/checkpoints/sprite&3d" --epoch 500 \\
    --styles 3d,sprite \\
    --output-dir inference/outputs/my_run

  # Or point directly at one checkpoint file
  python inference/scripts/conditional_diffusion_inference.py \\
    --ckpt-file inference/scripts/checkpoints/sprite/ckpt_epoch_400.pt \\
    --styles sprite

  Defaults: checkpoints live in inference/scripts/checkpoints/; PNGs go under
  inference/outputs/ unless you pass --output-dir.

Styles are comma-separated: 3d, sugimori, sprite (subset is fine for style-specific runs).

Optional: add --defaults to also sweep every checkpoint under --checkpoint-root
(missing files are skipped unless --strict).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Hyperparameters — must match `pokemon_diffusion_type_boost_updated.ipynb`
# ---------------------------------------------------------------------------
IMG_SIZE = 128
IMG_CHANNELS = 3
NUM_TYPES = 18
NUM_STYLES = 3
NUM_STAGES = 3
COND_VEC_DIM = NUM_TYPES + NUM_STYLES + NUM_STAGES
COND_EMBED_DIM = 256
TIMESTEPS = 1000
EMA_DECAY = 0.999
BASE_CH = 64
CH_MULTS = (1, 2, 2, 4)
NUM_RES_BLOCKS = 2
# Default for newly exported runs; older checkpoints in this repo use (16, 8) — see _infer_attn_resolutions.
ATTN_RESOLUTIONS = (32, 16, 8)
DROPOUT = 0.1

TYPE_TO_IDX = {
    "normal": 0, "fire": 1, "water": 2, "electric": 3,
    "grass": 4, "ice": 5, "fighting": 6, "poison": 7,
    "ground": 8, "flying": 9, "psychic": 10, "bug": 11,
    "rock": 12, "ghost": 13, "dragon": 14, "dark": 15,
    "steel": 16, "fairy": 17,
}
STYLE_TO_IDX = {"3d": 0, "sugimori": 1, "sprite": 2}
STAGE_TO_IDX = {"base": 0, "evo 1": 1, "evo 2": 2}

STYLES_ALL = ("3d", "sugimori", "sprite")
STYLES_3D_SPRITE = ("3d", "sprite")
STYLES_SPRITE = ("sprite",)
STYLES_3D = ("3d",)


# ---------------------------------------------------------------------------
# Noise schedule (unchanged vs notebook Cell 6)
# ---------------------------------------------------------------------------
class NoiseSchedule:
    def __init__(self, timesteps=1000, device="cpu", s=0.008):
        self.timesteps = timesteps
        self.device = device

        t = torch.linspace(0, 1, timesteps + 1, device=device)
        alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1mab * noise, noise

    def _apply_step(self, x_t, t, pred_noise):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1mab_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1)
        mean = sqrt_recip_t * (x_t - betas_t / sqrt_1mab_t * pred_noise)
        if t == 0:
            return mean
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    def _run_cfg(self, model, x_t, t_tensor, cond, prev, has_prev, guidance_scale):
        if guidance_scale == 1.0:
            return model(x_t, t_tensor, cond, prev, has_prev)
        cond_null = torch.zeros_like(cond)
        prev_null = torch.zeros_like(prev)
        has_prev_null = torch.zeros_like(has_prev)
        pred_cond = model(x_t, t_tensor, cond, prev, has_prev)
        pred_uncond = model(x_t, t_tensor, cond_null, prev_null, has_prev_null)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, cond, prev, has_prev, guidance_scale=1.0, progress=False):
        from tqdm.auto import tqdm

        x = torch.randn(shape, device=self.device)
        it = reversed(range(self.timesteps))
        if progress:
            it = tqdm(it, total=self.timesteps, desc="DDPM")
        for t in it:
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = self._run_cfg(model, x, t_tensor, cond, prev, has_prev, guidance_scale)
            x = self._apply_step(x, t, pred_noise)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, prev, has_prev, num_steps=50,
                    guidance_scale=1.0, eta=0.0, progress=False):
        from tqdm.auto import tqdm

        step_indices = torch.linspace(self.timesteps - 1, 0, num_steps + 1, device=self.device).long()
        x = torch.randn(shape, device=self.device)
        rng = range(num_steps)
        if progress:
            rng = tqdm(rng, desc="DDIM")
        for i in rng:
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item()
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps = self._run_cfg(model, x, t_tensor, cond, prev, has_prev, guidance_scale)

            a_t = self.alpha_cumprod[t]
            a_tp = self.alpha_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp))
            dir_xt = torch.sqrt((1 - a_tp - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_tp) * x0_pred + dir_xt
            if eta > 0 and t_prev > 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# U-Net (Cell 7–8, type-boost notebook)
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        hh = self.norm(x)
        qkv = self.qkv(hh).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        hh = F.scaled_dot_product_attention(q, k, v)
        hh = hh.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(hh)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, prev_channels=IMG_CHANNELS, cond_vec_dim=COND_VEC_DIM,
                 cond_embed_dim=COND_EMBED_DIM, base_ch=BASE_CH, ch_mults=CH_MULTS,
                 num_res_blocks=NUM_RES_BLOCKS, attn_resolutions=ATTN_RESOLUTIONS, dropout=DROPOUT, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.num_res_blocks = num_res_blocks
        self.ch_mults = ch_mults
        self.attn_resolutions = set(attn_resolutions)

        emb_dim = cond_embed_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_embed_dim),
            nn.Linear(cond_embed_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_vec_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.prev_encoder = nn.Sequential(
            nn.Conv2d(prev_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, emb_dim),
        )

        self.init_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_ch
        skip_channels = [ch]
        cur_res = img_size

        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleDict({
                    "res": ResBlock(ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                })
                self.downs.append(block)
                ch = out_ch
                skip_channels.append(ch)

            if level < len(ch_mults) - 1:
                self.downs.append(nn.ModuleDict({"down": Downsample(ch)}))
                skip_channels.append(ch)
                cur_res //= 2

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_dim, dropout),
            SelfAttention(ch),
            ResBlock(ch, ch, emb_dim, dropout),
        ])

        self.ups = nn.ModuleList()
        cur_res = img_size // (2 ** (len(ch_mults) - 1))
        ch = base_ch * ch_mults[-1]

        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + (1 if level > 0 else 0)):
                skip_ch = skip_channels.pop()
                block = nn.ModuleDict({
                    "res": ResBlock(ch + skip_ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                })
                self.ups.append(nn.ModuleDict({**block}))
                ch = out_ch

            if level > 0:
                self.ups.append(nn.ModuleDict({"up": Upsample(ch)}))
                cur_res *= 2

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, img_channels, 3, padding=1)

    def forward(self, x, t, cond_vec, prev_evo, has_prev):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond_vec)
        p_emb = self.prev_encoder(prev_evo) * has_prev.unsqueeze(1)
        emb = t_emb + c_emb + p_emb

        h = self.init_conv(x)
        skips = [h]
        for block in self.downs:
            if "down" in block:
                h = block["down"](h)
                skips.append(h)
            else:
                h = block["res"](h, emb)
                if not isinstance(block["attn"], nn.Identity):
                    h = block["attn"](h)
                skips.append(h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)

        for block in self.ups:
            if "up" in block:
                h = block["up"](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block["res"](h, emb)
                if not isinstance(block["attn"], nn.Identity):
                    h = block["attn"](h)

        return self.final_conv(F.silu(self.final_norm(h)))


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow, self.backup = {}, {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)


def make_cond_vec(type_names, style, stage):
    type_vec = torch.zeros(NUM_TYPES)
    if isinstance(type_names, str):
        type_names = [type_names]
    for t in type_names:
        k = t.lower().strip()
        if k in TYPE_TO_IDX:
            type_vec[TYPE_TO_IDX[k]] = 1.0

    style_vec = torch.zeros(NUM_STYLES)
    style_vec[STYLE_TO_IDX.get(style.lower().strip(), 0)] = 1.0

    stage_vec = torch.zeros(NUM_STAGES)
    key = stage.lower().strip() if isinstance(stage, str) else stage
    stage_vec[STAGE_TO_IDX.get(key, 0)] = 1.0

    return torch.cat([type_vec, style_vec, stage_vec])


@torch.no_grad()
def generate(model, noise_schedule, cond_vec, device, prev_evo_image=None,
             num_samples=1, guidance_scale=3.0, use_ddim=True, ddim_steps=100, progress=False):
    model.eval()
    cond = cond_vec.unsqueeze(0).expand(num_samples, -1).to(device)

    if prev_evo_image is not None:
        prev = prev_evo_image.unsqueeze(0).expand(num_samples, -1, -1, -1).to(device)
        has_prev = torch.ones(num_samples, device=device)
    else:
        prev = torch.zeros(num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        has_prev = torch.zeros(num_samples, device=device)

    shape = (num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    if use_ddim:
        x = noise_schedule.ddim_sample(
            model, shape, cond, prev, has_prev,
            num_steps=ddim_steps, guidance_scale=guidance_scale, progress=progress,
        )
    else:
        x = noise_schedule.ddpm_sample(
            model, shape, cond, prev, has_prev,
            guidance_scale=guidance_scale, progress=progress,
        )
    return (x + 1) / 2


def tensor_to_pil(tensor):
    """tensor [0,1], CHW → RGB PIL (matches notebook)."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


# Matches notebook `save_sample_grid` defaults (note spaced "evo 1")
DEFAULT_SAMPLE_CONFIGS = [
    {"type": "fire", "style": "3d", "stage": "base"},
    {"type": ["grass", "poison"], "style": "sugimori", "stage": "evo 1"},
    {"type": "water", "style": "3d", "stage": "base"},
    {"type": "fire", "style": "sprite", "stage": "base"},
]


@dataclass
class InferenceJob:
    """One checkpoint file + which art styles to render from DEFAULT_SAMPLE_CONFIGS."""

    path: Path
    label: str  # e.g. folder name + epoch for output grouping
    styles: tuple[str, ...]


def _default_checkpoint_jobs(checkpoint_root: Path) -> list[InferenceJob]:
    """Jobs aligned with inference/scripts/checkpoints as checked in to the repo."""
    r = checkpoint_root
    jobs: list[InferenceJob] = []

    def add(subdir: str, epoch: int, label: str, styles: tuple[str, ...]):
        p = r / subdir / f"ckpt_epoch_{epoch}.pt"
        jobs.append(InferenceJob(path=p, label=f"{subdir}__epoch_{epoch}", styles=styles))

    add("outputs_all_art_style_w_conditions", 250, "", STYLES_ALL)

    for ep in (400, 500):
        add("outputs_all_art_style_w_conditions", ep, "", STYLES_ALL)

    for ep in (400, 500):
        add("sprite&3d", ep, "", STYLES_3D_SPRITE)

    for ep in (250, 400, 500):
        add("sprite", ep, "", STYLES_SPRITE)

    for ep in (400, 450):
        add("3d", ep, "", STYLES_3D)

    return jobs


def _infer_attn_resolutions(state_dict: dict) -> tuple[int, ...]:
    """Match U-Net layout to checkpoint (two training configs exist in inference/scripts/checkpoints)."""
    keyset = state_dict.keys()
    if any(k.startswith("downs.6.attn.qkv") for k in keyset):
        return (32, 16, 8)
    if any(k.startswith("downs.9.attn.qkv") for k in keyset):
        return (16, 8)
    raise ValueError(
        "Could not infer attention layout from checkpoint weights "
        "(expected downs.6.attn.qkv* for 32,16,8 or downs.9.attn.qkv* for 16,8). "
        "Pass --attn-resolutions 32,16,8 or 16,8 explicitly."
    )


def load_for_inference(
    ckpt_path: str | Path,
    device: torch.device,
    attn_resolutions: tuple[int, ...] | None = None,
):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    if attn_resolutions is None:
        attn_resolutions = _infer_attn_resolutions(sd)
    model = ConditionalUNet(
        img_channels=IMG_CHANNELS, prev_channels=IMG_CHANNELS,
        cond_vec_dim=COND_VEC_DIM, cond_embed_dim=COND_EMBED_DIM,
        base_ch=BASE_CH, ch_mults=CH_MULTS, num_res_blocks=NUM_RES_BLOCKS,
        attn_resolutions=attn_resolutions, dropout=DROPOUT, img_size=IMG_SIZE,
    ).to(device)
    model.load_state_dict(sd)
    ema = EMA(model, decay=EMA_DECAY)
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.to(device)
    ep = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"Loaded {ckpt_path} | epoch={ep} | loss={loss} | attn_res={tuple(attn_resolutions)}")
    return model, ema, NoiseSchedule(TIMESTEPS, device=device)


def run_one_job(
    job: InferenceJob,
    out_root: Path,
    device: torch.device,
    num_samples: int,
    guidance_scale: float,
    ddim_steps: int,
    attn_resolutions: tuple[int, ...] | None = None,
) -> None:
    if not job.path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {job.path}")

    model, ema, noise_schedule = load_for_inference(job.path, device, attn_resolutions)
    subdir = out_root / job.label
    subdir.mkdir(parents=True, exist_ok=True)

    ema.apply_shadow()
    try:
        for cfg in DEFAULT_SAMPLE_CONFIGS:
            style = cfg["style"]
            if style not in job.styles:
                continue
            type_key = cfg["type"]
            type_slug = "_".join(type_key) if isinstance(type_key, list) else type_key
            cond = make_cond_vec(cfg["type"], cfg["style"], cfg["stage"]).to(device)
            gen = generate(
                model, noise_schedule, cond, device,
                prev_evo_image=None,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                use_ddim=True,
                ddim_steps=ddim_steps,
                progress=False,
            )
            for i in range(gen.shape[0]):
                pil = tensor_to_pil(gen[i])
                fname = f"{type_slug}_{cfg['style']}_{cfg['stage'].replace(' ', '_')}_sample{i:02d}.png"
                pil.save(subdir / fname)
    finally:
        ema.restore()

    print(f"Saved under {subdir}/")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    inference_dir = script_dir.parent
    default_root = script_dir / "checkpoints"
    default_output_dir = inference_dir / "outputs" / "inference_batch"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Folder containing ckpt_epoch_<epoch>.pt (use with --epoch)",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Training epoch number N → loads ckpt_epoch_N.pt inside --ckpt-dir",
    )
    p.add_argument(
        "--ckpt-file",
        type=Path,
        default=None,
        help="Full path to a single .pt checkpoint (alternative to --ckpt-dir + --epoch)",
    )
    p.add_argument(
        "--styles",
        type=str,
        default="3d,sugimori,sprite",
        help="Comma-separated art styles to sample (default: all three)",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=default_root,
        help=f"Used only with --defaults: repo of subfolders (default: {default_root})",
    )
    p.add_argument(
        "--defaults",
        action="store_true",
        help="Run the built-in sweep under --checkpoint-root (in addition to any manual runs)",
    )
    p.add_argument(
        "--extra",
        action="append",
        metavar="PATH::STYLES",
        help='Repeatable. Example: --extra ./ckpt_epoch_250.pt::3d,sugimori,sprite',
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Root directory for PNG outputs (default: {default_output_dir})",
    )
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument(
        "--attn-resolutions",
        type=str,
        default=None,
        metavar="R,R,...",
        help="Override U-Net attention resolutions, e.g. 32,16,8 or 16,8 (default: infer from checkpoint)",
    )
    p.add_argument("--strict", action="store_true", help="Fail if any default job checkpoint is missing")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (on Mac, omit this to use MPS GPU when PyTorch supports it)",
    )
    args = p.parse_args()

    has_dir = args.ckpt_dir is not None
    has_ep = args.epoch is not None
    has_file = args.ckpt_file is not None
    if has_dir ^ has_ep:
        p.error("You must pass both --ckpt-dir and --epoch together (or use --ckpt-file instead).")
    if has_dir and has_file:
        p.error("Use either --ckpt-dir/--epoch or --ckpt-file, not both.")
    if not (has_dir or has_file or args.defaults or args.extra):
        p.error(
            "Nothing to run. Choose one:\n"
            "  --ckpt-dir DIR --epoch N [--styles 3d,sprite]\n"
            "  --ckpt-file PATH.pt [--styles ...]\n"
            "  --defaults  (bulk sweep; see --checkpoint-root, default inference/scripts/checkpoints)\n"
            "  --extra PATH::STYLES  (repeatable)\n"
        )
    return args


def _parse_styles(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def pick_device(force_cpu: bool) -> torch.device:
    """CUDA (Linux/Windows) > MPS (Apple Silicon) > CPU."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def main():
    args = parse_args()
    attn_tuple: tuple[int, ...] | None = None
    if args.attn_resolutions is not None:
        attn_tuple = tuple(
            int(x.strip()) for x in args.attn_resolutions.split(",") if x.strip()
        )

    device = pick_device(args.cpu)
    print(f"Device: {device}")

    root = args.checkpoint_root.resolve()
    jobs: list[InferenceJob] = []
    styles = _parse_styles(args.styles)

    if args.defaults:
        jobs.extend(_default_checkpoint_jobs(root))

    if args.ckpt_dir is not None and args.epoch is not None:
        d = args.ckpt_dir.resolve()
        path = d / f"ckpt_epoch_{args.epoch}.pt"
        label = f"{d.name}__epoch_{args.epoch}"
        jobs.append(InferenceJob(path=path, label=label, styles=styles))

    if args.ckpt_file is not None:
        path = args.ckpt_file.resolve()
        label = f"{path.parent.name}__{path.stem}"
        jobs.append(InferenceJob(path=path, label=label, styles=styles))

    if args.extra:
        for spec in args.extra:
            if "::" not in spec:
                print(f"ERROR: --extra must be PATH::STYLES, got: {spec!r}", file=sys.stderr)
                sys.exit(2)
            path_s, styles_s = spec.split("::", 1)
            path_p = Path(path_s.strip()).resolve()
            st = tuple(s.strip() for s in styles_s.split(",") if s.strip())
            jobs.append(InferenceJob(path=path_p, label=path_p.stem, styles=st))

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ran = 0
    for j in jobs:
        print(f"\n=== {j.label} ===\n    {j.path}")
        try:
            run_one_job(
                j,
                out_root=out_root,
                device=device,
                num_samples=args.num_samples,
                guidance_scale=args.guidance_scale,
                ddim_steps=args.ddim_steps,
                attn_resolutions=attn_tuple,
            )
            ran += 1
        except FileNotFoundError as e:
            msg = str(e)
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: {msg}")

    print(f"\nDone. Ran {ran} job(s). Outputs under {out_root}/")


if __name__ == "__main__":
    main()
