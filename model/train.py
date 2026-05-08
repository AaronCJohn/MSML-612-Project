"""
Pokémon Diffusion Training Script

Supports two modes via --arch:
    baseline    – Unconditional U-Net (no type/style/stage conditioning)
    conditional – Full conditional U-Net with type/style/stage + prev-evo image

Usage:
    python model/train.py --arch conditional
    python model/train.py --arch baseline

All hyperparameters are read from model/config.json. The only CLI argument is --arch {baseline, conditional}.
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm

CONFIG_PATH = Path(__file__).parent / "train_config.json"

# Attribute mappings

TYPE_TO_IDX = {
    "normal": 0, "fire": 1, "water": 2, "electric": 3,
    "grass": 4, "ice": 5, "fighting": 6, "poison": 7,
    "ground": 8, "flying": 9, "psychic": 10, "bug": 11,
    "rock": 12, "ghost": 13, "dragon": 14, "dark": 15,
    "steel": 16, "fairy": 17,
}
STYLE_TO_IDX = {"3d": 0, "sugimori": 1, "sprite": 2}
STAGE_TO_IDX = {"base": 0, "evo 1": 1, "evo 2": 2}
TYPES = list(TYPE_TO_IDX.keys())
STYLES = list(STYLE_TO_IDX.keys())
STAGES = ["base", "evo 1", "evo 2"]
NUM_TYPES = len(TYPE_TO_IDX)
NUM_STYLES = len(STYLE_TO_IDX)
NUM_STAGES = len(STAGE_TO_IDX)

# Model building blocks

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
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
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(h)


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
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))



# Unconditional U-Net (baseline)

class UnconditionalUNet(nn.Module):
    def __init__(
        self, img_channels, time_embed_dim, base_ch, ch_mults,
        num_res_blocks, attn_resolutions, dropout, img_size,
    ):
        super().__init__()
        self.attn_resolutions = set(attn_resolutions)
        emb_dim = time_embed_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
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
                self.ups.append(block)
                ch = out_ch
            if level > 0:
                self.ups.append(nn.ModuleDict({"up": Upsample(ch)}))
                cur_res *= 2

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, img_channels, 3, padding=1)

    def forward(self, x, t):
        emb = self.time_mlp(t)
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

# Conditional U-Net

class ConditionalUNet(nn.Module):
    def __init__(
        self, img_channels, prev_channels, cond_vec_dim, cond_embed_dim,
        base_ch, ch_mults, num_res_blocks, attn_resolutions, dropout, img_size,
    ):
        super().__init__()
        self.attn_resolutions = set(attn_resolutions)
        emb_dim = cond_embed_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_embed_dim),
            nn.Linear(cond_embed_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_vec_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.prev_encoder = nn.Sequential(
            nn.Conv2d(prev_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
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
                self.ups.append(block)
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


# Noise schedule

class NoiseSchedule:
    def __init__(self, timesteps, device, s=0.008):
        self.timesteps = timesteps
        self.device = device
        t = torch.linspace(0, 1, timesteps + 1, device=device)
        ab = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sa = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        s1m = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sa * x_0 + s1m * noise, noise

    # Conditional CFG
    def _run_cfg(self, model, x_t, t_tensor, cond, prev, has_prev, guidance_scale):
        if guidance_scale == 1.0:
            return model(x_t, t_tensor, cond, prev, has_prev)
        null_cond = torch.zeros_like(cond)
        null_prev = torch.zeros_like(prev)
        null_has_prev = torch.zeros_like(has_prev)
        pred_cond = model(x_t, t_tensor, cond, prev, has_prev)
        pred_uncond = model(x_t, t_tensor, null_cond, null_prev, null_has_prev)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def ddim_sample_conditional(
        self, model, shape, cond, prev, has_prev,
        num_steps=100, guidance_scale=3.0, eta=0.0, progress=True,
    ):
        step_idx = torch.linspace(self.timesteps - 1, 0, num_steps + 1, device=self.device).long()
        x = torch.randn(shape, device=self.device)
        rng = range(num_steps)
        if progress:
            rng = tqdm(rng, desc="DDIM")
        for i in rng:
            t = step_idx[i].item()
            tp = step_idx[i + 1].item()
            tt = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps = self._run_cfg(model, x, tt, cond, prev, has_prev, guidance_scale)
            a_t = self.alpha_cumprod[t]
            a_tp = self.alpha_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=self.device)
            x0p = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
            sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp))
            dir_xt = torch.sqrt((1 - a_tp - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_tp) * x0p + dir_xt
            if eta > 0 and tp > 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)

    # Unconditional DDIM
    @torch.no_grad()
    def ddim_sample_unconditional(self, model, shape, num_steps=100, eta=0.0, progress=True):
        step_idx = torch.linspace(self.timesteps - 1, 0, num_steps + 1, device=self.device).long()
        x = torch.randn(shape, device=self.device)
        rng = range(num_steps)
        if progress:
            rng = tqdm(rng, desc="DDIM")
        for i in rng:
            t = step_idx[i].item()
            tp = step_idx[i + 1].item()
            tt = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps = model(x, tt)
            a_t = self.alpha_cumprod[t]
            a_tp = self.alpha_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=self.device)
            x0p = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
            sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp))
            dir_xt = torch.sqrt((1 - a_tp - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_tp) * x0p + dir_xt
            if eta > 0 and tp > 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)


# Exponential Moving Average (EMA)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow, self.backup = {}, {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def to(self, device):
        for n in self.shadow:
            self.shadow[n] = self.shadow[n].to(device)


# Pokemon Dataset Classes

class PokemonUnconditionalDataset(Dataset):

    def __init__(self, data_dir, json_files, img_size):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples = []

        for jf in json_files:
            path = self.data_dir / jf
            if not path.exists():
                print(f"Warning: {path} not found, skipping.")
                continue
            entries = json.load(open(path))
            entries = [e for e in entries if e.get("next_sprite") is not None]
            self.samples.extend(entries)
            print(f"Loaded {len(entries)} samples from {jf}")
        print(f"Total dataset size: {len(self.samples)}")

        self.rgb_jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)
        self.transform_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.transform_alpha = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_image(self, rel_path, flip=False):
        img = Image.open(self.data_dir / rel_path).convert("RGBA")
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        alpha = img.getchannel("A")
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img.convert("RGB"), mask=alpha)
        rgb = self.rgb_jitter(white_bg)
        rgb_tensor = self.transform_rgb(rgb) * 2 - 1
        alpha_tensor = self.transform_alpha(alpha)
        return rgb_tensor, alpha_tensor

    def __getitem__(self, idx):
        flipped = idx >= len(self.samples)
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        rgb_target, alpha_target = self._load_image(s["next_sprite"], flip=flipped)
        fg_mask = (alpha_target > 0.0).float()
        return {"target": rgb_target, "fg_mask": fg_mask}

    def __len__(self):
        return len(self.samples)


class PokemonConditionalDataset(Dataset):
    """Full conditional dataset: target + prev_evo + (types, stage, style)."""

    def __init__(self, data_dir, json_files, img_size, cfg_dropout=0.05):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.cfg_dropout = cfg_dropout
        self.samples = []

        for jf in json_files:
            path = self.data_dir / jf
            if not path.exists():
                print(f"Warning: {path} not found, skipping.")
                continue
            entries = json.load(open(path))
            entries = [e for e in entries if e.get("next_sprite") is not None]
            for e in entries:
                e["_source"] = jf
            self.samples.extend(entries)
            print(f"Loaded {len(entries)} samples from {jf}")
        print(f"Total dataset size: {len(self.samples)}")

        self.rgb_jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)
        self.transform_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.transform_alpha = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_image(self, rel_path, flip=False):
        img = Image.open(self.data_dir / rel_path).convert("RGBA")
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        alpha = img.getchannel("A")
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img.convert("RGB"), mask=alpha)
        rgb = self.rgb_jitter(white_bg)
        rgb_tensor = self.transform_rgb(rgb) * 2 - 1
        alpha_tensor = self.transform_alpha(alpha)
        return rgb_tensor, alpha_tensor

    @staticmethod
    def _encode_type(type_val):
        vec = torch.zeros(NUM_TYPES)
        if isinstance(type_val, str):
            type_val = [type_val]
        for t in type_val:
            k = t.lower().strip()
            if k in TYPE_TO_IDX:
                vec[TYPE_TO_IDX[k]] = 1.0
        return vec

    @staticmethod
    def _encode_style(style_val):
        vec = torch.zeros(NUM_STYLES)
        vec[STYLE_TO_IDX.get(style_val.lower().strip(), 0)] = 1.0
        return vec

    @staticmethod
    def _encode_stage(stage_val):
        vec = torch.zeros(NUM_STAGES)
        key = stage_val.lower().strip() if isinstance(stage_val, str) else stage_val
        vec[STAGE_TO_IDX.get(key, 0)] = 1.0
        return vec

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        flip = random.random() < 0.5

        rgb_target, alpha_target = self._load_image(s["next_sprite"], flip=flip)
        fg_mask = (alpha_target > 0.0).float()

        if s.get("prev_sprite") is not None:
            rgb_prev, _ = self._load_image(s["prev_sprite"], flip=flip)
            prev_evo = rgb_prev
            has_prev = torch.tensor(1.0)
        else:
            prev_evo = torch.zeros(3, self.img_size, self.img_size)
            has_prev = torch.tensor(0.0)

        cond_vec = torch.cat([
            self._encode_type(s["types"]),
            self._encode_style(s["art_style"]),
            self._encode_stage(s["evolution_stage"]),
        ])
        return {
            "target": rgb_target,
            "fg_mask": fg_mask,
            "prev_evo": prev_evo,
            "has_prev": has_prev,
            "cond_vec": cond_vec,
        }

    def __len__(self):
        return len(self.samples)

# Sampling helpers (for mid-training previews)

def tensor_to_pil(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).convert("RGB")


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
    stage_vec[STAGE_TO_IDX.get(stage.lower().strip(), 0)] = 1.0
    return torch.cat([type_vec, style_vec, stage_vec])


@torch.no_grad()
def save_baseline_grid(model, ema, noise_schedule, epoch, args):
    ema.apply_shadow()
    try:
        model.eval()
        n = 8
        shape = (n, args.img_channels, args.img_size, args.img_size)
        gen = noise_schedule.ddim_sample_unconditional(model, shape, num_steps=args.ddim_steps, progress=False)
        gen = (gen + 1) / 2
        cols = 4
        rows = (n + cols - 1) // cols
        grid = Image.new("RGB", (cols * args.img_size, rows * args.img_size), (255, 255, 255))
        for idx in range(n):
            r, c = divmod(idx, cols)
            grid.paste(tensor_to_pil(gen[idx]), (c * args.img_size, r * args.img_size))
        path = os.path.join(args.output_dir, f"samples_epoch_{epoch}.png")
        grid.save(path)
        print(f"  Saved samples to {path}")
    finally:
        ema.restore()


@torch.no_grad()
def save_conditional_grid(model, ema, noise_schedule, epoch, args):
    configs = [
        {"type": "fire",              "style": "3d",       "stage": "base"},
        {"type": ["grass", "poison"], "style": "sugimori", "stage": "evo 1"},
        {"type": "water",             "style": "3d",       "stage": "base"},
        {"type": "fire",              "style": "sprite",   "stage": "base"},
    ]
    samples_per_cfg = 2
    ema.apply_shadow()
    try:
        model.eval()
        all_imgs = []
        for cfg in configs:
            cfg_types = cfg["type"] if isinstance(cfg["type"], list) else [cfg["type"]]
            cond = make_cond_vec(cfg_types, cfg["style"], cfg["stage"])
            cond = cond.unsqueeze(0).expand(samples_per_cfg, -1).to(args.device)
            prev = torch.zeros(samples_per_cfg, args.img_channels, args.img_size, args.img_size, device=args.device)
            has_prev = torch.zeros(samples_per_cfg, device=args.device)
            shape = (samples_per_cfg, args.img_channels, args.img_size, args.img_size)
            gen = noise_schedule.ddim_sample_conditional(
                model, shape, cond, prev, has_prev,
                num_steps=args.ddim_steps, guidance_scale=args.guidance_scale, progress=False,
            )
            gen = (gen + 1) / 2
            for i in range(gen.shape[0]):
                all_imgs.append(tensor_to_pil(gen[i]))

        cols = samples_per_cfg
        rows = len(configs)
        grid = Image.new("RGB", (cols * args.img_size, rows * args.img_size), (255, 255, 255))
        for idx, img in enumerate(all_imgs):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * args.img_size, r * args.img_size))
        path = os.path.join(args.output_dir, f"samples_epoch_{epoch}.png")
        grid.save(path)
        print(f"  Saved samples to {path}")
    finally:
        ema.restore()


# Training loops

def train_baseline(args):
    dataset = PokemonUnconditionalDataset(args.data_dir, args.json_files, args.img_size)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = UnconditionalUNet(
        img_channels=args.img_channels, time_embed_dim=128,
        base_ch=args.base_ch, ch_mults=args.ch_mults,
        num_res_blocks=args.num_res_blocks, attn_resolutions=args.attn_resolutions,
        dropout=args.dropout, img_size=args.img_size,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_schedule = NoiseSchedule(args.timesteps, device=args.device)
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler(args.device.type, enabled=args.amp_enabled)

    start_epoch, best_loss = 0, float("inf")
    loss_history = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]
            ema.to(args.device)
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "loss_history" in ckpt:
            loss_history = ckpt["loss_history"]
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, loss: {best_loss:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - start_epoch)
    )

    print(f"Training BASELINE on {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss, n_batches = 0.0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            target = batch["target"].to(args.device, non_blocking=True)
            fg_mask = batch["fg_mask"].to(args.device, non_blocking=True)
            B = target.shape[0]

            t = torch.randint(0, args.timesteps, (B,), device=args.device)
            x_t, noise = noise_schedule.add_noise(target, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(args.device.type, enabled=args.amp_enabled):
                pred_noise = model(x_t, t)
                weight_map = (args.fg_weight_low + (args.fg_weight_high - args.fg_weight_low) * fg_mask)
                weight_map = weight_map.expand(-1, pred_noise.shape[1], -1, -1)
                sq_err = (pred_noise - noise) ** 2
                loss = (sq_err * weight_map).sum() / weight_map.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update()
            ep_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = ep_loss / max(1, n_batches)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

        if scheduler.get_last_lr()[0] <= 1e-7:
            print(f"  LR exhausted at epoch {epoch+1}, warm restarting at {args.lr}")
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch - 1
            )

        payload = {
            "epoch": epoch + 1,
            "arch": "baseline",
            "model_state_dict": model.state_dict(),
            "ema_shadow": ema.shadow,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": avg_loss,
            "loss_history": loss_history,
        }
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(payload, os.path.join(args.checkpoint_dir, "ckpt_best.pt"))
            print(f"  New best loss: {best_loss:.4f}")
        torch.save(payload, os.path.join(args.checkpoint_dir, "ckpt_latest.pt"))

        if (epoch + 1) % args.sample_every == 0:
            save_baseline_grid(model, ema, noise_schedule, epoch + 1, args)
            torch.save(payload, os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch+1}.pt"))

    return model, ema


def train_conditional(args):
    dataset = PokemonConditionalDataset(args.data_dir, args.json_files, args.img_size, args.cfg_dropout)

    style_counts = Counter(s["art_style"] for s in dataset.samples)
    print(f"Style counts: {dict(style_counts)}")
    style_weights = {style: 1.0 / cnt for style, cnt in style_counts.items()}
    sample_weights = [style_weights[s["art_style"]] for s in dataset.samples]
    weights = torch.tensor(sample_weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    cond_vec_dim = NUM_TYPES + NUM_STYLES + NUM_STAGES
    model = ConditionalUNet(
        img_channels=args.img_channels, prev_channels=args.img_channels,
        cond_vec_dim=cond_vec_dim, cond_embed_dim=args.cond_embed_dim,
        base_ch=args.base_ch, ch_mults=args.ch_mults,
        num_res_blocks=args.num_res_blocks, attn_resolutions=args.attn_resolutions,
        dropout=args.dropout, img_size=args.img_size,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_schedule = NoiseSchedule(args.timesteps, device=args.device)
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler(args.device.type, enabled=args.amp_enabled)

    start_epoch, best_loss = 0, float("inf")
    loss_history = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]
            ema.to(args.device)
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "loss_history" in ckpt:
            loss_history = ckpt["loss_history"]
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, loss: {best_loss:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - start_epoch)
    )

    print(f"Training CONDITIONAL on {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss, n_batches = 0.0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            target = batch["target"].to(args.device, non_blocking=True)
            prev_evo = batch["prev_evo"].to(args.device, non_blocking=True)
            has_prev = batch["has_prev"].to(args.device, non_blocking=True)
            cond_vec = batch["cond_vec"].to(args.device, non_blocking=True)
            B = target.shape[0]

            drop_mask = (torch.rand(B, device=args.device) < args.cfg_dropout).float()
            keep = 1.0 - drop_mask
            cond_vec = cond_vec * keep.unsqueeze(1)
            has_prev = has_prev * keep
            prev_evo = prev_evo * keep.view(B, 1, 1, 1)

            t = torch.randint(0, args.timesteps, (B,), device=args.device)
            x_t, noise = noise_schedule.add_noise(target, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(args.device.type, enabled=args.amp_enabled):
                pred_noise = model(x_t, t, cond_vec, prev_evo, has_prev)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update()
            ep_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = ep_loss / max(1, n_batches)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

        payload = {
            "epoch": epoch + 1,
            "arch": "conditional",
            "model_state_dict": model.state_dict(),
            "ema_shadow": ema.shadow,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": avg_loss,
            "loss_history": loss_history,
        }
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(payload, os.path.join(args.checkpoint_dir, "ckpt_best.pt"))
            print(f"  New best loss: {best_loss:.4f}")
        torch.save(payload, os.path.join(args.checkpoint_dir, "ckpt_epoch_latest.pt"))

        if (epoch + 1) % args.sample_every == 0:
            save_conditional_grid(model, ema, noise_schedule, epoch + 1, args)
            torch.save(payload, os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch+1}.pt"))

    return model, ema

# Config loading & CLI

def load_config(arch: str) -> SimpleNamespace:
    """Load the config for *arch* from config.json and return a SimpleNamespace."""
    with open(CONFIG_PATH) as f:
        all_cfg = json.load(f)
    if arch not in all_cfg:
        raise ValueError(f"Unknown arch '{arch}'. config.json has: {list(all_cfg.keys())}")
    cfg = all_cfg[arch]
    cfg["arch"] = arch
    cfg["ch_mults"] = tuple(cfg["ch_mults"])
    cfg["attn_resolutions"] = tuple(cfg["attn_resolutions"])
    return SimpleNamespace(**cfg)


def main(arch=None):
    if arch is None:
        p = argparse.ArgumentParser(
            description="Train Pokémon diffusion models (params from config.json)"
        )
        p.add_argument(
            "--arch", choices=["baseline", "conditional"], default="conditional",
            help="Model architecture: 'baseline' (unconditional) or 'conditional'",
        )
        cli = p.parse_args()
        arch = cli.arch
    args = load_config(arch)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    args.is_cuda = args.device.type == "cuda"
    args.pin_memory = args.is_cuda
    args.amp_enabled = args.use_amp and args.is_cuda
    print(f"Device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Architecture: {args.arch}")
    print(f"Data dir:     {args.data_dir}")
    print(f"JSON files:   {args.json_files}")
    print(f"Image size:   {args.img_size}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Base CH:      {args.base_ch}, Mults: {args.ch_mults}")
    print(f"Attn res:     {args.attn_resolutions}")
    print()

    if args.arch == "baseline":
        train_baseline(args)
    else:
        train_conditional(args)


if __name__ == "__main__":
    main()
