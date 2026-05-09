"""
Checkpoint Evaluation: FID / KID / ID Distance

For each checkpoint group configured in model/eval_config.json, this script:

1. Loads the U-Net + EMA shadow weights for every *.pt file in the group's directory.
2. Generates N samples via DDIM (null conditioning for conditional models).
3. Extracts Inception V3 pool-layer features for both real and generated images.
4. Computes FID, KID, and ID Distance (intra-generated diversity).
5. Writes metrics.json, metrics.csv, and metrics.png per group to the output directory.

Usage:
    python eval.py                                      # all groups
    python eval.py --group all_styles                   # one group
    python eval.py --group baseline_3d baseline_sprite  # specific groups
    python eval.py --epochs 300 450                     # only these epochs
"""

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm.auto import tqdm

CONFIG_PATH = Path(__file__).resolve().parent / "model" / "eval_config.json"
EPOCH_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


# U-Net building blocks

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
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


# Unconditional U-Net

class UnconditionalUNet(nn.Module):
    def __init__(self, img_channels, time_embed_dim, base_ch, ch_mults, num_res_blocks, attn_resolutions, dropout, img_size):
        super().__init__()
        self.img_size = img_size
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
                self.downs.append(nn.ModuleDict({
                    "res":  ResBlock(ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                }))
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
            for i in range(num_res_blocks + (1 if level > 0 else 0)):
                skip_ch = skip_channels.pop()
                self.ups.append(nn.ModuleDict({
                    "res":  ResBlock(ch + skip_ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                }))
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
    def __init__(self, img_channels, prev_channels, cond_vec_dim, cond_embed_dim, base_ch, ch_mults, num_res_blocks, attn_resolutions, dropout, img_size):
        super().__init__()
        self.img_size = img_size
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
            nn.Conv2d(32, 64, 4, 2, 1),            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          nn.SiLU(),
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
                self.downs.append(nn.ModuleDict({
                    "res":  ResBlock(ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                }))
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
            for i in range(num_res_blocks + (1 if level > 0 else 0)):
                skip_ch = skip_channels.pop()
                self.ups.append(nn.ModuleDict({
                    "res":  ResBlock(ch + skip_ch, out_ch, emb_dim, dropout),
                    "attn": SelfAttention(out_ch) if cur_res in self.attn_resolutions else nn.Identity(),
                }))
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


# Noise schedule (cosine)

class NoiseSchedule:
    def __init__(self, timesteps, device, s=0.008):
        self.timesteps = timesteps
        self.device = device
        t = torch.linspace(0, 1, timesteps + 1, device=device)
        ab = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def ddim_sample_uncond(self, model_fn, shape, num_steps, eta=0.0):
        step_idx = torch.linspace(
            self.timesteps - 1, 0, num_steps + 1, device=self.device
        ).long()
        x = torch.randn(shape, device=self.device)
        for i in range(num_steps):
            t = step_idx[i].item()
            tp = step_idx[i + 1].item()
            tt = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps = model_fn(x, tt)
            a_t = self.alpha_cumprod[t]
            a_tp = (
                self.alpha_cumprod[tp]
                if tp >= 0
                else torch.tensor(1.0, device=self.device)
            )
            x0p = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
            sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp))
            dir_xt = torch.sqrt((1 - a_tp - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_tp) * x0p + dir_xt
        return x.clamp(-1, 1)


# Inception V3 feature extractor (2048-dim pool)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.layers = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x * 2 - 1
        return self.layers(x)


@torch.no_grad()
def extract_features(imgs, extractor, device, batch_size=32, desc="features"):
    feats = []
    for i in tqdm(range(0, len(imgs), batch_size), desc=desc, leave=False):
        batch = imgs[i : i + batch_size].to(device)
        feats.append(extractor(batch).cpu())
    return torch.cat(feats).numpy()


# Real image loading

def _resolve_sprite_path(data_dir: Path, rel: str):
    rel_path = Path(rel)
    candidates = [
        data_dir / rel_path,
        data_dir.parent / rel_path,
        Path.cwd() / rel_path,
        rel_path,
    ]
    if rel_path.parts and rel_path.parts[0] == data_dir.name:
        candidates.insert(0, data_dir / Path(*rel_path.parts[1:]))
    for c in candidates:
        if c.exists():
            return c
    return None


def load_real_images(data_dir, json_files, img_size, max_n):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    data_dir = Path(data_dir)
    imgs, missing = [], 0
    for jf in json_files:
        path = data_dir / jf
        if not path.exists():
            print(f"  Warning: {path} not found")
            continue
        for entry in json.load(open(path)):
            if entry.get("next_sprite") is None:
                continue
            sprite_path = _resolve_sprite_path(data_dir, entry["next_sprite"])
            if sprite_path is None:
                missing += 1
                continue
            try:
                img = Image.open(sprite_path).convert("RGBA")
                white = Image.new("RGB", img.size, (255, 255, 255))
                white.paste(img.convert("RGB"), mask=img.getchannel("A"))
                imgs.append(tfm(white))
            except Exception:
                continue
            if len(imgs) >= max_n:
                break
        if len(imgs) >= max_n:
            break
    if missing:
        print(f"  Warning: {missing} sprite paths could not be resolved on disk")
    if not imgs:
        raise RuntimeError(
            f"No real images loaded from {data_dir} using {json_files}. "
            f"Check data_dir and that 'next_sprite' paths in the JSONs resolve."
        )
    return torch.stack(imgs)


# DDIM generation

@torch.no_grad()
def generate_samples(model, ema_shadow, noise_schedule, cfg, device):
    backup = {}
    if ema_shadow is not None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in ema_shadow:
                backup[n] = p.data.clone()
                p.data.copy_(ema_shadow[n].to(device))

    model.eval()
    try:
        if cfg.is_conditional:
            null_cond_dim = cfg.num_types + cfg.num_styles + cfg.num_stages

            def model_fn(x_t, tt):
                B = x_t.shape[0]
                cond = torch.zeros(B, null_cond_dim, device=device)
                prev_evo = torch.zeros(
                    B, cfg.img_channels, cfg.img_size, cfg.img_size, device=device
                )
                has_prev = torch.zeros(B, device=device)
                return model(x_t, tt, cond, prev_evo, has_prev)
        else:
            def model_fn(x_t, tt):
                return model(x_t, tt)

        all_imgs, produced = [], 0
        pbar = tqdm(total=cfg.n_samples, desc="generating", leave=False)
        while produced < cfg.n_samples:
            bs = min(cfg.batch_size, cfg.n_samples - produced)
            shape = (bs, cfg.img_channels, cfg.img_size, cfg.img_size)
            x = noise_schedule.ddim_sample_uncond(model_fn, shape, cfg.ddim_steps)
            all_imgs.append(((x + 1) / 2).clamp(0, 1).cpu())
            produced += bs
            pbar.update(bs)
        pbar.close()
        return torch.cat(all_imgs)[: cfg.n_samples]
    finally:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# Metrics

def compute_fid(real_feats, gen_feats):
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


def polynomial_kernel(x, y, degree=3, gamma=None, coef=1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef) ** degree


def compute_kid(real_feats, gen_feats, n_subsets=100, subset_size=1000):
    subset_size = min(subset_size, len(real_feats), len(gen_feats))
    if subset_size < 10:
        return float("nan"), float("nan")
    scores = []
    for _ in range(n_subsets):
        ri = np.random.choice(len(real_feats), subset_size, replace=False)
        gi = np.random.choice(len(gen_feats), subset_size, replace=False)
        r, g = real_feats[ri], gen_feats[gi]
        k_rr = polynomial_kernel(r, r)
        k_gg = polynomial_kernel(g, g)
        k_rg = polynomial_kernel(r, g)
        n = subset_size
        mmd = (
            (np.sum(k_rr) - np.trace(k_rr)) / (n * (n - 1))
            + (np.sum(k_gg) - np.trace(k_gg)) / (n * (n - 1))
            - 2 * k_rg.mean()
        )
        scores.append(mmd)
    return float(np.mean(scores)), float(np.std(scores))


def compute_id_distance(gen_feats, max_n=500):
    feats = torch.tensor(gen_feats)
    if len(feats) > max_n:
        idx = torch.randperm(len(feats))[:max_n]
        feats = feats[idx]
    dists = torch.cdist(feats, feats)
    n = len(feats)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    return float(dists[mask].mean())


# Checkpoint discovery & loading

def find_checkpoints(ckpt_dir, requested_epochs=None):
    ckpt_dir = Path(ckpt_dir)
    found = {}
    for p in ckpt_dir.glob("ckpt_epoch_*.pt"):
        m = EPOCH_RE.search(p.name)
        if m:
            found[int(m.group(1))] = p
    if requested_epochs is not None:
        return [(e, found[e]) for e in sorted(requested_epochs) if e in found]
    return sorted(found.items())


def load_checkpoint(ckpt_path, cfg, device):
    if cfg.is_conditional:
        cond_vec_dim = cfg.num_types + cfg.num_styles + cfg.num_stages
        model = ConditionalUNet(
            img_channels=cfg.img_channels,
            prev_channels=cfg.img_channels,
            cond_vec_dim=cond_vec_dim,
            cond_embed_dim=cfg.time_embed_dim,
            base_ch=cfg.base_ch,
            ch_mults=tuple(cfg.ch_mults),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            dropout=cfg.dropout,
            img_size=cfg.img_size,
        ).to(device)
    else:
        model = UnconditionalUNet(
            img_channels=cfg.img_channels,
            time_embed_dim=cfg.time_embed_dim,
            base_ch=cfg.base_ch,
            ch_mults=tuple(cfg.ch_mults),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
            dropout=cfg.dropout,
            img_size=cfg.img_size,
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("ema_shadow", None), ckpt.get("loss", None)


# Results I/O

def save_csv(results, out_dir):
    csv_path = Path(out_dir) / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,fid,kid_mean,kid_std,id_distance,train_loss\n")
        for r in results:
            loss = r["train_loss"] if r["train_loss"] is not None else ""
            f.write(
                f"{r['epoch']},{r['fid']:.4f},{r['kid_mean']:.6f},"
                f"{r['kid_std']:.6f},{r['id_distance']:.4f},{loss}\n"
            )
    print(f"Saved {csv_path}")


def save_plot(results, out_dir):
    if len(results) < 1:
        return
    epochs = [r["epoch"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [r["fid"] for r in results], marker="o")
    axes[0].set_title("FID \u2193")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)

    axes[1].plot(epochs, [r["kid_mean"] for r in results], marker="o")
    axes[1].fill_between(
        epochs,
        [r["kid_mean"] - r["kid_std"] for r in results],
        [r["kid_mean"] + r["kid_std"] for r in results],
        alpha=0.2,
    )
    axes[1].set_title("KID \u2193")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    axes[2].plot(epochs, [r["id_distance"] for r in results], marker="o")
    axes[2].set_title("ID Distance \u2191 (diversity)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = Path(out_dir) / "metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")


def print_summary(results):
    if not results:
        return
    best_fid = min(results, key=lambda r: r["fid"])
    best_kid = min(results, key=lambda r: r["kid_mean"])
    most_div = max(results, key=lambda r: r["id_distance"])
    print("--- Summary ---")
    print(f"  Best FID:  {best_fid['fid']:.2f}  (epoch {best_fid['epoch']})")
    print(f"  Best KID:  {best_kid['kid_mean']:.4f}  (epoch {best_kid['epoch']})")
    print(f"  Most diverse (ID): {most_div['id_distance']:.4f}  (epoch {most_div['epoch']})")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion checkpoints (FID / KID / ID Distance)",
    )
    parser.add_argument(
        "--group", nargs="+", default=None,
        help="Checkpoint group(s) to evaluate (default: all groups in config)",
    )
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=None,
        help="Only evaluate these epochs (default: all found in each group)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)

    all_groups = raw_cfg.pop("checkpoint_groups")
    if args.group:
        unknown = set(args.group) - set(all_groups)
        if unknown:
            parser.error(
                f"Unknown group(s): {unknown}. "
                f"Available: {list(all_groups.keys())}"
            )
        groups = {k: all_groups[k] for k in args.group}
    else:
        groups = all_groups

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Device: cuda | {torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: mps")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    print("Loading Inception V3 ...")
    extractor = InceptionFeatureExtractor().to(device)

    json_files = [f"{s}.json" for s in raw_cfg["styles"]]
    print(f"Loading real images (styles: {raw_cfg['styles']}) ...")
    real_imgs = load_real_images(
        raw_cfg["data_dir"], json_files, raw_cfg["img_size"], raw_cfg["n_samples"],
    )
    print(f"  {len(real_imgs)} real images loaded")
    real_feats = extract_features(
        real_imgs, extractor, device, raw_cfg["batch_size"], desc="real features",
    )
    del real_imgs

    for group_name, group_overrides in groups.items():
        print(f"\n")
        print(f"Group: {group_name}")

        merged = {**raw_cfg, **group_overrides}
        cfg = SimpleNamespace(**merged)

        ckpts = find_checkpoints(cfg.checkpoint_dir, args.epochs)
        if not ckpts:
            print("No checkpoints found, skipping")
            continue

        print(f"Checkpoints: {[e for e, _ in ckpts]}")
        print(f"Conditional: {cfg.is_conditional}")

        noise_schedule = NoiseSchedule(cfg.timesteps, device)
        out_dir = Path(cfg.output_dir) / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for epoch, ckpt_path in ckpts:
            print(f"\n  Epoch {epoch}")
            try:
                model, ema_shadow, train_loss = load_checkpoint(ckpt_path, cfg, device)
            except Exception as e:
                print(f"  Failed to load: {e}")
                continue

            gen_imgs = generate_samples(model, ema_shadow, noise_schedule, cfg, device)
            gen_feats = extract_features(
                gen_imgs, extractor, device, cfg.batch_size, desc="gen features",
            )
            del gen_imgs

            fid = compute_fid(real_feats, gen_feats)
            kid_mean, kid_std = compute_kid(real_feats, gen_feats)
            id_dist = compute_id_distance(gen_feats)

            print(f"FID: {fid:.2f} | KID: {kid_mean:.4f} \u00b1 {kid_std:.4f} | ID: {id_dist:.4f}")
            if train_loss is not None:
                print(f"(training loss: {train_loss:.4f})")

            results.append({
                "epoch": epoch,
                "fid": fid,
                "kid_mean": kid_mean,
                "kid_std": kid_std,
                "id_distance": id_dist,
                "train_loss": train_loss,
            })

            json.dump(results, open(out_dir / "metrics.json", "w"), indent=2)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        save_csv(results, out_dir)
        save_plot(results, out_dir)
        print_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
