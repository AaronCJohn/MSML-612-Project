"""
Standalone Pokémon generation script.
"""

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CHECKPOINT = "checkpoints/all_styles/ckpt_epoch_450.pt"   # path to .pt checkpoint
OUTPUT_DIR = "./generated"                   # where images are saved
SEED       = None                            # set an int for reproducibility

# --- Generation mode ---
# "single" : generate NUM_SAMPLES images for one type/style/stage combo
# "chain"  : evolution chain (base → evo1 → evo2), each image fed into next
# "random" : N_RANDOM random type/style/stage combinations
MODE = "single"

# --- Conditioning (used by "single" and "chain" modes) ---
GEN_TYPES    = ["grass"]     # 1 or 2 types, e.g. ["grass", "poison"]
GEN_STYLE    = "3d"             # "3d" | "sugimori" | "sprite"
GEN_STAGE    = "base"           # "base" | "evo 1" | "evo 2"
GEN_PREV_EVO = None            # path to a previous evolution image, or None

# --- Sampling ---
NUM_SAMPLES = 4      # images per generation call
GUIDANCE    = 3.0    # CFG guidance scale (1.0 = no guidance)
DDIM_STEPS  = 200    # more steps = slower but slightly higher quality

# --- Random mode ---
N_RANDOM = 16        # number of random generations (MODE = "random")
COND_EMBED_DIM   = 256
ATTN_RESOLUTIONS = (16, 8)
IMG_SIZE       = 128
IMG_CHANNELS   = 3
TIMESTEPS      = 1000
EMA_DECAY      = 0.999
BASE_CH        = 64
CH_MULTS       = (1, 2, 2, 4)
NUM_RES_BLOCKS = 2
DROPOUT        = 0.1

TYPE_TO_IDX = {
    "normal": 0, "fire": 1, "water": 2, "electric": 3,
    "grass": 4, "ice": 5, "fighting": 6, "poison": 7,
    "ground": 8, "flying": 9, "psychic": 10, "bug": 11,
    "rock": 12, "ghost": 13, "dragon": 14, "dark": 15,
    "steel": 16, "fairy": 17,
}
STYLE_TO_IDX = {"3d": 0, "sugimori": 1, "sprite": 2}
STAGE_TO_IDX = {"base": 0, "evo 1": 1, "evo 2": 2}
ALL_TYPES  = list(TYPE_TO_IDX.keys())
ALL_STYLES = list(STYLE_TO_IDX.keys())
ALL_STAGES = ["base", "evo 1", "evo 2"]
NUM_TYPES  = len(TYPE_TO_IDX)
NUM_STYLES = len(STYLE_TO_IDX)
NUM_STAGES = len(ALL_STAGES)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm      = nn.GroupNorm(8, channels)
        self.num_heads = num_heads
        self.qkv       = nn.Conv2d(channels, channels * 3, 1)
        self.proj      = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
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


class ConditionalUNet(nn.Module):
    def __init__(self, img_channels, prev_channels, cond_vec_dim, cond_embed_dim, base_ch, ch_mults, num_res_blocks, attn_resolutions, dropout, img_size):
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
            nn.Conv2d(prev_channels, 32, 4, 2, 1),  nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),             nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),           nn.SiLU(),
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
                    "res":  ResBlock(ch, out_ch, emb_dim, dropout),
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
                    "res":  ResBlock(ch + skip_ch, out_ch, emb_dim, dropout),
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
        emb   = t_emb + c_emb + p_emb

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

class NoiseSchedule:
    def __init__(self, timesteps, device, s=0.008):
        self.timesteps = timesteps
        self.device    = device
        t  = torch.linspace(0, 1, timesteps + 1, device=device)
        ab = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        self.alpha_cumprod                = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alpha_cumprod           = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def _run_cfg(self, model, x_t, t_tensor, cond, prev, has_prev, guidance_scale):
        if guidance_scale == 1.0:
            return model(x_t, t_tensor, cond, prev, has_prev)
        null_cond     = torch.zeros_like(cond)
        null_prev     = torch.zeros_like(prev)
        null_has_prev = torch.zeros_like(has_prev)
        pred_cond   = model(x_t, t_tensor, cond, prev, has_prev)
        pred_uncond = model(x_t, t_tensor, null_cond, null_prev, null_has_prev)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, prev, has_prev,
                    num_steps=200, guidance_scale=3.0, eta=0.0):
        step_idx = torch.linspace(self.timesteps - 1, 0, num_steps + 1, device=self.device).long()
        x = torch.randn(shape, device=self.device)
        for i in range(num_steps):
            t  = step_idx[i].item()
            tp = step_idx[i + 1].item()
            tt = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps    = self._run_cfg(model, x, tt, cond, prev, has_prev, guidance_scale)
            a_t    = self.alpha_cumprod[t]
            a_tp   = self.alpha_cumprod[tp] if tp >= 0 else torch.tensor(1.0, device=self.device)
            x0p    = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
            sigma  = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp))
            dir_xt = torch.sqrt((1 - a_tp - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_tp) * x0p + dir_xt
            if eta > 0 and tp > 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

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


def tensor_to_pil(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).convert("RGB")


def load_prev_evo_image(path, device):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    return tfm(Image.open(path).convert("RGB")).to(device)


def load_checkpoint(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Auto-detect attn_resolutions from checkpoint keys
    has_32        = any("downs.6.attn.norm" in k for k in state)
    detected_attn = (32, 16, 8) if has_32 else ATTN_RESOLUTIONS

    # Auto-detect cond_vec_dim from checkpoint weights
    detected_cond_dim = state["cond_mlp.0.weight"].shape[1]

    model = ConditionalUNet(
        img_channels=IMG_CHANNELS, prev_channels=IMG_CHANNELS,
        cond_vec_dim=detected_cond_dim, cond_embed_dim=COND_EMBED_DIM,
        base_ch=BASE_CH, ch_mults=CH_MULTS,
        num_res_blocks=NUM_RES_BLOCKS, attn_resolutions=detected_attn,
        dropout=DROPOUT, img_size=IMG_SIZE,
    ).to(device)

    model.load_state_dict(state)
    ema = EMA(model, EMA_DECAY)
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.to(device)

    print(f"Loaded epoch {ckpt['epoch']}, loss {ckpt.get('loss', float('nan')):.4f} "
          f"| attn={detected_attn} | cond_dim={detected_cond_dim}")
    return model, ema, NoiseSchedule(TIMESTEPS, device)


@torch.no_grad()
def generate(model, noise_schedule, cond_vec, device, prev_evo_tensor=None):
    model.eval()
    cond = cond_vec.unsqueeze(0).expand(NUM_SAMPLES, -1).to(device)
    if prev_evo_tensor is not None:
        prev     = prev_evo_tensor.unsqueeze(0).expand(NUM_SAMPLES, -1, -1, -1).to(device)
        has_prev = torch.ones(NUM_SAMPLES, device=device)
    else:
        prev     = torch.zeros(NUM_SAMPLES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        has_prev = torch.zeros(NUM_SAMPLES, device=device)
    x = noise_schedule.ddim_sample(
        model, (NUM_SAMPLES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
        cond, prev, has_prev,
        num_steps=DDIM_STEPS, guidance_scale=GUIDANCE,
    )
    return (x + 1) / 2   # [0, 1]


def save_grid(images, path, cols=4):
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * IMG_SIZE, rows * IMG_SIZE), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * IMG_SIZE, r * IMG_SIZE))
    grid.save(path)
    print(f"  Grid saved → {path}")


def run_single(model, ema, noise_schedule, device):
    ema.apply_shadow()
    try:
        cond = make_cond_vec(GEN_TYPES, GEN_STYLE, GEN_STAGE)
        prev = load_prev_evo_image(GEN_PREV_EVO, device) if GEN_PREV_EVO else None
        imgs = generate(model, noise_schedule, cond, device, prev)
        results = []
        tag = f"{'_'.join(GEN_TYPES)}_{GEN_STYLE}_{GEN_STAGE.replace(' ', '_')}"
        for i in range(imgs.shape[0]):
            pil = tensor_to_pil(imgs[i])
            results.append(pil)
            fname = f"{OUTPUT_DIR}/{tag}_{i}.png"
            pil.save(fname)
            print(f"  Saved {fname}")
        save_grid(results, f"{OUTPUT_DIR}/grid_{tag}.png")
    finally:
        ema.restore()


def run_chain(model, ema, noise_schedule, device):
    ema.apply_shadow()
    try:
        chain = []

        # If a base image is provided, use it as prev for evo 1 and start from evo 1
        # Otherwise generate from scratch starting at base
        if GEN_PREV_EVO is not None:
            base_img = load_prev_evo_image(GEN_PREV_EVO, device)
            chain.append(Image.open(GEN_PREV_EVO).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
            stages_to_generate = ["evo 1", "evo 2"]
            prev_tensor = base_img
        else:
            stages_to_generate = ALL_STAGES
            prev_tensor = None

        for stage in stages_to_generate:
            cond = make_cond_vec(GEN_TYPES, GEN_STYLE, stage)
            imgs = generate(model, noise_schedule, cond, device, prev_tensor)
            pil  = tensor_to_pil(imgs[0])
            chain.append(pil)
            prev_tensor = imgs[0] * 2 - 1   # back to [-1, 1] for next stage

        strip = Image.new("RGB", (IMG_SIZE * len(chain), IMG_SIZE), (255, 255, 255))
        for i, img in enumerate(chain):
            strip.paste(img, (i * IMG_SIZE, 0))
        fname = f"{OUTPUT_DIR}/chain_{'_'.join(GEN_TYPES)}_{GEN_STYLE}.png"
        strip.save(fname)
        print(f"  Evolution chain saved → {fname}")
    finally:
        ema.restore()


def run_random(model, ema, noise_schedule, device):
    all_imgs = []
    for i in range(N_RANDOM):
        type_names = random.sample(ALL_TYPES, random.choice([1, 1, 2]))
        style      = random.choice(ALL_STYLES)
        stage      = random.choice(ALL_STAGES)
        print(f"[{i+1}/{N_RANDOM}] types={type_names}, style={style}, stage={stage}")
        ema.apply_shadow()
        try:
            cond = make_cond_vec(type_names, style, stage)
            imgs = generate(model, noise_schedule, cond, device, None)
            stage_safe = stage.replace(" ", "_")
            for j in range(imgs.shape[0]):
                pil = tensor_to_pil(imgs[j])
                all_imgs.append(pil)
                pil.save(f"{OUTPUT_DIR}/rand_{i+1}_{'_'.join(type_names)}_{style}_{stage_safe}_{j}.png")
        finally:
            ema.restore()
    save_grid(all_imgs, f"{OUTPUT_DIR}/random_grid_gs{GUIDANCE}.png", cols=NUM_SAMPLES)


def main():
    if SEED is not None:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model, ema, noise_schedule = load_checkpoint(CHECKPOINT, device)

    if MODE == "single":
        run_single(model, ema, noise_schedule, device)
    elif MODE == "chain":
        run_chain(model, ema, noise_schedule, device)
    elif MODE == "random":
        run_random(model, ema, noise_schedule, device)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Choose 'single', 'chain', or 'random'.")


if __name__ == "__main__":
    main()