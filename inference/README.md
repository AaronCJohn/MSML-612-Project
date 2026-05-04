# Running conditional diffusion inference

Model weights (`.pt` files, ~380 MB each) are **not** in Git. Clone the repo, add checkpoints locally, then run the script from the **repository root**.

## 1. Environment

```bash
cd MSML-612-Project-1
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r inference/requirements-inference.txt
```

For NVIDIA GPUs, install `torch`/`torchvision` using the command from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version, then install the rest of this file if anything is missing.

## 2. Checkpoints

**Get the archive or folder from your team** (shared Drive, Dropbox, course submission bundle, etc.) and unpack so paths match the layout below.

Default location (matches script defaults and `.gitignore`):

```text
inference/scripts/checkpoints/
  3d/                      ckpt_epoch_400.pt, ckpt_epoch_450.pt
  sprite/                  ckpt_epoch_250.pt, ckpt_epoch_400.pt, ckpt_epoch_500.pt
  sprite&3d/               ckpt_epoch_400.pt, ckpt_epoch_500.pt
  outputs_all_art_style_w_conditions/   ckpt_epoch_250.pt (and 400/500 if you have them)
```

You do **not** need every file to try one run: a single `ckpt_epoch_*.pt` is enough if you pass `--ckpt-file` or `--ckpt-dir` + `--epoch`.

**Team maintainer:** add your real download link in this section (GitHub Release, Google Drive, Hugging Face, etc.).

## 3. Commands (from repo root)

Quick smoke test (matches checked-in sample outputs layout):

```bash
python inference/scripts/conditional_diffusion_inference.py \
  --ckpt-dir "inference/scripts/checkpoints/sprite&3d" --epoch 500 \
  --styles 3d,sprite \
  --output-dir inference/outputs/my_run
```

Single file:

```bash
python inference/scripts/conditional_diffusion_inference.py \
  --ckpt-file inference/scripts/checkpoints/sprite/ckpt_epoch_500.pt \
  --styles sprite
```

Bulk sweep (uses every checkpoint listed in the script under `inference/scripts/checkpoints/`; missing files are skipped unless you add `--strict`):

```bash
python inference/scripts/conditional_diffusion_inference.py --defaults \
  --checkpoint-root inference/scripts/checkpoints \
  --output-dir inference/outputs/batch
```

Use `--cpu` to force CPU. On Apple Silicon, omit `--cpu` to use MPS when available.

## 4. Verify files are present

```bash
find inference/scripts/checkpoints -name 'ckpt_epoch_*.pt' | sort
```

If a path is wrong, you will see `SKIP: Missing checkpoint:` or `FileNotFoundError` with the full path the script expected.
