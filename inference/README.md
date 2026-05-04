# Inference

Checkpoint files (`.pt`, large) are **not** in this repo since it's too large. Download them from drive and place them under `inference/scripts/checkpoints/` (same subfolder names as our training runs, e.g. `sprite&3d/`, `sprite/`, `3d/`, `outputs_all_art_style_w_conditions`).

From the **repository root**, with Python env that has PyTorch, Pillow, torchvision, numpy, tqdm:

```bash
python inference/scripts/conditional_diffusion_inference.py \
  --ckpt-dir "inference/scripts/checkpoints/sprite&3d" --epoch 500 \
  --styles 3d,sprite \
  --output-dir inference/outputs/my_run
```

```bash
python inference/scripts/conditional_diffusion_inference.py \
  --ckpt-file inference/scripts/checkpoints/sprite/ckpt_epoch_500.pt \
  --styles sprite
```

```bash
python inference/scripts/conditional_diffusion_inference.py --defaults \
  --checkpoint-root inference/scripts/checkpoints \
  --output-dir inference/outputs/batch
```

Use `--cpu` to force CPU. See the script docstring for more flags.
