# Pokemon Diffusion

A diffusion model for generating Pokemon images. Supports two architectures: a **conditional** model that accepts Pokemon type, art style, and evolution stage as inputs, and an **unconditional baseline** model trained per style.

## Project Structure

```
├── main.py                             # Unified CLI entry point (train / inference)
├── requirements.txt                    # Python dependencies
├── model/
│   ├── train.py                        # Training script (baseline + conditional)
│   ├── inference.py                    # Inference script (baseline + conditional)
│   ├── train_config.json               # Training hyperparameters
│   ├── inference_config.json           # Inference parameters and checkpoint paths
│   └── safe_checkpoints/               # Saved model checkpoints
│       ├── all_styles/                 #   Conditional model (all styles)
│       ├── baseline_3d/                #   Baseline model (3D renders)
│       └── baseline_sprite/            #   Baseline model (sprites)
├── poke-data/                          # Image datasets (not tracked in git)
├── preprocessing/                      # Bounding box detection and image padding/resizing
├── scrapers/                           # Data collection scripts (ProjectPokemon, PokeSprite, Sugimori)
├── mappings/                           # Style-to-style and type/evolution mapping JSONs
├── evolutions/                         # Pokemon evolution chain data
└── good_results/                       # Curated sample outputs
```

## Setup

Install PyTorch first, then the remaining dependencies:

```bash
# CUDA 12.1 (HPC / GPU machines)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# OR CPU / MPS (Apple Silicon)
pip install torch torchvision

# Then install the rest
pip install -r requirements.txt
```

## Usage

All commands are run from the project root via `main.py`.

### Inference

**Conditional model** (generates images conditioned on type, style, and stage):

```bash
# Generate water-type sprites (default)
python main.py inference conditional

# Specify types and style
python main.py inference conditional --type grass poison --style 3d
python main.py inference conditional --type fire --style sugimori
```

Supported types: `normal`, `fire`, `water`, `electric`, `grass`, `ice`, `fighting`, `poison`, `ground`, `flying`, `psychic`, `bug`, `rock`, `ghost`, `dragon`, `dark`, `steel`, `fairy`.

Supported styles for conditional: `3d`, `sugimori`, `sprite`.

**Baseline model** (unconditional generation, one checkpoint per style):

```bash
python main.py inference baseline --style sprite
python main.py inference baseline --style 3d
```

Supported styles for baseline: `3d`, `sprite`.

Running `python main.py` with no arguments defaults to `inference conditional --type water --style sprite`.

### Training

```bash
python main.py train --arch conditional
python main.py train --arch baseline
```

Training hyperparameters are stored in `model/train_config.json`. Each architecture has its own section with settings for learning rate, batch size, epochs, model dimensions, and more.

## Configuration

All parameters (checkpoint paths, sampling steps, model architecture, etc.) are externalized into JSON config files so the Python code does not need to be edited:

- `model/train_config.json` -- training hyperparameters for both architectures
- `model/inference_config.json` -- inference parameters, checkpoint paths, and model architecture settings

## Data Pipeline

1. **Scraping**: Scripts in `scrapers/` collect images from ProjectPokemon, PokeSprite, and Sugimori sources.
2. **Preprocessing**: `preprocessing/gen_bb_box.py` detects bounding boxes, then `preprocessing/pad_resize_images.py` pads and resizes images to 128x128.
3. **Mappings**: `mappings/` contains scripts and JSON files that map images across styles and attach type/evolution metadata.
4. **Evolution data**: `evolutions/` holds Pokemon evolution chain JSONs used for evolution-conditioned generation.

## Model Details

Both architectures use a U-Net with sinusoidal time embeddings, residual blocks, and self-attention at configurable resolutions. Training uses a cosine noise schedule, EMA weight averaging, and DDIM sampling.

- **Conditional**: Accepts a conditioning vector (multi-hot type encoding + one-hot style + one-hot stage) and an optional previous-evolution image encoder. Trained with classifier-free guidance (CFG dropout).
- **Baseline**: Time-conditioned only, no external conditioning. Separate checkpoints trained per art style.
