---
marp: true
theme: default
paginate: true
backgroundColor: '#ffffff'
style: |
    section {
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 22px;
      color: #1a1a2e;
      padding: 40px 60px;
    }
    h1 {
      font-size: 36px;
      color: #16213e;
      border-bottom: 3px solid #e94560;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }
    h2 {
      font-size: 28px;
      color: #0f3460;
      margin-bottom: 16px;
    }
    h3 {
      font-size: 22px;
      color: #e94560;
      margin-bottom: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 18px;
      margin-top: 12px;
    }
    th {
      background-color: #0f3460;
      color: #ffffff;
      padding: 10px 14px;
      text-align: left;
    }
    td {
      padding: 9px 14px;
      border-bottom: 1px solid #dde1e7;
    }
    tr:nth-child(even) td {
      background-color: #f4f6fb;
    }
    .highlight {
      background: #e94560;
      color: white;
      padding: 3px 10px;
      border-radius: 4px;
    }
    .pill {
      display: inline-block;
      background: #0f3460;
      color: white;
      padding: 2px 12px;
      border-radius: 20px;
      font-size: 16px;
      margin: 2px;
    }
    section.center {
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
    }
    section.compact {
      font-size: 20px;
    }
    section.compact h2 {
      margin-bottom: 10px;
    }
    section.compact ul {
      margin-top: 8px;
    }
    section.compact li {
      margin-bottom: 4px;
    }
    section.tight {
      font-size: 18px;
    }
    section.tight h2 {
      margin-bottom: 8px;
    }
    section.tight ul {
      margin-top: 6px;
    }
    section.tight li {
      margin-bottom: 2px;
    }
    section.arch-side {
      font-size: 17px;
    }
    section.arch-side h2 {
      margin-bottom: 10px;
    }
    section.arch-side .arch-wrap {
      display: flex;
      gap: 20px;
      align-items: flex-start;
    }
    section.arch-side .arch-image {
      width: 58%;
    }
    section.arch-side .arch-image img {
      width: 100%;
      display: block;
    }
    section.arch-side .arch-text {
      width: 42%;
      font-size: 17px;
    }
    section.arch-side .arch-text p {
      margin: 0 0 6px 0;
    }
    section.arch-side .arch-text ul {
      margin: 0;
      padding-left: 20px;
    }
    section.arch-side .arch-text li {
      margin-bottom: 5px;
    }
    section.gan-side {
      font-size: 17px;
    }
    section.gan-side h2 {
      margin-bottom: 10px;
    }
    section.gan-side .gan-wrap {
      display: flex;
      gap: 18px;
      align-items: flex-start;
    }
    section.gan-side .gan-visuals {
      width: 68%;
    }
    section.gan-side .gan-text {
      width: 32%;
      font-size: 16px;
      line-height: 1.35;
    }
    section.gan-side .gan-text p {
      margin: 0 0 10px 0;
    }
    section.gan-side .gan-text strong {
      font-weight: 700;
    }
    section.gan-side table.gan-grid {
      width: 100%;
      margin-top: 0;
      table-layout: fixed;
    }
    section.gan-side table.gan-grid th,
    section.gan-side table.gan-grid td {
      padding: 6px 4px;
      font-size: 14px;
    }
    section.gan-side table.gan-grid img {
      width: 100%;
      max-width: 150px;
      height: auto;
    }
    .centered-figure {
      text-align: center;
      margin: 6px 0 0 0;
    }
    .centered-figure img {
      width: 620px;
      max-width: 95%;
    }
    footer {
      font-size: 14px;
      color: #888;
    }
---

<!-- _class: center -->
<!-- _backgroundColor: "#0f3460" -->
<!-- _color: "#ffffff" -->

# Diffusion-Based Generation of Novel Pokémon

## with Conditional Evolution Modeling

**MSML 612**
Aaron Cyril John, Yugaank Kalia, Varen Maniktala

---

## Outline

1. Problem Motivation
2. Dataset Overview
3. Data Sources & Curation
4. Preprocessing Pipeline
5. System Architecture
6. Training Procedure
7. Evaluation Plan & Metrics
8. Current Status & Next Steps

---

## Problem Motivation

**Goal:** Generate novel, visually coherent Pokémon designs under structured conditional controls.

| Challenge                    | Why It Matters                                                                          |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| Multi-source style alignment | Official art, game sprites, and 3D renders must correspond                              |
| Attribute conditioning       | Type, evolution stage, and style must guide generation                                  |
| Evolution chain coherence    | Generated forms must remain visually related across a chain                             |
| Domain-specific structure    | Pokémon generation is not just image synthesis: it is structured conditional generation |

> Standard unconditional image generators cannot capture evolution progression or type-based stylistic cues.

---

## Dataset Overview

Three complementary image sources collected and aligned:

| Source            | Files | Folders | Role                                               |
| ----------------- | ----- | ------- | -------------------------------------------------- |
| SugimoriSprites   | 1,848 | 1,025   | High-quality 2D reference art                      |
| PokeSprite        | 2,853 | 905     | Target game-style sprite output                    |
| ProjectPokemon 3D | 2,799 | 905     | Alternate visual domain for cross-format alignment |

**Mappings generated:**

| Mapping File                            | Pairs | Pokémon Folders | Status              |
| --------------------------------------- | ----- | --------------- | ------------------- |
| `mapping_sugimori_to_sprite.csv`        | 1,710 | 905             | Auto-generated      |
| `edited_mapping_sugimori_to_sprite.csv` | 977   | 830             | Manually verified ✓ |
| `mapping_3d_to_sprite.csv`              | 2,676 | 905             | Auto-generated      |

---

## Data Sources: Visual Comparison

| Style                              | Bulbasaur                                                                                                        | Charizard                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **SugimoriSprites** (Official Art) | <img src="../poke-data/SugimoriSprites/0001bulbasaur/0001 Bulbasaur.png" height="100px">                         | <img src="../poke-data/SugimoriSprites/0006charizard/0006 Charizard.png" height="100px">                         |
| **PokeSprite** (Game Sprite)       | <img src="../poke-data/PokeSprite/0001bulbasaur/bulbasaur.png" height="100px">                                   | <img src="../poke-data/PokeSprite/0006charizard/charizard.png" height="100px">                                   |
| **ProjectPokemon 3D**              | <img src="../poke-data/ProjectPokemon/0001bulbasaur/poke_capture_0001_000_mf_n_00000000_f_n.png" height="100px"> | <img src="../poke-data/ProjectPokemon/0006charizard/poke_capture_0006_000_mf_n_00000000_f_n.png" height="100px"> |

> Each source captures a different visual style of the **same** Pokémon: cross-format alignment is critical.

---

## Preprocessing Pipeline

![w:900](images/data-preprocessing.png)

> Filename normalization → form-variant handling (Mega, Gigantamax) → CSV mapping generation → manual verification via contact-sheet review

---

<!-- _class: arch-side -->

## System Architecture

<div class="arch-wrap">
  <div class="arch-image">
    <img src="images/pokemon-diffusion.png" alt="System architecture diagram">
  </div>
  <div class="arch-text">
    <p><strong>Key components:</strong></p>
    <ul>
      <li><strong>Diffusion U-Net backbone</strong>: iterative denoising for image synthesis</li>
      <li><strong>Attribute conditioning</strong>: type embedding, evolution stage, and style vector</li>
      <li><strong>Reference-image conditioning path</strong>: guide evolution-chain generation</li>
      <li><strong>Sprite-focused output space</strong>: final images match game-style visual format</li>
    </ul>
  </div>
</div>

---

## Architecture: Conditioning Design

| Conditioning Input               | Encoding               | Purpose                        |
| -------------------------------- | ---------------------- | ------------------------------ |
| Pokémon Type (e.g., Fire, Water) | One-hot / embedding    | Style and color palette bias   |
| Evolution Stage (1 / 2 / 3)      | Scalar embedding       | Structural complexity control  |
| Visual Style                     | Domain label embedding | Sprite vs. official art target |
| Reference Image (optional)       | CNN feature extraction | Evolution-chain coherence      |

> All condition vectors are concatenated and injected into the U-Net via cross-attention at multiple resolutions.

---

<!-- _class: tight -->

## Training Procedure

<p class="centered-figure"><img src="images/training.png" alt="Training procedure diagram"></p>

**Training approach:**

- Adapt publicly available diffusion model implementations (Hugging Face Diffusers)
- Modify conditioning mechanism for structured Pokemon metadata
- Train on aligned Sugimori to Sprite pairs from the curated dataset

---

<!-- _class: gan-side -->

## Preliminary GAN Results

<div class="gan-wrap">
  <div class="gan-visuals">
    <table class="gan-grid">
      <tr>
        <th>Arceus</th>
        <th>Pachirisu</th>
        <th>Swampert</th>
      </tr>
      <tr>
        <td><img src="images/gan_result_3.png" alt="Arceus results"></td>
        <td><img src="images/gan_result_1.png" alt="Pachirisu results"></td>
        <td><img src="images/gan_result_2.png" alt="Naganadel results"></td>
      </tr>
    </table>
  </div>
  <div class="gan-text">
    <p><strong>Top:</strong> Input artwork</p>
    <p><strong>Middle:</strong> GAN-predicted sprite</p>
    <p><strong>Bottom:</strong> Ground-truth sprite</p>
    <p>GAN captures color palettes and rough silhouettes but lacks fine detail and sharpness compared with the ground-truth sprites.</p>
    <p>A diffusion transformer should improve global consistency and recover sharper fine-grained detail relative to this baseline.</p>
  </div>
</div>

---

## Evaluation Plan

| Metric                               | Description                                             | What It Measures     |
| ------------------------------------ | ------------------------------------------------------- | -------------------- |
| **FID** (Fréchet Inception Distance) | Distribution distance between generated and real images | Realism & diversity  |
| **SSIM**                             | Structural similarity for paired comparisons            | Pixel-level fidelity |
| **Diversity Score**                  | Pairwise distance across generated samples              | Mode coverage        |
| **Qualitative Review**               | Side-by-side visual inspection                          | Attribute adherence  |

**Planned ablations (If time allows):**

| Experiment                                 | Variable                          |
| ------------------------------------------ | --------------------------------- |
| Unconditional vs. attribute-conditioned    | Effect of conditioning on quality |
| With vs. without evolution-reference input | Evolution coherence               |
| Multiple target resolutions                | Resolution vs. fidelity trade-off |

---

## Current Status

| Component                             | Status                                |
| ------------------------------------- | ------------------------------------- |
| SugimoriSprites collection            | Complete (1,848 images)               |
| PokeSprite collection                 | Complete (2,853 images)               |
| Auto-generated mappings               | Complete (4,386 total pairs)          |
| Manual curation & verification        | 977 verified pairs across 830 Pokémon |
| Preprocessing & normalization scripts | Complete                              |
| Diffusion model architecture design   | Finalized                             |
| Training pipeline                     | In progress                           |
| Baseline training run                 | Pending                               |
| Quantitative evaluation               | Pending                               |

---

## Next Steps

1. **Finalize training pipeline** - convert curated dataset into DataLoader-ready format
2. **Run baseline experiment** - unconditional diffusion model on sprite images
3. **Add conditioning** - integrate type, stage, and style embeddings
4. **Evaluate** - compute FID, SSIM, and diversity scores on held-out Pokémon
5. **Ablation study** - measure impact of each conditioning component
6. **Visual summary** - generate contact sheets of novel Pokémon designs

---

<!-- _class: center -->
<!-- _backgroundColor: "#0f3460" -->
<!-- _color: "#ffffff" -->

## **References**

[1] Ho, J., Jain, A., and Abbeel, P. _Denoising Diffusion Probabilistic Models_. NeurIPS, 2020. https://arxiv.org/abs/2006.11239

[2] Ronneberger, O., Fischer, P., and Brox, T. _U-Net: Convolutional Networks for Biomedical Image Segmentation_. MICCAI, 2015. https://arxiv.org/abs/1505.04597

[3] Saharia, C., Chan, W., Saxena, S., et al. _Palette: Image-to-Image Diffusion Models_. SIGGRAPH, 2022. https://arxiv.org/abs/2111.05826

[4] Hugging Face. _Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch_. https://github.com/huggingface/diffusers

[5] PokéAPI. https://pokeapi.co/

[6] msikma. _pokesprite_. https://github.com/msikma/pokesprite

[7] Project Pokémon. Sprite and asset resources. https://projectpokemon.org/

[8] Pokémon Database. National Pokédex and image resources. https://pokemondb.net/pokedex/national
