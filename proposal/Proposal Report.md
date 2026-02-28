# Diffusion-Based Generation of Novel Pokémon with Conditional Evolution Modeling

## Members and Contributions

### Aaron Cyril John:

### Yugaank Kalia:

### Varen Maniktala:

## 1. Background

Generative models have become an important area of deep learning research, enabling the synthesis of realistic images, audio, and text. In recent years, diffusion models have emerged as one of the most powerful generative modeling approaches, surpassing generative adversarial networks (GANs) in many image generation tasks. Diffusion models learn to generate images by gradually transforming random noise into structured images through a learned denoising process.

The Pokémon franchise provides a well-structured dataset for studying generative models because Pokémon designs follow recognizable patterns such as elemental types (e.g., fire, water, electric), evolution stages, and consistent artistic styles. These characteristics make Pokémon imagery particularly suitable for conditional generative modeling, where the model generates images based on specific attributes.

The goal of this project is to develop a conditional diffusion-based generative model capable of producing novel Pokémon designs based on user-specified attributes such as Pokémon type, evolution stage, and art style. In addition, the model will support conditional evolution generation, where a newly generated Pokémon can be used as input to generate a plausible evolutionary form.

Transformers will be used as part of the architecture to model the conditional inputs and guide the diffusion process. Transformer-based embeddings allow the model to learn relationships between attributes such as type and evolution stage, providing a flexible mechanism for conditioning image generation.

---

# 2. Significance and Problem Statement

Designing new Pokémon species is traditionally a manual artistic process that requires balancing creativity with established visual patterns. The Pokémon universe contains hundreds of species with complex design principles, including:

* type-specific visual motifs (e.g., flames for fire-type Pokémon),
* evolution progressions that gradually increase complexity or size,
* stylistic consistency across official artwork and sprite formats.

The problem addressed in this project is:

Can a deep learning model generate new Pokémon designs that respect these structural patterns while producing novel species that do not already exist in the dataset?

Specifically, we aim to build a model capable of:

1. Generating new Pokémon images conditioned on attributes such as type, stage, and style.
2. Producing plausible evolutionary forms conditioned on previously generated Pokémon.
3. Maintaining stylistic consistency with known Pokémon artwork.

Transformers are used to encode attribute information because they are effective at learning relationships between structured inputs. The transformer embeddings provide a flexible conditioning mechanism that allows the diffusion model to adapt generation based on combinations of attributes. This architecture enables more structured generation compared to unconditional diffusion models.

The resulting system could demonstrate how structured generative models can incorporate both semantic attributes and visual conditioning, which has broader implications for conditional image generation in areas such as character design, game development, and creative AI systems.

---

# 3. Training Data

The training dataset will consist of Pokémon images paired with metadata describing attributes such as Pokémon type, evolution stage, and artistic style.

Possible data sources include:

* PokéAPI ([https://pokeapi.co/](https://pokeapi.co/)) for metadata about Pokémon types and evolution chains.
* Pokémon sprite repositories (e.g., Pokémon Showdown sprite datasets).
* Kaggle Pokémon datasets containing official artwork and sprites.
* Pokémon GO or Pokémon HOME image collections.

The dataset will include multiple image styles, including:

* official Sugimori-style artwork,
* pixel sprite images from early Pokémon games,
* 3D renders from modern Pokémon titles.

Each image will be labeled with:

* Pokémon type (one or two types),
* evolution stage (base, stage 1, stage 2),
* artistic style,
* previous evolution (if applicable).

The images will be preprocessed to ensure consistent resolution (e.g., 128×128 or 256×256) and normalized for model training.

To support evolution modeling, training samples will include pairs such as:

```
(previous Pokémon image, attributes) → evolved Pokémon image
```

For example:

```
Charmander → Charmeleon
Bulbasaur → Ivysaur
```

This allows the model to learn visual transformation patterns that occur during Pokémon evolution.

---

# 4. Model Architecture

The system will use a conditional diffusion model to generate images.

The model receives the following inputs:

* noisy image (during training),
* diffusion timestep,
* conditioning vector describing Pokémon attributes,
* optional reference image representing a previous evolution.

The conditioning information includes:

* Pokémon type
* evolution stage
* art style

These attributes are encoded into embeddings using a transformer-based encoder, which produces a conditioning representation used by the diffusion model.

The diffusion network then predicts the noise added to an image at each timestep, gradually learning to reconstruct the original image while incorporating the conditioning information.

At inference time, the model begins with random noise and iteratively denoises the image while respecting the specified attributes.

The architecture can be summarized as:

```
conditioning attributes → transformer encoder → conditioning embedding
noise image → diffusion UNet → predicted noise → generated Pokémon image
```

This architecture enables both attribute-controlled generation and evolution-based conditioning.

---

# 5. Evaluation Metrics

The quality of generated Pokémon images will be evaluated using both quantitative and qualitative metrics.

Quantitative evaluation will include:

* Fréchet Inception Distance (FID)
Measures similarity between generated images and the real Pokémon dataset.

* Structural Similarity Index (SSIM)
Measures similarity between generated images and training samples.

* Diversity metrics
Evaluate whether the model produces varied outputs instead of repeating similar images.

Qualitative evaluation will also be performed by visually examining generated Pokémon designs to assess whether they:

* respect the specified Pokémon type,
* follow plausible evolution patterns,
* maintain stylistic consistency with the training dataset.

---

# 6. Implementation Framework

The model will be implemented using PyTorch which provides flexible tools for building diffusion models and transformer architectures.

Existing diffusion implementations will be adapted from publicly available repositories, including:

* Hugging Face Diffusers library
* PyTorch diffusion model implementations
* open-source UNet diffusion architectures

These implementations will be modified to incorporate conditional embeddings representing Pokémon attributes and optional reference images for evolution generation.

The training pipeline will include:

* dataset preparation and preprocessing,
* conditioning vector encoding,
* diffusion model training,
* evaluation and visualization of generated Pokémon.
