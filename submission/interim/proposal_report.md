# Diffusion-Based Generation of Novel Pokémon with Conditional Evolution Modeling

## Members and Contributions

### Aaron Cyril John: Data processing and model evaluation

### Yugaank Kalia: Model training and architecture development

### Varen Maniktala: Data collection and pre-processing

## 1.1 Background

Generative models have become an important area of deep learning research, enabling the synthesis of realistic images, audio, and text. In recent years, diffusion models have emerged as one of the most powerful generative modeling approaches, surpassing generative adversarial networks (GANs) in many image generation tasks. Diffusion models learn to generate images by gradually transforming random noise into structured images through a learned denoising process.

The Pokémon franchise provides a well-structured dataset for studying generative models because Pokémon designs follow recognizable patterns such as elemental types (e.g., fire, water, electric), evolution stages, and consistent artistic styles. These characteristics make Pokémon imagery particularly suitable for conditional generative modeling, where the model generates images based on specific attributes.

The goal of this project is to develop a conditional diffusion-based generative model capable of producing novel Pokémon designs based on user-specified attributes such as Pokémon type, evolution stage, and art style. In addition, the model will support conditional evolution generation, where a newly generated Pokémon can be used as input to generate a plausible evolutionary form.

In addition to generating standalone Pokémon, the model will support conditional evolution generation, where a generated Pokémon can be used as a reference to produce a plausible evolved form. This allows the system to model visual progression across evolutionary stages while maintaining stylistic consistency. Through this approach, the project aims to demonstrate how conditional diffusion models can generate structured and coherent character designs while preserving relationships between related entities such as evolutionary chains.

## 1.2 Novelty
To generate sprite-style images for the generated Pokémons and their evolutions, we will use a U-Net–based image generation architecture, potentially implemented as an autoencoder or within a GAN framework. The encoder–decoder structure with skip connections allows the model to capture both global structure and fine visual details while preserving spatial information.

By conditioning the model on Pokémon attributes and, when applicable, a previously generated evolution stage, the system can produce coherent sprite representations that remain consistent across evolutionary chains. These generated sprites could potentially be used as assets for small game prototypes or other creative applications.

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
4. Generating sprites that could later be used for game design / development.

Attribute information will be incorporated into the model through learned embeddings that encode structured inputs such as Pokémon type, evolution stage, and artistic style. These attributes will first be represented using one‑hot encodings and then mapped to a dense embedding space. This embedding serves as a conditioninal embedding that guides the diffusion model during the image generation process.

By injecting the conditional embeddings into the U‑Net architecture, the model may learn relationships between attributes and their visual characteristics. This would allow the diffusion process to adapt generation based on specific attribute combinations, enabling more controlled and structured image synthesis compared to an unconditional model. This will be explored further during training of the diffusion model.

---

# 3. Training Data

The training dataset will consist of Pokémon images paired with metadata describing attributes such as Pokémon type, evolution stage, and artistic style.

We will use three primary datasets:

### PokeSprite
* **Content**: 68×56 pixel 2D Pokémon sprites (906 Pokémon)
* **Source**: [https://github.com/msikma/pokesprite/tree/master/pokemon-gen7x](https://github.com/msikma/pokesprite/tree/master/pokemon-gen7x) and [https://github.com/msikma/pokesprite/tree/master/pokemon-gen8](https://github.com/msikma/pokesprite/tree/master/pokemon-gen8)
* **Description**: Small-format pixel art sprites suitable for learning low-resolution generation patterns

### ProjectPokemon/PokemonDB
* **Content**: 3D Pokémon HOME models (all Pokémon)
* **Source**: [https://projectpokemon.org/home/docs/spriteindex_148/](https://projectpokemon.org/home/docs/spriteindex_148/) (home-sprites-gen-{X}-r{Y} where X = 1-8 (generation) and Y = 128-135 (ID)) and [https://pokemondb.net/pokedex/national](https://pokemondb.net/pokedex/national)
* **Description**: High-quality 3D renders with 512×512 resolution for Pokémon #1-898 (ProjectPokemon) and 256×256 resolution for Pokémon #899+ (PokemonDB)

### SugimoriSprites
* **Content**: Large quality 2D Pokémon sprites (all Pokémon)
* **Source**: [Google Drive collection](https://drive.google.com/drive/folders/1T2hF3ieas4mNBKQN6v94mlY8lbwT4KLx?usp=sharing) and [Reddit post](https://www.reddit.com/r/pokemon/comments/wx1qxp/all_officialsugimori_pokemon_art_collection_zip/)
* **Description**: Official Sugimori-style artwork (>500×500 pixels) representing the canonical artistic style

For metadata, we will use **PokéAPI** ([https://pokeapi.co/](https://pokeapi.co/)) to obtain information about Pokémon types and evolution chains.

The dataset will include multiple image styles, including:

* official Sugimori-style artwork,
* pixel sprite images from early Pokémon games,
* 3D renders from modern Pokémon titles.

Each image will be labeled with:

* Pokémon type (one or two types),
* evolution stage (base, stage 1, stage 2),
* artistic style,
* previous evolution.

The images will be preprocessed to ensure consistent resolution (e.g., 128×128 or 256×256) and normalized for model training.

To support evolution modeling, training samples will include pairs such as:

```
(previous Pokémon image, attributes) -> evolved Pokémon image
```

For example:

```
Charmander -> Charmeleon
Bulbasaur -> Ivysaur
```

This allows the model to learn visual transformation patterns that occur during Pokémon evolution.

---

# 4. Model Architecture

The system will use a conditional diffusion model to generate images.

The model receives the following inputs:

* noisy image (during training),
* diffusion timestep,
* conditional embedding describing Pokémon attributes,
* optional reference image representing a previous evolution.

The conditioning information includes:

* Pokémon type
* evolution stage
* art style

These attributes are encoded into a vector representation using a fully connected embedding layer, which produces a conditioning representation used by the diffusion model.

The diffusion network then predicts the noise added to an image at each timestep, gradually learning to reconstruct the original image while incorporating the conditioning information.

At inference time, the model begins with random noise and iteratively denoises the image while respecting the specified attributes.

The architecture can be summarized as:

```
conditional attributes -> embedding layers -> conditional embedding
noise image -> diffusion UNet -> predicted noise -> generated Pokémon image
```

This architecture enables both attribute-controlled generation and evolution-based conditioning, where the model can incorporate both attribute vectors and features from the previous evolution to guide generation.

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

The model will be implemented using PyTorch which provides flexible tools for building diffusion architectures.

Existing diffusion implementations will be adapted from publicly available repositories, including:

* Hugging Face Diffusers library
* PyTorch diffusion model implementations
* open-source UNet diffusion architectures

These implementations will be modified to incorporate conditional embeddings representing Pokémon attributes and optional reference images for evolution generation.

The training pipeline will include:

* dataset preparation and preprocessing,
* conditional embedding encoding,
* diffusion model training,
* evaluation and visualization of generated Pokémon.
* sprite generation using UNet auto-encoders or GANs.