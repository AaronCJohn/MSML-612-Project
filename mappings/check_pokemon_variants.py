#!/usr/bin/env python3
"""
Find all pokemon folders in ProjectPokemonCleaned with more than 2 images.
"""

from pathlib import Path
from collections import defaultdict

root = Path('poke-data/ProjectPokemonCleaned')
image_ext = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}

pokemon_counts = defaultdict(int)

# Iterate through each pokemon folder
for pokemon_folder in sorted(root.iterdir())[:906]:
    if pokemon_folder.is_dir():
        # Count image files in this pokemon folder
        image_count = sum(1 for f in pokemon_folder.iterdir() if f.is_file() and f.suffix.lower() in image_ext)
        if image_count > 0:
            pokemon_counts[pokemon_folder.name] = image_count

# Print folders with more than 2 images
print("Folders with more than 2 images per pokemon:")
print("=" * 50)
print(f"{'Pokemon Folder':<30} | {'Image Count'}")
print("-" * 50)

folders_with_more = []
for pokemon, count in sorted(pokemon_counts.items()):
    if count > 2:
        print(f"{pokemon:<30} | {count}")
        folders_with_more.append(pokemon)

print("-" * 50)
print(f"\nTotal pokemon folders with > 2 images: {len(folders_with_more)}")
