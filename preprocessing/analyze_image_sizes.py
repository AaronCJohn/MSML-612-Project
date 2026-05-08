"""
Script to analyze image sizes in the poke-data folder.
Lists all unique image dimensions with their counts.
"""

import os
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter


def get_image_sizes(root_dir):
    """
    Walk through directory tree and collect image sizes.
    
    Args:
        root_dir: Root directory to start searching from
    
    Returns:
        Counter with size counts
        Total images count
        Errors count
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    size_counter = Counter()
    total_images = 0
    errors = 0
    
    # Walk through all subdirectories
    for root, _, files in os.walk(root_dir):
        for file in files:
            # Check if file has an image extension
            if Path(file).suffix.lower() in image_extensions:
                file_path = os.path.join(root, file)
                
                try:
                    with Image.open(file_path) as img:
                        size = f"{img.width}x{img.height}"
                        size_counter[size] += 1
                        total_images += 1
                except Exception as e:
                    errors += 1
                    print(f"Error reading {file_path}: {e}")
    
    return size_counter, total_images, errors


def display_results(size_counter, total_images, errors, title="Image Size Analysis"):
    """
    Display the results in a pandas DataFrame.
    
    Args:
        size_counter: Counter with size counts
        total_images: Total number of images processed
        errors: Number of errors encountered
        title: Heading printed above the table
    """
    # Convert to DataFrame
    data_list = []
    for size, count in size_counter.items():
        data_list.append({
            'Count': count,
            'Size': size
        })
    
    df = pd.DataFrame(data_list)
    
    # Sort by count (descending), then by size
    df = df.sort_values(['Count', 'Size'], ascending=[False, True])
    df = df.reset_index(drop=True)
    
    # Display results
    print(f"\n{title}")
    print("=" * 30)
    print(df.to_string(index=False))
    print("=" * 30)
    
    # Print summary
    print(f"\nTotal images analyzed: {total_images}")
    print(f"Unique sizes found: {len(df)}")
    if len(df) > 0:
        parts = df['Size'].str.split('x', expand=True)
        max_width = int(parts[0].astype(int).max())
        max_height = int(parts[1].astype(int).max())
        print(f"Max width (any image): {max_width}")
        print(f"Max height (any image): {max_height}")
    if errors > 0:
        print(f"Errors encountered: {errors}")
    
    return df


def main():
    # Set the path to poke-data directory
    script_dir = Path(__file__).parent.parent
    pokesprite_dirs = [
        script_dir / "poke-data" / "PokeSprite",
        script_dir / "poke-data" / "SugimoriSprites",
        script_dir / "poke-data" / "ProjectPokemon",
    ]

    merged_counter = Counter()
    total_images = 0
    total_errors = 0

    for d in pokesprite_dirs:
        if not d.exists():
            print(f"Error: Directory '{d}' not found!")
            return

        print(f"Analyzing images in: {d}")
        print("This may take a moment...")

        size_counter, n_images, errors = get_image_sizes(d)
        merged_counter.update(size_counter)
        total_images += n_images
        total_errors += errors

    display_results(
        merged_counter,
        total_images,
        total_errors,
        title="Combined poke-data (PokeSprite, SugimoriSprites, ProjectPokemon)",
    )

if __name__ == "__main__":
    main()
