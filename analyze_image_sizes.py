#!/usr/bin/env python3
"""
Script to analyze image sizes in the PokeSprite folder.
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


def display_results(size_counter, total_images, errors):
    """
    Display the results in a pandas DataFrame.
    
    Args:
        size_counter: Counter with size counts
        total_images: Total number of images processed
        errors: Number of errors encountered
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
    print("\nPokeSprite Image Size Analysis")
    print("=" * 30)
    print(df.to_string(index=False))
    print("=" * 30)
    
    # Print summary
    print(f"\nTotal images analyzed: {total_images}")
    print(f"Unique sizes found: {len(df)}")
    if errors > 0:
        print(f"Errors encountered: {errors}")
    
    return df


def main():
    # Set the path to poke-data directory
    script_dir = Path(__file__).parent
    pokesprite_dir = script_dir / "poke-data" / "PokeSprite"
    
    if not pokesprite_dir.exists():
        print(f"Error: Directory '{pokesprite_dir}' not found!")
        return
    
    print(f"Analyzing images in: {pokesprite_dir}")
    print("This may take a moment...")
    
    # Get image sizes
    size_counter, total_images, errors = get_image_sizes(pokesprite_dir)
    
    # Display results and get DataFrame
    df = display_results(size_counter, total_images, errors)


if __name__ == "__main__":
    main()
