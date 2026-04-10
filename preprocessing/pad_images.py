#!/usr/bin/env python3
"""
Unified script to pad images to specified size with configurable background.
"""

import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def display_padded_image(original_img, padded_img, filename, display_count, target_size):
    """
    Display the original and padded image side by side.
    
    Args:
        original_img: PIL Image object of original image
        padded_img: PIL Image object of padded image
        filename: Name of the file being processed
        display_count: Counter for how many images to display
        target_size: Target size tuple (width, height)
    """
    if display_count >= 3:  # Show first 3 examples
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original {original_img.width}x{original_img.height} - {filename}")
    axes[0].axis('off')
    
    axes[1].imshow(padded_img)
    axes[1].set_title(f"Padded to {target_size[0]}x{target_size[1]} - {filename}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def pad_image(image_path, final_size, bg_type='transparent', output_path=None, show_display=False, display_count=None):
    """
    Pad an image to specified size by centering it.
    
    Args:
        image_path: Path to the input image
        final_size: Tuple (width, height) for target size
        bg_type: Background type - 'white', 'black', or 'transparent'
        output_path: Path to save the padded image (if None, overwrites original)
        show_display: If True, displays first few images
        display_count: Counter for displayed images as a one-item list
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            target_w, target_h = final_size
            
            # Skip if image is larger than target
            if width > target_w or height > target_h:
                return False
            
            # Skip if already at target size
            if width == target_w and height == target_h:
                return False
            
            # Store original for display
            original_img = img.copy()
            
            # Determine mode and background based on bg_type
            if bg_type == 'white':
                working_img = img.convert('RGB')
                new_img = Image.new('RGB', final_size, (255, 255, 255))
            elif bg_type == 'black':
                working_img = img.convert('RGB')
                new_img = Image.new('RGB', final_size, (0, 0, 0))
            else:  # transparent
                working_img = img.convert('RGBA')
                new_img = Image.new('RGBA', final_size, (0, 0, 0, 0))

            # Calculate centered position
            x = (target_w - width) // 2
            y = (target_h - height) // 2
            
            # Paste the original image in the center
            if working_img.mode == 'RGBA' and new_img.mode == 'RGBA':
                new_img.paste(working_img, (x, y), working_img)
            else:
                new_img.paste(working_img, (x, y))
            
            # Display if requested
            if show_display and display_count is not None and display_count[0] < 3:
                display_padded_image(original_img, new_img, Path(image_path).name, display_count[0], final_size)
                display_count[0] += 1
            
            # Save the padded image
            save_path = output_path if output_path else image_path
            new_img.save(save_path)
            
            return True
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_folder(folder_path, final_size, bg_type='transparent', show_display=False):
    """
    Process all images in the specified folder.
    
    Args:
        folder_path: Path to the folder to process
        final_size: Tuple (width, height) for target size
        bg_type: Background type - 'white', 'black', or 'transparent'
        show_display: If True, displays first few padded images
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    processed_count = 0
    skipped_count = 0
    display_count = [0]  # Use list to track count across function calls
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                file_path = os.path.join(root, file)
                
                try:
                    if pad_image(file_path, final_size, bg_type, show_display=show_display, display_count=display_count):
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"Processed {processed_count} images...")
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error checking {file_path}: {e}")
    
    return processed_count, skipped_count


def main():
    """
    Main function with configuration for different datasets.
    Modify the CONFIG section below to process different folders.
    """
    script_dir = Path(__file__).parent

    # Option 1: ProjectPokemonCleaned (resize 512x512 to 256x256)
    # Option 2: SugimoriSpritesCleaned (pad to 3329x3329)

    folder = script_dir / "poke-data" / "PokeSpriteCleaned"
    final_size = (72, 72)
    bg_type = 'trasparent'
    
    # ===================================
    
    if not folder.exists():
        print(f"Error: Directory '{folder}' not found!")
        return
    
    print(f"Processing images in: {folder}")
    print(f"Target size: {final_size[0]}x{final_size[1]}")
    print(f"Background type: {bg_type}")
    print("(Backup is assumed to be already created)")
    print("Showing the first 3 examples...")
    print()
    
    processed, skipped = process_folder(folder, final_size, bg_type, show_display=True)
    
    print()
    print("=" * 50)
    print(f"Processing complete!")
    print(f"Images padded: {processed}")
    print(f"Images skipped (larger than target or already at target size): {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()
