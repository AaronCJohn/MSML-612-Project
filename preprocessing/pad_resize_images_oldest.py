#!/usr/bin/env python3

import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# WHERE IMAGES WILL BE SAVED
output_root = r"H:\.shortcut-targets-by-id\1sDNc_3oujFrQjFhBM1lYLgCoEWNcBBqL\612 data\poke-data_128\SugimoriSprites"

# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def display_padded_image(original_img, padded_img, filename, display_count, target_size):
    if display_count >= 3:
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


# -----------------------------
# LOAD BBOX + ADD MARGIN
# -----------------------------
def load_bbox(label_path, img_w, img_h, margin=0.3):
    if not label_path.exists():
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip().split()

    _, cx, cy, w, h = map(float, line)

    # Convert YOLO → pixel coords
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    # 🔥 Add margin
    w *= (1 + margin)
    h *= (1 + margin)

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    # Clamp to bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return (x1, y1, x2, y2)


# -----------------------------
# CORE FUNCTION
# -----------------------------
def pad_and_resize_image(
    image_path,
    final_size,
    bg_type='white',
    is_sprite=False,
    show_display=False,
    display_count=None,
    save=False,
    use_bbox=True
):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            original_img = img.copy()

            orig_w, orig_h = img.size

            # -----------------------------
            # LOAD BBOX
            # -----------------------------
            label_path = (
                image_path.parent.parent / "annotations" / "labels" / "train" / f"{image_path.stem}.txt"
            )

            if use_bbox:
                bbox = load_bbox(label_path, orig_w, orig_h)

                if bbox:
                    x1, y1, x2, y2 = bbox
                    img = img.crop((x1, y1, x2, y2))
            #     else:
            #         print("WARNING: NOT USING BBOX")
            # else:
            #     print("WARNING: NOT USING BBOX")

            # -----------------------------
            # RESIZE (aspect ratio preserved)
            # -----------------------------
            target_w, target_h = final_size
            w, h = img.size

            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            # resample = Image.NEAREST if is_sprite else Image.BICUBIC
            resample = Image.NEAREST if is_sprite else Image.LANCZOS
            img = img.resize((new_w, new_h), resample=resample)

            # -----------------------------
            # BACKGROUND (strict handling)
            # -----------------------------
            bg_type = bg_type.lower()

            if bg_type == 'white':
                new_img = Image.new('RGBA', final_size, (255, 255, 255, 255))
            elif bg_type == 'black':
                new_img = Image.new('RGBA', final_size, (0, 0, 0, 255))
            elif bg_type == 'transparent':
                new_img = Image.new('RGBA', final_size, (0, 0, 0, 0))
            else:
                raise ValueError(f"Invalid bg_type: {bg_type}")

            # img is still RGBA here — paste using alpha as mask
            # -----------------------------
            # PAD (center)
            # -----------------------------
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            new_img.paste(img, (x, y), img)

            # Flatten only at the end (not for transparent)
            if bg_type in ('white', 'black'):
                new_img = new_img.convert("RGB")

            # -----------------------------
            # DISPLAY (no saving required)
            # -----------------------------
            if show_display and display_count is not None and display_count[0] < 3:
                display_padded_image(original_img, new_img, image_path.name, display_count[0], final_size)
                display_count[0] += 1

            # -----------------------------
            # SAVE (optional)
            # -----------------------------
            if save:
                if output_root is None:
                    raise ValueError("output_root must be provided when save=True")

                subfolder = image_path.parent.name
                save_dir = Path(output_root) / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / image_path.name
                new_img.save(save_path)
                # print(f"saved here: {save_path}")

            return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


# -----------------------------
# PROCESS DIRECTORY
# -----------------------------
def process_folder(
    folder_path,
    final_size,
    bg_type='white',
    is_sprite=False,
    show_display=False,
    save=False,
    use_bbox=True
):
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    processed = 0
    skipped = 0
    display_count = [0]

    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                path = Path(root) / file

                if pad_and_resize_image(
                    path,
                    final_size,
                    bg_type,
                    is_sprite,
                    show_display,
                    display_count,
                    save,
                    use_bbox=use_bbox
                ):
                    processed += 1
                else:
                    skipped += 1

    return processed, skipped


# -----------------------------
# PROCESS SINGLE IMAGE
# -----------------------------
def process_single_image(
    image_path,
    final_size,
    bg_type='white',
    is_sprite=False,
    save=False,
    use_bbox=True
):
    pad_and_resize_image(
        Path(image_path),
        final_size,
        bg_type,
        is_sprite,
        show_display=True,
        display_count=[0],
        save=save,
        use_bbox=use_bbox
    )


# -----------------------------
# MAIN
# -----------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # -------- CONFIG --------
    folder = project_root / "poke-data" / "SugimoriSprites"

    # set to None to process full dataset
    # single_image = folder / "1023iron crown" / "1023 Iron Crown.png"
    single_image = None

    final_size = (128,128)
    bg_type = 'white'
    is_sprite = False
    use_bbox = True

    save = True  # IMPORTANT: default = no saving
    # -----------------------

    if single_image:
        print(f"Testing single image: {single_image}")
        process_single_image(single_image, final_size, bg_type, is_sprite, save=save, use_bbox=use_bbox)
        return

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    print(f"Processing: {folder}")
    print(f"Target size: {final_size}")
    print(f"BG: {bg_type} | Sprite: {is_sprite}")
    print(f"Save enabled: {save}")

    processed, skipped = process_folder(
        folder,
        final_size,
        bg_type,
        is_sprite,
        show_display=False,
        save=save,
        use_bbox=use_bbox
    )

    print("\nDone")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()