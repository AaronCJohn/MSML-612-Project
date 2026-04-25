import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# WHERE IMAGES WILL BE SAVED
output_root = r"H:\.shortcut-targets-by-id\1sDNc_3oujFrQjFhBM1lYLgCoEWNcBBqL\612 data\poke-data resized\temp"

def display_padded_image(original_img, padded_img, filename, display_count, target_size):
    if display_count >= 3: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img); axes[0].set_title(f"Original {original_img.width}x{original_img.height}"); axes[0].axis('off')
    axes[1].imshow(padded_img); axes[1].set_title(f"Padded {target_size[0]}x{target_size[1]}"); axes[1].axis('off')
    plt.tight_layout(); plt.show()

def load_bbox(label_path, img_w, img_h, margin):
    if not label_path.exists(): return None
    with open(label_path, "r") as f:
        _, cx, cy, w, h = map(float, f.readline().strip().split())
    cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
    w *= (1 + margin); h *= (1 + margin)
    x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
    x2, y2 = min(img_w, int(cx + w/2)), min(img_h, int(cy + h/2))
    return (x1, y1, x2, y2)

def pad_and_resize_image(image_path, final_size, bg_type, is_sprite, show_display,
                         save, use_bbox, scale, bbox_margin, display_count=[0]):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            original_img = img.copy()
            orig_w, orig_h = img.size

            label_path = image_path.parent.parent / "annotations" / "labels" / "train" / f"{image_path.stem}.txt"
            if use_bbox:
                bbox = load_bbox(label_path, orig_w, orig_h, margin=bbox_margin)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    img = img.crop((x1, y1, x2, y2))

            w, h = img.size
            target_subject_height = int(final_size[1] * scale)  # ← use scale param
            s = target_subject_height / max(w, h)
            new_w = int(w * s)
            new_h = int(h * s)

            resample = Image.NEAREST if is_sprite else Image.LANCZOS
            img = img.resize((new_w, new_h), resample=resample)

            x = (final_size[0] - new_w) // 2
            y = (final_size[1] - new_h) // 2

            # PASTE (perfectly centered, no padding needed since we resized exactly)
            canvas = Image.new('RGBA', final_size, (0, 0, 0, 0))
            canvas.paste(img, (x,y), img)

            # --- Save alpha mask BEFORE converting to RGB ---
            # if save and bg_type in ('white', 'black'):
            #     alpha = new_img.getchannel('A')  # [0,255] grayscale, 0=bg, 255=fg
            #     subfolder = image_path.parent.name
            #     mask_dir = Path(output_root) / subfolder / "masks"
            #     mask_dir.mkdir(parents=True, exist_ok=True)
            #     alpha.save(mask_dir / f"{image_path.stem}.png")

            # BACKGROUND
            if bg_type == 'white':
                new_img = Image.new('RGBA', final_size, (255, 255, 255, 255))
            elif bg_type == 'black':
                new_img = Image.new('RGBA', final_size, (0, 0, 0, 255))
            elif bg_type == 'transparent':
                new_img = canvas  # already correct, skip flatten
            else:
                raise ValueError(f"Invalid bg_type: {bg_type}") 

            if bg_type in ('white', 'black'):
                new_img.paste(new_img, (0, 0), new_img)
                new_img = new_img.convert("RGB")

            if show_display and display_count is not None and display_count[0] < 3:
                display_padded_image(original_img, new_img, image_path.name, display_count[0], final_size)
                display_count[0] += 1

            if save:
                if output_root is None: raise ValueError("output_root must be provided when save=True")
                subfolder = image_path.parent.name
                save_dir = Path(output_root) / subfolder; save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / image_path.name; new_img.save(save_path)

            return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}"); return False

# Rest unchanged...
def process_folder(folder_path, final_size, bg_type, is_sprite, show_display, save, use_bbox, scale, bbox_margin):
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    processed = skipped = 0; display_count = [0]
    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                path = Path(root) / file
                if pad_and_resize_image(path, final_size, bg_type, is_sprite, show_display, 
                                      display_count, save, use_bbox, scale, bbox_margin): processed += 1
                else: skipped += 1
    return processed, skipped

def process_single_image(image_path, final_size, bg_type, is_sprite, show_display, save, use_bbox, scale, bbox_margin):
    pad_and_resize_image(Path(image_path), final_size, bg_type, is_sprite, show_display, display_count=[0], save=save, use_bbox=use_bbox, scale=scale, bbox_margin=bbox_margin)

def main():
    script_dir = Path(__file__).resolve().parent; project_root = script_dir.parent
    # folder = project_root / "poke-data" / "SugimoriSprites"
    # if not folder.exists(): print(f"Folder not found: {folder}"); return
    # single_image = folder / "0103exeggutor" / "0103 Exeggutor Alola.png"
    
    final_size = (512, 512)  # Use 512 for training data
    # bg_type, is_sprite, use_bbox, save, scale, bbox_margin = 'transparent', False, True, True, 1, 0.15
    
    # if single_image:
    #     print(f"Testing: {single_image}"); process_single_image(single_image, final_size, bg_type, is_sprite, save, use_bbox, scale, bbox_margin); return
    

    styles = [
        {
            "folder": project_root / "poke-data" / "ProjectPokemon",
            "bg_type": "transparent",
            "is_sprite": False,
            "use_bbox": True,
            "scale": 1,
            "bbox_margin": 0.05,
        },
        {
            "folder": project_root / "poke-data" / "PokeSprite",
            "bg_type": "transparent",
            "is_sprite": True,
            "use_bbox": True,
            "scale": 1,
            "bbox_margin": 0.2,
        },
        {
            "folder": project_root / "poke-data" / "SugimoriSprites",
            "bg_type": "transparent",
            "is_sprite": False,
            "use_bbox": True,
            "scale": 1,
            "bbox_margin": 0.15,
        },
    ]

    for cfg in styles:
        folder = cfg["folder"]
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue
        print(f"\nProcessing: {folder}")
        processed, skipped = process_folder(
            folder, final_size,
            bg_type=cfg["bg_type"],
            is_sprite=cfg["is_sprite"],
            use_bbox=cfg["use_bbox"],
            scale=cfg["scale"],
            bbox_margin=cfg["bbox_margin"],
            save=True,
        )
        print(f"Done. Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__": main()