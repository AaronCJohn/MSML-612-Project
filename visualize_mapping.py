#!/usr/bin/env python3
"""
Visualize project_to_sprite mappings side by side for manual verification.

Input:
- project_to_sprite_mapping.csv

Output:
- mapping_preview/page_XXXX.png (paginated contact sheets)
- mapping_preview/unresolved_rows.csv (rows without sprite mapping)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict, Any, cast

from PIL import Image, ImageDraw, ImageFont

MAPPING_CSV = Path("mapping_3d_to_sprite.csv")
OUTPUT_DIR = Path("mapping_preview")
UNRESOLVED_CSV = OUTPUT_DIR / "unresolved_rows.csv"

THUMB_SIZE = (192, 192)
ROWS_PER_PAGE = 20
PAGE_WIDTH = 1400
ROW_HEIGHT = 240
MARGIN_X = 20
if hasattr(Image, "Resampling"):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    RESAMPLE = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def fit_thumb(image_path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(image_path)
    if img.mode not in {"RGB", "RGBA"}:
        img = img.convert("RGBA")

    bg = Image.new("RGBA", size, (245, 245, 245, 255))
    img_copy = img.copy()
    img_copy.thumbnail(size, cast(Any, RESAMPLE))

    x = (size[0] - img_copy.width) // 2
    y = (size[1] - img_copy.height) // 2
    if img_copy.mode == "RGBA":
        bg.paste(img_copy, (x, y), img_copy)
    else:
        bg.paste(img_copy, (x, y))
    return bg.convert("RGB")


def draw_row(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: Dict[str, str],
    idx_on_page: int,
    global_index: int,
    font: Any,
) -> None:
    y0 = idx_on_page * ROW_HEIGHT
    y1 = y0 + ROW_HEIGHT

    draw.rectangle([(0, y0), (PAGE_WIDTH, y1)], outline=(220, 220, 220), width=1)

    project_file = Path("poke-data/ProjectPokemon") / Path(row.get("model_folder", "")) / Path(row.get("model_file", ""))
    sprite_file = Path("poke-data/PokeSprite") / Path(row.get("sprite_folder", "")) / Path(row.get("sprite_file", ""))

    left_img_x = MARGIN_X
    right_img_x = 320
    img_y = y0 + 20

    # Project image
    if project_file.exists():
        try:
            p_thumb = fit_thumb(project_file, THUMB_SIZE)
            page.paste(p_thumb, (left_img_x, img_y))
        except Exception as exc:
            draw.text((left_img_x, img_y + 80), f"Project read error: {exc}", fill=(180, 0, 0), font=font)
    else:
        draw.text((left_img_x, img_y + 80), "Project file missing", fill=(180, 0, 0), font=font)

    # Sprite image
    if sprite_file and sprite_file.exists():
        try:
            s_thumb = fit_thumb(sprite_file, THUMB_SIZE)
            page.paste(s_thumb, (right_img_x, img_y))
        except Exception as exc:
            draw.text((right_img_x, img_y + 80), f"Sprite read error: {exc}", fill=(180, 0, 0), font=font)
    else:
        draw.text((right_img_x, img_y + 80), "No sprite mapped", fill=(180, 0, 0), font=font)

    # Labels and metadata
    draw.text((left_img_x, y0 + 5), "ProjectPokemon", fill=(0, 0, 0), font=font)
    draw.text((right_img_x, y0 + 5), "PokeSprite", fill=(0, 0, 0), font=font)

    text_x = 560
    draw.text((text_x, y0 + 20), f"Row #{global_index + 1}", fill=(20, 20, 20), font=font)
    draw.text((text_x, y0 + 45), f"Status: {row.get('status', '')}", fill=(20, 20, 20), font=font)
    draw.text((text_x, y0 + 70), f"Reason: {row.get('reason', '')}", fill=(20, 20, 20), font=font)

    draw.text((text_x, y0 + 105), "Project:", fill=(20, 20, 20), font=font)
    draw.text((text_x, y0 + 125), str(project_file), fill=(60, 60, 60), font=font)

    draw.text((text_x, y0 + 160), "Sprite:", fill=(20, 20, 20), font=font)
    draw.text((text_x, y0 + 180), str(sprite_file) if sprite_file else "", fill=(60, 60, 60), font=font)


def save_unresolved(rows: List[Dict[str, str]]) -> int:
    unresolved = [r for r in rows if not r.get("sprite_file")]
    if not unresolved:
        return 0

    with UNRESOLVED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["project_file", "sprite_file", "status", "reason"])
        writer.writeheader()
        writer.writerows(unresolved)
    return len(unresolved)


def main() -> None:
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {MAPPING_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(MAPPING_CSV)

    if not rows:
        print("No rows found in mapping CSV.")
        return

    font = ImageFont.load_default()

    total_pages = math.ceil(len(rows) / ROWS_PER_PAGE)
    for page_index in range(total_pages):
        start = page_index * ROWS_PER_PAGE
        end = min(start + ROWS_PER_PAGE, len(rows))
        chunk = rows[start:end]

        page_height = len(chunk) * ROW_HEIGHT
        page = Image.new("RGB", (PAGE_WIDTH, page_height), (255, 255, 255))
        draw = ImageDraw.Draw(page)

        for i, row in enumerate(chunk):
            draw_row(page, draw, row, i, start + i, font)

        out_path = OUTPUT_DIR / f"page_{page_index + 1:04d}.png"
        page.save(out_path)

    unresolved_count = save_unresolved(rows)

    print(f"Total rows visualized: {len(rows)}")
    print(f"Pages created: {total_pages}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Unresolved rows: {unresolved_count}")
    if unresolved_count:
        print(f"Unresolved CSV: {UNRESOLVED_CSV}")


if __name__ == "__main__":
    main()
