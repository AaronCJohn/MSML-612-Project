#!/usr/bin/env python3
"""
Visualize sprite_to_sprite mappings side by side for manual verification.

Input:
- mappings/Diffusion_mapping/sprite_to_sprite.json

Output:
- mapping_preview_diffusion/page_XXXX.png  (paginated contact sheets)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont

MAPPING_JSON = Path(__file__).resolve().parents[1] / "diffusion mapping" / "sprite_to_sprite.json"
REPO_ROOT    = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = Path(__file__).parent / "mapping_preview_diffusion"

THUMB_SIZE   = (192, 192)
ROWS_PER_PAGE = 20
PAGE_WIDTH   = 1000
ROW_HEIGHT   = 240
MARGIN_X     = 20

if hasattr(Image, "Resampling"):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    RESAMPLE = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))


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
    entry: dict,
    idx_on_page: int,
    global_index: int,
    font: Any,
) -> None:
    y0 = idx_on_page * ROW_HEIGHT
    y1 = y0 + ROW_HEIGHT

    draw.rectangle([(0, y0), (PAGE_WIDTH, y1)], outline=(220, 220, 220), width=1)

    prev_name   = entry.get("prev") or "—"
    next_name   = entry.get("next") or "—"
    prev_sprite = entry.get("prev_sprite")
    next_sprite = entry.get("next_sprite")

    left_x  = MARGIN_X
    right_x = MARGIN_X + THUMB_SIZE[0] + 40
    img_y   = y0 + 30
    label_y = y0 + 6

    # Left thumbnail: prev sprite
    draw.text((left_x, label_y), "prev", fill=(80, 80, 80), font=font)
    if prev_sprite:
        path = REPO_ROOT / prev_sprite
        if path.exists():
            try:
                page.paste(fit_thumb(path, THUMB_SIZE), (left_x, img_y))
            except Exception as exc:
                draw.text((left_x, img_y + 80), f"Error: {exc}", fill=(180, 0, 0), font=font)
        else:
            draw.text((left_x, img_y + 80), "File missing", fill=(180, 0, 0), font=font)
    else:
        draw.text((left_x, img_y + 80), "null", fill=(160, 160, 160), font=font)

    # Right thumbnail: next sprite
    draw.text((right_x, label_y), "next", fill=(80, 80, 80), font=font)
    if next_sprite:
        path = REPO_ROOT / next_sprite
        if path.exists():
            try:
                page.paste(fit_thumb(path, THUMB_SIZE), (right_x, img_y))
            except Exception as exc:
                draw.text((right_x, img_y + 80), f"Error: {exc}", fill=(180, 0, 0), font=font)
        else:
            draw.text((right_x, img_y + 80), "File missing", fill=(180, 0, 0), font=font)
    else:
        draw.text((right_x, img_y + 80), "null", fill=(160, 160, 160), font=font)

    # Metadata column
    text_x = right_x + THUMB_SIZE[0] + 30
    draw.text((text_x, y0 + 20), f"#{global_index + 1}", fill=(120, 120, 120), font=font)
    draw.text((text_x, y0 + 40), f"prev: {prev_name}", fill=(20, 20, 20), font=font)
    draw.text((text_x, y0 + 60), f"next: {next_name}", fill=(20, 20, 20), font=font)

    if prev_sprite:
        draw.text((text_x, y0 + 95),  "prev sprite:", fill=(60, 60, 60), font=font)
        draw.text((text_x, y0 + 112), prev_sprite, fill=(100, 100, 100), font=font)
    if next_sprite:
        draw.text((text_x, y0 + 140), "next sprite:", fill=(60, 60, 60), font=font)
        draw.text((text_x, y0 + 157), next_sprite, fill=(100, 100, 100), font=font)


def main() -> None:
    if not MAPPING_JSON.exists():
        raise FileNotFoundError(f"Missing mapping JSON: {MAPPING_JSON}")

    with MAPPING_JSON.open(encoding="utf-8") as f:
        entries = json.load(f)

    # Only visualize entries where at least one sprite is present
    entries = [e for e in entries if e.get("prev_sprite") or e.get("next_sprite")]

    if not entries:
        print("No entries with sprites found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT_DIR.glob("page_*.png"):
        old.unlink()

    font = ImageFont.load_default()

    total_pages = math.ceil(len(entries) / ROWS_PER_PAGE)
    for page_index in range(total_pages):
        start = page_index * ROWS_PER_PAGE
        chunk = entries[start : start + ROWS_PER_PAGE]

        page_height = len(chunk) * ROW_HEIGHT
        page = Image.new("RGB", (PAGE_WIDTH, page_height), (255, 255, 255))
        draw = ImageDraw.Draw(page)

        for i, entry in enumerate(chunk):
            draw_row(page, draw, entry, i, start + i, font)

        out_path = OUTPUT_DIR / f"page_{page_index + 1:04d}.png"
        page.save(out_path)

    print(f"Entries visualized : {len(entries)}")
    print(f"Pages created      : {total_pages}")
    print(f"Output directory   : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
