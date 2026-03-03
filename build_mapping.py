#!/usr/bin/env python3
"""
Build mapping between ProjectPokemonCleaned (3D renders) and PokeSpriteCleaned (sprites).

Outputs:
- project_to_sprite_mapping.csv: one row per ProjectPokemon file
- sprite_to_project_mapping.csv: one row per PokeSprite file with many ProjectPokemon files
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("poke-data/ProjectPokemonCleaned")
SPRITE_ROOT = Path("poke-data/PokeSpriteCleaned")

PROJECT_TO_SPRITE_CSV = Path("project_to_sprite_mapping.csv")
SPRITE_TO_PROJECT_CSV = Path("sprite_to_project_mapping.csv")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

# PokeSprite only exists up to dex #905
MAX_POKESPRITE_DEX = 905
# Pokemon after #898 came from pokemondb (not projectpokemon), have different image types and no variants
POKEMONDB_START_DEX = 899

PROJECT_PATTERN = re.compile(
    r"^poke_capture_(\d{4})_(\d{3})_([a-z]{2})_([a-z])_([0-9]{8})_f_([nr])\.png$"
)


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def extract_dex_number(folder_name: str) -> int | None:
    """Extract dex number from folder name like '0001bulbasaur' -> 1"""
    try:
        return int(folder_name[:4])
    except (ValueError, IndexError):
        return None


def parse_project_filename(name: str) -> dict | None:
    match = PROJECT_PATTERN.match(name)
    if not match:
        return None

    dex, variant, sex_form_code, gmax_flag, _, shiny_flag = match.groups()
    return {
        "dex": dex,
        "variant": int(variant),
        "sex_form_code": sex_form_code,
        "gmax_flag": gmax_flag,
        "is_shiny": shiny_flag == "r",
    }


def sprite_key(sprite_file: Path) -> tuple[str, str]:
    """
    Returns (base_stem, form_part) for sorting/matching.

    Examples:
    - charizard_shiny.png       -> (charizard, "")
    - charizard-mega-x.png      -> (charizard, "mega-x")
    - charizard-gmax_shiny.png  -> (charizard, "gmax")
    """
    stem = sprite_file.stem
    if stem.endswith("_shiny"):
        stem = stem[:-6]

    if "-" in stem:
        base, form = stem.split("-", 1)
    else:
        base, form = stem, ""

    return base, form


def is_shiny_sprite(sprite_file: Path) -> bool:
    return sprite_file.stem.endswith("_shiny")


def choose_sprite_candidate(project_info: dict, sprite_files: list[Path]) -> tuple[Path | None, str]:
    """
    Deterministic mapping strategy:
    1) match shiny/non-shiny
    2) if gmax flag in project file, force gmax form
    3) else exclude gmax and map variant index to ordered forms (base first, then forms alphabetically)
    """
    shiny = project_info["is_shiny"]
    variant = project_info["variant"]
    gmax_flag = project_info["gmax_flag"]

    candidates = [p for p in sprite_files if is_shiny_sprite(p) == shiny]
    if not candidates:
        return None, "no-shiny-match"

    if gmax_flag == "g":
        gmax_candidates = [p for p in candidates if "-gmax" in p.stem]
        if gmax_candidates:
            return sorted(gmax_candidates)[0], "gmax-direct"
        return None, "gmax-missing"

    non_gmax = [p for p in candidates if "-gmax" not in p.stem]
    if not non_gmax:
        non_gmax = candidates

    base_files = []
    form_files = []

    for p in non_gmax:
        _, form = sprite_key(p)
        if form == "":
            base_files.append(p)
        else:
            form_files.append(p)

    ordered = sorted(base_files) + sorted(form_files, key=lambda p: sprite_key(p)[1])
    if not ordered:
        return None, "no-candidates"

    if variant < len(ordered):
        return ordered[variant], "variant-index"

    return ordered[-1], "variant-out-of-range-fallback"


def build_mapping(project_root: Path, sprite_root: Path):
    project_to_sprite_rows: list[dict] = []
    sprite_to_project: defaultdict[str, list[str]] = defaultdict(list)

    total_project_files = 0
    matched_files = 0
    unresolved_files = 0

    for project_folder in sorted([p for p in project_root.iterdir() if p.is_dir()]):
        sprite_folder = sprite_root / project_folder.name

        project_files = list_image_files(project_folder)
        total_project_files += len(project_files)

        if not sprite_folder.exists() or not sprite_folder.is_dir():
            for project_file in project_files:
                project_to_sprite_rows.append(
                    {
                        "project_file": str(project_file),
                        "sprite_file": "",
                        "status": "missing-sprite-folder",
                        "reason": "matching dex folder not found in PokeSpriteCleaned",
                    }
                )
                unresolved_files += 1
            continue

        sprite_files = list_image_files(sprite_folder)

        for project_file in project_files:
            parsed = parse_project_filename(project_file.name)
            if parsed is None:
                project_to_sprite_rows.append(
                    {
                        "project_file": str(project_file),
                        "sprite_file": "",
                        "status": "unparsed",
                        "reason": "filename does not match expected poke_capture pattern",
                    }
                )
                unresolved_files += 1
                continue

            sprite_match, method = choose_sprite_candidate(parsed, sprite_files)

            if sprite_match is None:
                project_to_sprite_rows.append(
                    {
                        "project_file": str(project_file),
                        "sprite_file": "",
                        "status": "unresolved",
                        "reason": method,
                    }
                )
                unresolved_files += 1
                continue

            project_to_sprite_rows.append(
                {
                    "project_file": str(project_file),
                    "sprite_file": str(sprite_match),
                    "status": "mapped",
                    "reason": method,
                }
            )
            sprite_to_project[str(sprite_match)].append(str(project_file))
            matched_files += 1

    sprite_to_project_rows = []
    for sprite_file, project_list in sorted(sprite_to_project.items()):
        sprite_to_project_rows.append(
            {
                "sprite_file": sprite_file,
                "project_count": len(project_list),
                "project_files": ";".join(project_list),
            }
        )

    stats = {
        "total_project_files": total_project_files,
        "matched_files": matched_files,
        "unresolved_files": unresolved_files,
        "unique_sprites_matched": len(sprite_to_project_rows),
    }

    return project_to_sprite_rows, sprite_to_project_rows, stats


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Missing folder: {PROJECT_ROOT}")
    if not SPRITE_ROOT.exists():
        raise FileNotFoundError(f"Missing folder: {SPRITE_ROOT}")

    project_to_sprite_rows, sprite_to_project_rows, stats = build_mapping(PROJECT_ROOT, SPRITE_ROOT)

    write_csv(
        PROJECT_TO_SPRITE_CSV,
        project_to_sprite_rows,
        ["project_file", "sprite_file", "status", "reason"],
    )
    write_csv(
        SPRITE_TO_PROJECT_CSV,
        sprite_to_project_rows,
        ["sprite_file", "project_count", "project_files"],
    )

    print("Mapping complete")
    print(f"Total ProjectPokemon files: {stats['total_project_files']}")
    print(f"Mapped files: {stats['matched_files']}")
    print(f"Unresolved files: {stats['unresolved_files']}")
    print(f"Unique sprites matched: {stats['unique_sprites_matched']}")
    print(f"Wrote: {PROJECT_TO_SPRITE_CSV}")
    print(f"Wrote: {SPRITE_TO_PROJECT_CSV}")


if __name__ == "__main__":
    main()
