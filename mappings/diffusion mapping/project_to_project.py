"""
Generates project_to_project.json from evolutions/evolution.json.

For each evolution pair (prev -> next):
    - Maps every valid prev ProjectPokemon image -> best-matching next image.
    - Chain starts (prev=null) get one entry per valid image in next's folder.

Filename format: poke_capture_DDDD_VVV_GG_F_00000000_f_S.png
    DDDD = dex number
    VVV  = variant (000 = default; 001+ = alternate forms, regional, mega)
    GG   = gender/form: mf | md | fd | mo | fo | uk
    F    = form type:   n (normal) | g (gigantamax)  ← g is skipped
    S    = shiny flag:  n (normal) | r (shiny/rare)

Matching strategy per evolution entry:
    1. Exact (variant_num, gender) match in next.
    2. Same gender, variant 000 in next.
    3. Gender mf, same variant in next.
    4. Gender mf, variant 000 in next (ultimate fallback).
    Shiny always maps to the corresponding shiny; falls back to normal if
    the next pokemon has no shiny for that key.

Regional form names in evolution.json (e.g. "Meowth-Galar") cannot be
mapped to specific variant numbers from filenames alone and are logged
as unresolved in project_to_project_unresolved.json.

Output
------
mappings/GAN mapping/project_to_project.json
mappings/GAN mapping/project_to_project_unresolved.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT      = Path(__file__).resolve().parents[2]
PROJECT_ROOT   = REPO_ROOT / "poke-data" / "ProjectPokemon"
EVOLUTION_JSON = REPO_ROOT / "evolutions" / "evolution.json"
OUTPUT_JSON    = Path(__file__).parent / "project_to_project.json"
UNRESOLVED_JSON = Path(__file__).parent / "project_to_project_unresolved.json"


# ---------------------------------------------------------------------------
# Image dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpriteFile:
    path: Path
    variant: int      # 0, 1, 2, …
    gender: str       # mf | md | fd | mo | fo | uk
    is_shiny: bool
    
    @classmethod
    def parse(cls, path: Path) -> "SpriteFile | None":
        """
        Parse a ProjectPokemon filename into its components.
        Returns None if the file should be skipped (gmax, bad format).

        Supports:
        1. Standard poke_capture format
        2. Fallback: simple 'name.png'
        """
        parts = path.stem.split("_")

        # ---- Standard format ----
        if len(parts) >= 9 and parts[0] == "poke" and parts[1] == "capture":
            form = parts[5]       # n or g
            if form == "g":       # skip gigantamax
                return None

            try:
                variant = int(parts[3])
            except ValueError:
                return None

            gender   = parts[4]
            is_shiny = parts[8] == "r"

            return cls(path=path, variant=variant, gender=gender, is_shiny=is_shiny)

        # ---- Fallback: simple naming (e.g., "annihilape.png") ----
        # Only accept if it's a clean single-token filename (no underscores)
        if "_" not in path.stem:
            return cls(
                path=path,
                variant=0,
                gender="mf",   # default assumption
                is_shiny=False
            )

        return None


# ---------------------------------------------------------------------------
# Folder index
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Collapse all separators (space, hyphen, dot, apostrophe) for matching."""
    return re.sub(r"[.\-'\s]", "", s.lower())


def build_folder_index(root: Path) -> dict[str, Path]:
    """Maps _norm(folder_basename_without_digits) -> folder_path."""
    index: dict[str, Path] = {}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        base = re.sub(r"^\d+", "", folder.name)
        index[_norm(base)] = folder
    return index


def resolve_folder(
    evo_name: str, folder_index: dict[str, Path]
) -> tuple[Path | None, str | None]:
    """
    Returns (folder, variant_str_or_None).
    variant_str is set when the name encodes a regional form (e.g. "Meowth-Galar"
    → variant_str="galar"). These cases cannot be mapped to variant numbers and
    are logged as unresolved.
    """
    key = _norm(evo_name)
    if key in folder_index:
        return folder_index[key], None

    parts = evo_name.split("-")
    for i in range(len(parts) - 1, 0, -1):
        base_key = _norm("-".join(parts[:i]))
        if base_key in folder_index:
            variant = "-".join(parts[i:]).lower()
            return folder_index[base_key], variant

    return None, None


# ---------------------------------------------------------------------------
# Per-folder image index
# ---------------------------------------------------------------------------

def load_sprites(folder: Path) -> list[SpriteFile]:
    """Return all valid (non-gmax) SpriteFile objects from a folder."""
    sprites = []
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() != ".png":
            continue
        sf = SpriteFile.parse(img)
        if sf is not None:
            sprites.append(sf)
    return sprites


def build_sprite_lookup(
    sprites: list[SpriteFile],
) -> dict[tuple[int, str], tuple[SpriteFile | None, SpriteFile | None]]:
    """
    Build { (variant, gender) -> (normal_sprite, shiny_sprite) }.
    """
    lookup: dict[tuple[int, str], list[SpriteFile | None]] = {}
    for sf in sprites:
        key = (sf.variant, sf.gender)
        if key not in lookup:
            lookup[key] = [None, None]   # [normal, shiny]
        if sf.is_shiny:
            lookup[key][1] = sf
        else:
            lookup[key][0] = sf
    return {k: (v[0], v[1]) for k, v in lookup.items()}

def gender_priority_list(gender: str) -> list[str]:
    """
    Returns fallback priority list for genders.
    """
    if gender == "mf":
        return ["mf", "md", "fd"]
    if gender == "md":
        return ["md", "mf"]
    if gender == "fd":
        return ["fd", "mf"]
    # rare/edge cases
    return [gender, "mf", "md", "fd", "uk", "mo", "fo"]


def find_best_next(
    variant: int,
    gender: str,
    next_lookup: dict[tuple[int, str], tuple[SpriteFile | None, SpriteFile | None]],
) -> tuple[SpriteFile | None, SpriteFile | None]:
    """
    Improved matching with gender-aware fallback.

    Priority:
        1. Exact (variant, gender)
        2. (0, gender)
        3. (variant, compatible gender)
        4. (0, compatible gender)
        5. (variant, any gender)
        6. (0, any gender)
    """

    gender_priority = gender_priority_list(gender)

    # 1 & 2: exact + same gender fallback
    for g in gender_priority:
        if (variant, g) in next_lookup:
            return next_lookup[(variant, g)]
    for g in gender_priority:
        if (0, g) in next_lookup:
            return next_lookup[(0, g)]

    # 5: same variant, ANY gender
    for (v, g), pair in next_lookup.items():
        if v == variant:
            return pair

    # 6: default variant, ANY gender
    for (v, g), pair in next_lookup.items():
        if v == 0:
            return pair

    return None, None
# def find_best_next(
#     variant: int,
#     gender: str,
#     next_lookup: dict[tuple[int, str], tuple[SpriteFile | None, SpriteFile | None]],
# ) -> tuple[SpriteFile | None, SpriteFile | None]:
#     """
#     Return the best (normal, shiny) pair from next_lookup for given (variant, gender).

#     Fallback chain:
#         1. Exact (variant, gender)
#         2. (0, gender)         — same gender, default variant
#         3. (variant, "mf")     — default gender, same variant
#         4. (0, "mf")           — default gender, default variant
#     """
#     for v, g in [(variant, gender), (0, gender), (variant, "mf"), (0, "mf")]:
#         if (v, g) in next_lookup:
#             return next_lookup[(v, g)]
#     return None, None


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

_FOLDER_NAME_RE = re.compile(r"^\d+(.+)$")

def folder_to_pokemon_name(folder: Path) -> str:
    """Extract pokemon name from folder like '0001bulbasaur' -> 'bulbasaur'."""
    m = _FOLDER_NAME_RE.match(folder.name)
    return m.group(1) if m else folder.name


def make_entry(
    prev_sf: SpriteFile | None,
    next_sf: SpriteFile | None,
) -> dict:
    def rel(sf: SpriteFile | None) -> str | None:
        return str(sf.path.relative_to(REPO_ROOT)) if sf else None

    def name(sf: SpriteFile | None) -> str | None:
        return folder_to_pokemon_name(sf.path.parent) if sf else None

    return {
        "prev":        name(prev_sf),
        "next":        name(next_sf),
        "prev_sprite": rel(prev_sf),
        "next_sprite": rel(next_sf),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(EVOLUTION_JSON, encoding="utf-8") as f:
        evolutions = json.load(f)

    folder_index = build_folder_index(PROJECT_ROOT)

    results: list[dict] = []
    unresolved: list[dict] = []

    for entry in evolutions:
        prev_name: str | None = entry.get("prev")
        next_name: str | None = entry.get("next")

        if not next_name:
            continue

        # ---- Resolve next ----
        next_folder, next_variant_str = resolve_folder(next_name, folder_index)
        if next_folder is None:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": "next folder not found",
            })
            continue
        if next_variant_str is not None:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": f"next is regional form '{next_variant_str}' — variant number unknown",
            })
            continue

        next_sprites = load_sprites(next_folder)
        next_lookup  = build_sprite_lookup(next_sprites)

        if not next_lookup:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": "next folder has no valid images",
            })
            continue

        # ---- Chain start (prev = null) ----
        if prev_name is None:
            for (var, gen), (normal_sf, shiny_sf) in sorted(next_lookup.items()):
                if normal_sf:
                    results.append(make_entry(None, normal_sf))
                if shiny_sf:
                    results.append(make_entry(None, shiny_sf))
            continue

        # ---- Resolve prev ----
        prev_folder, prev_variant_str = resolve_folder(prev_name, folder_index)
        if prev_folder is None:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": "prev folder not found",
            })
            continue
        if prev_variant_str is not None:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": f"prev is regional form '{prev_variant_str}' — variant number unknown",
            })
            continue

        prev_sprites = load_sprites(prev_folder)
        prev_lookup  = build_sprite_lookup(prev_sprites)

        if not prev_lookup:
            unresolved.append({
                "prev": prev_name, "next": next_name,
                "reason": "prev folder has no valid images",
            })
            continue

        # ---- Match and emit pairs ----
        for (var, gen), (prev_normal, prev_shiny) in sorted(prev_lookup.items()):
            next_normal, next_shiny = find_best_next(var, gen, next_lookup)

            if prev_normal:
                if next_normal:
                    results.append(make_entry(prev_normal, next_normal))
                else:
                    unresolved.append({
                        "prev": prev_name,
                        "next": next_name,
                        "reason": f"no NORMAL match for variant={var}, gender={gen}"
                    })
                # # normal → normal; if next has no normal fall back to its shiny
                # results.append(make_entry(prev_normal, next_normal or next_shiny))
            if prev_shiny:
                if next_shiny:
                    results.append(make_entry(prev_shiny, next_shiny))
                else:
                    unresolved.append({
                        "prev": prev_name,
                        "next": next_name,
                        "reason": f"no NORMAL match for variant={var}, gender={gen}"
                    })
                # # shiny → shiny; if next has no shiny fall back to its normal
                # results.append(make_entry(prev_shiny, next_shiny or next_normal))

    # ---- Write outputs ----
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent="\t")

    unresolved_report = {
        "total_pairs": len(results),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }
    with open(UNRESOLVED_JSON, "w", encoding="utf-8") as f:
        json.dump(unresolved_report, f, indent="\t")

    print(f"Written         : {OUTPUT_JSON}")
    print(f"Total pairs     : {len(results)}")
    print(f"Unresolved      : {len(unresolved)}")
    if unresolved:
        print(f"Unresolved log  : {UNRESOLVED_JSON}")


if __name__ == "__main__":
    main()
