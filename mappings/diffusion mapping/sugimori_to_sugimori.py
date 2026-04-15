"""
Generates sugimori_to_sugimori.json from evolutions/evolution.json.

For each evolution pair (prev -> next):
    - Maps every valid prev Sugimori image -> best-matching next Sugimori image.
    - Chain starts (prev=null) get one entry per valid image in next's folder.

Mega and Gigantamax images are skipped entirely.
All 1025 pokemon are present in poke-data/SugimoriSprites so nothing is removed.

Output
--
mappings/diffusion mapping/sugimori_to_sugimori.json
mappings/diffusion mapping/sugimori_to_sugimori_unresolved.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[2]
SUGIMORI_ROOT  = REPO_ROOT / "poke-data" / "SugimoriSprites"
EVOLUTION_JSON = REPO_ROOT / "evolutions" / "evolution.json"
OUTPUT_JSON    = Path(__file__).parent / "sugimori_to_sugimori.json"
UNRESOLVED_JSON = Path(__file__).parent / "sugimori_to_sugimori_unresolved.json"

# Words whose presence in an image stem marks it as a Mega / Gigantamax form.
_SKIP_WORDS = {"mega", "gigantamax"}

# Folder index

def _norm(s: str) -> str:
    """Collapse all separators (space, hyphen, dot, apostrophe) for fuzzy matching."""
    return re.sub(r"[.\-'\s]", "", s.lower())


def build_folder_index(root: Path) -> dict[str, Path]:
    """
    Maps normalize(folder_base_name) -> folder_path.
    Leading digits are stripped from folder names before normalizing.
    """
    index: dict[str, Path] = {}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        base = re.sub(r"^\d+", "", folder.name)   # strip "0001", "0006", …
        index[_norm(base)] = folder
    return index


def resolve_folder(evo_name: str, folder_index: dict[str, Path]) -> tuple[Path | None, str | None]:
    """
    Given a pokemon name from evolution.json return (folder, variant_str).

    Resolution order:
        1. Direct normalized match  →  variant = None  (it IS the base pokemon).
        2. Strip trailing hyphen-separated components right-to-left until we
            find a folder match  →  the stripped part becomes the variant string.

    Examples
    
    "bulbasaur"           → (bulbasaur_folder, None)
    "nidoran-f"           → (nidoran-f_folder, None)   # exact match
    "Meowth-Galar"        → (meowth_folder,    "galar")
    "Mr-Mime-Galar"       → (mr.mime_folder,   "galar")
    "basculin-white-striped" → (basculin_folder, "white-striped")
    "tapu-koko"           → (tapukoko_folder,  None)   # exact match
    """
    # 1. Exact match
    key = _norm(evo_name)
    if key in folder_index:
        return folder_index[key], None

    # 2. Progressive right-strip on hyphens
    parts = evo_name.split("-")
    for i in range(len(parts) - 1, 0, -1):
        base_key = _norm("-".join(parts[:i]))
        if base_key in folder_index:
            variant = "-".join(parts[i:]).lower()
            return folder_index[base_key], variant

    return None, None

# Image helpers

def valid_images(folder: Path) -> list[Path]:
    """PNGs in the folder, excluding Mega and Gigantamax forms."""
    imgs = []
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() != ".png":
            continue
        stem_lower = img.stem.lower()
        if any(w in stem_lower for w in _SKIP_WORDS):
            continue
        imgs.append(img)
    return imgs


def image_variant_map(images: list[Path]) -> dict[str, Path]:
    """
    Build { variant_key (lowercase) -> image_path } for a list of images.

    The variant is extracted by finding the longest common word-prefix across
    all stems, then stripping it.  The leftover (possibly empty) string is the
    variant key.

    e.g.  ["0001 Bulbasaur", "0001 Bulbasaur RB", "0001 Bulbasaur RG"]
            common prefix = "0001 Bulbasaur"
            variants      = {"": …, "rb": …, "rg": …}
    """
    if not images:
        return {}

    word_lists = [img.stem.split() for img in images]
    min_len = min(len(w) for w in word_lists)

    common_count = 0
    for i in range(min_len):
        if all(wl[i] == word_lists[0][i] for wl in word_lists):
            common_count = i + 1
        else:
            break

    common_prefix = " ".join(word_lists[0][:common_count])

    result: dict[str, Path] = {}
    for img in images:
        variant = img.stem[len(common_prefix):].strip().lower()
        result[variant] = img
    return result


def filter_by_variant(variant_map: dict[str, Path], variant_str: str) -> dict[str, Path]:
    """
    Keep only entries whose key contains at least one word from variant_str.
    e.g. variant_str="white-striped" keeps keys containing "white" or "striped".
    """
    words = set(variant_str.lower().split("-"))
    return {k: v for k, v in variant_map.items() if any(w in k for w in words)}

# Pair generation

def make_pairs(
    prev_vmap: dict[str, Path] | None,   # None → chain start (prev = null)
    next_vmap: dict[str, Path],
    repo_root: Path,
) -> list[dict]:
    """
    Generate (prev_image, next_image) JSON entries.

    Matching strategy (prev_variant → next_variant):
        - Exact key match first.
        - Fall back to the base image ("" key) in next.
        - If next has no base, use the first available next image.
    """
    pairs: list[dict] = []
    base_next = next_vmap.get("") or (next(iter(next_vmap.values()), None) if next_vmap else None)

    if prev_vmap is None:
        # Chain start: one null-prev entry per next image
        for next_img in next_vmap.values():
            pairs.append({
                "prev": None,
                "next": next_img.stem,
                "prev_sprite": None,
                "next_sprite": str(next_img.relative_to(repo_root)),
            })
        return pairs

    for variant_key, prev_img in prev_vmap.items():
        next_img = next_vmap.get(variant_key) or base_next
        pairs.append({
            "prev": prev_img.stem,
            "next": next_img.stem if next_img else None,
            "prev_sprite": str(prev_img.relative_to(repo_root)),
            "next_sprite": str(next_img.relative_to(repo_root)) if next_img else None,
        })

    return pairs

# Main

def main() -> None:
    with open(EVOLUTION_JSON, encoding="utf-8") as f:
        evolutions = json.load(f)

    folder_index = build_folder_index(SUGIMORI_ROOT)

    results: list[dict] = []
    unresolved: list[dict] = []

    for entry in evolutions:
        prev_name: str | None = entry.get("prev")
        next_name: str | None = entry.get("next")

        if not next_name:
            continue  # malformed entry

        #  Resolve next folder 
        next_folder, next_variant = resolve_folder(next_name, folder_index)
        if next_folder is None:
            unresolved.append({"prev": prev_name, "next": next_name, "missing": ["next_folder"]})
            continue

        next_imgs = valid_images(next_folder)
        next_vmap = image_variant_map(next_imgs)
        if next_variant:
            next_vmap = filter_by_variant(next_vmap, next_variant) or next_vmap

        #  Resolve prev folder (null prev = chain start) 
        if prev_name is None:
            pairs = make_pairs(None, next_vmap, REPO_ROOT)
            results.extend(pairs)
            continue

        prev_folder, prev_variant = resolve_folder(prev_name, folder_index)
        if prev_folder is None:
            unresolved.append({"prev": prev_name, "next": next_name, "missing": ["prev_folder"]})
            continue

        prev_imgs = valid_images(prev_folder)
        prev_vmap = image_variant_map(prev_imgs)
        if prev_variant:
            prev_vmap = filter_by_variant(prev_vmap, prev_variant) or prev_vmap

        if not prev_vmap:
            unresolved.append({"prev": prev_name, "next": next_name, "missing": ["prev_images"]})
            continue

        pairs = make_pairs(prev_vmap, next_vmap, REPO_ROOT)
        results.extend(pairs)

    #  Write outputs 
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent="\t")

    unresolved_report = {
        "total_pairs": len(results),
        "resolved": len(results),
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
