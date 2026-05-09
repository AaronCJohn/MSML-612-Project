"""
Checks whether every Pokemon listed in evolutions/evolution.json appears at
least once (as prev or next) in each mapping JSON in this folder.

Coverage is determined by normalised name matching: separators (hyphens,
spaces, dots, apostrophes) are stripped and comparison is case-insensitive.
Each mapping entry contributes two name tokens per side:

    1. The 'prev'/'next' field  (folder-derived base name, e.g. "arcanine")
    2. The image stem extracted from 'prev_sprite'/'next_sprite'
        (e.g. "0059 Arcanine Hisui" → includes the variant suffix)

This catches regional forms like "Arcanine-Hisui" whose base folder name is
just "arcanine" but whose sprite stem contains "arcaninehisui".

Reports:
    - Per-mapping coverage count / percentage and list of missing pokemon
    - Pokemon missing from ALL checked mappings

Usage:
    python check_coverage.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[2]
EVOLUTION_JSON = REPO_ROOT / "evolutions" / "evolution.json"
MAPPING_DIR    = Path(__file__).parent

MAPPING_FILES = [
    MAPPING_DIR / "sugimori_to_sugimori.json",
    MAPPING_DIR / "sprite_to_sprite.json",
    MAPPING_DIR / "project_to_project.json"
]

LOG_FILE = MAPPING_DIR / "check_coverage.log"


def _norm(s: str) -> str:
    """Lowercase and strip all separators for fuzzy comparison."""
    return re.sub(r"[.\-'\s]", "", s.lower())


def all_evo_pokemon(evolution_json: Path) -> dict[str, str]:
    """
    Return { norm_name -> display_name } for every unique pokemon in
    evolution.json (both prev and next sides).
    """
    with open(evolution_json, encoding="utf-8") as f:
        data = json.load(f)
    pokemon: dict[str, str] = {}
    for entry in data:
        for field in ("prev", "next"):
            name = entry.get(field)
            if name:
                pokemon[_norm(name)] = name
    return pokemon


def _norm_tokens_from_entry(entry: dict) -> set[str]:
    """
    Extract normalised name tokens from one mapping entry.

    Two sources per side:
        - The 'prev'/'next' name field (base folder name).
        - The image stem from 'prev_sprite'/'next_sprite' (includes variant).
    """
    tokens: set[str] = set()
    for name_key, sprite_key in (("prev", "prev_sprite"), ("next", "next_sprite")):
        name = entry.get(name_key)
        if name:
            tokens.add(_norm(name))

        sprite = entry.get(sprite_key)
        if sprite:
            stem = Path(sprite).stem          # e.g. "0059 Arcanine Hisui"
            tokens.add(_norm(stem))           # e.g. "0059arcaninehisui"
    return tokens


def covered_pokemon(
    mapping_json: Path, evo_norm_names: set[str]
) -> set[str]:
    """
    Return the subset of evo_norm_names that are covered by this mapping.

    A pokemon is considered covered if its normalised name is a substring of
    any normalised token produced by the mapping entries.
    """
    with open(mapping_json, encoding="utf-8") as f:
        data = json.load(f)

    # Build one big set of all normalised tokens from the mapping
    all_tokens: set[str] = set()
    for entry in data:
        all_tokens |= _norm_tokens_from_entry(entry)

    covered: set[str] = set()
    for norm_name in evo_norm_names:
        # A pokemon is covered if its norm name appears as a substring in any token
        if any(norm_name in token for token in all_tokens):
            covered.add(norm_name)
    return covered


def main() -> None:
    evo_pokemon = all_evo_pokemon(EVOLUTION_JSON)   # norm -> display
    evo_norms   = set(evo_pokemon)

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"Total unique Pokemon in evolution.json : {len(evo_norms)}")
    out()

    covered_sets: dict[str, set[str]] = {}

    for mapping_path in MAPPING_FILES:
        if not mapping_path.exists():
            out(f"[SKIP] {mapping_path.name}, file not found")
            out()
            continue

        covered = covered_pokemon(mapping_path, evo_norms)
        missing_norms = evo_norms - covered
        covered_sets[mapping_path.name] = covered

        pct = 100 * len(covered) / len(evo_norms)
        out("─" * 60)
        out(f"Mapping : {mapping_path.name}")
        out(f"Covered : {len(covered)} / {len(evo_norms)}  ({pct:.1f}%)")
        out(f"Missing : {len(missing_norms)}")
        if missing_norms:
            for norm in sorted(missing_norms):
                out(f"  - {evo_pokemon[norm]}")
        out()

    # Summary: missing from ALL mappings
    checked = [p for p in MAPPING_FILES if p.name in covered_sets]
    if len(checked) > 1:
        globally_covered = set.intersection(*(covered_sets[p.name] for p in checked))
        globally_missing = evo_norms - globally_covered
        out("═" * 60)
        out(f"Missing from ALL mappings : {len(globally_missing)}")
        if globally_missing:
            for norm in sorted(globally_missing):
                out(f"  - {evo_pokemon[norm]}")

    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog written to {LOG_FILE}")


if __name__ == "__main__":
    main()
