"""
Generates sprite_to_sprite.json from evolutions/evolution.json.

For each evolution pair (prev -> next):
    - Maps prev sprite image -> next sprite image (normal)
    - Maps prev shiny image  -> next shiny image  (shiny)

If prev is null, the entry is kept with null prev (chain start).
If a sprite cannot be found in poke-data/PokeSprite, the path is set to null
and the pair is recorded in sprite_to_sprite_unresolved.json.

Pokemon in REMOVE_POKEMON (Gen 9 / #906+, no sprites available) are skipped entirely.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPRITE_ROOT = REPO_ROOT / "poke-data" / "PokeSprite"
EVOLUTION_JSON = REPO_ROOT / "evolutions" / "evolution.json"
OUTPUT_JSON = Path(__file__).parent / "sprite_to_sprite.json"
UNRESOLVED_JSON = Path(__file__).parent / "sprite_to_sprite_unresolved.json"

# Pokemon with no sprites (Gen 9 / #906+ and a handful of missing variants).
# Any evolution entry whose prev or next appears here is skipped.
REMOVE_POKEMON = {
    # Gen 9 starters & evolutions
    "sprigatito", "floragato", "meowscarada",
    "fuecoco", "crocalor", "skeledirge",
    "quaxly", "quaxwell", "quaquaval",
    # Gen 9 pokemon
    "lechonk", "oinkologne",
    "tarountula", "spidops",
    "nymble", "lokix",
    "pawmi", "pawmo", "pawmot",
    "tandemaus", "maushold",
    "fidough", "dachsbun",
    "smoliv", "dolliv", "arboliva",
    "squawkabilly",
    "nacli", "naclstack", "garganacl",
    "charcadet", "armarouge", "ceruledge",
    "tadbulb", "bellibolt",
    "wattrel", "kilowattrel",
    "maschiff", "mabosstiff",
    "shroodle", "grafaiai",
    "bramblin", "brambleghast",
    "toedscool", "toedscruel",
    "capsakid", "scovillain",
    "rellor", "rabsca",
    "flittle", "espathra",
    "tinkatink", "tinkatuff", "tinkaton",
    "wiglett", "wugtrio",
    "bombirdier",
    "cetoddle", "cetitan",
    "varoom", "revavroom",
    "cyclizar",
    "orthworm",
    "greavard", "houndstone",
    "flamigo",
    "klawf",
    "finizen", "palafin", "palafin-hero",
    "annihilape",
    "clodsire",
    "farigiraf",
    "dudunsparce",
    "kingambit",
    "great-tusk", "scream-tail", "brute-bonnet", "flutter-mane",
    "slither-wing", "sandy-shocks", "iron-treads", "iron-bundle",
    "iron-hands", "iron-jugulis", "iron-moth", "iron-thorns",
    "iron-valiant", "roaring-moon", "walking-wake", "gouging-fire",
    "raging-bolt", "iron-boulder", "iron-crown",
    "koraidon", "miraidon",
    "wo-chien", "chien-pao", "ting-lu", "chi-yu",
    "gimmighoul", "gimmighoul-roaming", "gholdengo",
    "ogerpon", "ogerpon-cornerstone-mask", "ogerpon-hearthflame-mask", "ogerpon-wellspring-mask",
    "dipplin", "hydrapple",
    "archaludon",
    "fezandipiti", "okidogi", "munkidori",
    "terapagos", "terapagos-terastal", "terapagos-stellar",
    "pecharunt",
    "sinistcha", "poltchageist",
    "dondozo",
    "tatsugiri", "tatsugiri-droopy", "tatsugiri-stretchy",
    "veluza",
    "glimmet", "glimmora",
    "ursaluna-bloodmoon",
    # Missing variants (no dedicated sprite file)
    "giratina-altered",
    "calyrex-ice", "calyrex-shadow",
    "necrozma-dawn-wings", "necrozma-dusk-mane",
    "oricorio-pompom",
    "zygarde-10", "zygarde-complete",
}

# Name aliases: evolution.json name (lowercase) -> sprite file stem
# Covers pokemon whose folder/file uses special characters.
NAME_ALIASES = {
    "farfetchd":        "farfetch_d",
    "farfetchd-galar":  "farfetch_d_galar",
    "sirfetchd":        "sirfetch_d",
    "mr-mime":          "mr. mime",
    "mr-mime-galar":    "mr. mime_galar",
    "mr-rime":          "mr. rime",
}


def build_sprite_index(sprite_root: Path) -> dict[str, str]:
    """
    Walk every folder in PokeSprite and build a mapping:
        lowercase-stem -> relative image path

    Both normal and shiny entries are indexed separately:
        "charizard"       -> "poke-data/PokeSprite/0006charizard/charizard.png"
        "charizard_shiny" -> "poke-data/PokeSprite/0006charizard/charizard_shiny.png"
    """
    index: dict[str, str] = {}
    for folder in sorted(sprite_root.iterdir()):
        if not folder.is_dir() or folder.name == "processed":
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif"}:
                continue
            stem = img.stem.lower()
            rel = str(img.relative_to(REPO_ROOT))
            index[stem] = rel
    return index


def resolve_sprite(name: str, shiny: bool, index: dict[str, str]) -> str | None:
    """
    Given a pokemon name from evolution.json, return the relative path to its sprite.

    Resolution order:
        1. Check NAME_ALIASES for special-character remaps.
        2. Direct lowercase lookup.
        3. Hyphen -> space fallback (covers regional forms like Meowth-Galar).

    Returns None if no matching image is found.
    """
    if name is None:
        return None

    base = name.lower()
    shiny_suffix = "_shiny" if shiny else ""

    # Apply alias if present
    base = NAME_ALIASES.get(base, base)

    # Primary lookup
    key = base + shiny_suffix
    if key in index:
        return index[key]

    # Fallback: hyphens -> spaces (e.g. "meowth-galar" -> "meowth galar")
    key_spaced = base.replace("-", " ") + shiny_suffix
    if key_spaced in index:
        return index[key_spaced]

    return None


def main():
    with open(EVOLUTION_JSON, encoding="utf-8") as f:
        evolutions = json.load(f)

    index = build_sprite_index(SPRITE_ROOT)

    results = []
    seen: set[tuple] = set()
    unresolved = []
    skipped = 0

    for entry in evolutions:
        prev_name = entry.get("prev")
        next_name = entry.get("next")

        # Skip entries that involve a pokemon with no sprites
        if (prev_name and prev_name.lower() in REMOVE_POKEMON) or \
            (next_name and next_name.lower() in REMOVE_POKEMON):
            skipped += 1
            continue

        for shiny in (False, True):
            suffix = " shiny" if shiny else ""

            prev_sprite = resolve_sprite(prev_name, shiny, index) if prev_name else None
            next_sprite = resolve_sprite(next_name, shiny, index) if next_name else None

            # Collect unresolved sides (null prev is intentional, not a miss)
            missing = []
            if prev_name and prev_sprite is None:
                missing.append("prev")
            if next_name and next_sprite is None:
                missing.append("next")
            if missing:
                unresolved.append(
                    {
                        "prev": f"{prev_name}{suffix}" if prev_name else None,
                        "next": f"{next_name}{suffix}" if next_name else None,
                        "missing": missing,
                    }
                )

            key = (prev_sprite, next_sprite)
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "prev": f"{prev_name}{suffix}" if prev_name else None,
                        "next": f"{next_name}{suffix}" if next_name else None,
                        "prev_sprite": prev_sprite,
                        "next_sprite": next_sprite,
                    }
                )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent="\t")

    unresolved_report = {
        "total_pairs": len(results),
        "resolved": len(results) - len(unresolved),
        "unresolved_count": len(unresolved),
        "skipped_no_sprite": skipped,
        "unresolved": unresolved,
    }
    with open(UNRESOLVED_JSON, "w", encoding="utf-8") as f:
        json.dump(unresolved_report, f, indent="\t")

    print(f"Written         : {OUTPUT_JSON}")
    print(f"Total pairs     : {len(results)} (normal + shiny)")
    print(f"Resolved        : {len(results) - len(unresolved)} / {len(results)}")
    print(f"Unresolved      : {len(unresolved)} / {len(results)}")
    print(f"Skipped (no sprite): {skipped} evolution entries")
    if unresolved:
        print(f"Unresolved log  : {UNRESOLVED_JSON}")


if __name__ == "__main__":
    main()
