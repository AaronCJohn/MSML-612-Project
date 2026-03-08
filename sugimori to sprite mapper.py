import json
import os
from pathlib import Path
from collections import defaultdict
import csv
import sys

# ==== CONFIG ====
POKEMON_JSON = Path("all_pokemon_safe.json")  # ["bulbasaur", "ivysaur", ...]
SUGIMORI_ROOT = Path(r"C:\Users\varen\VSCode Projects\MSML612\PokeProj\MSML-612-Project\poke-data\SugimoriSprites")  # contains 0001bulbasaur, 0002ivysaur, ...
SPRITES_ROOT = Path(r"C:\Users\varen\VSCode Projects\MSML612\PokeProj\MSML-612-Project\poke-data\PokeSprite")    # contains 0001bulbasaur, 0002ivysaur, ...
OUTPUT_MAPPING_JSON = Path("mapping_sugimori_to_sprite.json")
OUTPUT_MAPPING_CSV = Path("mapping_sugimori_to_sprite.csv")
LOG_FILE = Path("mapping_sugimori_log.txt")

# regions to detect in names
REGIONS = ["alola", "galar", "hisui", "paldea"]


# ==== LOGGING SETUP ====
class TeeLogger:
    def __init__(self, logfile_path):
        self.logfile = open(logfile_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.stdout.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


# ==== HELPERS ====
def load_pokemon_names(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    dex_to_name = {}
    name_to_dex = {}
    for i, name in enumerate(names, start=1):
        dex_str = f"{i:04d}"
        dex_to_name[dex_str] = name
        name_to_dex[name] = dex_str
    return dex_to_name, name_to_dex


def normalize_name_for_match(name: str) -> str:
    """Lowercase, replace spaces and apostrophes for fuzzy folder/sprite matching."""
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("’", "")
        .replace("'", "")
        .replace(".", "")
        .replace("é", "e")
    )


def parse_sugimori_filename(fname: str):
    """
    Examples:
      '0001 Bulbasaur.png'
      '0006 Charizard Mega Y.png'
      '0018 Pidgeot Mega.png'
      '0001 Bulbasaur RB.png'
      '0001 Bulbasaur RG.png'
      '0019 Rattata Alola.png'
      '0006 Charizard Gigantamax.png'
      '0006 Charizard Gigantamax 2.png'
    """
    stem = os.path.splitext(fname)[0]
    parts = stem.split(" ", 1)
    if len(parts) < 2:
        return None

    dex = parts[0]  # "0001"
    rest = parts[1]  # "Bulbasaur", "Charizard Mega Y", etc.
    tokens = rest.split(" ")

    # base name accumulates until we hit a recognized keyword
    name_tokens = []
    form = "base"      # base / mega / gmax
    shiny = False      # Sugimori art is typically non-shiny, keep for completeness
    region = None      # alola / galar / hisui / paldea
    game = None        # RB / RG

    i = 0
    while i < len(tokens):
        t = tokens[i]

        low = t.lower()
        # Game tags
        if t in ("RB", "RG"):
            game = t
            i += 1
            continue

        # Region tags
        if low in REGIONS:
            region = low
            i += 1
            continue

        # Mega
        if low == "mega":
            form = "mega"
            i += 1
            # Optional X/Y
            if i < len(tokens) and tokens[i] in ("X", "Y"):
                # You might distinguish between mega-x/mega-y later if you have separate sprites
                # For now we treat both as 'mega'
                i += 1
            continue

        # Gigantamax
        if low.startswith("gigantamax"):
            form = "gmax"
            i += 1
            # Optional "2"
            if i < len(tokens) and tokens[i] == "2":
                i += 1
            continue

        # Default: part of name
        name_tokens.append(t)
        i += 1

    poke_name = " ".join(name_tokens)
    return {
        "dex": dex,
        "poke_name_raw": poke_name,
        "form": form,
        "is_shiny": shiny,
        "region": region,
        "game": game,
    }


def sprite_basename_from_parsed(poke_name: str, parsed: dict):
    """
    Build ideal sprite basename (without .png) from parsed Sugimori info.
    Sprite patterns (PokéSprite style):
      {name}
      {name}-shiny (note dash!)
      {name}-gmax  
      {name}-gmax-shiny
      {name}-mega (DASH, not underscore!)
      {name}-mega-shiny
      {name}-mega-x
      {name}-mega-y
      {name}-female
      {name}-shiny-female
      {name}-alola, {name}-galar, etc.
    """
    base = poke_name

    names = []

    # Form variants with PokéSprite dash convention
    if parsed["form"] == "gmax":
        if parsed["is_shiny"]:
            names.append(f"{base}-gmax-shiny")
        names.append(f"{base}-gmax")
    elif parsed["form"] == "mega":
        if parsed["is_shiny"]:
            names.append(f"{base}-mega-shiny")
        names.append(f"{base}-mega")
        # Also try mega-x/mega-y for dual-mega Pokémon
        names.append(f"{base}-mega-x")
        names.append(f"{base}-mega-y")
    else:
        # base form
        if parsed["is_shiny"]:
            names.append(f"{base}-shiny")
        names.append(base)

    # Region variants (dash first)
    region_names = []
    if parsed["region"]:
        r = parsed["region"]
        for n in names:
            region_names.extend([f"{n}-{r}", f"{n}_{r}", f"{n} {r}"])  # dash, underscore, space variants

    # Region variants first, then non-region
    all_names = region_names + names

    # unique while preserving order
    seen = set()
    ordered = []
    for n in all_names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def fuzzy_sprite_match(poke_name: str, sprite_dict: dict):
    """
    Very simple fuzzy: match on normalized name containment.
    """
    norm = normalize_name_for_match(poke_name)
    best = None
    for base in sprite_dict.keys():
        if norm in normalize_name_for_match(base):
            best = base
            break
    if best:
        return sprite_dict[best]
    return None


def try_sprite_for_sugimori(sprite_dict, candidates, poke_name, parsed):
    """
    Try list of candidate sprite basenames in order, then fuzzy fallback.
    """
    # exact candidates
    for cand in candidates:
        if cand in sprite_dict:
            return sprite_dict[cand]

    # fuzzy fallback
    return fuzzy_sprite_match(poke_name, sprite_dict)


# ==== MAIN ====
def build_mapping():
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger

    dex_to_name, _ = load_pokemon_names(POKEMON_JSON)

    mapping_rows = []
    unmapped_art = []
    extra_sprites = []

    # Index sprites: {dex: {basename: (folder_name, filename)}}
    sprite_index = defaultdict(dict)
    for folder in SPRITES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name   # "0001bulbasaur"
        dex = folder_name[:4]
        for f in folder.glob("*.png"):
            base = f.stem
            sprite_index[dex][base] = (folder_name, f.name)

    used_sprite_keys = set()

    # Map Sugimori art to sprites
    for folder in SUGIMORI_ROOT.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name   # "0001bulbasaur"
        dex = folder_name[:4]
        poke_name_json = dex_to_name.get(dex)
        if not poke_name_json:
            print(f"[WARN] Unknown dex folder in Sugimori: {folder_name}")
            continue

        sprites_for_dex = sprite_index.get(dex, {})

        for f in folder.glob("*.png"):
            parsed = parse_sugimori_filename(f.name)
            if not parsed:
                unmapped_art.append((folder_name, f.name, "parse_failed"))
                continue

            if parsed["dex"] != dex:
                print(f"[WARN] Dex mismatch: folder {dex}, file {parsed['dex']} ({folder_name}/{f.name})")

            # prefer folder/json-based name for sprite naming (since sprite naming uses that)
            poke_name = poke_name_json

            candidates = sprite_basename_from_parsed(poke_name, parsed)
            sprite_info = try_sprite_for_sugimori(sprites_for_dex, candidates, poke_name, parsed)

            if sprite_info is None:
                unmapped_art.append((folder_name, f.name, ", ".join(candidates)))
            else:
                sprite_folder, sprite_file = sprite_info
                mapping_rows.append(
                    {
                        "art_folder": folder_name,
                        "art_file": f.name,
                        "sprite_folder": sprite_folder,
                        "sprite_file": sprite_file,
                    }
                )
                used_sprite_keys.add((sprite_folder, sprite_file))

    # Identify unused sprites
    for dex, sprites in sprite_index.items():
        for base, (folder_name, filename) in sprites.items():
            if (folder_name, filename) not in used_sprite_keys:
                extra_sprites.append((folder_name, filename))

    # Save mapping JSON (folder/file only)
    mapping_dict = {}
    for row in mapping_rows:
        key = f"{row['art_folder']}/{row['art_file']}"
        val = f"{row['sprite_folder']}/{row['sprite_file']}"
        mapping_dict[key] = val

    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as out:
        json.dump(mapping_dict, out, indent=2, ensure_ascii=False)

    # Save mapping CSV
    with open(OUTPUT_MAPPING_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["art_folder", "art_file", "sprite_folder", "sprite_file"],
        )
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    # Logging
    print("=== Sugimori→Sprite mapping complete ===")
    print(f"Total mapped: {len(mapping_rows)}")

    if unmapped_art:
        print("\n[UNMAPPED SUGIMORI FILES]")
        for folder_name, fname, expected in unmapped_art:
            print(f"{folder_name}/{fname} | Tried: {expected}")

    if extra_sprites:
        print("\n[UNUSED SPRITE FILES]")
        for folder_name, fname in extra_sprites:
            print(f"{folder_name}/{fname}")

    sys.stdout = logger.stdout
    logger.close()


if __name__ == "__main__":
    build_mapping()