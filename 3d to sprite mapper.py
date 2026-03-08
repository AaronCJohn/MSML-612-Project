import json
import os
from pathlib import Path
from collections import defaultdict
import csv
import sys

# ==== CONFIG ====
POKEMON_JSON = Path("all_pokemon_safe.json")
MODELS_ROOT = Path(r"C:\Users\varen\VSCode Projects\MSML612\PokeProj\MSML-612-Project\poke-data\ProjectPokemon")
SPRITES_ROOT = Path(r"C:\Users\varen\VSCode Projects\MSML612\PokeProj\MSML-612-Project\poke-data\PokeSprite")
OUTPUT_MAPPING_JSON = Path("mapping_3d_to_sprite.json")
OUTPUT_MAPPING_CSV = Path("mapping_3d_to_sprite.csv")
LOG_FILE = Path("mapping_log.txt")

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

def fuzzy_match_sprite(poke_name, sprite_dict, threshold=80):
    """Find best sprite match using fuzzy substring matching. Returns (match_name, (folder, file)) or None"""
    best_score = 0
    best_match = None
    
    for sprite_base in sprite_dict.keys():
        # Check if poke_name is contained in sprite_base (handles farfetch'd -> farfetch_d)
        if poke_name.lower() in sprite_base.lower():
            score = 100  # Perfect substring match
        else:
            # Simple character overlap score
            common_chars = set(poke_name.lower()) & set(sprite_base.lower())
            score = len(common_chars) / max(len(poke_name), len(sprite_base)) * 100
            
        if score > best_score and score >= threshold:
            best_score = score
            best_match = sprite_base
    
    if best_match:
        print(f"[FUZZY MATCH] '{poke_name}' -> '{best_match}' (score: {best_score:.1f}%)")
        return best_match, sprite_dict[best_match]
    return None

def generate_candidate_names(poke_name, parsed):

    candidates = []

    # ---------- GMAX ----------
    if parsed["is_gmax"]:
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-gmax_shiny")
        candidates.append(f"{poke_name}-gmax")

    # ---------- VARIANT (Mega / regional) ----------
    if parsed["variant"] != "000":

        # Mega X/Y
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-mega-x_shiny")
            candidates.append(f"{poke_name}-mega-y_shiny")

        candidates.append(f"{poke_name}-mega-x")
        candidates.append(f"{poke_name}-mega-y")

        # Generic mega
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-mega_shiny")

        candidates.append(f"{poke_name}-mega")

    # ---------- FEMALE ----------
    if parsed["gender"] == "female":
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}_shiny_female")
        candidates.append(f"{poke_name}_female")

    # ---------- BASE ----------
    if parsed["is_shiny"]:
        candidates.append(f"{poke_name}_shiny")

    candidates.append(poke_name)

    return candidates

def try_sprite_variants(sprite_dict, poke_name, parsed):
    """
    Generate candidate sprite names in priority order.
    Supports: Gmax, Mega, Mega X/Y, regional forms, female, shiny, base
    """

    candidates = []

    # ---------- GMAX ----------
    if parsed["is_gmax"]:
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-gmax_shiny")
        candidates.append(f"{poke_name}-gmax")

    # ---------- MEGA ----------
    if parsed["variant"] != "000" and not parsed.get("is_regional", False):
        # Mega X/Y support
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-mega-x_shiny")
            candidates.append(f"{poke_name}-mega-y_shiny")
        candidates.append(f"{poke_name}-mega-x")
        candidates.append(f"{poke_name}-mega-y")
        # generic mega
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}-mega_shiny")
        candidates.append(f"{poke_name}-mega")

    # ---------- REGIONAL FORMS ----------
    if parsed["variant"] != "000" and not parsed["is_gmax"] and not parsed.get("is_mega", False):
        # Example: Alola
        form_name = "alola"
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name} {form_name}_shiny")
        candidates.append(f"{poke_name} {form_name}")

    # ---------- FEMALE ----------
    if parsed["gender"] == "female":
        if parsed["is_shiny"]:
            candidates.append(f"{poke_name}_shiny_female")
        candidates.append(f"{poke_name}_female")

    # ---------- BASE ----------
    if parsed["is_shiny"]:
        candidates.append(f"{poke_name}_shiny")
    candidates.append(poke_name)

    # ---------- SEARCH ----------
    for name in candidates:
        if name in sprite_dict:
            print(f"[MATCH] {name}")
            return sprite_dict[name]

    # ---------- FUZZY FALLBACK ----------
    fuzzy_result = fuzzy_match_sprite(poke_name, sprite_dict)
    if fuzzy_result:
        return fuzzy_result[1]

    return None

# def try_sprite_variants(sprite_dict, poke_name, parsed):
#     """
#     Generate candidate sprite names in priority order.
#     """

#     candidates = []

#     # ---------- GMAX ----------
#     if parsed["is_gmax"]:
#         if parsed["is_shiny"]:
#             candidates.append(f"{poke_name}-gmax_shiny")
#         candidates.append(f"{poke_name}-gmax")

#     # ---------- MEGA ----------
#     if parsed["variant"] != "000":
#         # Generic mega
#         if parsed["is_shiny"]:
#             candidates.append(f"{poke_name}-mega_shiny")
#         candidates.append(f"{poke_name}-mega")

#     # ---------- FEMALE ----------
#     if parsed["gender"] == "female":
#         if parsed["is_shiny"]:
#             candidates.append(f"{poke_name}_shiny_female")
#         candidates.append(f"{poke_name}_female")

#     # ---------- BASE ----------
#     if parsed["is_shiny"]:
#         candidates.append(f"{poke_name}_shiny")
#     candidates.append(poke_name)

#     # ---------- SEARCH ----------
#     for name in candidates:
#         if name in sprite_dict:
#             print(f"[MATCH] {name}")
#             return sprite_dict[name]

#     # ---------- FUZZY FALLBACK ----------
#     fuzzy_result = fuzzy_match_sprite(poke_name, sprite_dict)
#     if fuzzy_result:
#         return fuzzy_result[1]

#     return None

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

def parse_model_filename(fname: str):
    """Parse poke_capture_XXXX_YYY_ZZ_A_BBBBBBBB_C_D.png format"""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")

    if len(parts) < 9 or parts[0] != "poke" or parts[1] != "capture":
        return None

    dex = parts[2]
    variant = parts[3]
    gender_code = parts[4]
    form_code = parts[5]
    shiny_code = parts[8]

    is_shiny = shiny_code == "r"
    is_gmax = form_code == "g"

    # Gender mapping
    if gender_code in ("fd", "fo"):
        gender = "female"
    elif gender_code in ("md", "mo"):
        gender = "male"
    else:
        gender = "default"

    return {
        "dex": dex,
        "variant": variant,
        "gender": gender,
        "is_shiny": is_shiny,
        "is_gmax": is_gmax,
    }

def sprite_name_for(poke_name: str, parsed: dict):
    """Build ideal sprite name (before fallback)"""
    pieces = [poke_name]

    # Form first (gmax > mega > base)
    if parsed["is_gmax"]:
        pieces.append("gmax")
    elif parsed["form"] == "mega":
        pieces.append("mega")

    # Shiny + gender
    if parsed["is_shiny"]:
        if parsed["gender"] == "female" and parsed["form"] == "base" and not parsed["is_gmax"]:
            pieces.append("shiny_female")
        else:
            pieces.append("shiny")
    elif parsed["gender"] == "female" and parsed["form"] == "base" and not parsed["is_gmax"]:
        pieces.append("female")

    return "_".join(pieces)

def build_mapping():
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger

    dex_to_name, _ = load_pokemon_names(POKEMON_JSON)

    mapping_rows = []
    unmapped_models = []
    missing_sprites = []
    extra_sprites = []

    # Index sprites
    sprite_index = defaultdict(dict)
    for folder in SPRITES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name
        dex = folder_name[:4]
        for f in folder.glob("*.png"):
            base = f.stem
            sprite_index[dex][base] = (folder_name, f.name)

    used_sprite_keys = set()

    # Map models to sprites
    for folder in MODELS_ROOT.iterdir():
        if not folder.is_dir():
            continue
        folder_name = folder.name
        dex = folder_name[:4]
        poke_name = dex_to_name.get(dex)
        if not poke_name:
            print(f"[WARN] Unknown dex folder in models: {folder_name}")
            continue

        for f in folder.glob("*.png"):
            parsed = parse_model_filename(f.name)
            if not parsed:
                unmapped_models.append((folder_name, f.name))
                continue

            if parsed["dex"] != dex:
                print(f"[WARN] Dex mismatch: folder {dex}, file {parsed['dex']} ({folder_name}/{f.name})")

            sprite_dict = sprite_index.get(dex, {})
            sprite_info = try_sprite_variants(sprite_dict, poke_name, parsed)

            if sprite_info is None:
                unmapped_models.append((folder_name, f.name))
                expected_candidates = generate_candidate_names(poke_name, parsed)
                missing_sprites.append((folder_name, f.name, expected_candidates, dex))
            else:
                sprite_folder, sprite_file = sprite_info
                mapping_rows.append({
                    "model_folder": folder_name,
                    "model_file": f.name,
                    "sprite_folder": sprite_folder,
                    "sprite_file": sprite_file,
                })
                used_sprite_keys.add((sprite_folder, sprite_file))

    # Find unused sprites
    for dex, sprites in sprite_index.items():
        for base, (folder_name, filename) in sprites.items():
            if (folder_name, filename) not in used_sprite_keys:
                extra_sprites.append((folder_name, filename))

    # Save outputs
    mapping_dict = {f"{row['model_folder']}/{row['model_file']}": f"{row['sprite_folder']}/{row['sprite_file']}" 
                   for row in mapping_rows}

    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as out:
        json.dump(mapping_dict, out, indent=2, ensure_ascii=False)

    with open(OUTPUT_MAPPING_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["model_folder", "model_file", "sprite_folder", "sprite_file"])
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    # Log results
    print("=== Mapping complete ===")
    print(f"Total mapped: {len(mapping_rows)}")
    
    if unmapped_models:
        print("\n[UNMAPPED MODEL FILES]")
        for folder_name, fname in unmapped_models:
            print(f"{folder_name}/{fname}")

    if missing_sprites:
        print("\n[MISSING SPRITES FOR MODELS]")
        for folder_name, fname, expected, dex in missing_sprites:
            print(f"Model: {folder_name}/{fname} | Expected: {expected} (dex {dex})")

    if extra_sprites:
        print("\n[UNUSED SPRITE FILES]")
        for folder_name, fname in extra_sprites:
            print(f"{folder_name}/{fname}")

    sys.stdout = logger.stdout
    logger.close()

if __name__ == "__main__":
    build_mapping()