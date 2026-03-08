import csv
from pathlib import Path

# ==== CONFIG ====
SPRITES_ROOT = Path(r"C:\Users\varen\VSCode Projects\MSML612\PokeProj\MSML-612-Project\poke-data\PokeSprite")
MAPPING_CSV = Path("mapping_sugimori_to_sprite.csv")
LOG_UNUSED = Path("unused_sprites.txt")
LOG_MISSING = Path("missing_sprites.txt")

def get_all_sprites(sprite_root: Path):
    """Return a set of all (folder, file) tuples in the sprite directory."""
    all_sprites = set()
    for folder in sprite_root.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.glob("*.png"):
            all_sprites.add((folder.name, f.name))
    return all_sprites

def get_mapped_sprites(mapping_csv: Path):
    """Return a set of all (folder, file) tuples used in the mapping CSV."""
    mapped_sprites = set()
    missing_sprites = []
    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sprite_tuple = (row["sprite_folder"], row["sprite_file"])
            sprite_path = SPRITES_ROOT / row["sprite_folder"] / row["sprite_file"]
            if not sprite_path.exists():
                missing_sprites.append(sprite_tuple)
            mapped_sprites.add(sprite_tuple)
    return mapped_sprites, missing_sprites

def main():
    all_sprites = get_all_sprites(SPRITES_ROOT)
    mapped_sprites, missing_sprites = get_mapped_sprites(MAPPING_CSV)

    # Unused sprites
    unused_sprites = all_sprites - mapped_sprites

    # Log unused sprites
    if unused_sprites:
        print(f"[INFO] {len(unused_sprites)} unused sprite(s) found. Logging to {LOG_UNUSED}")
        with open(LOG_UNUSED, "w", encoding="utf-8") as log:
            for folder, fname in sorted(unused_sprites):
                log.write(f"{folder}/{fname}\n")
                print(f"UNUSED: {folder}/{fname}")
    else:
        print("[INFO] All sprites are used in the mapping.")

    # Log missing or malformed sprite references
    if missing_sprites:
        print(f"[WARNING] {len(missing_sprites)} mapped sprite(s) missing or malformed. Logging to {LOG_MISSING}")
        with open(LOG_MISSING, "w", encoding="utf-8") as log:
            for folder, fname in sorted(missing_sprites):
                log.write(f"{folder}/{fname}\n")
                print(f"MISSING: {folder}/{fname}")
    else:
        print("[INFO] All mapped sprites exist in the sprite directory.")

if __name__ == "__main__":
    main()