import json
import requests

BASE_URL = "https://pokeapi.co/api/v2"

REGIONAL_SUFFIXES = ("-alola", "-galar", "-hisui", "-paldea")

FORM_VARIANTS = {
    "basculin-white-striped",
    "castform-sunny", "castform-rainy", "castform-snowy",
    "deoxys-attack", "deoxys-defense", "deoxys-speed",
    "burmy-trash", "burmy-sandy",
    "wormadam-trash", "wormadam-sandy",
    "cherrim-sunshine",
    "rotom-heat", "rotom-wash", "rotom-frost", "rotom-fan", "rotom-mow",
    "dialga-origin", "palkia-origin",
    "giratina-altered",
    "shaymin-sky",
    "darmanitan-zen",
    "tornadus-therian", "thundurus-therian", "landorus-therian", "enamorus-therian",
    "kyurem-white", "kyurem-black",
    "meloetta-pirouette",
    "zygarde-10", "zygarde-complete",
    "hoopa-unbound",
    "oricorio-pompom", "oricorio-pau", "oricorio-sensu",
    "wishiwashi-school",
    "necrozma-dusk-mane", "necrozma-dawn-wings", "necrozma-ultra",
    "eternatus-eternamax",
    "ursaluna-bloodmoon",
    "palafin-hero",
    "tatsugiri-droopy", "tatsugiri-stretchy",
    "gimmighoul-roaming",
    "ogerpon-wellspring-mask", "ogerpon-hearthflame-mask", "ogerpon-cornerstone-mask",
    "terapagos-terastal", "terapagos-stellar",
    "lycanroc-midnight",
    "calyrex-ice", "calyrex-shadow",
}


def is_regional_form(name: str) -> bool:
    lower = name.lower()
    return any(suffix in lower for suffix in REGIONAL_SUFFIXES)


def is_form_variant(name: str) -> bool:
    return name.lower() in FORM_VARIANTS


def get_species(name: str):
    url = f"{BASE_URL}/pokemon-species/{name}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def get_evolution_chain(chain_url: str):
    resp = requests.get(chain_url)
    resp.raise_for_status()
    return resp.json()


def build_evolution_pairs(chain_obj, valid_pokemon: set):
    pairs = []
    visited = set()

    def dfs(node, parent_name=None):
        current_name = node["species"]["name"]

        if current_name not in visited and current_name in valid_pokemon:
            visited.add(current_name)
            if parent_name is None or parent_name in valid_pokemon:
                pairs.append((parent_name, current_name))

        for child in node["evolves_to"]:
            dfs(child, current_name)

    dfs(chain_obj["chain"])
    return pairs, visited


def write_log(not_found: list, skipped: list, form_variants: list, all_pairs: list):
    with open("log.txt", "w") as f:
        f.write("=== Evolution Mapping Log ===\n\n")
        f.write(f"Total pairs written: {len(all_pairs)}\n")
        f.write(f"Total skipped (regional forms): {len(skipped)}\n")
        f.write(f"Total form variants (added as base): {len(form_variants)}\n")
        f.write(f"Total 404s: {len(not_found)}\n\n")

        f.write("--- Skipped (regional forms) ---\n")
        for name in skipped:
            f.write(f"  SKIPPED: {name}\n")

        f.write("\n--- Form Variants (added as standalone base) ---\n")
        for name in form_variants:
            f.write(f"  FORM: {name}\n")

        f.write("\n--- 404 Not Found ---\n")
        for name in not_found:
            f.write(f"  404: {name}\n")


def main():
    with open("all_pokemon_evo.json", "r") as f:
        pokemon_list = json.load(f)

    # only exclude regional forms from valid set
    valid_pokemon = {name.lower() for name in pokemon_list if not is_regional_form(name)}

    all_pairs = []
    globally_visited = set()
    not_found = []
    skipped = []
    form_variants_added = []

    for name in pokemon_list:

        # regional forms → skip entirely
        if is_regional_form(name):
            skipped.append(name)
            continue

        sanitized = name.lower()

        # form variants → add as standalone base, no API call
        if is_form_variant(name):
            all_pairs.append((None, sanitized))
            form_variants_added.append(sanitized)
            continue

        # skip if already processed as part of another chain
        if sanitized in globally_visited:
            continue

        # fetch species
        try:
            species = get_species(sanitized)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"404 NOT FOUND: {name}")
                not_found.append(name)
                continue
            raise

        # fetch evolution chain
        evo_chain_url = species["evolution_chain"]["url"]
        try:
            evo_chain = get_evolution_chain(evo_chain_url)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"404 NOT FOUND (evo chain): {name}")
                not_found.append(name)
                continue
            raise

        pairs, visited = build_evolution_pairs(evo_chain, valid_pokemon)
        globally_visited.update(visited)
        all_pairs.extend(pairs)

    # serialize: None becomes JSON null
    output = [{"prev": prev, "next": current} for prev, current in all_pairs]

    with open("evolution.json", "w") as f:
        json.dump(output, f, indent=2)

    write_log(not_found, skipped, form_variants_added, all_pairs)
    print(f"\nDone. {len(all_pairs)} pairs written to evolution.json")
    print(f"Skipped (regional): {len(skipped)} | Form variants (as base): {len(form_variants_added)} | 404s ({len(not_found)}): {not_found}")


if __name__ == "__main__":
    main()
