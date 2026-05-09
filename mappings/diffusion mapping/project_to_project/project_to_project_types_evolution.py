"""
Enriches safe_project_to_project.json with type + evolution-stage info.

For each entry we add:
    - `types`          :    list of types for `next`.
                            Base species use poke-data/pokedex.json.
                            Regional variants and other alternate forms are
                            resolved via PokeAPI's
                                /pokemon-species/{name} -> varieties
                                -> /pokemon/{slug}
                            pipeline so region-specific typing (e.g. dark Alolan
                            Meowth, fairy Galarian Ponyta, Hisuian Typhlosion
                            fire/ghost) is preserved. Two cases are handled:

                            1. Explicit region tokens in `next`, e.g.
                                "Meowth-Alola", "Persian-Alola", "mr. mime galar".
                            2. ProjectPokemon sprite filenames with a non-zero
                                form index (e.g. poke_capture_0019_001_* for
                                Alolan Rattata, _002_ for Galarian Meowth,
                                _003_ for Galarian Zen Darmanitan).
                            3. HOME-style sprite filenames with a variant tag
                                after the dex number. Tag letters map to:
                                    A = Alola, G = Galar, H = Hisui, P = Paldea.
                                Two-letter tags like PA / PB / PC pick the 1st
                                / 2nd / 3rd Paldean variety in PokeAPI order
                                (e.g. Tauros PA = combat, PB = blaze,
                                PC = aqua). Other species-specific tags (O for
                                Dialga-Origin, B for Ursaluna-Bloodmoon, etc.)
                                are resolved by first-letter token match.

                            Requires --use-api (network).
    - `evolution_stage`: string label for the pokemon's depth within its chain:
                            "base"  -> 0 (prev: null)
                            "evo 1" -> 1 (first evolution)
                            "evo 2" -> 2 (second evolution)
                            "evo N" -> N for any deeper chains.
    - `art_style`      : hard-coded to "project".

Stages are derived from the `prev` / `next` edges in safe_project_to_project.json
via a BFS starting at every `prev: null` entry (chain starts). Any pokemon not
reachable that way is optionally looked up against the PokeAPI so nothing is
silently dropped, network use is opt-in via --use-api.

Output
--
mappings/diffusion mapping/project_to_project/safe_project_to_project_types_evolution.json
mappings/diffusion mapping/project_to_project/project_to_project_types_evolution_unresolved.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT       = Path(__file__).resolve().parents[3]
POKEDEX_JSON    = REPO_ROOT / "poke-data" / "pokedex.json"
INPUT_JSON      = Path(__file__).parent / "safe_project_to_project.json"
OUTPUT_JSON     = Path(__file__).parent / "safe_project_to_project_types_evolution.json"
UNRESOLVED_JSON = Path(__file__).parent / "project_to_project_types_evolution_unresolved.json"

ART_STYLE = "3d"

REGIONS = ("alola", "galar", "hisui", "paldea")
SPRITE_FORM_RE = re.compile(r"poke_capture_\d{4}_(\d{3})_")
HOME_RE        = re.compile(r"HOME\d{4}([A-Za-z]*)(?:_s)?\.png$", re.IGNORECASE)

HOME_REGION_LETTER: dict[str, str] = {
    "A": "alola",
    "G": "galar",
    "H": "hisui",
    "P": "paldea",
}


def stage_label(stage: int | None) -> str | None:
    """Map an integer stage to its human-readable label ("base", "evo 1", ...)."""
    if stage is None:
        return None
    if stage == 0:
        return "base"
    return f"evo {stage}"


def _norm(s: str) -> str:
    """Collapse all separators (space, hyphen, dot, apostrophe) for fuzzy matching."""
    return re.sub(r"[.\-'\s]", "", s.lower())


def build_type_index(pokedex: dict) -> dict[str, list[str]]:
    """Map normalized pokemon name -> list of types."""
    index: dict[str, list[str]] = {}
    for name, data in pokedex.items():
        type_str = data.get("type", "") or ""
        types = [t.strip() for t in type_str.split("/") if t.strip()]
        index[_norm(name)] = types
    return index


def lookup_types(name: str | None, type_index: dict[str, list[str]]) -> list[str]:
    """
    Return types for a pokemon name, falling back to progressively shorter
    hyphen/space-split forms (e.g. "meowth-galar" -> "meowth") to handle
    regional variants whose base form shares the same typing in the pokedex.
    """
    if not name:
        return []

    key = _norm(name)
    if key in type_index:
        return type_index[key]

    parts = re.split(r"[\s\-]+", name.strip())
    for i in range(len(parts) - 1, 0, -1):
        base_key = _norm(" ".join(parts[:i]))
        if base_key in type_index:
            return type_index[base_key]

    return []


# Evolution-stage derivation

def build_stage_map(entries: list[dict]) -> tuple[dict[str, int], set[str]]:
    """
    Build { pokemon_name -> evolution_stage } using the prev/next edges in
    safe_project_to_project.json. BFS from every `prev: null` entry (chain
    start): base = 0, its evolutions = 1, their evolutions = 2, etc.

    For forks (e.g. Eevee -> Vaporeon/Jolteon/...) every branch is stage 1.
    For joins (multiple prevs for same next), the shortest distance wins.

    Returns (stage_map, all_names_seen).
    """
    adj: dict[str, set[str]] = defaultdict(set)
    all_names: set[str] = set()
    bases: set[str] = set()

    for e in entries:
        prv = e.get("prev")
        nxt = e.get("next")
        if nxt:
            all_names.add(nxt)
        if prv:
            all_names.add(prv)
            if nxt:
                adj[prv].add(nxt)
        else:
            if nxt:
                bases.add(nxt)

    stage: dict[str, int] = {}
    q: deque[tuple[str, int]] = deque()
    for b in bases:
        q.append((b, 0))

    while q:
        name, s = q.popleft()
        if name in stage and stage[name] <= s:
            continue
        stage[name] = s
        for child in adj.get(name, ()):
            q.append((child, s + 1))

    return stage, all_names


# PokeAPI fallback (opt-in)

_POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon-species/{name}"


def _pokeapi_slug(name: str) -> str:
    """Turn a folder-derived name like 'mr. mime' into 'mr-mime' for PokeAPI."""
    s = name.lower().strip()
    s = s.replace(".", "").replace("'", "").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


def _fetch_species(name: str) -> dict | None:
    try:
        url = _POKEAPI_URL.format(name=quote(_pokeapi_slug(name)))
        req = Request(url, headers={"User-Agent": "project-types/1.0"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  PokeAPI miss for {name!r}: {exc}")
        return None


def _fetch_json(url: str) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": "project-types/1.0"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  PokeAPI miss for {url}: {exc}")
        return None


def detect_form_index(sprite_path: str | None) -> int:
    """
    Pull the form index out of a ProjectPokemon sprite filename like
    'poke_capture_0019_001_mf_n_00000000_f_n.png' -> 1. HOME-style or
    unrecognised filenames return 0 (base form).
    """
    if not sprite_path:
        return 0
    m = SPRITE_FORM_RE.search(sprite_path)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def detect_variant_from_name(
    next_name: str | None,
) -> tuple[str | None, str | None, list[str]]:
    """
    Detect a regional variant encoded directly in a `next` value such as
    'Meowth-Alola', 'Persian-Alola', or 'mr. mime galar'. Returns
    (base_species, region, subtokens); (None, None, []) when no region token
    is present.
    """
    if not next_name:
        return None, None, []
    tokens = [t for t in re.split(r"[\s\-_]+", next_name.lower().strip()) if t]
    for i, tok in enumerate(tokens):
        if tok in REGIONS:
            base = " ".join(tokens[:i]).strip()
            if not base:
                return None, None, []
            subtokens = [t for t in tokens[i + 1:] if t.isalpha()]
            return base, tok, subtokens
    return None, None, []


def _variant_score(variant_name: str, region: str, subtokens: list[str]) -> int:
    """
    Rank a candidate PokeAPI variety name. Higher is better. Must contain
    the region; subtoken matches boost, extras penalise.
    """
    parts = variant_name.split("-")
    if region not in parts:
        return -1
    score = 100
    for st in subtokens:
        score += 50 if st in parts else -50
    base = variant_name.split("-", 1)[0]
    extras = [
        p for p in parts
        if p != region and p not in subtokens and p != base and not p.isdigit()
    ]
    score -= 5 * len(extras)
    return score


def _types_from_pokemon_payload(data: dict | None) -> list[str]:
    if not data:
        return []
    out = []
    for t in data.get("types") or []:
        name = (t.get("type") or {}).get("name")
        if name:
            out.append(name.lower())
    return out


def resolve_variant_types_by_region(
    species: str,
    region: str,
    subtokens: list[str],
    species_cache: dict[str, dict | None],
    variant_type_cache: dict[str, list[str]],
) -> tuple[list[str], str | None]:
    """
    Pick the PokeAPI variety best matching (region, subtokens) and return
    (types, matched_variant_name). Returns ([], None) on any failure.
    """
    if species not in species_cache:
        print(f"  Fetching species {species!r} for {region} variant …")
        species_cache[species] = _fetch_species(species)
    species_data = species_cache[species]
    if not species_data:
        return [], None

    varieties = species_data.get("varieties") or []
    best_name: str | None = None
    best_url: str | None = None
    best_score = -1
    for v in varieties:
        poke = v.get("pokemon") or {}
        name = poke.get("name") or ""
        score = _variant_score(name, region, subtokens)
        if score > best_score:
            best_score, best_name, best_url = score, name, poke.get("url")

    if not best_name or best_score < 0 or not best_url:
        return [], None

    if best_name in variant_type_cache:
        return variant_type_cache[best_name], best_name

    print(f"  Fetching variant {best_name!r} …")
    types = _types_from_pokemon_payload(_fetch_json(best_url))
    variant_type_cache[best_name] = types
    return types, best_name


def resolve_variant_types_by_form_index(
    species: str,
    form_index: int,
    species_cache: dict[str, dict | None],
    variant_type_cache: dict[str, list[str]],
) -> tuple[list[str], str | None]:
    """
    Return (types, variety_name) for the PokeAPI variety at position
    `form_index` in the species' `varieties` list. This mirrors the game's
    internal FormIndex that ProjectPokemon filenames embed. Returns
    ([], None) if the species has fewer varieties than the requested index
    or the request fails.
    """
    if form_index <= 0:
        return [], None
    if species not in species_cache:
        print(f"  Fetching species {species!r} for form index {form_index} …")
        species_cache[species] = _fetch_species(species)
    data = species_cache[species]
    if not data:
        return [], None

    varieties = data.get("varieties") or []
    if form_index >= len(varieties):
        return [], None

    poke = varieties[form_index].get("pokemon") or {}
    name = poke.get("name") or ""
    url = poke.get("url") or ""
    if not name or not url:
        return [], None

    if name in variant_type_cache:
        return variant_type_cache[name], name

    print(f"  Fetching variant {name!r} (form {form_index}) …")
    types = _types_from_pokemon_payload(_fetch_json(url))
    variant_type_cache[name] = types
    return types, name


def detect_home_tag(sprite_path: str | None) -> str | None:
    """
    Parse a HOME-style ProjectPokemon filename like 'HOME0059H_s.png' and
    return the upper-cased variant tag ('H'), 'PA' / 'PB' / 'PC' for the
    three Paldean Tauros forms, etc. Returns None for base HOME sprites
    ('HOME0001.png' / 'HOME0001_s.png') or non-HOME filenames.
    """
    if not sprite_path:
        return None
    m = HOME_RE.search(sprite_path)
    if not m:
        return None
    tag = (m.group(1) or "").upper()
    return tag or None


def _distinctive_tokens(variety_name: str) -> list[str]:
    """Variety-name tokens with the species base and filler words removed."""
    noise = {"breed", "mask", "form", "standard"}
    parts = variety_name.split("-")[1:]  # drop species base segment
    return [p for p in parts if p and p not in noise]


def resolve_variant_types_by_home_tag(
    species: str,
    tag: str,
    species_cache: dict[str, dict | None],
    variant_type_cache: dict[str, list[str]],
) -> tuple[list[str], str | None]:
    """
    Resolve the variety that corresponds to a HOME-style file tag.

    Tag rules
    ---------
        * Single-letter region tags ``A`` / ``G`` / ``H`` / ``P`` pick the variety
            whose name contains ``alola`` / ``galar`` / ``hisui`` / ``paldea``.
            Shortest matching variety wins (so ``wooper`` + ``P`` -> ``wooper-paldea``
            and never a sub-form).
        * Two-letter region + ordinal tags ``PA`` / ``PB`` / ``PC`` (generally
            ``<R><X>`` with ``R`` in ``AGHP`` and ``X`` in ``A-Z``) pick the Nth
            variety containing that region, in PokeAPI order
            (e.g. Tauros ``PA`` -> combat-breed, ``PB`` -> blaze-breed, ``PC`` ->
            aqua-breed).
        * Any other tag falls back to a species-specific first-letter match:
            pick the variety that has a distinctive token starting with the tag
            (e.g. ``dialga`` + ``O`` -> ``dialga-origin``, ``ursaluna`` + ``B`` ->
            ``ursaluna-bloodmoon``, ``palafin`` + ``H`` -> ``palafin-hero``,
            ``ogerpon`` + ``C`` -> ``ogerpon-cornerstone-mask``).

    Returns ``([], None)`` when nothing matches or the PokeAPI call fails.
    """
    if not tag:
        return [], None

    if species not in species_cache:
        print(f"  Fetching species {species!r} for HOME tag {tag!r} …")
        species_cache[species] = _fetch_species(species)
    data = species_cache[species]
    if not data:
        return [], None

    varieties = data.get("varieties") or []
    t = tag.upper()
    chosen: dict | None = None

    def _region_matches(region: str) -> list[dict]:
        out = []
        for v in varieties:
            name = (v.get("pokemon") or {}).get("name") or ""
            if region in name.split("-"):
                out.append(v)
        return out

    # Single-letter regional tag.
    if chosen is None and len(t) == 1 and t in HOME_REGION_LETTER:
        matches = _region_matches(HOME_REGION_LETTER[t])
        if matches:
            matches.sort(
                key=lambda v: len(
                    ((v.get("pokemon") or {}).get("name") or "").split("-")
                )
            )
            chosen = matches[0]

    # Two-letter region + ordinal tag.
    if (
        chosen is None
        and len(t) == 2
        and t[0] in HOME_REGION_LETTER
        and "A" <= t[1] <= "Z"
    ):
        matches = _region_matches(HOME_REGION_LETTER[t[0]])
        idx = ord(t[1]) - ord("A")
        if 0 <= idx < len(matches):
            chosen = matches[idx]

    # Generic fallback: match any distinctive token whose first letter == tag.
    if chosen is None and len(t) == 1:
        region_tokens = set(REGIONS)
        candidates: list[dict] = []
        for v in varieties:
            name = (v.get("pokemon") or {}).get("name") or ""
            for tok in _distinctive_tokens(name):
                if tok in region_tokens:
                    continue
                if tok[:1].upper() == t:
                    candidates.append(v)
                    break
        candidates.sort(
            key=lambda v: len(((v.get("pokemon") or {}).get("name") or "").split("-"))
        )
        if candidates:
            chosen = candidates[0]

    if chosen is None:
        return [], None

    poke = chosen.get("pokemon") or {}
    name = poke.get("name") or ""
    url = poke.get("url") or ""
    if not name or not url:
        return [], None

    if name in variant_type_cache:
        return variant_type_cache[name], name

    print(f"  Fetching variant {name!r} (HOME tag {tag}) …")
    types = _types_from_pokemon_payload(_fetch_json(url))
    variant_type_cache[name] = types
    return types, name


def _stage_from_evolution_chain(chain_url: str, target_norm: str) -> int | None:
    """Walk the PokeAPI evolution-chain tree and return the target's depth."""
    try:
        req = Request(chain_url, headers={"User-Agent": "project-types/1.0"})
        with urlopen(req, timeout=10) as resp:
            chain_root = json.loads(resp.read().decode("utf-8")).get("chain")
    except Exception as exc:
        print(f"  PokeAPI chain miss for {chain_url}: {exc}")
        return None

    stack = [(chain_root, 0)]
    while stack:
        node, depth = stack.pop()
        if not node:
            continue
        species_name = (node.get("species") or {}).get("name", "")
        if _norm(species_name) == target_norm:
            return depth
        for child in node.get("evolves_to", []) or []:
            stack.append((child, depth + 1))
    return None


def lookup_stage_via_api(name: str) -> int | None:
    """Return evolution_stage for `name` via PokeAPI, or None on any failure."""
    species = _fetch_species(name)
    if not species:
        return None
    chain = species.get("evolution_chain") or {}
    chain_url = chain.get("url")
    if not chain_url:
        return None
    return _stage_from_evolution_chain(chain_url, _norm(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-api",
        action= "store_true",
        help=   "Use PokeAPI as a fallback for pokemon whose stage can't be derived "
                "from safe_project_to_project.json alone (slower).",
    )
    args = parser.parse_args()

    with open(POKEDEX_JSON, encoding="utf-8") as f:
        pokedex = json.load(f)
    with open(INPUT_JSON, encoding="utf-8") as f:
        entries = json.load(f)

    type_index = build_type_index(pokedex)
    stage_map, all_chain_names = build_stage_map(entries)

    # Names seen only as `prev` that never appear as `next` are chain starts too.
    for n in all_chain_names:
        if n not in stage_map:
            stage_map[n] = 0

    api_cache: dict[str, int | None] = {}
    species_cache: dict[str, dict | None] = {}
    variant_type_cache: dict[str, list[str]] = {}
    variants_used: dict[str, list[str]] = {}

    def resolve_stage(name: str | None) -> int | None:
        if not name:
            return None
        if name in stage_map:
            return stage_map[name]
        if not args.use_api:
            return None
        if name not in api_cache:
            print(f"  Looking up {name!r} on PokeAPI …")
            api_cache[name] = lookup_stage_via_api(name)
        return api_cache[name]

    enriched: list[dict] = []
    all_next_names: set[str] = set()
    missing_types_names: set[str] = set()
    missing_stage_names: set[str] = set()
    unresolved_entries: list[dict] = []

    for entry in entries:
        next_name = entry.get("next")
        next_sprite = entry.get("next_sprite")

        types: list[str] = []
        matched_variant: str | None = None

        if args.use_api and next_name:
            base_species, region, subtokens = detect_variant_from_name(next_name)
            if region and base_species:
                types, matched_variant = resolve_variant_types_by_region(
                    base_species, region, subtokens,
                    species_cache, variant_type_cache,
                )
            else:
                home_tag = detect_home_tag(next_sprite)
                if home_tag:
                    types, matched_variant = resolve_variant_types_by_home_tag(
                        next_name, home_tag,
                        species_cache, variant_type_cache,
                    )
                else:
                    form_idx = detect_form_index(next_sprite)
                    if form_idx > 0:
                        types, matched_variant = resolve_variant_types_by_form_index(
                            next_name, form_idx,
                            species_cache, variant_type_cache,
                        )

        if matched_variant and types:
            variants_used[matched_variant] = types

        if not types:
            types = lookup_types(next_name, type_index)

        stage = resolve_stage(next_name)

        if next_name:
            all_next_names.add(next_name)

            missing_type = not types
            missing_stage = stage is None

            if missing_type:
                missing_types_names.add(next_name)
            if missing_stage:
                missing_stage_names.add(next_name)

            if missing_type or missing_stage:
                unresolved_entries.append({
                    "prev":           entry.get("prev"),
                    "next":           next_name,
                    "prev_sprite":    entry.get("prev_sprite"),
                    "next_sprite":    entry.get("next_sprite"),
                    "missing_types":  missing_type,
                    "missing_stage":  missing_stage,
                })

        enriched.append({
            "prev":            entry.get("prev"),
            "next":            next_name,
            "prev_sprite":     entry.get("prev_sprite"),
            "next_sprite":     entry.get("next_sprite"),
            "types":           [t.lower() for t in types],
            "evolution_stage": stage_label(stage),
            "art_style":       ART_STYLE,
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent="\t")

    unresolved_report = {
        "total_entries":          len(enriched),
        "unique_next_pokemon":    len(all_next_names),
        "missing_types_count":    len(missing_types_names),
        "missing_stage_count":    len(missing_stage_names),
        "missing_types_pokemon":  sorted(missing_types_names),
        "missing_stage_pokemon":  sorted(missing_stage_names),
        "unresolved_entries":     unresolved_entries,
    }
    with open(UNRESOLVED_JSON, "w", encoding="utf-8") as f:
        json.dump(unresolved_report, f, indent="\t")

    stage_hist: dict[str, int] = defaultdict(int)
    for n in all_next_names:
        s = stage_map.get(n)
        if s is None and args.use_api:
            s = api_cache.get(n)
        label = stage_label(s)
        if label is not None:
            stage_hist[label] += 1

    print(f"Written              : {OUTPUT_JSON}")
    print(f"Total entries        : {len(enriched)}")
    print(f"Unique next pokemon  : {len(all_next_names)}")
    print(f"Missing types        : {len(missing_types_names)}")
    print(f"Missing stage        : {len(missing_stage_names)}")
    print(f"Stage distribution   : {dict(sorted(stage_hist.items()))}")
    print(f"Variants resolved    : {len(variants_used)}")
    print(f"Unresolved log       : {UNRESOLVED_JSON}")
    if variants_used:
        for key, t in sorted(variants_used.items()):
            print(f"  {key:35s} -> {t}")
    if missing_types_names:
        print(f"Missing types for    : {sorted(missing_types_names)}")
    if missing_stage_names:
        print(f"Missing stage for    : {sorted(missing_stage_names)}")


if __name__ == "__main__":
    main()
