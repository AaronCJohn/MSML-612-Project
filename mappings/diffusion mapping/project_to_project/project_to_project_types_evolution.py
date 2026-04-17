"""
Enriches safe_project_to_project.json with type + evolution-stage info.

For each entry we add:
    - `types`          : list of types for `next` (from poke-data/pokedex.json).
    - `evolution_stage`: string label for the pokemon's depth within its chain:
                            "base"  -> 0 (prev: null)
                            "evo 1" -> 1 (first evolution)
                            "evo 2" -> 2 (second evolution)
                            "evo N" -> N for any deeper chains.
    - `art_style`      : hard-coded to "project".

Stages are derived from the `prev` / `next` edges in safe_project_to_project.json
via a BFS starting at every `prev: null` entry (chain starts). Any pokemon not
reachable that way is optionally looked up against the PokeAPI
(https://pokeapi.co/api/v2/pokemon-species/{name}) so nothing is silently
dropped — network use is opt-in via --use-api.

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
    print(f"Unresolved log       : {UNRESOLVED_JSON}")
    if missing_types_names:
        print(f"Missing types for    : {sorted(missing_types_names)}")
    if missing_stage_names:
        print(f"Missing stage for    : {sorted(missing_stage_names)}")


if __name__ == "__main__":
    main()
