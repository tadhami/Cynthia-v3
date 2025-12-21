"""
Builds a consolidated Pokémon knowledge base from CSV sources.

What this script does at a high level:
- Resolves paths under the project `data/` folder
- Loads multiple CSVs (types chart, Pokémon types and stats, moves, evolutions, locations, items)
- Computes type effectiveness and assembles per-Pokémon summaries
- Writes JSON Lines to `data/processed/pokemon_kb.txt` (one JSON object per line)

Tip: Read the `build_kb()` function first for the write flow. Helper loaders above it each return
plain Python structures (lists or dicts) with predictable keys used during assembly.
"""
import csv  # CSV parsing for all data files
import json # JSON lines output formatting
import ast   # Safe parsing of stringified lists (e.g., abilities)
from pathlib import Path  # Cross-platform filesystem paths

ROOT = Path(__file__).resolve().parents[1]  # repository root
DATA = ROOT / "data"                         # raw/curated CSVs live here
PROCESSED = DATA / "processed"               # final KB text output folder

# Input CSVs by role
TYPES_CHART_CSV = DATA / "types_chart.csv"                 # attacker vs defender multipliers
POKEMON_TYPES_CSV = DATA / "Pokemon_data_types.csv"        # names + typing (primary/secondary)
POKEMON_STATS_CSV = DATA / "Pokemon_data.csv"              # base stats and misc metadata
POKEMON_MOVES_CSV = DATA / "pokemon_moves.csv"             # move names + attributes
POKEMON_EVOS_CSV = DATA / "evolution_criteria.csv"         # evolution descriptions
POKEMON_LOCS_CSV = DATA / "Pokemon_locations.csv"          # per-game Pokémon locations
ITEM_LOCS_CSV = DATA / "item_locations.csv"                # per-game item locations

# Output KB file (JSON Lines in a .txt: one JSON object per line)
OUTPUT_TXT = PROCESSED / "pokemon_kb.txt"

# Map numeric codes in types_chart.csv to multipliers
# -5 -> 0x (immune), -1 -> 0.5x (resist), 0 -> 1x (neutral), 1 -> 2x (weak)
# Mapping of type chart codes to numeric multipliers used in effectiveness math
CODE_TO_MULT = {
    -5: 0.0,
    -1: 0.5,
    0: 1.0,
    1: 2.0,
}


def load_type_chart(path: Path):
    """Read the type chart and return (defender_types, chart).

    defender_types: list[str] of column names representing defender types
    chart: dict[str, dict[str, float]] mapping attacker -> defender -> multiplier
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # header like: ["Attacker/Defender", "Normal", "Fire", ...]
        defender_types = header[1:]
        chart = {}  # attacker -> defender -> multiplier
        for row in reader:
            attacker = row[0].strip()
            chart[attacker] = {}
            for i, cell in enumerate(row[1:]):
                cell = cell.strip()
                # Some cells may be like "-5" or "0" or "1"
                try:
                    code = int(cell)
                except Exception:
                    # Default neutral if missing/invalid
                    code = 0
                mult = CODE_TO_MULT.get(code, 1.0)
                chart[attacker][defender_types[i]] = mult
        return defender_types, chart


def load_pokemon_types(path: Path):
    """Load Pokémon names with primary/secondary types.

    Returns: list of dicts {name, type1, type2 or None}
    """
    pokemons = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name") or row.get("Pokemon")
            type1 = (row.get("type1") or row.get("Primary Type") or "").strip()
            type2 = (row.get("type2") or row.get("Secondary Type") or "").strip()
            if type2 in ("", "None", "null", "NULL"):
                type2 = None
            if not name:
                continue
            pokemons.append({
                "name": name.strip(),
                "type1": type1,
                "type2": type2,
            })
    return pokemons


def load_pokemon_meta(path: Path):
    """Load auxiliary metadata per Pokémon keyed by name.

    Includes: pokedex_number, generation, classification, abilities,
    height_m, weight_kg, base_total
    """
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or row.get("Name") or "").strip()
            if not name:
                continue
            # Extract and normalize abilities
            abilities_raw = (row.get("abilities") or row.get("Abilities") or "").strip()
            abilities_fmt = ""
            if abilities_raw:
                try:
                    parsed = ast.literal_eval(abilities_raw)
                    if isinstance(parsed, (list, tuple)):
                        abilities_fmt = ", ".join(str(x) for x in parsed)
                    else:
                        abilities_fmt = str(parsed)
                except Exception:
                    # Fallback: strip brackets/quotes common in CSVs
                    abilities_fmt = abilities_raw.strip("[]").replace("'", "").replace('"', "")

            meta[name] = {
                "pokedex_number": (row.get("pokedex_number") or "").strip(),
                "generation": (row.get("generation") or "").strip(),
                "classification": (row.get("classification") or "").strip(),
                "abilities": abilities_fmt,
                "height_m": (row.get("height_m") or "").strip(),
                "weight_kg": (row.get("weight_kg") or "").strip(),
                "base_total": (row.get("base_total") or row.get("base_total")) or "",
            }
    return meta


def load_stats_lookup(path: Path):
    """Build a name -> base stats lookup dict."""
    lookup = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Name") or row.get("name") or "").strip()
            if not name:
                continue
            stats = {
                "HP": row.get("HP", ""),
                "Attack": row.get("Attack", ""),
                "Defense": row.get("Defense", ""),
                "Sp.Atk": row.get("Sp.Atk", row.get("sp_attack", "")),
                "Sp.Def": row.get("Sp.Def", row.get("sp_defense", "")),
                "Speed": row.get("Speed", ""),
            }
            lookup.setdefault(name, stats)
    return lookup

def load_id_to_name(path: Path):
    """Resolve Pokédex IDs to canonical names (handles variants)."""
    id_to_name = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("pokedex_number") or row.get("pokemon_id") or "").strip()
            name = (row.get("Name") or row.get("name") or "").strip()
            variant = (row.get("Variant") or "").strip()
            if not pid or not name:
                continue
            if pid not in id_to_name:
                id_to_name[pid] = name
            elif not variant and id_to_name.get(pid, ""): 
                id_to_name[pid] = name
    return id_to_name

def load_moves(path: Path):
    """Load move entries with type/category/power/accuracy/PP/effect."""
    moves = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return moves
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        def key(*candidates):
            for c in candidates:
                if c in fieldnames:
                    return c
            return candidates[0]
        name_key = key("Name", fieldnames[0])
        type_key = key("Type")
        cat_key = key("Cat.", "Category")
        power_key = key("Power")
        acc_key = key("Acc.", "Accuracy")
        pp_key = key("PP")
        effect_key = key("Effect")

        for row in reader:
            name = (row.get(name_key) or "").strip()
            if not name:
                continue
            m = {
                "Name": name,
                "Type": (row.get(type_key) or "").strip() or "Unknown",
                "Category": (row.get(cat_key) or "").strip() or "—",
                "Power": (row.get(power_key) or "").strip() or "—",
                "Accuracy": (row.get(acc_key) or "").strip() or "—",
                "PP": (row.get(pp_key) or "").strip() or "—",
                "Effect": (row.get(effect_key) or "").strip() or "—",
            }
            moves.append(m)
    return moves

def clean_text(s: str) -> str:
    """Normalize whitespace and remove placeholder artifacts."""
    if not s:
        return ""
    # Remove placeholder artifacts and tidy whitespace
    s = s.replace("{a WEIRD string}", "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("  ", " ")
    return s.strip()

def load_evolutions(path: Path):
    """Load evolution criteria per Pokémon name."""
    evo = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("pokemon_name") or row.get("name") or "").strip()
            crit = clean_text(row.get("evolution_criteria") or "")
            if not name:
                continue
            if crit and crit.lower().startswith("not found"):
                crit = "Evolution details unavailable."
            evo[name] = crit or "Evolution details unavailable."
    return evo

def load_pokemon_locations(path: Path, id_to_name: dict[str, str]):
    """Build name -> semicolon-joined per-game locations string.

    All columns except identifiers are treated as game labels; any non-empty value
    becomes "GameName: value" and is joined with semicolons.
    """
    locs: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [fn.strip() for fn in (reader.fieldnames or [])]
        # Treat all columns except id/name-like fields as game columns
        exclude = {"pokemon_id", "pokedex_number", "name", "Name", "Variant"}
        game_cols = [c for c in fields if c not in exclude]
        for row in reader:
            pid = (row.get("pokemon_id") or row.get("pokedex_number") or "").strip()
            if not pid:
                continue
            name = id_to_name.get(pid)
            if not name:
                continue
            parts = []
            for g in game_cols:
                val = clean_text(row.get(g) or "")
                if val:
                    parts.append(f"{g}: {val}")
            if parts:
                locs[name] = "; ".join(parts)
    return locs

def load_item_locations(path: Path):
    """Return list of {name, summary} where summary aggregates per-game item locations."""
    items = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return items
        fields = [fn.strip() for fn in reader.fieldnames]
        # Identify item name column
        name_key = "item_name" if "item_name" in fields else fields[0]
        exclude = {name_key, "item_id", "id", "ID"}
        game_cols = [c for c in fields if c not in exclude]
        for row in reader:
            name = (row.get(name_key) or "").strip()
            if not name:
                continue
            parts = []
            for g in game_cols:
                val = clean_text(row.get(g) or "")
                if val:
                    parts.append(f"{g}: {val}")
            summary = "; ".join(parts) if parts else "Locations unavailable"
            items.append({"name": name, "summary": summary})
    return items


def combine_effects(type1: str, type2: str | None, attacker_types: list[str], chart: dict):
    """Compute weaknesses/resistances/immunities vs all attacking types.

    Multiplier product across the defending types decides bucket:
    - 0.0 -> immune
    - >1.0 -> weak (mark 4x explicitly)
    - 0<mult<1 -> resist (mark 0.25x explicitly)
    """
    weaknesses = []
    resistances = []
    immunities = []

    for atk in attacker_types:
        m1 = chart.get(atk, {}).get(type1, 1.0)
        m2 = 1.0
        if type2:
            m2 = chart.get(atk, {}).get(type2, 1.0)
        total = m1 * m2
        if total == 0.0:
            immunities.append(atk)
        elif total > 1.0:
            # Mark 4x explicitly
            label = f"{atk} (4x)" if total >= 4.0 else atk
            weaknesses.append(label)
        elif 0.0 < total < 1.0:
            # Mark 0.25x explicitly
            label = f"{atk} (0.25x)" if total <= 0.25 else atk
            resistances.append(label)
    return weaknesses, resistances, immunities


def build_kb():
    """Main assembly: writes KB as JSON Lines (one JSON object per line)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    defender_types, chart = load_type_chart(TYPES_CHART_CSV)
    pokemons = load_pokemon_types(POKEMON_TYPES_CSV)
    meta_lookup = load_pokemon_meta(POKEMON_TYPES_CSV)
    stats_lookup = load_stats_lookup(POKEMON_STATS_CSV)
    moves = load_moves(POKEMON_MOVES_CSV)
    id_to_name = load_id_to_name(POKEMON_STATS_CSV)
    evolutions = load_evolutions(POKEMON_EVOS_CSV)
    locations = load_pokemon_locations(POKEMON_LOCS_CSV, id_to_name)
    items = load_item_locations(ITEM_LOCS_CSV)
    print(f"Loaded {len(pokemons)} Pokémon, {len(moves)} moves, {len(evolutions)} evolutions, {len(locations)} location rows, {len(items)} items.")

    with OUTPUT_TXT.open("w", encoding="utf-8") as out:
        # Types as JSON objects
        for atk in chart.keys():
            weak = [d for d, mult in chart[atk].items() if mult == 2.0]
            resist = [d for d, mult in chart[atk].items() if mult == 0.5]
            immune = [d for d, mult in chart[atk].items() if mult == 0.0]
            obj = {
                "id": atk,
                "category": "Type",
                "super_effective": weak,
                "not_very_effective": resist,
                "no_effect": immune,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Per-Pokémon entries as JSON objects
        for p in sorted(pokemons, key=lambda x: x["name"].lower()):
            name = p["name"]
            t1 = p["type1"] or "Unknown"
            t2 = p["type2"]
            weaknesses, resistances, immunities = combine_effects(t1, t2, defender_types, chart)

            # Selected metadata fields
            meta = meta_lookup.get(name, {})
            dex = meta.get("pokedex_number") or "?"
            gen = meta.get("generation") or "?"
            classification = meta.get("classification") or "Unknown"
            abilities = meta.get("abilities") or "Unknown"
            height_m = meta.get("height_m") or "?"
            weight_kg = meta.get("weight_kg") or "?"
            base_total = str(meta.get("base_total") or "?")

            # Stats dictionary (may contain empty strings depending on source CSV)
            stats = stats_lookup.get(name)

            obj = {
                "id": name,
                "category": "Pokemon",
                "types": [t1] if not t2 else [t1, t2],
                "weak_to": weaknesses,
                "resists": resistances,
                "immune_to": immunities,
                "stats": stats or {},
                "evolution": evolutions.get(name) or "",
                "locations": locations.get(name) or "",
                "metadata": {
                    "pokedex_number": dex,
                    "generation": gen,
                    "classification": classification,
                    "abilities": abilities,
                    "height_m": height_m,
                    "weight_kg": weight_kg,
                    "base_total": base_total,
                },
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Per-Move entries as JSON objects
        for m in sorted(moves, key=lambda x: x["Name"].lower()):
            name = m["Name"]
            type_ = m["Type"]
            cat = m["Category"]
            power = m["Power"]
            acc = m["Accuracy"]
            pp = m["PP"]
            effect = m["Effect"] if m["Effect"] else "—"
            obj = {
                "id": name,
                "category": "Move",
                "type": type_,
                "move_category": cat,
                "power": power,
                "accuracy": acc,
                "pp": pp,
                "effect": effect,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Item entries as JSON objects
        for item in sorted(items, key=lambda x: x["name"].lower()):
            obj = {
                "id": item["name"],
                "category": "Item",
                "locations": item["summary"],
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {OUTPUT_TXT.relative_to(ROOT)}")


if __name__ == "__main__":
    # Entry point: run the builder and write the KB file
    build_kb()
