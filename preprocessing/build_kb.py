import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

TYPES_CHART_CSV = DATA / "types_chart.csv"
POKEMON_TYPES_CSV = DATA / "Pokemon_data_types.csv"
POKEMON_STATS_CSV = DATA / "Pokemon_data.csv"
POKEMON_MOVES_CSV = DATA / "pokemon_moves.csv"
POKEMON_EVOS_CSV = DATA / "evolution_criteria.csv"
POKEMON_LOCS_CSV = DATA / "Pokemon_locations.csv"
ITEM_LOCS_CSV = DATA / "item_locations.csv"
OUTPUT_TXT = PROCESSED / "pokemon_kb.txt"

# Map numeric codes in types_chart.csv to multipliers
# -5 -> 0x (immune), -1 -> 0.5x (resist), 0 -> 1x (neutral), 1 -> 2x (weak)
CODE_TO_MULT = {
    -5: 0.0,
    -1: 0.5,
    0: 1.0,
    1: 2.0,
}


def load_type_chart(path: Path):
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


def load_stats_lookup(path: Path):
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
    id_to_name = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("pokedex_number") or row.get("pokemon_id") or "").strip()
            name = (row.get("Name") or row.get("name") or "").strip()
            variant = (row.get("Variant") or "").strip()
            if not pid or not name:
                continue
            # Prefer non-variant base names; do not overwrite an existing base with a variant
            if pid not in id_to_name:
                id_to_name[pid] = name
            elif not variant and id_to_name.get(pid, ""):  # keep existing base
                id_to_name[pid] = name
    return id_to_name

def load_moves(path: Path):
    moves = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return moves
        # Resolve potential BOM or unexpected header names
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        # Map canonical keys
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
    if not s:
        return ""
    # Remove placeholder artifacts and tidy whitespace
    s = s.replace("{a WEIRD string}", "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("  ", " ")
    return s.strip()

def load_evolutions(path: Path):
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
    """Load per-Pokémon locations across ALL game columns present in the CSV."""
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
    """Load item names and their locations across ALL game columns present in the CSV."""
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
    # For each attacking type, compute multiplier product across defending types
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
    PROCESSED.mkdir(parents=True, exist_ok=True)
    defender_types, chart = load_type_chart(TYPES_CHART_CSV)
    pokemons = load_pokemon_types(POKEMON_TYPES_CSV)
    stats_lookup = load_stats_lookup(POKEMON_STATS_CSV)
    moves = load_moves(POKEMON_MOVES_CSV)
    id_to_name = load_id_to_name(POKEMON_STATS_CSV)
    evolutions = load_evolutions(POKEMON_EVOS_CSV)
    locations = load_pokemon_locations(POKEMON_LOCS_CSV, id_to_name)
    items = load_item_locations(ITEM_LOCS_CSV)
    print(f"Loaded {len(pokemons)} Pokémon, {len(moves)} moves, {len(evolutions)} evolutions, {len(locations)} location rows, {len(items)} items.")

    # Index list for quick name matching
    names_index = sorted(set(p["name"] for p in pokemons))

    with OUTPUT_TXT.open("w", encoding="utf-8") as out:
        out.write("Pokémon Knowledge Base.\n")
        out.write("Index: " + ", ".join(names_index[:200]) + ("..." if len(names_index) > 200 else "") + "\n\n")

        # Type Chart Summary
        out.write("Type Chart Summary:\n")
        for atk in chart.keys():
            weak = [d for d, mult in chart[atk].items() if mult == 2.0]
            resist = [d for d, mult in chart[atk].items() if mult == 0.5]
            immune = [d for d, mult in chart[atk].items() if mult == 0.0]
            out.write(f"{atk}: Super effective against {', '.join(weak) or 'None'}. ")
            out.write(f"Not very effective against {', '.join(resist) or 'None'}. ")
            out.write(f"No effect on {', '.join(immune) or 'None'}.\n")
        out.write("\n")

        # Per-Pokémon entries
        for p in sorted(pokemons, key=lambda x: x["name"].lower()):
            name = p["name"]
            t1 = p["type1"] or "Unknown"
            t2 = p["type2"]
            weaknesses, resistances, immunities = combine_effects(t1, t2, defender_types, chart)

            # Stats summary if available
            stats = stats_lookup.get(name)
            if stats:
                stats_text = (
                    f"Stats: HP {stats['HP']}/Atk {stats['Attack']}/Def {stats['Defense']}/"
                    f"SpA {stats['Sp.Atk']}/SpD {stats['Sp.Def']}/Spe {stats['Speed']}."
                )
            else:
                stats_text = "Stats: Unknown."

            type_label = f"{t1}/{t2}" if t2 else t1
            weak_text = ", ".join(weaknesses) if weaknesses else "None"
            resist_text = ", ".join(resistances) if resistances else "None"
            immune_text = ", ".join(immunities) if immunities else "None"

            out.write(f"{name} is a {type_label}-type Pokémon. ")
            out.write(f"Weak to: {weak_text}. ")
            out.write(f"Resists: {resist_text}. ")
            out.write(f"Immune to: {immune_text}. ")
            # Evolution summary
            evo_text = evolutions.get(name)
            if evo_text:
                out.write(f"Evolution: {evo_text} ")
            # Locations summary (modern games)
            loc_text = locations.get(name)
            if loc_text:
                out.write(f"Locations: {loc_text}. ")
            out.write(stats_text + "\n")

        out.write("\n")
        out.write("Moves Index:\n")
        if moves:
            out.write(
                ", ".join(sorted(set(m["Name"] for m in moves))[:200])
                + ("..." if len(moves) > 200 else "")
                + "\n\n"
            )
        else:
            out.write("(No moves found)\n\n")

        # Per-Move entries
        out.write("Move Summaries:\n")
        for m in sorted(moves, key=lambda x: x["Name"].lower()):
            name = m["Name"]
            type_ = m["Type"]
            cat = m["Category"]
            power = m["Power"]
            acc = m["Accuracy"]
            pp = m["PP"]
            effect = m["Effect"] if m["Effect"] else "—"
            out.write(
                f"{name} — Type={type_}; Category={cat}; Power={power}; Accuracy={acc}; PP={pp}. "
                f"Effect: {effect}.\n"
            )

        # Item Locations
        out.write("\nItem Locations:\n")
        if items:
            for item in sorted(items, key=lambda x: x["name"].lower()):
                out.write(f"{item['name']}: {item['summary']}\n")
        else:
            out.write("(No item locations found)\n")

    print(f"Wrote {OUTPUT_TXT.relative_to(ROOT)}")


if __name__ == "__main__":
    build_kb()
