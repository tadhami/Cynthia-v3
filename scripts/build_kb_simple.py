import pandas as pd  # DataFrame operations for CSVs
from pathlib import Path  # Filesystem-safe path handling
import ast  # Safe parsing of stringified Python literals (e.g., lists)

ROOT = Path(__file__).resolve().parents[1]  # Project root: two levels up from this file
DATA = ROOT / "data"  # Root data folder containing source CSVs
PROCESSED = DATA / "processed"  # Output folder for generated artifacts

TYPES_CHART_CSV = DATA / "types_chart.csv"  # Attack vs. defender type effectiveness chart
POKEMON_TYPES_CSV = DATA / "Pokemon_data_types.csv"  # Names + primary/secondary types
POKEMON_STATS_CSV = DATA / "Pokemon_data.csv"  # Base stats and IDs used for lookups
POKEMON_MOVES_CSV = DATA / "pokemon_moves.csv"  # Move data (type, category, power, etc.)
POKEMON_EVOS_CSV = DATA / "evolution_criteria.csv"  # Evolution requirements text
POKEMON_LOCS_CSV = DATA / "Pokemon_locations.csv"  # Pokémon encounter locations per game
ITEM_LOCS_CSV = DATA / "item_locations.csv"  # Item locations per game/version
OUTPUT_TXT = PROCESSED / "pokemon_kb.txt"  # Consolidated KB output path

CODE_TO_MULT = {-5: 0.0, -1: 0.5, 0: 1.0, 1: 2.0}  # Map CSV codes to damage multipliers


def pick_col(df: pd.DataFrame, *names):
    """Return the first matching column by name (tolerates schema differences)."""
    for n in names:  # Try provided fallbacks in order
        if n in df.columns:
            return df[n]  # Found a matching column; return its Series
    return None  # No suggested column present

def load_type_chart_df(path: Path):
    """Load a type effectiveness chart into a nested dict: chart[attacker][defender] → multiplier."""
    df = pd.read_csv(path)  # Read the type chart table
    defenders = [c for c in df.columns if c != df.columns[0]]  # All defender types except header
    chart = {}  # {attacker_type: {defender_type: multiplier}}
    for _, row in df.iterrows():  # Iterate each attacker type row
        atk = str(row.iloc[0]).strip()  # Attacker type in first column
        chart[atk] = {}  # Initialize defender mapping
        for i, d in enumerate(defenders):  # For each defender type column
            val = row.iloc[i + 1]  # Raw code value at (attacker, defender)
            try:
                code = int(val)  # Convert to int code when possible
            except Exception:
                code = 0  # Default to neutral if conversion fails
            chart[atk][d] = CODE_TO_MULT.get(code, 1.0)  # Map to damage multiplier
    return defenders, chart  # Return defender list + chart


def load_pokemon_types_df(path: Path):
    """Return a list of Pokémon with normalized primary/secondary types."""
    df = pd.read_csv(path)  # Load type data
    name = pick_col(df, "name", "Name", "Pokemon")  # Schema-flexible name column
    type1 = pick_col(df, "type1", "Primary Type")  # Primary type column
    type2 = pick_col(df, "type2", "Secondary Type")  # Secondary type column
    res = []  # Collect normalized entries
    for i in range(len(df)):  # Walk each row by index
        n = str(name.iloc[i]).strip() if name is not None else ""  # Pokémon name
        if not n:  # Skip empty names
            continue
        t1 = str(type1.iloc[i]).strip() if type1 is not None else ""  # Primary type value
        t2 = str(type2.iloc[i]).strip() if type2 is not None else ""  # Secondary type value
        if t2 in ("", "None", "null", "NULL"):  # Normalize empty/none-like values
            t2 = None
        res.append({"name": n, "type1": t1 or "Unknown", "type2": t2})  # Default unknowns
    return res  # List of dicts with name + types


def load_pokemon_meta_df(path: Path):
    """Build a per-Pokémon metadata lookup (dex, gen, classification, abilities, size)."""
    df = pd.read_csv(path)  # Load metadata source CSV
    meta = {}  # {name: {field: value}}
    name = pick_col(df, "name", "Name")  # Name column (schema tolerant)
    abl = pick_col(df, "abilities", "Abilities")  # Abilities column (may be list-like text)
    for i in range(len(df)):
        n = str(name.iloc[i]).strip() if name is not None else ""  # Pokémon name
        if not n:  # Skip rows without a name
            continue
        abilities_raw = str(abl.iloc[i]).strip() if abl is not None else ""  # Raw abilities text
        abilities_fmt = ""  # Normalized abilities string
        if abilities_raw:
            try:
                parsed = ast.literal_eval(abilities_raw)  # Try parse string as Python literal
                if isinstance(parsed, (list, tuple)):
                    abilities_fmt = ", ".join(str(x) for x in parsed)  # Join lists into CSV
                else:
                    abilities_fmt = str(parsed)  # Fallback to scalar string
            except Exception:
                # Fallback: strip brackets/quotes for simple list-like strings
                abilities_fmt = abilities_raw.strip("[]").replace("'", "").replace('"', "")
        # Collect commonly used metadata, defaulting to placeholders when missing
        meta[n] = {
            "pokedex_number": str(df.get("pokedex_number", pd.Series([""]*len(df))).iloc[i]).strip(),
            "generation": str(df.get("generation", pd.Series([""]*len(df))).iloc[i]).strip(),
            "classification": str(df.get("classification", pd.Series([""]*len(df))).iloc[i]).strip() or "Unknown",
            "abilities": abilities_fmt or "Unknown",
            "height_m": str(df.get("height_m", pd.Series([""]*len(df))).iloc[i]).strip() or "?",
            "weight_kg": str(df.get("weight_kg", pd.Series([""]*len(df))).iloc[i]).strip() or "?",
            "base_total": str(df.get("base_total", pd.Series([""]*len(df))).iloc[i]).strip() or "?",
        }
    return meta  # Name → metadata dict


def load_stats_lookup_df(path: Path):
    """Create a name → base stats mapping with schema fallbacks for Sp.Atk/Sp.Def."""
    df = pd.read_csv(path)  # Read stats table
    name = pick_col(df, "Name", "name")  # Name column aliasing
    lookup = {}  # {name: {stat_name: value}}
    for i in range(len(df)):
        n = str(name.iloc[i]).strip() if name is not None else ""  # Pokémon name
        if not n:  # Skip unnamed rows
            continue
        stats = {
            "HP": str(df.get("HP", pd.Series([""]*len(df))).iloc[i]).strip(),  # HP value
            "Attack": str(df.get("Attack", pd.Series([""]*len(df))).iloc[i]).strip(),  # Attack value
            "Defense": str(df.get("Defense", pd.Series([""]*len(df))).iloc[i]).strip(),  # Defense value
            "Sp.Atk": str(df.get("Sp.Atk", df.get("sp_attack", pd.Series([""]*len(df)))).iloc[i]).strip(),  # Special Attack
            "Sp.Def": str(df.get("Sp.Def", df.get("sp_defense", pd.Series([""]*len(df)))).iloc[i]).strip(),  # Special Defense
            "Speed": str(df.get("Speed", pd.Series([""]*len(df))).iloc[i]).strip(),  # Speed value
        }
        lookup[n] = stats  # Store stats keyed by name
    return lookup  # Final stats lookup


def load_id_to_name_df(path: Path):
    """Map Pokémon numeric IDs (dex or internal) to their names."""
    df = pd.read_csv(path)  # Load dataset with IDs
    pid = pick_col(df, "pokedex_number", "pokemon_id")  # ID column (dex or alt)
    name = pick_col(df, "Name", "name")  # Name column
    id_to_name = {}  # {id_str: name}
    for i in range(len(df)):
        p = str(pid.iloc[i]).strip() if pid is not None else ""  # ID as string
        n = str(name.iloc[i]).strip() if name is not None else ""  # Name
        if p and n:  # Only map valid pairs
            id_to_name[p] = n
    return id_to_name  # ID → name mapping


def load_moves_df(path: Path):
    """Load moves with flexible column detection and normalize missing values."""
    df = pd.read_csv(path)  # Move data
    cols = list(df.columns)  # All column names
    def key(*candidates):  # Helper to choose first present column among candidates
        for c in candidates:
            if c in cols:
                return c
        return candidates[0]  # Fall back to first candidate if none present
    name_k = key("Name", cols[0])  # Move name column
    type_k = key("Type")  # Move type column
    cat_k = key("Cat.", "Category")  # Category: Physical/Special/Status
    pow_k = key("Power")  # Base power
    acc_k = key("Acc.", "Accuracy")  # Accuracy percentage
    pp_k = key("PP")  # PP count
    eff_k = key("Effect")  # Effect description
    moves = []  # Collect normalized records
    for _, row in df.iterrows():  # Iterate rows as dict-like
        name = str(row.get(name_k, "")).strip()  # Move name
        if not name:  # Skip unnamed moves
            continue
        moves.append({
            "Name": name,
            "Type": str(row.get(type_k, "Unknown")).strip() or "Unknown",  # Default unknown
            "Category": str(row.get(cat_k, "—")).strip() or "—",  # Em dash for missing
            "Power": str(row.get(pow_k, "—")).strip() or "—",  # Missing power → em dash
            "Accuracy": str(row.get(acc_k, "—")).strip() or "—",  # Missing accuracy → em dash
            "PP": str(row.get(pp_k, "—")).strip() or "—",  # Missing PP → em dash
            "Effect": str(row.get(eff_k, "—")).strip() or "—",  # Missing effect → em dash
        })
    return moves  # List of move dicts


def load_evolutions_df(path: Path):
    """Read evolution criteria per Pokémon; normalize unknowns to a friendly message."""
    df = pd.read_csv(path)  # Evolution criteria table
    name_k = "pokemon_name" if "pokemon_name" in df.columns else ("name" if "name" in df.columns else None)  # Flexible name key
    evo = {}  # {name: criteria}
    for _, row in df.iterrows():  # Walk rows
        n = str(row.get(name_k, "")).strip()  # Pokémon name
        crit = str(row.get("evolution_criteria", "")).strip()  # Criteria text
        if not n:  # Skip if no name present
            continue
        if crit.lower().startswith("not found"):  # Normalize "not found" markers
            crit = "Evolution details unavailable."
        evo[n] = crit or "Evolution details unavailable."  # Default to friendly message
    return evo  # Name → evolution text


def clean_text(s: str) -> str:
    """Collapse whitespace and remove newlines for concise, single-line summaries."""
    if not s:
        return ""  # Preserve empty values
    s = str(s).replace("\n", " ").replace("\r", " ")  # Replace line breaks with spaces
    while "  " in s:
        s = s.replace("  ", " ")  # Deduplicate consecutive spaces
    return s.strip()  # Trim leading/trailing spaces


def load_pokemon_locations_df(path: Path, id_to_name: dict):
    """Aggregate per-game encounter locations, keyed by Pokémon name via ID lookup."""
    df = pd.read_csv(path)  # Locations per game/version
    exclude = {"pokemon_id", "pokedex_number", "name", "Name", "Variant"}  # Non-game columns to skip
    game_cols = [c for c in df.columns if c not in exclude]  # Game/version columns
    locs = {}  # {name: "Game1: places; Game2: places"}
    for _, row in df.iterrows():  # Iterate rows
        pid_val = row.get("pokemon_id", row.get("pokedex_number", ""))  # Prefer pokemon_id, fallback to dex
        pid = str(pid_val).strip()  # ID as string
        name = id_to_name.get(pid)  # Resolve to Pokémon name
        if not name:  # Skip unknown IDs
            continue
        parts = []  # Build summary across games
        for g in game_cols:  # Walk each game/version
            val = clean_text(row.get(g, ""))  # Normalize value text
            if val:  # Include non-empty entries
                parts.append(f"{g}: {val}")
        if parts:  # Only store if we have some location data
            locs[name] = "; ".join(parts)  # Join per-game summaries with semicolons
    return locs  # Name → locations summary


def load_item_locations_df(path: Path):
    """Summarize item locations across games; default to 'unavailable' when empty."""
    df = pd.read_csv(path)  # Item locations dataset
    name_k = "item_name" if "item_name" in df.columns else df.columns[0]  # Prefer item_name, fallback to first col
    exclude = {name_k, "item_id", "id", "ID"}  # Non-game columns to ignore
    game_cols = [c for c in df.columns if c not in exclude]  # Game/version columns
    items = []  # [{name, summary}]
    for _, row in df.iterrows():  # Iterate items
        name = str(row.get(name_k, "")).strip()  # Item name
        if not name:  # Skip missing names
            continue
        parts = []  # Build per-game location fragments
        for g in game_cols:
            val = clean_text(row.get(g, ""))  # Normalize text
            if val:
                parts.append(f"{g}: {val}")
        summary = "; ".join(parts) if parts else "Locations unavailable"  # Default fallback
        items.append({"name": name, "summary": summary})  # Append normalized record
    return items  # List of item summaries


def combine_effects(type1: str, type2: str | None, defender_types: list[str], chart: dict):
    """Compute weaknesses/resistances/immunities for a Pokémon's typing using the chart."""
    weaknesses, resistances, immunities = [], [], []  # Buckets for output
    for atk in defender_types:  # For each attacking type
        m1 = chart.get(atk, {}).get(type1, 1.0)  # Multiplier vs. primary type
        m2 = chart.get(atk, {}).get(type2, 1.0) if type2 else 1.0  # Multiplier vs. secondary type
        total = m1 * m2  # Combined multiplier (e.g., 4.0 for double weakness)
        if total == 0.0:  # No effect
            immunities.append(atk)
        elif total > 1.0:  # Weakness
            label = f"{atk} (4x)" if total >= 4.0 else atk  # Flag 4x weaknesses
            weaknesses.append(label)
        elif 0.0 < total < 1.0:  # Resistance
            label = f"{atk} (0.25x)" if total <= 0.25 else atk  # Flag 0.25x resistances
            resistances.append(label)
    return weaknesses, resistances, immunities  # Categorized effectiveness


def build_kb():
    """Construct and write a human-readable Pokémon knowledge base to OUTPUT_TXT."""
    PROCESSED.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists
    defender_types, chart = load_type_chart_df(TYPES_CHART_CSV)  # Type effectiveness
    pokemons = load_pokemon_types_df(POKEMON_TYPES_CSV)  # Names + types
    meta_lookup = load_pokemon_meta_df(POKEMON_TYPES_CSV)  # Dex/gen/classification/abilities/size
    stats_lookup = load_stats_lookup_df(POKEMON_STATS_CSV)  # Base stats
    moves = load_moves_df(POKEMON_MOVES_CSV)  # Move list
    id_to_name = load_id_to_name_df(POKEMON_STATS_CSV)  # ID → name mapping for locations
    evolutions = load_evolutions_df(POKEMON_EVOS_CSV)  # Evolution criteria
    locations = load_pokemon_locations_df(POKEMON_LOCS_CSV, id_to_name)  # Encounter locations
    items = load_item_locations_df(ITEM_LOCS_CSV)  # Item locations

    names_index = sorted(set(p["name"] for p in pokemons))  # Alphabetical index of Pokémon names

    with OUTPUT_TXT.open("w", encoding="utf-8") as out:  # Open output file for writing
        out.write("Pokémon Knowledge Base.\n")  # Header line
        # Index preview (first 200 names; append ellipsis if longer)
        out.write("Index: " + ", ".join(names_index[:200]) + ("..." if len(names_index) > 200 else "") + "\n\n")

        out.write("Type Chart Summary:\n")  # Section header
        for atk in chart.keys():  # For each attacking type
            weak = [d for d, mult in chart[atk].items() if mult == 2.0]  # 2x damage targets
            resist = [d for d, mult in chart[atk].items() if mult == 0.5]  # 0.5x targets
            immune = [d for d, mult in chart[atk].items() if mult == 0.0]  # 0x targets
            out.write(f"Type — {atk} — Super effective against {', '.join(weak) or 'None'}. ")
            out.write(f"Not very effective against {', '.join(resist) or 'None'}. ")
            out.write(f"No effect on {', '.join(immune) or 'None'}.\n")
        out.write("\n")  # Blank line between sections

        # Per-Pokémon summaries
        for p in sorted(pokemons, key=lambda x: x["name"].lower()):  # Stable name order
            name = p["name"]  # Pokémon name
            t1 = p["type1"]  # Primary type
            t2 = p["type2"]  # Optional secondary type
            weak, resist, immune = combine_effects(t1, t2, defender_types, chart)  # Effectiveness buckets
            stats = stats_lookup.get(name)  # Base stats lookup
            if stats:
                stats_text = (
                    f"Stats: HP {stats['HP']}/Atk {stats['Attack']}/Def {stats['Defense']}/"
                    f"SpA {stats['Sp.Atk']}/SpD {stats['Sp.Def']}/Spe {stats['Speed']}."
                )  # Inline stat summary
            else:
                stats_text = "Stats: Unknown."  # Placeholder when stats missing
            type_label = f"{t1}/{t2}" if t2 else t1  # Human-readable typing
            weak_text = ", ".join(weak) if weak else "None"  # Weakness list
            resist_text = ", ".join(resist) if resist else "None"  # Resistance list
            immune_text = ", ".join(immune) if immune else "None"  # Immunity list
            meta = meta_lookup.get(name, {})  # Metadata dict (may be empty)
            dex = meta.get("pokedex_number") or "?"  # Pokedex number
            gen = meta.get("generation") or "?"  # Generation label
            classification = meta.get("classification") or "Unknown"  # Species classification
            abilities = meta.get("abilities") or "Unknown"  # Known abilities
            height_m = meta.get("height_m") or "?"  # Height in meters
            weight_kg = meta.get("weight_kg") or "?"  # Weight in kilograms
            base_total = str(meta.get("base_total") or "?")  # Base stat total

            # Compose the Pokémon summary line(s)
            out.write(f"Pokemon — {name} — Dex# {dex}, Gen {gen}, Classification: {classification}. ")
            out.write(f"Abilities: {abilities}. Height: {height_m} m, Weight: {weight_kg} kg. Base total: {base_total}. ")
            out.write(f"{name} is a {type_label}-type Pokémon. ")
            out.write(f"Weak to: {weak_text}. ")
            out.write(f"Resists: {resist_text}. ")
            out.write(f"Immune to: {immune_text}. ")
            evo_text = evolutions.get(name)  # Evolution criteria
            if evo_text:
                out.write(f"Evolution: {evo_text} ")
            loc_text = locations.get(name)  # Encounter locations
            if loc_text:
                out.write(f"Locations: {loc_text}. ")
            out.write(stats_text + "\n")  # Terminate this Pokémon's entry

        out.write("\n")  # Spacing before moves
        out.write("Moves Index:\n")  # Moves section header
        if moves:
            # Preview first 200 move names; add ellipsis when truncated
            out.write(
                ", ".join(sorted(set(m["Name"] for m in moves))[:200])
                + ("..." if len(moves) > 200 else "")
                + "\n\n"
            )
        else:
            out.write("(No moves found)\n\n")  # Handle empty dataset gracefully

        out.write("Move Summaries:\n")  # Detailed move entries
        for m in sorted(moves, key=lambda x: x["Name"].lower()):  # Stable alphabetical order
            name = m["Name"]  # Move name
            type_ = m["Type"]  # Move type
            cat = m["Category"]  # Physical/Special/Status
            power = m["Power"]  # Base power or em dash
            acc = m["Accuracy"]  # Accuracy % or em dash
            pp = m["PP"]  # PP count or em dash
            effect = m["Effect"] if m["Effect"] else "—"  # Effect text or em dash
            out.write(
                f"Move — {name} — Type={type_}; Category={cat}; Power={power}; Accuracy={acc}; PP={pp}. "
                f"Effect: {effect}.\n"
            )  # Single-line move summary

        out.write("\nItem Locations:\n")  # Item section header
        if items:
            for item in sorted(items, key=lambda x: x["name"].lower()):  # Alphabetical by item name
                out.write(f"Item — {item['name']} — {item['summary']}\n")  # One-line item entry
        else:
            out.write("(No item locations found)\n")  # Handle empty dataset gracefully

    print(f"Wrote {OUTPUT_TXT.relative_to(ROOT)}")  # Console confirmation of output path


if __name__ == "__main__":  # Allow running as a script
    build_kb()  # Generate and write the knowledge base
