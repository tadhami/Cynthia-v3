import csv
from pathlib import Path

PREFERRED_KEYS = [
    "Name",
    "Pokemon",
    "Move",
    "Type",
    "Attacking Type",
    "Defending Type",
]

def choose_key_column(fieldnames):
    for key in PREFERRED_KEYS:
        if key in fieldnames:
            return key
    # fallback to the first column
    return fieldnames[0] if fieldnames else "Entry"

def format_row(row: dict, key_col: str):
    key = row.get(key_col, "Entry")
    parts = []
    for col, val in row.items():
        if col == key_col:
            continue
        if val is None:
            continue
        text = str(val).strip()
        if text == "":
            continue
        parts.append(f"{col}={text}")
    values = "; ".join(parts)
    return f"{key}: {values}" if values else f"{key}"

def csv_to_text(input_csv: Path, output_txt: Path):
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        key_col = choose_key_column(reader.fieldnames)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with output_txt.open("w", encoding="utf-8") as out:
            for row in reader:
                line = format_row(row, key_col)
                out.write(line + "\n")

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    targets = [
        data_dir / "Pokemon_data.csv",
        data_dir / "Pokemon_data_types.csv",
        data_dir / "types_chart.csv",
        data_dir / "pokemon_moves.csv",
    ]
    for csv_path in targets:
        if not csv_path.exists():
            continue
        out_name = csv_path.stem + ".txt"
        out_path = processed_dir / out_name
        csv_to_text(csv_path, out_path)
        print(f"Wrote {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()