"""
illuminance_summary.py
Computes illuminance_lux statistics per light_regime for all *_Data.csv
files in a folder, and combines the results into a single summary table.

Usage:
    python illuminance_summary.py <folder> [output.csv]

If no output path is given, the summary is printed to the console only.
"""

import sys
from pathlib import Path

import pandas as pd


def summarise_illuminance(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean, min, max, and range of illuminance_lux per light_regime."""
    return (
        df.groupby("light_regime")["illuminance_lux"]
        .agg(
            mean_lux="mean",
            min_lux="min",
            max_lux="max",
        )
        .assign(range_lux=lambda x: x["max_lux"] - x["min_lux"])
        .reset_index()
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python illuminance_summary.py <folder> [output.csv]")
        sys.exit(1)

    folder = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    csv_files = sorted(folder.glob("*_Data.csv"))
    if not csv_files:
        print(f"No files matching '*_Data.csv' found in '{folder}'.")
        sys.exit(1)

    summaries = []
    skipped = []

    for path in csv_files:
        df = pd.read_csv(path)
        missing = {"light_regime", "illuminance_lux"} - set(df.columns)
        if missing:
            skipped.append((path.name, missing))
            continue

        summary = summarise_illuminance(df)
        summary.insert(0, "file", path.name)
        summaries.append(summary)

    if not summaries:
        print("No valid files could be processed.")
        sys.exit(1)

    combined = pd.concat(summaries, ignore_index=True)

    print(f"\nIlluminance summary — {len(summaries)} file(s) in '{folder}':\n")
    print(combined.to_string(index=False, float_format="%.6f"))

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} file(s) due to missing columns:")
        for name, cols in skipped:
            print(f"   {name}: missing {cols}")

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Saved summary → {output_path}")


if __name__ == "__main__":
    main()
