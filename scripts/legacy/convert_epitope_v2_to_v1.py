"""
Convert confluencia 2.0 epitope training data to 1.0-compatible format.

Source (READ ONLY): confluencia-2.0-epitope/data/epitope_training_confluencia.csv
Target:             新建文件夹/data/epitope_from_v2.csv

Column mapping:
    epitope_seq    → sequence
    dose           → concentration
    treatment_time → incubation_hours
    freq, circ_expr, ifn_score → kept as additional env_cols
    efficacy       → efficacy (unchanged)

NOTE: This script never modifies the source 2.0 data file.
"""

import pandas as pd
from pathlib import Path

SRC = Path(r"D:\IGEM集成方案\confluencia-2.0-epitope\data\epitope_training_confluencia.csv")
DST = Path(r"D:\IGEM集成方案\新建文件夹\data\epitope_from_v2.csv")

RENAME_MAP = {
    "epitope_seq": "sequence",
    "dose": "concentration",
    "treatment_time": "incubation_hours",
}

# Desired column order: sequence first, then env_cols, then target
COL_ORDER = [
    "sequence",
    "concentration",
    "incubation_hours",
    "freq",
    "circ_expr",
    "ifn_score",
    "efficacy",
]


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source file not found: {SRC}")

    print(f"Reading source: {SRC}")
    df = pd.read_csv(SRC)
    print(f"  Source shape: {df.shape}")
    print(f"  Source columns: {list(df.columns)}")

    # Rename columns
    out = df.rename(columns=RENAME_MAP)

    # Verify all expected columns exist after rename
    missing = [c for c in COL_ORDER if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}")

    # Select and reorder columns
    out = out[COL_ORDER]

    # Filter: drop rows with NaN in sequence or efficacy
    before = len(out)
    out = out.dropna(subset=["sequence", "efficacy"])
    print(f"  Dropped {before - len(out)} rows with NaN in sequence/efficacy")

    # Filter: sequence length >= 8 (matching epitope predictor validation)
    before = len(out)
    out = out[out["sequence"].str.strip().str.len() >= 8]
    print(f"  Dropped {before - len(out)} rows with sequence length < 8")

    # Filter: efficacy must be numeric
    out["efficacy"] = pd.to_numeric(out["efficacy"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["efficacy"])
    print(f"  Dropped {before - len(out)} rows with non-numeric efficacy")

    # Reset index
    out = out.reset_index(drop=True)

    # Save
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"\nSaved {len(out)} rows to {DST}")
    print(f"  Columns: {list(out.columns)}")
    print(f"  env_cols (auto-detected): concentration, incubation_hours, freq, circ_expr, ifn_score")
    print(f"  Efficacy range: [{out['efficacy'].min():.4f}, {out['efficacy'].max():.4f}]")
    print(f"  Sequence length range: [{out['sequence'].str.len().min()}, {out['sequence'].str.len().max()}]")


if __name__ == "__main__":
    main()
