"""
Convert confluencia 2.0 drug training data to 1.0-compatible format.

Source (READ ONLY): confluencia-2.0-drug/data/breast_cancer_drug_dataset.csv
Target:             新建文件夹/data/drug_from_v2.csv
                    新建文件夹/data/drug_from_v2_minimal.csv (4 columns only)

Column handling:
    smiles, dose, freq, efficacy → kept (core columns)
    treatment_time, target_binding, immune_activation,
    immune_cell_activation, inflammation_risk, toxicity_risk → kept as env_cols
    epitope_seq, group_id → DROPPED (strings, not used by drug featurizer)

NOTE: This script never modifies the source 2.0 data file.
"""

import pandas as pd
from pathlib import Path

SRC = Path(r"D:\IGEM集成方案\confluencia-2.0-drug\data\breast_cancer_drug_dataset.csv")
DST = Path(r"D:\IGEM集成方案\新建文件夹\data\drug_from_v2.csv")
DST_MINIMAL = Path(r"D:\IGEM集成方案\新建文件夹\data\drug_from_v2_minimal.csv")

DROP_COLS = ["epitope_seq", "group_id"]
MINIMAL_COLS = ["smiles", "dose", "freq", "efficacy"]


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source file not found: {SRC}")

    print(f"Reading source: {SRC}")
    df = pd.read_csv(SRC)
    print(f"  Source shape: {df.shape}")
    print(f"  Source columns: {list(df.columns)}")

    # Drop string columns the drug featurizer cannot use
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Reorder: smiles first, then env_cols (all numeric except smiles+efficacy), then target
    env_cols = [c for c in out.columns if c not in ("smiles", "efficacy")]
    out = out[["smiles"] + env_cols + ["efficacy"]]

    # Filter: drop rows with NaN in smiles or efficacy
    before = len(out)
    out = out.dropna(subset=["smiles", "efficacy"])
    print(f"  Dropped {before - len(out)} rows with NaN in smiles/efficacy")

    # Filter: efficacy must be numeric
    out["efficacy"] = pd.to_numeric(out["efficacy"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["efficacy"])
    print(f"  Dropped {before - len(out)} rows with non-numeric efficacy")

    # Reset index
    out = out.reset_index(drop=True)

    # Save full version
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"\nSaved {len(out)} rows to {DST}")
    print(f"  Columns: {list(out.columns)}")
    print(f"  env_cols (auto-detected): {env_cols}")
    print(f"  Efficacy range: [{out['efficacy'].min():.4f}, {out['efficacy'].max():.4f}]")

    # Save minimal version (matching original 1.0 example format exactly)
    minimal = out[MINIMAL_COLS].copy()
    minimal.to_csv(DST_MINIMAL, index=False)
    print(f"\nSaved {len(minimal)} rows to {DST_MINIMAL}")
    print(f"  Columns: {MINIMAL_COLS}")


if __name__ == "__main__":
    main()
