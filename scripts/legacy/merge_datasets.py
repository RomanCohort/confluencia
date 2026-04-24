"""
Merge all collected datasets into final unified files.

Epitope merge:  epitope_from_v2.csv + epitope_iedb.csv → epitope_merged.csv
Drug merge:     drug_from_v2.csv + drug_chembl.csv → drug_merged.csv

Deduplication strategy:
  - Epitope: dedupe by (sequence, concentration, incubation_hours) — keep first occurrence
  - Drug: dedupe by (smiles, dose, freq) — keep first occurrence

NOTE: Source files are never modified.
"""

import pandas as pd
from pathlib import Path

BASE = Path(r"D:\IGEM集成方案\新建文件夹\data")


def merge_epitope():
    print("=" * 60)
    print("Merging epitope datasets")
    print("=" * 60)

    dfs = []

    # 1. v2 converted data (full env_cols)
    f1 = BASE / "epitope_from_v2.csv"
    if f1.exists():
        df1 = pd.read_csv(f1)
        df1["_source"] = "v2_converted"
        dfs.append(df1)
        print(f"  epitope_from_v2.csv: {len(df1)} rows, columns={list(df1.columns)}")

    # 2. IEDB data
    f2 = BASE / "epitope_iedb.csv"
    if f2.exists():
        df2 = pd.read_csv(f2)
        df2["_source"] = "iedb"
        dfs.append(df2)
        print(f"  epitope_iedb.csv: {len(df2)} rows, columns={list(df2.columns)}")

    if not dfs:
        print("  No epitope datasets to merge!")
        return

    # Concatenate
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Before dedup: {len(merged)} rows")

    # Drop rows with NaN in required columns
    merged = merged.dropna(subset=["sequence", "efficacy"])
    print(f"  After dropna: {len(merged)} rows")

    # Deduplicate by sequence + key env_cols
    # For IEDB data with default env values, dedupe by sequence only
    dedup_cols = ["sequence"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_cols, keep="first")
    print(f"  After dedup (by {dedup_cols}): {len(merged)} rows (removed {before - len(merged)})")

    # Remove _source column before saving
    merged = merged.drop(columns=["_source"])

    # Reset index
    merged = merged.reset_index(drop=True)

    # Save
    dst = BASE / "epitope_merged.csv"
    merged.to_csv(dst, index=False)
    print(f"\nSaved {len(merged)} rows to {dst}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"  Efficacy range: [{merged['efficacy'].min():.4f}, {merged['efficacy'].max():.4f}]")
    print(f"  Unique sequences: {merged['sequence'].nunique()}")


def merge_drug():
    print("\n" + "=" * 60)
    print("Merging drug datasets")
    print("=" * 60)

    dfs = []

    # 1. v2 converted data (full env_cols)
    f1 = BASE / "drug_from_v2.csv"
    if f1.exists():
        df1 = pd.read_csv(f1)
        df1["_source"] = "v2_converted"
        dfs.append(df1)
        print(f"  drug_from_v2.csv: {len(df1)} rows, columns={list(df1.columns)}")

    # 2. ChEMBL data (minimal columns: smiles, dose, freq, efficacy)
    f2 = BASE / "drug_chembl.csv"
    if f2.exists():
        df2 = pd.read_csv(f2)
        df2["_source"] = "chembl"
        dfs.append(df2)
        print(f"  drug_chembl.csv: {len(df2)} rows, columns={list(df2.columns)}")

    if not dfs:
        print("  No drug datasets to merge!")
        return

    # Concatenate
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Before dedup: {len(merged)} rows")

    # Drop rows with NaN in required columns
    merged = merged.dropna(subset=["smiles", "efficacy"])
    print(f"  After dropna: {len(merged)} rows")

    # Deduplicate by SMILES (keep first occurrence — v2 data has more features)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["smiles"], keep="first")
    print(f"  After dedup (by smiles): {len(merged)} rows (removed {before - len(merged)})")

    # Remove _source column before saving
    merged = merged.drop(columns=["_source"])

    # Reset index
    merged = merged.reset_index(drop=True)

    # Save
    dst = BASE / "drug_merged.csv"
    merged.to_csv(dst, index=False)
    print(f"\nSaved {len(merged)} rows to {dst}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"  Efficacy range: [{merged['efficacy'].min():.4f}, {merged['efficacy'].max():.4f}]")
    print(f"  Unique SMILES: {merged['smiles'].nunique()}")


def main():
    merge_epitope()
    merge_drug()


if __name__ == "__main__":
    main()
