"""
Validate converted data files — column names, NaN, numeric types, env detection.

Works for both 1.0 (早期版) and 2.0 format data.

Usage
-----
    python -m scripts.validate_data data/
    python -m scripts.validate_data data/epitope_merged.csv data/drug_merged.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List

import pandas as pd


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_epitope(path: Path) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Validating epitope: {path.name}")
    print(f"{'=' * 60}")

    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    ok = True

    # Required columns
    for col in ("sequence", "efficacy"):
        if col not in df.columns:
            print(f"  [FAIL] Missing required column: {col}")
            ok = False
        else:
            print(f"  [OK] Required column present: {col}")

    if not ok:
        return False

    # NaN check
    for col in ("sequence", "efficacy"):
        na_count = df[col].isna().sum()
        if na_count > 0:
            print(f"  [WARN] {col}: {na_count} NaN values")
        else:
            print(f"  [OK] {col}: no NaN")

    # Numeric efficacy
    if not pd.api.types.is_numeric_dtype(df["efficacy"]):
        print(f"  [FAIL] efficacy is not numeric")
        return False
    print(f"  [OK] efficacy is numeric, range: [{df['efficacy'].min():.4f}, {df['efficacy'].max():.4f}]")

    # Sequence length
    seq_lens = df["sequence"].str.len()
    print(f"  [OK] Sequence length range: [{seq_lens.min()}, {seq_lens.max()}]")

    # Auto-detect env_cols
    env_cols = [
        c for c in df.columns
        if c not in ("sequence", "efficacy") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if env_cols:
        print(f"  [OK] Auto-detected env_cols: {env_cols}")
    else:
        print(f"  [WARN] No env_cols auto-detected")

    print(f"  [PASS] Epitope validation complete")
    return True


def validate_drug(path: Path) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Validating drug: {path.name}")
    print(f"{'=' * 60}")

    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    ok = True

    for col in ("smiles", "efficacy"):
        if col not in df.columns:
            print(f"  [FAIL] Missing required column: {col}")
            ok = False
        else:
            print(f"  [OK] Required column present: {col}")

    if not ok:
        return False

    for col in ("smiles", "efficacy"):
        na_count = df[col].isna().sum()
        if na_count > 0:
            print(f"  [WARN] {col}: {na_count} NaN values")
        else:
            print(f"  [OK] {col}: no NaN")

    if not pd.api.types.is_numeric_dtype(df["efficacy"]):
        print(f"  [FAIL] efficacy is not numeric")
        return False
    print(f"  [OK] efficacy is numeric, range: [{df['efficacy'].min():.4f}, {df['efficacy'].max():.4f}]")

    env_cols = [
        c for c in df.columns
        if c not in ("smiles", "efficacy") and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  [OK] Auto-detected env_cols: {env_cols}")
    print(f"  [OK] Unique SMILES: {df['smiles'].nunique()}")
    print(f"  [PASS] Drug validation complete")
    return True


def validate_file(path: Path) -> bool:
    """Auto-detect type and validate."""
    name = path.name.lower()
    if "epitope" in name:
        return validate_epitope(path)
    elif "drug" in name:
        return validate_drug(path)
    else:
        # Try both
        print(f"\n  Unknown type for {path.name}, trying drug columns ...")
        df = pd.read_csv(path, nrows=0)
        if "smiles" in df.columns:
            return validate_drug(path)
        elif "sequence" in df.columns:
            return validate_epitope(path)
        else:
            print(f"  [SKIP] Cannot determine data type for {path.name}")
            return True


def main():
    parser = argparse.ArgumentParser(description="Validate Confluencia datasets")
    parser.add_argument("paths", nargs="+", type=Path,
                        help="CSV files or directory to validate")
    args = parser.parse_args()

    errors: List[str] = []
    files: List[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        else:
            files.append(p)

    for f in files:
        if not f.exists():
            print(f"  [SKIP] Not found: {f}")
            continue
        md5 = _md5_file(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB, MD5={md5}")
        try:
            if not validate_file(f):
                errors.append(f.name)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            errors.append(f.name)

    print(f"\n{'=' * 60}")
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} file(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
