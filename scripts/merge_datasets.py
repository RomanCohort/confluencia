"""
Merge multiple collected datasets into final unified files.

Supports both 1.0 (早期版) and 2.0 data paths.

Epitope merge: epitope_from_v2 + epitope_iedb [+ any extra] → epitope_merged.csv
Drug merge:    drug_from_v2 + drug_chembl + pubchem_crawl → drug_merged.csv

Deduplication:
  - Epitope: by ``sequence`` (keep first)
  - Drug:    by ``smiles`` (keep first)

Usage
-----
    python -m scripts.merge_datasets --data-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def _load_if_exists(path: Path, *, source_label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    df["_source"] = source_label
    print(f"  Loaded {path.name}: {len(df)} rows, columns={list(df.columns)}")
    return df


def merge_epitope(
    data_dir: Path,
    *,
    extra_paths: Optional[Sequence[Path]] = None,
    dedup_col: str = "sequence",
) -> pd.DataFrame:
    """Merge all epitope datasets under *data_dir*."""
    print("=" * 60)
    print("Merging epitope datasets")
    print("=" * 60)

    candidates = [
        (data_dir / "epitope_from_v2.csv", "v2_converted"),
        (data_dir / "epitope_from_v2_minimal.csv", "v2_minimal"),
        (data_dir / "epitope_iedb.csv", "iedb"),
        (data_dir / "epitope_crawl_out.csv", "crawled"),
    ]
    if extra_paths:
        for p in extra_paths:
            candidates.append((Path(p), Path(p).stem))

    frames: List[pd.DataFrame] = []
    for path, label in candidates:
        df = _load_if_exists(path, source_label=label)
        if df is not None:
            frames.append(df)

    if not frames:
        print("  No epitope datasets found!")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Before dedup: {len(merged)} rows")

    merged = merged.dropna(subset=["sequence", "efficacy"])
    print(f"  After dropna: {len(merged)} rows")

    before = len(merged)
    merged = merged.drop_duplicates(subset=[dedup_col], keep="first")
    print(f"  After dedup (by {dedup_col}): {len(merged)} rows (removed {before - len(merged)})")

    merged = merged.drop(columns=["_source"], errors="ignore")
    merged = merged.reset_index(drop=True)

    dst = data_dir / "epitope_merged.csv"
    merged.to_csv(dst, index=False)
    print(f"\nSaved {len(merged)} rows to {dst}")
    return merged


def merge_drug(
    data_dir: Path,
    *,
    extra_paths: Optional[Sequence[Path]] = None,
    dedup_col: str = "smiles",
) -> pd.DataFrame:
    """Merge all drug datasets under *data_dir*."""
    print("\n" + "=" * 60)
    print("Merging drug datasets")
    print("=" * 60)

    candidates = [
        (data_dir / "drug_from_v2.csv", "v2_converted"),
        (data_dir / "drug_from_v2_minimal.csv", "v2_minimal"),
        (data_dir / "drug_chembl.csv", "chembl"),
        (data_dir / "drug_crawl_out.csv", "pubchem_crawled"),
        (data_dir / "drug_screen_out.csv", "pubchem_screened"),
    ]
    if extra_paths:
        for p in extra_paths:
            candidates.append((Path(p), Path(p).stem))

    frames: List[pd.DataFrame] = []
    for path, label in candidates:
        df = _load_if_exists(path, source_label=label)
        if df is not None:
            frames.append(df)

    if not frames:
        print("  No drug datasets found!")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Before dedup: {len(merged)} rows")

    merged = merged.dropna(subset=["smiles", "efficacy"])
    print(f"  After dropna: {len(merged)} rows")

    before = len(merged)
    merged = merged.drop_duplicates(subset=[dedup_col], keep="first")
    print(f"  After dedup (by {dedup_col}): {len(merged)} rows (removed {before - len(merged)})")

    merged = merged.drop(columns=["_source"], errors="ignore")
    merged = merged.reset_index(drop=True)

    dst = data_dir / "drug_merged.csv"
    merged.to_csv(dst, index=False)
    print(f"\nSaved {len(merged)} rows to {dst}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge Confluencia datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory containing source CSV files")
    parser.add_argument("--extra", nargs="*", help="Additional CSV paths to include")
    args = parser.parse_args()

    merge_epitope(args.data_dir, extra_paths=args.extra)
    merge_drug(args.data_dir, extra_paths=args.extra)


if __name__ == "__main__":
    main()
