"""
Extract epitope data from IEDB T-cell assay data (already downloaded in 2.0 raw).

Source (READ ONLY): confluencia-2.0-epitope/data/raw/iedb_tcell_full_v3.zip
Target:             新建文件夹/data/epitope_iedb.csv

Key IEDB columns (by index, since column names are duplicated):
    11  - Epitope Name (peptide sequence)
    122 - Qualitative Measurement (Positive/Negative)
    124 - Quantitative measurement (numeric, e.g. IC50 nM)
    127 - Response Frequency (%)

Mapping to 1.0 format:
    sequence  <- Epitope Name
    efficacy  <- derived from Qualitative/Quantitative measurement
    concentration, incubation_hours, freq, circ_expr, ifn_score <- filled with median values

NOTE: This script never modifies the source 2.0 data file.
"""

import zipfile
import csv
import io
import re
from pathlib import Path
import pandas as pd
import numpy as np

SRC_ZIP = Path(r"D:\IGEM集成方案\confluencia-2.0-epitope\data\raw\iedb_tcell_full_v3.zip")
DST = Path(r"D:\IGEM集成方案\新建文件夹\data\epitope_iedb.csv")

# IEDB column indices
COL_EPITYPE_NAME = 11
COL_QUALITATIVE = 122
COL_QUANTITATIVE = 124
COL_RESPONSE_FREQ = 127

# Qualitative measurement mapping
QUALITATIVE_MAP = {
    "Positive": 1.0,
    "Positive-High": 1.5,
    "Positive-Intermediate": 1.0,
    "Positive-Low": 0.5,
    "Negative": -1.0,
    "Negative-Low": -0.5,
}

# Median env values from existing epitope_from_v2.csv (to fill missing env_cols)
MEDIAN_ENV = {
    "concentration": 2.5,
    "incubation_hours": 24.0,
    "freq": 1.0,
    "circ_expr": 1.0,
    "ifn_score": 0.5,
}

# Valid amino acid characters
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_quantitative(val: str) -> float | None:
    """Try to extract a numeric value from quantitative measurement field."""
    if not val or val.strip() == "":
        return None
    # Remove common prefixes/suffixes
    val = val.strip()
    # Try direct float conversion
    try:
        return float(val)
    except ValueError:
        pass
    # Try extracting first number from string like ">10000" or "< 50"
    match = re.search(r"[\d.]+", val)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def is_valid_peptide(seq: str) -> bool:
    """Check if sequence is a valid peptide (8-30 AA, only standard amino acids)."""
    if not seq or len(seq) < 8 or len(seq) > 30:
        return False
    return all(c in VALID_AA for c in seq.upper())


def main():
    if not SRC_ZIP.exists():
        raise FileNotFoundError(f"Source file not found: {SRC_ZIP}")

    print(f"Reading IEDB data from: {SRC_ZIP}")

    records = []
    total_rows = 0
    valid_peptides = 0
    has_efficacy = 0

    with zipfile.ZipFile(SRC_ZIP, "r") as zf:
        with zf.open("tcell_full_v3.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            next(reader)  # skip header

            for row in reader:
                total_rows += 1
                if total_rows % 500000 == 0:
                    print(f"  Processed {total_rows} rows, collected {len(records)} records...")

                # Extract epitope name
                seq = row[COL_EPITYPE_NAME].strip() if len(row) > COL_EPITYPE_NAME else ""
                if not is_valid_peptide(seq):
                    continue
                valid_peptides += 1

                # Derive efficacy from qualitative or quantitative measurement
                efficacy = None

                # Try qualitative first
                qual = row[COL_QUALITATIVE].strip() if len(row) > COL_QUALITATIVE else ""
                if qual in QUALITATIVE_MAP:
                    efficacy = QUALITATIVE_MAP[qual]

                # Try quantitative if qualitative is missing or ambiguous
                quant_str = row[COL_QUANTITATIVE].strip() if len(row) > COL_QUANTITATIVE else ""
                quant_val = parse_quantitative(quant_str)
                if quant_val is not None and quant_val > 0:
                    # Convert to log-scale efficacy (lower IC50 = stronger binding)
                    # Assume nM units; use -log10(IC50_nM / 1e6) to get reasonable range
                    efficacy = -np.log10(max(quant_val, 1e-3) / 1e6)

                # Try response frequency as fallback
                if efficacy is None:
                    freq_str = row[COL_RESPONSE_FREQ].strip() if len(row) > COL_RESPONSE_FREQ else ""
                    if freq_str:
                        try:
                            rf = float(freq_str)
                            # Map 0-100% to efficacy range
                            efficacy = (rf / 100.0) * 2.0 - 1.0  # [-1, 1]
                        except ValueError:
                            pass

                if efficacy is None:
                    continue

                has_efficacy += 1
                records.append({
                    "sequence": seq.upper(),
                    "concentration": MEDIAN_ENV["concentration"],
                    "incubation_hours": MEDIAN_ENV["incubation_hours"],
                    "freq": MEDIAN_ENV["freq"],
                    "circ_expr": MEDIAN_ENV["circ_expr"],
                    "ifn_score": MEDIAN_ENV["ifn_score"],
                    "efficacy": round(efficacy, 6),
                })

    print(f"\n  Total IEDB rows: {total_rows}")
    print(f"  Valid peptide sequences (8-30 AA): {valid_peptides}")
    print(f"  Records with efficacy value: {has_efficacy}")

    if not records:
        print("ERROR: No valid records extracted!")
        return

    df = pd.DataFrame(records)
    print(f"\n  DataFrame shape: {df.shape}")

    # Drop exact duplicates (same sequence + efficacy)
    before = len(df)
    df = df.drop_duplicates(subset=["sequence", "efficacy"])
    print(f"  After dedup: {len(df)} rows (removed {before - len(df)} duplicates)")

    # Clip efficacy to reasonable range
    df["efficacy"] = df["efficacy"].clip(-2.0, 6.0)

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"\nSaved {len(df)} rows to {DST}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Efficacy range: [{df['efficacy'].min():.4f}, {df['efficacy'].max():.4f}]")
    print(f"  Unique sequences: {df['sequence'].nunique()}")


if __name__ == "__main__":
    main()
