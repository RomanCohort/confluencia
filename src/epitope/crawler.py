"""
src.epitope.crawler — unified epitope data crawler for Confluencia.

Combines:
  - IEDB FASTA / CSV crawling   (from confluencia-2.0-epitope/tools/iedb_crawler.py)
  - IEDB raw T-cell extraction  (from scripts/legacy/fetch_iedb_epitope.py)

Public API
----------
crawl_epitope_fasta_sources(...)
    Fetch epitope sequences from FASTA URLs / UniProt / PDB / local files.

crawl_epitope_csv_datasets(...)
    Fetch and concatenate user-provided CSV/TSV/Excel epitope datasets.

crawl_epitope_iedb_raw(...)
    Extract epitope data from IEDB T-cell assay ZIP file.

crawl_all_epitope(...)
    One-shot: run all sources and return concatenated DataFrame.

clean_epitope_table(...)
    Clean / validate / deduplicate an epitope DataFrame.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ── Re-export the full 2.0 IEDB crawler ──────────────────────────────────
from confluencia_2_0_epitope.tools.iedb_crawler import (  # type: ignore[import-not-found]  # noqa: F401
    crawl_epitope_csv_datasets,
    crawl_epitope_fasta_sources,
    clean_epitope_table,
    parse_fasta_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

_QUALITATIVE_MAP: Dict[str, float] = {
    "Positive": 1.0,
    "Positive-High": 1.5,
    "Positive-Intermediate": 1.0,
    "Positive-Low": 0.5,
    "Negative": -1.0,
    "Negative-Low": -0.5,
}

# Median env values (for filling missing columns when extracting from IEDB raw)
_MEDIAN_ENV: Dict[str, float] = {
    "concentration": 2.5,
    "incubation_hours": 24.0,
    "freq": 1.0,
    "circ_expr": 1.0,
    "ifn_score": 0.5,
}

# ---------------------------------------------------------------------------
# IEDB raw T-cell extraction (legacy)
# ---------------------------------------------------------------------------

_COL_EPITOPE_NAME = 11
_COL_QUALITATIVE = 122
_COL_QUANTITATIVE = 124
_COL_RESPONSE_FREQ = 127


def _parse_quantitative(val: str) -> Optional[float]:
    """Extract numeric value from IEDB quantitative measurement field."""
    if not val or val.strip() == "":
        return None
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    match = re.search(r"[\d.]+", val)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _is_valid_peptide(seq: str, min_len: int = 8, max_len: int = 30) -> bool:
    if not seq or len(seq) < min_len or len(seq) > max_len:
        return False
    return all(c in AA_STANDARD for c in seq.upper())


def crawl_epitope_iedb_raw(
    zip_path: Union[str, Path],
    *,
    inner_csv: str = "tcell_full_v3.csv",
    min_len: int = 8,
    max_len: int = 30,
    clip_range: Tuple[float, float] = (-2.0, 6.0),
    drop_duplicates: bool = True,
    fill_env: bool = True,
    progress_every: int = 500_000,
) -> pd.DataFrame:
    """Extract epitope data from an IEDB T-cell assay ZIP file.

    Parameters
    ----------
    zip_path : str | Path
        Path to the IEDB ``tcell_full_v3.zip``.
    inner_csv : str
        CSV file name inside the ZIP archive.
    min_len, max_len : int
        Peptide length filter.
    clip_range : (float, float)
        Clip derived efficacy to this range.
    fill_env : bool
        Fill concentration / incubation_hours etc. with median defaults.
    drop_duplicates : bool
        Deduplicate by (sequence, efficacy).
    progress_every : int
        Print progress every N rows.

    Returns
    -------
    pd.DataFrame  columns include ``[sequence, efficacy]`` and optionally env_cols.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"IEDB ZIP not found: {zip_path}")

    records: List[Dict[str, float | str]] = []
    total = 0
    valid = 0
    has_eff = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_csv) as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            next(reader)  # skip header

            for row in reader:
                total += 1
                if total % progress_every == 0:
                    print(f"  IEDB raw: {total} rows, {len(records)} records collected ...")

                seq = row[_COL_EPITOPE_NAME].strip() if len(row) > _COL_EPITOPE_NAME else ""
                if not _is_valid_peptide(seq, min_len, max_len):
                    continue
                valid += 1

                efficacy: Optional[float] = None

                # Try qualitative
                qual = row[_COL_QUALITATIVE].strip() if len(row) > _COL_QUALITATIVE else ""
                if qual in _QUALITATIVE_MAP:
                    efficacy = _QUALITATIVE_MAP[qual]

                # Try quantitative
                quant_str = row[_COL_QUANTITATIVE].strip() if len(row) > _COL_QUANTITATIVE else ""
                quant_val = _parse_quantitative(quant_str)
                if quant_val is not None and quant_val > 0:
                    efficacy = -np.log10(max(quant_val, 1e-3) / 1e6)

                # Try response frequency as fallback
                if efficacy is None:
                    freq_str = row[_COL_RESPONSE_FREQ].strip() if len(row) > _COL_RESPONSE_FREQ else ""
                    if freq_str:
                        try:
                            rf = float(freq_str)
                            efficacy = (rf / 100.0) * 2.0 - 1.0
                        except ValueError:
                            pass

                if efficacy is None:
                    continue

                has_eff += 1
                rec: Dict[str, float | str] = {
                    "sequence": seq.upper(),
                    "efficacy": round(efficacy, 6),
                }
                if fill_env:
                    for k, v in _MEDIAN_ENV.items():
                        rec[k] = v
                records.append(rec)

    print(f"  IEDB raw: total={total}, valid_peptides={valid}, with_efficacy={has_eff}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["sequence", "efficacy"])
        print(f"  After dedup: {len(df)} rows (removed {before - len(df)})")

    df["efficacy"] = df["efficacy"].clip(clip_range[0], clip_range[1])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified "crawl all" convenience
# ---------------------------------------------------------------------------

def crawl_all_epitope(
    *,
    # FASTA / CSV sources
    fasta_urls: Optional[Sequence[str]] = None,
    csv_urls: Optional[Sequence[str]] = None,
    sequence_col: Optional[str] = None,
    # IEDB raw
    iedb_zip_path: Optional[Union[str, Path]] = None,
    iedb_inner_csv: str = "tcell_full_v3.csv",
    # General
    cache_dir: Union[str, Path] = "data/cache/epitope",
    min_len: int = 8,
    max_len: int = 25,
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "epitope-crawler/1.0 (research; contact: local)",
) -> pd.DataFrame:
    """Run all epitope data sources and concatenate.

    Parameters
    ----------
    fasta_urls : list[str], optional
        FASTA URLs / UniProt refs / PDB refs / local FASTA paths.
    csv_urls : list[str], optional
        CSV/TSV/Excel URLs or local paths containing epitope data.
    iedb_zip_path : str | Path, optional
        Path to IEDB T-cell ZIP for raw extraction.
    sequence_col : str, optional
        Name of the sequence column in CSV sources.

    Returns
    -------
    pd.DataFrame  deduplicated epitope dataset.
    """
    frames: List[pd.DataFrame] = []

    # 1. FASTA sources
    if fasta_urls:
        fasta_df = crawl_epitope_fasta_sources(
            list(fasta_urls),
            cache_dir=cache_dir,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            user_agent=user_agent,
            min_len=min_len,
            max_len=max_len,
        )
        if not fasta_df.empty:
            frames.append(fasta_df)

    # 2. CSV sources
    if csv_urls:
        csv_df = crawl_epitope_csv_datasets(
            list(csv_urls),
            cache_dir=cache_dir,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            user_agent=user_agent,
            sequence_col=sequence_col,
            min_len=min_len,
            max_len=max_len,
        )
        if not csv_df.empty:
            frames.append(csv_df)

    # 3. IEDB raw extraction
    if iedb_zip_path is not None:
        iedb_df = crawl_epitope_iedb_raw(
            iedb_zip_path,
            inner_csv=iedb_inner_csv,
            min_len=min_len,
            max_len=max_len,
        )
        if not iedb_df.empty:
            frames.append(iedb_df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if "sequence" in merged.columns:
        merged = merged.dropna(subset=["sequence"])
        merged = merged.drop_duplicates(subset=["sequence"])
    return merged.reset_index(drop=True)
