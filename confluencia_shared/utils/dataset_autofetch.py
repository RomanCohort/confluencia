from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from confluencia_shared.utils.dataset_fetch import download_to_cache, is_http_url, read_table_any

# Heuristic column vocab for quick training-ready checks
_INPUT_COLS = {
    "drug": ["smiles", "canonical_smiles", "isomeric_smiles", "ligand_smiles", "compound_smiles", "sequence"],
    "epitope": ["sequence", "peptide", "aa", "amino"],
    "docking": ["ligand_smiles", "smiles", "protein_sequence", "protein", "pdb", "pdb_id", "binding_score"],
    "multiscale": ["smiles", "target", "D", "Vmax", "Km"],
    "custom": ["smiles", "sequence", "input"],
}
_TARGET_COLS = {
    "drug": ["activity", "activity_score", "label", "y", "response", "affinity", "ic50", "ec50"],
    "epitope": ["label", "binder", "immunogenic", "positivity", "target"],
    "docking": ["binding_score", "affinity", "score", "y"],
    "multiscale": ["target", "y", "D", "Vmax", "Km"],
    "custom": ["label", "target", "y"],
}


@dataclass(frozen=True)
class FetchedDataset:
    url: str
    path: str
    n_rows: int
    n_cols: int
    columns: List[str]
    training_ready: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "path": self.path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": ", ".join(self.columns),
            "training_ready": self.training_ready,
            "reason": self.reason,
        }


def _has_overlap(cols: Sequence[str], vocab: Sequence[str]) -> bool:
    cols_l = [c.lower() for c in cols]
    vocab_l = [v.lower() for v in vocab]
    return any(c in cols_l for c in vocab_l)


def _detect_training_ready(cols: Sequence[str], domain: str) -> tuple[bool, str]:
    domain_key = domain if domain in _INPUT_COLS else "custom"
    has_inp = _has_overlap(cols, _INPUT_COLS.get(domain_key, []))
    has_tgt = _has_overlap(cols, _TARGET_COLS.get(domain_key, []))
    if has_inp and has_tgt:
        return True, "found input & target columns"
    if has_inp:
        return False, "found input columns, missing target"
    if has_tgt:
        return False, "found target columns, missing input"
    return False, "no known input/target columns"


def fetch_datasets(
    *,
    urls: Sequence[str],
    domain: str = "custom",
    cache_dir: str | Path = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.0,
    max_rows_per_file: int = 5000,
) -> Dict[str, Any]:
    summaries: List[FetchedDataset] = []
    frames: List[pd.DataFrame] = []

    for raw_url in urls:
        url = str(raw_url).strip()
        if not url:
            continue
        try:
            if is_http_url(url):
                art = download_to_cache(url, cache_dir=cache_dir, timeout=timeout, sleep_seconds=sleep_seconds)
                path = art.path
            else:
                path = Path(url)
                if not path.exists():
                    summaries.append(FetchedDataset(url=url, path=str(path), n_rows=0, n_cols=0, columns=[], training_ready=False, reason="file not found"))
                    continue

            ext = path.suffix.lower()
            if ext not in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
                summaries.append(FetchedDataset(url=url, path=str(path), n_rows=0, n_cols=0, columns=[], training_ready=False, reason=f"skip (ext {ext})"))
                continue

            df = read_table_any(str(path))
            if len(df) > max_rows_per_file:
                df = df.head(max_rows_per_file).copy()

            cols = list(df.columns.astype(str))
            ready, reason = _detect_training_ready(cols, domain)
            summaries.append(
                FetchedDataset(
                    url=url,
                    path=str(path),
                    n_rows=int(len(df)),
                    n_cols=int(len(cols)),
                    columns=cols,
                    training_ready=ready,
                    reason=reason,
                )
            )
            frames.append(df)
        except Exception as exc:
            summaries.append(FetchedDataset(url=url, path=str(url), n_rows=0, n_cols=0, columns=[], training_ready=False, reason=str(exc)))
            continue

    out: Dict[str, Any] = {
        "items": [s.to_dict() for s in summaries],
    }

    if frames:
        concat_df = pd.concat(frames, axis=0, ignore_index=True)
        buf = io.BytesIO()
        concat_df.to_csv(buf, index=False)
        out["result_csv_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        out["n_rows"] = int(len(concat_df))
        out["n_cols"] = int(len(concat_df.columns))
    else:
        out["n_rows"] = 0
        out["n_cols"] = 0

    return out
