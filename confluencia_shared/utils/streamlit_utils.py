"""
Shared Streamlit utilities for Confluencia apps.

Common Streamlit helper functions used across drug and epitope modules.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

import numpy as np
import pandas as pd
import streamlit as st

try:
    from importlib.metadata import version as importlib_metadata_version
except ImportError:
    from importlib_metadata import version as importlib_metadata_version  # type: ignore


# =============================================================================
# File I/O Utilities
# =============================================================================

def read_uploaded_csv(uploaded_file: "UploadedFile") -> pd.DataFrame:
    """Read uploaded CSV with multi-encoding fallback.

    Tries utf-8, utf-8-sig, gbk, gb18030 encodings in order.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Parsed DataFrame.

    Raises:
        Exception: If all encoding attempts fail.
    """
    raw = uploaded_file.getvalue()
    last_err: Exception | None = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(io.BytesIO(raw))


def read_uploaded_file(uploaded_file: "UploadedFile") -> pd.DataFrame:
    """Read uploaded CSV or Parquet file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Parsed DataFrame.
    """
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(uploaded_file.getvalue()))
    return read_uploaded_csv(uploaded_file)


# =============================================================================
# Data Validation Utilities
# =============================================================================

def missing_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return list of missing required column names.

    Args:
        df: DataFrame to check.
        required: List of required column names.

    Returns:
        List of column names that are missing from df.
    """
    return [c for c in required if c not in df.columns]


def core_ready_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    """Compute ratio of required columns present in DataFrame.

    Args:
        df: DataFrame to check.
        cols: List of required column names.

    Returns:
        Ratio of columns present (0.0 to 1.0). Returns 1.0 if cols is empty.
    """
    if not cols:
        return 1.0
    hit = sum(1 for c in cols if c in df.columns)
    return float(hit) / float(len(cols))


@st.cache_data(show_spinner=False)
def data_quality_report(df: pd.DataFrame, validate_smiles: bool = False) -> pd.DataFrame:
    """Build data quality report with missing rate per column.

    Args:
        df: DataFrame to analyze.
        validate_smiles: If True, validate SMILES strings and report invalid ratio.

    Returns:
        DataFrame with column names and missing rates.
    """
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        miss = float(df[c].isna().sum()) / float(n)
        rows.append({"列名": c, "缺失率": miss})

    if validate_smiles and "smiles" in df.columns:
        smiles = df["smiles"].astype(str).tolist()
        try:
            from rdkit import Chem  # type: ignore
            mol_from_smiles = getattr(Chem, "MolFromSmiles", None)
            if mol_from_smiles is None:
                raise RuntimeError("MolFromSmiles unavailable")

            bad = 0
            for s in smiles:
                if mol_from_smiles(str(s)) is None:
                    bad += 1
            invalid_ratio = float(bad) / float(max(len(smiles), 1))
        except Exception:
            # Fallback heuristic when RDKit is unavailable.
            bad = 0
            for s in smiles:
                t = str(s).strip()
                if (not t) or (" " in t) or ("%" in t):
                    bad += 1
            invalid_ratio = float(bad) / float(max(len(smiles), 1))
        rows.append({"列名": "smiles_invalid_ratio", "缺失率": invalid_ratio})

    return pd.DataFrame(rows)


# =============================================================================
# Experiment Logging Utilities
# =============================================================================

def append_experiment_log(module: str, config: dict, metrics: dict, base_dir: Path | None = None) -> None:
    """Append experiment run metadata to logs/experiments.csv.

    Args:
        module: Module name (e.g., "drug-main", "epitope-main").
        config: Configuration dictionary.
        metrics: Metrics dictionary.
        base_dir: Base directory for logs. If None, uses parent of calling module.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
    }
    csv_path = base_dir / "experiments.csv"
    line_df = pd.DataFrame([row])
    if csv_path.exists():
        line_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        line_df.to_csv(csv_path, index=False)


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of DataFrame CSV representation.

    Args:
        df: DataFrame to hash.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    csv_text = df.to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def snapshot_env_deps(extra_packages: list[str] | None = None) -> dict[str, str]:
    """Capture Python and key package versions.

    Args:
        extra_packages: Additional packages to check. Defaults to standard ML packages.

    Returns:
        Dictionary mapping package names to versions.
    """
    packages = ["python", "numpy", "pandas", "scikit-learn", "streamlit", "torch"]
    if extra_packages:
        packages = list(dict.fromkeys(packages + extra_packages))

    deps: dict[str, str] = {}
    for pkg in packages:
        if pkg == "python":
            deps[pkg] = platform.python_version()
            continue
        try:
            deps[pkg] = importlib_metadata_version(pkg)
        except Exception:
            deps[pkg] = "not-installed"
    return deps


def save_repro_bundle(
    module: str,
    data_df: pd.DataFrame,
    config: dict,
    metrics: dict,
    base_dir: Path | None = None,
    extra_packages: list[str] | None = None,
) -> str:
    """Save full reproducibility bundle (CSV + MD report) to logs/reproduce/.

    Args:
        module: Module name.
        data_df: Input DataFrame.
        config: Configuration dictionary.
        metrics: Metrics dictionary.
        base_dir: Base directory for logs. If None, uses parent of calling module.
        extra_packages: Additional packages to snapshot.

    Returns:
        Run ID string.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "logs" / "reproduce"
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_hash = hash_dataframe(data_df)
    run_id = f"{module}_{ts}_{data_hash[:8]}"
    env_deps = snapshot_env_deps(extra_packages)

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "rows": int(len(data_df)),
        "data_sha256": data_hash,
        "python_executable": sys.executable,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "env_deps": json.dumps(env_deps, ensure_ascii=False),
    }

    csv_path = base_dir / "runs.csv"
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)

    md_path = base_dir / f"{run_id}.md"
    md_lines = [
        f"# Repro Report ({module})",
        f"- run_id: {run_id}",
        f"- timestamp: {row['timestamp']}",
        f"- rows: {row['rows']}",
        f"- data_sha256: {data_hash}",
        "",
        "## Config",
        f"```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        f"```",
        "",
        "## Metrics",
        f"```json",
        json.dumps(metrics, indent=2, ensure_ascii=False),
        f"```",
        "",
        "## Environment",
        f"```json",
        json.dumps(env_deps, indent=2, ensure_ascii=False),
        f"```",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return run_id


# =============================================================================
# Module-specific column alias maps
# =============================================================================

EPITOPE_ALIAS_MAP: dict[str, list[str]] = {
    "epitope_seq": ["epitope", "sequence", "peptide_seq"],
    "dose": ["dosage", "dose_mg"],
    "freq": ["frequency", "times", "dose_freq"],
    "treatment_time": ["time", "duration_h", "treatment_hours"],
    "circ_expr": ["circRNA_expr", "circ_expression", "circ_score"],
    "ifn_score": ["ifn", "ifn_gamma", "ifn_g_score"],
    "efficacy": ["label", "target", "y"],
}

DRUG_ALIAS_MAP: dict[str, list[str]] = {
    "smiles": ["SMILES", "canonical_smiles", "mol_smiles"],
    "epitope_seq": ["epitope", "sequence", "peptide_seq"],
    "dose": ["dosage", "dose_mg"],
    "freq": ["frequency", "times", "dose_freq"],
    "treatment_time": ["time", "duration_h", "treatment_hours"],
    "efficacy": ["label", "target", "y"],
}


def apply_column_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    """Rename common column aliases to canonical names.

    Args:
        df: DataFrame with potentially aliased column names.
        alias_map: Dictionary mapping canonical names to lists of aliases.

    Returns:
        DataFrame with canonical column names.
    """
    out = df.copy()
    for canonical, aliases in alias_map.items():
        if canonical in out.columns:
            continue
        for a in aliases:
            if a in out.columns:
                out = out.rename(columns={a: canonical})
                break
    return out


# =============================================================================
# Metric Utilities
# =============================================================================

def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R2 with safety checks.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, RMSE, R2 metrics.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    return {
        "mae": float(mean_absolute_error(y_true_clean, y_pred_clean)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
        "r2": float(r2_score(y_true_clean, y_pred_clean)),
    }


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    """Compute mean, std, and 95% CI for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Tuple of (mean, std, ci_95).
    """
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) == 1 else 0.0, 0.0, 0.0

    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / np.sqrt(len(arr))
    return mean, std, float(ci)


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # File I/O
    "read_uploaded_csv",
    "read_uploaded_file",
    # Data Validation
    "missing_required_columns",
    "core_ready_ratio",
    "data_quality_report",
    # Experiment Logging
    "append_experiment_log",
    "hash_dataframe",
    "snapshot_env_deps",
    "save_repro_bundle",
    # Column Aliases
    "EPITOPE_ALIAS_MAP",
    "DRUG_ALIAS_MAP",
    "apply_column_aliases",
    # Metrics
    "safe_metrics",
    "mean_std_ci",
]
