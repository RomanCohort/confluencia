"""
Confluencia Ablation Study Framework
=====================================
Systematically removes components to quantify each contribution.

Usage:
    python -m benchmarks.ablation --module epitope --data data/example_epitope.csv
    python -m benchmarks.ablation --module drug --data data/example_drug.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EPITOPE_DIR = _PROJECT_ROOT / "confluencia-2.0-epitope"
_DRUG_DIR = _PROJECT_ROOT / "confluencia-2.0-drug"


def _ensure_epitope_path():
    p = str(_EPITOPE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_drug_path():
    p = str(_DRUG_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# Column name mapping: raw CSV -> internal names
_EPITOPE_COL_MAP = {
    "sequence": "epitope_seq",
    "concentration": "dose",
    "cell_density": "circ_expr",
    "incubation_hours": "treatment_time",
}


def _normalise_epitope_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw CSV column names to internal feature names."""
    out = df.copy()
    for raw, internal in _EPITOPE_COL_MAP.items():
        if raw in out.columns and internal not in out.columns:
            out[internal] = out[raw]
    if "epitope_seq" not in out.columns and "sequence" in out.columns:
        out["epitope_seq"] = out["sequence"]
    return out


# ---------------------------------------------------------------------------
# Epitope feature builders (with ablation toggles)
# ---------------------------------------------------------------------------


@dataclass
class EpitopeAblationConfig:
    """Toggle each feature component on/off for ablation."""
    use_mamba_summary: bool = True
    use_mamba_local: bool = True
    use_mamba_meso: bool = True
    use_mamba_global: bool = True
    use_kmer2: bool = True
    use_kmer3: bool = True
    use_biochem: bool = True
    use_env: bool = True
    # model choice
    model_name: str = "hgb"  # hgb, ridge, gbr
    kmer_dim: int = 64
    d_model: int = 24


def _build_ablation_features_epitope(
    df: pd.DataFrame,
    cfg: EpitopeAblationConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix with selective component inclusion."""
    _ensure_epitope_path()
    from core.features import (
        FeatureSpec, Mamba3Config, Mamba3LiteEncoder,
        _hash_kmer, _biochem_stats, _clean_seq, ensure_columns,
    )
    from core.features import AA_TO_IDX

    mamba_cfg = Mamba3Config(d_model=cfg.d_model)
    encoder = Mamba3LiteEncoder(mamba_cfg)
    work = ensure_columns(_normalise_epitope_columns(df))

    xs: List[np.ndarray] = []
    all_names: List[str] = []

    for idx, row in work.iterrows():
        seq = str(row.get("epitope_seq", ""))
        parts: List[np.ndarray] = []
        names: List[str] = []

        m = encoder.encode(seq)

        if cfg.use_mamba_summary:
            parts.append(m["summary"])
            d = cfg.d_model
            names += [f"summary_mean_{i}" for i in range(d)]
            names += [f"summary_max_{i}" for i in range(d)]
            names += [f"summary_last_{i}" for i in range(d)]
            names += [f"summary_mix_{i}" for i in range(d)]

        if cfg.use_mamba_local:
            parts.append(m["local_pool"])
            names += [f"nb_local_{i}" for i in range(cfg.d_model)]

        if cfg.use_mamba_meso:
            parts.append(m["meso_pool"])
            names += [f"nb_meso_{i}" for i in range(cfg.d_model)]

        if cfg.use_mamba_global:
            parts.append(m["global_pool"])
            names += [f"nb_global_{i}" for i in range(cfg.d_model)]

        if cfg.use_kmer2:
            parts.append(_hash_kmer(seq, k=2, dim=cfg.kmer_dim))
            names += [f"kmer2_{i}" for i in range(cfg.kmer_dim)]

        if cfg.use_kmer3:
            parts.append(_hash_kmer(seq, k=3, dim=cfg.kmer_dim))
            names += [f"kmer3_{i}" for i in range(cfg.kmer_dim)]

        if cfg.use_biochem:
            parts.append(_biochem_stats(seq))
            names += [
                "bio_length", "bio_hydrophobic_frac", "bio_polar_frac",
                "bio_acidic_frac", "bio_basic_frac", "bio_entropy",
                "bio_n_hydrophobic", "bio_c_hydrophobic",
                "bio_proline_frac", "bio_glycine_frac", "bio_aromatic_frac",
                "bio_basic2_frac", "bio_acidic2_frac", "bio_amide_frac",
                "bio_unique_residue_ratio", "bio_unknown_ratio",
            ]

        if cfg.use_env:
            env_cols = [c for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
                        if c in work.columns]
            env = np.array(
                [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in env_cols],
                dtype=np.float32,
            )
            parts.append(env)
            names += [f"env_{c}" for c in env_cols]

        if parts:
            xs.append(np.concatenate(parts).astype(np.float32))
        else:
            xs.append(np.zeros((1,), dtype=np.float32))
            names = ["dummy"]

        if idx == 0:
            all_names = list(names)

    return np.stack(xs, axis=0), all_names


# ---------------------------------------------------------------------------
# Drug feature builders (with ablation toggles)
# ---------------------------------------------------------------------------


@dataclass
class DrugAblationConfig:
    """Toggle drug feature components."""
    use_morgan_fp: bool = True
    use_descriptors: bool = True
    use_context: bool = True
    use_epitope_features: bool = True
    model_name: str = "hgb"


def _build_ablation_features_drug(
    df: pd.DataFrame,
    cfg: DrugAblationConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Build drug feature matrix with selective component inclusion."""
    _ensure_drug_path()
    from core.features import build_feature_matrix, build_feature_names, MixedFeatureSpec, ensure_columns

    work = ensure_columns(df)
    spec = MixedFeatureSpec(prefer_rdkit=True)  # Use RDKit Morgan FP

    X, env_cols, backend = build_feature_matrix(work, spec)
    names = build_feature_names(spec, env_cols)

    # Ensure names length matches X columns
    if len(names) != X.shape[1]:
        # Generate fallback names
        names = [f"feat_{i}" for i in range(X.shape[1])]

    # Identify column groups by prefix for ablation masking
    mask = np.ones(X.shape[1], dtype=bool)

    if not cfg.use_morgan_fp:
        fp_mask = np.array([n.startswith("smiles_hash_") or n.startswith("smiles_morgan_") for n in names])
        mask[fp_mask] = False
    if not cfg.use_descriptors:
        desc_mask = np.array([n.startswith("smiles_desc_") for n in names])
        mask[desc_mask] = False
    if not cfg.use_context:
        ctx_mask = np.array([n.startswith("ctx_") or n in ("dose", "freq", "treatment_time") for n in names])
        mask[ctx_mask] = False
    if not cfg.use_epitope_features:
        epi_mask = np.array([n.startswith("epitope_") or n.startswith("epi_") for n in names])
        mask[epi_mask] = False

    return X[:, mask], [n for n, m in zip(names, mask) if m]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _make_model(name: str, seed: int = 42):
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Sequence-aware splitting
# ---------------------------------------------------------------------------


def sequence_aware_split(
    df: pd.DataFrame,
    seq_col: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by unique sequences: no sequence appears in both train and test."""
    unique_seqs = df[seq_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)
    n_test = max(1, int(len(unique_seqs) * test_ratio))
    test_seqs = set(unique_seqs[:n_test])
    test_idx = np.array([i for i, s in enumerate(df[seq_col]) if s in test_seqs])
    train_idx = np.array([i for i, s in enumerate(df[seq_col]) if s not in test_seqs])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_kfold(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "hgb",
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run repeated k-fold cross-validation and return aggregate metrics."""
    results = {
        "mae": [], "rmse": [], "r2": [],
        "train_time": [],
    }

    for rep in range(n_repeats):
        kf = KFold(n_splits=min(n_folds, max(2, len(y) // 5)),
                    shuffle=True, random_state=seed + rep)
        for tr_idx, va_idx in kf.split(X):
            model = _make_model(model_name, seed=seed + rep)
            t0 = time.time()
            model.fit(X[tr_idx], y[tr_idx])
            elapsed = time.time() - t0

            pred = model.predict(X[va_idx])
            results["mae"].append(mean_absolute_error(y[va_idx], pred))
            results["rmse"].append(np.sqrt(mean_squared_error(y[va_idx], pred)))
            results["r2"].append(r2_score(y[va_idx], pred))
            results["train_time"].append(elapsed)

    def _stats(vals):
        a = np.array(vals)
        return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    return {k: _stats(v) for k, v in results.items()}


def evaluate_sequence_split(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    seq_col: str,
    model_name: str = "hgb",
    seed: int = 42,
) -> Dict[str, Any]:
    """Single train/test split by sequence, report metrics."""
    train_idx, test_idx = sequence_aware_split(df, seq_col, seed=seed)

    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"error": "Cannot split: too few unique sequences"}

    model = _make_model(model_name, seed)
    t0 = time.time()
    model.fit(X[train_idx], y[train_idx])
    elapsed = time.time() - t0

    pred = model.predict(X[test_idx])
    return {
        "mae": float(mean_absolute_error(y[test_idx], pred)),
        "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
        "r2": float(r2_score(y[test_idx], pred)),
        "train_time": float(elapsed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

EPITOPE_ABLATION_CARDS: Dict[str, Dict[str, bool]] = {
    "Full (all components)": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- Mamba summary": dict(
        use_mamba_summary=False, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- Mamba local pool": dict(
        use_mamba_summary=True, use_mamba_local=False, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- Mamba meso pool": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=False,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- Mamba global pool": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=False, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- k-mer (2)": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=False, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "- k-mer (3)": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=False,
        use_biochem=True, use_env=True,
    ),
    "- Biochem stats": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=False, use_env=True,
    ),
    "- Environment": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=False,
    ),
    "Only Mamba+env (no kmer/bio)": dict(
        use_mamba_summary=True, use_mamba_local=True, use_mamba_meso=True,
        use_mamba_global=True, use_kmer2=False, use_kmer3=False,
        use_biochem=False, use_env=True,
    ),
    "Only kmer+bio+env (no Mamba)": dict(
        use_mamba_summary=False, use_mamba_local=False, use_mamba_meso=False,
        use_mamba_global=False, use_kmer2=True, use_kmer3=True,
        use_biochem=True, use_env=True,
    ),
    "Only env (baseline)": dict(
        use_mamba_summary=False, use_mamba_local=False, use_mamba_meso=False,
        use_mamba_global=False, use_kmer2=False, use_kmer3=False,
        use_biochem=False, use_env=True,
    ),
}

DRUG_ABLATION_CARDS: Dict[str, Dict[str, bool]] = {
    "Full (all components)": dict(
        use_morgan_fp=True, use_descriptors=True, use_context=True,
        use_epitope_features=True,
    ),
    "- Morgan FP": dict(
        use_morgan_fp=False, use_descriptors=True, use_context=True,
        use_epitope_features=True,
    ),
    "- Descriptors": dict(
        use_morgan_fp=True, use_descriptors=False, use_context=True,
        use_epitope_features=True,
    ),
    "- Context (dose/freq/time)": dict(
        use_morgan_fp=True, use_descriptors=True, use_context=False,
        use_epitope_features=True,
    ),
    "- Epitope features": dict(
        use_morgan_fp=True, use_descriptors=True, use_context=True,
        use_epitope_features=False,
    ),
    "Only FP + context": dict(
        use_morgan_fp=True, use_descriptors=False, use_context=True,
        use_epitope_features=False,
    ),
    "Only context (baseline)": dict(
        use_morgan_fp=False, use_descriptors=False, use_context=True,
        use_epitope_features=False,
    ),
}


def run_epitope_ablation(
    data_path: str,
    model_name: str = "hgb",
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    split_mode: str = "sequence",  # "sequence" or "random"
    output_dir: str = "benchmarks/results",
) -> str:
    """Run full ablation study on epitope module."""
    project_root = Path(__file__).resolve().parents[1]
    data_path_resolved = project_root / data_path
    df = pd.read_csv(data_path_resolved)

    if "efficacy" not in df.columns:
        raise ValueError("Data must have 'efficacy' column for ablation study.")

    # Detect sequence column (raw data may use 'sequence' instead of 'epitope_seq')
    seq_col = None
    for col in ("epitope_seq", "sequence", "seq"):
        if col in df.columns:
            seq_col = col
            break

    y = df["efficacy"].to_numpy(dtype=np.float32)
    results: Dict[str, Any] = {}

    for card_name, toggles in EPITOPE_ABLATION_CARDS.items():
        cfg = EpitopeAblationConfig(
            model_name=model_name,
            **toggles,  # type: ignore[arg-type]
        )
        X, names = _build_ablation_features_epitope(df, cfg)
        print(f"[ablation] {card_name:40s} dim={X.shape[1]:4d}", end=" ... ", flush=True)

        if split_mode == "sequence" and seq_col is not None:
            metrics = evaluate_sequence_split(X, y, df, seq_col, model_name, seed)
        else:
            metrics = evaluate_kfold(X, y, model_name, n_folds, n_repeats, seed)

        results[card_name] = {"feature_dim": int(X.shape[1]), **metrics}
        if "mae" in metrics and isinstance(metrics["mae"], dict):
            print(f"MAE={metrics['mae']['mean']:.4f} R2={metrics['r2']['mean']:.4f}")
        elif "mae" in metrics:
            print(f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_epitope.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return str(out_path)


def run_drug_ablation(
    data_path: str,
    model_name: str = "hgb",
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    output_dir: str = "benchmarks/results",
) -> str:
    """Run full ablation study on drug module."""
    project_root = Path(__file__).resolve().parents[1]
    data_path_resolved = project_root / data_path
    df = pd.read_csv(data_path_resolved)

    if "efficacy" not in df.columns:
        raise ValueError("Data must have 'efficacy' column for ablation study.")

    y = df["efficacy"].to_numpy(dtype=np.float32)
    results: Dict[str, Any] = {}

    for card_name, toggles in DRUG_ABLATION_CARDS.items():
        cfg = DrugAblationConfig(model_name=model_name, **toggles)
        X, names = _build_ablation_features_drug(df, cfg)
        print(f"[ablation] {card_name:40s} dim={X.shape[1]:4d}", end=" ... ", flush=True)

        if "smiles" in df.columns:
            metrics = evaluate_sequence_split(X, y, df, "smiles", model_name, seed)
        else:
            metrics = evaluate_kfold(X, y, model_name, n_folds, n_repeats, seed)

        results[card_name] = {"feature_dim": int(X.shape[1]), **metrics}
        if "mae" in metrics and isinstance(metrics["mae"], dict):
            print(f"MAE={metrics['mae']['mean']:.4f} R2={metrics['r2']['mean']:.4f}")
        elif "mae" in metrics:
            print(f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_drug.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Confluencia Ablation Study")
    parser.add_argument("--module", choices=["epitope", "drug"], required=True)
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--model", default="hgb", help="Model backend (hgb, ridge, gbr)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["sequence", "random"], default="sequence")
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    if args.module == "epitope":
        run_epitope_ablation(
            args.data, args.model, args.folds, args.repeats, args.seed, args.split, args.output,
        )
    else:
        run_drug_ablation(
            args.data, args.model, args.folds, args.repeats, args.seed, args.output,
        )


if __name__ == "__main__":
    main()
