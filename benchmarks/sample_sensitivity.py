"""
Confluencia Sample Size Sensitivity (Learning Curve) Experiment
================================================================
Measure model performance as training data size increases.

Usage:
    python -m benchmarks.sample_sensitivity --module epitope --data data/example_epitope.csv
    python -m benchmarks.sample_sensitivity --module drug --data data/example_drug.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EPITOPE_DIR = _PROJECT_ROOT / "confluencia-2.0-epitope"
_DRUG_DIR = _PROJECT_ROOT / "confluencia-2.0-drug"

# Default sample sizes to test
SAMPLE_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_SAMPLES = 15  # absolute minimum


def _ensure_path(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)


_EPITOPE_COL_MAP = {
    "sequence": "epitope_seq",
    "concentration": "dose",
    "cell_density": "circ_expr",
    "incubation_hours": "treatment_time",
}


def _normalise_epitope_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw, internal in _EPITOPE_COL_MAP.items():
        if raw in out.columns and internal not in out.columns:
            out[internal] = out[raw]
    return out


def _build_features(df: pd.DataFrame, module: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build features for the given module."""
    if module == "epitope":
        _ensure_path(str(_EPITOPE_DIR))
        from core.features import FeatureSpec, build_feature_matrix, ensure_columns
        work = ensure_columns(_normalise_epitope_columns(df))
        X, _, _ = build_feature_matrix(work, FeatureSpec())
    else:
        _ensure_path(str(_DRUG_DIR))
        from core.features import build_feature_matrix, MixedFeatureSpec
        spec = MixedFeatureSpec(prefer_rdkit=False)
        X, _, _ = build_feature_matrix(df, spec)

    y = df["efficacy"].to_numpy(dtype=np.float32) if "efficacy" in df.columns else None
    return X, y


def _evaluate_at_size(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    model_name: str = "hgb",
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate model with a fixed training set size using held-out test."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train_actual = min(n_train, int(0.8 * n))
    n_test = min(max(10, n - n_train_actual), n - n_train_actual)

    train_idx = indices[:n_train_actual]
    test_idx = indices[n_train_actual:n_train_actual + n_test]

    if len(train_idx) < 5 or len(test_idx) < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    if model_name == "hgb":
        model = HistGradientBoostingRegressor(random_state=seed)
    elif model_name == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    else:
        model = HistGradientBoostingRegressor(random_state=seed)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    pred = model.predict(X_te)
    return {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "train_time": float(elapsed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def run_sensitivity_experiment(
    module: str,
    data_path: str,
    model_name: str = "hgb",
    fractions: List[float] = None,
    n_repeats: int = 5,
    seed: int = 42,
    output_dir: str = "benchmarks/results",
) -> str:
    """Run learning curve experiment: vary sample size and measure performance."""
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / data_path)

    X, y = _build_features(df, module)
    if y is None:
        raise ValueError("Data must have 'efficacy' column.")

    n_total = len(y)
    fractions = fractions or SAMPLE_FRACTIONS

    results: Dict[str, Any] = {
        "module": module,
        "n_total": n_total,
        "model": model_name,
        "curve": [],
    }

    for frac in fractions:
        n_train = max(MIN_SAMPLES, int(frac * n_total * 0.8))
        if n_train > n_total * 0.8:
            n_train = int(n_total * 0.8)

        point = {"fraction": frac, "n_train": n_train, "repeats": []}

        for rep in range(n_repeats):
            metrics = _evaluate_at_size(X, y, n_train, model_name, seed=seed + rep)
            point["repeats"].append(metrics)

        # Aggregate
        mae_vals = [r["mae"] for r in point["repeats"] if not np.isnan(r.get("mae", float("nan")))]
        rmse_vals = [r["rmse"] for r in point["repeats"] if not np.isnan(r.get("rmse", float("nan")))]
        r2_vals = [r["r2"] for r in point["repeats"] if not np.isnan(r.get("r2", float("nan")))]

        point["mae_mean"] = float(np.mean(mae_vals)) if mae_vals else float("nan")
        point["mae_std"] = float(np.std(mae_vals)) if mae_vals else float("nan")
        point["rmse_mean"] = float(np.mean(rmse_vals)) if rmse_vals else float("nan")
        point["r2_mean"] = float(np.mean(r2_vals)) if r2_vals else float("nan")
        point["r2_std"] = float(np.std(r2_vals)) if r2_vals else float("nan")

        print(
            f"frac={frac:.2f} n_train={n_train:4d} "
            f"MAE={point['mae_mean']:.4f}±{point['mae_std']:.4f} "
            f"R2={point['r2_mean']:.4f}+/-{point['r2_std']:.4f}"
        )
        results["curve"].append(point)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_sensitivity_{module}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Confluencia Sample Size Sensitivity")
    parser.add_argument("--module", choices=["epitope", "drug"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    run_sensitivity_experiment(
        args.module, args.data, args.model, n_repeats=args.repeats,
        seed=args.seed, output_dir=args.output,
    )


if __name__ == "__main__":
    main()
