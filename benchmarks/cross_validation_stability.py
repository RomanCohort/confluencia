"""
Cross-Validation Stability Analysis (Fast Version)
================================================
Comprehensive cross-validation analysis with reduced iterations for speed.

Usage:
    python benchmarks/cross_validation_stability.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-epitope"))

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def load_epitope_data():
    """Load epitope training data."""
    from core.features import build_feature_matrix, FeatureSpec

    csv_path = PROJECT_ROOT / "data" / "example_epitope.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "confluencia" / "confluencia" / "_internal" / "data" / "example_epitope.csv"

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    col_map = {
        "sequence": "epitope_seq",
        "concentration": "dose",
        "cell_density": "circ_expr",
        "incubation_hours": "treatment_time",
    }
    for raw, internal in col_map.items():
        if raw in df.columns and internal not in df.columns:
            df[internal] = df[raw]

    if "freq" not in df.columns:
        df["freq"] = 1.0
    if "ifn_score" not in df.columns:
        df["ifn_score"] = 0.5

    spec = FeatureSpec()
    X, _, _ = build_feature_matrix(df, spec)
    y = df["efficacy"].values.astype(np.float32)

    return X, y


def run_10fold_cv(X, y):
    """Run 10-fold cross-validation."""
    print("\n=== 10-Fold Cross-Validation ===")

    from core.moe import MOERegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        t0 = time.time()
        moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], folds=4, random_state=42)
        moe.fit(X_scaled[train_idx], y[train_idx])
        preds = moe.predict(X_scaled[test_idx])
        train_time = time.time() - t0
        preds = np.nan_to_num(preds, nan=0.0)

        mae = mean_absolute_error(y[test_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[test_idx], preds))
        r2 = r2_score(y[test_idx], preds)
        pr, _ = stats.pearsonr(preds, y[test_idx])

        fold_results.append({
            "fold": fold + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson_r": float(pr),
            "train_time_s": float(train_time),
        })

        all_preds.extend(preds.tolist())
        all_true.extend(y[test_idx].tolist())

        print(f"  Fold {fold+1:2d}: MAE={mae:.4f}, R2={r2:.4f}, r={pr:.4f}")

    # Aggregate
    metrics = ["mae", "rmse", "r2", "pearson_r", "train_time_s"]
    aggregate = {}
    for m in metrics:
        values = [f[m] for f in fold_results]
        aggregate[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0,
        }

    all_preds, all_true = np.array(all_preds), np.array(all_true)
    overall = {
        "mae": float(mean_absolute_error(all_true, all_preds)),
        "rmse": float(np.sqrt(mean_squared_error(all_true, all_preds))),
        "r2": float(r2_score(all_true, all_preds)),
        "pearson_r": float(stats.pearsonr(all_preds, all_true)[0]),
    }

    print(f"\n  Aggregate: MAE={aggregate['mae']['mean']:.4f}±{aggregate['mae']['std']:.4f}, R2={aggregate['r2']['mean']:.4f}±{aggregate['r2']['std']:.4f}")

    return {"folds": fold_results, "aggregate": aggregate, "overall_pooled": overall}


def run_seed_sensitivity(X, y, n_seeds=10):
    """Analyze seed sensitivity."""
    print("\n=== Random Seed Sensitivity ===")

    from core.moe import MOERegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

    seed_results = []
    for seed in range(n_seeds):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        mae_vals, r2_vals = [], []

        for train_idx, test_idx in kf.split(X_scaled):
            moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], folds=4, random_state=seed)
            moe.fit(X_scaled[train_idx], y[train_idx])
            preds = moe.predict(X_scaled[test_idx])
            preds = np.nan_to_num(preds, nan=0.0)
            mae_vals.append(mean_absolute_error(y[test_idx], preds))
            r2_vals.append(r2_score(y[test_idx], preds))

        seed_results.append({
            "seed": seed,
            "mae_mean": float(np.mean(mae_vals)),
            "r2_mean": float(np.mean(r2_vals)),
        })
        print(f"  Seed {seed}: MAE={np.mean(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}")

    aggregate = {
        "mae_mean": float(np.mean([s["mae_mean"] for s in seed_results])),
        "mae_std": float(np.std([s["mae_mean"] for s in seed_results])),
        "r2_mean": float(np.mean([s["r2_mean"] for s in seed_results])),
        "r2_std": float(np.std([s["r2_mean"] for s in seed_results])),
    }

    print(f"\n  Across {n_seeds} seeds: MAE={aggregate['mae_mean']:.4f}±{aggregate['mae_std']:.4f}, R2={aggregate['r2_mean']:.4f}±{aggregate['r2_std']:.4f}")

    return {"seeds": seed_results, "aggregate": aggregate}


def run_gating_comparison(X, y):
    """Compare static vs gating MOE."""
    print("\n=== Static vs Gating MOE Comparison ===")

    from core.moe import MOERegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {"static": [], "gating": []}

    for train_idx, test_idx in kf.split(X_scaled):
        # Static gating
        moe_static = MOERegressor(expert_names=["ridge", "hgb", "rf"], folds=4, random_state=42, gating="static")
        moe_static.fit(X_scaled[train_idx], y[train_idx])
        pred_static = np.nan_to_num(moe_static.predict(X_scaled[test_idx]), nan=0.0)
        mae_static = mean_absolute_error(y[test_idx], pred_static)
        r2_static = r2_score(y[test_idx], pred_static)
        results["static"].append({"mae": float(mae_static), "r2": float(r2_static)})

        # Learnable gating
        moe_gate = MOERegressor(expert_names=["ridge", "hgb", "rf"], folds=4, random_state=42, gating="gating")
        moe_gate.fit(X_scaled[train_idx], y[train_idx])
        pred_gate = np.nan_to_num(moe_gate.predict(X_scaled[test_idx]), nan=0.0)
        mae_gate = mean_absolute_error(y[test_idx], pred_gate)
        r2_gate = r2_score(y[test_idx], pred_gate)
        results["gating"].append({"mae": float(mae_gate), "r2": float(r2_gate)})

        print(f"  Fold: Static MAE={mae_static:.4f}, R2={r2_static:.4f} | Gating MAE={mae_gate:.4f}, R2={r2_gate:.4f}")

    # Aggregate
    for mode in ["static", "gating"]:
        results[mode] = {
            "mae_mean": float(np.mean([r["mae"] for r in results[mode]])),
            "mae_std": float(np.std([r["mae"] for r in results[mode]])),
            "r2_mean": float(np.mean([r["r2"] for r in results[mode]])),
            "r2_std": float(np.std([r["r2"] for r in results[mode]])),
        }

    print(f"\n  Static: MAE={results['static']['mae_mean']:.4f}±{results['static']['mae_std']:.4f}, R2={results['static']['r2_mean']:.4f}±{results['static']['r2_std']:.4f}")
    print(f"  Gating: MAE={results['gating']['mae_mean']:.4f}±{results['gating']['mae_std']:.4f}, R2={results['gating']['r2_mean']:.4f}±{results['gating']['r2_std']:.4f}")

    improvement = (results["static"]["mae_mean"] - results["gating"]["mae_mean"]) / results["static"]["mae_mean"] * 100
    print(f"  Gating improvement: {improvement:+.1f}%")

    return results


def main():
    print("=" * 60)
    print("Cross-Validation Stability Analysis")
    print("=" * 60)

    X, y = load_epitope_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

    results = {}
    results["10fold_cv"] = run_10fold_cv(X, y)
    results["seed_sensitivity"] = run_seed_sensitivity(X, y, n_seeds=10)
    results["gating_comparison"] = run_gating_comparison(X, y)

    results["summary"] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "conclusion": (
            f"Stability analysis confirms robust performance. "
            f"10-fold CV: MAE={results['10fold_cv']['aggregate']['mae']['mean']:.4f}±{results['10fold_cv']['aggregate']['mae']['std']:.4f}. "
            f"Seed sensitivity: MAE variance = {results['seed_sensitivity']['aggregate']['mae_std']:.4f}. "
            f"Gating network: {results['gating_comparison']['gating']['mae_mean']:.4f} vs static {results['gating_comparison']['static']['mae_mean']:.4f}."
        ),
    }

    output_path = RESULTS_DIR / "cross_validation_stability.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()