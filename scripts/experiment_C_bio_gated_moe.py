#!/usr/bin/env python3
"""
Confluencia Experiment C: BioGatedMOE vs MOERegressor

Tests whether bio-mimetic gating improves prediction over basic inverse-RMSE MOE.
Uses correct API: MOERegressor(expert_names, folds, config=ExpertConfig())

Usage: python experiment_C_bio_gated_moe.py
Output: benchmarks/results/experiment_C_bio_gated_moe.json
"""

import sys
import os
import json
import time
import traceback
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_shared.moe import MOERegressor, ExpertConfig

# BioGatedMOE may not exist in all versions — try to import
try:
    from confluencia_shared.moe import BioGatedMOERegressor
    HAS_BIO_GATED = True
except ImportError:
    HAS_BIO_GATED = False
    print("WARNING: BioGatedMOERegressor not available in this code version")


def generate_epitope_data(n=300, seed=42):
    """Generate synthetic epitope-like data for MOE comparison."""
    rng = np.random.RandomState(seed)
    n_features = 317
    X = rng.randn(n, n_features)

    y = np.zeros(n)
    y[:100] = X[:100, :5].sum(axis=1) * 0.3 + rng.randn(100) * 0.1
    y[100:200] = (X[100:200, 0] * X[100:200, 1] + X[100:200, 2] ** 2) * 0.5 + rng.randn(100) * 0.15
    y[200:300] = X[200:300, :3].sum(axis=1) * 0.2 + X[200:300, 3] * X[200:300, 4] * 0.3 + rng.randn(100) * 0.12

    y = (y - y.min()) / (y.max() - y.min())
    return X, y


def evaluate_moe_variant(X, y, moe_class, moe_name, n_folds=5):
    """Evaluate a MOE variant with K-fold cross-validation."""
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mae_scores = []
    r2_scores = []
    fold_details = []
    total_train_time = 0

    config = ExpertConfig()  # Use default hyperparameters

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Medium profile: 3 experts (ridge, hgb, rf) for N=300
        expert_names = ["ridge", "hgb", "rf"]

        start_time = time.time()

        if moe_name == "BioGatedMOE":
            moe = BioGatedMOERegressor(expert_names=expert_names, folds=4, config=config)
        else:
            moe = MOERegressor(expert_names=expert_names, folds=4, config=config)

        moe.fit(X_train, y_train)
        y_pred = moe.predict(X_test)

        elapsed = time.time() - start_time
        total_train_time += elapsed

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        r2_scores.append(r2)
        fold_details.append({
            "fold": fold_idx,
            "mae": float(mae),
            "r2": float(r2),
            "train_time_s": float(elapsed),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    return {
        "moe_variant": moe_name,
        "mean_mae": float(np.mean(mae_scores)),
        "std_mae": float(np.std(mae_scores)),
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "total_train_time_s": float(total_train_time),
        "n_folds": n_folds,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "expert_names": expert_names,
        "fold_details": fold_details,
    }


def main():
    print("=" * 70)
    print("Confluencia Experiment C: BioGatedMOE vs MOERegressor")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"BioGatedMOE available: {HAS_BIO_GATED}")
    print()

    X, y = generate_epitope_data(n=300)
    print(f"  Data: X shape {X.shape}, y range [{y.min():.3f}, {y.max():.3f}]")
    print()

    all_results = {
        "experiment": "C_bio_gated_moe_comparison",
        "timestamp": datetime.now().isoformat(),
        "bio_gated_available": HAS_BIO_GATED,
        "data": {"n_samples": 300, "n_features": 317},
    }

    # ── Basic MOERegressor ──
    print("Running MOERegressor (basic inverse-RMSE weighting)...")
    try:
        basic_result = evaluate_moe_variant(X, y, MOERegressor, "MOERegressor")
        print(f"  MAE: {basic_result['mean_mae']:.4f} ± {basic_result['std_mae']:.4f}")
        print(f"  R²: {basic_result['mean_r2']:.4f} ± {basic_result['std_r2']:.4f}")
        print(f"  Time: {basic_result['total_train_time_s']:.2f}s")
    except Exception as e:
        basic_result = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"  ERROR: {str(e)}")

    all_results["basic_moe"] = basic_result

    # ── BioGatedMOE ──
    if HAS_BIO_GATED:
        print("\nRunning BioGatedMOE (membrane + emotional + lateral inhibition)...")
        try:
            bio_result = evaluate_moe_variant(X, y, BioGatedMOERegressor, "BioGatedMOE")
            print(f"  MAE: {bio_result['mean_mae']:.4f} ± {bio_result['std_mae']:.4f}")
            print(f"  R²: {bio_result['mean_r2']:.4f} ± {bio_result['std_r2']:.4f}")
            print(f"  Time: {bio_result['total_train_time_s']:.2f}s")

            # Comparison
            if "error" not in basic_result and "error" not in bio_result:
                mae_delta = basic_result['mean_mae'] - bio_result['mean_mae']
                r2_delta = bio_result['mean_r2'] - basic_result['mean_r2']
                comparison = {
                    "mae_improvement": float(mae_delta),
                    "mae_improvement_pct": float((mae_delta / basic_result['mean_mae']) * 100),
                    "r2_improvement": float(r2_delta),
                    "bio_gated_wins_mae": mae_delta > 0,
                    "bio_gated_wins_r2": r2_delta > 0,
                }
                print(f"\n  MAE improvement: {mae_delta:+.4f} ({(mae_delta/basic_result['mean_mae']*100):+.1f}%)")
                print(f"  R² improvement: {r2_delta:+.4f}")
            else:
                comparison = {"error": "One or both variants failed"}

        except Exception as e:
            bio_result = {"error": str(e), "traceback": traceback.format_exc()}
            comparison = {"error": f"BioGatedMOE failed: {str(e)}"}
            print(f"  ERROR: {str(e)}")

        all_results["bio_gated_moe"] = bio_result
        all_results["comparison"] = comparison
    else:
        all_results["bio_gated_moe"] = {"error": "BioGatedMOERegressor not available in this version"}
        all_results["comparison"] = {"error": "Cannot compare — BioGatedMOE not available"}
        print("  SKIPPED: BioGatedMOE not available")

    # ── Save ──
    output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_C_bio_gated_moe.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()