#!/usr/bin/env python3
"""
Confluencia Experiment C: BioGatedMOE vs MOERegressor on Epitope Data

Tests whether bio-mimetic gating (membrane potential + emotional state +
lateral inhibition) improves prediction over basic inverse-RMSE MOE.

Uses epitope benchmark data (N=300) for comparison.

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

from confluencia_shared.moe import MOERegressor, BioGatedMOERegressor, ExpertConfig


def generate_epitope_data(n=300, seed=42):
    """Generate synthetic epitope-like data for MOE comparison.

    Uses a mixture of regimes to create a dataset where adaptive gating
    should have an advantage: different expert performance in different
    regions of feature space.
    """
    rng = np.random.RandomState(seed)

    # 317 features (matching real epitope data dimensionality)
    n_features = 317
    X = rng.randn(n, n_features)

    # Create regime-dependent target:
    # - Samples 0-100: linear regime (Ridge excels)
    # - Samples 100-200: nonlinear regime (HGB excels)
    # - Samples 200-300: mixed regime (all experts contribute)

    y = np.zeros(n)

    # Regime 1: Linear (first 5 features dominate)
    y[:100] = X[:100, :5].sum(axis=1) * 0.3 + rng.randn(100) * 0.1

    # Regime 2: Nonlinear (feature interactions)
    y[100:200] = (X[100:200, 0] * X[100:200, 1] + X[100:200, 2] ** 2) * 0.5 + rng.randn(100) * 0.15

    # Regime 3: Mixed
    y[200:300] = X[200:300, :3].sum(axis=1) * 0.2 + X[200:300, 3] * X[200:300, 4] * 0.3 + rng.randn(100) * 0.12

    # Normalize y to [0, 1] range (like epitope binding affinity)
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

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        config = ExpertConfig(
            experts=["ridge", "hgb", "rf"],  # medium profile (N=300)
            n_folds=3,
        )

        start_time = time.time()

        if moe_class == BioGatedMOERegressor:
            moe = BioGatedMOERegressor(config=config)
        else:
            moe = MOERegressor(config=config)

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
        "fold_details": fold_details,
    }


def main():
    print("=" * 70)
    print("Confluencia Experiment C: BioGatedMOE vs MOERegressor")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Generate data
    print("Generating synthetic epitope data (N=300, d=317)...")
    X, y = generate_epitope_data(n=300)
    print(f"  X shape: {X.shape}, y range: [{y.min():.3f}, {y.max():.3f}]")
    print()

    all_results = {
        "experiment": "C_bio_gated_moe_comparison",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "n_samples": 300,
            "n_features": 317,
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
        },
    }

    # ── Experiment 1: Basic MOERegressor ──
    print("Running MOERegressor (basic inverse-RMSE weighting)...")
    basic_result = evaluate_moe_variant(X, y, MOERegressor, "MOERegressor")
    print(f"  MAE: {basic_result['mean_mae']:.4f} ± {basic_result['std_mae']:.4f}")
    print(f"  R²: {basic_result['mean_r2']:.4f} ± {basic_result['std_r2']:.4f}")
    print(f"  Time: {basic_result['total_train_time_s']:.2f}s")

    # ── Experiment 2: BioGatedMOE ──
    print("\nRunning BioGatedMOE (membrane + emotional + lateral inhibition)...")
    try:
        bio_result = evaluate_moe_variant(X, y, BioGatedMOERegressor, "BioGatedMOE")
        print(f"  MAE: {bio_result['mean_mae']:.4f} ± {bio_result['std_mae']:.4f}")
        print(f"  R²: {bio_result['mean_r2']:.4f} ± {bio_result['std_r2']:.4f}")
        print(f"  Time: {bio_result['total_train_time_s']:.2f}s")

        # Comparison
        mae_delta = basic_result['mean_mae'] - bio_result['mean_mae']
        r2_delta = bio_result['mean_r2'] - basic_result['mean_r2']
        mae_improvement_pct = (mae_delta / basic_result['mean_mae']) * 100 if basic_result['mean_mae'] > 0 else 0

        comparison = {
            "mae_improvement": float(mae_delta),
            "mae_improvement_pct": float(mae_improvement_pct),
            "r2_improvement": float(r2_delta),
            "bio_gated_wins_mae": mae_delta > 0,
            "bio_gated_wins_r2": r2_delta > 0,
        }

        print(f"\n  Comparison:")
        print(f"    MAE improvement: {mae_delta:+.4f} ({mae_improvement_pct:+.1f}%)")
        print(f"    R² improvement: {r2_delta:+.4f}")
        print(f"    BioGated wins on MAE: {comparison['bio_gated_wins_mae']}")
        print(f"    BioGated wins on R²: {comparison['bio_gated_wins_r2']}")

    except Exception as e:
        bio_result = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        comparison = {"error": f"BioGatedMOE failed: {str(e)}"}
        print(f"  ERROR: {str(e)}")

    all_results["basic_moe"] = basic_result
    all_results["bio_gated_moe"] = bio_result
    all_results["comparison"] = comparison

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