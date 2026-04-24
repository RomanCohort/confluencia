"""
Full-Scale Drug Training (91k Extended Dataset)
================================================
Trains MOE ensemble on 91k drug data with multi-task prediction.

Targets:
- efficacy (primary)
- target_binding, immune_activation, immune_cell_activation, inflammation_risk, toxicity_risk (micro-targets)

Output:
- benchmarks/results/train_drug_91k.json
- data/cache/drug_model_91k.joblib (trained model)
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import joblib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-drug"))
sys.path.insert(0, str(PROJECT / "confluencia_shared"))
sys.path.insert(0, str(PROJECT))

from core.features import build_feature_matrix

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = PROJECT / "confluencia-2.0-drug" / "data" / "breast_cancer_drug_dataset_extended.csv"
OUT_PATH = PROJECT / "benchmarks" / "results" / "train_drug_91k.json"
MODEL_PATH = PROJECT / "data" / "cache" / "drug_model_91k.joblib"

# Targets to predict
TARGETS = ["efficacy", "target_binding", "immune_activation",
           "immune_cell_activation", "inflammation_risk", "toxicity_risk"]


def train_regressor(name, model, X_tr, y_tr, X_te, y_te, use_scaled=False):
    """Train regressor and return metrics."""
    t0 = time.time()

    if use_scaled:
        scaler = StandardScaler()
        X_tr_use = scaler.fit_transform(X_tr)
        X_te_use = scaler.transform(X_te)
    else:
        scaler = None
        X_tr_use = X_tr
        X_te_use = X_te

    model.fit(X_tr_use, y_tr)
    elapsed = time.time() - t0

    pred = model.predict(X_te_use)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)

    # Pearson correlation
    r = np.corrcoef(y_te, pred)[0, 1]

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(r),
        "train_time": elapsed,
    }, model, scaler


def train_moe_ensemble(X_tr, y_tr, X_te, y_te, folds=5):
    """Train MOE ensemble with OOF weighting."""
    t0 = time.time()

    experts = {
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05,
                                             min_samples_leaf=20, random_state=42),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    }

    # OOF predictions for weighting
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(y_tr)) for name in experts}
    oof_rmse = {}

    for name, model_template in experts.items():
        fold_rmse = []
        for tr_idx, va_idx in kf.split(X_tr):
            m = clone(model_template)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            pred = m.predict(X_tr[va_idx])
            oof_preds[name][va_idx] = pred
            fold_rmse.append(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))
        oof_rmse[name] = np.mean(fold_rmse)

    # Inverse RMSE weighting
    inv_rmse = {name: 1.0 / max(rmse, 1e-6) for name, rmse in oof_rmse.items()}
    total = sum(inv_rmse.values())
    weights = {name: w / total for name, w in inv_rmse.items()}

    # Train final models
    trained = {}
    for name, model_template in experts.items():
        m = clone(model_template)
        m.fit(X_tr, y_tr)
        trained[name] = m

    # Ensemble prediction
    pred_ens = sum(weights[name] * trained[name].predict(X_te) for name in experts)

    elapsed = time.time() - t0

    mae = mean_absolute_error(y_te, pred_ens)
    rmse = np.sqrt(mean_squared_error(y_te, pred_ens))
    r2 = r2_score(y_te, pred_ens)
    r = np.corrcoef(y_te, pred_ens)[0, 1]

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(r),
        "train_time": elapsed,
        "weights": weights,
        "oof_rmse": oof_rmse,
    }, trained, weights


def main():
    print("=" * 60)
    print("Full-Scale Drug Training (91k Extended Dataset)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")
    print(f"  Columns: {list(df.columns)}")

    # 2. Split by group_id (to avoid data leakage)
    print("\n[2] Group-aware split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["group_id"].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Train groups: {train_df['group_id'].nunique()}, Test groups: {test_df['group_id'].nunique()}")

    # 3. Feature extraction
    print("\n[3] Feature extraction (this may take a while)...")
    t0 = time.time()
    X_train, env_cols, backend = build_feature_matrix(train_df)
    t_feat_train = time.time() - t0
    print(f"  Train features: {X_train.shape}, backend={backend} ({t_feat_train:.1f}s)")

    t0 = time.time()
    X_test, _, _ = build_feature_matrix(test_df)
    t_feat_test = time.time() - t0
    print(f"  Test features: {X_test.shape} ({t_feat_test:.1f}s)")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Train models for each target
    all_results = {}
    all_models = {}
    best_models = {}

    for target in TARGETS:
        print(f"\n[4] Training for target: {target}")
        print("-" * 40)

        y_train = train_df[target].values.astype(np.float32)
        y_test = test_df[target].values.astype(np.float32)

        target_results = {}

        # MOE Ensemble
        print("  [MOE]", end=" ", flush=True)
        r, models, weights = train_moe_ensemble(X_train, y_train, X_test, y_test)
        target_results["MOE"] = r
        print(f"MAE={r['mae']:.4f} R2={r['r2']:.4f} Pearson={r['pearson_r']:.4f} ({r['train_time']:.1f}s)")

        # HGB
        print("  [HGB]", end=" ", flush=True)
        r, model, _ = train_regressor("HGB",
            HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05,
                                         min_samples_leaf=20, random_state=42),
            X_train, y_train, X_test, y_test)
        target_results["HGB"] = r
        print(f"MAE={r['mae']:.4f} R2={r['r2']:.4f} Pearson={r['pearson_r']:.4f} ({r['train_time']:.1f}s)")

        # RF
        print("  [RF]", end=" ", flush=True)
        r, model, _ = train_regressor("RF",
            RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            X_train, y_train, X_test, y_test)
        target_results["RF"] = r
        print(f"MAE={r['mae']:.4f} R2={r['r2']:.4f} Pearson={r['pearson_r']:.4f} ({r['train_time']:.1f}s)")

        # Ridge
        print("  [Ridge]", end=" ", flush=True)
        r, model, _ = train_regressor("Ridge",
            Ridge(alpha=1.0),
            X_train_s, y_train, X_test_s, y_test, use_scaled=True)
        target_results["Ridge"] = r
        print(f"MAE={r['mae']:.4f} R2={r['r2']:.4f} Pearson={r['pearson_r']:.4f} ({r['train_time']:.1f}s)")

        all_results[target] = target_results

        # Find best for this target
        best = max(target_results.keys(), key=lambda k: target_results[k]["r2"])
        best_models[target] = best
        print(f"  Best: {best} (R2={target_results[best]['r2']:.4f})")

    # 6. Save results
    output = {
        "targets": TARGETS,
        "best_models": best_models,
        "results": all_results,
        "data": {
            "total": len(df),
            "train": len(train_df),
            "test": len(test_df),
            "features": X_train.shape[1],
            "feature_backend": backend,
        },
        "timing": {
            "feature_extraction_train": t_feat_train,
            "feature_extraction_test": t_feat_test,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUT_PATH}")

    # 7. Save model bundle (primary target: efficacy)
    print("\n[5] Saving model for primary target (efficacy)...")

    # Retrain best model for efficacy on full training data
    y_train_eff = train_df["efficacy"].values.astype(np.float32)
    y_test_eff = test_df["efficacy"].values.astype(np.float32)

    best_name = best_models["efficacy"]
    if best_name == "MOE":
        # Retrain MOE
        _, trained_models, weights = train_moe_ensemble(X_train, y_train_eff, X_test, y_test_eff)
        final_model = {"type": "MOE", "models": trained_models, "weights": weights}
    else:
        # Retrain single model
        if best_name == "Ridge":
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train_eff)
            final_model = {"type": "Ridge", "model": model}
        elif best_name == "HGB":
            model = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05,
                                                  min_samples_leaf=20, random_state=42)
            model.fit(X_train, y_train_eff)
            final_model = {"type": "HGB", "model": model}
        elif best_name == "RF":
            model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train_eff)
            final_model = {"type": "RF", "model": model}
        else:
            final_model = None

    model_bundle = {
        "model": final_model,
        "scaler": scaler,
        "env_cols": env_cols,
        "feature_backend": backend,
        "config": {
            "best_model": best_name,
            "n_train": len(train_df),
            "r2": all_results["efficacy"][best_name]["r2"],
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Target':<25s} {'Best':<8s} {'R2':>8s} {'Pearson':>8s}")
    print("-" * 55)
    for target in TARGETS:
        best = best_models[target]
        r = all_results[target][best]
        print(f"{target:<25s} {best:<8s} {r['r2']:8.4f} {r['pearson_r']:8.4f}")


if __name__ == "__main__":
    main()
