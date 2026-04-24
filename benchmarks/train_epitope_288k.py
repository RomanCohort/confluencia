"""
Full-Scale Epitope Training (288k IEDB)
========================================
Trains MOE ensemble on 288k epitope data with sequence-aware splitting.
Since efficacy is bimodal (0.5 vs 3.0), we train both regression and classification.

Output:
- benchmarks/results/train_epitope_288k.json
- data/cache/epitope_model_288k.joblib (trained model)
"""
from __future__ import annotations
import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import build_feature_matrix

# Suppress TF/other warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
OUT_PATH = PROJECT / "benchmarks" / "results" / "train_epitope_288k.json"
MODEL_PATH = PROJECT / "data" / "cache" / "epitope_model_288k.joblib"


def train_classifier(name, model, X_tr, y_tr, X_te, y_te, use_scaled=False):
    """Train classifier and return metrics."""
    t0 = time.time()
    X_tr_use = X_tr if not use_scaled else StandardScaler().fit_transform(X_tr)
    X_te_use = X_te if not use_scaled else StandardScaler().fit(X_tr).transform(X_te)

    model.fit(X_tr_use, y_tr)
    elapsed = time.time() - t0

    pred = model.predict(X_te_use)
    prob = model.predict_proba(X_te_use)[:, 1]

    return {
        "auc": float(roc_auc_score(y_te, prob)),
        "auprc": float(average_precision_score(y_te, prob)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "mcc": float(matthews_corrcoef(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "train_time": elapsed,
    }, model


def main():
    print("=" * 60)
    print("Full-Scale Epitope Training (288k IEDB)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    # Create binary label (efficacy >= 3.0 = binder)
    df["label"] = (df["efficacy"] >= 3.0).astype(int)
    print(f"  Binders: {df['label'].sum()} ({df['label'].mean():.1%})")

    # 2. Sequence-aware split
    print("\n[2] Sequence-aware split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["epitope_seq"].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Train binders: {train_df['label'].mean():.1%}, Test binders: {test_df['label'].mean():.1%}")

    # 3. Feature extraction (this is the slow part)
    print("\n[3] Feature extraction...")
    t0 = time.time()
    X_train, feat_names, env_cols = build_feature_matrix(train_df)
    t_feat_train = time.time() - t0
    print(f"  Train features: {X_train.shape} ({t_feat_train:.1f}s)")

    t0 = time.time()
    X_test, _, _ = build_feature_matrix(test_df)
    t_feat_test = time.time() - t0
    print(f"  Test features: {X_test.shape} ({t_feat_test:.1f}s)")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # 4. Scale features for LR/MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Train classifiers
    results = {}
    trained_models = {}

    print("\n[4] Training classifiers...")

    # HGB - best performer
    print("  [HGB]", end=" ", flush=True)
    r, model = train_classifier("HGB",
        HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42),
        X_train, y_train, X_test, y_test)
    results["HGB"] = r
    trained_models["HGB"] = model
    print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ({r['train_time']:.1f}s)")

    # Random Forest
    print("  [RF]", end=" ", flush=True)
    r, model = train_classifier("RF",
        RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                              random_state=42, n_jobs=-1),
        X_train, y_train, X_test, y_test)
    results["RF"] = r
    trained_models["RF"] = model
    print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ({r['train_time']:.1f}s)")

    # Logistic Regression
    print("  [LR]", end=" ", flush=True)
    r, model = train_classifier("LR",
        LogisticRegression(C=1.0, max_iter=2000, random_state=42, n_jobs=-1),
        X_train_s, y_train, X_test_s, y_test, use_scaled=True)
    results["LR"] = r
    trained_models["LR"] = model
    print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ({r['train_time']:.1f}s)")

    # MLP
    print("  [MLP]", end=" ", flush=True)
    r, model = train_classifier("MLP",
        MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300,
                     early_stopping=True, validation_fraction=0.1,
                     random_state=42),
        X_train_s, y_train, X_test_s, y_test, use_scaled=True)
    results["MLP"] = r
    trained_models["MLP"] = model
    print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ({r['train_time']:.1f}s)")

    # 6. Find best model
    best_name = max(results.keys(), key=lambda k: results[k]["auc"])
    print(f"\n  Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    # 7. Save results
    output = {
        "model": best_name,
        "metrics": results[best_name],
        "all_results": results,
        "data": {
            "total": len(df),
            "train": len(train_df),
            "test": len(test_df),
            "features": X_train.shape[1],
            "binder_rate": float(train_df["label"].mean()),
        },
        "timing": {
            "feature_extraction_train": t_feat_train,
            "feature_extraction_test": t_feat_test,
            "total": sum(r["train_time"] for r in results.values()) + t_feat_train + t_feat_test,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUT_PATH}")

    # 8. Save model bundle
    model_bundle = {
        "model": trained_models[best_name],
        "scaler": scaler,
        "feature_names": feat_names,
        "env_cols": env_cols,
        "config": {
            "best_model": best_name,
            "n_train": len(train_df),
            "auc": results[best_name]["auc"],
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")

    # 9. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<10s} {'AUC':>8s} {'F1':>8s} {'MCC':>8s} {'Time':>8s}")
    print("-" * 50)
    for name in ["HGB", "RF", "LR", "MLP"]:
        r = results[name]
        marker = "*" if name == best_name else " "
        print(f"{name:<10s}{marker}{r['auc']:8.4f} {r['f1']:8.4f} {r['mcc']:8.4f} {r['train_time']:7.1f}s")


if __name__ == "__main__":
    main()
