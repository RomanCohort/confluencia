"""
Full-scale Binary Classification Benchmark (288k IEDB data)
============================================================
Sequence-aware split, trains on 231k rows, tests on 57k rows.

Options:
  --no-esm2      Disable ESM-2 features (default: enabled)
  --esm2-model   ESM-2 model size: 8M, 35M, 150M, 650M (default: 650M)
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import build_feature_matrix, FeatureSpec

DATA_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
OUT_PATH = PROJECT / "benchmarks" / "results" / "baselines_288k_binary.json"


def main(use_esm2: bool = True, esm2_model: str = "650M"):
    print("=" * 60)
    print("Binary Classification Benchmark (288k IEDB)")
    print(f"ESM-2: {use_esm2} ({esm2_model if use_esm2 else 'N/A'})")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["label"] = (df["efficacy"] >= 3.0).astype(int)
    print(f"  Total: {len(df)}, binders: {df['label'].sum()} ({df['label'].mean():.1%})")

    # 2. Sequence-aware split
    print("\n[2] Sequence-aware split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["epitope_seq"].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Train binders: {train_df['label'].mean():.1%}, Test binders: {test_df['label'].mean():.1%}")

    # 3. Feature extraction
    spec = FeatureSpec(
        use_esm2=use_esm2,
        esm2_model_size=esm2_model,
        esm2_cache_dir=str(PROJECT / "data" / "cache" / "esm2"),
    )
    print(f"\n[3] Feature extraction (ESM-2={use_esm2})...")
    t0 = time.time()
    X_train, names, _ = build_feature_matrix(train_df, spec)
    t_feat = time.time() - t0
    print(f"  Train features: {X_train.shape} ({t_feat:.1f}s)")

    t0 = time.time()
    X_test, _, _ = build_feature_matrix(test_df, spec)
    t_feat2 = time.time() - t0
    print(f"  Test features: {X_test.shape} ({t_feat2:.1f}s)")

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    y_reg_train = train_df["efficacy"].values.astype(np.float32)
    y_reg_test = test_df["efficacy"].values.astype(np.float32)

    # 4. Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Train and evaluate
    results = {}

    def eval_clf(name, y_true, y_pred, y_prob, train_time):
        r = {
            "auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "train_time": train_time,
        }
        results[name] = r
        print(f"  AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}, MCC={r['mcc']:.4f} ({train_time:.1f}s)")
        return r

    # --- Logistic Regression ---
    print("\n[4a] Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_s, y_train)
    eval_clf("LogisticRegression", y_test, lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1], time.time() - t0)

    # --- HGB Classifier ---
    print("\n[4b] HGB Classifier...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)
    hgb.fit(X_train, y_train)
    eval_clf("HGB", y_test, hgb.predict(X_test), hgb.predict_proba(X_test)[:, 1], time.time() - t0)

    # --- Random Forest ---
    print("\n[4c] Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    eval_clf("RF", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1], time.time() - t0)

    # --- MLP Classifier ---
    print("\n[4d] MLP Classifier...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=42)
    mlp.fit(X_train_s, y_train)
    eval_clf("MLP", y_test, mlp.predict(X_test_s), mlp.predict_proba(X_test_s)[:, 1], time.time() - t0)

    # --- MOE Ensemble (regression-based, threshold at 3.0) ---
    print("\n[4e] MOE Ensemble...")
    t0 = time.time()
    experts = {}
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for ename, emodel in [
        ("ridge", Ridge(alpha=1.2)),
        ("hgb_reg", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)),
    ]:
        oof = np.zeros(len(y_train), dtype=np.float32)
        for tr, va in kf.split(X_train):
            m = clone(emodel)
            m.fit(X_train[tr], y_reg_train[tr])
            oof[va] = m.predict(X_train[va]).astype(np.float32)
        m_full = clone(emodel)
        m_full.fit(X_train, y_reg_train)
        rmse = float(np.sqrt(np.mean((y_reg_train - oof) ** 2)))
        experts[ename] = (m_full, rmse)
        print(f"    {ename}: OOF-RMSE={rmse:.4f}")

    inv = np.array([1.0 / max(experts[k][1], 1e-6) for k in experts])
    weights = inv / inv.sum()
    print(f"    MOE weights: {dict(zip(experts.keys(), [f'{w:.3f}' for w in weights]))}")

    pred_ens = sum(w * experts[k][0].predict(X_test) for k, w in zip(experts.keys(), weights))
    pred_label = (pred_ens >= 3.0).astype(int)

    r_moe = {
        "auc": float(roc_auc_score(y_test, pred_ens)),
        "accuracy": float(accuracy_score(y_test, pred_label)),
        "f1": float(f1_score(y_test, pred_label)),
        "mcc": float(matthews_corrcoef(y_test, pred_label)),
        "train_time": time.time() - t0,
        "weights": {k: float(w) for k, w in zip(experts.keys(), weights)},
    }
    results["MOE"] = r_moe
    print(f"  AUC={r_moe['auc']:.4f}, Acc={r_moe['accuracy']:.4f}, F1={r_moe['f1']:.4f}, MCC={r_moe['mcc']:.4f}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY (288k IEDB Binary Classification)")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Features: {X_train.shape[1]}")
    print("=" * 60)
    print(f"{'Model':20s} {'AUC':>6s} {'Acc':>6s} {'F1':>6s} {'MCC':>6s} {'Time':>8s}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:20s} {r['auc']:6.4f} {r['accuracy']:6.4f} {r['f1']:6.4f} {r['mcc']:6.4f} {r['train_time']:7.1f}s")

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="288k Binary Classification Benchmark")
    parser.add_argument("--no-esm2", action="store_true", help="Disable ESM-2 features")
    parser.add_argument("--esm2-model", default="650M", choices=["8M", "35M", "150M", "650M"],
                        help="ESM-2 model size (default: 650M)")
    args = parser.parse_args()
    main(use_esm2=not args.no_esm2, esm2_model=args.esm2_model)
