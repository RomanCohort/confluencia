"""
288k 全量数据 + MHC 特征基准测试
=================================
在 288k IEDB 数据上测试 MHC 特征效果
（无 allele 信息时使用默认 HLA-A*02:01）
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import FeatureSpec, build_feature_matrix

DATA_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
OUT_PATH = PROJECT / "benchmarks" / "results" / "mhc_288k_benchmark.json"


def main():
    print("=" * 60)
    print("288k + MHC Benchmark")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df["label"] = (df["efficacy"] >= 3.0).astype(int)
    print(f"[Data] {len(df)} rows, binders: {df['label'].mean():.1%}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["epitope_seq"].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"[Split] Train: {len(train_df)}, Test: {len(test_df)}")

    results = {}

    for name, spec in [
        ("baseline", FeatureSpec(use_mhc=False, use_esm2=False)),
        ("mhc", FeatureSpec(use_mhc=True, use_esm2=False)),
    ]:
        print(f"\n[{name}] Extracting features...")
        t0 = time.time()
        X_train, feat_names, _ = build_feature_matrix(train_df, spec)
        X_test, _, _ = build_feature_matrix(test_df, spec)
        t_feat = time.time() - t0
        print(f"  Features: {X_train.shape} ({t_feat:.1f}s)")

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        t0 = time.time()
        hgb = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=8,
            l2_regularization=1.0, random_state=42,
        )
        hgb.fit(X_train, y_train)
        t_train = time.time() - t0

        y_prob = hgb.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results[name] = {
            "auc": float(roc_auc_score(y_test, y_prob)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "n_features": int(X_train.shape[1]),
            "feat_time": float(t_feat),
            "train_time": float(t_train),
        }
        r = results[name]
        print(f"  AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}, MCC={r['mcc']:.4f}")

    print(f"\n{'Config':15s} {'AUC':>6s} {'F1':>6s} {'MCC':>6s} {'Feats':>6s}")
    print("-" * 42)
    for n, r in results.items():
        print(f"{n:15s} {r['auc']:6.4f} {r['f1']:6.4f} {r['mcc']:6.4f} {r['n_features']:6d}")

    delta = results["mhc"]["auc"] - results["baseline"]["auc"]
    print(f"\nMHC delta: {delta:+.4f}")

    output = {"results": results, "delta_mhc": float(delta)}
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()