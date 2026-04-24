"""
快速基准测试：去重 + ESM-2 缓存
=================================
优化点：
1. 按 epitope_seq 去重，减少编码量 (288k → 139k)
2. ESM-2 特征按唯一序列编码后广播
3. 多种阈值同时评估
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

DATA_PATH = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
RESULTS_DIR = PROJECT / "benchmarks" / "results"


def main(esm2_model: str = "35M", thresholds: list = None):
    if thresholds is None:
        thresholds = [2.0, 2.5, 3.0]

    print("=" * 60)
    print(f"快速基准测试 (去重 + ESM-2 {esm2_model})")
    print("=" * 60)

    # 1. 加载数据
    df = pd.read_csv(DATA_PATH)
    print(f"\n[Data] 原始: {len(df)} 行, {df['epitope_seq'].nunique()} 唯一序列")

    # 2. 去重：每个序列取一行（保留第一个出现的）
    df_dedup = df.drop_duplicates(subset=['epitope_seq'], keep='first').reset_index(drop=True)
    print(f"[Data] 去重后: {len(df_dedup)} 行")

    # 3. 分割
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_dedup, groups=df_dedup["epitope_seq"].values))
    train_df = df_dedup.iloc[train_idx].reset_index(drop=True)
    test_df = df_dedup.iloc[test_idx].reset_index(drop=True)
    print(f"[Split] Train: {len(train_df)}, Test: {len(test_df)}")

    # 4. 特征提取
    from core.features import FeatureSpec, build_feature_matrix, ensure_columns
    from core.esm2_encoder import ESM2Encoder

    # 4a. ESM-2 编码 (唯一序列)
    print(f"\n[ESM-2] 编码 {esm2_model}...")
    encoder = ESM2Encoder(model_size=esm2_model, cache_dir=str(PROJECT / "data" / "cache" / "esm2"))

    t0 = time.time()
    train_seqs = train_df['epitope_seq'].tolist()
    test_seqs = test_df['epitope_seq'].tolist()
    esm2_train = encoder.encode(train_seqs)
    esm2_test = encoder.encode(test_seqs)
    t_esm2 = time.time() - t0
    print(f"  Train ESM-2: {esm2_train.shape} ({t_esm2:.1f}s)")
    print(f"  Test ESM-2: {esm2_test.shape}")

    # 4b. 其他特征
    spec = FeatureSpec(use_esm2=False)  # 不用 build_feature_matrix 的 ESM-2
    X_train_base, names_base, _ = build_feature_matrix(train_df, spec)
    X_test_base, _, _ = build_feature_matrix(test_df, spec)
    print(f"  Base features: {X_train_base.shape}")

    # 4c. 拼接
    X_train = np.hstack([X_train_base, esm2_train])
    X_test = np.hstack([X_test_base, esm2_test])
    print(f"  Final features: {X_train.shape}")

    # 5. 多阈值评估
    all_results = {}

    for thresh in thresholds:
        y_train = (train_df["efficacy"] >= thresh).astype(int).values
        y_test = (test_df["efficacy"] >= thresh).astype(int).values

        if y_test.sum() < 5 or (1 - y_test).sum() < 5:
            print(f"\n[threshold={thresh}] 跳过 (类别不平衡)")
            continue

        print(f"\n[threshold={thresh}] binder_rate: {y_test.mean():.1%}")

        # HGB
        t0 = time.time()
        hgb = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=8,
            l2_regularization=1.0, random_state=42
        )
        hgb.fit(X_train, y_train)
        y_prob = hgb.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        t_train = time.time() - t0

        print(f"  HGB: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f} ({t_train:.1f}s)")

        all_results[f"thresh_{thresh}"] = {
            "threshold": thresh,
            "binder_rate": float(y_test.mean()),
            "auc": float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "mcc": float(mcc),
            "train_time": float(t_train),
        }

    # 保存
    output = {
        "config": {
            "esm2_model": esm2_model,
            "dedup": True,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(X_train.shape[1]),
            "esm2_encode_time": float(t_esm2),
        },
        "results": all_results,
    }

    out_path = RESULTS_DIR / f"quick_benchmark_{esm2_model}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n保存至: {out_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2-model", default="35M", choices=["8M", "35M", "150M", "650M"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.0, 2.5, 3.0])
    args = parser.parse_args()
    main(esm2_model=args.esm2_model, thresholds=args.thresholds)