"""
MHC 增强特征基准测试
====================
在 iedb_heldout_mhc.csv (2167行，含 mhc_allele) 上评估 MHC 特征的效果

特征组合:
  - Baseline: Mamba3Lite + kmer + biochem
  - +MHC: + MHC 伪序列 + HLA one-hot + 结合位置特征 (979维)
  - +ESM2: + ESM-2 嵌入
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "benchmarks" / "data"
RESULTS_DIR = PROJECT / "benchmarks" / "results"


def load_mhc_data():
    """加载带 MHC allele 的数据"""
    path = DATA_DIR / "iedb_heldout_mhc.csv"
    if not path.exists():
        raise FileNotFoundError(f"数据不存在: {path}")

    df = pd.read_csv(path)
    print(f"[Data] 加载 {len(df)} 行")
    print(f"  Binders: {df['is_binder'].sum()} ({df['is_binder'].mean():.1%})")
    print(f"  唯一 alleles: {df['mhc_allele'].nunique()}")
    print(f"  Top alleles: {df['mhc_allele'].value_counts().head(5).to_dict()}")
    return df


def extract_features(df: pd.DataFrame, spec) -> tuple:
    """提取特征"""
    from core.features import build_feature_matrix

    # 确保有必要的列
    work = df.copy()
    for col in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]:
        if col not in work.columns:
            work[col] = 0.0

    t0 = time.time()
    X, names, env_cols = build_feature_matrix(work, spec)
    t_feat = time.time() - t0
    print(f"  Features: {X.shape} ({t_feat:.1f}s)")

    y = df["is_binder"].astype(int).values
    return X, y, names


def run_benchmark():
    """运行 MHC 增强基准测试"""
    print("=" * 60)
    print("MHC 增强特征基准测试")
    print("=" * 60)

    from core.features import FeatureSpec

    df = load_mhc_data()

    # 数据分割 (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_binder"]
    )
    print(f"\n[Split] Train: {len(train_df)}, Test: {len(test_df)}")

    results = {}

    # ========== Baseline (no MHC, no ESM2) ==========
    print("\n[Baseline] 无 MHC, 无 ESM-2...")
    spec_baseline = FeatureSpec(use_mhc=False, use_esm2=False)
    X_train_base, y_train, _ = extract_features(train_df, spec_baseline)
    X_test_base, y_test, _ = extract_features(test_df, spec_baseline)

    # HGB
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=8, random_state=42
    )
    hgb.fit(X_train_base, y_train)
    y_prob = hgb.predict_proba(X_test_base)[:, 1]
    auc_base = roc_auc_score(y_test, y_prob)
    results["baseline_hgb"] = {
        "auc": float(auc_base),
        "acc": float(accuracy_score(y_test, (y_prob >= 0.5).astype(int))),
        "f1": float(f1_score(y_test, (y_prob >= 0.5).astype(int))),
        "mcc": float(matthews_corrcoef(y_test, (y_prob >= 0.5).astype(int))),
        "n_features": int(X_train_base.shape[1]),
        "time": float(time.time() - t0),
    }
    print(f"  HGB AUC: {auc_base:.4f}")

    # ========== +MHC ==========
    print("\n[+MHC] 添加 MHC 特征...")
    spec_mhc = FeatureSpec(use_mhc=True, use_esm2=False)
    X_train_mhc, y_train, _ = extract_features(train_df, spec_mhc)
    X_test_mhc, y_test, _ = extract_features(test_df, spec_mhc)

    t0 = time.time()
    hgb_mhc = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=8, random_state=42
    )
    hgb_mhc.fit(X_train_mhc, y_train)
    y_prob_mhc = hgb_mhc.predict_proba(X_test_mhc)[:, 1]
    auc_mhc = roc_auc_score(y_test, y_prob_mhc)
    results["mhc_hgb"] = {
        "auc": float(auc_mhc),
        "acc": float(accuracy_score(y_test, (y_prob_mhc >= 0.5).astype(int))),
        "f1": float(f1_score(y_test, (y_prob_mhc >= 0.5).astype(int))),
        "mcc": float(matthews_corrcoef(y_test, (y_prob_mhc >= 0.5).astype(int))),
        "n_features": int(X_train_mhc.shape[1]),
        "time": float(time.time() - t0),
    }
    print(f"  HGB AUC: {auc_mhc:.4f} (delta: {auc_mhc - auc_base:+.4f})")

    # ========== +MHC +ESM2 (8M, 轻量) ==========
    print("\n[+MHC +ESM2-8M] 添加 ESM-2 8M...")
    try:
        spec_full = FeatureSpec(use_mhc=True, use_esm2=True, esm2_model_size="8M")
        X_train_full, y_train, _ = extract_features(train_df, spec_full)
        X_test_full, y_test, _ = extract_features(test_df, spec_full)

        t0 = time.time()
        hgb_full = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=8, random_state=42
        )
        hgb_full.fit(X_train_full, y_train)
        y_prob_full = hgb_full.predict_proba(X_test_full)[:, 1]
        auc_full = roc_auc_score(y_test, y_prob_full)
        results["mhc_esm2_8m_hgb"] = {
            "auc": float(auc_full),
            "acc": float(accuracy_score(y_test, (y_prob_full >= 0.5).astype(int))),
            "f1": float(f1_score(y_test, (y_prob_full >= 0.5).astype(int))),
            "mcc": float(matthews_corrcoef(y_test, (y_prob_full >= 0.5).astype(int))),
            "n_features": int(X_train_full.shape[1]),
            "time": float(time.time() - t0),
        }
        print(f"  HGB AUC: {auc_full:.4f} (delta from baseline: {auc_full - auc_base:+.4f})")
    except Exception as e:
        print(f"  ESM-2 加载失败: {e}")
        results["mhc_esm2_8m_hgb"] = {"error": str(e)}

    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':20s} {'AUC':>6s} {'F1':>6s} {'MCC':>6s} {'Features':>8s}")
    print("-" * 50)
    for name, r in results.items():
        if "error" not in r:
            print(f"{name:20s} {r['auc']:6.4f} {r['f1']:6.4f} {r['mcc']:6.4f} {r['n_features']:8d}")

    # 保存
    output = {
        "config": {
            "data": "iedb_heldout_mhc.csv",
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
        "results": results,
        "netmhcpan_reference": {"auc_range": [0.92, 0.96]},
    }

    out_path = RESULTS_DIR / "mhc_benchmark.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n保存至: {out_path}")

    return output


if __name__ == "__main__":
    run_benchmark()