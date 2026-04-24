"""
NetMHCpan-4.1 Enhanced Comparison (ESM-2 650M)
================================================
使用 ESM-2 增强特征与 NetMHCpan-4.1 进行 head-to-head 对比。

评估数据集:
  1. netmhcpan_heldout.csv (61 peptides, ic50 + is_binder)
  2. iedb_heldout_mhc.csv (2167 peptides, ic50 + is_binder + mhc_allele)

参考: NetMHCpan-4.1 AUC ~0.92-0.96 on MHC-I binding prediction
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import FeatureSpec, build_feature_matrix, ensure_columns

DATA_DIR = PROJECT / "benchmarks" / "data"
RESULTS_DIR = PROJECT / "benchmarks" / "results"


def load_benchmark_data(name: str) -> pd.DataFrame:
    """加载基准数据集"""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Benchmark data not found: {path}")

    df = pd.read_csv(path)
    print(f"[{name}] Loaded {len(df)} peptides")
    if "is_binder" in df.columns:
        print(f"  Binders: {df['is_binder'].sum()}, Non-binders: {(~df['is_binder'].astype(bool)).sum()}")
    return df


def train_on_iedb_full(
    exclude_seqs: set,
    spec: FeatureSpec,
    max_unique_seqs: int = 20000,
) -> HistGradientBoostingClassifier:
    """在 IEDB 全量数据上训练分类器 (排除 benchmark 序列)"""
    train_path = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
    train_df = pd.read_csv(train_path)

    # 排除 benchmark 序列
    train_df = train_df[~train_df["epitope_seq"].str.upper().isin(exclude_seqs)].reset_index(drop=True)

    # 二值化
    train_df["label"] = (train_df["efficacy"] >= 3.0).astype(int)

    # 子采样 (如果序列太多)
    unique_seqs = train_df["epitope_seq"].unique()
    if len(unique_seqs) > max_unique_seqs:
        rng = np.random.default_rng(42)
        selected = rng.choice(unique_seqs, size=max_unique_seqs, replace=False)
        train_df = train_df[train_df["epitope_seq"].isin(selected)].reset_index(drop=True)

    print(f"  Training on {len(train_df)} rows ({train_df['label'].mean():.1%} binders)")

    # 特征提取
    X_train, _, _ = build_feature_matrix(ensure_columns(train_df), spec)
    y_train = train_df["label"].values

    # 训练
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model


def evaluate_on_benchmark(
    model,
    benchmark_df: pd.DataFrame,
    spec: FeatureSpec,
    dataset_name: str,
) -> Dict[str, Any]:
    """在 benchmark 数据集上评估"""
    work = benchmark_df.copy()

    # 确保特征兼容
    work = ensure_columns(work)
    X, _, _ = build_feature_matrix(work, spec)

    # 预测
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except AttributeError:
        # 回归模型 fallback
        y_pred_raw = model.predict(X)
        y_prob = (y_pred_raw - y_pred_raw.min()) / max(y_pred_raw.max() - y_pred_raw.min(), 1e-8)

    y_true = benchmark_df["is_binder"].values.astype(int)

    # AUC
    auc = roc_auc_score(y_true, y_prob)

    # 最优阈值 (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_prob >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # IC50 相关性 (如果有)
    ic50_corr = None
    if "ic50_nm" in benchmark_df.columns:
        log_ic50 = np.log10(np.maximum(benchmark_df["ic50_nm"].values, 1.0))
        corr, pval = stats.pearsonr(y_prob, log_ic50)
        ic50_corr = {"corr": float(corr), "p_value": float(pval)}

    result = {
        "dataset": dataset_name,
        "n_samples": int(len(y_true)),
        "n_binders": int(y_true.sum()),
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
        "optimal_threshold": float(optimal_threshold),
        "ic50_correlation": ic50_corr,
    }

    print(f"\n[{dataset_name}] Results:")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  MCC:        {mcc:.4f}")
    if ic50_corr:
        print(f"  Corr(logIC50): {ic50_corr['corr']:.4f} (p={ic50_corr['p_value']:.4f})")

    return result


def main(
    esm2_model_size: str = "650M",
    esm2_pca_dim: int = 64,  # PCA降维，默认64维
    output_path: Optional[str] = None,
):
    """主评估流程"""
    print("=" * 60)
    print("NetMHCpan Enhanced Comparison (ESM-2)")
    print(f"ESM-2 model: {esm2_model_size}")
    print(f"ESM-2 PCA dim: {esm2_pca_dim}")
    print("=" * 60)

    # 特征配置
    spec = FeatureSpec(
        use_esm2=True,
        esm2_model_size=esm2_model_size,
        esm2_cache_dir=str(PROJECT / "data" / "cache" / "esm2"),
        esm2_pca_dim=esm2_pca_dim,
    )

    # 加载 benchmark 数据
    results = {}

    # --- Benchmark 1: NetMHCpan heldout (61 peptides) ---
    try:
        print("\n[Benchmark 1] NetMHCpan heldout (61 peptides)")
        bm1 = load_benchmark_data("netmhcpan_heldout.csv")
        exclude_seqs = set(bm1["epitope_seq"].str.upper())

        print("\nTraining classifier...")
        t0 = time.time()
        model = train_on_iedb_full(exclude_seqs, spec)
        print(f"Training time: {time.time() - t0:.1f}s")

        r1 = evaluate_on_benchmark(model, bm1, spec, "netmhcpan_heldout_61")
        results["netmhcpan_61"] = r1
    except Exception as e:
        print(f"[Benchmark 1] Failed: {e}")
        results["netmhcpan_61"] = {"error": str(e)}

    # --- Benchmark 2: IEDB heldout MHC (2167 peptides) ---
    try:
        print("\n[Benchmark 2] IEDB heldout MHC (2167 peptides)")
        bm2 = load_benchmark_data("iedb_heldout_mhc.csv")
        exclude_seqs2 = set(bm2["epitope_seq"].str.upper())

        print("\nTraining classifier...")
        t0 = time.time()
        model2 = train_on_iedb_full(exclude_seqs2, spec, max_unique_seqs=20000)
        print(f"Training time: {time.time() - t0:.1f}s")

        r2 = evaluate_on_benchmark(model2, bm2, spec, "iedb_heldout_2167")
        results["iedb_2167"] = r2
    except Exception as e:
        print(f"[Benchmark 2] Failed: {e}")
        results["iedb_2167"] = {"error": str(e)}

    # --- 对比 NetMHCpan reference ---
    netmhcpan_ref = {
        "auc_range": [0.92, 0.96],
        "training_size": 180000,
    }

    comparison = {
        "config": {
            "esm2_model_size": esm2_model_size,
            "feature_spec": str(spec),
        },
        "results": results,
        "netmhcpan_reference": netmhcpan_ref,
        "summary": {},
    }

    # 汇总
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        if "error" not in r:
            gap = netmhcpan_ref["auc_range"][0] - r["auc"]
            print(f"  {name}: AUC={r['auc']:.4f} (gap to NetMHCpan: {gap:+.4f})")

    # 保存
    if output_path is None:
        output_path = str(RESULTS_DIR / "netmhcpan_esm2_benchmark.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {output_path}")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--esm2-model", default="650M", choices=["8M", "35M", "150M", "650M"])
    parser.add_argument("--esm2-pca-dim", type=int, default=64, help="PCA降维维度，0表示不做PCA")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    main(esm2_model_size=args.esm2_model, esm2_pca_dim=args.esm2_pca_dim, output_path=args.output)