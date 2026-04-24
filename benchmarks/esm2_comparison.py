"""
ESM-2 增强对比实验 (ESM-2 35M, CPU优化版)
=========================================
使用 ESM-2 35M 模型避免 CPU OOM，快速验证策略

实验:
  A. 传统特征 (317维) - 基线
  B. 传统 + ESM-2 PCA 32维补充
  C. 传统 + ESM-2 PCA 64维补充
  D. 传统 + ESM-2 PCA 128维补充
"""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, roc_curve
from sklearn.decomposition import IncrementalPCA
from scipy import stats
import joblib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "benchmarks" / "data"
RESULTS_DIR = PROJECT / "benchmarks" / "results"
CACHE_DIR = PROJECT / "data" / "cache" / "esm2"

# 使用 35M 模型 (480维)
ESM2_MODEL_SIZE = "35M"


def load_benchmark(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path)
    print(f"[{name}] {len(df)} peptides")
    return df


def get_esm2_encoder():
    """获取 ESM-2 编码器 (35M)"""
    from core.features import _get_esm2_encoder
    return _get_esm2_encoder(ESM2_MODEL_SIZE, str(CACHE_DIR))


def fit_pca(esm2_features: np.ndarray, pca_dim: int, cache_path: str) -> IncrementalPCA:
    """拟合 PCA 并缓存"""
    actual_dim = min(pca_dim, esm2_features.shape[1], esm2_features.shape[0])
    print(f"[PCA] 拟合: {esm2_features.shape[1]} -> {actual_dim} 维")
    pca = IncrementalPCA(n_components=actual_dim, batch_size=4096)
    pca.fit(esm2_features)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[PCA] 解释方差比: {explained:.4f}")
    joblib.dump(pca, cache_path)
    print(f"[PCA] 已缓存: {cache_path}")
    return pca


def apply_pca(esm2_features: np.ndarray, pca: IncrementalPCA) -> np.ndarray:
    """应用 PCA"""
    result = pca.transform(esm2_features)
    print(f"[PCA] 降维: {esm2_features.shape} -> {result.shape}")
    return result.astype(np.float32)


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec,
    name: str,
    esm2_train: np.ndarray = None,
    esm2_test: np.ndarray = None,
    pca_model: IncrementalPCA = None,
) -> Dict[str, Any]:
    """训练+评估"""
    from core.features import build_feature_matrix, ensure_columns

    # 特征构建 (传统特征)
    X_train, _, _ = build_feature_matrix(ensure_columns(train_df), spec)
    X_test, _, _ = build_feature_matrix(ensure_columns(test_df.copy()), spec)

    # 拼接 ESM-2 特征
    if esm2_train is not None and esm2_test is not None and pca_model is not None:
        esm2_train_pca = apply_pca(esm2_train, pca_model)
        esm2_test_pca = apply_pca(esm2_test, pca_model)
        X_train = np.concatenate([X_train, esm2_train_pca], axis=1)
        X_test = np.concatenate([X_test, esm2_test_pca], axis=1)

    y_train = train_df["label"].values

    print(f"  [{name}] 特征维度: {X_train.shape[1]}, 训练样本: {len(X_train)}")

    # 训练
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  [{name}] 训练时间: {train_time:.1f}s")

    # 评估
    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = test_df["is_binder"].values.astype(int)
    auc = roc_auc_score(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    result = {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "optimal_threshold": float(optimal_threshold),
        "train_time": float(train_time),
        "feature_dim": int(X_train.shape[1]),
    }

    if "ic50_nm" in test_df.columns:
        log_ic50 = np.log10(np.maximum(test_df["ic50_nm"].values, 1.0))
        corr, pval = stats.pearsonr(y_prob, log_ic50)
        result["ic50_corr"] = float(corr)

    print(f"  [{name}] AUC={auc:.4f} Acc={result['accuracy']:.4f} F1={result['f1']:.4f} MCC={result['mcc']:.4f}")
    return result


def main():
    print("=" * 60)
    print(f"ESM-2 增强策略对比实验 (ESM-2 {ESM2_MODEL_SIZE})")
    print("=" * 60)

    # 加载数据
    train_path = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
    train_full = pd.read_csv(train_path)

    bm_netmhc = load_benchmark("netmhcpan_heldout.csv")
    bm_iedb = load_benchmark("iedb_heldout_mhc.csv")

    # 排除测试集序列
    exclude_netmhc = set(bm_netmhc["epitope_seq"].str.upper())
    exclude_iedb = set(bm_iedb["epitope_seq"].str.upper())

    train_netmhc = train_full[~train_full["epitope_seq"].str.upper().isin(exclude_netmhc)].reset_index(drop=True)
    train_netmhc["label"] = (train_netmhc["efficacy"] >= 3.0).astype(int)

    # CPU 优化: 采样 1 万条训练序列
    np.random.seed(42)
    train_sample = train_netmhc.sample(n=min(10000, len(train_netmhc)), random_state=42).reset_index(drop=True)
    print(f"\n[采样] 训练集: {len(train_sample)} 条 (从 {len(train_netmhc)} 条采样)")

    # 特征配置
    from core.features import FeatureSpec
    spec_no_esm2 = FeatureSpec(use_esm2=False)

    results = {}

    # === 实验 A: 传统特征基线 ===
    print("\n" + "=" * 40)
    print("实验 A: 传统特征 (317维)")
    results["A_baseline"] = train_and_eval(
        train_sample, bm_netmhc,
        spec=spec_no_esm2, name="A_baseline"
    )

    # === ESM-2 编码 ===
    print(f"\n[ESM-2 {ESM2_MODEL_SIZE}] 编码序列...")
    encoder = get_esm2_encoder()

    netmhc_seqs = bm_netmhc["epitope_seq"].astype(str).tolist()
    train_seqs = train_sample["epitope_seq"].astype(str).tolist()

    t0 = time.time()
    print("  编码 benchmark...")
    esm2_test = encoder.encode(netmhc_seqs)
    print(f"  Benchmark done: {esm2_test.shape}")

    print("  编码训练集...")
    esm2_train = encoder.encode(train_seqs)
    print(f"  Train done: {esm2_train.shape}, time: {time.time()-t0:.1f}s")

    # === 实验 B/C/D: 传统 + ESM-2 PCA ===
    for pca_dim in [32, 64, 128]:
        label = f"{'B' if pca_dim==32 else 'C' if pca_dim==64 else 'D'}_esm2_{pca_dim}d"
        print(f"\n{'='*40}")
        print(f"实验 {label}: 传统 + ESM-2 PCA {pca_dim}维")

        # 拟合 PCA
        pca_path = CACHE_DIR / f"pca_{ESM2_MODEL_SIZE}_{pca_dim}d.joblib"
        pca = fit_pca(esm2_train, pca_dim, str(pca_path))

        results[label] = train_and_eval(
            train_sample, bm_netmhc,
            spec=spec_no_esm2,
            name=label,
            esm2_train=esm2_train,
            esm2_test=esm2_test,
            pca_model=pca,
        )

    # === 汇总 ===
    print("\n" + "=" * 60)
    print("NetMHCpan heldout (61 peptides) 结果对比")
    print("=" * 60)
    print(f"{'实验':<20} {'AUC':>8} {'Acc':>8} {'F1':>8} {'MCC':>8} {'Dim':>5}")
    print("-" * 60)
    for label, r in results.items():
        print(f"{label:<20} {r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} {r['feature_dim']:>5}")

    # 保存
    output = {
        "model": f"ESM-2 {ESM2_MODEL_SIZE}",
        "results": results,
        "baseline_ref": {"auc": 0.92, "model": "NetMHCpan-4.1"},
    }
    out_path = RESULTS_DIR / "esm2_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    main()
