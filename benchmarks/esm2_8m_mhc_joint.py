"""
ESM-2 8M + MHC 联合基准测试
============================
策略: 传统特征(317D) + MHC(979D) + ESM-2 8M(320D) 联合

实验:
  A. 传统特征 (317D)
  B. 传统 + MHC (1296D) — 当前最优 AUC=0.917
  C. 传统 + MHC + ESM-2 8M (1616D)
  D. 传统 + MHC + ESM-2 8M PCA 32D (1328D)
  E. 传统 + MHC + ESM-2 8M PCA 64D (1360D)

评估: IEDB heldout MHC (N=2166) + NetMHCpan heldout (N=61)
"""
from __future__ import annotations

import json
import time
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

from core.features import FeatureSpec, build_feature_matrix, ensure_columns, _get_esm2_encoder
from core.mhc_features import MHCFeatureEncoder

DATA_DIR = PROJECT / "benchmarks" / "data"
RESULTS_DIR = PROJECT / "benchmarks" / "results"
CACHE_DIR = PROJECT / "data" / "cache" / "esm2"

ESM2_SIZE = "8M"


def load_benchmark(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path)
    print(f"[{name}] {len(df)} peptides")
    return df


def get_esm2_encoder():
    return _get_esm2_encoder(ESM2_SIZE, str(CACHE_DIR))


def fit_pca(features: np.ndarray, dim: int, path: str):
    actual = min(dim, features.shape[1], features.shape[0])
    print(f"  [PCA] {features.shape[1]} -> {actual}D")
    pca = IncrementalPCA(n_components=actual, batch_size=4096)
    pca.fit(features)
    print(f"  [PCA] explained var: {pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(pca, path)
    return pca


def train_and_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_true: np.ndarray,
    name: str, test_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """训练 HistGradientBoosting 并评估"""
    print(f"  [{name}] dim={X_train.shape[1]}, train={len(X_train)}, test={len(X_test)}")

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05,
        max_depth=6, l2_regularization=0.3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[opt_idx]
    y_pred = (y_prob >= opt_thresh).astype(int)

    result = {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "optimal_threshold": float(opt_thresh),
        "train_time": float(train_time),
        "feature_dim": int(X_train.shape[1]),
    }

    if test_df is not None and "ic50_nm" in test_df.columns:
        log_ic50 = np.log10(np.maximum(test_df["ic50_nm"].values, 1.0))
        corr, pval = stats.pearsonr(y_prob, log_ic50)
        result["ic50_corr"] = float(corr)
        result["ic50_pval"] = float(pval)

    print(f"  [{name}] AUC={auc:.4f} Acc={result['accuracy']:.4f} F1={result['f1']:.4f} MCC={result['mcc']:.4f} t={train_time:.1f}s")
    return result


def build_traditional_features(df: pd.DataFrame) -> np.ndarray:
    """构建传统特征 (Mamba3Lite + k-mer + biochem + env)"""
    spec = FeatureSpec(use_esm2=False, use_mhc=False)
    X, _, _ = build_feature_matrix(ensure_columns(df), spec)
    return X


def build_mhc_features(df: pd.DataFrame) -> np.ndarray:
    """构建 MHC 特征 (979D)"""
    encoder = MHCFeatureEncoder()
    sequences = df["epitope_seq"].astype(str).tolist()
    if "mhc_allele" in df.columns:
        alleles = df["mhc_allele"].fillna("HLA-A*02:01").astype(str).tolist()
    else:
        alleles = ["HLA-A*02:01"] * len(df)
    return encoder.encode_batch(sequences, alleles)


def main():
    print("=" * 60)
    print(f"ESM-2 {ESM2_SIZE} + MHC 联合基准测试")
    print("=" * 60)

    # 加载数据
    train_path = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
    train_full = pd.read_csv(train_path)
    print(f"训练数据: {len(train_full)} rows, {train_full['epitope_seq'].nunique()} unique seqs")

    # 两个 benchmark
    bm_iedb = load_benchmark("iedb_heldout_mhc.csv")
    bm_netmhc = load_benchmark("netmhcpan_heldout.csv")

    results = {}

    # ---- 对每个 benchmark 独立训练和评估 ----
    for bm_name, bm_df in [("iedb_2166", bm_iedb), ("netmhc_61", bm_netmhc)]:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bm_name} (N={len(bm_df)})")
        print(f"{'='*60}")

        # 排除 benchmark 序列
        exclude = set(bm_df["epitope_seq"].str.upper())
        train = train_full[~train_full["epitope_seq"].str.upper().isin(exclude)].reset_index(drop=True)
        train["label"] = (train["efficacy"] >= 3.0).astype(int)
        print(f"训练集: {len(train)} rows ({train['label'].mean():.1%} binders)")

        # 确保 benchmark 有 mhc_allele
        if "mhc_allele" not in bm_df.columns:
            bm_df["mhc_allele"] = "HLA-A*02:01"
        if "is_binder" not in bm_df.columns:
            bm_df["is_binder"] = (bm_df.get("ic50_nm", 500) < 500).astype(int)

        y_train = train["label"].values
        y_true = bm_df["is_binder"].values.astype(int)

        # === 预计算所有特征 ===
        print("\n[1/4] 构建传统特征...")
        X_trad_train = build_traditional_features(train)
        X_trad_test = build_traditional_features(bm_df)
        print(f"  传统特征: {X_trad_train.shape[1]}D")

        print("\n[2/4] 构建 MHC 特征...")
        X_mhc_train = build_mhc_features(train)
        X_mhc_test = build_mhc_features(bm_df)
        print(f"  MHC 特征: {X_mhc_train.shape[1]}D")

        print(f"\n[3/4] 构建 ESM-2 {ESM2_SIZE} 特征...")
        t0 = time.time()
        encoder = get_esm2_encoder()
        train_seqs = train["epitope_seq"].astype(str).tolist()
        test_seqs = bm_df["epitope_seq"].astype(str).tolist()
        X_esm2_test = encoder.encode(test_seqs)
        print(f"  Test ESM-2: {X_esm2_test.shape}, {time.time()-t0:.1f}s")
        X_esm2_train = encoder.encode(train_seqs)
        print(f"  Train ESM-2: {X_esm2_train.shape}, {time.time()-t0:.1f}s total")

        # PCA 缓存
        pca_dim_32 = min(32, X_esm2_train.shape[1], X_esm2_train.shape[0])
        pca_dim_64 = min(64, X_esm2_train.shape[1], X_esm2_train.shape[0])

        pca32 = fit_pca(X_esm2_train, 32, str(CACHE_DIR / f"pca_{ESM2_SIZE}_32d.joblib"))
        pca64 = fit_pca(X_esm2_train, 64, str(CACHE_DIR / f"pca_{ESM2_SIZE}_64d.joblib"))

        X_esm2_pca32_train = pca32.transform(X_esm2_train).astype(np.float32)
        X_esm2_pca32_test = pca32.transform(X_esm2_test).astype(np.float32)
        X_esm2_pca64_train = pca64.transform(X_esm2_train).astype(np.float32)
        X_esm2_pca64_test = pca64.transform(X_esm2_test).astype(np.float32)

        # === 实验 ===
        print(f"\n[4/4] 训练与评估...")

        # A: 传统特征
        results[f"{bm_name}_A_trad"] = train_and_eval(
            X_trad_train, y_train, X_trad_test, y_true,
            f"{bm_name}/A_trad", bm_df)

        # B: 传统 + MHC (当前最优)
        Xb_train = np.concatenate([X_trad_train, X_mhc_train], axis=1)
        Xb_test = np.concatenate([X_trad_test, X_mhc_test], axis=1)
        results[f"{bm_name}_B_trad_mhc"] = train_and_eval(
            Xb_train, y_train, Xb_test, y_true,
            f"{bm_name}/B_trad+mhc", bm_df)

        # C: 传统 + MHC + ESM-2 全量
        Xc_train = np.concatenate([X_trad_train, X_mhc_train, X_esm2_train], axis=1)
        Xc_test = np.concatenate([X_trad_test, X_mhc_test, X_esm2_test], axis=1)
        results[f"{bm_name}_C_trad_mhc_esm2"] = train_and_eval(
            Xc_train, y_train, Xc_test, y_true,
            f"{bm_name}/C_trad+mhc+esm2", bm_df)

        # D: 传统 + MHC + ESM-2 PCA 32D
        Xd_train = np.concatenate([X_trad_train, X_mhc_train, X_esm2_pca32_train], axis=1)
        Xd_test = np.concatenate([X_trad_test, X_mhc_test, X_esm2_pca32_test], axis=1)
        results[f"{bm_name}_D_trad_mhc_esm2_32"] = train_and_eval(
            Xd_train, y_train, Xd_test, y_true,
            f"{bm_name}/D_trad+mhc+esm2_32", bm_df)

        # E: 传统 + MHC + ESM-2 PCA 64D
        Xe_train = np.concatenate([X_trad_train, X_mhc_train, X_esm2_pca64_train], axis=1)
        Xe_test = np.concatenate([X_trad_test, X_mhc_test, X_esm2_pca64_test], axis=1)
        results[f"{bm_name}_E_trad_mhc_esm2_64"] = train_and_eval(
            Xe_train, y_train, Xe_test, y_true,
            f"{bm_name}/E_trad+mhc+esm2_64", bm_df)

    # === 汇总 ===
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"{'实验':<40} {'AUC':>8} {'F1':>8} {'MCC':>8} {'Dim':>6}")
    print("-" * 70)
    for label, r in results.items():
        print(f"{label:<40} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} {r['feature_dim']:>6}")

    # 保存
    output = {
        "model": f"ESM-2 {ESM2_SIZE} + MHC 联合",
        "results": results,
        "baseline_ref": {"auc": 0.92, "model": "NetMHCpan-4.1"},
    }
    out_path = RESULTS_DIR / "esm2_8m_mhc_joint.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    main()
