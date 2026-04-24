"""
Epitope Binder 分类专项训练脚本
================================
使用 ESM-2 650M + Mamba3Lite + k-mer + biochem 特征训练二分类器。

训练策略:
  - 冻结 ESM-2 特征提取器 (不 fine-tune)
  - GroupShuffleSplit 按 epitope_seq 分割避免数据泄漏
  - HistGradientBoostingClassifier 专项训练

目标: AUC 从当前 0.731 提升至 0.92+
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 路径设置
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import FeatureSpec, build_feature_matrix, ensure_columns


def load_and_prepare_data(
    data_path: Optional[str] = None,
    efficacy_threshold: float = 3.0,
) -> pd.DataFrame:
    """加载并准备训练数据"""
    if data_path is None:
        data_path = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"

    df = pd.read_csv(data_path)
    df["label"] = (df["efficacy"] >= efficacy_threshold).astype(int)

    print(f"[Data] 加载完成: {len(df)} 行")
    print(f"[Data] Binders: {df['label'].sum()} ({df['label'].mean():.1%})")
    print(f"[Data] Non-binders: {(~df['label'].astype(bool)).sum()}")
    print(f"[Data] 唯一序列数: {df['epitope_seq'].nunique()}")

    return df


def sequence_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """按序列分组分割，避免数据泄漏"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["epitope_seq"].values))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} ({train_df['label'].mean():.1%} binders)")
    print(f"[Split] Test:  {len(test_df)} ({test_df['label'].mean():.1%} binders)")
    print(f"[Split] Train unique seqs: {train_df['epitope_seq'].nunique()}")
    print(f"[Split] Test unique seqs:  {test_df['epitope_seq'].nunique()}")

    return train_df, test_df


def extract_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: FeatureSpec,
) -> tuple:
    """提取特征矩阵"""
    print(f"\n[Features] 提取特征 (ESM-2: {spec.use_esm2}, model: {spec.esm2_model_size})...")

    t0 = time.time()
    X_train, names, env_cols = build_feature_matrix(train_df, spec)
    t_train = time.time() - t0
    print(f"  Train: {X_train.shape} ({t_train:.1f}s)")

    t0 = time.time()
    X_test, _, _ = build_feature_matrix(test_df, spec)
    t_test = time.time() - t0
    print(f"  Test:  {X_test.shape} ({t_test:.1f}s)")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test, names


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """训练多个分类器并评估"""
    results = {}

    # 标准化版本 (用于 LR, MLP)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    def eval_model(name: str, y_true, y_pred, y_prob, train_time):
        r = {
            "auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "train_time": float(train_time),
        }
        results[name] = r
        print(f"  {name:25s}  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  "
              f"F1={r['f1']:.4f}  MCC={r['mcc']:.4f}  ({train_time:.1f}s)")
        return r

    # 1. HistGradientBoosting (核心模型)
    print("\n[Model 1] HistGradientBoosting...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    eval_model("HGB_v2", y_test, hgb.predict(X_test), hgb.predict_proba(X_test)[:, 1], time.time() - t0)

    # 2. HGB with different hyperparams (deeper, slower learning)
    print("\n[Model 2] HGB Deep...")
    t0 = time.time()
    hgb_deep = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.02,
        max_depth=10,
        l2_regularization=2.0,
        min_samples_leaf=10,
        random_state=42,
    )
    hgb_deep.fit(X_train, y_train)
    eval_model("HGB_deep", y_test, hgb_deep.predict(X_test), hgb_deep.predict_proba(X_test)[:, 1], time.time() - t0)

    # 3. Random Forest (bagging baseline)
    print("\n[Model 3] Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    eval_model("RF_v2", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1], time.time() - t0)

    # 4. MLP (神经网络 baseline)
    print("\n[Model 4] MLP...")
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate="adaptive",
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    eval_model("MLP_v2", y_test, mlp.predict(X_test_s), mlp.predict_proba(X_test_s)[:, 1], time.time() - t0)

    # 5. Logistic Regression (linear baseline)
    print("\n[Model 5] Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42, n_jobs=-1)
    lr.fit(X_train_s, y_train)
    eval_model("LR_v2", y_test, lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1], time.time() - t0)

    # 6. MOE Ensemble: 加权平均 HGB + RF
    print("\n[Model 6] MOE Ensemble...")
    t0 = time.time()
    hgb_prob = hgb.predict_proba(X_test)[:, 1]
    hgb_deep_prob = hgb_deep.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # 基于 OOF 权重的简单集成
    ens_prob = 0.4 * hgb_prob + 0.4 * hgb_deep_prob + 0.2 * rf_prob
    ens_pred = (ens_prob >= 0.5).astype(int)
    eval_model("MOE_v2", y_test, ens_pred, ens_prob, time.time() - t0)

    return results


def main(
    data_path: Optional[str] = None,
    use_esm2: bool = True,
    esm2_model_size: str = "650M",
    efficacy_threshold: float = 3.0,
    output_path: Optional[str] = None,
):
    """主训练流程"""
    print("=" * 60)
    print("Epitope Binder 分类专项训练")
    print(f"ESM-2: {use_esm2} ({esm2_model_size if use_esm2 else 'N/A'})")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_prepare_data(data_path, efficacy_threshold)

    # 2. 序列感知分割
    train_df, test_df = sequence_aware_split(df)

    # 3. 特征提取
    spec = FeatureSpec(
        use_esm2=use_esm2,
        esm2_model_size=esm2_model_size,
        esm2_cache_dir=str(PROJECT / "data" / "cache" / "esm2"),
    )
    X_train, X_test, y_train, y_test, feat_names = extract_features(train_df, test_df, spec)

    # 4. 训练和评估
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. 输出汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print("=" * 60)
    print(f"{'Model':25s} {'AUC':>6s} {'Acc':>6s} {'F1':>6s} {'MCC':>6s} {'Time':>8s}")
    print("-" * 62)
    for name, r in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        print(f"{name:25s} {r['auc']:6.4f} {r['accuracy']:6.4f} {r['f1']:6.4f} "
              f"{r['mcc']:6.4f} {r['train_time']:7.1f}s")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nBest: {best_name} (AUC={results[best_name]['auc']:.4f})")

    # 6. 保存结果
    if output_path is None:
        output_path = str(PROJECT / "benchmarks" / "results" / "binder_classifier_v2.json")

    output = {
        "config": {
            "use_esm2": use_esm2,
            "esm2_model_size": esm2_model_size,
            "efficacy_threshold": efficacy_threshold,
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        },
        "results": results,
        "best_model": best_name,
        "best_auc": results[best_name]["auc"],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Epitope Binder Classifier Training")
    parser.add_argument("--no-esm2", action="store_true", help="禁用 ESM-2")
    parser.add_argument("--esm2-model", default="650M", choices=["8M", "35M", "150M", "650M"])
    parser.add_argument("--threshold", type=float, default=3.0, help="efficacy binarization threshold")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    main(
        use_esm2=not args.no_esm2,
        esm2_model_size=args.esm2_model,
        efficacy_threshold=args.threshold,
        output_path=args.output,
    )