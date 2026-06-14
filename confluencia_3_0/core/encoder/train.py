#!/usr/bin/env python
"""
train.py — Train circRNA immunogenicity prediction models.

Usage:
    python confluencia_circrna/encoder/train.py \
        --data confluencia_circrna/data/training/circrna_training_pairs_v3.csv \
        --output confluencia_circrna/data/models \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# 项目路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]


def load_training_data(data_path: str) -> pd.DataFrame:
    """加载训练数据."""
    path = Path(data_path)
    if not path.exists():
        path = _PROJECT_ROOT / data_path
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(path)
    print(f"[1] Loaded {len(df)} training samples from {path}")
    return df


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """构建特征矩阵."""
    # 基因表达特征
    gene_cols = [c for c in df.columns if c.startswith('gene_')]

    # 序列特征
    seq_cols = ['seq_length'] if 'seq_length' in df.columns else []

    # 合并特征
    feature_cols = gene_cols + seq_cols

    X = df[feature_cols].fillna(0).values
    print(f"[2] Features: {len(feature_cols)} columns")

    return X, feature_cols


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'rf',
    epochs: int = 100,
) -> Tuple[any, StandardScaler]:
    """训练模型."""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 选择模型
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=epochs,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=epochs,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_scaled, y_train)

    return model, scaler


def evaluate_model(
    model: any,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """评估模型."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    return {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def main():
    parser = argparse.ArgumentParser(description="Train circRNA prediction models")
    parser.add_argument('--data', default='confluencia_circrna/data/training/circrna_training_pairs_v3.csv',
                       help='Training data path')
    parser.add_argument('--output', default='confluencia_circrna/data/models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of trees/epochs')
    parser.add_argument('--model-type', default='rf', choices=['rf', 'gbr'],
                       help='Model type: rf (RandomForest) or gbr (GradientBoosting)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("circRNA Immunogenicity Model Training")
    print("=" * 60)

    np.random.seed(args.seed)

    # 加载数据
    df = load_training_data(args.data)

    # 构建特征
    X, feature_cols = build_features(df)

    # 目标变量
    targets = {
        'ips': 'target_ips',
        'immunotherapy_score': 'target_immunotherapy_score',
        'overall_immunogenicity': 'target_overall_immunogenicity',
        'rig_i_score': 'target_rig_i_score',
        'tlr_score': 'target_tlr_score',
        'pkr_score': 'target_pkr_score',
    }

    # 输出目录
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir = _PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3] Training {len(targets)} models with {args.model_type}...")

    results = {}
    all_metrics = {}

    for target_name, target_col in targets.items():
        if target_col not in df.columns:
            print(f"  {target_name}: SKIP (column not found)")
            continue

        y = df[target_col].values

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )

        # 训练
        model, scaler = train_model(X_train, y_train, args.model_type, args.epochs)

        # 评估
        metrics = evaluate_model(model, scaler, X_test, y_test)

        # 保存
        model_path = output_dir / f'{target_name}_predictor.pkl'
        scaler_path = output_dir / f'{target_name}_scaler.pkl'

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        results[target_name] = {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
        }
        all_metrics[target_name] = metrics

        print(f"  {target_name}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")

    # 保存训练报告
    report = {
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'targets': list(targets.keys()),
        'model_type': args.model_type,
        'epochs': args.epochs,
        'metrics': all_metrics,
    }

    report_path = output_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[4] Models saved to {output_dir}")
    print(f"[5] Training report: {report_path}")

    # 总体统计
    avg_r2 = np.mean([m['r2'] for m in all_metrics.values()])
    print(f"\nAverage R²: {avg_r2:.3f}")

    print("=" * 60)
    print("Training Complete!")

    return results


if __name__ == "__main__":
    main()