#!/usr/bin/env python
"""
test_models.py — Test trained circRNA prediction models.

Usage:
    python confluencia_circrna/encoder/test_models.py
"""

import joblib
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("Testing circRNA Prediction Models")
    print("=" * 60)

    # 尝试多个可能的路径
    possible_paths = [
        Path(__file__).resolve().parents[2] / "data" / "models",  # confluencia_circrna/data/models
        Path(__file__).resolve().parents[3] / "confluencia_circrna" / "data" / "models",  # 项目根/confluencia_circrna/data/models
        Path("/root/autodl-tmp/confluencia_circrna/data/models"),  # AutoDL绝对路径
        Path.cwd() / "confluencia_circrna" / "data" / "models",
    ]

    model_dir = None
    for p in possible_paths:
        if p.exists() and (p / "ips_predictor.pkl").exists():
            model_dir = p
            break

    if model_dir is None:
        print("ERROR: Model files not found. Please train models first:")
        print("  python confluencia_circrna/encoder/train.py")
        return

    # 测试IPS模型
    model = joblib.load(model_dir / "ips_predictor.pkl")
    scaler = joblib.load(model_dir / "ips_scaler.pkl")

    # 测试用例 (TROP2, NECTIN4, LIV-1, B7-H4, MKI67, MYC, seq_length)
    test_cases = [
        [0.8, 0.6, 0.4, 0.7, 0.5, 0.5, 500],   # 高表达
        [0.3, 0.3, 0.3, 0.3, 0.8, 0.6, 2000],  # 低表达
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1000],  # 中等
    ]

    print("\nIPS Predictions:")
    for i, test_X in enumerate(test_cases):
        test_scaled = scaler.transform(np.array([test_X]))
        ips_pred = model.predict(test_scaled)
        print(f"  Test {i+1}: IPS={ips_pred[0]:.2f}")

    print("\n" + "=" * 60)
    print("Test Complete!")

if __name__ == "__main__":
    main()