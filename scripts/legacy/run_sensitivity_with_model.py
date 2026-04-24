#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.epitope.featurizer import SequenceFeatures
from src.epitope.sensitivity import format_sensitivity_report, sensitivity_from_bundle, sensitivity_report


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="模型路径（默认 models/epitope_model.joblib）")
    p.add_argument("--sequence", default="ACDEFGHIKLMNPQRSTVWY", help="要解释的序列")
    p.add_argument("--eps", type=float, default=1e-3, help="数值梯度扰动")
    p.add_argument("--batch", type=int, default=2048, help="数值梯度批大小")
    p.add_argument("--topk", type=int, default=10, help="报告 Top-K 特征")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    model_path = Path(args.model) if args.model else (Path(__file__).resolve().parents[1] / "models" / "epitope_model.joblib")
    print(f"Checking model path: {model_path}")
    if not model_path.exists():
        print(f"模型文件未找到: {model_path}")
        return

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    # Choose an example sequence (short synthetic) and use env medians
    seq = str(args.sequence)
    env = {c: float(v) for c, v in getattr(bundle, "env_medians", {}).items()}

    feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
    featurizer = SequenceFeatures(version=feat_v)
    seq_x = featurizer.transform_one(seq).astype(np.float32)

    env_cols = list(getattr(bundle, "env_cols", []))
    env_vec = [float(env.get(c, float(getattr(bundle, "env_medians", {}).get(c, 0.0)))) for c in env_cols]
    x = np.concatenate([seq_x, np.asarray(env_vec, dtype=np.float32)], axis=0).astype(np.float32)

    res = sensitivity_from_bundle(bundle, x=x, eps=float(args.eps), batch_size=int(args.batch))

    report = sensitivity_report(res, top_k=int(args.topk))
    print("=== 格式化报告 ===")
    print(format_sensitivity_report(report, top_k=int(args.topk)))


if __name__ == "__main__":
    main()
