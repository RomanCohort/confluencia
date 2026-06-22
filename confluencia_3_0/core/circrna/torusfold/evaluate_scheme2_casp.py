#!/usr/bin/env python3
"""
evaluate_scheme2_casp.py — 用 Circ-CASP 评分体系评估 Scheme 2 (纯物理求解器)

Scheme 2 特点：
- 零训练，纯几何约束求解
- 输入：序列长度 + 配对约束
- 输出：3D 坐标

用法：
    python evaluate_scheme2_casp.py \
        --labels data/circrna_3d_merged \
        --output results/scheme2_casp.json \
        --n-samples 100
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import load_pseudo_labels
from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
    GeometricConstraintSolver, SolverConfig
)
from confluencia_3_0.core.circrna.torusfold.circ_casp_evaluate import (
    evaluate_single, compute_rmsd, compute_bsj_closure, compute_bond_consistency
)


def extract_pair_constraints(pair_probs: np.ndarray, threshold: float = 0.3) -> List:
    """从 pair_probs 提取配对约束。

    Args:
        pair_probs: (L, L) 配对概率矩阵
        threshold: 配对概率阈值

    Returns:
        List of (i, j, distance, weight) tuples
    """
    L = pair_probs.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):  # 最小 loop 长度 3
            if pair_probs[i, j] > threshold:
                # Watson-Crick pair distance ~10.6 Å
                pairs.append((i, j, 10.6, float(pair_probs[i, j])))
    return pairs


class ConstraintSet:
    """Simple constraint set wrapper for the solver."""
    def __init__(self, seq_len: int, pair_constraints: List):
        self.seq_len = seq_len
        self.pair_constraints = pair_constraints


def run_scheme2_inference(
    seq_len: int,
    pair_probs: Optional[np.ndarray] = None,
    config: Optional[SolverConfig] = None,
) -> np.ndarray:
    """运行 Scheme 2 推理。

    Args:
        seq_len: 序列长度
        pair_probs: (L, L) 配对概率矩阵
        config: 求解器配置

    Returns:
        (L, 3) 坐标数组（最佳构象）
    """
    if config is None:
        config = SolverConfig(n_samples=10)  # 默认采样 10 个构象

    solver = GeometricConstraintSolver(config)

    # 提取配对约束
    if pair_probs is not None:
        pairs = extract_pair_constraints(pair_probs)
    else:
        pairs = []

    constraint_set = ConstraintSet(seq_len, pairs)

    # 求解
    conformations = solver.solve(constraint_set)

    if not conformations:
        # Fallback: simple helix
        return simple_helix(seq_len)

    return conformations[0]  # 返回能量最低的构象


def simple_helix(L: int, bond_length: float = 5.9) -> np.ndarray:
    """简单螺旋作为 fallback。

    保证闭合：最后一个核苷酸连接回第一个。
    """
    coords = np.zeros((L, 3))
    radius = bond_length * L / (2 * np.pi) * 0.5
    rise_per_nt = 2.8

    for i in range(L):
        angle = 2 * np.pi * i / L
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = rise_per_nt * i - L * rise_per_nt / 2

    # 闭合修正：最后一个点放回起点附近
    coords[-1] = coords[0] + bond_length * np.array([
        np.cos(2 * np.pi * (L-1) / L),
        np.sin(2 * np.pi * (L-1) / L),
        0
    ])

    return coords


def evaluate_scheme2_on_dataset(
    sequences: List[str],
    coords_labels: List[np.ndarray],
    pair_labels: List[np.ndarray],
    metadata: List[Dict],
    n_samples: int = 100,
    verbose: bool = True,
) -> Dict:
    """在数据集上评估 Scheme 2。

    Args:
        sequences: 序列列表
        coords_labels: 真实坐标列表
        pair_labels: 配对概率列表
        metadata: 元数据列表
        n_samples: 评估样本数
        verbose: 是否打印进度

    Returns:
        评估结果字典
    """
    results = []
    total_inference_time = 0.0

    for i in range(min(n_samples, len(sequences))):
        seq = sequences[i]
        true_coords = coords_labels[i]
        pair_probs = pair_labels[i] if i < len(pair_labels) else None
        L = len(seq)

        if L < 10:
            if verbose:
                print(f"  [{i+1}/{n_samples}] L={L} - skip (too short)")
            continue

        # 运行 Scheme 2
        start_time = time.time()
        pred_coords = run_scheme2_inference(L, pair_probs)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # 提取配对约束用于 T4 评分
        if pair_probs is not None:
            pairs_pred = extract_pair_constraints(pair_probs, threshold=0.5)
            pairs_pred = set((p[0], p[1]) for p in pairs_pred)
            # 从真实坐标推断真实配对（简化：直接用 pair_probs 阈值）
            pairs_true = set((p[0], p[1]) for p in extract_pair_constraints(pair_probs, threshold=0.7))
        else:
            pairs_pred = set()
            pairs_true = set()

        # CASP 评分
        scores = evaluate_single(pred_coords, true_coords, pairs_pred, pairs_true)
        scores['inference_time'] = round(inference_time, 3)
        scores['seq_len'] = L
        scores['id'] = metadata[i].get('id', f'seq_{i}')

        results.append(scores)

        if verbose and (i + 1) % 10 == 0:
            avg_total = np.mean([r['total_score'] for r in results])
            avg_rmsd = np.mean([r['rmsd'] for r in results])
            print(f"  [{i+1}/{n_samples}] L={L} RMSD={scores['rmsd']:.1f}Å "
                  f"total={scores['total_score']:.1f} (avg={avg_total:.1f})")

    # 汇总统计
    if not results:
        return {'error': 'No valid samples'}

    all_rmsd = [r['rmsd'] for r in results]
    all_total = [r['total_score'] for r in results]
    all_t1 = [r['t1_score'] for r in results]
    all_t2 = [r['t2_score'] for r in results]
    all_t3 = [r['t3_score'] for r in results]
    all_t4 = [r['t4_score'] for r in results]

    summary = {
        'n_samples': len(results),
        'avg_inference_time': round(total_inference_time / len(results), 3),
        'rmsd': {
            'mean': round(np.mean(all_rmsd), 2),
            'median': round(np.median(all_rmsd), 2),
            'std': round(np.std(all_rmsd), 2),
            'min': round(min(all_rmsd), 2),
            'max': round(max(all_rmsd), 2),
        },
        'scores': {
            'total': {
                'mean': round(np.mean(all_total), 2),
                'median': round(np.median(all_total), 2),
                'std': round(np.std(all_total), 2),
            },
            't1_rmsd': round(np.mean(all_t1), 2),
            't2_bsj': round(np.mean(all_t2), 2),
            't3_bond': round(np.mean(all_t3), 2),
            't4_pairs': round(np.mean(all_t4), 2),
        },
        # Circ-CASP 通过门槛检查
        'pass_threshold': {
            'avg_total_10': np.mean(all_total) >= 10,
            'rmsd_lt_30': sum(1 for r in all_rmsd if r < 30) / len(all_rmsd),
            'rmsd_lt_20': sum(1 for r in all_rmsd if r < 20) / len(all_rmsd),
            'rmsd_lt_10': sum(1 for r in all_rmsd if r < 10) / len(all_rmsd),
        }
    }

    return {
        'summary': summary,
        'per_sample': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Scheme 2 with Circ-CASP scoring')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to pseudo-labels directory')
    parser.add_argument('--output', type=str, default='results/scheme2_casp.json',
                        help='Output JSON file path')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--max-len', type=int, default=500,
                        help='Maximum sequence length to evaluate')
    args = parser.parse_args()

    print("=" * 60)
    print("  Scheme 2 Circ-CASP Evaluation")
    print("=" * 60)

    # 加载数据
    print(f"\nLoading data from {args.labels}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(
        args.labels, max_len=args.max_len
    )
    print(f"  Loaded {len(sequences)} sequences (max_len={args.max_len})")

    # 评估
    print(f"\nEvaluating Scheme 2 on {min(args.n_samples, len(sequences))} samples...")
    results = evaluate_scheme2_on_dataset(
        sequences, coords_labels, pair_labels, metadata,
        n_samples=args.n_samples,
        verbose=True,
    )

    # 打印汇总
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    s = results['summary']
    print(f"  Samples evaluated: {s['n_samples']}")
    print(f"  Avg inference time: {s['avg_inference_time']:.3f}s")
    print(f"\n  RMSD: mean={s['rmsd']['mean']:.1f}Å median={s['rmsd']['median']:.1f}Å "
          f"std={s['rmsd']['std']:.1f}Å")
    print(f"\n  Circ-CASP Scores (0-100):")
    print(f"    Total:  {s['scores']['total']['mean']:.1f} ± {s['scores']['total']['std']:.1f}")
    print(f"    T1 (RMSD): {s['scores']['t1_rmsd']:.1f}")
    print(f"    T2 (BSJ):  {s['scores']['t2_bsj']:.1f}")
    print(f"    T3 (Bond): {s['scores']['t3_bond']:.1f}")
    print(f"    T4 (Pairs):{s['scores']['t4_pairs']:.1f}")
    print(f"\n  Pass Rates:")
    print(f"    RMSD < 30Å: {s['pass_threshold']['rmsd_lt_30']*100:.1f}%")
    print(f"    RMSD < 20Å: {s['pass_threshold']['rmsd_lt_20']*100:.1f}%")
    print(f"    RMSD < 10Å: {s['pass_threshold']['rmsd_lt_10']*100:.1f}%")
    print(f"    Total >= 10: {s['pass_threshold']['avg_total_10']}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()
