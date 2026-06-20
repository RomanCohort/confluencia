#!/usr/bin/env python3
"""
circ_casp_evaluate.py — Circ-CASP 竞赛评分脚本。

用法：
    python circ_casp_evaluate.py \
        --predictions submissions/team_A/ \
        --ground-truth data/circ_casp_test/ \
        --output results/team_A_scores.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def compute_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """计算 RMSD（先做最优对齐）。"""
    # Center both
    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)

    # Kabsch alignment
    H = true_c.T @ pred_c
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])

    R = Vt.T @ D @ U.T
    pred_aligned = (R @ pred_c.T).T

    rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true_c) ** 2, axis=1)))
    return rmsd


def compute_bsj_closure(coords: np.ndarray) -> float:
    """计算 BSJ 闭合距离。"""
    return float(np.linalg.norm(coords[0] - coords[-1]))


def compute_bond_consistency(coords: np.ndarray, circular: bool = True) -> Tuple[float, float]:
    """计算骨架距离一致性。

    Returns:
        (mean_distance, std_distance)
    """
    if circular:
        # 包含 BSJ 连接
        next_indices = np.roll(np.arange(len(coords)), -1)
        bonds = np.linalg.norm(coords[next_indices] - coords, axis=1)
    else:
        bonds = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

    return float(bonds.mean()), float(bonds.std())


def score_t1_rmsd(rmsd: float) -> float:
    """T1: 整体结构评分。"""
    if rmsd < 5:
        return 100.0
    elif rmsd < 10:
        return 80.0 + 20.0 * (10 - rmsd) / 5.0
    elif rmsd < 15:
        return 60.0 + 20.0 * (15 - rmsd) / 5.0
    elif rmsd < 20:
        return 40.0 + 20.0 * (20 - rmsd) / 5.0
    elif rmsd < 30:
        return 20.0 + 20.0 * (30 - rmsd) / 10.0
    else:
        return max(0.0, 20.0 * max(0, 50 - rmsd) / 20.0)


def score_t2_bsj(bsj_pred: float, bsj_true: float) -> float:
    """T2: BSJ 闭合评分。"""
    error = abs(bsj_pred - bsj_true)
    if error < 1:
        return 100.0
    elif error < 2:
        return 80.0 + 20.0 * (2 - error)
    elif error < 5:
        return 60.0 + 20.0 * (5 - error) / 3.0
    elif error < 10:
        return 40.0 + 20.0 * (10 - error) / 5.0
    else:
        return max(0.0, 40.0 - 4.0 * (error - 10))


def score_t3_bond(bond_error: float) -> float:
    """T3: 骨架一致性评分。"""
    return max(0.0, 100.0 - 20.0 * bond_error)


def score_t4_pairs(pairs_pred: set, pairs_true: set) -> float:
    """T4: 二级结构配对 F1。"""
    if not pairs_true and not pairs_pred:
        return 100.0
    if not pairs_true:
        return 0.0

    tp = len(pairs_pred & pairs_true)
    fp = len(pairs_pred - pairs_true)
    fn = len(pairs_true - pairs_pred)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return f1 * 100.0


def evaluate_single(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    pairs_pred: Optional[set] = None,
    pairs_true: Optional[set] = None,
) -> Dict:
    """评估单个 circRNA 预测。"""

    # T1: RMSD
    rmsd = compute_rmsd(pred_coords, true_coords)
    t1_score = score_t1_rmsd(rmsd)

    # T2: BSJ closure
    bsj_pred = compute_bsj_closure(pred_coords)
    bsj_true = compute_bsj_closure(true_coords)
    t2_score = score_t2_bsj(bsj_pred, bsj_true)

    # T3: Bond consistency
    bond_mean_pred, bond_std_pred = compute_bond_consistency(pred_coords, circular=True)
    bond_mean_true, bond_std_true = compute_bond_consistency(true_coords, circular=True)
    bond_error = abs(bond_mean_pred - bond_mean_true)
    t3_score = score_t3_bond(bond_error)

    # T4: Pair F1
    t4_score = 0.0
    if pairs_pred is not None and pairs_true is not None:
        t4_score = score_t4_pairs(pairs_pred, pairs_true)

    # Total (T5 暂不计入)
    weights = {'t1': 0.4, 't2': 0.2, 't3': 0.15, 't4': 0.15, 't5': 0.1}
    total = (weights['t1'] * t1_score +
             weights['t2'] * t2_score +
             weights['t3'] * t3_score +
             weights['t4'] * t4_score)

    return {
        'rmsd': round(rmsd, 2),
        'bsj_pred': round(bsj_pred, 2),
        'bsj_true': round(bsj_true, 2),
        'bsj_error': round(abs(bsj_pred - bsj_true), 2),
        'bond_mean_pred': round(bond_mean_pred, 2),
        'bond_mean_true': round(bond_mean_true, 2),
        'bond_error': round(bond_error, 2),
        't1_score': round(t1_score, 2),
        't2_score': round(t2_score, 2),
        't3_score': round(t3_score, 2),
        't4_score': round(t4_score, 2),
        'total_score': round(total, 2),
    }


def evaluate_submission(
    pred_dir: str,
    truth_dir: str,
) -> Dict:
    """评估整个提交。"""

    # Load ground truth
    with open(os.path.join(truth_dir, 'sequences.json'), 'r') as f:
        truth_seqs = json.load(f)

    results = {}
    all_scores = []

    for item in truth_seqs:
        seq_id = item['id']

        # Load prediction
        pred_path = os.path.join(pred_dir, f"{seq_id}_coords.npy")
        if not os.path.exists(pred_path):
            print(f"  WARNING: Missing prediction for {seq_id}")
            results[seq_id] = {'error': 'missing', 'total_score': 0.0}
            all_scores.append(0.0)
            continue

        pred_coords = np.load(pred_path)

        # Load ground truth
        truth_path = os.path.join(truth_dir, 'coords', f"{seq_id}.npy")
        true_coords = np.load(truth_path)

        # Load pairs (optional)
        pairs_pred = None
        pairs_true = None
        pairs_pred_path = os.path.join(pred_dir, f"{seq_id}_pairs.json")
        pairs_true_path = os.path.join(truth_dir, 'pairs', f"{seq_id}.json")

        if os.path.exists(pairs_pred_path) and os.path.exists(pairs_true_path):
            with open(pairs_pred_path) as f:
                pairs_pred = set(map(tuple, json.load(f)))
            with open(pairs_true_path) as f:
                pairs_true = set(map(tuple, json.load(f)))

        # Evaluate
        result = evaluate_single(pred_coords, true_coords, pairs_pred, pairs_true)
        results[seq_id] = result
        all_scores.append(result['total_score'])

        print(f"  {seq_id}: RMSD={result['rmsd']:.1f}Å "
              f"BSJ_err={result['bsj_error']:.1f}Å "
              f"Bond_err={result['bond_error']:.1f}Å "
              f"Score={result['total_score']:.1f}")

    # Summary
    summary = {
        'per_target': results,
        'mean_score': round(np.mean(all_scores), 2),
        'median_score': round(np.median(all_scores), 2),
        'std_score': round(np.std(all_scores), 2),
        'n_targets': len(results),
        'n_evaluated': sum(1 for r in results.values() if 'error' not in r),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Circ-CASP Evaluation')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Directory with prediction files')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='Directory with ground truth files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for scores')
    args = parser.parse_args()

    print("=" * 60)
    print("  Circ-CASP Evaluation")
    print("=" * 60)
    print(f"  Predictions: {args.predictions}")
    print(f"  Ground truth: {args.ground_truth}")

    results = evaluate_submission(args.predictions, args.ground_truth)

    print(f"\n{'='*60}")
    print(f"  Mean Score: {results['mean_score']:.2f}")
    print(f"  Median Score: {results['median_score']:.2f}")
    print(f"  Std Score: {results['std_score']:.2f}")
    print(f"  Targets Evaluated: {results['n_evaluated']}/{results['n_targets']}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
