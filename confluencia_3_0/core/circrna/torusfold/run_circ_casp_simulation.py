#!/usr/bin/env python3
"""
run_circ_casp_simulation.py — Complete Circ-CASP simulation pipeline.

Simulates CASP-style blind testing:
1. Split merged data into train/test (30 targets held out)
2. Train all 7 schemes on training data
3. Run predictions on test set
4. Score using Circ-CASP metrics
5. Generate leaderboard

Usage:
    python run_circ_casp_simulation.py \
        --merged data/circrna_3d_merged \
        --output results/circ_casp_simulation \
        --device cuda \
        --epochs 30
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

import torch


def split_train_test(sequences: list, coords_labels: list, pair_labels: list,
                     confidence_weights: list, metadata: list,
                     n_test: int = 30, seed: int = 2026) -> Dict:
    """Split merged data into train and test sets.

    Test set is held out (blind) and stratified by length.
    """
    rng = np.random.RandomState(seed)
    N = len(sequences)

    # Stratified sampling by length
    lengths = [m['length'] for m in metadata]
    length_bins = [(50, 100), (100, 200), (200, 300), (300, 500), (500, 1000)]

    test_indices = []
    per_bin = max(1, n_test // len(length_bins))

    for lo, hi in length_bins:
        candidates = [i for i, l in enumerate(lengths) if lo <= l < hi]
        if candidates:
            n_sample = min(per_bin, len(candidates))
            selected = rng.choice(candidates, n_sample, replace=False).tolist()
            test_indices.extend(selected)

    # Fill remaining
    remaining = n_test - len(test_indices)
    if remaining > 0:
        all_candidates = [i for i in range(N) if i not in test_indices]
        if all_candidates:
            extra = rng.choice(all_candidates, min(remaining, len(all_candidates)), replace=False).tolist()
            test_indices.extend(extra)

    test_indices = sorted(test_indices)
    train_indices = [i for i in range(N) if i not in test_indices]

    # Build test set
    test_set = {
        'indices': test_indices,
        'sequences': [sequences[i] for i in test_indices],
        'coords': [coords_labels[i] for i in test_indices],
        'pair_probs': [pair_labels[i] for i in test_indices],
        'metadata': [metadata[i] for i in test_indices],
    }

    # Build train set (for actual training)
    train_set = {
        'indices': train_indices,
        'sequences': [sequences[i] for i in train_indices],
        'coords': [coords_labels[i] for i in train_indices],
        'pair_probs': [pair_labels[i] for i in train_indices],
        'confidence': [confidence_weights[i] for i in train_indices],
        'metadata': [metadata[i] for i in train_indices],
    }

    print(f"  Total data: {N}")
    print(f"  Train set: {len(train_indices)}")
    print(f"  Test set (blind): {len(test_indices)}")
    test_lengths = [m['length'] for m in test_set['metadata']]
    print(f"  Test lengths: {min(test_lengths)}-{max(test_lengths)} nt, mean {np.mean(test_lengths):.0f}")

    return {'train': train_set, 'test': test_set}


def save_test_set_as_ground_truth(test_set: Dict, output_dir: str):
    """Save test set in Circ-CASP ground truth format."""
    truth_dir = os.path.join(output_dir, 'ground_truth')
    coords_dir = os.path.join(truth_dir, 'coords')
    pairs_dir = os.path.join(truth_dir, 'pairs')
    os.makedirs(coords_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)

    # Save sequences.json
    seq_entries = []
    for i, (seq, coords, pp, meta) in enumerate(zip(
        test_set['sequences'],
        test_set['coords'],
        test_set['pair_probs'],
        test_set['metadata']
    )):
        seq_id = f"circ_{i:03d}"
        L = len(seq)

        # Save coords
        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)

        # Extract pairs from pair_probs
        pairs = []
        for p in range(L):
            for q in range(p + 1, L):
                if pp[p, q] > 0.5:
                    pairs.append([p, q])

        # Save pairs
        with open(os.path.join(pairs_dir, f"{seq_id}.json"), 'w') as f:
            json.dump(pairs, f)

        seq_entries.append({
            'id': seq_id,
            'sequence': seq,
            'length': L,
            'source': meta.get('source', 'unknown'),
        })

    with open(os.path.join(truth_dir, 'sequences.json'), 'w') as f:
        json.dump(seq_entries, f, indent=2)

    print(f"  Ground truth saved to {truth_dir}")


def main():
    parser = argparse.ArgumentParser(description="Circ-CASP simulation pipeline")
    parser.add_argument('--merged', type=str, required=True,
                        help='Merged dataset directory')
    parser.add_argument('--output', type=str, default='results/circ_casp_simulation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-test', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    print("=" * 60)
    print("  Circ-CASP Simulation Pipeline")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # Load merged data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_all_schemes import load_pseudo_labels

    print(f"\n[1/4] Loading merged data from {args.merged}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = \
        load_pseudo_labels(args.merged)

    # Split train/test
    print(f"\n[2/4] Splitting train/test ({args.n_test} test targets)...")
    split = split_train_test(
        sequences, coords_labels, pair_labels,
        confidence_weights, metadata,
        n_test=args.n_test, seed=args.seed
    )

    # Save test set as ground truth (blind)
    print(f"\n  Saving test set as ground truth...")
    save_test_set_as_ground_truth(split['test'], args.output)

    # Save train set for training
    train_dir = os.path.join(args.output, 'train_data')
    coords_dir = os.path.join(train_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    train_entries = []
    for i, (seq, coords, pp, conf, meta) in enumerate(zip(
        split['train']['sequences'],
        split['train']['coords'],
        split['train']['pair_probs'],
        split['train']['confidence'],
        split['train']['metadata']
    )):
        seq_id = f"train_{i:05d}"
        L = len(seq)

        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)

        pairs = []
        for p in range(L):
            for q in range(p + 1, L):
                if pp[p, q] > 0.5:
                    pairs.append([p, q])

        train_entries.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': '.' * L,  # Placeholder
            'pair_constraints': pairs,
            'length': L,
            'source': meta.get('source', 'unknown'),
            'confidence': conf,
        })

    with open(os.path.join(train_dir, 'sequences.json'), 'w') as f:
        json.dump(train_entries, f, indent=2)

    print(f"  Train data saved to {train_dir}")

    # Train all schemes
    print(f"\n[3/4] Training all 7 schemes on training data...")
    print(f"  (This step is typically done separately)")
    print(f"  To train, run:")
    print(f"    python train_all_schemes.py --labels {train_dir} --output {args.output}/models --device {args.device} --epochs {args.epochs}")

    # Evaluate
    print(f"\n[4/4] Evaluation...")
    print(f"  After training completes, run:")
    print(f"    python circ_casp_evaluate.py --predictions {args.output}/models --ground-truth {args.output}/ground_truth --output {args.output}/scores.json")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Setup complete. Train and evaluate separately.")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Return instructions
    print(f"\n  Next steps:")
    print(f"    1. Train: python train_all_schemes.py --labels {train_dir} --output {args.output}/models --device {args.device} --epochs {args.epochs}")
    print(f"    2. Eval: python circ_casp_evaluate.py --predictions {args.output}/predictions --ground-truth {args.output}/ground_truth --output {args.output}/scores.json")


if __name__ == "__main__":
    main()