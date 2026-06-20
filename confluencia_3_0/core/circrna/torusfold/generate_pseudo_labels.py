#!/usr/bin/env python3
"""
generate_pseudo_labels.py — Generate 3D pseudo-labels for TorusFold training.

Pipeline:
    circRNA sequence → ViennaRNA (secondary structure, circ mode)
    → Physics Solver (3D coordinates) → Save to disk

Output:
    pseudo_labels/
    ├── metadata.json          # Summary + per-sample info
    ├── sequences.json         # All sequences
    └── coords/
        ├── pseudo_0000.npy    # (L, 3) coordinate arrays
        ├── pseudo_0001.npy
        └── ...

Usage:
    python generate_pseudo_labels.py --n 1000 --output data/pseudo_labels
    python generate_pseudo_labels.py --n 500 --min-len 30 --max-len 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
    GeometricConstraintSolver, SolverConfig
)


def generate_pseudo_labels(n_seqs=1000, min_len=30, max_len=500, seed=42):
    """Generate 3D coordinate pseudo-labels using ViennaRNA + Physics Solver."""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    sequences = []
    coords_list = []
    structures = []  # dot-bracket
    pair_constraints_all = []
    metadata = []

    try:
        import RNA
        has_vienna = True
        print("  ViennaRNA: available (circ mode)")
    except ImportError:
        has_vienna = False
        print("  ViennaRNA: NOT available, using heuristic pairing")

    # Physics solver
    config = SolverConfig(
        n_samples=10,
        use_annealing_closure=True,
        bond_length=5.9,
        pair_distance=10.6,
    )
    solver = GeometricConstraintSolver(config)

    print(f"  Generating {n_seqs} pseudo-labels (L={min_len}-{max_len})...")
    t0 = time.time()

    for i in range(n_seqs):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, size=L))

        # ViennaRNA secondary structure (circ mode)
        pair_constraints = []
        ss = '.' * L
        mfe = 0.0

        if has_vienna:
            try:
                md = RNA.md()
                md.circ = True
                fc = RNA.fold_compound(seq, md)
                ss, mfe = fc.mfe()

                stack = []
                for pos, char in enumerate(ss):
                    if char == '(':
                        stack.append(pos)
                    elif char == ')' and stack:
                        j_pos = stack.pop()
                        pair_constraints.append((j_pos, pos, 10.6, 1.0))
            except Exception:
                pass

        if not pair_constraints:
            # Heuristic: complement pairing
            complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
            for j in range(L):
                for k in range(j + 4, min(j + 20, L)):
                    if complement.get(seq[j]) == seq[k] and rng.random() < 0.3:
                        pair_constraints.append((j, k, 10.6, 1.0))

        # Build constraint set
        class CS:
            def __init__(self, n, pairs):
                self.seq_len = n
                self.pair_constraints = pairs

        cs = CS(L, pair_constraints)
        conformations = solver.solve(cs)

        if conformations and len(conformations) > 0:
            best_coords = conformations[0]
            closure_err = abs(np.linalg.norm(best_coords[0] - best_coords[-1]) - 5.9)

            if closure_err < 2.0:
                sequences.append(seq)
                coords_list.append(best_coords)
                structures.append(ss)
                pair_constraints_all.append(pair_constraints)

                metadata.append({
                    'id': f'pseudo_{len(sequences)-1:04d}',
                    'length': L,
                    'n_pairs': len(pair_constraints),
                    'closure_error': float(closure_err),
                    'mfe': float(mfe),
                    'source': 'ViennaRNA+Physics' if has_vienna else 'Heuristic+Physics',
                })

                if len(sequences) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = len(sequences) / elapsed
                    print(f"    {len(sequences)}/{n_seqs} "
                          f"({rate:.1f} samples/s, "
                          f"elapsed={elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(sequences)}/{n_seqs} in {elapsed:.1f}s")

    return sequences, coords_list, structures, pair_constraints_all, metadata


def save_pseudo_labels(sequences, coords_list, structures,
                       pair_constraints_all, metadata, output_dir):
    """Save pseudo-labels to disk."""
    os.makedirs(output_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    # Save each coordinate array as .npy
    for i, coords in enumerate(coords_list):
        np.save(os.path.join(coords_dir, f'pseudo_{i:04d}.npy'), coords)

    # Save sequences
    seq_data = []
    for i, (seq, ss, pairs) in enumerate(zip(sequences, structures, pair_constraints_all)):
        seq_data.append({
            'id': f'pseudo_{i:04d}',
            'sequence': seq,
            'secondary_structure': ss,
            'pair_constraints': [(p[0], p[1]) for p in pairs],
        })

    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(seq_data, f, indent=2)

    # Save metadata
    summary = {
        'total': len(sequences),
        'length_range': [min(m['length'] for m in metadata),
                         max(m['length'] for m in metadata)],
        'mean_closure_error': float(np.mean([m['closure_error'] for m in metadata])),
        'mean_n_pairs': float(np.mean([m['n_pairs'] for m in metadata])),
        'vienna_used': any(m['source'] == 'ViennaRNA+Physics' for m in metadata),
        'samples': metadata,
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    sequences.json: {len(sequences)} sequences")
    print(f"    coords/: {len(coords_list)} .npy files")
    print(f"    metadata.json: summary + per-sample info")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D pseudo-labels')
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of sequences to generate (recommend 1000 for scheme 4/6)')
    parser.add_argument('--min-len', type=int, default=30)
    parser.add_argument('--max-len', type=int, default=500)
    parser.add_argument('--output', type=str, default='data/pseudo_labels')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold 3D Pseudo-label Generation")
    print("=" * 60)
    print(f"  Target: {args.n} sequences")
    print(f"  Length: {args.min_len}-{args.max_len}")
    print(f"  Output: {args.output}")

    sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels(
        n_seqs=args.n,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed
    )

    save_pseudo_labels(sequences, coords_list, structures, pairs, metadata, args.output)

    print("\n" + "=" * 60)
    print("  Next: Train schemes with these labels")
    print("  python train_all_schemes.py --labels data/pseudo_labels --device cuda")
    print("=" * 60)


if __name__ == '__main__':
    main()
