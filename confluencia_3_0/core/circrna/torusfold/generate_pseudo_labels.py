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


def generate_pseudo_labels_3drna(n_seqs=1000, min_len=30, max_len=500, seed=42, use_3drna=True):
    """Generate 3D pseudo-labels using ViennaRNA + 3dRNA (if available)."""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    sequences = []
    coords_list = []
    structures = []
    metadata = []

    try:
        import RNA
        has_vienna = True
        print("  ViennaRNA: available (circ mode)")
    except ImportError:
        has_vienna = False
        print("  ViennaRNA: NOT available")
        return sequences, coords_list, structures, [], metadata

    # Check 3dRNA
    if use_3drna:
        # Try local 3dRNA binary
        import subprocess
        result = subprocess.run(['which', '3dRNA'], capture_output=True)
        has_3drna_local = result.returncode == 0

        if has_3drna_local:
            print("  3dRNA: available (local binary)")
        else:
            print("  3dRNA: NOT installed locally")
            print("    Download from: http://biophy.hust.edu.cn/3dRNA")
            print("    Or use web server (manual submission)")
            use_3drna = False

    print(f"  Generating {n_seqs} pseudo-labels...")
    t0 = time.time()

    for i in range(n_seqs):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, size=L))

        # ViennaRNA secondary structure (circ mode)
        md = RNA.md()
        md.circ = True
        fc = RNA.fold_compound(seq, md)
        ss, mfe = fc.mfe()

        sequences.append(seq)
        structures.append(ss)

        if use_3drna:
            # Call 3dRNA
            coords = run_3drna_local(seq, ss)
            if coords is not None:
                coords_list.append(coords)
                metadata.append({
                    'id': f'pseudo_{i:04d}',
                    'length': L,
                    'source': 'ViennaRNA+3dRNA',
                    'mfe': float(mfe),
                })
        else:
            # Fallback: generate helical coords (much faster)
            coords = generate_helical_coords(L)
            coords_list.append(coords)
            metadata.append({
                'id': f'pseudo_{i:04d}',
                'length': L,
                'source': 'ViennaRNA+Helical',
                'mfe': float(mfe),
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_seqs} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(sequences)} in {elapsed:.1f}s")
    return sequences, coords_list, structures, [], metadata


def run_3drna_local(sequence: str, structure: str) -> np.ndarray:
    """Run local 3dRNA binary to generate 3D coordinates."""
    import subprocess
    import tempfile
    import os

    try:
        # Create input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{sequence}\n{structure}\n")
            input_file = f.name

        output_file = input_file.replace('.txt', '.pdb')

        # Run 3dRNA
        result = subprocess.run(
            ['3dRNA', input_file, output_file],
            capture_output=True, timeout=60
        )

        if result.returncode == 0 and os.path.exists(output_file):
            # Parse PDB
            coords = parse_pdb_coords(output_file)
            os.unlink(input_file)
            os.unlink(output_file)
            return coords

        os.unlink(input_file)

    except Exception as e:
        pass

    return None


def parse_pdb_coords(pdb_file: str) -> np.ndarray:
    """Parse PDB file to extract backbone P atom coordinates."""
    coords = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 8:
                    atom_name = parts[2]
                    if atom_name == 'P':  # Phosphate backbone
                        x = float(parts[-4])
                        y = float(parts[-3])
                        z = float(parts[-2])
                        coords.append([x, y, z])

    return np.array(coords) if coords else None


def generate_helical_coords(L: int, bond_length: float = 5.9) -> np.ndarray:
    """Generate ideal A-form helical coordinates (fast fallback).

    This is NOT a real 3D structure, just a helical backbone.
    Use when 3dRNA is not available.
    """
    coords = np.zeros((L, 3))
    rise_per_nt = 2.8  # A-form RNA rise
    radius = 10.0  # Helix radius

    for i in range(L):
        angle = 2 * np.pi * i / 10  # 10 nt per turn
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = rise_per_nt * i

    # Center
    coords = coords - coords.mean(axis=0)

    # Adjust to approximate closure (for circRNA)
    if L > 20:
        # Bend into circle
        theta = 2 * np.pi * np.arange(L) / L
        coords[:, 0] = bond_length * L / (2 * np.pi) * np.cos(theta)
        coords[:, 1] = bond_length * L / (2 * np.pi) * np.sin(theta)
        coords[:, 2] = np.arange(L) * 2.8 - L * 1.4

    return coords


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
    for i, (seq, ss) in enumerate(zip(sequences, structures)):
        pairs = pair_constraints_all[i] if i < len(pair_constraints_all) else []
        seq_data.append({
            'id': f'pseudo_{i:04d}',
            'sequence': seq,
            'secondary_structure': ss,
            'pair_constraints': [(p[0], p[1]) for p in pairs] if pairs else [],
        })

    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(seq_data, f, indent=2)

    # Save metadata
    closure_errors = [m.get('closure_error', 0) for m in metadata if 'closure_error' in m]
    n_pairs_list = [m.get('n_pairs', 0) for m in metadata if 'n_pairs' in m]

    summary = {
        'total': len(sequences),
        'length_range': [min(m['length'] for m in metadata),
                         max(m['length'] for m in metadata)],
        'mean_closure_error': float(np.mean(closure_errors)) if closure_errors else 0.0,
        'mean_n_pairs': float(np.mean(n_pairs_list)) if n_pairs_list else 0.0,
        'vienna_used': any('ViennaRNA' in m.get('source', '') for m in metadata),
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
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', '3drna', 'physics', 'helical'],
                        help='Method: auto=3dRNA>physics>helical, 3drna=3dRNA only, '
                             'physics=Physics Solver, helical=fast helical coords')
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold 3D Pseudo-label Generation")
    print("=" * 60)
    print(f"  Target: {args.n} sequences")
    print(f"  Length: {args.min_len}-{args.max_len}")
    print(f"  Method: {args.method}")
    print(f"  Output: {args.output}")

    if args.method == '3drna':
        sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels_3drna(
            n_seqs=args.n, min_len=args.min_len, max_len=args.max_len,
            seed=args.seed, use_3drna=True
        )
    elif args.method == 'helical':
        sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels_3drna(
            n_seqs=args.n, min_len=args.min_len, max_len=args.max_len,
            seed=args.seed, use_3drna=False  # Falls back to helical
        )
    elif args.method == 'physics':
        sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels(
            n_seqs=args.n, min_len=args.min_len, max_len=args.max_len,
            seed=args.seed
        )
    else:  # auto
        # Try 3dRNA first, then physics, then helical
        try:
            import subprocess
            result = subprocess.run(['which', '3dRNA'], capture_output=True)
            has_3drna = result.returncode == 0
        except Exception:
            has_3drna = False

        if has_3drna:
            print("  Auto-detected: using 3dRNA")
            sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels_3drna(
                n_seqs=args.n, min_len=args.min_len, max_len=args.max_len,
                seed=args.seed, use_3drna=True
            )
        else:
            print("  Auto-detected: 3dRNA not found, using Physics Solver")
            sequences, coords_list, structures, pairs, metadata = generate_pseudo_labels(
                n_seqs=args.n, min_len=args.min_len, max_len=args.max_len,
                seed=args.seed
            )

    save_pseudo_labels(sequences, coords_list, structures, pairs, metadata, args.output)

    print("\n" + "=" * 60)
    print("  Next: Train schemes with these labels")
    print("  python train_all_schemes.py --labels data/pseudo_labels --device cuda")
    print("=" * 60)


if __name__ == '__main__':
    main()
