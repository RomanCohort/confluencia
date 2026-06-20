#!/usr/bin/env python3
"""
generate_medium_length_dataset.py — Generate medium-length (500-1000 nt) circRNA 3D dataset.

Fills the length gap between build_training_dataset.py (50-500 nt) and
the very long sequences that are computationally prohibitive.

Composition:
    60% from real circBase sequences (500-1000 nt) + ViennaRNA circ-mode + GeometricConstraintSolver
    40% from synthetic random sequences (500-1000 nt) + same pipeline
    Target: 2,000 samples

Output:
    <output_dir>/
    ├── sequences.json     # All sequences with secondary_structure and pair_constraints
    ├── coords/            # .npy coordinate arrays
    │   ├── medium_0000.npy
    │   └── ...
    └── metadata.json      # Summary statistics

Usage:
    python generate_medium_length_dataset.py --output data/medium_length_3d
    python generate_medium_length_dataset.py --output data/medium_length_3d --n-samples 2000
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import ViennaRNA
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

# Try to import GeometricConstraintSolver
try:
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig,
    )
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False


# ═══════════════════════════════════════════════════════════════
# circBase download and parse (from generate_real_dataset.py)
# ═══════════════════════════════════════════════════════════════

CIRCBASE_URL = "http://www.circbase.org/cgi-bin/download.cgi"


def download_circbase(output_dir: str, species: str = "hsa") -> Optional[str]:
    """Download circBase data. Returns file path or None on failure."""
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"circBase_{species}.txt.gz")

    if os.path.exists(output_path):
        print(f"  Found existing: {output_path}")
        return output_path

    print(f"  Downloading circBase ({species})...")
    url = f"{CIRCBASE_URL}?sp={species}&db=hsa&type=circ"

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def parse_circbase(filepath: str, min_len: int = 500, max_len: int = 1000) -> List[Dict]:
    """Parse circBase file, extract circRNA sequences in length range.

    circBase format (BED-like):
    chrom  start  end  name  score  strand  circRNA_type  seq
    """
    sequences = []

    print(f"  Parsing: {filepath}")

    if filepath.endswith('.gz'):
        opener = lambda: gzip.open(filepath, 'rt')
    else:
        opener = lambda: open(filepath, 'r')

    with opener() as f:
        for i, line in enumerate(f):
            if line.startswith('#') or line.startswith('chrom'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom, start, end, name, score, strand = parts[:6]
            seq = parts[7] if len(parts) > 7 else None

            if not seq:
                continue

            L = len(seq)
            if L < min_len or L > max_len:
                continue

            valid_bases = set('ACGUacgu')
            if not all(b in valid_bases for b in seq):
                continue

            sequences.append({
                'id': f"circBase_{name}_{i}",
                'sequence': seq.upper().replace('T', 'U'),
                'chrom': chrom,
                'start': int(start),
                'end': int(end),
                'strand': strand,
                'length': L,
            })

            if len(sequences) % 500 == 0:
                print(f"    Parsed {len(sequences)} sequences (500-1000 nt)...")

    print(f"  Total valid sequences (500-1000 nt): {len(sequences)}")
    return sequences


# ═══════════════════════════════════════════════════════════════
# 3D structure generation
# ═══════════════════════════════════════════════════════════════

def simple_constraint_solve(seq_len: int, pair_constraints: list, bond_length: float = 5.9) -> np.ndarray:
    """Generate 3D coords from constraints using gradient descent.

    Fallback when GeometricConstraintSolver is not available.
    """
    coords = np.zeros((seq_len, 3))
    # Initialize as circular helix
    for i in range(seq_len):
        angle = 2 * np.pi * i / seq_len
        radius = bond_length * seq_len / (2 * np.pi) * 0.5
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0]

    # Refine with constraint satisfaction (simple gradient descent)
    for step in range(200):
        grad = np.zeros_like(coords)
        # Bond constraints
        for i in range(seq_len - 1):
            nxt = (i + 1) % seq_len
            diff = coords[nxt] - coords[i]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = (dist - bond_length) * diff / dist
                grad[i] += force * 0.1
                grad[nxt] -= force * 0.1
        # Pair constraints
        for pi, pj, target_d, w in pair_constraints:
            diff = coords[pj] - coords[pi]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = w * (dist - target_d) * diff / dist
                grad[pi] += force * 0.05
                grad[pj] -= force * 0.05
        coords -= grad
    return coords


def predict_secondary_structure(sequence: str, rng: np.random.RandomState):
    """Predict secondary structure and extract pair constraints.

    Uses ViennaRNA with circ mode if available, otherwise heuristic pairing.

    Returns:
        (ss, pair_constraints, mfe) tuple
    """
    L = len(sequence)
    pair_constraints = []
    ss = '.' * L
    mfe = 0.0

    if HAS_VIENNA:
        try:
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()

            # Extract base pairs from dot-bracket
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
                if complement.get(sequence[j]) == sequence[k] and rng.random() < 0.3:
                    pair_constraints.append((j, k, 10.6, 1.0))

    return ss, pair_constraints, mfe


def generate_3d_coords(seq_len: int, pair_constraints: list,
                       solver=None, rng: np.random.RandomState = None) -> np.ndarray:
    """Generate 3D coordinates from pair constraints.

    Uses GeometricConstraintSolver if available, otherwise gradient descent fallback.
    """
    coords = None

    if solver and pair_constraints:
        # Build constraint set
        class CS:
            def __init__(self, n, pairs):
                self.seq_len = n
                self.pair_constraints = pairs

        cs = CS(seq_len, pair_constraints)
        conformations = solver.solve(cs)
        if conformations and len(conformations) > 0:
            coords = conformations[0]

    if coords is None:
        if pair_constraints:
            coords = simple_constraint_solve(seq_len, pair_constraints, bond_length=5.9)
        else:
            # Last resort: helical coords
            rise_per_nt = 2.8
            twist_per_nt = 32.7
            bond_length = 5.9
            radius = max(5.0, seq_len * rise_per_nt / (2 * np.pi) * 0.8)
            coords = np.zeros((seq_len, 3))
            for j in range(seq_len):
                angle = np.deg2rad(twist_per_nt * j)
                coords[j, 0] = radius * np.cos(angle)
                coords[j, 1] = radius * np.sin(angle)
                coords[j, 2] = rise_per_nt * j
            coords = coords - coords.mean(axis=0)
            if rng is not None:
                coords += rng.normal(0, 0.3, (seq_len, 3))

    return coords


# ═══════════════════════════════════════════════════════════════
# Dataset generation
# ═══════════════════════════════════════════════════════════════

def generate_real_samples(sequences: List[Dict], n_target: int,
                          solver, rng: np.random.RandomState,
                          output_dir: str):
    """Generate 3D structures for real circBase sequences."""
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    result_seqs = []
    result_coords = []

    # Subsample if more sequences available than needed
    if len(sequences) > n_target:
        indices = rng.choice(len(sequences), n_target, replace=False)
        sequences = [sequences[i] for i in indices]

    print(f"  Processing {len(sequences)} real circBase sequences...")

    for i, item in enumerate(sequences):
        seq = item['sequence']
        L = item['length']
        seq_id = item['id']

        ss, pair_constraints, mfe = predict_secondary_structure(seq, rng)
        coords = generate_3d_coords(L, pair_constraints, solver, rng)

        # Save coords
        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)

        result_seqs.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': ss,
            'pair_constraints': [(p[0], p[1]) for p in pair_constraints],
            'length': L,
            'source': 'circbase_real',
            'chrom': item.get('chrom', ''),
            'mfe': float(mfe) if mfe != 0.0 else None,
        })
        result_coords.append(coords)

        if (i + 1) % 100 == 0:
            print(f"    Real: {i+1}/{len(sequences)}")

    return result_seqs, result_coords


def generate_synthetic_samples(n_samples: int, min_len: int, max_len: int,
                               solver, rng: np.random.RandomState,
                               output_dir: str):
    """Generate synthetic random sequences + 3D structures."""
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    bases = ['A', 'C', 'G', 'U']
    result_seqs = []
    result_coords = []

    print(f"  Generating {n_samples} synthetic sequences ({min_len}-{max_len} nt)...")

    for i in range(n_samples):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, L))

        ss, pair_constraints, mfe = predict_secondary_structure(seq, rng)
        coords = generate_3d_coords(L, pair_constraints, solver, rng)

        seq_id = f'medium_synth_{i:05d}'

        # Save coords
        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)

        result_seqs.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': ss,
            'pair_constraints': [(p[0], p[1]) for p in pair_constraints],
            'length': L,
            'source': 'synthetic',
            'mfe': float(mfe) if mfe != 0.0 else None,
        })
        result_coords.append(coords)

        if (i + 1) % 100 == 0:
            print(f"    Synthetic: {i+1}/{n_samples}")

    return result_seqs, result_coords


def main():
    parser = argparse.ArgumentParser(
        description='Generate medium-length (500-1000 nt) circRNA 3D dataset'
    )
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Total number of samples (default: 2000)')
    parser.add_argument('--min-len', type=int, default=500,
                        help='Minimum sequence length')
    parser.add_argument('--max-len', type=int, default=1000,
                        help='Maximum sequence length')
    parser.add_argument('--real-fraction', type=float, default=0.6,
                        help='Fraction of samples from circBase (default: 0.6)')
    parser.add_argument('--species', type=str, default='hsa',
                        help='Species for circBase (hsa=human, mmu=mouse)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    print("=" * 60)
    print("  Medium-Length circRNA 3D Dataset Generation")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Target samples: {args.n_samples}")
    print(f"  Length range: {args.min_len}-{args.max_len} nt")
    print(f"  Real fraction: {args.real_fraction:.0%}")
    print(f"  ViennaRNA: {'available (circ mode)' if HAS_VIENNA else 'NOT available, using heuristic pairing'}")
    print(f"  GeometricConstraintSolver: {'available' if HAS_SOLVER else 'NOT available, using gradient descent fallback'}")

    # Initialize solver if available
    solver = None
    if HAS_SOLVER:
        config = SolverConfig(
            n_samples=10,
            use_annealing_closure=True,
            bond_length=5.9,
            pair_distance=10.6,
        )
        solver = GeometricConstraintSolver(config)

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    n_real = int(args.n_samples * args.real_fraction)
    n_synth = args.n_samples - n_real

    t0 = time.time()

    # ── Phase 1: Real circBase sequences ──────────────────────
    print(f"\n[1/2] Loading circBase sequences ({n_real} target)...")

    circbase_file = download_circbase(output_dir, args.species)

    real_seqs = []
    real_coords = []

    if circbase_file and os.path.exists(circbase_file):
        circbase_sequences = parse_circbase(
            circbase_file, args.min_len, args.max_len
        )
        if circbase_sequences:
            real_seqs, real_coords = generate_real_samples(
                circbase_sequences, n_real, solver, rng, output_dir
            )
        else:
            print("  No circBase sequences in 500-1000 nt range found.")
    else:
        print("  circBase download failed, skipping real samples.")

    # Adjust synthetic count if we got fewer real samples than target
    actual_real = len(real_seqs)
    if actual_real < n_real:
        shortfall = n_real - actual_real
        n_synth += shortfall
        print(f"  Real samples shortfall: {shortfall}, added to synthetic ({n_synth})")

    # ── Phase 2: Synthetic random sequences ───────────────────
    print(f"\n[2/2] Generating {n_synth} synthetic samples...")

    synth_seqs, synth_coords = generate_synthetic_samples(
        n_synth, args.min_len, args.max_len, solver, rng, output_dir
    )

    # ── Combine and save ──────────────────────────────────────
    all_sequences = real_seqs + synth_seqs
    all_coords = real_coords + synth_coords

    # Save sequences.json
    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(all_sequences, f, indent=2)

    # Save metadata.json
    lengths = [s['length'] for s in all_sequences]
    n_pairs_list = [len(s['pair_constraints']) for s in all_sequences]

    metadata = {
        'total': len(all_sequences),
        'length_range': [min(lengths), max(lengths)] if lengths else [0, 0],
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'mean_n_pairs': float(np.mean(n_pairs_list)) if n_pairs_list else 0.0,
        'sources': {
            'circbase_real': sum(1 for s in all_sequences if s['source'] == 'circbase_real'),
            'synthetic': sum(1 for s in all_sequences if s['source'] == 'synthetic'),
        },
        'pipeline': {
            'secondary_structure': 'ViennaRNA circ-mode' if HAS_VIENNA else 'heuristic',
            '3d_generation': 'GeometricConstraintSolver' if HAS_SOLVER else 'gradient_descent',
        },
        'params': {
            'min_len': args.min_len,
            'max_len': args.max_len,
            'real_fraction_target': args.real_fraction,
            'real_fraction_actual': actual_real / len(all_sequences) if all_sequences else 0.0,
            'seed': args.seed,
        },
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Dataset: {output_dir}/")
    print(f"  Total: {metadata['total']}")
    print(f"  circBase (real): {metadata['sources']['circbase_real']}")
    print(f"  Synthetic: {metadata['sources']['synthetic']}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"  Mean length: {metadata['mean_length']:.1f}")
    print(f"  Mean pairs: {metadata['mean_n_pairs']:.1f}")
    print(f"  Pipeline: {metadata['pipeline']['secondary_structure']} + {metadata['pipeline']['3d_generation']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Next: Train with medium-length data")
    print(f"  python train_all_schemes.py --labels {output_dir} --device cuda")


if __name__ == '__main__':
    main()
