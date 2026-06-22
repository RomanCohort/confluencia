#!/usr/bin/env python3
"""
circbase_to_training.py — Generate training data from circBase fasta sequences.

Reads circBase fasta, filters by length, samples N sequences,
predicts secondary structure with ViennaRNA circ-mode,
generates 3D coordinates with constraint solver.

Output format compatible with train_all_schemes.py load_pseudo_labels().

Usage:
    python circbase_to_training.py \
        --fasta data/circrna/circbase_seqs.fa.gz \
        --output data/circbase_real_3d \
        --n-samples 6000 \
        --min-len 50 \
        --max-len 1000
"""

import argparse
import json
import os
import sys
import gzip
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try ViennaRNA
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

# Try GeometricConstraintSolver
try:
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig
    )
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False


def extract_pairs_from_dot_bracket(ss: str) -> list:
    """Extract base pairs from dot-bracket. Supports ( ) [ ]."""
    pairs = []
    stack_a = []
    stack_b = []
    for i, char in enumerate(ss):
        if char == "(":
            stack_a.append(i)
        elif char == ")" and stack_a:
            j = stack_a.pop()
            pairs.append([j, i])
        elif char == "[":
            stack_b.append(i)
        elif char == "]" and stack_b:
            j = stack_b.pop()
            pairs.append([j, i])
    return pairs


def predict_ss_circ(sequence: str) -> tuple:
    """Predict secondary structure using ViennaRNA circ mode.
    Returns (dot_bracket, mfe) or ('.'*L, 0.0).
    """
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    if HAS_VIENNA:
        try:
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()
            return ss, float(mfe)
        except Exception:
            pass

    return '.' * L, 0.0


def generate_coords_from_constraints(L: int, pair_constraints: list) -> np.ndarray:
    """Generate 3D coords from pair constraints using gradient descent.

    Includes BSJ closure constraint: first and last nucleotide must be
    ~bond_length apart (5.9 Å) to form the back-splice junction.
    """
    bond_length = 5.9
    pair_distance = 10.6

    coords = np.zeros((L, 3))
    # Initialize as circular helix (already circular by construction)
    for i in range(L):
        angle = 2 * np.pi * i / L
        radius = bond_length * L / (2 * np.pi) * 0.5
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0]

    # Refine with constraint satisfaction
    for step in range(300):
        grad = np.zeros_like(coords)
        # Bond constraints (circular, includes BSJ: last→first)
        for i in range(L):
            nxt = (i + 1) % L
            diff = coords[nxt] - coords[i]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = (dist - bond_length) * diff / dist
                grad[i] += force * 0.1
                grad[nxt] -= force * 0.1
        # Pair constraints
        for pi, pj in pair_constraints:
            diff = coords[pj] - coords[pi]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = (dist - pair_distance) * diff / dist
                grad[pi] += force * 0.05
                grad[pj] -= force * 0.05
        # BSJ closure: strengthen first-last bond (higher weight)
        bsj_diff = coords[0] - coords[L - 1]
        bsj_dist = np.linalg.norm(bsj_diff)
        if bsj_dist > 0:
            bsj_force = 0.15 * (bsj_dist - bond_length) * bsj_diff / bsj_dist
            grad[L - 1] += bsj_force
            grad[0] -= bsj_force
        # Centering
        grad -= coords.mean(axis=0) * 0.01
        coords -= grad

    # Center
    coords -= coords.mean(axis=0)
    return coords


def read_fasta(fasta_path: str, min_len: int, max_len: int) -> list:
    """Read fasta, filter by length. Returns list of (header, sequence)."""
    opener = gzip.open if fasta_path.endswith('.gz') else open
    entries = []

    with opener(fasta_path, 'rt') as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq_parts:
                    seq = ''.join(seq_parts)
                    if min_len <= len(seq) <= max_len:
                        entries.append((header, seq))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper().replace('T', 'U'))
        # Last entry
        if header and seq_parts:
            seq = ''.join(seq_parts)
            if min_len <= len(seq) <= max_len:
                entries.append((header, seq))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from circBase fasta sequences"
    )
    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to circBase fasta file (.fa or .fa.gz)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=6000,
                        help='Number of sequences to sample')
    parser.add_argument('--min-len', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max-len', type=int, default=1000,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-if-exists', action='store_true',
                        help='Skip sequences that already have coords')
    args = parser.parse_args()

    print("=" * 60)
    print("  circBase Fasta → Training Data")
    print("=" * 60)

    t0 = time.time()

    print(f"\n  ViennaRNA: {'available (circ mode)' if HAS_VIENNA else 'NOT available'}")
    print(f"  GeometricConstraintSolver: {'available' if HAS_SOLVER else 'NOT available'}")

    # Read fasta
    print(f"\n[1/4] Reading fasta: {args.fasta}")
    print(f"  Filter: {args.min_len}-{args.max_len} nt")
    entries = read_fasta(args.fasta, args.min_len, args.max_len)
    print(f"  Sequences in range: {len(entries)}")

    if not entries:
        print("  ERROR: No sequences found in length range!")
        sys.exit(1)

    # Sample
    rng = np.random.RandomState(args.seed)
    if len(entries) > args.n_samples:
        indices = rng.choice(len(entries), args.n_samples, replace=False)
        indices.sort()
        entries = [entries[i] for i in indices]
        print(f"  Sampled: {len(entries)}")
    else:
        print(f"  Using all {len(entries)} sequences (less than n_samples)")

    # Generate structures
    print(f"\n[2/4] Predicting secondary structure + generating 3D coords...")
    coords_dir = os.path.join(args.output, "coords")
    os.makedirs(coords_dir, exist_ok=True)

    results = []
    n_with_ss = 0
    n_with_pairs = 0

    for i, (header, seq) in enumerate(entries):
        L = len(seq)

        # Extract circRNA ID from header
        # Format: hsa_circ_XXXXXXX|chr1:... or just use index
        parts = header.split('|')
        circ_id = parts[0] if parts else f"circbase_{i:05d}"
        seq_id = f"circbase_{circ_id}"

        # Predict SS with ViennaRNA circ mode
        ss, mfe = predict_ss_circ(seq)
        pair_constraints = extract_pairs_from_dot_bracket(ss) if ss != '.' * L else []

        if ss != '.' * L:
            n_with_ss += 1
        if len(pair_constraints) > 0:
            n_with_pairs += 1

        # Generate 3D coords
        if HAS_SOLVER and pair_constraints:
            try:
                config = SolverConfig(
                    n_samples=1, use_annealing_closure=False,
                    bond_length=5.9, pair_distance=10.6, max_iterations=50,
                )
                solver = GeometricConstraintSolver(config)

                class CS:
                    def __init__(self, n, pairs):
                        self.seq_len = n
                        self.pair_constraints = pairs

                cs = CS(L, [(p[0], p[1], 10.6, 1.0) for p in pair_constraints])
                conformations = solver.solve(cs)
                coords = conformations[0] if conformations and len(conformations) > 0 else None
            except Exception:
                coords = None

        if not HAS_SOLVER or coords is None:
            coords = generate_coords_from_constraints(L, pair_constraints)

        # Center coords
        coords = coords - coords.mean(axis=0)

        # Save coords
        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)

        results.append({
            "id": seq_id,
            "sequence": seq,
            "secondary_structure": ss,
            "pair_constraints": pair_constraints,
            "length": L,
            "source": "circbase_real",
            "mfe": mfe if mfe != 0.0 else None,
            "chrom": parts[1].split(':')[0] if len(parts) > 1 else None,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(entries)} | SS: {n_with_ss} | Pairs: {n_with_pairs} | "
                  f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s")

    # Save sequences.json
    print(f"\n[3/4] Saving sequences.json...")
    with open(os.path.join(args.output, "sequences.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save metadata
    print(f"[4/4] Saving metadata.json...")
    lengths = [r['length'] for r in results]
    source_counts = {}
    for r in results:
        src = r['source']
        source_counts[src] = source_counts.get(src, 0) + 1

    metadata = {
        "total": len(results),
        "length_range": [min(lengths), max(lengths)],
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "n_with_pair_constraints": n_with_pairs,
        "n_with_real_ss": n_with_ss,
        "fraction_with_pairs": n_with_pairs / len(results) if results else 0.0,
        "fraction_with_real_ss": n_with_ss / len(results) if results else 0.0,
        "sources": source_counts,
        "input_fasta": args.fasta,
        "filter": {"min_len": args.min_len, "max_len": args.max_len},
    }
    with open(os.path.join(args.output, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Output: {args.output}/")
    print(f"  Total: {len(results)}")
    print(f"  With pair_constraints: {n_with_pairs} ({100*n_with_pairs/len(results):.1f}%)")
    print(f"  With real SS: {n_with_ss} ({100*n_with_ss/len(results):.1f}%)")
    print(f"  Length range: {min(lengths)}-{max(lengths)} nt")
    print(f"  Mean length: {np.mean(lengths):.0f} nt")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
