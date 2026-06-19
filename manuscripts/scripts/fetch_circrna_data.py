#!/usr/bin/env python3
"""
fetch_circrna_data.py — Fetch all available circRNA 3D structure data.

Data sources:
1. PDB experimental structures (9H8A, 8xtp, 8xtq, 8xtr, 8xts, 9is7)
2. IsRNAcirc 34-structure test set
3. circBank sequences (for pseudo-label generation)

Usage:
    python fetch_circrna_data.py --output data/circrna_3d --pdb --isrnacirc
"""

import argparse
import os
import sys
from pathlib import Path


# ── PDB Experimental Structures ──────────────────────────────

CIRCRNA_PDB_IDS = {
    '9H8A': 'cryo-EM circRNA dimer, 9.60 Å',
    '8xtp': 'bacterial type II intron, circularization, 2.5-2.9 Å',
    '8xtq': 'bacterial type II intron, circularization, 2.5-2.9 Å',
    '8xtr': 'bacterial type II intron, circularization, 2.5-2.9 Å',
    '8xts': 'bacterial type II intron, circularization, 2.5-2.9 Å',
    '9is7': 'bacterial type II intron, circularization, 2.5-2.9 Å',
}


def fetch_pdb(pdb_id: str, output_dir: str) -> bool:
    """Fetch PDB structure file."""
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    out_path = os.path.join(output_dir, f"{pdb_id}.cif")

    if os.path.exists(out_path):
        print(f"  {pdb_id}: already exists")
        return True

    try:
        import urllib.request
        urllib.request.urlretrieve(pdb_url, out_path)
        print(f"  {pdb_id}: downloaded ({CIRCRNA_PDB_IDS.get(pdb_id, '')})")
        return True
    except Exception as e:
        print(f"  {pdb_id}: FAILED - {e}")
        return False


def fetch_all_pdb(output_dir: str):
    """Fetch all known circRNA PDB structures."""
    pdb_dir = os.path.join(output_dir, 'pdb_experimental')
    os.makedirs(pdb_dir, exist_ok=True)

    print("\nFetching PDB experimental structures:")
    print(f"  Known circRNA structures: {len(CIRCRNA_PDB_IDS)}")

    results = {}
    for pdb_id, description in CIRCRNA_PDB_IDS.items():
        results[pdb_id] = fetch_pdb(pdb_id, pdb_dir)

    # Write metadata
    meta_path = os.path.join(pdb_dir, 'metadata.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump({
            'structures': {k: {'description': v, 'downloaded': results[k]}
                          for k, v in CIRCRNA_PDB_IDS.items()},
            'note': '9H8A is the only true circRNA; others are circularization-related introns',
        }, f, indent=2)

    print(f"\n  Downloaded: {sum(results.values())}/{len(results)}")
    return results


# ── IsRNAcirc Test Set ───────────────────────────────────────

ISRNACIRC_REPO = "https://github.com/DongZhangRNA/IsRNAcirc"


def fetch_isrnacirc(output_dir: str):
    """Clone IsRNAcirc repository for 34-structure test set."""
    isrnacirc_dir = os.path.join(output_dir, 'isrnacirc_test_set')

    if os.path.exists(isrnacirc_dir):
        print(f"  IsRNAcirc: already cloned at {isrnacirc_dir}")
        return True

    print("\nFetching IsRNAcirc test set:")
    print(f"  Repository: {ISRNACIRC_REPO}")
    print(f"  Contains: 34 predicted circRNA 3D structures")

    try:
        import subprocess
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', ISRNACIRC_REPO, isrnacirc_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"  IsRNAcirc: cloned successfully")
            return True
        else:
            print(f"  IsRNAcirc: clone failed - {result.stderr}")
            return False
    except Exception as e:
        print(f"  IsRNAcirc: FAILED - {e}")
        return False


# ── circBank Sequences ───────────────────────────────────────

CIRCBANK_URL = "http://www.circbank.cn/"


def fetch_circbank_info(output_dir: str):
    """Write circBank access information (no direct API)."""
    info_path = os.path.join(output_dir, 'circbank_info.json')

    import json
    with open(info_path, 'w') as f:
        json.dump({
            'url': CIRCBANK_URL,
            'description': 'circBank 2.0: circRNA sequences, annotations, miRNA binding',
            'note': 'No direct API. Use web interface to download sequences by species/filter.',
            'usage': 'Download circRNA sequences → ViennaRNA pseudo-labels → TorusFold training',
        }, f, indent=2)

    print(f"\n  circBank info saved to {info_path}")
    print(f"  Access at: {CIRCBANK_URL}")
    print(f"  Usage: Download sequences → generate pseudo-labels → train TorusFold")


# ── Pseudo-label Generation Pipeline ────────────────────────

def generate_pseudo_labels(output_dir: str, n_sequences: int = 100):
    """Generate pseudo-labels using ViennaRNA + physics refinement.

    This creates a training dataset for TorusFold from:
    1. Random circRNA sequences
    2. ViennaRNA secondary structure prediction
    3. Physics-based 3D structure generation
    """
    pseudo_dir = os.path.join(output_dir, 'pseudo_labels')
    os.makedirs(pseudo_dir, exist_ok=True)

    print(f"\nGenerating {n_sequences} pseudo-labels:")

    try:
        import RNA  # ViennaRNA Python bindings
        has_vienna = True
    except ImportError:
        has_vienna = False
        print("  ViennaRNA not available. Install: pip install ViennaRNA")
        print("  Skipping pseudo-label generation.")
        return False

    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig
    )

    rng = np.random.RandomState(42)
    labels = []

    for i in range(n_sequences):
        # Random sequence
        L = rng.randint(50, 300)
        seq = ''.join(rng.choice(['A', 'C', 'G', 'U'], size=L))

        # ViennaRNA secondary structure (circ mode)
        try:
            md = RNA.md()
            md.circ = True  # circRNA mode
            fc = RNA.fold_compound(seq, md)
            ss, mfe = fc.mfe()
        except Exception:
            continue

        # 3D structure from physics solver
        config = SolverConfig(n_samples=5, use_annealing_closure=True)
        solver = GeometricConstraintSolver(config)

        class CS:
            def __init__(self, n):
                self.seq_len = n
                self.pair_constraints = []

        confs = solver.solve(CS(L))
        if not confs:
            continue

        labels.append({
            'id': f'pseudo_{i:04d}',
            'sequence': seq,
            'secondary_structure': ss,
            'mfe': float(mfe),
            'length': L,
        })

    # Save
    import json
    label_path = os.path.join(pseudo_dir, 'pseudo_labels.json')
    with open(label_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"  Generated: {len(labels)} pseudo-labels")
    print(f"  Saved to: {label_path}")
    return True


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fetch circRNA 3D structure data')
    parser.add_argument('--output', type=str, default='data/circrna_3d')
    parser.add_argument('--pdb', action='store_true', help='Fetch PDB structures')
    parser.add_argument('--isrnacirc', action='store_true', help='Clone IsRNAcirc')
    parser.add_argument('--circbank', action='store_true', help='Write circBank info')
    parser.add_argument('--pseudo-labels', type=int, default=0,
                        help='Generate N pseudo-labels (requires ViennaRNA)')
    parser.add_argument('--all', action='store_true', help='Fetch everything')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  circRNA 3D Structure Data Fetcher")
    print("=" * 60)

    if args.all or args.pdb:
        fetch_all_pdb(args.output)

    if args.all or args.isrnacirc:
        fetch_isrnacirc(args.output)

    if args.all or args.circbank:
        fetch_circbank_info(args.output)

    if args.pseudo_labels > 0:
        generate_pseudo_labels(args.output, args.pseudo_labels)

    print("\n" + "=" * 60)
    print("  Data Summary")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  PDB structures: {len(CIRCRNA_PDB_IDS)} known")
    print(f"  IsRNAcirc: 34 predicted structures")
    print(f"  circBank: sequence data (web interface)")
    print("=" * 60)


if __name__ == '__main__':
    import numpy as np
    main()
