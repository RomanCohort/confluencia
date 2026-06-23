#!/usr/bin/env python3
"""
quick_pdb_testset.py -- Generate test set from cached PDB files (no download needed).

Usage:
    python quick_pdb_testset.py --pdb-cache data/pdb_cache --output data/pdb_3d --n-samples 7
"""

import argparse
import glob
import json
import math
import os

import numpy as np


RESIDUE_MAP = {
    'A': 'A', 'ADE': 'A',
    'U': 'U', 'URI': 'U',
    'G': 'G', 'GUA': 'G',
    'C': 'C', 'CYT': 'C',
}


def parse_pdb_rna(pdb_path):
    """Extract C3' coordinates and sequence from a PDB file."""
    coords = []
    seq = []
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "C3'":
                continue

            resname = line[17:20].strip()
            chain = line[21]
            resseq = int(line[22:26])

            # Deduplicate by chain+resseq
            key = (chain, resseq)
            if key in seen:
                continue
            seen.add(key)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Get base from resname
            base = RESIDUE_MAP.get(resname[-1], RESIDUE_MAP.get(resname, None))
            if base is None:
                base = 'N'

            coords.append([x, y, z])
            seq.append(base)

    return np.array(coords, dtype=np.float64), ''.join(seq)


def circularize(coords, bond_length=5.9, n_steps=200, lr=0.01):
    """Simple annealing closure for linear RNA -> circular."""
    coords = coords.copy()
    for step in range(n_steps):
        diff = coords[0] - coords[-1]
        dist = np.linalg.norm(diff)
        if dist < bond_length * 1.1:
            break
        correction = lr * (dist - bond_length) * diff / max(dist, 1e-6)
        coords[0] -= correction * 0.5
        coords[-1] += correction * 0.5
        # Distribute small adjustment to neighbors
        for i in range(min(5, len(coords) - 1)):
            coords[i] -= correction * 0.02
            coords[-(i + 1)] += correction * 0.02
    return coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-cache', type=str, default='data/pdb_cache')
    parser.add_argument('--output', type=str, default='data/pdb_3d')
    parser.add_argument('--n-samples', type=int, default=7)
    parser.add_argument('--min-length', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=200)
    parser.add_argument('--max-closure', type=float, default=15.0)
    args = parser.parse_args()

    pdb_files = sorted(glob.glob(f'{args.pdb_cache}/*.pdb'))
    print(f'Found {len(pdb_files)} PDB files in {args.pdb_cache}')

    results = []
    for pdb_file in pdb_files:
        try:
            coords, seq = parse_pdb_rna(pdb_file)
        except Exception:
            continue

        L = len(coords)
        if L < args.min_length or L > args.max_length:
            continue

        # Check closure distance before circularization
        closure = np.linalg.norm(coords[0] - coords[-1])
        if closure > args.max_closure:
            continue

        # Circularize
        coords_circ = circularize(coords)
        closure_after = np.linalg.norm(coords_circ[0] - coords_circ[-1])

        pdb_id = os.path.basename(pdb_file).replace('.pdb', '')
        results.append({
            'id': pdb_id,
            'sequence': seq,
            'coords': coords_circ.astype(np.float32),
            'closure': closure_after,
        })
        print(f'  [{len(results)}/{args.n_samples}] {pdb_id} L={L} closure={closure:.1f}->{closure_after:.1f}A')

        if len(results) >= args.n_samples:
            break

    if not results:
        print('No valid samples found!')
        return

    # Save
    os.makedirs(f'{args.output}/coords', exist_ok=True)
    sequences = []
    for i, r in enumerate(results):
        seq_id = r['id']
        sequences.append({
            'id': seq_id,
            'sequence': r['sequence'],
            'length': len(r['sequence']),
            'pair_constraints': [],
        })
        # File name must match seq_id (load_pseudo_labels uses f'{seq_id}.npy')
        np.save(f'{args.output}/coords/{seq_id}.npy', r['coords'])

    with open(f'{args.output}/sequences.json', 'w') as f:
        json.dump(sequences, f, indent=2)

    metadata = {
        'n_samples': len(results),
        'source': 'pdb_cache_circularized',
        'circularization': 'annealing_closure',
    }
    with open(f'{args.output}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nSaved {len(results)} samples to {args.output}')


if __name__ == '__main__':
    main()
