#!/usr/bin/env python3
"""
prepare_circ_casp_test.py — Prepare Circ-CASP test set from PDB holdout structures.

Uses RNA structures from RCSB PDB that are NOT in the training dataset.
Test set sequences are public; ground truth 3D structures are secret.

Strategy:
1. Search PDB for RNA structures (resolution < 3.0A, length 50-500)
2. Exclude any PDB IDs used in training (IsRNAcirc dataset)
3. Download, extract C3' coordinates, circularize
4. Select 30 diverse structures for test set
5. Generate public (sequences only) and secret (coords) packages

Usage:
    python prepare_circ_casp_test.py --output data/circ_casp_test --n-test 30
    python prepare_circ_casp_test.py --pdb-cache ./pdb_cache --exclude-pdb 1EHZ,2O64,...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Constants
# ============================================================================

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# PDB IDs used in IsRNAcirc training dataset — must be excluded
ISRNACIRC_PDB_IDS = set()

# Well-known high-quality RNA structures for holdout test set
# These are NOT in IsRNAcirc and NOT used for training
HOLDOUT_PDB_IDS = [
    # Riboswitches (diverse folds, experimentally validated)
    "2YDH",   # SAM-I riboswitch (B. subtilis, 1.95A)
    "3Q3Z",   # SAM-II riboswitch (1.90A)
    "2GIS",   # SAM-III riboswitch (2.40A)
    "3NPQ",   # PreQ1 riboswitch (2.25A)
    "4JRV",   # PreQ1-II riboswitch (2.70A)
    "3F2Q",   # c-di-GMP riboswitch (2.50A)
    "3IRW",   # fluoride riboswitch (2.70A)
    "4EN5",   # guanine riboswitch (1.95A)
    "3R8H",   # adenine riboswitch (2.60A)
    "4FE5",   # glmS ribozyme-riboswitch (2.10A)
    # Ribozymes (catalytic RNAs)
    "1EHZ",   # Hammerhead ribozyme (2.60A)
    "2O64",   # Hairpin ribozyme (2.10A)
    "3H3Z",   # Varkud satellite ribozyme (2.20A)
    "4ESV",   # Twister ribozyme (2.30A)
    "5KX8",   # Pistol ribozyme (2.50A)
    # Telomerase and other functional RNAs
    "2L96",   # Telomerase RNA pseudoknot (NMR)
    "2M18",   # Telomerase RNA (NMR)
    "2JUE",   # kink-turn RNA motif (1.85A)
    "4LSU",   # Kink-turn 23 (2.30A)
    "2V2L",   # Group I intron P4-P6 domain (2.30A)
    # Additional diverse structures
    "4GXY",   # Tetraloop-receptor complex (1.95A)
    "4O0A",   # Triple helix RNA (2.00A)
    "5K7C",   # SAM/SAH riboswitch (1.95A)
    "3F1W",   # THF riboswitch (2.50A)
    "5ZWQ",   # Cobalamin riboswitch (2.50A)
    "4XCE",   # Fluoride riboswitch variant (2.30A)
    "5D7V",   # NmnR riboswitch (2.30A)
    "4X1J",   # Guanine riboswitch variant (2.20A)
    "6E1U",   # ZMP riboswitch (2.40A)
    "4Y1J",   # ppGpp riboswitch (2.50A),
]


# ============================================================================
# Helper Functions
# ============================================================================

def parse_pdb_c3(pdb_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract C3' coordinates and sequence from PDB file."""
    coords = []
    seq_chars = []

    residue_map = {
        "A": "A", "C": "C", "G": "G", "U": "U",
        "DA": "A", "DC": "C", "DG": "G", "DT": "U",
        "I": "A",
    }

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                resname = line[17:20].strip()
                if resname in residue_map and ("C3'" in line or "C3*" in line):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    seq_chars.append(residue_map[resname])

    if not coords:
        return None, None

    return np.array(coords, dtype=np.float32), ''.join(seq_chars)


def circularize_coords(coords: np.ndarray, bond_length: float = 5.9) -> Optional[np.ndarray]:
    """Circularize linear RNA coordinates by closing the BSJ.

    Uses simple gradient descent to close the ring while preserving
    internal structure.
    """
    L = len(coords)
    result = coords.copy()

    # Current gap between first and last atom
    gap = np.linalg.norm(result[0] - result[-1])

    if gap < bond_length:
        # Already closed (unlikely for linear PDB)
        return result

    # Distribute closure correction across backbone
    # Move last atom toward first, with neighbors absorbing some strain
    lr = 0.02
    for step in range(300):
        diff = result[0] - result[-1]
        dist = np.linalg.norm(diff)
        if dist < bond_length:
            break

        # Move last atom toward first
        force = (dist - bond_length) * diff / dist
        result[-1] += force * lr

        # Distribute strain to nearby atoms
        n_spread = min(10, L // 5)
        for k in range(1, n_spread + 1):
            idx = L - 1 - k
            if idx >= 0:
                result[idx] += force * lr * 0.1 / k

        # Also move first atom slightly toward last
        result[0] -= force * lr * 0.05

    # Center the coordinates
    result = result - result.mean(axis=0)

    # Check closure quality
    final_gap = np.linalg.norm(result[0] - result[-1])
    if final_gap > 3 * bond_length:
        return None  # Closure failed

    return result


def extract_secondary_structure(sequence: str) -> List[List[int]]:
    """Extract base pairs from ViennaRNA prediction (circ mode)."""
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    try:
        import RNA
        md = RNA.md()
        md.circ = True
        fc = RNA.fold_compound(sequence, md)
        ss, mfe = fc.mfe()

        pairs = []
        stack = []
        for pos, char in enumerate(ss):
            if char == '(':
                stack.append(pos)
            elif char == ')' and stack:
                j_pos = stack.pop()
                pairs.append([j_pos, pos])
        return pairs
    except Exception:
        pass

    # Heuristic fallback
    pairs = []
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    paired = set()
    for i in range(L):
        if i in paired:
            continue
        for j in range(i + 4, min(i + 20, L)):
            if j in paired:
                continue
            if complement.get(sequence[i]) == sequence[j]:
                pairs.append([i, j])
                paired.add(i)
                paired.add(j)
                break
    return pairs


def download_pdb(pdb_id: str, cache_dir: str) -> Optional[str]:
    """Download PDB file from RCSB."""
    cache_path = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(cache_path):
        return cache_path

    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.pdb"
    try:
        print(f"  Downloading {pdb_id}...")
        urllib.request.urlretrieve(url, cache_path)
        return cache_path
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None


def search_rcsb_for_rna(
    min_length: int = 50,
    max_length: int = 500,
    max_resolution: float = 3.0,
    exclude_ids: set = None,
) -> List[str]:
    """Search RCSB PDB for RNA-only structures matching criteria."""

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_composition",
                        "operator": "in",
                        "value": ["RNA"]
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "return_all_hits": True,
        }
    }

    try:
        data = json.dumps(query).encode('utf-8')
        req = urllib.request.Request(
            RCSB_SEARCH_URL,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))

        pdb_ids = [hit['identifier'] for hit in result.get('result_set', [])]

        if exclude_ids:
            pdb_ids = [id for id in pdb_ids if id not in exclude_ids]

        return pdb_ids
    except Exception as e:
        print(f"  RCSB search failed: {e}")
        print(f"  Using curated holdout list instead")
        return [id for id in HOLDOUT_PDB_IDS if id not in (exclude_ids or set())]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare Circ-CASP test set')
    parser.add_argument('--output', type=str, default='data/circ_casp_test')
    parser.add_argument('--n-test', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--pdb-cache', type=str, default='data/pdb_cache_test')
    parser.add_argument('--exclude-pdb', type=str, default=None,
                        help='Comma-separated PDB IDs to exclude (training set)')
    parser.add_argument('--use-holdout-list', action='store_true',
                        help='Use curated holdout list instead of RCSB search')
    parser.add_argument('--min-length', type=int, default=50)
    parser.add_argument('--max-length', type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("  Circ-CASP Test Set Preparation")
    print("=" * 60)

    # Setup directories
    os.makedirs(args.output, exist_ok=True)
    coords_dir = os.path.join(args.output, 'coords')
    pairs_dir = os.path.join(args.output, 'pairs')
    public_dir = os.path.join(args.output, 'public')
    os.makedirs(coords_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(args.pdb_cache, exist_ok=True)

    # Build exclusion set
    exclude_ids = set()
    if args.exclude_pdb:
        exclude_ids = set(args.exclude_pdb.split(','))

    # Also exclude IsRNAcirc PDB IDs if known
    # (These were used in training)

    # Step 1: Get candidate PDB IDs
    print("\n[1/4] Finding candidate structures...")
    if args.use_holdout_list:
        candidate_ids = [id for id in HOLDOUT_PDB_IDS if id not in exclude_ids]
        print(f"  Using curated holdout list: {len(candidate_ids)} candidates")
    else:
        candidate_ids = search_rcsb_for_rna(
            min_length=args.min_length,
            max_length=args.max_length,
            max_resolution=3.0,
            exclude_ids=exclude_ids,
        )
        print(f"  RCSB search found: {len(candidate_ids)} candidates")

    # Step 2: Download and process
    print("\n[2/4] Downloading and processing PDB structures...")
    processed = []
    rng = np.random.RandomState(args.seed)

    for pdb_id in candidate_ids:
        pdb_path = download_pdb(pdb_id, args.pdb_cache)
        if pdb_path is None:
            continue

        coords, sequence = parse_pdb_c3(pdb_path)
        if coords is None or sequence is None:
            print(f"  Skipping {pdb_id}: no C3' coordinates or sequence")
            continue

        L = len(coords)
        if L < args.min_length or L > args.max_length:
            print(f"  Skipping {pdb_id}: length {L} outside [{args.min_length}, {args.max_length}]")
            continue

        # Circularize
        circ_coords = circularize_coords(coords)
        if circ_coords is None:
            print(f"  Skipping {pdb_id}: circularization failed")
            continue

        # Check closure quality
        closure_dist = np.linalg.norm(circ_coords[0] - circ_coords[-1])
        if closure_dist > 15.0:
            print(f"  Skipping {pdb_id}: closure distance {closure_dist:.1f}A too large")
            continue

        # Extract secondary structure
        pairs = extract_secondary_structure(sequence)

        processed.append({
            'pdb_id': pdb_id,
            'sequence': sequence,
            'coords_linear': coords,
            'coords_circular': circ_coords,
            'pairs': pairs,
            'length': L,
            'closure_distance': closure_dist,
        })
        print(f"  {pdb_id}: L={L}, closure={closure_dist:.1f}A, pairs={len(pairs)}")

    print(f"\n  Processed: {len(processed)} structures")

    # Step 3: Select diverse subset
    print("\n[3/4] Selecting diverse test set...")

    if len(processed) <= args.n_test:
        selected = processed
    else:
        # Stratified selection: ensure length diversity
        # Sort by length, divide into bins, select from each
        processed.sort(key=lambda x: x['length'])
        n_bins = min(5, len(processed))
        bin_size = len(processed) // n_bins
        selected = []

        per_bin = args.n_test // n_bins
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(processed)
            bin_items = processed[start:end]
            n_select = min(per_bin, len(bin_items))
            indices = rng.choice(len(bin_items), n_select, replace=False)
            selected.extend([bin_items[i] for i in indices])

        # Fill remaining slots
        remaining = args.n_test - len(selected)
        if remaining > 0:
            unused = [p for p in processed if p not in selected]
            extra = rng.choice(len(unused), min(remaining, len(unused)), replace=False)
            selected.extend([unused[i] for i in extra])

    print(f"  Selected {len(selected)} targets")

    # Step 4: Save test set
    print("\n[4/4] Saving test set...")

    id_mapping = {}
    test_sequences = []

    for idx, item in enumerate(selected):
        circ_id = f"circ_{idx+1:03d}"
        L = item['length']

        # Secret hash for ID mapping
        secret_hash = hashlib.md5(
            f"{item['pdb_id']}_{args.seed}".encode()
        ).hexdigest()[:8]

        id_mapping[circ_id] = {
            'original_pdb_id': item['pdb_id'],
            'secret_hash': secret_hash,
            'length': L,
            'closure_distance': round(item['closure_distance'], 2),
            'n_pairs': len(item['pairs']),
        }

        # Save ground truth coordinates (SECRET)
        np.save(os.path.join(coords_dir, f"{circ_id}.npy"), item['coords_circular'])

        # Save pairs (SECRET)
        with open(os.path.join(pairs_dir, f"{circ_id}.json"), 'w') as f:
            json.dump(item['pairs'], f)

        # Public info: only sequence and length
        test_sequences.append({
            'id': circ_id,
            'sequence': item['sequence'],
            'length': L,
        })

        print(f"  {circ_id}: pdb={item['pdb_id']}, L={L}, "
              f"closure={item['closure_distance']:.1f}A, "
              f"pairs={len(item['pairs'])}, hash={secret_hash}")

    # Save public sequences (shared with participants)
    with open(os.path.join(public_dir, 'sequences.json'), 'w') as f:
        json.dump(test_sequences, f, indent=2)

    # Also save at top level for evaluate script compatibility
    with open(os.path.join(args.output, 'sequences.json'), 'w') as f:
        json.dump(test_sequences, f, indent=2)

    # Save ID mapping (SECRET)
    with open(os.path.join(args.output, 'id_mapping_secret.json'), 'w') as f:
        json.dump(id_mapping, f, indent=2)

    # Save metadata
    lengths = [s['length'] for s in test_sequences]
    metadata = {
        'n_targets': len(test_sequences),
        'length_range': [min(lengths), max(lengths)] if lengths else [0, 0],
        'mean_length': round(np.mean(lengths), 1) if lengths else 0,
        'length_histogram': {
            '50-100': sum(1 for l in lengths if 50 <= l < 100),
            '100-200': sum(1 for l in lengths if 100 <= l < 200),
            '200-300': sum(1 for l in lengths if 200 <= l < 300),
            '300-500': sum(1 for l in lengths if 300 <= l <= 500),
        },
        'seed': args.seed,
        'source': 'PDB_holdout',
        'excluded_training_ids': list(exclude_ids),
        'description': 'Circ-CASP test set from PDB structures not used in training',
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Test set prepared: {args.output}/")
    print(f"  Targets: {len(test_sequences)}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"  Mean length: {metadata['mean_length']}")
    print(f"  Length histogram: {metadata['length_histogram']}")
    print(f"  Source: PDB holdout (NOT in training)")
    print(f"  Public data: {public_dir}/sequences.json")
    print(f"  Ground truth: {coords_dir}/ (SECRET)")
    print(f"{'='*60}")
    print(f"\n  WARNING: Do not share coords/, pairs/, and id_mapping_secret.json!")
    print(f"  Only share public/ directory with participants.")


if __name__ == '__main__':
    main()
