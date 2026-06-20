#!/usr/bin/env python3
"""merge_expanded_dataset.py — 合并所有数据源到最终训练数据集。

Usage:
    python -m confluencia_3_0.core.circrna.torusfold.merge_expanded_dataset \
        --output data/circbase_real_3d_v2 \
        --sources data/isrnacirc_3d data/shape_3d data/pdb_3d data/medium_length_3d data/circbase_real_3d

每个 source 目录必须包含:
    sequences.json   — [{id, sequence, secondary_structure, pair_constraints}]
    coords/          — {id}.npy (L,3) C3' 坐标
    metadata.json    — 可选
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def load_source(source_dir):
    """Load a source dataset directory."""
    seq_path = os.path.join(source_dir, 'sequences.json')
    coords_dir = os.path.join(source_dir, 'coords')

    if not os.path.exists(seq_path):
        print(f"  WARNING: No sequences.json in {source_dir}, skipping")
        return [], []

    with open(seq_path, 'r') as f:
        seq_data = json.load(f)

    if not os.path.isdir(coords_dir):
        print(f"  WARNING: No coords/ dir in {source_dir}, skipping")
        return [], []

    sequences = []
    coords_list = []

    skipped = 0
    for item in seq_data:
        seq_id = item['id']
        coords_path = os.path.join(coords_dir, f"{seq_id}.npy")

        if not os.path.exists(coords_path):
            skipped += 1
            continue

        coords = np.load(coords_path)
        L = len(item.get('sequence', ''))

        # Validate: coords shape must match sequence length
        if coords.shape[0] != L:
            skipped += 1
            continue

        # Validate: coords must be (L, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            skipped += 1
            continue

        # Validate: no NaN or Inf
        if not np.isfinite(coords).all():
            skipped += 1
            continue

        sequences.append(item)
        coords_list.append(coords)

    if skipped > 0:
        print(f"  Skipped {skipped} invalid samples from {os.path.basename(source_dir)}")

    return sequences, coords_list


def deduplicate(sequences, coords_list):
    """Remove duplicate sequences, keeping the one from the highest-quality source."""
    source_priority = {
        'isrnacirc': 0,
        'isrnacirc_aug': 1,
        'shape_constrained': 2,
        'pdb_circularized': 3,
        'rna_puzzles': 4,
        'medium_length': 5,
        'synthetic': 6,
    }

    seen = {}
    for i, item in enumerate(sequences):
        seq = item['sequence']
        source = item.get('source', 'unknown')
        priority = source_priority.get(source, 99)

        if seq not in seen or priority < seen[seq][1]:
            seen[seq] = (i, priority)

    keep_indices = set(v[0] for v in seen.values())
    deduped_seqs = [sequences[i] for i in sorted(keep_indices)]
    deduped_coords = [coords_list[i] for i in sorted(keep_indices)]

    removed = len(sequences) - len(deduped_seqs)
    if removed > 0:
        print(f"  Deduplicated: removed {removed} duplicate sequences")

    return deduped_seqs, deduped_coords


def compute_metadata(sequences):
    """Compute metadata summary."""
    sources = Counter(item.get('source', 'unknown') for item in sequences)
    lengths = [len(item['sequence']) for item in sequences]

    # Length histogram
    bins = [(0, 100), (100, 200), (200, 300), (300, 500),
            (500, 700), (700, 1000), (1000, 1500), (1500, 3000)]
    length_hist = {}
    for lo, hi in bins:
        n = sum(1 for l in lengths if lo <= l < hi)
        if n > 0:
            length_hist[f"{lo}-{hi}"] = n

    # Pair constraints stats
    with_pairs = sum(1 for item in sequences if item.get('pair_constraints'))
    with_ss = sum(1 for item in sequences if item.get('secondary_structure', '.').strip('.'))

    return {
        'total': len(sequences),
        'length_range': [min(lengths), max(lengths)],
        'length_mean': float(np.mean(lengths)),
        'length_median': float(np.median(lengths)),
        'sources': dict(sources),
        'length_histogram': length_hist,
        'with_pair_constraints': with_pairs,
        'with_secondary_structure': with_ss,
    }


def merge_all_sources(source_dirs, output_dir, max_len=None):
    """Merge all source datasets into final training dataset."""
    print("=" * 60)
    print("  Merging Expanded circRNA 3D Dataset")
    print("=" * 60)

    all_sequences = []
    all_coords = []

    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            print(f"  Source not found: {src_dir}, skipping")
            continue

        seqs, coords = load_source(src_dir)
        src_name = os.path.basename(src_dir)
        print(f"  {src_name}: {len(seqs)} valid samples")
        all_sequences.extend(seqs)
        all_coords.extend(coords)

    print(f"\n  Total before dedup: {len(all_sequences)}")

    # Deduplicate
    all_sequences, all_coords = deduplicate(all_sequences, all_coords)
    print(f"  Total after dedup: {len(all_sequences)}")

    # Filter by max_len
    if max_len is not None:
        keep = [i for i, item in enumerate(all_sequences)
                if len(item['sequence']) <= max_len]
        all_sequences = [all_sequences[i] for i in keep]
        all_coords = [all_coords[i] for i in keep]
        print(f"  After max_len={max_len} filter: {len(all_sequences)}")

    if len(all_sequences) == 0:
        print("ERROR: No valid samples after merging!")
        return

    # Save to output directory
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    # Save sequences.json
    seq_path = os.path.join(output_dir, 'sequences.json')
    with open(seq_path, 'w') as f:
        json.dump(all_sequences, f, indent=2)

    # Save coords
    for item, coords in zip(all_sequences, all_coords):
        np.save(os.path.join(coords_dir, f"{item['id']}.npy"), coords)

    # Save metadata.json
    metadata = compute_metadata(all_sequences)
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Final dataset: {len(all_sequences)} samples")
    print(f"  Saved to: {output_dir}")
    print(f"\n  Sources:")
    for src, count in sorted(metadata['sources'].items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    print(f"\n  Length distribution:")
    for rng, count in metadata['length_histogram'].items():
        bar = '#' * (count // 100)
        print(f"    {rng:>10s} nt: {count:>5d} {bar}")
    print(f"\n  With pair_constraints: {metadata['with_pair_constraints']}/{metadata['total']}")
    print(f"  With secondary_structure: {metadata['with_secondary_structure']}/{metadata['total']}")


def main():
    parser = argparse.ArgumentParser(description='Merge expanded circRNA datasets')
    parser.add_argument('--output', type=str, default='data/circbase_real_3d_v2',
                        help='Output directory')
    parser.add_argument('--sources', nargs='+', default=[
                        'data/isrnacirc_3d',
                        'data/shape_3d',
                        'data/pdb_3d',
                        'data/medium_length_3d',
                        'data/circbase_real_3d',
                        ], help='Source directories to merge')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Maximum sequence length filter')
    args = parser.parse_args()

    merge_all_sources(args.sources, args.output, args.max_len)


if __name__ == '__main__':
    main()
