#!/usr/bin/env python3
"""check_merged_data.py — Validate merged circRNA 3D dataset quality."""

import json
import os
import sys
import numpy as np


def check_data(data_dir):
    seq_path = os.path.join(data_dir, "sequences.json")
    coords_dir = os.path.join(data_dir, "coords")

    if not os.path.exists(seq_path):
        print(f"ERROR: {seq_path} not found")
        return

    with open(seq_path) as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    n_mismatch = 0
    n_nan = 0
    n_inf = 0
    n_empty_pairs = 0
    n_bad_pairs = 0
    n_missing_coords = 0
    n_zero_coords = 0

    lengths = []
    pair_counts = []
    sources = {}

    for i, item in enumerate(data):
        seq = item["sequence"]
        seq_id = item["id"]
        length = item.get("length", len(seq))
        source = item.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        lengths.append(length)

        # Check coords
        coord_path = os.path.join(coords_dir, f"{seq_id}.npy")
        if not os.path.exists(coord_path):
            n_missing_coords += 1
            continue

        coords = np.load(coord_path)
        if coords.shape[0] != len(seq):
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"  MISMATCH {seq_id}: seq={len(seq)} coords={coords.shape[0]}")
        if np.isnan(coords).any():
            n_nan += 1
            if n_nan <= 5:
                print(f"  NaN {seq_id}: {np.isnan(coords).sum()} values")
        if np.isinf(coords).any():
            n_inf += 1
            if n_inf <= 5:
                print(f"  Inf {seq_id}: {np.isinf(coords).sum()} values")
        if np.abs(coords).max() == 0:
            n_zero_coords += 1

        # Check pairs
        pairs = item.get("pair_constraints", [])
        pair_counts.append(len(pairs))
        if not pairs:
            n_empty_pairs += 1
        else:
            for p in pairs:
                if p[0] < 0 or p[0] >= length or p[1] < 0 or p[1] >= length:
                    n_bad_pairs += 1

        if (i + 1) % 2000 == 0:
            print(f"  Checked {i+1}/{len(data)}...")

    print(f"\n--- Summary ---")
    print(f"Length mismatch (seq vs coords): {n_mismatch}")
    print(f"NaN in coords: {n_nan}")
    print(f"Inf in coords: {n_inf}")
    print(f"All-zero coords: {n_zero_coords}")
    print(f"Missing coords: {n_missing_coords}")
    print(f"Empty pair_constraints: {n_empty_pairs}")
    print(f"Out-of-bounds pairs: {n_bad_pairs}")
    print(f"\nLength stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"Pair stats: min={min(pair_counts)}, max={max(pair_counts)}, mean={np.mean(pair_counts):.1f}")
    print(f"\nSources:")
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    # Show first 3 samples in detail
    print(f"\n--- First 3 samples ---")
    for i in range(min(3, len(data))):
        item = data[i]
        seq_id = item["id"]
        seq = item["sequence"]
        coord_path = os.path.join(coords_dir, f"{seq_id}.npy")
        if os.path.exists(coord_path):
            coords = np.load(coord_path)
            print(f"\n  {seq_id}:")
            print(f"    seq length: {len(seq)}, declared: {item.get('length')}")
            print(f"    coords shape: {coords.shape}")
            print(f"    coords range: [{coords.min():.2f}, {coords.max():.2f}]")
            print(f"    has NaN: {np.isnan(coords).any()}, has Inf: {np.isinf(coords).any()}")
            print(f"    n_pairs: {len(item.get('pair_constraints', []))}")
            print(f"    source: {item.get('source')}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/circrna_3d_merged"
    check_data(data_dir)
