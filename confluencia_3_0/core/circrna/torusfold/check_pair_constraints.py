#!/usr/bin/env python3
"""
check_pair_constraints.py — Diagnose pair_constraints coverage in merged dataset.

Usage:
    python check_pair_constraints.py --data data/circrna_3d_merged
"""

import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Check pair_constraints coverage")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to merged dataset directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print details for samples without pairs")
    args = parser.parse_args()

    seq_path = os.path.join(args.data, "sequences.json")
    if not os.path.exists(seq_path):
        print(f"ERROR: sequences.json not found at {seq_path}")
        return

    with open(seq_path, "r") as f:
        data = json.load(f)

    print("=" * 60)
    print("  pair_constraints Coverage Report")
    print("=" * 60)
    print(f"  Dataset: {args.data}")
    print(f"  Total samples: {len(data)}")

    # Count by coverage type
    has_pairs = []
    has_real_ss = []
    has_both = []
    has_none = []

    for entry in data:
        seq_id = entry.get("id", "unknown")
        length = entry.get("length", 0)
        sequence = entry.get("sequence", "")
        ss = entry.get("secondary_structure", "")
        pairs = entry.get("pair_constraints", [])
        source = entry.get("source", "unknown")

        # Check pair_constraints
        pairs_valid = len(pairs) > 0

        # Check secondary_structure (not all dots)
        ss_valid = ss and ss != "." * length

        if pairs_valid:
            has_pairs.append(entry)
        if ss_valid:
            has_real_ss.append(entry)
        if pairs_valid and ss_valid:
            has_both.append(entry)
        if not pairs_valid and not ss_valid:
            has_none.append(entry)

    print(f"\n  Coverage Summary:")
    print(f"    With pair_constraints: {len(has_pairs)} ({100*len(has_pairs)/len(data):.1f}%)")
    print(f"    With real SS: {len(has_real_ss)} ({100*len(has_real_ss)/len(data):.1f}%)")
    print(f"    With both: {len(has_both)} ({100*len(has_both)/len(data):.1f}%)")
    print(f"    With neither: {len(has_none)} ({100*len(has_none)/len(data):.1f}%)")

    # Breakdown by source
    print(f"\n  Breakdown by source:")
    source_counts = Counter(e.get("source", "unknown") for e in data)
    source_with_pairs = Counter(e.get("source", "unknown") for e in has_pairs)
    source_with_none = Counter(e.get("source", "unknown") for e in has_none)

    for source in sorted(source_counts.keys()):
        total = source_counts[source]
        with_pairs = source_with_pairs.get(source, 0)
        without = source_with_none.get(source, 0)
        pct = 100 * with_pairs / total if total > 0 else 0
        print(f"    {source}: {with_pairs}/{total} ({pct:.1f}% have pairs, {without} missing)")

    # Length distribution of samples without pairs
    if has_none:
        print(f"\n  Length distribution (samples missing pairs):")
        length_bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 5000)]
        for lo, hi in length_bins:
            count = sum(1 for e in has_none if lo <= e.get("length", 0) < hi)
            if count > 0:
                print(f"    {lo}-{hi} nt: {count}")

    if args.verbose and has_none:
        print(f"\n  Samples without pairs (first 20):")
        for entry in has_none[:20]:
            print(f"    {entry.get('id', 'unknown')}: L={entry.get('length', 0)}, "
                  f"source={entry.get('source', 'unknown')}")

    # Diagnosis
    print(f"\n" + "=" * 60)
    if len(has_pairs) == 0:
        print("  CRITICAL: No samples have pair_constraints!")
        print("  All schemes will train on trivial helix structures.")
        print("  Action: Regenerate data with updated pipelines.")
    elif len(has_pairs) < len(data) * 0.5:
        print(f"  WARNING: Only {len(has_pairs)} samples ({100*len(has_pairs)/len(data):.1f}%) have pairs.")
        print("  Models will have weak pair supervision.")
        print("  Action: Check data generation scripts for missing pair_constraints.")
    else:
        print("  OK: Majority of samples have pair_constraints.")
    print("=" * 60)


if __name__ == "__main__":
    main()