#!/usr/bin/env python3
"""
merge_expanded_dataset.py — Merge all circRNA 3D data sources into a unified training dataset.

Loads data from:
  1. IsRNAcirc (build_training_dataset.py output)
  2. icSHAPE-constrained (shape_to_3d_pipeline.py output)
  3. PDB circularized (pdb_rna_circularize.py output)
  4. Medium-length (generate_medium_length_dataset.py output)

Normalizes inconsistent fields across sources:
  - Extracts pair_constraints from secondary_structure when missing (SHAPE, PDB)
  - Normalizes pair_constraints to [[i,j], ...] format
  - Re-indexes all entries as merged_NNNNN
  - Validates coords shape matches sequence length

Deduplicates by sequence (keeping higher-quality source), validates entries,
and saves in the format compatible with train_all_schemes.py load_pseudo_labels().

Output:
    <output_dir>/
    ├── sequences.json     # All sequences with unified fields
    ├── coords/            # .npy coordinate arrays
    │   ├── merged_0000.npy
    │   └── ...
    └── metadata.json      # Source breakdown + length histogram

Usage:
    python merge_expanded_dataset.py \
        --isrnacirc-dir data/circbase_real_3d \
        --shape-dir data/shape_3d \
        --pdb-dir data/pdb_circularized \
        --medium-dir data/medium_length_3d \
        --output data/circrna_3d_merged
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np


# Source quality priority (lower = higher quality, kept on dedup)
SOURCE_PRIORITY = {
    "isrnacirc": 0,
    "isrnacirc_aug": 1,
    "shape_experimental": 2,
    "shape_expanded": 2,
    "rfam_consensus": 3,
    "circbase_real": 4,
    "pdb_circularized": 5,
    "pdb_circularized_aug": 6,
    "medium_synth": 7,
    "synthetic": 8,
}

# Source confidence weights for training loss weighting
# IMPORTANT: Augmented samples have lower confidence because they are noisy copies
# of original structures, not independent high-quality predictions.
SOURCE_CONFIDENCE = {
    "pdb_circularized": 1.0,       # Real PDB structure, highest quality
    "pdb_circularized_aug": 0.5,   # NOISY COPY - reduced from 0.95 to prevent overfitting
    "af3_predicted": 1.0,          # AlphaFold3 prediction
    "shape_experimental": 0.9,     # SHAPE-constrained folding
    "shape_expanded": 0.85,        # SHAPE-constrained (expanded sources)
    "rfam_consensus": 0.8,         # Rfam consensus structure
    "isrnacirc": 0.7,              # Predicted but validated
    "isrnacirc_aug": 0.35,         # NOISY COPY - reduced from 0.65
    "circbase_real": 0.5,          # ViennaRNA predicted
    "medium_synth": 0.4,           # Synthetic with constraints
    "synthetic": 0.3,              # Pure synthetic
}


def _extract_pairs_from_dot_bracket(ss: str) -> list:
    """Extract base pairs from dot-bracket notation. Supports ( ) [ ]."""
    pairs = []
    stack_a = []  # for ( )
    stack_b = []  # for [ ]
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


def _normalize_pair_constraints(pairs: list, ss: str, length: int, sequence: str = "", seq_id: str = "") -> list:
    """Normalize pair_constraints to [[i,j], ...] format.

    If pairs is empty but ss has brackets, extract from ss.
    If both empty, use ViennaRNA circ-mode fold as merge-time fallback only.
    This fallback is intentional: it runs once during merge, not during training.
    """
    normalized = []
    for p in pairs:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            i, j = int(p[0]), int(p[1])
            if 0 <= i < length and 0 <= j < length:
                normalized.append([i, j])

    # If no pairs from constraints, try extracting from dot-bracket
    if not normalized and ss and ss != "." * length:
        normalized = _extract_pairs_from_dot_bracket(ss)

    # Merge-time ViennaRNA fallback (runs once, results saved to disk)
    if not normalized and sequence and len(sequence) == length:
        try:
            import RNA
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            (ss_vrna, _) = fc.mfe()
            normalized = _extract_pairs_from_dot_bracket(ss_vrna)
            if normalized:
                print(f"    ViennaRNA fallback: {seq_id} (L={length}) -> {len(normalized)} pairs")
        except ImportError:
            pass

    return normalized


def load_source(source_name: str, source_dir: str) -> Tuple[list, int, int]:
    """Load sequences.json and validate coords from a source directory.

    Returns:
        (entries, n_loaded, n_skipped)
    """
    seq_path = os.path.join(source_dir, "sequences.json")
    coords_dir = os.path.join(source_dir, "coords")

    if not os.path.exists(seq_path):
        print(f"  [{source_name}] sequences.json not found, skipping")
        return [], 0, 0

    with open(seq_path, "r") as f:
        raw_entries = json.load(f)

    print(f"  [{source_name}] Loading {len(raw_entries)} entries...")

    entries = []
    n_skipped = 0
    n_no_pairs = 0

    for idx, item in enumerate(raw_entries):
        seq_id = item.get("id", "")
        sequence = item.get("sequence", "")
        length = item.get("length", 0)

        # Use sequence length if declared length is wrong
        if len(sequence) > 0 and length != len(sequence):
            length = len(sequence)

        if not seq_id or not sequence or length == 0:
            n_skipped += 1
            continue

        # Normalize pair_constraints
        ss = item.get("secondary_structure", "")
        raw_pairs = item.get("pair_constraints", [])
        pair_constraints = _normalize_pair_constraints(raw_pairs, ss, length, sequence, seq_id)

        if not pair_constraints:
            n_no_pairs += 1

        # If ss is missing, construct from pairs or use all-dots
        if not ss:
            ss = "." * length

        # Find coords file
        coord_file = os.path.join(coords_dir, f"{seq_id}.npy")
        if not os.path.exists(coord_file):
            n_skipped += 1
            continue

        # Determine source name for confidence lookup
        source_name_disp = item.get("source", source_name)

        entry = {
            "id": seq_id,
            "sequence": sequence,
            "secondary_structure": ss,
            "pair_constraints": pair_constraints,
            "length": length,
            "source": source_name_disp,
            "confidence": SOURCE_CONFIDENCE.get(source_name_disp, 0.3),
            "mfe": item.get("mfe", None),
            "coords_path": coord_file,
        }

        # Preserve source-specific metadata
        for key in ("structure_type", "has_real_ss", "pdb_id", "chain_id",
                     "closure_distance", "bond_rmsd", "chrom",
                     "reactivities_mean", "n_pairs",
                     "augmentation_index"):
            if key in item and item[key] is not None:
                entry[key] = item[key]

        entries.append(entry)

    print(f"  [{source_name}] Loaded {len(entries)}/{len(raw_entries)} "
          f"({n_skipped} skipped, {n_no_pairs} no pairs)")
    return entries, len(entries), n_skipped


def validate_entry(entry: dict) -> Optional[str]:
    """Validate a single entry. Returns error message or None if valid."""
    seq = entry.get("sequence", "")
    length = entry.get("length", 0)

    if len(seq) != length:
        return f"sequence length {len(seq)} != declared length {length}"

    coord_path = entry.get("coords_path", "")
    if not os.path.exists(coord_path):
        return f"coords file missing: {coord_path}"

    try:
        coords = np.load(coord_path)
        if coords.shape[0] != length:
            return f"coords shape[0]={coords.shape[0]} != length {length}"
        if coords.ndim != 2 or coords.shape[1] != 3:
            return f"coords shape {coords.shape}, expected (L, 3)"
        if not np.isfinite(coords).all():
            return "coords contain NaN or Inf"
    except Exception as e:
        return f"coords load error: {e}"

    for p in entry.get("pair_constraints", []):
        if p[0] < 0 or p[0] >= length or p[1] < 0 or p[1] >= length:
            return f"pair_constraint {p} out of bounds for length {length}"

    return None


def deduplicate(entries: list) -> list:
    """Deduplicate entries by (sequence, source) to preserve augmentations.

    Augmented samples (pdb_circularized_aug, isrnacirc_aug) have the same
    sequence but different coords (rotation + noise), which are valuable
    for training. We deduplicate by (sequence, source) instead of just
    sequence to keep these augmented variants.

    Within each (sequence, source) group, keep the entry with best quality
    metrics (lower augmentation_index, lower closure_distance, etc).
    """
    # Group by (sequence, source)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, entry in enumerate(entries):
        seq = entry["sequence"]
        source = entry.get("source", "unknown")
        key = (seq, source)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # For each group, keep all entries (augmentations are preserved)
    # But for duplicate entries within same source, prefer:
    # 1. augmentation_index=0 or None (original) over aug copies
    # 2. lower closure_distance (better circularization quality)
    keep_indices = []
    for key, indices in groups.items():
        if len(indices) == 1:
            keep_indices.append(indices[0])
        else:
            # Sort by quality: prefer original (aug_idx=0 or None) then by quality metrics
            best = indices[0]
            best_aug = entries[best].get("augmentation_index", 0) or 0
            best_closure = entries[best].get("closure_distance", 999) or 999

            for idx in indices[1:]:
                entry = entries[idx]
                aug_idx = entry.get("augmentation_index", 0) or 0
                closure = entry.get("closure_distance", 999) or 999

                # Keep if better quality
                if aug_idx < best_aug or (aug_idx == best_aug and closure < best_closure):
                    best = idx
                    best_aug = aug_idx
                    best_closure = closure

            # For augmented sources, keep ALL variants (they have different coords)
            source = key[1]
            if source in ("pdb_circularized_aug", "isrnacirc_aug", "medium_aug"):
                keep_indices.extend(indices)  # Keep all augmentations
            else:
                keep_indices.append(best)  # Keep only best for non-aug sources

    keep_indices = sorted(set(keep_indices))
    unique = [entries[i] for i in keep_indices]
    n_dup = len(entries) - len(unique)

    if n_dup > 0:
        print(f"  Removed {n_dup} duplicate entries (kept unique (seq,source) pairs)")
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Merge all circRNA 3D data sources into unified training dataset"
    )
    parser.add_argument("--isrnacirc-dir", type=str, default="",
                        help="IsRNAcirc dataset directory")
    parser.add_argument("--shape-dir", type=str, default="",
                        help="SHAPE-constrained dataset directory")
    parser.add_argument("--pdb-dir", type=str, default="",
                        help="PDB circularized dataset directory")
    parser.add_argument("--medium-dir", type=str, default="",
                        help="Medium-length dataset directory")
    parser.add_argument("--rfam-dir", type=str, default="",
                        help="Rfam consensus structure dataset directory")
    parser.add_argument("--shape-exp-dir", type=str, default="",
                        help="Expanded SHAPE dataset directory")
    parser.add_argument("--rosetta-dir", type=str, default="",
                        help="Rosetta predicted dataset directory")
    parser.add_argument("--circbase-dir", type=str, default="",
                        help="CircBase real 3D dataset directory")
    parser.add_argument("--pdb3d-dir", type=str, default="",
                        help="PDB 3D dataset directory (non-circularized)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged dataset")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip coordinate validation (faster)")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Minimum sequence length to include")
    parser.add_argument("--max-length", type=int, default=10000,
                        help="Maximum sequence length to include")
    args = parser.parse_args()

    print("=" * 60)
    print("  Merge Expanded circRNA 3D Dataset")
    print("=" * 60)

    t0 = time.time()

    # ── Load all sources ──────────────────────────────────────
    print("\n[1/5] Loading data sources...")

    # Load in priority order so dedup keeps earlier (higher quality) entries
    all_entries = []
    sources = [
        ("isrnacirc", args.isrnacirc_dir),
        ("shape", args.shape_dir),
        ("shape_expanded", args.shape_exp_dir),
        ("pdb", args.pdb_dir),
        ("pdb3d", args.pdb3d_dir),
        ("rosetta", args.rosetta_dir),
        ("circbase", args.circbase_dir),
        ("rfam", args.rfam_dir),
        ("medium", args.medium_dir),
    ]

    for name, path in sources:
        if path and os.path.exists(path):
            entries, _, _ = load_source(name, path)
            all_entries.extend(entries)
        elif path:
            print(f"  [{name}] Directory not found: {path}")

    print(f"\n  Total loaded: {len(all_entries)}")

    if not all_entries:
        print("  ERROR: No entries loaded from any source!")
        sys.exit(1)

    # ── Filter by length ──────────────────────────────────────
    print(f"\n[2/5] Filtering by length ({args.min_length}-{args.max_length} nt)...")
    before = len(all_entries)
    all_entries = [e for e in all_entries
                   if args.min_length <= e["length"] <= args.max_length]
    print(f"  Kept {len(all_entries)}/{before}")

    # ── Deduplicate ───────────────────────────────────────────
    print("\n[3/5] Deduplicating by sequence...")
    all_entries = deduplicate(all_entries)
    print(f"  Unique sequences: {len(all_entries)}")

    # ── Validate ──────────────────────────────────────────────
    print("\n[4/5] Validating entries...")
    valid_entries = []
    n_invalid = 0

    for i, entry in enumerate(all_entries):
        if not args.skip_validation:
            error = validate_entry(entry)
            if error:
                if n_invalid < 10:
                    print(f"  INVALID {entry['id']}: {error}")
                n_invalid += 1
                continue
        valid_entries.append(entry)

        if (i + 1) % 2000 == 0:
            print(f"  Validated: {i+1}/{len(all_entries)}")

    if n_invalid > 10:
        print(f"  ... and {n_invalid - 10} more invalid entries")
    print(f"  Valid: {len(valid_entries)}, Invalid: {n_invalid}")

    all_entries = valid_entries

    # ── Save merged dataset ───────────────────────────────────
    print(f"\n[5/5] Saving merged dataset ({len(all_entries)} samples)...")

    output_dir = args.output
    coords_dir = os.path.join(output_dir, "coords")
    os.makedirs(coords_dir, exist_ok=True)

    # Re-index with merged_ prefix
    json_entries = []
    n_skipped_mismatch = 0
    for i, entry in enumerate(all_entries):
        new_id = f"merged_{i:05d}"

        # Copy coords and VERIFY length matches sequence
        src_coords = entry["coords_path"]
        dst_coords = os.path.join(coords_dir, f"{new_id}.npy")
        coords = np.load(src_coords)

        seq_len = len(entry["sequence"])
        coord_len = coords.shape[0]
        if seq_len != coord_len:
            n_skipped_mismatch += 1
            continue  # Skip mismatched entries

        np.save(dst_coords, coords)

        # Build JSON entry (without internal fields like coords_path)
        json_entry = {
            "id": new_id,
            "original_id": entry["id"],
            "sequence": entry["sequence"],
            "secondary_structure": entry["secondary_structure"],
            "pair_constraints": entry["pair_constraints"],
            "length": entry["length"],
            "source": entry["source"],
            "confidence": entry.get("confidence", 0.3),
            "mfe": entry.get("mfe"),
        }

        # Preserve source-specific metadata
        for key in ("structure_type", "has_real_ss", "pdb_id", "chain_id",
                     "closure_distance", "bond_rmsd", "chrom",
                     "reactivities_mean", "n_pairs",
                     "augmentation_index"):
            if key in entry and entry[key] is not None:
                json_entry[key] = entry[key]

        json_entries.append(json_entry)

        if (i + 1) % 2000 == 0:
            print(f"  Saved: {i+1}/{len(all_entries)}")

    if n_skipped_mismatch > 0:
        print(f"  Skipped {n_skipped_mismatch} entries with seq/coords length mismatch")

    # Save sequences.json
    with open(os.path.join(output_dir, "sequences.json"), "w") as f:
        json.dump(json_entries, f, indent=2)

    # Build metadata
    lengths = [e["length"] for e in all_entries]
    source_counts = Counter(e["source"] for e in all_entries)
    n_with_pairs = sum(1 for e in all_entries
                       if len(e.get("pair_constraints", [])) > 0)
    n_with_real_ss = sum(1 for e in all_entries
                         if e.get("secondary_structure", "")
                         and e["secondary_structure"] != "." * e["length"])

    # Length histogram
    length_bins = [(0, 100), (100, 200), (200, 300), (300, 500),
                   (500, 750), (750, 1000), (1000, 1500), (1500, 3000)]
    length_hist = {}
    for lo, hi in length_bins:
        count = sum(1 for l in lengths if lo <= l < hi)
        if count > 0:
            length_hist[f"{lo}-{hi}"] = count

    metadata = {
        "total": len(all_entries),
        "length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "n_with_pair_constraints": n_with_pairs,
        "n_with_real_secondary_structure": n_with_real_ss,
        "fraction_with_pairs": n_with_pairs / len(all_entries) if all_entries else 0.0,
        "fraction_with_real_ss": n_with_real_ss / len(all_entries) if all_entries else 0.0,
        "sources": dict(sorted(source_counts.items())),
        "length_histogram": length_hist,
        "input_dirs": {
            "isrnacirc": args.isrnacirc_dir,
            "shape": args.shape_dir,
            "pdb": args.pdb_dir,
            "medium": args.medium_dir,
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Merged Dataset: {output_dir}/")
    print(f"  Total samples: {metadata['total']}")
    print(f"  Length range: {metadata['length_range'][0]}-{metadata['length_range'][1]} nt")
    print(f"  Mean length: {metadata['mean_length']:.1f} nt")
    print(f"  With pair constraints: {n_with_pairs} ({metadata['fraction_with_pairs']:.1%})")
    print(f"  With real SS: {n_with_real_ss} ({metadata['fraction_with_real_ss']:.1%})")
    print(f"\n  Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    print(f"\n  Length distribution:")
    max_count = max(length_hist.values()) if length_hist else 1
    for label, count in length_hist.items():
        bar = "#" * min(50, int(count / max_count * 50))
        print(f"    {label:>12s} nt: {count:5d} {bar}")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Next: Train with merged data")
    print(f"  bash run_training_sequential.sh --labels {output_dir} --device cuda")


if __name__ == "__main__":
    main()
