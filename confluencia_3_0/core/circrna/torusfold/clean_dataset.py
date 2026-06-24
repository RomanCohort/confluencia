#!/usr/bin/env python3
"""
clean_dataset.py — Filter and clean TorusFold training data.

Removes samples with:
- NaN/Inf coordinates
- Extreme values (>1e6 Å, unrealistic for RNA)
- Zero coordinates (placeholder data)
- Length mismatch (coords shape != sequence length)

Outputs a cleaned dataset to a new directory.
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm


def load_raw_dataset(data_dir):
    """Load raw dataset (sequences.json + coords/*.npy).

    Format follows train_all_schemes.py load_pseudo_labels:
        sequences.json: [{"id": ..., "sequence": ..., "secondary_structure": ..., ...}, ...]
        coords/: {id}.npy files
        metadata.json: optional summary
    """
    seq_path = Path(data_dir) / "sequences.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"sequences.json not found in {data_dir}")

    with open(seq_path, 'r') as f:
        seq_data = json.load(f)

    # seq_data is a list of dicts
    coords_dir = Path(data_dir) / "coords"

    sequences = []
    seq_ids = []
    coords = []
    pair_probs = []
    confidence = []
    metadata = []

    for i, item in enumerate(seq_data):
        seq_id = item.get('id', f'pseudo_{i:05d}')
        seq = item.get('sequence', '')
        ss = item.get('secondary_structure', None)
        pair = item.get('pair_constraints', None)
        source = item.get('source', 'unknown')
        item_conf = item.get('confidence', 0.5)

        coord_path = coords_dir / f"{seq_id}.npy"
        if not coord_path.exists():
            # Try alternate naming
            coord_path = coords_dir / f"pseudo_{i:05d}.npy"
            if not coord_path.exists():
                continue

        coord = np.load(coord_path) if coord_path.exists() else None

        sequences.append(seq)
        seq_ids.append(seq_id)
        coords.append(coord)
        pair_probs.append(pair)
        # Use confidence from item, fallback to metadata.json
        if item_conf is not None and item_conf > 0:
            confidence.append(item_conf)
        else:
            meta_path = Path(data_dir) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    if isinstance(meta, list) and i < len(meta):
                        confidence.append(meta[i].get('confidence', 0.5))
                    else:
                        confidence.append(0.5)
            else:
                confidence.append(0.5)
        metadata.append({
            'id': seq_id,
            'length': len(seq),
            'source': source,
            'confidence': confidence[-1],
        })

    return sequences, seq_ids, coords, pair_probs, confidence, metadata


def validate_sample(seq, coord, pair_prob, conf, meta, args):
    """Check if a sample is valid.

    Returns: (is_valid, reason)
    """
    if coord is None:
        return False, "missing_coords"

    L = len(seq)
    if coord.shape[0] != L:
        return False, f"length_mismatch ({coord.shape[0]} vs {L})"

    # NaN/Inf check
    if np.isnan(coord).any():
        return False, "nan_coords"
    if np.isinf(coord).any():
        return False, "inf_coords"

    # Zero check (placeholder data)
    if np.abs(coord).sum() < 1e-3:
        return False, "zero_coords"

    # Extreme value check
    abs_max = np.abs(coord).max()
    if abs_max > args.max_coord:
        return False, f"extreme_value ({abs_max:.2e} > {args.max_coord})"

    # Length range check
    if L < args.min_length:
        return False, f"too_short ({L} < {args.min_length})"
    if args.max_length > 0 and L > args.max_length:
        return False, f"too_long ({L} > {args.max_length})"

    # Confidence check (optional)
    if args.min_confidence > 0 and conf < args.min_confidence:
        return False, f"low_confidence ({conf:.3f} < {args.min_confidence})"

    return True, "valid"


def clean_dataset(input_dir, output_dir, args):
    """Filter dataset and save clean version."""
    print(f"Loading dataset from: {input_dir}")
    sequences, seq_ids, coords, pair_probs, confidence, metadata = load_raw_dataset(input_dir)

    n_total = len(sequences)
    print(f"Total samples: {n_total}")

    # Validate all samples
    valid_indices = []
    rejection_reasons = {}

    print("Validating samples...")
    for i in tqdm(range(n_total)):
        seq = sequences[i]
        coord = coords[i]
        pp = pair_probs[i] if i < len(pair_probs) else None
        conf = confidence[i] if i < len(confidence) else 0.5
        meta = metadata[i] if i < len(metadata) else {}

        is_valid, reason = validate_sample(seq, coord, pp, conf, meta, args)

        if is_valid:
            valid_indices.append(i)
        else:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    n_valid = len(valid_indices)
    print(f"\nValid samples: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"Rejected samples: {n_total - n_valid}")

    if rejection_reasons:
        print("\nRejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if n_valid == 0:
        print("ERROR: No valid samples found!")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    coords_out_dir = output_path / "coords"
    coords_out_dir.mkdir(exist_ok=True)

    # Save clean data
    print(f"\nSaving clean dataset to: {output_dir}")

    clean_seq_data = []

    for idx, i in enumerate(tqdm(valid_indices)):
        item = {
            'id': seq_ids[i],
            'sequence': sequences[i],
            'source': metadata[i].get('source', 'unknown'),
            'confidence': confidence[i],
        }
        if pair_probs[i] is not None:
            item['pair_constraints'] = pair_probs[i]
        if metadata[i]:
            item['metadata'] = metadata[i]

        clean_seq_data.append(item)

        # Save coordinate file with same id
        np.save(coords_out_dir / f"{seq_ids[i]}.npy", coords[i])

    # Save sequences.json (list of dicts, same format as input)
    with open(output_path / "sequences.json", 'w') as f:
        json.dump(clean_seq_data, f, indent=2)

    # Save cleaning report
    report = {
        'source': str(input_dir),
        'original_samples': n_total,
        'cleaned_samples': n_valid,
        'rejection_reasons': rejection_reasons,
        'params': {
            'max_coord': args.max_coord,
            'min_length': args.min_length,
            'max_length': args.max_length,
            'min_confidence': args.min_confidence,
        }
    }
    with open(output_path / "clean_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDone! Clean dataset saved.")
    print(f"  Sequences: {len(clean_seq_data)}")
    if clean_seq_data:
        lens = [len(item['sequence']) for item in clean_seq_data]
        print(f"  Length range: {min(lens)} - {max(lens)}")
    if coords and coords[valid_indices[0]] is not None:
        valid_coords = [coords[i] for i in valid_indices[:100]]  # Sample for stats
        print(f"  Coord range: {min(c.min() for c in valid_coords):.2f} - {max(c.max() for c in valid_coords):.2f} Å")


def main():
    parser = argparse.ArgumentParser(description='Clean TorusFold dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output (clean) dataset directory')
    parser.add_argument('--max-coord', type=float, default=1e6,
                        help='Maximum coordinate value in Å (default 1e6)')
    parser.add_argument('--min-length', type=int, default=10,
                        help='Minimum sequence length (default 10)')
    parser.add_argument('--max-length', type=int, default=0,
                        help='Maximum sequence length (default 0 = no limit)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence score (default 0.0, no filter)')
    args = parser.parse_args()

    clean_dataset(args.input, args.output, args)


if __name__ == '__main__':
    main()