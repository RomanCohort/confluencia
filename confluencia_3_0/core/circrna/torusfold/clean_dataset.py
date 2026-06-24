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
    """Load raw dataset (sequences.json + coords/*.npy)."""
    seq_path = Path(data_dir) / "sequences.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"sequences.json not found in {data_dir}")

    with open(seq_path, 'r') as f:
        data = json.load(f)

    sequences = data['sequences']
    coords_dir = Path(data_dir) / "coords"

    # Load coordinates
    coords = []
    for i, seq in enumerate(sequences):
        coord_path = coords_dir / f"sample_{i}.npy"
        if coord_path.exists():
            coords.append(np.load(coord_path))
        else:
            coords.append(None)

    # Load optional fields
    pair_probs = data.get('pair_probs', [None] * len(sequences))
    confidence = data.get('confidence', [0.5] * len(sequences))
    metadata = data.get('metadata', [{}] * len(sequences))

    return sequences, coords, pair_probs, confidence, metadata


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
    if L > args.max_length:
        return False, f"too_long ({L} > {args.max_length})"

    # Confidence check (optional)
    if args.min_confidence > 0 and conf < args.min_confidence:
        return False, f"low_confidence ({conf:.3f} < {args.min_confidence})"

    return True, "valid"


def clean_dataset(input_dir, output_dir, args):
    """Filter dataset and save clean version."""
    print(f"Loading dataset from: {input_dir}")
    sequences, coords, pair_probs, confidence, metadata = load_raw_dataset(input_dir)

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

    clean_sequences = []
    clean_coords = []
    clean_pair_probs = []
    clean_confidence = []
    clean_metadata = []

    for i in tqdm(valid_indices):
        clean_sequences.append(sequences[i])
        clean_coords.append(coords[i])
        if pair_probs and i < len(pair_probs):
            clean_pair_probs.append(pair_probs[i])
        else:
            clean_pair_probs.append(None)
        clean_confidence.append(confidence[i] if i < len(confidence) else 0.5)
        clean_metadata.append(metadata[i] if i < len(metadata) else {})

        # Save coordinate file
        np.save(coords_out_dir / f"sample_{len(clean_sequences)-1}.npy", coords[i])

    # Save sequences.json
    output_data = {
        'sequences': clean_sequences,
        'pair_probs': clean_pair_probs,
        'confidence': clean_confidence,
        'metadata': clean_metadata,
        'source': str(input_dir),
        'cleaned_samples': n_valid,
        'original_samples': n_total,
        'rejection_reasons': rejection_reasons,
    }

    with open(output_path / "sequences.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone! Clean dataset saved.")
    print(f"  Sequences: {len(clean_sequences)}")
    print(f"  Length range: {min(len(s) for s in clean_sequences)} - {max(len(s) for s in clean_sequences)}")
    if clean_coords[0] is not None:
        print(f"  Coord range: {min(c.min() for c in clean_coords):.2f} - {max(c.max() for c in clean_coords):.2f} Å")


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
    parser.add_argument('--max-length', type=int, default=1000,
                        help='Maximum sequence length (default 1000)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence score (default 0.0, no filter)')
    args = parser.parse_args()

    clean_dataset(args.input, args.output, args)


if __name__ == '__main__':
    main()