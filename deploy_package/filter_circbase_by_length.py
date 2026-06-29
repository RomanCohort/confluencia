#!/usr/bin/env python3
"""
Filter circBase sequences by length.

ViennaRNA has a limit on sequence length (unsigned int max ~65535).
Filter out sequences > max_length to avoid OverflowError in hc_add_bp.
"""

import gzip
import argparse


def filter_circbase(input_path, output_path, max_length=5000):
    """
    Filter circBase sequences by length.

    Args:
        input_path: Path to circBase FASTA (gzipped or plain)
        output_path: Path to filtered FASTA output
        max_length: Maximum sequence length to keep
    """
    # Read input (support gzipped or plain)
    if input_path.endswith('.gz'):
        opener = lambda: gzip.open(input_path, 'rt')
    else:
        opener = lambda: open(input_path, 'r')

    with opener() as f:
        lines = f.readlines()

    # Parse sequences
    sequences = []
    current_id = None
    current_seq = []
    total_count = 0

    for line in lines:
        if line.startswith('>'):
            if current_id and current_seq:
                total_count += 1
                seq = ''.join(current_seq)
                if len(seq) <= max_length:
                    sequences.append((current_id, seq))
            current_id = line.strip()
            current_seq = []
        else:
            current_seq.append(line.strip())

    # Handle last sequence
    if current_id and current_seq:
        total_count += 1
        seq = ''.join(current_seq)
        if len(seq) <= max_length:
            sequences.append((current_id, seq))

    # Write filtered sequences
    with open(output_path, 'w') as f:
        for i, (seq_id, seq) in enumerate(sequences):
            f.write(f">circ_{i:06d} bsj_start=0 bsj_end={len(seq)}\n{seq}\n")

    print(f"Total sequences: {total_count}")
    print(f"Kept (≤{max_length}nt): {len(sequences)} ({100*len(sequences)/total_count:.1f}%)")
    print(f"Filtered out: {total_count - len(sequences)}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter circBase sequences by length')
    parser.add_argument('--input', required=True, help='Input FASTA (gzipped or plain)')
    parser.add_argument('--output', required=True, help='Output filtered FASTA')
    parser.add_argument('--max-length', type=int, default=5000, help='Max sequence length (default: 5000)')

    args = parser.parse_args()
    filter_circbase(args.input, args.output, args.max_length)
