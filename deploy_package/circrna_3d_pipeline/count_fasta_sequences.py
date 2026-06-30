#!/usr/bin/env python3
"""
Count sequences and lengths in FASTA file.

Usage:
  python count_fasta_sequences.py input.fa
"""

import sys


def count_fasta_sequences(fasta_path):
    """Count sequences and their lengths."""

    print(f"Analyzing FASTA: {fasta_path}")

    sequences = []
    current_id = None
    current_seq = ""

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append((current_id, len(current_seq)))
                current_id = line[1:].split()[0]  # Get ID
                current_seq = ""
            else:
                current_seq += line

        # Don't forget the last sequence
        if current_id:
            sequences.append((current_id, len(current_seq)))

    print(f"\nTotal sequences: {len(sequences)}")

    if sequences:
        lengths = [length for _, length in sequences]

        print(f"\nLength statistics:")
        print(f"  Min length: {min(lengths)} nt")
        print(f"  Max length: {max(lengths)} nt")
        print(f"  Average length: {sum(lengths) / len(lengths):.1f} nt")
        print(f"  Total nucleotides: {sum(lengths)} nt")

        print(f"\nFirst 10 sequences:")
        for i, (seq_id, length) in enumerate(sequences[:10]):
            print(f"  {i+1}. {seq_id}: {length} nt")

        print(f"\nLast sequence:")
        print(f"  {len(sequences)}. {sequences[-1][0]}: {sequences[-1][1]} nt")

        # Length distribution
        print(f"\nLength distribution:")
        ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, float('inf'))]
        for min_len, max_len in ranges:
            count = sum(1 for length in lengths if min_len <= length < max_len)
            if max_len == float('inf'):
                print(f"  {min_len}+ nt: {count} sequences")
            else:
                print(f"  {min_len}-{max_len} nt: {count} sequences")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python count_fasta_sequences.py input.fa", file=sys.stderr)
        sys.exit(1)

    count_fasta_sequences(sys.argv[1])