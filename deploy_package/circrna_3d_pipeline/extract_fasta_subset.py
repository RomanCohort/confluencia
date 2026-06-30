#!/usr/bin/env python3
"""
Extract subset of sequences from FASTA file for testing.

Usage:
  python extract_fasta_subset.py input.fa output.fa num_sequences
"""

import sys


def extract_fasta_subset(input_fa, output_fa, num_seqs):
    """Extract first N sequences from FASTA."""

    print(f"Extracting {num_seqs} sequences from {input_fa} to {output_fa}")

    count = 0
    writing = False

    with open(input_fa, 'r') as f_in, open(output_fa, 'w') as f_out:
        for line in f_in:
            if line.startswith('>'):
                if count >= num_seqs:
                    break
                writing = True
                count += 1
            if writing:
                f_out.write(line)

    print(f"Extracted {count} sequences")
    print(f"Output saved to: {output_fa}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python extract_fasta_subset.py input.fa output.fa num_sequences")
        sys.exit(1)

    extract_fasta_subset(sys.argv[1], sys.argv[2], int(sys.argv[3]))