#!/usr/bin/env python3
"""Split FASTA into N parts for parallel processing"""

import sys
import os

def split_fasta(fasta_path, num_parts, output_dir):
    """Split FASTA file into multiple parts"""

    # 读取所有序列
    sequences = []
    with open(fasta_path) as f:
        current_header = None
        current_seq = ""

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_header, current_seq))
                current_header = line
                current_seq = ""
            else:
                current_seq += line

        if current_header:
            sequences.append((current_header, current_seq))

    total = len(sequences)
    part_size = (total + num_parts - 1) // num_parts

    print(f"Total sequences: {total}")
    print(f"Splitting into {num_parts} parts (~{part_size} sequences each)")

    os.makedirs(output_dir, exist_ok=True)

    # 分割并保存
    for i in range(num_parts):
        start = i * part_size
        end = min((i + 1) * part_size, total)

        part_file = os.path.join(output_dir, f'part_{i:02d}.fa')
        with open(part_file, 'w') as f:
            for header, seq in sequences[start:end]:
                f.write(f"{header}\n{seq}\n")

        print(f"  Part {i:02d}: {end - start} sequences → {part_file}")

    print(f"\n✓ Split complete!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='circbase_filtered_5000.fa')
    parser.add_argument('--num-parts', type=int, default=10)
    parser.add_argument('--output', default='fasta_parts')
    args = parser.parse_args()

    split_fasta(args.fasta, args.num_parts, args.output)