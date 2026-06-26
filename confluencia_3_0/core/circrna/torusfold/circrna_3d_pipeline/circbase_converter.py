"""
circBase to FASTA Converter.

Downloads and converts circBase circRNA sequences to FASTA format
for TorusFold 3D structure generation.
"""

import os
import sys
import csv
import argparse
import random
import requests
from pathlib import Path


def download_circbase(output_path='circbase_raw.csv', species='hsa'):
    """
    Download circBase database.

    Args:
        output_path: Path to save raw CSV
        species: Species code (hsa=human, mmu=mouse, etc.)

    Returns:
        Path to downloaded file
    """
    # circBase download URL
    url = f"http://circbase.org/download/{species}_all.csv"

    print(f"Downloading circBase ({species}) from {url}...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Downloaded {len(response.text.splitlines())} entries to {output_path}")
        return output_path

    except Exception as e:
        print(f"Download failed: {e}")
        print("Falling back to generating synthetic circRNA sequences...")
        return None


def parse_circbase_csv(csv_path):
    """
    Parse circBase CSV file.

    Expected columns:
        gene, chrom, start, end, strand, circRNA_id, seq

    Returns:
        list of dicts with circRNA info
    """
    circrnas = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # circBase fields
                seq = row.get('seq', '')
                if not seq or len(seq) < 30:
                    continue

                circ_id = row.get('circRNA_id', f"circ_{len(circrnas):05d}")
                gene = row.get('gene', 'unknown')
                chrom = row.get('chrom', 'unknown')
                start = int(row.get('start', 0))
                end = int(row.get('end', 0))

                circrnas.append({
                    'id': circ_id,
                    'sequence': seq,
                    'gene': gene,
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'length': len(seq),
                    'bsj_start': 0,  # circBase sequences are already circular
                    'bsj_end': len(seq),
                })

            except Exception as e:
                continue

    return circrnas


def filter_by_length(circrnas, min_length=50, max_length=500):
    """Filter circRNAs by length for TorusFold compatibility."""
    filtered = [c for c in circrnas if min_length <= c['length'] <= max_length]

    print(f"Length filter ({min_length}-{max_length}): {len(circrnas)} → {len(filtered)}")
    return filtered


def filter_by_gc_content(circrnas, min_gc=0.3, max_gc=0.7):
    """Filter circRNAs by GC content (stable RNA requires moderate GC)."""
    filtered = []
    for c in circrnas:
        seq = c['sequence']
        gc = sum(1 for b in seq if b in 'GC') / len(seq)
        if min_gc <= gc <= max_gc:
            c['gc_content'] = gc
            filtered.append(c)

    print(f"GC filter ({min_gc}-{max_gc}): {len(circrnas)} → {len(filtered)}")
    return filtered


def write_fasta(circrnas, output_path, max_samples=None):
    """Write circRNAs to FASTA format."""
    if max_samples and len(circrnas) > max_samples:
        random.seed(42)
        circrnas = random.sample(circrnas, max_samples)

    with open(output_path, 'w') as f:
        for c in circrnas:
            # FASTA header with BSJ annotation
            header = f">{c['id']} bsj_start={c['bsj_start']} bsj_end={c['bsj_end']} " \
                    f"gene={c['gene']} len={c['length']} gc={c.get('gc_content', 0.5):.2f}"
            f.write(header + '\n')
            f.write(c['sequence'] + '\n')

    print(f"Wrote {len(circrnas)} circRNAs to {output_path}")


def generate_synthetic_circrna(n_sequences=10000, min_length=50, max_length=500):
    """
    Generate synthetic circRNA sequences when circBase is unavailable.

    Uses biologically plausible GC content and base composition.
    """
    print(f"Generating {n_sequences} synthetic circRNA sequences...")

    random.seed(42)
    circrnas = []

    for i in range(n_sequences):
        L = random.randint(min_length, max_length)

        # circBase typical GC: 40-60%
        gc_target = random.uniform(0.4, 0.6)
        n_gc = int(L * gc_target)
        n_au = L - n_gc

        # Build sequence with biological composition
        seq = []
        for _ in range(n_gc // 2):
            seq.append(random.choice(['G', 'C']))
        for _ in range(n_au // 2):
            seq.append(random.choice(['A', 'U']))
        random.shuffle(seq)
        seq = ''.join(seq)

        gc_content = sum(1 for b in seq if b in 'GC') / len(seq)

        circrnas.append({
            'id': f"synthetic_{i:05d}",
            'sequence': seq,
            'gene': 'synthetic',
            'chromosome': 'synthetic',
            'start': 0,
            'end': L,
            'length': L,
            'bsj_start': 0,
            'bsj_end': L,
            'gc_content': gc_content,
        })

    return circrnas


def main():
    parser = argparse.ArgumentParser(description='circBase to FASTA Converter')
    parser.add_argument('--download', action='store_true', help='Download circBase')
    parser.add_argument('--input', default='circbase_raw.csv', help='Input CSV')
    parser.add_argument('--output', default='circrna_sequences.fasta', help='Output FASTA')
    parser.add_argument('--species', default='hsa', help='Species (hsa, mmu, etc.)')
    parser.add_argument('--min-length', type=int, default=50, help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=500, help='Maximum sequence length')
    parser.add_argument('--max-samples', type=int, default=20000, help='Maximum samples')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic if download fails')

    args = parser.parse_args()

    # Download or use existing
    if args.download:
        csv_path = download_circbase(args.input, args.species)
        if csv_path is None and args.synthetic:
            circrnas = generate_synthetic_circrna(args.max_samples, args.min_length, args.max_length)
        else:
            circrnas = parse_circbase_csv(csv_path)
    else:
        if os.path.exists(args.input):
            circrnas = parse_circbase_csv(args.input)
        elif args.synthetic:
            circrnas = generate_synthetic_circrna(args.max_samples, args.min_length, args.max_length)
        else:
            print(f"Error: {args.input} not found. Use --download or --synthetic.")
            sys.exit(1)

    # Filter
    circrnas = filter_by_length(circrnas, args.min_length, args.max_length)
    circrnas = filter_by_gc_content(circrnas, min_gc=0.3, max_gc=0.7)

    # Write FASTA
    write_fasta(circrnas, args.output, args.max_samples)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    print(f"  Total circRNAs: {len(circrnas)}")
    print(f"  Length range: {min(c['length'] for c in circrnas)}-{max(c['length'] for c in circrnas)}")
    print(f"  GC content range: {min(c['gc_content'] for c in circrnas):.2f}-{max(c['gc_content'] for c in circrnas):.2f}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()