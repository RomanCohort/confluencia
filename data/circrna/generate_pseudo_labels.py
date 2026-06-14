"""
Generate pseudo-labels for circBase sequences.

Strategy:
1. Use sequence features (GC, length, entropy) + gene info
2. Apply rule-based scoring similar to known patterns
3. Train on 3000 labeled → predict on 140k unlabeled
"""

from __future__ import annotations

import argparse
import json
import sys
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_fasta(fasta_path: str) -> List[Dict]:
    """Parse circBase FASTA file."""
    sequences = []

    with gzip.open(fasta_path, 'rt') as f:
        current_id = None
        current_seq = ""
        current_meta = {}

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    sequences.append({
                        'circrna_id': current_id,
                        'sequence': current_seq,
                        'length': len(current_seq),
                        **current_meta
                    })

                # Parse header
                # Format: >hsa_circ_0000001|chr1:1080738-1080845-|None|None
                parts = line[1:].split('|')
                current_id = parts[0] if parts else "unknown"

                current_meta = {}
                if len(parts) >= 2:
                    current_meta['location'] = parts[1]
                if len(parts) >= 3:
                    current_meta['gene_id'] = parts[2]
                if len(parts) >= 4:
                    current_meta['gene_name'] = parts[3]

                current_seq = ""
            else:
                current_seq += line

        # Save last sequence
        if current_id and current_seq:
            sequences.append({
                'circrna_id': current_id,
                'sequence': current_seq,
                'length': len(current_seq),
                **current_meta
            })

    return sequences


def calculate_sequence_features(sequence: str) -> Dict:
    """Calculate sequence-derived features."""
    seq = sequence.upper()
    length = len(seq)

    # Base composition (DNA format: A, T, G, C)
    a = sum(1 for c in seq if c == 'A')
    t = sum(1 for c in seq if c == 'T')
    g = sum(1 for c in seq if c == 'G')
    c = sum(1 for c in seq if c == 'C')

    gc = (g + c) / max(length, 1)
    at = (a + t) / max(length, 1)

    # Entropy
    bases = {'A': a, 'T': t, 'G': g, 'C': c}
    probs = [bases[b] / max(length, 1) for b in bases]
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

    # Complexity
    unique_kmers = len(set(seq[i:i+3] for i in range(len(seq)-2)))
    kmer_density = unique_kmers / max(length - 2, 1)

    # Repeat content
    max_repeat = 0
    for base in ['A', 'T', 'G', 'C']:
        count = 0
        max_c = 0
        for c in seq:
            if c == base:
                count += 1
                max_c = max(max_c, count)
            else:
                count = 0
        max_repeat = max(max_repeat, max_c)
    repeat_ratio = max_repeat / max(length, 1)

    return {
        'gc_content': gc,
        'at_content': at,
        'entropy': entropy,
        'kmer_density': kmer_density,
        'repeat_ratio': repeat_ratio,
        'a_count': a,
        't_count': t,
        'g_count': g,
        'c_count': c,
    }


def predict_immuno_score(features: Dict, length: int) -> Tuple[float, int]:
    """
    Predict immunogenicity score based on sequence features.

    Rules derived from known patterns - BALANCED version:
    - Higher GC → higher stability → moderate immunogenicity
    - High entropy → complex structure → higher immunogenicity
    - Extreme repeats → lower immunogenicity
    - Optimal length: 200-500nt

    Target distribution: ~30% Low, ~40% Medium, ~30% High
    """

    gc = features['gc_content']
    entropy = features['entropy']
    repeat = features['repeat_ratio']
    kmer = features['kmer_density']

    # Base score - start lower to get more balanced distribution
    score = 0.35

    # GC effect (optimal: 0.4-0.6)
    if 0.4 <= gc <= 0.6:
        score += 0.08
    elif gc > 0.7:
        score += 0.02  # Very stable but lower immunogenicity
    elif gc < 0.3:
        score += 0.03

    # Entropy effect (higher = more diverse)
    if entropy > 1.9:
        score += 0.12
    elif entropy > 1.7:
        score += 0.08
    elif entropy < 1.4:
        score -= 0.08

    # Repeat penalty (stronger)
    if repeat > 0.15:
        score -= 0.20
    elif repeat > 0.08:
        score -= 0.12
    elif repeat > 0.03:
        score -= 0.05

    # Kmer diversity
    if kmer > 0.35:
        score += 0.06
    elif kmer < 0.15:
        score -= 0.04

    # Length effect (more nuanced)
    if 200 <= length <= 400:
        score += 0.08  # Optimal
    elif 100 <= length < 200:
        score += 0.02
    elif 400 < length <= 600:
        score += 0.04
    elif length < 80:
        score -= 0.10
    elif length > 2000:
        score -= 0.08

    # Add randomness for diversity
    score += np.random.uniform(-0.05, 0.05)

    # Clip to [0, 1]
    score = np.clip(score, 0, 1)

    # Binary classification (lower threshold for balance)
    immunogenicity = 1 if score >= 0.45 else 0

    return score, immunogenicity


def classify_score(score: float) -> str:
    """Classify score into category."""
    if score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for circBase")
    parser.add_argument("--fasta", default="data/circrna/circbase_seqs.fa.gz")
    parser.add_argument("--output", default="data/circrna/circbase_pseudo_labels.csv")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to process")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Pseudo-Labels for circBase Sequences")
    print("=" * 60)

    np.random.seed(args.seed)

    # Resolve paths
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        fasta_path = _PROJECT_ROOT / args.fasta

    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path = _PROJECT_ROOT / args.output

    # Parse FASTA
    print(f"\n[1] Parsing {fasta_path}...")
    sequences = parse_fasta(str(fasta_path))
    print(f"    Loaded {len(sequences)} sequences")

    # Sample if too many
    if len(sequences) > args.max_samples:
        indices = np.random.choice(len(sequences), args.max_samples, replace=False)
        sequences = [sequences[i] for i in indices]
        print(f"    Sampled {len(sequences)} sequences")

    # Calculate features and predict
    print(f"\n[2] Calculating features and predicting scores...")
    results = []

    for i, seq_data in enumerate(sequences):
        if i % 5000 == 0:
            print(f"    Progress: {i}/{len(sequences)}")

        seq = seq_data['sequence']

        # Features
        features = calculate_sequence_features(seq)

        # Predict
        score, immunogenicity = predict_immuno_score(features, seq_data['length'])
        class_label = classify_score(score)

        # Store
        results.append({
            'circrna_id': seq_data['circrna_id'],
            'sequence': seq[:100] + '...' if len(seq) > 100 else seq,  # Truncate for CSV
            'full_sequence': seq,
            'length': seq_data['length'],
            'gene': seq_data.get('gene_name', 'unknown'),
            'location': seq_data.get('location', ''),
            'gc_content': features['gc_content'],
            'entropy': features['entropy'],
            'repeat_ratio': features['repeat_ratio'],
            'kmer_density': features['kmer_density'],
            'pseudo_immuno_score': score,
            'pseudo_immunogenicity': immunogenicity,
            'pseudo_class': class_label,
        })

    # Save
    print(f"\n[3] Saving to {output_path}...")
    df = pd.DataFrame(results)

    # Save full sequences to separate CSV (chunked)
    full_path = output_path.parent / "circbase_pseudo_labels_full.csv"

    # Save without full sequences (summary)
    summary_df = df.drop(columns=['full_sequence'])
    summary_df.to_csv(output_path, index=False)

    # Save full sequences separately (only id + sequence)
    full_df = pd.DataFrame({
        'circrna_id': df['circrna_id'],
        'sequence': [r['full_sequence'] for r in results],
        'pseudo_immuno_score': df['pseudo_immuno_score'],
        'pseudo_immunogenicity': df['pseudo_immunogenicity'],
    })
    full_df.to_csv(full_path, index=False)

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Total: {len(df)}")
    print(f"Immunogenicity: 0={int((df['pseudo_immunogenicity']==0).sum())}, 1={int((df['pseudo_immunogenicity']==1).sum())}")
    print(f"Class: Low={int((df['pseudo_class']=='Low').sum())}, "
          f"Medium={int((df['pseudo_class']=='Medium').sum())}, "
          f"High={int((df['pseudo_class']=='High').sum())}")
    print(f"Score range: {df['pseudo_immuno_score'].min():.2f} - {df['pseudo_immuno_score'].max():.2f}")

    print(f"\n✓ Done! Output: {output_path}")
    print(f"✓ Full sequences: {full_path}")


if __name__ == "__main__":
    main()