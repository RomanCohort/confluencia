"""
Batch Prefilter: Quickly screen circRNA sequences before full 3D generation.

Uses ViennaRNA + quick minimization to eliminate unstable sequences,
reducing computation time for full pipeline.
"""

import os
import sys
import numpy as np
import json
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from stage1_vienna import ViennaRNAPredictor


class BatchPrefilter:
    """Quick prefilter to eliminate unstable circRNA sequences."""

    def __init__(self, config):
        self.vienna = ViennaRNAPredictor(config.get('vienna', {}))
        self.max_mfe = config.get('prefilter', {}).get('max_mfe_kjmol', -50.0)  # Stable = negative MFE
        self.min_gc_content = config.get('prefilter', {}).get('min_gc_content', 0.3)
        self.max_gc_content = config.get('prefilter', {}).get('max_gc_content', 0.7)
        self.min_length = config.get('prefilter', {}).get('min_length', 30)
        self.max_length = config.get('prefilter', {}).get('max_length', 500)

    def prefilter(self, sequences: List[str], bsj_positions: List[Tuple[int, int]]) -> dict:
        """
        Batch prefilter sequences by stability criteria.

        Args:
            sequences: List of RNA sequences
            bsj_positions: List of (bsj_start, bsj_end) tuples

        Returns:
            dict with 'kept_indices', 'rejected_indices', 'reasons', 'stats'
        """
        kept_indices = []
        rejected_indices = []
        reasons = {}

        stats = {
            'total': len(sequences),
            'too_short': 0,
            'too_long': 0,
            'bad_gc': 0,
            'unstable': 0,
            'passed': 0
        }

        for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
            # Length check
            L = len(seq)
            if L < self.min_length:
                rejected_indices.append(i)
                reasons[i] = f"Too short: {L} < {self.min_length}"
                stats['too_short'] += 1
                continue
            if L > self.max_length:
                rejected_indices.append(i)
                reasons[i] = f"Too long: {L} > {self.max_length}"
                stats['too_long'] += 1
                continue

            # GC content check (stable RNA needs moderate GC)
            gc_count = sum(1 for b in seq if b in 'GC')
            gc_content = gc_count / L
            if gc_content < self.min_gc_content or gc_content > self.max_gc_content:
                rejected_indices.append(i)
                reasons[i] = f"Bad GC: {gc_content:.2f} (want {self.min_gc_content}-{self.max_gc_content})"
                stats['bad_gc'] += 1
                continue

            # ViennaRNA stability check
            try:
                ss_result = self.vienna.predict(seq, bsj_start, bsj_end)
                mfe = ss_result['mfe']

                # Stable RNA should have negative MFE
                if mfe > self.max_mfe:
                    rejected_indices.append(i)
                    reasons[i] = f"Unstable: MFE={mfe:.1f} > {self.max_mfe}"
                    stats['unstable'] += 1
                    continue

                # Passed all checks
                kept_indices.append(i)
                stats['passed'] += 1

            except Exception as e:
                rejected_indices.append(i)
                reasons[i] = f"ViennaRNA error: {str(e)}"
                stats['unstable'] += 1

        return {
            'kept_indices': kept_indices,
            'rejected_indices': rejected_indices,
            'reasons': reasons,
            'stats': stats
        }

    def print_report(self, result):
        """Print prefilter report."""
        stats = result['stats']
        print(f"\n{'='*50}")
        print(f"  Prefilter Report")
        print(f"{'='*50}")
        print(f"  Total sequences: {stats['total']}")
        print(f"  Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)")
        print(f"  Rejected:")
        print(f"    - Too short: {stats['too_short']}")
        print(f"    - Too long: {stats['too_long']}")
        print(f"    - Bad GC: {stats['bad_gc']}")
        print(f"    - Unstable: {stats['unstable']}")
        print(f"{'='*50}")

        if stats['passed'] > 0:
            print(f"  Retention rate: {100*stats['passed']/stats['total']:.1f}%")
            print(f"  Recommended for generation: {stats['passed']} sequences")


def generate_sequences_from_circbase(n_sequences=20000, output_path='circrna_sequences.fasta'):
    """
    Generate FASTA file from circBase or random circRNA sequences.

    Args:
        n_sequences: Number of sequences to generate
        output_path: Path to save FASTA
    """
    import random

    bases = ['A', 'U', 'G', 'C']

    # circBase typical lengths: 50-500 nt
    # circBase typical GC: 0.4-0.6

    with open(output_path, 'w') as f:
        for i in range(n_sequences):
            # Random length with circBase distribution
            L = random.randint(50, min(500, 300 + random.randint(-100, 100)))

            # GC content in stable range
            gc_target = random.uniform(0.4, 0.6)
            n_gc = int(L * gc_target)
            n_au = L - n_gc

            # Build sequence
            seq = ['G'] * (n_gc // 2) + ['C'] * (n_gc // 2) + \
                  ['A'] * (n_au // 2) + ['U'] * (n_au // 2)
            random.shuffle(seq)
            seq = ''.join(seq)

            # BSJ at sequence ends (circRNA typical)
            bsj_start = 0
            bsj_end = L

            f.write(f">circRNA_{i:05d} bsj_start={bsj_start} bsj_end={bsj_end}\n")
            f.write(f"{seq}\n")

    print(f"Generated {n_sequences} sequences to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch Prefilter for circRNA 3D Pipeline')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--output', default='prefiltered.fasta', help='Output filtered FASTA')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--report', default='prefilter_report.json', help='Report JSON')

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load sequences
    from pipeline import load_fasta_with_bsj
    sequences, bsj_positions = load_fasta_with_bsj(args.fasta)

    # Prefilter
    prefilter = BatchPrefilter(config)
    result = prefilter.prefilter(sequences, bsj_positions)
    prefilter.print_report(result)

    # Save filtered sequences
    kept = result['kept_indices']
    with open(args.output, 'w') as f:
        for i in kept:
            seq = sequences[i]
            bsj_start, bsj_end = bsj_positions[i]
            f.write(f">circRNA_{i:05d} bsj_start={bsj_start} bsj_end={bsj_end}\n")
            f.write(f"{seq}\n")

    print(f"Saved {len(kept)} sequences to {args.output}")

    # Save report
    with open(args.report, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()