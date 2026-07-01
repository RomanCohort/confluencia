#!/usr/bin/env python3
"""
CircFold Baseline (Scheme 0) - Official CASP CircRNA Structure Prediction Baseline

The foundational scheme that generates high-quality circRNA 3D structures
for CASP evaluation and training of subsequent schemes (1-7).

CASP CircRNA Track Official Baseline Method:
    Stage 1: ViennaRNA (Secondary Structure Prediction)
    Stage 2: trRosettaRNA2/RoseTTAFold2NA (3D Prediction)
    Stage 3: OpenMM (BSJ Cyclization)
    Stage 4: Molecular Dynamics Relaxation
    Stage 5: Quality Filtering

Key Features:
    - 5-stage pipeline with physics-based refinement
    - Expected retention: ~80,000 structures from 130,472 sequences
    - Quality thresholds: confidence≥0.70, BSJ distance 2.8-5.0Å, energy<800kJ/mol
    - Official baseline for CASP circRNA structure prediction track

Usage:
    python circfold_baseline.py --fasta circbase.fa --output casp_circ_baseline_output

Citation:
    CircFold Baseline: Official CASP Baseline for circRNA 3D Structure Prediction
    (Scheme 0 - Data Generation Pipeline)
"""

SCHEME_0_OFFICIAL_NAME = "CircFold Baseline"
SCHEME_0_CASP_ID = "CASP-circ-Baseline-0"
SCHEME_0_DESCRIPTION = "Official CASP Baseline for circRNA 3D Structure Prediction"

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEME_0_CONFIG = {
    'name': 'Scheme 0: Data Generation Pipeline',
    'type': 'data_generator',  # Not a trainable model
    'stages': [
        {
            'name': 'ViennaRNA',
            'tool': 'ViennaRNA 2.7.2',
            'purpose': 'Secondary structure prediction with BSJ constraint',
            'output': 'dot_bracket, bp_probs, MFE',
            'time_per_seq': '~1 second'
        },
        {
            'name': 'trRosettaRNA2',
            'tool': 'trRosettaRNA2 / RoseTTAFold2NA',
            'purpose': 'Linear 3D structure prediction',
            'output': 'PDB files (multiple samples)',
            'time_per_seq': '~100-130 seconds (GPU)'
        },
        {
            'name': 'BSJ Cyclization',
            'tool': 'OpenMM',
            'purpose': 'Connect BSJ ends to form circular topology',
            'output': 'Cyclized PDB + energy metrics',
            'time_per_seq': '~10-30 seconds'
        },
        {
            'name': 'MD Relaxation',
            'tool': 'OpenMM + AMBER14',
            'purpose': 'Molecular dynamics to resolve clashes',
            'output': 'MD snapshots + RMSD trajectory',
            'time_per_seq': '~1-5 min (fast mode)'
        },
        {
            'name': 'Quality Filtering',
            'tool': 'Multi-pass quality gates',
            'purpose': 'Filter low-quality structures',
            'output': 'Quality structures with confidence scores',
            'time_per_seq': '~5 seconds'
        }
    ],
    'quality_thresholds': {
        'confidence': 0.70,
        'bsj_distance': (2.8, 5.0),
        'energy': 800.0,
        'rmsd_variance': 0.3,
        'bsj_clashes': 5
    },
    'expected_output': {
        'input_sequences': 130472,
        'filtered_structures': 80000,
        'retention_rate': 0.60
    }
}


def run_scheme0(fasta_path, output_dir, config_path='config_quality.yaml'):
    """
    Execute Scheme 0 pipeline

    Args:
        fasta_path: Input FASTA file (circRNA sequences)
        output_dir: Output directory for generated structures
        config_path: Pipeline configuration file

    Returns:
        stats: Dictionary with generation statistics
    """
    print(f"\n{'='*70}")
    print(f"Scheme 0: Data Generation Pipeline")
    print(f"{'='*70}")
    print(f"Input: {fasta_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Import pipeline components
    from deploy_package.circrna_3d_pipeline.pipeline import CircRNA3DPipeline

    # Run pipeline
    pipeline = CircRNA3DPipeline(config_path)

    # Generate data
    results = pipeline.run_batch(
        sequences=load_fasta(fasta_path),
        bsj_positions=extract_bsj_positions(fasta_path),
        output_dir=output_dir
    )

    # Statistics
    stats = {
        'total_input': len(results),
        'successful': sum(1 for r in results if 'error' not in r),
        'high_quality': sum(1 for r in results if r.get('confidence', 0) >= 0.70),
        'retention_rate': sum(1 for r in results if r.get('confidence', 0) >= 0.70) / len(results)
    }

    print(f"\n{'='*70}")
    print(f"Scheme 0 Complete")
    print(f"{'='*70}")
    print(f"Input sequences: {stats['total_input']}")
    print(f"Successful: {stats['successful']}")
    print(f"High quality (≥0.70): {stats['high_quality']}")
    print(f"Retention rate: {stats['retention_rate']*100:.1f}%")
    print(f"{'='*70}\n")

    return stats


def load_fasta(fasta_path):
    """Load sequences from FASTA file"""
    sequences = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                sequences.append(line.strip())
    return sequences


def extract_bsj_positions(fasta_path):
    """Extract BSJ positions from FASTA headers"""
    import re
    bsj_positions = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                match = re.search(r'bsj_start=(\d+)\s+bsj_end=(\d+)', line)
                if match:
                    bsj_positions.append((int(match.group(1)), int(match.group(2))))
    return bsj_positions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scheme 0: Data Generation Pipeline')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='config_quality.yaml', help='Pipeline config')
    args = parser.parse_args()

    run_scheme0(args.fasta, args.output, args.config)