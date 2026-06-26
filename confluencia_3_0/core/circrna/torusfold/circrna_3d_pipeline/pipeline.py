"""
Full Pipeline Orchestration.

Integrates all 5 stages:
  Stage 1: ViennaRNA secondary structure prediction
  Stage 2: RoseTTAFold2NA 3D prediction (linear)
  Stage 3: OpenMM BSJ cyclization
  Stage 4: OpenMM MD relaxation
  Stage 5: Quality filtering & confidence scoring
"""

import os
import sys
import json
import time
import numpy as np
import yaml
from pathlib import Path

from stage1_vienna import ViennaRNAPredictor
from stage2_rosetta import RoseTTAFold2NAPredictor
from stage3_cyclize import BSJCyclizer
from stage4_md import MDRelaxation
from stage5_quality import QualityFilter, save_dataset, convert_to_torusfold_format


class CircRNA3DPipeline:
    """Full pipeline for circRNA 3D structure generation."""

    def __init__(self, config_path=None, mode='fast'):
        """
        Args:
            config_path: Path to config.yaml
            mode: 'prefilter' (quick check), 'fast' (2ns), 'high_quality' (10ns)
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mode = mode

        # Initialize stages
        self.stage1 = ViennaRNAPredictor(self.config['vienna'])
        self.stage2 = RoseTTAFold2NAPredictor(self.config['rosetta'])
        self.stage3 = BSJCyclizer(self.config['cyclize'])
        self.stage4 = MDRelaxation(self.config['md'], mode=mode)
        self.stage5 = QualityFilter(self.config['quality'])

        self.output_dir = self.config['output'].get('output_dir', 'pipeline_output/')
        self.save_pdbs = self.config['output'].get('save_pdbs', True)
        self.save_energies = self.config['output'].get('save_energies', True)

    def run_single(self, sequence, bsj_start, bsj_end, seq_id=0):
        """
        Run full pipeline for a single circRNA.

        Args:
            sequence: RNA sequence string
            bsj_start: BSJ start index (0-based)
            bsj_end: BSJ end index (0-based)
            seq_id: Sequence identifier

        Returns:
            dict with 'quality_structures', 'confidence', 'stats'
        """
        start_time = time.time()

        seq_dir = os.path.join(self.output_dir, f'seq_{seq_id}')
        os.makedirs(seq_dir, exist_ok=True)

        # Stage 1: Secondary structure
        print(f"[Stage 1] Predicting secondary structure for seq_{seq_id}...")
        ss_result = self.stage1.predict(sequence, bsj_start, bsj_end)
        ss_result['seq_id'] = seq_id

        # Stage 2: 3D prediction (linear)
        print(f"[Stage 2] Predicting 3D structure for seq_{seq_id}...")
        linear_dir = os.path.join(seq_dir, 'linear')
        linear_results = self.stage2.predict(
            sequence=sequence,
            dot_bracket=ss_result['dot_bracket'],
            bp_probs=ss_result['bp_probs'],
            output_dir=linear_dir
        )

        # Stage 3: Cyclization
        print(f"[Stage 3] Cyclizing BSJ for seq_{seq_id}...")
        cyclize_dir = os.path.join(seq_dir, 'cyclized')
        os.makedirs(cyclize_dir, exist_ok=True)

        cyclized_results = []
        for sample in linear_results:
            cycl = self.stage3.cyclize(
                pdb_path=sample['pdb_path'],
                bsj_start=bsj_start,
                bsj_end=bsj_end,
                ss_pairs=self._parse_ss_pairs(ss_result['dot_bracket']),
                output_path=os.path.join(cyclize_dir, f'cyclized_{sample["sample_id"]}.pdb')
            )
            cycl['seq_id'] = seq_id
            cycl['sample_id'] = sample['sample_id']
            cyclized_results.append(cycl)

        # Stage 4: MD relaxation
        print(f"[Stage 4] Running MD relaxation for seq_{seq_id}...")
        md_dir = os.path.join(seq_dir, 'md')
        os.makedirs(md_dir, exist_ok=True)

        md_results = []
        for cycl in cyclized_results:
            md = self.stage4.relax(
                pdb_path=cycl['pdb_path'],
                bsj_start=bsj_start,
                bsj_end=bsj_end,
                output_dir=os.path.join(md_dir, f'sample_{cycl["sample_id"]}')
            )
            md['seq_id'] = seq_id
            md['sample_id'] = cycl['sample_id']
            md_results.append(md)

        # Stage 5: Quality filtering
        print(f"[Stage 5] Filtering and scoring for seq_{seq_id}...")
        quality_structures = self.stage5.filter_and_score_batch(
            md_results, cyclized_results, ss_result
        )

        elapsed = time.time() - start_time
        print(f"Pipeline completed for seq_{seq_id} in {elapsed:.1f}s")
        print(f"  Generated {len(quality_structures)} quality structures")

        # Compute overall confidence
        avg_confidence = np.mean([s['confidence'] for s in quality_structures]) if quality_structures else 0.0

        return {
            'seq_id': seq_id,
            'sequence': sequence,
            'bsj_start': bsj_start,
            'bsj_end': bsj_end,
            'ss_result': ss_result,
            'quality_structures': quality_structures,
            'avg_confidence': float(avg_confidence),
            'num_structures': len(quality_structures),
            'elapsed_seconds': elapsed
        }

    def run_batch(self, sequences, bsj_positions):
        """
        Run pipeline for multiple circRNAs.

        Args:
            sequences: list of RNA sequence strings
            bsj_positions: list of (bsj_start, bsj_end) tuples

        Returns:
            list of result dicts
        """
        results = []
        for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
            result = self.run_single(seq, bsj_start, bsj_end, seq_id=i)
            results.append(result)
        return results

    def export_dataset(self, results, output_path=None):
        """
        Export all quality structures to a unified dataset.

        Args:
            results: list of results from run_batch
            output_path: path to save dataset JSON

        Returns:
            dataset summary dict
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'dataset.json')

        all_structures = []
        for result in results:
            for struct in result['quality_structures']:
                struct['sequence'] = result['sequence']
                struct['bsj_start'] = result['bsj_start']
                struct['bsj_end'] = result['bsj_end']
                all_structures.append(struct)

        # Save dataset
        save_dataset(all_structures, output_path)

        # Generate report
        report = self.stage5.generate_dataset_report(all_structures)

        # Save report
        report_path = output_path.replace('.json', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDataset Summary:")
        print(f"  Total structures: {report['total_structures']}")
        print(f"  Confidence: {report['confidence_stats']['mean']:.3f} ± {report['confidence_stats']['std']:.3f}")
        print(f"  Energy: {report['energy_stats']['mean']:.1f} ± {report['energy_stats']['std']:.1f} kJ/mol")
        print(f"  BSJ distance: {report['bsj_distance_stats']['mean']:.2f} ± {report['bsj_distance_stats']['std']:.2f} Å")

        return report

    def export_torusfold(self, results, output_dir=None):
        """
        Export results in TorusFold training format.

        Args:
            results: list of results from run_batch
            output_dir: directory for TorusFold format files
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'torusfold_format')

        os.makedirs(output_dir, exist_ok=True)

        dataset = []
        for result in results:
            for struct in result['quality_structures']:
                tf_data = convert_to_torusfold_format(struct)
                tf_data['sequence'] = result['sequence']
                tf_data['bsj_start'] = result['bsj_start']
                tf_data['bsj_end'] = result['bsj_end']
                dataset.append(tf_data)

        # Save as numpy arrays
        coords = np.stack([d['coords'] for d in dataset])
        confidences = np.array([d['confidence'] for d in dataset])
        sequences = [d['sequence'] for d in dataset]

        np.save(os.path.join(output_dir, 'coords.npy'), coords)
        np.save(os.path.join(output_dir, 'confidences.npy'), confidences)

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'num_structures': len(dataset),
                'sequences': sequences,
                'avg_confidence': float(np.mean(confidences)),
                'avg_coords_shape': list(coords.shape)
            }, f, indent=2)

        print(f"Exported {len(dataset)} structures to TorusFold format at {output_dir}")

    def _parse_ss_pairs(self, dot_bracket):
        """Parse dot-bracket notation into base pair list."""
        stack = []
        pairs = []
        for i, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        return pairs


def main():
    """Example usage of the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='circRNA 3D Structure Generation Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--fasta', help='Input FASTA file with circRNA sequences')
    parser.add_argument('--sequence', help='Single RNA sequence')
    parser.add_argument('--bsj-start', type=int, default=0, help='BSJ start index')
    parser.add_argument('--bsj-end', type=int, default=-1, help='BSJ end index')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--export-torusfold', action='store_true', help='Export in TorusFold format')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = CircRNA3DPipeline(args.config)

    if args.output:
        pipeline.output_dir = args.output

    if args.sequence:
        # Single sequence mode
        bsj_end = args.bsj_end if args.bsj_end > 0 else len(args.sequence)
        result = pipeline.run_single(args.sequence, args.bsj_start, bsj_end)

        if args.export_torusfold:
            pipeline.export_torusfold([result])

    elif args.fasta:
        # Batch mode from FASTA
        sequences, bsj_positions = load_fasta_with_bsj(args.fasta)
        results = pipeline.run_batch(sequences, bsj_positions)
        pipeline.export_dataset(results)

        if args.export_torusfold:
            pipeline.export_torusfold(results)

    else:
        print("Usage: python pipeline.py --sequence <RNA_SEQ> [--bsj-start 0 --bsj-end 100]")
        print("   or: python pipeline.py --fasta <input.fasta>")


def load_fasta_with_bsj(fasta_path):
    """
    Load FASTA file with BSJ annotations.

    Expected FASTA header format:
    >circRNA_001 bsj_start=0 bsj_end=100
    ACGUACGU...

    Or simple format (assume BSJ at sequence ends):
    >circRNA_001
    ACGUACGU...
    """
    sequences = []
    bsj_positions = []

    with open(fasta_path, 'r') as f:
        current_seq = ''
        current_bsj = (0, -1)

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    bsj_positions.append(current_bsj)

                # Parse header for BSJ annotation
                header = line[1:]
                current_seq = ''
                current_bsj = (0, -1)

                if 'bsj_start=' in header:
                    import re
                    start = int(re.search(r'bsj_start=(\d+)', header).group(1))
                    end = int(re.search(r'bsj_end=(\d+)', header).group(1))
                    current_bsj = (start, end)
            else:
                current_seq += line

        # Last sequence
        if current_seq:
            sequences.append(current_seq)
            if current_bsj[1] == -1:
                current_bsj = (0, len(current_seq))
            bsj_positions.append(current_bsj)

    return sequences, bsj_positions


if __name__ == '__main__':
    main()
