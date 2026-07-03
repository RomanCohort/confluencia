"""
Parallel worker for trRosettaRNA2 circRNA pipeline.

Target: 6000 sequences/month on DGX Spark (8× GPU)

Architecture:
  - Ray-based parallel execution
  - Each worker runs on one GPU
  - Sequences distributed evenly across workers
  - Progress tracking and checkpointing
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from stage1_vienna import ViennaRNAPredictor
from stage2_trrosetta import trRosettaRNA2Predictor
from stage3_cyclize import BSJCyclizer
from stage4_md import MDRelaxation
from stage5_quality import QualityFilter, save_dataset, convert_to_torusfold_format


class HighThroughputPipeline:
    """
    High-throughput circRNA 3D structure generation pipeline.

    Optimized for trRosettaRNA2 + fast cyclization.
    Target: 6000 sequences/month.
    """

    def __init__(self, config_path: str, mode: str = 'fast', gpu_id: int = 0):
        import yaml

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mode = mode
        self.gpu_id = gpu_id
        device = f'cuda:{gpu_id}'

        # Override device in config
        if 'trrosetta' in self.config:
            self.config['trrosetta']['device'] = device

        # Initialize stages
        self.stage1 = ViennaRNAPredictor(self.config['vienna'])
        self.stage2 = trRosettaRNA2Predictor(self.config.get('trrosetta', self.config.get('rosetta', {})))
        self.stage3 = BSJCyclizer(self.config['cyclize'])
        self.stage4 = MDRelaxation(self.config['md'], mode=mode)
        self.stage5 = QualityFilter(self.config['quality'])

        self.output_dir = self.config['output'].get('output_dir', 'pipeline_output/')

    def run_single(
        self,
        sequence: str,
        bsj_start: int,
        bsj_end: int,
        seq_id: int = 0,
        source: str = 'real'
    ) -> Dict:
        """
        Run full pipeline for a single circRNA sequence.

        Returns:
            dict with 'quality_structures', 'rejected_structures', 'confidence', 'stats', 'status', 'source'
        """
        start_time = time.time()

        seq_dir = os.path.join(self.output_dir, f'seq_{seq_id}')
        os.makedirs(seq_dir, exist_ok=True)

        try:
            # Stage 1: Secondary structure
            ss_result = self.stage1.predict(sequence, bsj_start, bsj_end)
            ss_result['seq_id'] = seq_id

            # Stage 2: trRosettaRNA2 3D prediction
            linear_dir = os.path.join(seq_dir, 'linear')
            linear_results = self.stage2.predict(
                sequence=sequence,
                dot_bracket=ss_result.get('dot_bracket'),
                bp_probs=ss_result.get('bp_probs'),
                output_dir=linear_dir
            )

            # Stage 3: Cyclization
            cyclize_dir = os.path.join(seq_dir, 'cyclized')
            os.makedirs(cyclize_dir, exist_ok=True)

            cyclized_results = []
            for sample in linear_results:
                cycl = self.stage3.cyclize(
                    pdb_path=sample['pdb_path'],
                    bsj_start=bsj_start,
                    bsj_end=bsj_end,
                    ss_pairs=self._parse_ss_pairs(ss_result.get('dot_bracket', '')),
                    output_path=os.path.join(cyclize_dir, f'cyclized_{sample["sample_id"]}.pdb')
                )
                cycl['seq_id'] = seq_id
                cycl['sample_id'] = sample['sample_id']
                cyclized_results.append(cycl)

            # Stage 4: MD relaxation
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

            # Stage 5: Quality filtering (now returns structured rejections)
            quality_structures, rejected_by_gate = self.stage5.filter_and_score_batch(
                md_results, cyclized_results, [ss_result]
            )

            elapsed = time.time() - start_time

            avg_confidence = np.mean([s['confidence'] for s in quality_structures]) if quality_structures else 0.0

            # Flatten rejected for export
            rejected_structures = rejected_by_gate.get('B', []) + rejected_by_gate.get('C', [])

            return {
                'seq_id': seq_id,
                'sequence': sequence,
                'bsj_start': bsj_start,
                'bsj_end': bsj_end,
                'source': source,
                'ss_result': ss_result,
                'quality_structures': quality_structures,
                'rejected_structures': rejected_structures,
                'rejected_by_gate': rejected_by_gate,
                'avg_confidence': float(avg_confidence),
                'num_structures': len(quality_structures),
                'num_rejected_b': len(rejected_by_gate.get('B', [])),
                'num_rejected_c': len(rejected_by_gate.get('C', [])),
                'elapsed_seconds': elapsed,
                'status': 'success',
            }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[GPU {self.gpu_id}] Error on seq_{seq_id}: {e}")
            return {
                'seq_id': seq_id,
                'sequence': sequence,
                'source': source,
                'status': 'failed',
                'error': str(e),
                'elapsed_seconds': elapsed,
            }

    def run_batch(
        self,
        sequences: List[str],
        bsj_positions: List[Tuple[int, int]],
        sources: List[str] = None,
        checkpoint_path: str = None
    ) -> List[Dict]:
        """
        Run pipeline for a batch of sequences.

        Args:
            sequences: list of RNA sequences
            bsj_positions: list of (bsj_start, bsj_end) tuples
            sources: optional list of source labels ('real'/'synthetic'/'benchmark')
            checkpoint_path: path to save progress

        Returns:
            list of result dicts
        """
        results = []
        completed_ids = set()

        # Resume from checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        results.append(r)
                        if r.get('status') == 'success':
                            completed_ids.add(r['seq_id'])

            print(f"[GPU {self.gpu_id}] Resumed {len(results)} results from checkpoint")

        total = len(sequences)
        for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
            if i in completed_ids:
                continue

            src = sources[i] if sources and i < len(sources) else 'real'
            result = self.run_single(seq, bsj_start, bsj_end, seq_id=i, source=src)
            results.append(result)

            # Save checkpoint after each sequence
            if checkpoint_path:
                with open(checkpoint_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')

            # Progress
            done = len(completed_ids) + (i + 1 - len(completed_ids))
            if done % 10 == 0 or done == total:
                success_count = sum(1 for r in results if r.get('status') == 'success')
                avg_time = np.mean([r['elapsed_seconds'] for r in results if r.get('elapsed_seconds')])
                print(f"[GPU {self.gpu_id}] Progress: {done}/{total} | "
                      f"Success: {success_count} | "
                      f"Avg time: {avg_time:.1f}s/seq")

        return results

    def _parse_ss_pairs(self, dot_bracket: str) -> List[Tuple[int, int]]:
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


# ============================================================
# Ray-based parallel execution
# ============================================================

def run_parallel_ray(
    config_path: str,
    sequences: List[str],
    bsj_positions: List[Tuple[int, int]],
    sources: List[str] = None,
    num_workers: int = 8,
    mode: str = 'fast',
    output_dir: str = 'pipeline_output/'
) -> List[Dict]:
    """
    Run pipeline in parallel using Ray.

    Target: 6000 sequences in ~38 hours with 8 GPUs (fast mode).
    """
    import ray

    if not ray.is_initialized():
        ray.init()

    @ray.remote(num_gpus=1)
    class PipelineWorker:
        def __init__(self, config_path, mode, gpu_id):
            self.pipeline = HighThroughputPipeline(config_path, mode, gpu_id)

        def process_batch(self, sequences, bsj_positions, sources, checkpoint_path):
            return self.pipeline.run_batch(sequences, bsj_positions, sources, checkpoint_path)

    # Distribute sequences across workers
    chunks = []
    chunk_size = len(sequences) // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(sequences)
        chunk_sources = sources[start:end] if sources else ['real'] * (end - start)
        chunks.append((sequences[start:end], bsj_positions[start:end], chunk_sources))

    print(f"Distributing {len(sequences)} sequences across {num_workers} workers")
    print(f"Chunk sizes: {[len(c[0]) for c in chunks]}")

    # Create workers
    workers = [
        PipelineWorker.remote(config_path, mode, i)
        for i in range(num_workers)
    ]

    # Submit jobs
    futures = []
    for i, (worker, (seqs, bsjs, srcs)) in enumerate(zip(workers, chunks)):
        checkpoint = os.path.join(output_dir, f'checkpoint_worker_{i}.jsonl')
        future = worker.process_batch.remote(seqs, bsjs, srcs, checkpoint)
        futures.append(future)

    # Collect results
    print("Waiting for workers to complete...")
    all_results = ray.get(futures)

    # Flatten
    flat_results = []
    for batch in all_results:
        flat_results.extend(batch)

    return flat_results


def run_parallel_multiprocessing(
    config_path: str,
    sequences: List[str],
    bsj_positions: List[Tuple[int, int]],
    sources: List[str] = None,
    num_workers: int = 8,
    mode: str = 'fast',
    output_dir: str = 'pipeline_output/'
) -> List[Dict]:
    """
    Run pipeline using multiprocessing (fallback if Ray unavailable).
    """
    from multiprocessing import Pool, Manager

    def process_chunk(args):
        chunk_seqs, chunk_bsjs, chunk_srcs, gpu_id, checkpoint_path = args
        pipeline = HighThroughputPipeline(config_path, mode, gpu_id)
        return pipeline.run_batch(chunk_seqs, chunk_bsjs, chunk_srcs, checkpoint_path)

    # Distribute
    chunks = []
    chunk_size = len(sequences) // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(sequences)
        checkpoint = os.path.join(output_dir, f'checkpoint_worker_{i}.jsonl')
        chunk_sources = sources[start:end] if sources else ['real'] * (end - start)
        chunks.append((sequences[start:end], bsj_positions[start:end], chunk_sources, i, checkpoint))

    print(f"Distributing {len(sequences)} sequences across {num_workers} workers")

    with Pool(num_workers) as pool:
        all_results = pool.map(process_chunk, chunks)

    flat_results = []
    for batch in all_results:
        flat_results.extend(batch)

    return flat_results


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='trRosettaRNA2 High-Throughput circRNA Pipeline'
    )
    parser.add_argument('--config', default='config_trrosetta.yaml',
                        help='Path to config file')
    parser.add_argument('--fasta', required=True,
                        help='Input FASTA file with circRNA sequences')
    parser.add_argument('--output', default='pipeline_output/',
                        help='Output directory')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers (GPUs)')
    parser.add_argument('--mode', default='fast',
                        choices=['fast', 'high_quality', 'prefilter'],
                        help='Pipeline mode')
    parser.add_argument('--use-ray', action='store_true',
                        help='Use Ray for parallelization')
    parser.add_argument('--export-torusfold', action='store_true',
                        help='Export results in TorusFold format')
    parser.add_argument('--include-soft-noise', action='store_true',
                        help='Include B-layer soft-noise rejections (Gate1/Gate3) in TorusFold export with weight=0.3')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode (10 sequences)')

    args = parser.parse_args()

    # Load sequences
    sequences, bsj_positions, sources = load_fasta_with_bsj(args.fasta)

    if args.test:
        sequences = sequences[:10]
        bsj_positions = bsj_positions[:10]
        sources = sources[:10]
        print(f"Test mode: processing {len(sequences)} sequences")

    os.makedirs(args.output, exist_ok=True)

    # Time estimate
    est_time_per_seq = 3.0 if args.mode == 'fast' else 10.0
    est_total = len(sequences) * est_time_per_seq / 60  # minutes
    est_parallel = est_total / args.num_workers

    print(f"\n{'='*60}")
    print(f"  trRosettaRNA2 High-Throughput Pipeline")
    print(f"{'='*60}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Mode: {args.mode}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Est. time/seq: {est_time_per_seq:.0f}s")
    print(f"  Est. total: {est_total:.0f} min (serial)")
    print(f"  Est. parallel: {est_parallel:.0f} min ({args.num_workers} GPUs)")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Run pipeline
    if args.use_ray:
        results = run_parallel_ray(
            args.config, sequences, bsj_positions, sources,
            args.num_workers, args.mode, args.output
        )
    else:
        results = run_parallel_multiprocessing(
            args.config, sequences, bsj_positions, sources,
            args.num_workers, args.mode, args.output
        )

    elapsed = time.time() - start_time

    # Summary
    success = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']
    total_structures = sum(r.get('num_structures', 0) for r in success)
    avg_confidence = np.mean([r.get('avg_confidence', 0) for r in success]) if success else 0

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"  Successful: {len(success)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total structures: {total_structures}")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"  Throughput: {len(success) / (elapsed/3600):.0f} seq/hour")
    print(f"{'='*60}")

    # Save summary
    summary = {
        'pipeline': 'trRosettaRNA2',
        'mode': args.mode,
        'total_sequences': len(sequences),
        'successful': len(success),
        'failed': len(failed),
        'total_structures': total_structures,
        'avg_confidence': float(avg_confidence),
        'elapsed_minutes': elapsed / 60,
        'throughput_seqs_per_hour': len(success) / (elapsed / 3600),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(args.output, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Export TorusFold format (three-layer sampling)
    if args.export_torusfold:
        tf_dir = os.path.join(args.output, 'torusfold_format')
        os.makedirs(tf_dir, exist_ok=True)

        dataset, sources, weights, hard_negatives = [], [], [], []
        for result in success:
            source = result.get('source', 'real')
            # A-layer: quality structures
            for struct in result.get('quality_structures', []):
                tf_data = convert_to_torusfold_format(struct)
                tf_data['sequence'] = result['sequence']
                tf_data['bsj_start'] = result['bsj_start']
                tf_data['bsj_end'] = result['bsj_end']
                dataset.append(tf_data)
                weights.append(1.0)
                sources.append(source)
            # C-layer: hard negative rejections (exclude from training)
            for r in result.get('rejected_structures', []):
                if r.get('reject_layer') == 'C':
                    hard_negatives.append({
                        'pdb_path': r.get('pdb_path'),
                        'frame': r.get('frame'),
                        'reject_gate': r.get('reject_gate'),
                        'reject_reason': r.get('reject_reason'),
                        'confidence': r.get('confidence', 0.0),
                        'seq_id': r.get('seq_id'),
                        'sample_id': r.get('sample_id'),
                    })
            # B-layer: soft noise (include only if --include-soft-noise is set)
            if getattr(args, 'include_soft_noise', False):
                for r in result.get('rejected_structures', []):
                    if r.get('reject_layer') == 'B':
                        tf_data = convert_to_torusfold_format({
                            'pdb_path': r.get('pdb_path'),
                            'frame': r.get('frame'),
                            'time_ps': r.get('time_ps'),
                            'energy_kjmol': r.get('energy_kjmol'),
                            'bsj_distance_angstrom': r.get('bsj_distance_angstrom'),
                            'confidence': r.get('confidence', 0.0),
                        })
                        tf_data['sequence'] = ''
                        tf_data['bsj_start'] = -1
                        tf_data['bsj_end'] = -1
                        dataset.append(tf_data)
                        weights.append(0.3)
                        sources.append(source)

        if dataset:
            coords = np.stack([d['coords'] for d in dataset])
            confidences = np.array([d['confidence'] for d in dataset])
            sequences = [d['sequence'] for d in dataset]
            weights = np.array(weights, dtype=np.float32)
            sources_arr = np.array(sources)

            np.save(os.path.join(tf_dir, 'coords.npy'), coords)
            np.save(os.path.join(tf_dir, 'confidences.npy'), confidences)
            np.save(os.path.join(tf_dir, 'sample_weights.npy'), weights)
            np.save(os.path.join(tf_dir, 'sources.npy'), sources_arr)

            n_a = int(np.sum(weights == 1.0))
            n_b = int(np.sum(weights == 0.3))
            n_synthetic = int(np.sum(sources_arr == 'synthetic'))

            with open(os.path.join(tf_dir, 'metadata.json'), 'w') as f:
                json.dump({
                    'num_training': len(dataset),
                    'num_hard_negatives': len(hard_negatives),
                    'avg_confidence': float(np.mean(confidences)),
                    'avg_coords_shape': list(coords.shape),
                    'layers': {
                        'A (quality, weight=1.0)': n_a,
                        'B (soft noise, weight=0.3)': n_b,
                        'C (hard negative, excluded from training)': int(len(hard_negatives)),
                    },
                    'sources': {
                        'real': int(np.sum(sources_arr == 'real')),
                        'synthetic': n_synthetic,
                        'benchmark': int(np.sum(sources_arr == 'benchmark')),
                    },
                    'include_soft_noise': getattr(args, 'include_soft_noise', False),
                }, f, indent=2)

            if hard_negatives:
                np.save(os.path.join(tf_dir, 'hard_negatives.npy'),
                        np.array(hard_negatives, dtype=object))

            print(f"Exported {len(dataset)} training structures + {len(hard_negatives)} hard negatives "
                  f"to TorusFold format at {tf_dir}")
            print(f"  A-layer (quality): {n_a} | B-layer (soft noise): {n_b} | "
                  f"C-layer (hard neg): {len(hard_negatives)}")
            if n_synthetic > 0:
                print(f"  Synthetic-source samples: {n_synthetic} "
                      f"({n_synthetic/len(dataset)*100:.1f}%) — consent confirmed by user")


def load_fasta_with_bsj(fasta_path: str) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
    """
    Load FASTA file with optional BSJ and source annotations.

    Header formats:
      >seq_id bsj_start=0 bsj_end=100 source=synthetic
      >seq_id bsj_start=0 bsj_end=100
      >seq_id  (assumes BSJ at sequence ends, source='real')

    Returns:
        (sequences, bsj_positions, sources)
    """
    import re
    sequences = []
    bsj_positions = []
    sources = []

    with open(fasta_path, 'r') as f:
        current_seq = ''
        current_bsj = (0, -1)
        current_source = 'real'

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    bsj_positions.append(current_bsj)
                    sources.append(current_source)

                header = line[1:]
                current_seq = ''
                current_bsj = (0, -1)
                current_source = 'real'

                if 'bsj_start=' in header:
                    start = int(re.search(r'bsj_start=(\d+)', header).group(1))
                    end = int(re.search(r'bsj_end=(\d+)', header).group(1))
                    current_bsj = (start, end)
                m = re.search(r'source=(\w+)', header)
                if m:
                    current_source = m.group(1)
            else:
                current_seq += line

        if current_seq:
            sequences.append(current_seq)
            if current_bsj[1] == -1:
                current_bsj = (0, len(current_seq))
            bsj_positions.append(current_bsj)
            sources.append(current_source)

    return sequences, bsj_positions, sources


if __name__ == '__main__':
    main()
