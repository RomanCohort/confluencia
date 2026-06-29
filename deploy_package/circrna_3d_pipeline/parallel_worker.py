"""
Parallel Worker using Ray for DGX Spark deployment.

Distributes pipeline execution across multiple GPUs.
"""

import os
import sys
import time
import json
import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("Warning: Ray not installed. Install with: pip install ray")


class ParallelPipeline:
    """Parallel pipeline execution using Ray actors."""

    def __init__(self, config_path=None, num_workers=8):
        if not HAS_RAY:
            raise RuntimeError("Ray is required for parallel execution")

        self.config_path = config_path
        self.num_workers = num_workers

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create workers
        self.workers = [
            PipelineWorker.options(num_gpus=1).remote(config_path, i)
            for i in range(num_workers)
        ]

        print(f"Initialized {num_workers} parallel workers")

    def run_batch(self, sequences, bsj_positions):
        """
        Run pipeline in parallel on multiple workers.

        Args:
            sequences: list of RNA sequence strings
            bsj_positions: list of (bsj_start, bsj_end) tuples

        Returns:
            list of results
        """
        futures = []

        for i, (seq, (bsj_start, bsj_end)) in enumerate(zip(sequences, bsj_positions)):
            worker_id = i % self.num_workers
            future = self.workers[worker_id].process.remote(
                seq, bsj_start, bsj_end, i
            )
            futures.append(future)

        # Collect results with progress tracking
        results = []
        remaining = list(enumerate(futures))

        while remaining:
            ready_ids = ray.wait([f for _, f in remaining], timeout=5.0)[0]

            for i, future in remaining[:]:
                if future in ready_ids:
                    try:
                        result = ray.get(future)
                        results.append(result)
                        print(f"Completed seq_{result['seq_id']} ({len(results)}/{len(futures)})")
                    except Exception as e:
                        print(f"Failed seq_{i}: {e}")
                        results.append({'seq_id': i, 'error': str(e)})

                    remaining.remove((i, future))

        # Sort by seq_id
        results.sort(key=lambda x: x.get('seq_id', 0))

        return results

    def shutdown(self):
        """Shutdown Ray."""
        ray.shutdown()


@ray.remote
class PipelineWorker:
    """Ray actor for processing circRNA sequences."""

    def __init__(self, config_path, worker_id):
        self.worker_id = worker_id
        self.config_path = config_path

        # Import pipeline (deferred to worker process)
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline import CircRNA3DPipeline

        self.pipeline = CircRNA3DPipeline(config_path)
        print(f"Worker {worker_id} initialized on GPU {ray.get_gpu_ids()}")

    def process(self, sequence, bsj_start, bsj_end, seq_id):
        """Process a single circRNA sequence."""
        try:
            result = self.pipeline.run_single(sequence, bsj_start, bsj_end, seq_id)
            return result
        except Exception as e:
            import traceback
            return {
                'seq_id': seq_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


def main():
    """Example usage for DGX Spark."""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel circRNA 3D Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--output', default='pipeline_output/', help='Output directory')
    parser.add_argument('--export-torusfold', action='store_true', help='Export TorusFold format')

    args = parser.parse_args()

    # Load sequences
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import load_fasta_with_bsj

    sequences, bsj_positions = load_fasta_with_bsj(args.fasta)
    print(f"Loaded {len(sequences)} sequences from {args.fasta}")

    # Initialize parallel pipeline
    parallel = ParallelPipeline(args.config, args.num_workers)

    # Run
    start_time = time.time()
    results = parallel.run_batch(sequences, bsj_positions)
    elapsed = time.time() - start_time

    # Export results
    os.makedirs(args.output, exist_ok=True)

    from pipeline import CircRNA3DPipeline
    pipeline = CircRNA3DPipeline(args.config)
    pipeline.output_dir = args.output

    # Filter out failed results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"\nCompleted: {len(successful)}/{len(results)} sequences")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average: {elapsed/len(sequences):.1f}s per sequence")

    if successful:
        pipeline.export_dataset(successful)

        if args.export_torusfold:
            pipeline.export_torusfold(successful)

    # Save failed sequences for retry
    if failed:
        with open(os.path.join(args.output, 'failed.json'), 'w') as f:
            json.dump(failed, f, indent=2)

    parallel.shutdown()


if __name__ == '__main__':
    main()
