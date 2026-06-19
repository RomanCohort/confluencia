#!/usr/bin/env python3
"""
benchmark_schemes.py — Benchmark all 6 structure prediction schemes.

Usage:
    python benchmark_schemes.py --output results/benchmark.csv
    python benchmark_schemes.py --schemes 1 2 3 --lengths 100 200 500
    python benchmark_schemes.py --n-sequences 20 --device cuda

Experiments:
    1. Physics consistency (energy, closure, clashes)
    2. Scalability (time, memory vs sequence length)
    3. Diversity (pairwise RMSD across multiple samples)
"""

import argparse
import os
import sys
import time
import json
import traceback
from pathlib import Path

import numpy as np

# ── Sequence Generation ──────────────────────────────────────

def random_circrna_sequence(length: int, seed: int = 42) -> str:
    """Generate random circRNA sequence."""
    rng = np.random.RandomState(seed)
    return ''.join(rng.choice(['A', 'C', 'G', 'U'], size=length))


# ── Metrics ──────────────────────────────────────────────────

def compute_bond_energy(coords: np.ndarray, bond_length: float = 5.9) -> float:
    L = len(coords)
    energy = 0.0
    for i in range(L):
        j = (i + 1) % L
        d = np.linalg.norm(coords[j] - coords[i])
        energy += (d - bond_length) ** 2
    return energy


def compute_closure_error(coords: np.ndarray, bond_length: float = 5.9) -> float:
    return abs(np.linalg.norm(coords[0] - coords[-1]) - bond_length)


def count_clashes(coords: np.ndarray, clash_dist: float = 3.0) -> int:
    L = len(coords)
    count = 0
    for i in range(L):
        for j in range(i + 2, L):
            if i == 0 and j == L - 1:
                continue
            d = np.linalg.norm(coords[j] - coords[i])
            if d < clash_dist:
                count += 1
    return count


def compute_stacking_energy(coords: np.ndarray, stack_dist: float = 3.4) -> float:
    L = len(coords)
    energy = 0.0
    for i in range(L):
        j = (i + 1) % L
        dz = abs(coords[j, 2] - coords[i, 2])
        energy += (dz - stack_dist) ** 2
    return energy


def compute_total_energy(coords: np.ndarray) -> float:
    e_bond = compute_bond_energy(coords)
    e_clash = count_clashes(coords) * 10.0  # weight
    e_stack = compute_stacking_energy(coords) * 0.3  # weight
    return e_bond + e_clash + e_stack


def compute_pairwise_rmsd(coords_list: list) -> float:
    """Average pairwise RMSD across conformations."""
    n = len(coords_list)
    if n < 2:
        return 0.0
    rmsds = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords_list[i] - coords_list[j]
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
            rmsds.append(rmsd)
    return np.mean(rmsds)


# ── Scheme Implementations ───────────────────────────────────

def run_scheme_1(sequence: str, n_samples: int = 5) -> dict:
    """Scheme 1: DL prediction + physics refinement (cascade)."""
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig
    )

    L = len(sequence)
    config = SolverConfig(
        n_samples=n_samples,
        use_annealing_closure=True,
    )
    solver = GeometricConstraintSolver(config)

    class MinimalConstraintSet:
        def __init__(self, seq_len):
            self.seq_len = seq_len
            self.pair_constraints = []

    constraint_set = MinimalConstraintSet(L)
    conformations = solver.solve(constraint_set)

    if conformations:
        best = conformations[0]
    else:
        best = np.zeros((L, 3))

    return {'coords': best, 'n_conformations': len(conformations)}


def run_scheme_2(sequence: str, n_samples: int = 100) -> dict:
    """Scheme 2: Batch generation + physics scoring."""
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig
    )

    L = len(sequence)
    config = SolverConfig(
        n_samples=n_samples,
        use_annealing_closure=True,
    )
    solver = GeometricConstraintSolver(config)

    class MinimalConstraintSet:
        def __init__(self, seq_len):
            self.seq_len = seq_len
            self.pair_constraints = []

    constraint_set = MinimalConstraintSet(L)
    conformations = solver.solve(constraint_set)

    if conformations:
        best = conformations[0]
    else:
        best = np.zeros((L, 3))

    return {'coords': best, 'n_conformations': len(conformations)}


def run_scheme_3(sequence: str, n_iterations: int = 3) -> dict:
    """Scheme 3: Dual-engine iterative evolution."""
    from confluencia_3_0.core.circrna.torusfold.dual_engine import (
        DualEngineTorusFold, DualEngineConfig
    )

    L = len(sequence)
    config = DualEngineConfig(
        n_iterations=n_iterations,
        n_candidates=20,
    )
    engine = DualEngineTorusFold(config)

    result = engine.predict(sequence, pair_constraints=[])

    return {
        'coords': result['coords'],
        'energy_history': result['energy_history'],
        'n_iterations': result['n_iterations'],
    }


def run_scheme_4(sequence: str) -> dict:
    """Scheme 4: DDPM + EGNN + Guided Diffusion."""
    import torch
    from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
        CircRNADiffusionModel, CircDiffusionConfig
    )

    L = len(sequence)
    config = CircDiffusionConfig(n_diffusion_steps=20)
    model = CircRNADiffusionModel(config)

    # Tokenize sequence
    token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens = torch.tensor([[token_map.get(c, 4) for c in sequence]])

    with torch.no_grad():
        result = model(tokens, temperature=310.0, pH=7.4, Mg_conc=1.0, Na_conc=150.0)

    coords = result['coords'][0].numpy()
    return {'coords': coords}


def run_scheme_5(sequence: str) -> dict:
    """Scheme 5: CircPairformer with physics bias (test architecture)."""
    import torch
    from confluencia_3_0.core.circrna.torusfold.triangle_update import (
        CircPairformerBlock, PhysicsConstraintBias
    )

    L = len(sequence)
    block = CircPairformerBlock(c_z=128, use_physics_bias=True)

    # Dummy pair representation
    z = torch.randn(1, L, L, 128) * 0.01

    # Dummy coordinates for physics bias
    coords = torch.randn(1, L, 3) * 5.9

    with torch.no_grad():
        z_out = block(z, coords=coords)

    # Extract pseudo-coords from pair repr (diagonal)
    diag = z_out.diagonal(dim1=1, dim2=2)  # (1, L, 128)
    # Simple projection to 3D
    proj = torch.nn.Linear(128, 3)
    pseudo_coords = proj(diag)[0].detach().numpy()

    return {'coords': pseudo_coords}


def run_scheme_6(sequence: str) -> dict:
    """Scheme 6 Path 2: GNN encoder + latent diffusion + decoder."""
    import torch
    from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
        GNNLatentDiffusionModel, GNNLatentConfig
    )

    L = len(sequence)
    config = GNNLatentConfig(n_diffusion_steps=10)
    model = GNNLatentDiffusionModel(config)

    # Tokenize sequence
    token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    tokens = torch.tensor([[token_map.get(c, 4) for c in sequence]])

    with torch.no_grad():
        result = model(tokens, mode='sample')

    coords = result['coords'][0].numpy()
    return {'coords': coords}


SCHEME_RUNNERS = {
    1: run_scheme_1,
    2: run_scheme_2,
    3: run_scheme_3,
    4: run_scheme_4,
    5: run_scheme_5,
    6: run_scheme_6,
}

SCHEME_NAMES = {
    1: "DL+Physics Cascade",
    2: "Batch+Physics Filter",
    3: "Dual-Engine Iterative",
    4: "Conditional Diffusion",
    5: "Physics-Biased Attention",
    6: "Diffusion+GNN Hybrid",
}


# ── Main Benchmark ───────────────────────────────────────────

def benchmark_physics_consistency(schemes, lengths, n_sequences, output_dir):
    """Experiment 1: Physics consistency metrics."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Physics Consistency")
    print("="*60)

    results = []
    for length in lengths:
        for seq_idx in range(n_sequences):
            sequence = random_circrna_sequence(length, seed=seq_idx)
            print(f"\n  L={length}, seq={seq_idx}: {sequence[:20]}...")

            for scheme_id in schemes:
                runner = SCHEME_RUNNERS[scheme_id]
                name = SCHEME_NAMES[scheme_id]

                try:
                    t0 = time.time()
                    result = runner(sequence)
                    elapsed = time.time() - t0

                    coords = result['coords']
                    metrics = {
                        'scheme': scheme_id,
                        'scheme_name': name,
                        'length': length,
                        'seq_idx': seq_idx,
                        'bond_energy': compute_bond_energy(coords),
                        'closure_error': compute_closure_error(coords),
                        'clash_count': count_clashes(coords),
                        'stacking_energy': compute_stacking_energy(coords),
                        'total_energy': compute_total_energy(coords),
                        'time_seconds': elapsed,
                        'success': True,
                    }
                    # Extra info from scheme 3
                    if scheme_id == 3 and 'energy_history' in result:
                        metrics['energy_history'] = result['energy_history']
                        metrics['n_iterations'] = result['n_iterations']

                except Exception as e:
                    print(f"    Scheme {scheme_id} FAILED: {e}")
                    metrics = {
                        'scheme': scheme_id,
                        'scheme_name': name,
                        'length': length,
                        'seq_idx': seq_idx,
                        'success': False,
                        'error': str(e),
                    }

                results.append(metrics)
                if metrics.get('success'):
                    print(f"    {name}: E={metrics['total_energy']:.2f}, "
                          f"closure={metrics['closure_error']:.3f}Å, "
                          f"clashes={metrics['clash_count']}, "
                          f"time={metrics['time_seconds']:.2f}s")

    # Save
    out_path = os.path.join(output_dir, 'physics_consistency.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return results


def benchmark_scalability(schemes, max_length=2000, output_dir='.'):
    """Experiment 2: Scalability (time + memory vs length)."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Scalability")
    print("="*60)

    lengths = [50, 100, 200, 500, 1000, 2000]
    results = []

    for length in lengths:
        if length > max_length:
            break
        sequence = random_circrna_sequence(length, seed=0)
        print(f"\n  L={length}")

        for scheme_id in schemes:
            runner = SCHEME_RUNNERS[scheme_id]
            name = SCHEME_NAMES[scheme_id]

            try:
                import tracemalloc
                tracemalloc.start()

                t0 = time.time()
                result = runner(sequence)
                elapsed = time.time() - t0

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                metrics = {
                    'scheme': scheme_id,
                    'scheme_name': name,
                    'length': length,
                    'time_seconds': elapsed,
                    'peak_memory_mb': peak / 1024 / 1024,
                    'success': True,
                }

            except Exception as e:
                tracemalloc.stop()
                print(f"    Scheme {scheme_id} FAILED at L={length}: {e}")
                metrics = {
                    'scheme': scheme_id,
                    'scheme_name': name,
                    'length': length,
                    'success': False,
                    'error': str(e),
                }

            results.append(metrics)
            if metrics.get('success'):
                print(f"    {name}: time={metrics['time_seconds']:.2f}s, "
                      f"mem={metrics['peak_memory_mb']:.1f}MB")

    out_path = os.path.join(output_dir, 'scalability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return results


def benchmark_diversity(schemes, length=200, n_samples=20, output_dir='.'):
    """Experiment 3: Conformation diversity."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Conformation Diversity")
    print("="*60)

    sequence = random_circrna_sequence(length, seed=0)
    results = []

    for scheme_id in schemes:
        if scheme_id in [5]:  # Scheme 5 doesn't support multi-sample
            continue

        runner = SCHEME_RUNNERS[scheme_id]
        name = SCHEME_NAMES[scheme_id]
        print(f"\n  {name}:")

        candidates = []
        for i in range(n_samples):
            try:
                # Use different random seed for diversity
                np.random.seed(i * 42)
                result = runner(sequence)
                candidates.append(result['coords'])
            except Exception as e:
                print(f"    Sample {i} failed: {e}")

        if len(candidates) >= 2:
            diversity = compute_pairwise_rmsd(candidates)
            energies = [compute_total_energy(c) for c in candidates]
            metrics = {
                'scheme': scheme_id,
                'scheme_name': name,
                'length': length,
                'n_candidates': len(candidates),
                'diversity_rmsd': diversity,
                'energy_mean': np.mean(energies),
                'energy_std': np.std(energies),
                'energy_min': np.min(energies),
                'success': True,
            }
            print(f"    diversity={diversity:.2f}Å, "
                  f"E_mean={metrics['energy_mean']:.2f}, "
                  f"E_min={metrics['energy_min']:.2f}")
        else:
            metrics = {
                'scheme': scheme_id,
                'scheme_name': name,
                'success': False,
                'error': 'insufficient candidates',
            }

        results.append(metrics)

    out_path = os.path.join(output_dir, 'diversity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark 6 structure schemes')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1,2,3,4,5,6],
                        help='Schemes to benchmark')
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[50, 100, 200],
                        help='Sequence lengths for physics test')
    parser.add_argument('--n-sequences', type=int, default=5,
                        help='Number of sequences per length')
    parser.add_argument('--max-length', type=int, default=1000,
                        help='Max length for scalability test')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'physics', 'scalability', 'diversity'],
                        help='Which experiment to run')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Confluencia Scheme Benchmark")
    print(f"Schemes: {args.schemes}")
    print(f"Output: {args.output}")

    if args.experiment in ['all', 'physics']:
        benchmark_physics_consistency(
            args.schemes, args.lengths, args.n_sequences, args.output
        )

    if args.experiment in ['all', 'scalability']:
        benchmark_scalability(args.schemes, args.max_length, args.output)

    if args.experiment in ['all', 'diversity']:
        benchmark_diversity(args.schemes, output_dir=args.output)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
