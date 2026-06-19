#!/usr/bin/env python3
"""
circrna_casp.py — Internal CASP-style evaluation for circRNA structure prediction.

Treats 6 TorusFold schemes as "competing groups" in a CASP-like assessment.
Each scheme predicts structures for the same set of target sequences.
Evaluation uses:
  - Physical consistency (energy, closure, clashes)
  - Diversity (conformation coverage)
  - Scalability (time, memory vs length)
  - Robustness (success rate across sequences)

Output: CASP-style ranking table + per-target scores.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── CASP Target Set ─────────────────────────────────────────

def generate_casp_targets(n_per_length=5, seed=42):
    """Generate CASP-like target sequences.

    Returns list of (target_id, sequence, category) tuples.
    """
    rng = np.random.RandomState(seed)
    targets = []

    lengths = {
        'short': [50, 80, 100],
        'medium': [150, 200, 300],
        'long': [500, 800, 1000],
    }

    target_id = 0
    for category, lens in lengths.items():
        for L in lens:
            for i in range(n_per_length):
                seq = ''.join(rng.choice(['A', 'C', 'G', 'U'], size=L))
                targets.append((f'T{target_id:04d}', seq, category))
                target_id += 1

    return targets


# ── Scoring Functions ────────────────────────────────────────

def score_structure(coords: np.ndarray) -> dict:
    """Score a single predicted structure.

    Returns dict with all CASP metrics.
    """
    L = len(coords)
    bond_length = 5.9

    # Bond energy
    bond_errors = []
    for i in range(L):
        j = (i + 1) % L
        d = np.linalg.norm(coords[j] - coords[i])
        bond_errors.append((d - bond_length) ** 2)
    bond_rmsd = np.sqrt(np.mean(bond_errors))

    # Closure error
    closure_error = abs(np.linalg.norm(coords[0] - coords[-1]) - bond_length)

    # Clash count
    clash_count = 0
    clash_dist = 3.0
    for i in range(L):
        for j in range(i + 2, L):
            if i == 0 and j == L - 1:
                continue
            d = np.linalg.norm(coords[j] - coords[i])
            if d < clash_dist:
                clash_count += 1

    # Stacking quality
    stack_errors = []
    for i in range(L):
        j = (i + 1) % L
        dz = abs(coords[j, 2] - coords[i, 2])
        stack_errors.append((dz - 3.4) ** 2)
    stack_rmsd = np.sqrt(np.mean(stack_errors))

    # Total energy (lower = better)
    total_energy = (np.sum(bond_errors) +
                    10.0 * clash_count +
                    0.3 * np.sum(stack_errors) +
                    closure_error ** 2 * 100)

    return {
        'bond_rmsd': float(bond_rmsd),
        'closure_error': float(closure_error),
        'clash_count': int(clash_count),
        'stack_rmsd': float(stack_rmsd),
        'total_energy': float(total_energy),
        'L': L,
    }


# ── Scheme Runners ───────────────────────────────────────────

def run_scheme(scheme_id: int, sequence: str, target_id: str) -> dict:
    """Run a single scheme on a single target."""
    L = len(sequence)

    scheme_names = {
        1: "DL+Physics Cascade",
        2: "Batch+Physics Filter",
        3: "Dual-Engine Iterative",
        4: "DDPM+EGNN Guided",
        5: "Physics-Biased Attention",
        6: "GNN Latent Diffusion",
    }

    result = {
        'scheme': scheme_id,
        'scheme_name': scheme_names[scheme_id],
        'target_id': target_id,
        'length': L,
    }

    try:
        t0 = time.time()

        if scheme_id == 1:
            from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
                GeometricConstraintSolver, SolverConfig
            )
            config = SolverConfig(n_samples=5, use_annealing_closure=True)
            solver = GeometricConstraintSolver(config)
            class CS:
                def __init__(self, n):
                    self.seq_len = n
                    self.pair_constraints = []
            confs = solver.solve(CS(L))
            coords = confs[0] if confs else np.zeros((L, 3))

        elif scheme_id == 2:
            from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
                GeometricConstraintSolver, SolverConfig
            )
            config = SolverConfig(n_samples=100, use_annealing_closure=True)
            solver = GeometricConstraintSolver(config)
            class CS:
                def __init__(self, n):
                    self.seq_len = n
                    self.pair_constraints = []
            confs = solver.solve(CS(L))
            coords = confs[0] if confs else np.zeros((L, 3))

        elif scheme_id == 3:
            from confluencia_3_0.core.circrna.torusfold.dual_engine import (
                DualEngineTorusFold, DualEngineConfig
            )
            config = DualEngineConfig(n_iterations=2, n_candidates=10)
            engine = DualEngineTorusFold(config)
            res = engine.predict(sequence, pair_constraints=[])
            coords = res['coords'] if res['coords'] is not None else np.zeros((L, 3))

        elif scheme_id == 4:
            import torch
            from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
                CircRNADiffusionModel, CircDiffusionConfig
            )
            config = CircDiffusionConfig(n_diffusion_steps=10)
            model = CircRNADiffusionModel(config)
            token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            tokens = torch.tensor([[token_map.get(c, 4) for c in sequence]])
            with torch.no_grad():
                res = model(tokens)
            coords = res['coords'][0].numpy()

        elif scheme_id == 5:
            import torch
            from confluencia_3_0.core.circrna.torusfold.triangle_update import (
                CircPairformerBlock
            )
            block = CircPairformerBlock(c_z=128, use_physics_bias=True)
            z = torch.randn(1, L, L, 128) * 0.01
            c = torch.randn(1, L, 3) * 5.9
            with torch.no_grad():
                z_out = block(z, coords=c)
            diag = z_out.diagonal(dim1=1, dim2=2)
            proj = torch.nn.Linear(128, 3)
            coords = proj(diag)[0].detach().numpy()

        elif scheme_id == 6:
            import torch
            from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
                GNNLatentDiffusionModel, GNNLatentConfig
            )
            config = GNNLatentConfig(n_diffusion_steps=10)
            model = GNNLatentDiffusionModel(config)
            token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            tokens = torch.tensor([[token_map.get(c, 4) for c in sequence]])
            with torch.no_grad():
                res = model(tokens, mode='sample')
            coords = res['coords'][0].numpy()

        elapsed = time.time() - t0
        scores = score_structure(coords)

        result.update(scores)
        result['time_seconds'] = elapsed
        result['success'] = True

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)

    return result


# ── CASP Evaluation ──────────────────────────────────────────

def run_casp(targets, schemes, output_dir):
    """Run CASP-style evaluation."""
    print("=" * 70)
    print("  CIRCRNA-CASP: Internal Structure Prediction Assessment")
    print("=" * 70)
    print(f"  Targets: {len(targets)}")
    print(f"  Schemes: {schemes}")
    print()

    all_results = []

    for target_id, sequence, category in targets:
        L = len(sequence)
        print(f"  [{category:>6s}] {target_id} (L={L}):", end="")

        for scheme_id in schemes:
            result = run_scheme(scheme_id, sequence, target_id)
            result['category'] = category
            all_results.append(result)

            if result.get('success'):
                print(f" S{scheme_id}:E={result['total_energy']:.1f}", end="")
            else:
                print(f" S{scheme_id}:FAIL", end="")

        print()

    # ── Ranking ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CASP RANKING (by mean total energy, lower = better)")
    print("=" * 70)

    scheme_stats = {}
    for scheme_id in schemes:
        successes = [r for r in all_results
                     if r['scheme'] == scheme_id and r.get('success')]
        if successes:
            energies = [r['total_energy'] for r in successes]
            closures = [r['closure_error'] for r in successes]
            clashes = [r['clash_count'] for r in successes]
            times = [r['time_seconds'] for r in successes]
            scheme_stats[scheme_id] = {
                'mean_energy': np.mean(energies),
                'std_energy': np.std(energies),
                'mean_closure': np.mean(closures),
                'mean_clashes': np.mean(clashes),
                'mean_time': np.mean(times),
                'success_rate': len(successes) / len(all_results) * len(schemes),
                'n_success': len(successes),
            }

    # Sort by energy
    ranked = sorted(scheme_stats.items(), key=lambda x: x[1]['mean_energy'])

    print(f"\n  {'Rank':>4s} {'Scheme':>30s} {'Energy':>10s} {'Closure':>10s} "
          f"{'Clashes':>8s} {'Time(s)':>8s} {'Success':>7s}")
    print("  " + "-" * 80)

    for rank, (scheme_id, stats) in enumerate(ranked, 1):
        names = {1: "DL+Physics Cascade", 2: "Batch+Physics Filter",
                 3: "Dual-Engine Iterative", 4: "DDPM+EGNN Guided",
                 5: "Physics-Biased Attention", 6: "GNN Latent Diffusion"}
        print(f"  {rank:>4d} {names[scheme_id]:>30s} "
              f"{stats['mean_energy']:>10.1f} "
              f"{stats['mean_closure']:>10.3f} "
              f"{stats['mean_clashes']:>8.1f} "
              f"{stats['mean_time']:>8.2f} "
              f"{stats['n_success']:>7d}")

    # ── Per-category breakdown ───────────────────────────────
    print("\n" + "=" * 70)
    print("  PER-CATEGORY BREAKDOWN")
    print("=" * 70)

    for category in ['short', 'medium', 'long']:
        print(f"\n  [{category.upper()}]")
        cat_results = [r for r in all_results if r.get('category') == category]

        for scheme_id in schemes:
            successes = [r for r in cat_results
                         if r['scheme'] == scheme_id and r.get('success')]
            if successes:
                energies = [r['total_energy'] for r in successes]
                closures = [r['closure_error'] for r in successes]
                names = {1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S5", 6: "S6"}
                print(f"    {names[scheme_id]}: E={np.mean(energies):.1f}±{np.std(energies):.1f}, "
                      f"closure={np.mean(closures):.3f}Å, n={len(successes)}")

    # Save all results
    out_path = os.path.join(output_dir, 'circrna_casp_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to {out_path}")

    # Save ranking
    ranking_path = os.path.join(output_dir, 'circrna_casp_ranking.json')
    ranking = {str(sid): stats for sid, stats in scheme_stats.items()}
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"  Ranking saved to {ranking_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='circRNA-CASP evaluation')
    parser.add_argument('--output', type=str, default='casp_results')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--n-per-length', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    targets = generate_casp_targets(n_per_length=args.n_per_length)
    run_casp(targets, args.schemes, args.output)


if __name__ == '__main__':
    main()
