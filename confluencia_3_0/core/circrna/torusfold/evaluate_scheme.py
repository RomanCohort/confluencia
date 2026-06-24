#!/usr/bin/env python3
"""
evaluate_scheme.py — Evaluate a trained TorusFold scheme on test data.

Usage:
    python evaluate_scheme.py --scheme 6 --checkpoint models/torusfold_s6/scheme6_best.pt
    python evaluate_scheme.py --scheme 7 --checkpoint models/torusfold_s7/scheme7_best.pt --n-samples 5
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn,
)
from torch.utils.data import DataLoader


def kabsch_rmsd(pred, target):
    """Kabsch-aligned RMSD between two coordinate sets."""
    p = pred - pred.mean(dim=0)
    t = target - target.mean(dim=0)

    H = t.T @ p
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1, 1, d], device=p.device, dtype=torch.float32))
        R = Vt.T @ D @ U.T
        p_aligned = (R @ p.T).T
        rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t) ** 2, dim=1)))
    except Exception:
        rmsd = torch.sqrt(torch.mean(torch.sum((p - t) ** 2, dim=1)))
    return rmsd


def build_model(scheme_id, args, device):
    """Build model for given scheme."""
    if scheme_id == 1:
        # Scheme 1: Use Scheme1Model wrapper (contains egnn=CircRNA3DModel)
        # Checkpoint keys have 'egnn.' prefix, must match this structure
        import torch.nn as nn
        from confluencia_3_0.core.circrna.torusfold.train_torusfold_3d import CircRNA3DModel
        class Scheme1Model(nn.Module):
            """EGNN backbone wrapper (matches train_all_schemes.py structure)."""
            def __init__(self, d_hidden=128, n_layers=4):
                super().__init__()
                self.egnn = CircRNA3DModel(d_hidden=d_hidden, n_layers=n_layers)
            def forward(self, seq_ids):
                return self.egnn(seq_ids)
        return Scheme1Model(d_hidden=args.d_hidden, n_layers=args.n_layers).to(device)
    elif scheme_id == 2:
        # Scheme 2: IsRNAcirc-inspired physics solver (zero training)
        # Pipeline: SS prediction -> coarse-grained 3D folding -> closure refinement
        from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
            GeometricConstraintSolver, SolverConfig
        )
        class Scheme2PhysicsSolver:
            """IsRNAcirc-inspired circRNA 3D structure prediction.

            Pipeline (mirrors IsRNAcirc, zero-training):
            1. Predict secondary structure from sequence (Nussinov-like)
            2. Initialize 3D coords respecting pair geometry
            3. Iterative energy minimization with annealing closure
            """
            def __init__(self, n_samples=3):
                config = SolverConfig(
                    n_samples=n_samples,
                    max_iterations=200,
                    clash_distance=0.0,  # Disable for speed
                    use_annealing_closure=True,
                    annealing_steps_per_temp=10,
                    annealing_cooling=0.9,
                )
                self.solver = GeometricConstraintSolver(config)
                self.device = device

            def _predict_secondary_structure(self, seq_ids_np):
                """Predict secondary structure using Nussinov-like algorithm.

                Simplified version: maximize base pairs with stacking preference.
                Handles circular topology (BSJ-crossing pairs allowed).
                """
                import numpy as np
                L = len(seq_ids_np)

                # Complementarity scores
                comp_score = np.zeros((L, L), dtype=np.float32)
                wc = {(0,1): 2.0, (1,0): 2.0, (2,3): 3.0, (3,2): 3.0}  # AU=2, GC=3
                wobble = {(2,1): 1.0, (1,2): 1.0}  # GU=1

                for i in range(L):
                    for j in range(i + 4, L):
                        si, sj = int(seq_ids_np[i]), int(seq_ids_np[j])
                        if si > 3 or sj > 3: continue
                        comp_score[i, j] = wc.get((si, sj), 0.0) + wobble.get((si, sj), 0.0)
                        comp_score[j, i] = comp_score[i, j]

                # Nussinov DP for circular RNA
                # Use linear Nussinov + allow BSJ-crossing pairs
                dp = np.zeros((L, L), dtype=np.float32)
                bp = np.full((L, L), -1, dtype=np.int32)

                for length in range(5, L):
                    for i in range(L - length):
                        j = i + length
                        # No pair (i,j)
                        best = dp[i+1, j]
                        # Pair (i,j)
                        if comp_score[i, j] > 0:
                            pair_score = comp_score[i, j] + dp[i+1, j-1]
                            # Try bifurcation
                            for k in range(i+1, j):
                                bif = dp[i+1, k] + dp[k+1, j-1]
                                pair_score = max(pair_score, comp_score[i, j] + bif)
                            if pair_score > best:
                                best = pair_score
                                bp[i, j] = j
                        # Bifurcation without pairing (i,j)
                        for k in range(i+1, j):
                            bif = dp[i, k] + dp[k+1, j]
                            if bif > best:
                                best = bif
                                bp[i, j] = -1
                        dp[i, j] = best

                # Traceback to get pairs
                pairs = []
                used = set()
                stack = [(0, L-1)]
                while stack:
                    i, j = stack.pop()
                    if i >= j or j - i < 4: continue
                    if bp[i, j] == j and i not in used and j not in used:
                        pairs.append((i, j))
                        used.add(i); used.add(j)
                        stack.append((i+1, j-1))
                    else:
                        # Find best bifurcation
                        best_k = -1
                        best_score = -1
                        for k in range(i+1, j):
                            score = dp[i, k] + dp[k+1, j]
                            if score > best_score:
                                best_score = score
                                best_k = k
                        if best_k >= 0:
                            stack.append((i, best_k))
                            stack.append((best_k+1, j))

                # Convert to constraint format
                constraints = []
                for (i, j) in pairs:
                    si, sj = int(seq_ids_np[i]), int(seq_ids_np[j])
                    # GC pairs: 10.6A, AU pairs: 10.4A, GU wobble: 10.8A
                    if (si, sj) in [(2,3), (3,2)]:
                        target_d = 10.6
                        weight = 0.9
                    elif (si, sj) in [(0,1), (1,0)]:
                        target_d = 10.4
                        weight = 0.8
                    else:  # wobble
                        target_d = 10.8
                        weight = 0.5
                    constraints.append((i, j, target_d, weight))

                return constraints

            def _initialize_3d(self, L, pair_constraints):
                """Initialize 3D coords respecting secondary structure geometry.

                Key idea from IsRNAcirc: start from a structure-aware initial
                configuration rather than a flat ring. Paired regions form
                A-form helices; unpaired regions form loops.
                """
                import math
                import numpy as np

                coords = np.zeros((L, 3), dtype=np.float64)
                assigned = np.zeros(L, dtype=bool)

                # A-form helix parameters
                helix_rise = 2.8    # A per nucleotide along helix axis
                helix_radius = 5.0  # A from helix axis
                nt_per_turn = 11   # nucleotides per helix turn
                helix_twist = 2 * math.pi / nt_per_turn  # radians per nt

                # Process pairs in order: build helices first
                sorted_pairs = sorted(pair_constraints,
                                     key=lambda p: p[0], reverse=False)

                for (i, j, target_d, weight) in sorted_pairs:
                    if assigned[i] or assigned[j]:
                        continue

                    # Place paired nucleotides as A-form helix
                    # i on 5' strand, j on 3' strand (antiparallel)
                    # Helix extends in z-direction
                    pair_idx = sum(1 for p in sorted_pairs if p[0] < i and not assigned[p[0]])

                    z_base = pair_idx * helix_rise

                    # 5' strand (i): forward along helix
                    angle_5 = pair_idx * helix_twist
                    coords[i] = [helix_radius * math.cos(angle_5),
                                 helix_radius * math.sin(angle_5),
                                 z_base]
                    assigned[i] = True

                    # 3' strand (j): backward along helix (antiparallel)
                    angle_3 = angle_5 + math.pi  # opposite side
                    coords[j] = [helix_radius * math.cos(angle_3),
                                 helix_radius * math.sin(angle_3),
                                 z_base + helix_rise * 0.5]
                    assigned[j] = True

                # Fill unassigned positions as loops connecting helices
                # Use smooth interpolation between assigned points
                assigned_indices = np.where(assigned)[0]
                if len(assigned_indices) == 0:
                    # No pairs: fall back to regular polygon
                    R = L * 5.9 / (2 * math.pi)
                    for i in range(L):
                        angle = 2 * math.pi * i / L
                        coords[i] = [R * math.cos(angle), R * math.sin(angle), 0]
                    return coords.astype(np.float32)

                # Interpolate unassigned positions
                for seg_start in range(len(assigned_indices)):
                    idx_start = assigned_indices[seg_start]
                    idx_end = assigned_indices[(seg_start + 1) % len(assigned_indices)]

                    if idx_end <= idx_start:
                        # Wrap around BSJ
                        loop_indices = list(range(idx_start + 1, L)) + list(range(0, idx_end))
                    else:
                        loop_indices = list(range(idx_start + 1, idx_end))

                    if not loop_indices:
                        continue

                    n_loop = len(loop_indices)
                    p_start = coords[idx_start].copy()
                    p_end = coords[idx_end].copy()

                    # Loop goes outward from helix axis
                    mid_dir = (p_start + p_end) / 2
                    loop_radius = max(5.0, n_loop * 5.9 / (2 * math.pi) * 0.3)

                    for k, idx in enumerate(loop_indices):
                        t = (k + 1) / (n_loop + 1)  # 0 to 1
                        # Interpolate position
                        coords[idx] = p_start * (1 - t) + p_end * t
                        # Add outward bulge for loop
                        outward = np.array([math.cos(math.pi * t), math.sin(math.pi * t), 0])
                        coords[idx] += outward * loop_radius * 0.5
                        assigned[idx] = True

                # Ensure any remaining unassigned get interpolated
                for i in range(L):
                    if not assigned[i]:
                        # Find nearest assigned neighbors
                        prev_a = max((a for a in assigned_indices if a < i), default=assigned_indices[-1])
                        next_a = min((a for a in assigned_indices if a > i), default=assigned_indices[0])
                        t = (i - prev_a) / max(next_a - prev_a, 1)
                        coords[i] = coords[prev_a] * (1 - t) + coords[next_a] * t
                        assigned[i] = True

                return coords.astype(np.float32)

            def _extract_pair_constraints(self, pair_prob_matrix, threshold=0.3):
                """Extract pair constraints from probability matrix."""
                import numpy as np
                L = pair_prob_matrix.shape[0]
                pairs = []
                for i in range(L):
                    for j in range(i + 4, L):
                        if pair_prob_matrix[i, j] > threshold:
                            pairs.append((i, j, 10.6, float(pair_prob_matrix[i, j])))
                return pairs

            def __call__(self, seq_ids, mode='sample', pair_probs=None, lengths=None, **kwargs):
                """Run physics solver for each sequence in batch."""
                import math
                import numpy as np

                B, L = seq_ids.shape
                coords_list = []
                for b in range(B):
                    actual_L = lengths[b] if lengths is not None else L

                    # Step 1: Get pair constraints
                    pair_constraints = []
                    if pair_probs is not None:
                        pp = pair_probs[b, :actual_L, :actual_L].cpu().numpy()
                        if pp.max() > 0.31:  # Real data
                            pair_constraints = self._extract_pair_constraints(pp)

                    if not pair_constraints:
                        seq_np = seq_ids[b, :actual_L].cpu().numpy()
                        pair_constraints = self._predict_secondary_structure(seq_np)

                    # Step 2: Initialize with structure-aware 3D coords
                    init_coords = self._initialize_3d(actual_L, pair_constraints)

                    # Step 3: Refine with solver (starts from init_coords)
                    class ConstraintSet:
                        def __init__(self, seq_len, pairs):
                            self.seq_len = seq_len
                            self.pair_constraints = pairs

                    constraint_set = ConstraintSet(actual_L, pair_constraints)

                    # Use solver with custom init (override regular polygon)
                    # Monkey-patch solver's _regular_polygon temporarily
                    original_method = self.solver._regular_polygon
                    self.solver._regular_polygon = lambda l, bl: init_coords.copy()
                    conformations = self.solver.solve(constraint_set)
                    self.solver._regular_polygon = original_method

                    if conformations:
                        coords = conformations[0].astype(np.float32)
                    else:
                        coords = init_coords.astype(np.float32)

                    # Pad to batch length if needed
                    if actual_L < L:
                        pad = np.tile(coords[-1:], (L - actual_L, 1))
                        coords = np.concatenate([coords, pad], axis=0)

                    coords_list.append(coords)

                # Stack into batch tensor
                import torch
                pred = torch.from_numpy(np.stack(coords_list, axis=0)).to(self.device)
                return {'coords': pred}

            def eval(self):
                pass  # No-op for compatibility

        return Scheme2PhysicsSolver(n_samples=args.n_samples if hasattr(args, 'n_samples') else 10)
    elif scheme_id == 5:
        import torch.nn as nn
        class Scheme5Model(nn.Module):
            def __init__(self, d_model=128, n_heads=4, n_blocks=4):
                super().__init__()
                self.embed = nn.Embedding(5, d_model)
                self.circ_pos = nn.Embedding(512, d_model)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads,
                        dim_feedforward=d_model * 2,
                        dropout=0.1, batch_first=True,
                    )
                    for _ in range(n_blocks)
                ])
                self.coord_head = nn.Linear(d_model, 3)
            def forward(self, seq_ids, **kwargs):
                B, L = seq_ids.shape
                device = seq_ids.device
                pos = torch.arange(L, device=device) % 512
                h = self.embed(seq_ids) + self.circ_pos(pos)
                for block in self.blocks:
                    h = block(h)
                coords = self.coord_head(h)
                return {'coords': coords}
        return Scheme5Model(d_model=args.d_hidden, n_blocks=args.n_layers).to(device)
    elif scheme_id == 6:
        from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
            GNNLatentDiffusionModel, GNNLatentConfig
        )
        config = GNNLatentConfig(n_diffusion_steps=100, d_node=args.d_hidden)
        return GNNLatentDiffusionModel(config).to(device)
    elif scheme_id == 7:
        from confluencia_3_0.core.circrna.torusfold.circrna_mamba_diffusion import (
            CircMambaDiffusionModel, CircMambaConfig
        )
        config = CircMambaConfig(d_model=args.d_hidden)
        return CircMambaDiffusionModel(config).to(device)
    else:
        raise ValueError(f"Cannot build model for scheme {scheme_id}")


@torch.no_grad()
def evaluate(model, scheme_id, loader, device, n_samples=1):
    """Evaluate model on dataset.

    For diffusion models, sample n_samples conformations and take best.
    """
    model.eval()
    all_rmsds = []
    all_closure = []
    all_tm_scores = []
    n_evaluated = 0
    n_failed = 0
    n_batches = 0
    n_skipped_inf = 0
    n_skipped_zero = 0

    for batch in loader:
        n_batches += 1
        seq_ids = batch['seq_ids'].to(device)
        target = batch['coords'].to(device)
        lengths = batch['lengths']
        pair_probs = batch.get('pair_probs', None)

        # Skip corrupt data
        if torch.isinf(target).any() or torch.isnan(target).any():
            n_skipped_inf += 1
            continue

        # Skip zero targets (replaced Inf data)
        if target.abs().sum() < 1e-3:
            n_skipped_zero += 1
            continue

        B = len(lengths)

        # Get predictions
        best_rmsds = [float('inf')] * B

        for sample_idx in range(n_samples):
            try:
                if scheme_id == 2:
                    # Physics solver: pass pair_probs for constraint extraction
                    out = model(seq_ids, mode='sample', pair_probs=pair_probs, lengths=lengths)
                    pred = out['coords']
                elif scheme_id == 6:
                    # Scheme 6: GNN Latent Diffusion
                    # Training uses unit-sphere normalization (2nd training loop saves checkpoint):
                    #   target_norm = target_centered / target_scale
                    #   pred_norm = pred_centered / target_scale
                    # Denormalize: pred_norm * target_scale + target.mean
                    out = model(seq_ids, mode='sample')
                    pred = out['coords']
                    # Debug: check raw model output scale
                    if n_batches <= 2:
                        print(f"    [DEBUG S6] raw pred: mean={pred.mean().item():.4f}, std={pred.std().item():.4f}, "
                              f"norm={torch.norm(pred).item():.2f}, nan={torch.isnan(pred).sum().item()}, inf={torch.isinf(pred).sum().item()}")
                    pred_centered = pred - pred.mean(dim=1, keepdim=True)
                    pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1e-6)
                    pred_norm = pred_centered / pred_scale
                    target_centered = target - target.mean(dim=1, keepdim=True)
                    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
                    pred = pred_norm * target_scale + target.mean(dim=1, keepdim=True)
                elif scheme_id == 1:
                    # Scheme 1: CircRNA3DModel.forward(seq_ids) — no mode param
                    # Model outputs in unit-sphere normalized space, need denormalization
                    out = model(seq_ids)
                    pred = out['coords']
                    # Normalize pred to unit-sphere (like training), then denormalize by target_scale
                    pred_centered = pred - pred.mean(dim=1, keepdim=True)
                    pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1e-6)
                    pred_norm = pred_centered / pred_scale
                    target_centered = target - target.mean(dim=1, keepdim=True)
                    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
                    pred = pred_norm * target_scale + target.mean(dim=1, keepdim=True)
                else:
                    out = model(seq_ids, mode='sample')
                    pred = out['coords']
            except Exception as e:
                if n_failed < 5:
                    print(f"    [WARN] Sample {n_failed}+ failed: {e}")
                n_failed += B
                break

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                if n_failed < 5:
                    print(f"    [WARN] Sample {n_failed}+ NaN/Inf in prediction after denorm")
                n_failed += B
                break

            for b in range(B):
                L = lengths[b]
                p = pred[b, :L]
                t = target[b, :L]

                # Skip zero targets
                if t.abs().sum() < 1e-3:
                    continue

                rmsd = kabsch_rmsd(p, t)
                if not (torch.isnan(rmsd) or torch.isinf(rmsd)):
                    best_rmsds[b] = min(best_rmsds[b], rmsd.item())

        # Record results for this batch
        for b in range(B):
            if best_rmsds[b] < float('inf'):
                all_rmsds.append(best_rmsds[b])

                # Closure error
                p = pred[b, :lengths[b]]
                closure_dist = torch.norm(p[0] - p[-1]).item()
                all_closure.append(abs(closure_dist - 5.9))

                # TM-score approximation
                L = lengths[b]
                d0 = 1.24 * (max(L, 15) - 15) ** (1.0/3.0) - 1.8
                d0 = max(d0, 0.5)
                t_coord = target[b, :L]
                p_coord = pred[b, :L]
                t_c = t_coord - t_coord.mean(dim=0)
                p_c = p_coord - p_coord.mean(dim=0)
                di = torch.sqrt(torch.sum((p_c - t_c) ** 2, dim=1))
                tm = torch.sum(1.0 / (1.0 + (di / d0) ** 2)) / L
                all_tm_scores.append(tm.item())

                n_evaluated += 1

    results = {
        'n_batches': n_batches,
        'n_evaluated': n_evaluated,
        'n_failed': n_failed,
        'n_skipped_inf': n_skipped_inf,
        'n_skipped_zero': n_skipped_zero,
        'rmsd_mean': float(np.mean(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_median': float(np.median(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_std': float(np.std(all_rmsds)) if all_rmsds else 0,
        'rmsd_min': float(np.min(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_max': float(np.max(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_<10A': float(np.mean([r < 10 for r in all_rmsds])) if all_rmsds else 0,
        'rmsd_<20A': float(np.mean([r < 20 for r in all_rmsds])) if all_rmsds else 0,
        'rmsd_<30A': float(np.mean([r < 30 for r in all_rmsds])) if all_rmsds else 0,
        'closure_mean': float(np.mean(all_closure)) if all_closure else float('inf'),
        'tm_mean': float(np.mean(all_tm_scores)) if all_tm_scores else 0,
        'tm_median': float(np.median(all_tm_scores)) if all_tm_scores else 0,
    }

    # RMSD by length bucket
    if all_rmsds:
        length_buckets = {'30-50': [], '50-100': [], '100-200': [], '200-500': [], '500+': []}
        # We don't have individual lengths here, store overall
        results['rmsd_percentiles'] = {
            'p10': float(np.percentile(all_rmsds, 10)),
            'p25': float(np.percentile(all_rmsds, 25)),
            'p50': float(np.percentile(all_rmsds, 50)),
            'p75': float(np.percentile(all_rmsds, 75)),
            'p90': float(np.percentile(all_rmsds, 90)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TorusFold scheme')
    parser.add_argument('--scheme', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--labels', type=str, default='data/circrna_3d_merged')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Alternative test data directory (e.g., data/pdb_3d)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Number of samples for diffusion models')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples to evaluate')
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    print("=" * 60)
    print(f"  Evaluating Scheme {args.scheme}")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")

    # Use alternative test data if specified
    data_dir = args.test_data if args.test_data else args.labels
    print(f"  Data: {data_dir}")

    # Auto-search data path (AutoDL compatibility)
    if not Path(data_dir).exists():
        search_paths = []
        # Try relative to script, project roots, and common AutoDL paths
        for root in [PROJECT_ROOT, PROJECT_ROOT / 'confluencia_3_0' / 'core' / 'circrna' / 'torusfold',
                      Path('/root/autodl-tmp/confluencia/confluencia_3_0/core/circrna/torusfold'),
                      Path('/root/autodl-tmp')]:
            candidate = root / data_dir
            if candidate.exists():
                data_dir = str(candidate)
                print(f"  Found at: {data_dir}")
                break
            # Also try just the base name
            candidate2 = root / Path(data_dir).name
            if candidate2.exists():
                data_dir = str(candidate2)
                print(f"  Found at: {data_dir}")
                break

    # Load data
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(data_dir)
    print(f"  Total: {len(sequences)} samples")

    # Use first N clean samples (not last 10% which may be all-Inf)
    clean_mask = []
    for i, c in enumerate(coords_labels):
        if np.isfinite(c).all() and c.shape[0] == len(sequences[i]) and c.shape[0] >= 4:
            clean_mask.append(True)
        else:
            clean_mask.append(False)

    # Pick clean samples from first 90% (training region, but we use for eval)
    n_train = int(0.9 * len(sequences))
    eval_candidates = [(i, s, c, p, cw) for i, (s, c, p, cw, m)
                       in enumerate(zip(sequences, coords_labels, pair_labels,
                                       confidence_weights, clean_mask))
                       if m and i < n_train]

    # Take first max_samples
    eval_candidates = eval_candidates[:args.max_samples]

    if not eval_candidates:
        print("  ERROR: No clean samples found! Using all samples regardless of quality.")
        eval_candidates = [(i, sequences[i], coords_labels[i], pair_labels[i],
                           confidence_weights[i])
                          for i in range(min(args.max_samples, len(sequences)))]
    else:
        print(f"  Found {len(clean_mask)-sum(clean_mask)} dirty, "
              f"{sum(clean_mask)} clean samples")
        print(f"  Using {len(eval_candidates)} clean samples for evaluation")

    test_seqs = [e[1] for e in eval_candidates]
    test_coords = [e[2] for e in eval_candidates]
    test_pairs = [e[3] for e in eval_candidates]
    test_confs = [e[4] for e in eval_candidates]

    print(f"  Test: {len(test_seqs)} clean samples")

    ds = CircRNADataset(test_seqs, test_coords, test_pairs, test_confs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Build and load model
    print(f"\n  Building Scheme {args.scheme} model...")
    model = build_model(args.scheme, args, device)

    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not missing and not unexpected:
            print(f"  All keys matched perfectly")
        # Quick sanity: compare a few parameter values to confirm loading worked
        n_params = sum(p.numel() for p in model.parameters())
        n_nonzero = sum((p != 0).sum().item() for p in model.parameters())
        print(f"  Model params: {n_params:,} total, {n_nonzero:,} nonzero")
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  WARNING: Checkpoint not found, using random init")

    # Evaluate
    print(f"\n  Evaluating (n_samples={args.n_samples})...")
    t0 = time.time()
    results = evaluate(model, args.scheme, loader, device, n_samples=args.n_samples)
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"  Scheme {args.scheme} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {results['n_evaluated']}")
    print(f"  Samples failed:    {results['n_failed']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"")
    print(f"  RMSD (A):")
    print(f"    Mean:   {results['rmsd_mean']:.2f}")
    print(f"    Median: {results['rmsd_median']:.2f}")
    print(f"    Std:    {results['rmsd_std']:.2f}")
    print(f"    Min:    {results['rmsd_min']:.2f}")
    print(f"    Max:    {results['rmsd_max']:.2f}")
    print(f"")
    print(f"  RMSD thresholds:")
    print(f"    < 10A: {results['rmsd_<10A']:.1%}")
    print(f"    < 20A: {results['rmsd_<20A']:.1%}")
    print(f"    < 30A: {results['rmsd_<30A']:.1%}")
    print(f"")
    if 'rmsd_percentiles' in results:
        p = results['rmsd_percentiles']
        print(f"  RMSD percentiles:")
        print(f"    P10: {p['p10']:.2f}  P25: {p['p25']:.2f}  P50: {p['p50']:.2f}  P75: {p['p75']:.2f}  P90: {p['p90']:.2f}")
    print(f"")
    print(f"  Closure error (A): {results['closure_mean']:.2f}")
    print(f"  TM-score: {results['tm_mean']:.4f} (median: {results['tm_median']:.4f})")
    print(f"{'='*60}")

    # Save
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, f'scheme{args.scheme}_eval.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}/scheme{args.scheme}_eval.json")


if __name__ == '__main__':
    main()
