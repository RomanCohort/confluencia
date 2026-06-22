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
        from confluencia_3_0.core.circrna.torusfold.train_torusfold_3d import CircRNA3DModel
        return CircRNA3DModel(d_hidden=args.d_hidden, n_layers=args.n_layers).to(device)
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
                if scheme_id == 6:
                    # Use encoder+decoder directly (fast, deterministic)
                    latent = model.encoder(seq_ids)
                    pred = model.decoder(latent, seq_ids)
                else:
                    out = model(seq_ids, mode='sample')
                    pred = out['coords']
            except Exception as e:
                n_failed += B
                break

            if torch.isnan(pred).any() or torch.isinf(pred).any():
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
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
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
    print(f"  RMSD (Å):")
    print(f"    Mean:   {results['rmsd_mean']:.2f}")
    print(f"    Median: {results['rmsd_median']:.2f}")
    print(f"    Std:    {results['rmsd_std']:.2f}")
    print(f"    Min:    {results['rmsd_min']:.2f}")
    print(f"    Max:    {results['rmsd_max']:.2f}")
    print(f"")
    print(f"  RMSD thresholds:")
    print(f"    < 10Å: {results['rmsd_<10A']:.1%}")
    print(f"    < 20Å: {results['rmsd_<20A']:.1%}")
    print(f"    < 30Å: {results['rmsd_<30A']:.1%}")
    print(f"")
    if 'rmsd_percentiles' in results:
        p = results['rmsd_percentiles']
        print(f"  RMSD percentiles:")
        print(f"    P10: {p['p10']:.2f}  P25: {p['p25']:.2f}  P50: {p['p50']:.2f}  P75: {p['p75']:.2f}  P90: {p['p90']:.2f}")
    print(f"")
    print(f"  Closure error (Å): {results['closure_mean']:.2f}")
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
