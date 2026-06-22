#!/usr/bin/env python3
"""Diagnose Scheme 6 NaN issue with REAL training data."""

import torch
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
    GNNLatentDiffusionModel, GNNLatentConfig
)
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load real data
import argparse
args = argparse.Namespace(
    d_hidden=128, n_layers=4, diffusion_steps=100,
    lr=1e-3, epochs=50, batch_size=4, output='models/torusfold_s6',
    seed=42,
)

print("\n=== Loading data ===")
labels_dir = 'data/circrna_3d_merged'
try:
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir)
    print(f"  Loaded {len(sequences)} samples")

    # Check for data issues
    nan_coords = 0
    inf_coords = 0
    mismatch = 0
    for i in range(len(sequences)):
        c = coords_labels[i]
        if np.isnan(c).any():
            nan_coords += 1
        if np.isinf(c).any():
            inf_coords += 1
        if c.shape[0] != len(sequences[i]):
            mismatch += 1
    print(f"  NaN coords: {nan_coords}, Inf coords: {inf_coords}, Length mismatch: {mismatch}")

    # Build dataset and test collation
    print("\n=== Testing collate_fn ===")
    ds = CircRNADataset(sequences[:20], coords_labels[:20], pair_labels[:20], confidence_weights[:20] if confidence_weights else None)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    for i, batch in enumerate(loader):
        seq_ids = batch['seq_ids']
        coords = batch['coords']
        lengths = batch['lengths']
        print(f"  Batch {i}: seq_ids={seq_ids.shape}, coords={coords.shape}, "
              f"NaN={torch.isnan(coords).any().item()}, "
              f"Inf={torch.isinf(coords).any().item()}, "
              f"Range=[{coords.min().item():.2f}, {coords.max().item():.2f}]")

        # Check individual samples
        for b in range(len(lengths)):
            L = lengths[b]
            c = coords[b, :L]
            if torch.isnan(c).any() or torch.isinf(c).any():
                print(f"    Sample {b} (L={L}): HAS NaN/Inf!")

except Exception as e:
    print(f"  Data loading failed: {e}")
    print("  Using synthetic data instead")
    sequences = None

# Test model with real batch
if sequences:
    print("\n=== Testing model with real data ===")
    config = GNNLatentConfig(n_diffusion_steps=10, d_node=64)
    model = GNNLatentDiffusionModel(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_nan = 0
    n_ok = 0
    for i, batch in enumerate(loader):
        if i >= 5:
            break

        seq_ids = batch['seq_ids'].to(device)
        target = batch['coords'].to(device)
        lengths = batch['lengths']

        # Normalize
        B, L, _ = target.shape
        target_centered = target - target.mean(dim=1, keepdim=True)
        target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
        target_norm = target_centered / target_scale

        # Check for zero scales (all same coords = padding artifact)
        for b in range(B):
            if target_scale[b].item() < 1e-6:
                print(f"  WARNING: Batch {i}, sample {b} has near-zero scale (all coords same)")

        out = model(seq_ids, mode='train')
        pred_coords = out['coords']
        diff_loss = out.get('diffusion_loss', None)

        pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
        pred_norm = pred_centered / target_scale

        coord_loss = 0
        for b in range(B):
            valid_L = lengths[b]
            diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
            coord_loss += torch.mean(diff ** 2)
        coord_loss /= B

        if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
            total = diff_loss + 0.1 * coord_loss
        else:
            total = coord_loss

        is_nan = torch.isnan(total).any().item()
        if is_nan:
            n_nan += 1
            print(f"  Batch {i}: LOSS=NaN! diff_loss={diff_loss.item() if diff_loss is not None else 'None'}, "
                  f"coord_loss={coord_loss.item():.6f}")
            # Check intermediate values
            print(f"    target NaN: {torch.isnan(target).any().item()}, range: [{target.min().item():.2f}, {target.max().item():.2f}]")
            print(f"    target_scale: {target_scale.squeeze().tolist()}")
            print(f"    pred NaN: {torch.isnan(pred_coords).any().item()}")
        else:
            n_ok += 1
            print(f"  Batch {i}: OK loss={total.item():.6f}")

        if not is_nan:
            total.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(f"\n  Results: {n_ok} OK, {n_nan} NaN out of {n_ok+n_nan} batches")

print("\n=== DONE ===")
