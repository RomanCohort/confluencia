#!/usr/bin/env python3
"""Scan ALL data for NaN triggers in Scheme 6."""

import torch
import sys
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
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load ALL data
labels_dir = 'data/circrna_3d_merged'
sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir)
print(f"Loaded {len(sequences)} samples")

# Filter for Scheme 6 (max_len=800)
keep = [i for i, m in enumerate(metadata) if m['length'] <= 800]
print(f"Filtered to {len(keep)} (max_len=800)")

sequences = [sequences[i] for i in keep]
coords_labels = [coords_labels[i] for i in keep]
pair_labels = [pair_labels[i] for i in keep]
confidence_weights = [confidence_weights[i] for i in keep]

# Check data
print("\n=== Scanning all data ===")
bad_indices = []
for i in range(len(sequences)):
    c = coords_labels[i]
    s = sequences[i]
    if c.shape[0] != len(s):
        bad_indices.append((i, f"mismatch: coords={c.shape[0]}, seq={len(s)}"))
    if c.shape[0] < 4:
        bad_indices.append((i, f"too short: {c.shape[0]}"))
    if c.shape[0] > 800:
        bad_indices.append((i, f"too long: {c.shape[0]}"))
print(f"  Bad data: {len(bad_indices)}")
for idx, reason in bad_indices[:10]:
    print(f"    [{idx}] {reason}")

# Build full dataset
ds = CircRNADataset(sequences, coords_labels, pair_labels, confidence_weights)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Test model
config = GNNLatentConfig(n_diffusion_steps=10, d_node=64)
model = GNNLatentDiffusionModel(config).to(device)
model.train()

print(f"\n=== Testing {len(loader)} batches ===")
n_nan = 0
n_ok = 0
first_nan_batch = None

for i, batch in enumerate(loader):
    seq_ids = batch['seq_ids'].to(device)
    target = batch['coords'].to(device)
    lengths = batch['lengths']

    B, L, _ = target.shape
    target_centered = target - target.mean(dim=1, keepdim=True)
    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
    target_norm = target_centered / target_scale

    out = model(seq_ids, mode='train')
    pred_coords = out['coords']
    diff_loss = out.get('diffusion_loss', None)

    pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
    pred_norm = pred_centered / target_scale

    coord_loss = 0
    n_valid = 0
    for b in range(B):
        valid_L = lengths[b]
        if valid_L < 4:
            continue
        diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
        coord_loss += torch.mean(diff ** 2)
        n_valid += 1
    coord_loss = coord_loss / max(n_valid, 1)

    if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
        loss = diff_loss + 0.1 * coord_loss
    else:
        loss = coord_loss

    if torch.isnan(loss) or torch.isinf(loss):
        n_nan += 1
        if first_nan_batch is None:
            first_nan_batch = i
            # Detailed debug
            print(f"\n  FIRST NaN at batch {i}:")
            print(f"    seq_ids shape: {seq_ids.shape}")
            print(f"    lengths: {lengths}")
            print(f"    target range: [{target.min().item():.2f}, {target.max().item():.2f}]")
            print(f"    target_scale: {target_scale.squeeze().tolist()}")
            print(f"    pred range: [{pred_coords.min().item():.2f}, {pred_coords.max().item():.2f}]")
            print(f"    diff_loss: {diff_loss.item() if diff_loss is not None else 'None'}")
            print(f"    coord_loss: {coord_loss.item():.6f}")
            # Check individual samples
            for b in range(B):
                Lb = lengths[b]
                c = target[b, :Lb]
                p = pred_coords[b, :Lb]
                print(f"    Sample {b}: L={Lb}, target NaN={torch.isnan(c).any().item()}, "
                      f"pred NaN={torch.isnan(p).any().item()}, "
                      f"target_range=[{c.min().item():.2f}, {c.max().item():.2f}]")
    else:
        n_ok += 1

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(loader)}: {n_ok} OK, {n_nan} NaN")

print(f"\n  Final: {n_ok} OK, {n_nan} NaN out of {n_ok+n_nan} batches")
print(f"  First NaN batch: {first_nan_batch}")
print("=== DONE ===")
