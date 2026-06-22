#!/usr/bin/env python3
"""debug_first_batch.py — Check first batch for NaN causes."""

import os
import sys
import json
import torch
import numpy as np

_here = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here)))))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn, CircRNA3DModel
)


def main():
    labels_dir = sys.argv[1] if len(sys.argv) > 1 else "data/circrna_3d_merged"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {labels_dir}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Confidence weights: min={min(confidence_weights)}, max={max(confidence_weights)}, mean={np.mean(confidence_weights):.3f}")
    print(f"Any NaN confidence: {any(np.isnan(c) for c in confidence_weights)}")

    # Create dataset
    ds = CircRNADataset(sequences[:10], coords_labels[:10], pair_labels[:10], confidence_weights[:10])

    # Get first item
    item = ds[0]
    print(f"\nFirst item:")
    print(f"  seq_ids shape: {item['seq_ids'].shape}")
    print(f"  coords shape: {item['coords'].shape}")
    print(f"  length: {item['length']}")
    print(f"  confidence: {item['confidence']}")
    print(f"  has NaN in seq_ids: {torch.isnan(item['seq_ids']).any()}")
    print(f"  has NaN in coords: {torch.isnan(item['coords']).any()}")

    # Collate
    batch = collate_fn([ds[i] for i in range(4)])
    print(f"\nFirst batch:")
    print(f"  seq_ids shape: {batch['seq_ids'].shape}")
    print(f"  coords shape: {batch['coords'].shape}")
    print(f"  lengths: {batch['lengths']}")
    print(f"  confidence: {batch['confidence']}")
    print(f"  has NaN in seq_ids: {torch.isnan(batch['seq_ids']).any()}")
    print(f"  has NaN in coords: {torch.isnan(batch['coords']).any()}")
    print(f"  coords range: [{batch['coords'].min():.2f}, {batch['coords'].max():.2f}]")

    # Model forward
    model = CircRNA3DModel(d_hidden=128, n_layers=4).to(device)
    seq_ids = batch['seq_ids'].to(device)
    target = batch['coords'].to(device)

    print(f"\nModel forward...")
    with torch.no_grad():
        out = model(seq_ids)
        pred = out['coords']
        print(f"  pred shape: {pred.shape}")
        print(f"  pred has NaN: {torch.isnan(pred).any()}")
        print(f"  pred range: [{pred.min():.2f}, {pred.max():.2f}]")
        print(f"  bond_loss: {out['bond_loss']:.4f}")
        print(f"  closure_loss: {out['closure_loss']:.4f}")

    # Compute training loss
    B, L, _ = target.shape
    lengths = batch['lengths']

    target_centered = target - target.mean(dim=1, keepdim=True)
    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
    target_norm = target_centered / target_scale

    print(f"\nNormalization:")
    print(f"  target_scale: {target_scale.squeeze()}")
    print(f"  target_norm has NaN: {torch.isnan(target_norm).any()}")

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    pred_norm = pred_centered / target_scale

    print(f"  pred_norm has NaN: {torch.isnan(pred_norm).any()}")

    loss = 0
    for b in range(B):
        valid_L = lengths[b]
        diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
        batch_loss = torch.mean(diff ** 2)
        print(f"  batch {b} loss (valid_L={valid_L}): {batch_loss:.4f}, NaN={torch.isnan(batch_loss).any()}")
        loss += batch_loss
    loss /= B

    conf_scale = batch['confidence'].mean().item()
    final_loss = loss * conf_scale * 2.0

    print(f"\nFinal loss: {final_loss:.4f}, NaN={np.isnan(final_loss)}")


if __name__ == "__main__":
    main()