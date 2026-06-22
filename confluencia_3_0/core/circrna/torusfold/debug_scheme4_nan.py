#!/usr/bin/env python3
"""debug_scheme4_nan.py — Step-by-step NaN diagnosis for Scheme 4 with AMP."""

import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn
)


def check_tensor(name, t, track_nan=False):
    """Check a tensor for NaN/Inf and print stats."""
    if t is None:
        print(f"  {name}: None")
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    dtype = t.dtype
    shape = t.shape
    if has_nan or has_inf:
        print(f"  {name}: shape={shape} dtype={dtype} *** NaN={has_nan} Inf={has_inf} ***")
        return True
    if t.numel() > 0 and t.is_floating_point():
        print(f"  {name}: shape={shape} dtype={dtype} range=[{t.min().item():.4f}, {t.max().item():.4f}] mean={t.mean().item():.4f}")
    else:
        print(f"  {name}: shape={shape} dtype={dtype}")
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    labels_dir = sys.argv[1] if len(sys.argv) > 1 else "data/circrna_3d_merged"
    print(f"Loading from {labels_dir}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir, max_len=500)
    print(f"Loaded {len(sequences)} sequences (max_len=500 filtered)")

    # Create dataset and batch
    ds = CircRNADataset(sequences[:4], coords_labels[:4], pair_labels[:4], confidence_weights[:4])
    batch = collate_fn([ds[i] for i in range(4)])

    seq_ids = batch['seq_ids'].to(device)
    coords_target = batch['coords'].to(device)
    pair_probs = batch.get('pair_probs', None)
    if pair_probs is not None:
        pair_probs = pair_probs.to(device)

    print(f"\n=== Input Data ===")
    check_tensor("seq_ids", seq_ids)
    check_tensor("coords_target", coords_target)
    print(f"  lengths: {batch['lengths']}")

    # Normalize
    B, L, _ = coords_target.shape
    coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
    coords_scale = torch.norm(coords_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
    coords_norm = coords_centered / coords_scale

    print(f"\n=== Normalized Target (L={L}) ===")
    check_tensor("coords_norm", coords_norm)

    from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
        CircRNADiffusionModel, CircDiffusionConfig
    )

    config = CircDiffusionConfig(
        n_diffusion_steps=50,
        d_node=128,
        d_edge=64,
    )
    model = CircRNADiffusionModel(config).to(device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} params")

    # Test with AMP enabled
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n=== Testing Forward + Backward with AMP ===")

    with torch.cuda.amp.autocast():
        # Forward
        t = torch.randint(0, config.n_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(coords_norm)
        alpha_bar = model.alpha_bars[t].view(B, 1, 1)
        coords_noisy = torch.sqrt(alpha_bar) * coords_norm + torch.sqrt(1 - alpha_bar) * noise

        cond = model.condition_encoder(seq_ids, None, 310.0, 7.4, 1.0, 150.0)
        t_emb = model.time_embed(t.float())

        # Denoise
        noise_pred = model._denoise(coords_noisy, cond, t_emb, L, pair_probs)

        check_tensor("noise_pred", noise_pred)

        noise_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        closure_dist = torch.norm(noise_pred[:, 0] - noise_pred[:, -1], dim=-1)
        closure_loss = ((closure_dist - config.bond_length) ** 2).mean()
        loss = noise_loss + 0.1 * closure_loss

        check_tensor("loss", loss)
        print(f"  loss value: {loss.item():.6f}")

    # Backward with scaler
    print(f"\n=== Testing Backward with GradScaler ===")
    scaler.scale(loss).backward()

    # Check gradients
    print(f"\n=== Checking Gradients ===")
    nan_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"  *** NaN gradient in {name} ***")
                nan_grads += 1
    print(f"  Total params with NaN gradients: {nan_grads}")

    if nan_grads == 0:
        print(f"\n=== SUCCESS: No NaN in forward or backward ===")
    else:
        print(f"\n=== FAILURE: NaN gradients detected ===")


if __name__ == "__main__":
    main()
