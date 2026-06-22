#!/usr/bin/env python3
"""debug_scheme4_nan.py — Scan many batches to find which ones cause NaN."""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    labels_dir = sys.argv[1] if len(sys.argv) > 1 else "data/circrna_3d_merged"
    print(f"Loading from {labels_dir}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir, max_len=500)
    print(f"Loaded {len(sequences)} sequences (max_len=500 filtered)")

    # Split same as training
    split = int(0.9 * len(sequences))
    train_ds = CircRNADataset(sequences[:split], coords_labels[:split], pair_labels[:split], confidence_weights[:split])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
        CircRNADiffusionModel, CircDiffusionConfig
    )

    config = CircDiffusionConfig(n_diffusion_steps=50, d_node=128, d_edge=64)
    model = CircRNADiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"\nScanning batches for NaN (max 100 batches)...\n")

    n_ok = 0
    n_nan_loss = 0
    n_nan_grad = 0
    nan_examples = []  # Store examples of NaN batches

    for i, batch in enumerate(train_loader):
        if i >= 100:
            break

        seq_ids = batch['seq_ids'].to(device)
        coords_target = batch['coords'].to(device)
        pair_probs = batch.get('pair_probs', None)
        if pair_probs is not None:
            pair_probs = pair_probs.to(device)
        lengths = batch['lengths']

        B, L, _ = coords_target.shape
        coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
        coords_scale = torch.norm(coords_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
        coords_norm = coords_centered / coords_scale

        optimizer.zero_grad()

        try:
            with torch.cuda.amp.autocast():
                out = model(seq_tokens=seq_ids, coords_target=coords_norm, pair_probs=pair_probs)
                noise_loss = out.get('noise_loss', torch.tensor(0.0, device=device))
                closure_loss = out.get('closure_loss', torch.tensor(0.0, device=device))
                loss = out.get('total_loss', noise_loss + 0.1 * closure_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                n_nan_loss += 1
                nan_examples.append(('nan_loss', i, L, lengths, loss.item() if not torch.isnan(loss) else 'NaN'))
                if len(nan_examples) <= 5:
                    print(f"  Batch {i}: L={L} lengths={lengths} -> NaN/Inf loss (noise={noise_loss.item():.4f}, closure={closure_loss.item():.4f})")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check gradients
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                n_nan_grad += 1
                nan_examples.append(('nan_grad', i, L, lengths, loss.item()))
                if len(nan_examples) <= 5:
                    print(f"  Batch {i}: L={L} lengths={lengths} -> NaN gradients (loss={loss.item():.4f})")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            n_ok += 1

            if (i + 1) % 20 == 0:
                print(f"  Checked {i+1} batches: OK={n_ok}, NaN_loss={n_nan_loss}, NaN_grad={n_nan_grad}")

        except RuntimeError as e:
            print(f"  Batch {i}: L={L} lengths={lengths} -> RuntimeError: {e}")
            nan_examples.append(('error', i, L, lengths, str(e)[:80]))
            optimizer.zero_grad()
            continue

    print(f"\n{'='*50}")
    print(f"Results from {min(i+1, 100)} batches:")
    print(f"  OK: {n_ok}")
    print(f"  NaN loss: {n_nan_loss}")
    print(f"  NaN gradients: {n_nan_grad}")
    print(f"  Errors: {len(nan_examples) - n_nan_loss - n_nan_grad}")

    if nan_examples:
        print(f"\n  NaN batch details (first 5):")
        for ex in nan_examples[:5]:
            print(f"    {ex}")

    # If all OK with AMP + GradScaler, try WITHOUT AMP
    if n_nan_loss == 0 and n_nan_grad == 0:
        print(f"\n  All batches OK with AMP. Training should work now.")
    elif n_nan_loss > 0 or n_nan_grad > 0:
        print(f"\n  Still have NaN batches. Need further investigation.")


if __name__ == "__main__":
    main()
