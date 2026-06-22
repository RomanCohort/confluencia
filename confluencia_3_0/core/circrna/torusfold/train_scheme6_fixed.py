#!/usr/bin/env python3
"""
train_scheme6_fixed.py — Fixed Scheme 6 training with physics-aware losses.

Key fixes:
1. Multiple loss components: diffusion + coord + closure + bond
2. CosineAnnealing scheduler (no val dependency)
3. Encoder+Decoder validation (fast, no sampling)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def kabsch_rmsd(p, t):
    """Kabsch-aligned RMSD."""
    p_c = p - p.mean(dim=0)
    t_c = t - t.mean(dim=0)
    H = t_c.T @ p_c
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1, 1, d], device=p.device, dtype=torch.float32))
        R = Vt.T @ D @ U.T
        p_aligned = (R @ p_c.T).T
        return torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
    except:
        return torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))


def train_scheme6_fixed(labels_dir, output_dir, epochs=100, batch_size=4, lr=5e-5, device='cuda'):
    print("=" * 60)
    print("  Scheme 6 Fixed: GNN Latent Diffusion")
    print("=" * 60)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  LR: {lr}, Epochs: {epochs}")
    print(f"  Loss weights: diff*1, coord*10, closure*5, bond*2")

    # Load data
    sequences, coords, pairs, confs, meta = load_pseudo_labels(labels_dir)
    print(f"  Loaded {len(sequences)} samples")

    # Split
    n = len(sequences)
    split = int(0.9 * n)
    train_ds = CircRNADataset(sequences[:split], coords[:split], pairs[:split], confs[:split])
    val_ds = CircRNADataset(sequences[split:], coords[split:], pairs[split:], confs[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    config = GNNLatentConfig(n_diffusion_steps=100, d_node=128)
    model = GNNLatentDiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val = float('inf')
    bond_length = 5.9

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        metrics = {'diff': 0, 'coord': 0, 'closure': 0, 'bond': 0}
        nan_batches = 0

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']

            # Skip corrupt
            if torch.isinf(target).any() or torch.isnan(target).any():
                nan_batches += 1
                optimizer.zero_grad()
                continue

            B, L, _ = target.shape

            # Normalize: center per sample, scale per sample (more stable)
            target_centered = target - target.mean(dim=1, keepdim=True)
            # Use per-sample norm, not global std
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            # Forward
            out = model(seq_ids, mode='train')
            pred_coords = out['coords']
            diff_loss = out.get('diffusion_loss', None)

            # Denormalize for physics losses (use target scale)
            pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
            pred_denorm = pred_centered * target_scale + target.mean(dim=1, keepdim=True)

            # Losses
            pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
            coord_loss = F.mse_loss(pred_centered, target_norm)

            pred_closure = torch.norm(pred_denorm[:, 0] - pred_denorm[:, -1], dim=-1)
            closure_loss = F.mse_loss(pred_closure, torch.full_like(pred_closure, bond_length))

            bond_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                valid_L = lengths[b]
                if valid_L < 4:
                    continue
                bonds = torch.norm(pred_denorm[b, 1:valid_L] - pred_denorm[b, :valid_L-1], dim=-1)
                bsj = torch.norm(pred_denorm[b, 0] - pred_denorm[b, valid_L-1])
                all_bonds = torch.cat([bonds, bsj.unsqueeze(0)])
                bond_loss += F.mse_loss(all_bonds, torch.full_like(all_bonds, bond_length))
            bond_loss /= max(B, 1)

            # Total
            if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
                loss = diff_loss * 1.0 + coord_loss * 10.0 + closure_loss * 5.0 + bond_loss * 2.0
            else:
                loss = coord_loss * 10.0 + closure_loss * 5.0 + bond_loss * 2.0

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            metrics['diff'] += diff_loss.item() if diff_loss is not None and torch.isfinite(diff_loss) else 0
            metrics['coord'] += coord_loss.item()
            metrics['closure'] += closure_loss.item()
            metrics['bond'] += bond_loss.item()

        scheduler.step()

        n_valid_batches = max(len(train_loader) - nan_batches, 1)
        avg_train = train_loss / n_valid_batches
        avg_diff = metrics['diff'] / n_valid_batches
        avg_coord = metrics['coord'] / n_valid_batches
        avg_closure = metrics['closure'] / n_valid_batches
        avg_bond = metrics['bond'] / n_valid_batches

        # Validation: encoder+decoder RMSD
        model.eval()
        val_rmsd = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                if target.abs().sum() < 1e-3:
                    continue

                latent = model.encoder(seq_ids)
                pred = model.decoder(latent, seq_ids)

                for b in range(len(lengths)):
                    L = lengths[b]
                    p = pred[b, :L]
                    t = target[b, :L]
                    if torch.isnan(p).any() or t.abs().sum() < 1e-3:
                        continue

                    rmsd = kabsch_rmsd(p, t)
                    if torch.isfinite(rmsd):
                        val_rmsd += rmsd.item()
                        n_val += 1

        avg_val = val_rmsd / max(n_val, 1) if n_val > 0 else avg_train * 100

        if avg_val < best_val:
            best_val = avg_val
            import os
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/scheme6_best.pt")

        print(f"  Epoch {epoch+1}/{epochs} train={avg_train:.4f} "
              f"(diff={avg_diff:.3f} coord={avg_coord:.3f} cls={avg_closure:.2f} bond={avg_bond:.2f}) "
              f"val={avg_val:.1f}Å (n={n_val}) nan={nan_batches}")

    print(f"\n  Best val RMSD: {best_val:.2f}Å")
    return best_val


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='data/circrna_3d_merged')
    parser.add_argument('--output', default='models/torusfold_s6_fixed')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    train_scheme6_fixed(
        args.labels, args.output, args.epochs, args.batch_size, args.lr, args.device
    )