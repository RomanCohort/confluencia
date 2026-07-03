#!/usr/bin/env python3
"""
train_scheme6_pipeline.py — Scheme 6 training on pipeline-exported data
with three-layer sample weighting (A=1.0, B=0.3) per training_strategy_v2.

Differences from train_scheme6_fixed.py:
  1. Data source: pipeline export (coords.npy/confidences.npy/sample_weights.npy/
     sources.npy) via load_pipeline_dataset.build_scheme_c_loaders()
  2. Scheme C split: real -> 85% train / 5% val / 10% test-B; synthetic -> train only
  3. Sample-weighted coord loss: per-sample weight = sample_weight * confidence
     (A-layer=1.0, B-layer=0.3) — soft noise contributes less to the gradient
  4. LR=1e-4 with 5-epoch warmup (per feedback_torusfold_scheme1_lr)
  5. Test-B held-out evaluation (statistical), in addition to kabsch val RMSD

Loss components (same as train_scheme6_fixed.py):
  diff*1 + coord*10 + closure*5 + bond*2
  where coord_loss is per-sample weighted.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
    GNNLatentDiffusionModel, GNNLatentConfig
)
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    CircRNADataset, collate_fn,
)
from confluencia_3_0.core.circrna.torusfold.load_pipeline_dataset import (
    build_scheme_c_loaders,
)
from confluencia_3_0.core.circrna.torusfold.train_scheme6_fixed import kabsch_rmsd


class EMAWrapper:
    """Exponential moving average of model parameters.

    Maintains a shadow copy updated each step as:
        ema_p = decay * ema_p + (1 - decay) * p
    Use the EMA copy for evaluation (often more stable than raw weights).
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Load shadow params into model (for eval). Returns backup to restore."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            p.data.copy_(backup[n])


def schedule_b_weight(epoch, warmup_epochs=5, full_at_epoch=20,
                      target_b_weight=0.3, enabled=True):
    """Curriculum schedule for B-layer (soft noise) weight.

    Phase 1 (epoch < warmup):  pure A-layer, B weight = 0
    Phase 2 (warmup..full):    B weight linear 0 -> target
    Phase 3 (full+):           B weight = target (static)

    Returns 0.0 if not enabled (static mode uses loader's fixed 0.3).
    """
    if not enabled:
        return target_b_weight
    if epoch < warmup_epochs:
        return 0.0
    if epoch >= full_at_epoch:
        return target_b_weight
    progress = (epoch - warmup_epochs) / max(full_at_epoch - warmup_epochs, 1)
    return target_b_weight * progress


def weighted_coord_loss(pred_centered, target_norm, sample_weights):
    """Per-sample weighted MSE in normalized space."""
    per_sample = ((pred_centered - target_norm) ** 2).mean(dim=(1, 2))  # (B,)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)


def train_scheme6_pipeline(export_dir, output_dir, epochs=100, batch_size=4,
                            lr=1e-4, device='cuda', val_ratio=0.05,
                            test_b_ratio=0.10, max_len=None,
                            include_soft_noise=True, seed=42,
                            curriculum_b=False, full_at_epoch=20,
                            use_ema=False, ema_decay=0.999,
                            d_node=128, d_edge=64, d_latent=256,
                            n_encoder_layers=6, n_decoder_layers=6,
                            n_diffusion_steps=100, n_heads=8,
                            self_distill=False, distill_weight=2.0, distill_start_epoch=10):
    print("=" * 60)
    print("  Scheme 6 Pipeline: GNN Latent Diffusion (three-layer weighted)")
    print("=" * 60)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  LR: {lr} (5-epoch warmup), Epochs: {epochs}")
    print(f"  Loss: diff*1 + coord*10 (weighted) + closure*5 + bond*2")
    print(f"  Split: real {1-val_ratio-test_b_ratio:.0%}/{val_ratio:.0%}/{test_b_ratio:.0%} "
          f"(train/val/test-B), synthetic -> train")
    if curriculum_b:
        print(f"  Curriculum: B-layer 0 -> 0.3 over epoch 5..{full_at_epoch}")
    else:
        print(f"  B-layer: static 0.3 (no curriculum)")
    if use_ema:
        print(f"  EMA: enabled (decay={ema_decay})")
    if self_distill:
        if not use_ema:
            print("  WARNING: self-distill requires EMA teacher; auto-enabling EMA")
        print(f"  Self-distill: enabled (weight={distill_weight}, start epoch={distill_start_epoch})")

    # Load + split
    train_loader, val_loader, test_loader, splits = build_scheme_c_loaders(
        export_dir, batch_size=batch_size, max_len=max_len,
        val_ratio=val_ratio, test_b_ratio=test_b_ratio, seed=seed,
        include_soft_noise=include_soft_noise,
    )
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test-B: {len(test_loader.dataset)}")

    # Model (P1: configurable capacity)
    config = GNNLatentConfig(
        d_node=d_node, d_edge=d_edge, d_latent=d_latent,
        n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
        n_diffusion_steps=n_diffusion_steps, n_heads=n_heads
    )
    print(f"  Model: d_node={d_node}, d_edge={d_edge}, d_latent={d_latent}, "
          f"layers={n_encoder_layers}/{n_decoder_layers}, steps={n_diffusion_steps}")
    model = GNNLatentDiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # EMA (P2): shadow params for stable evaluation; required as teacher for P3
    ema_enabled = use_ema or self_distill
    ema = EMAWrapper(model, decay=ema_decay) if ema_enabled else None

    # 5-epoch warmup (per feedback_torusfold_scheme1_lr) + cosine decay
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5 / lr, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )

    best_val = float('inf')
    bond_length = 5.9

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        metrics = {'diff': 0, 'coord': 0, 'closure': 0, 'bond': 0, 'distill': 0}
        nan_batches = 0
        n_batches = 0

        # Curriculum: rescale B-layer base weight (0.3) to scheduled value.
        # Loader gives sample_w = base_weight * confidence. B-layer base=0.3
        # so sample_w < 0.5 identifies B samples (A is 1.0*conf >= 0.6).
        b_scale = 1.0
        if curriculum_b:
            scheduled = schedule_b_weight(epoch, warmup_epochs, full_at_epoch)
            b_scale = scheduled / 0.3 if 0.3 > 0 else 0.0  # ratio vs loader's 0.3

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']
            # Per-sample weights from confidence (carries A/B layer * confidence)
            # collate_fn puts confidence per sample; fallback to 1.0
            if 'confidence' in batch:
                sample_w = batch['confidence'].to(device).float()
            else:
                sample_w = torch.ones(target.shape[0], device=device)

            # Apply curriculum rescaling to B-layer samples
            if curriculum_b and b_scale != 1.0:
                b_mask = sample_w < 0.5  # B-layer (base 0.3 * conf)
                sample_w = torch.where(b_mask, sample_w * b_scale, sample_w)

            if torch.isinf(target).any() or torch.isnan(target).any():
                nan_batches += 1
                optimizer.zero_grad()
                continue

            B, L, _ = target.shape

            # Normalize target (per-sample center + scale)
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1, 2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            # Forward
            out = model(seq_ids, mode='train')
            pred_coords = out['coords']
            diff_loss = out.get('diffusion_loss', None)

            # Weighted coord loss (per-sample)
            pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
            coord_loss = weighted_coord_loss(pred_centered, target_norm, sample_w)

            # Closure loss (per-sample weighted)
            bond_length_norm = bond_length / target_scale.squeeze(-1)  # (B,)
            pred_closure_norm = torch.norm(
                pred_centered[:, 0] - pred_centered[:, -1], dim=-1
            )  # (B,)
            per_sample_closure = (pred_closure_norm - bond_length_norm.squeeze(-1)) ** 2
            closure_loss = (per_sample_closure * sample_w).sum() / sample_w.sum().clamp(min=1e-6)

            # Bond consistency (per-sample weighted)
            bond_loss = torch.tensor(0.0, device=device)
            total_w = 0.0
            for b in range(B):
                valid_L = lengths[b]
                if valid_L < 4:
                    continue
                bonds = torch.norm(
                    pred_centered[b, 1:valid_L] - pred_centered[b, :valid_L - 1], dim=-1
                )
                bsj = torch.norm(pred_centered[b, 0] - pred_centered[b, valid_L - 1])
                all_bonds = torch.cat([bonds, bsj.unsqueeze(0)])
                target_bond = bond_length_norm[b].expand(all_bonds.shape[0])
                per_sample_bond = ((all_bonds - target_bond) ** 2).mean()
                bond_loss = bond_loss + per_sample_bond * sample_w[b]
                total_w += sample_w[b].item()
            bond_loss = bond_loss / max(total_w, 1e-6)

            # Total
            if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
                loss = diff_loss * 1.0 + coord_loss * 10.0 + closure_loss * 5.0 + bond_loss * 2.0
            else:
                loss = coord_loss * 10.0 + closure_loss * 5.0 + bond_loss * 2.0

            # P3: Self-distillation — EMA teacher provides softened coord targets.
            # Teacher pred (no grad) blended with ground truth via confidence:
            #   soft_target = conf * gt + (1 - conf) * teacher_pred
            # Student matches soft_target. Teacher must be more stable than
            # student, so only active after distill_start_epoch.
            distill_loss = torch.tensor(0.0, device=device)
            if (self_distill and ema is not None
                    and epoch >= distill_start_epoch):
                backup = ema.apply_to(model)
                with torch.no_grad():
                    teacher_out = model(seq_ids, mode='train')
                    teacher_coords = teacher_out['coords']
                    teacher_centered = teacher_coords - teacher_coords.mean(dim=1, keepdim=True)
                ema.restore(model, backup)
                # Blend: high-confidence samples keep GT; low-conf lean on teacher
                conf = sample_w.clamp(0.0, 1.0).view(-1, 1, 1)
                soft_target = conf * target_norm + (1.0 - conf) * teacher_centered
                distill_loss = ((pred_centered - soft_target.detach()) ** 2).mean()
                loss = loss + distill_loss * distill_weight

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

            # EMA update (after optimizer step)
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            metrics['diff'] += diff_loss.item() if diff_loss is not None and torch.isfinite(diff_loss) else 0
            metrics['coord'] += coord_loss.item()
            metrics['closure'] += closure_loss.item()
            metrics['bond'] += bond_loss.item()
            metrics['distill'] += distill_loss.item() if torch.isfinite(distill_loss) else 0
            n_batches += 1

        # Step the right scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        n_valid = max(n_batches, 1)
        avg_train = train_loss / n_valid
        avg_coord = metrics['coord'] / n_valid
        avg_closure = metrics['closure'] / n_valid
        avg_bond = metrics['bond'] / n_valid
        avg_diff = metrics['diff'] / n_valid
        avg_distill = metrics['distill'] / n_valid

        # Validation: kabsch RMSD on val set
        model.eval()
        val_rmsd, n_val = _eval_rmsd(model, val_loader, device)
        avg_val = val_rmsd / max(n_val, 1) if n_val > 0 else avg_train * 100

        if avg_val < best_val:
            best_val = avg_val
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/scheme6_pipeline_best.pt")

        cur_b = schedule_b_weight(epoch, warmup_epochs, full_at_epoch) if curriculum_b else 0.3
        disp_str = f" distil={avg_distill:.4f}" if self_distill and epoch >= distill_start_epoch else ""
        print(f"  Epoch {epoch+1}/{epochs} train={avg_train:.4f} "
              f"(diff={avg_diff:.3f} coord={avg_coord:.3f} cls={avg_closure:.2f} bond={avg_bond:.2f}{disp_str}) "
              f"val={avg_val:.1f}Å (n={n_val}) nan={nan_batches} b_w={cur_b:.2f}")

    # Final test-B evaluation
    model.eval()
    test_rmsd, n_test = _eval_rmsd(model, test_loader, device)
    avg_test = test_rmsd / max(n_test, 1) if n_test > 0 else 0.0
    print(f"\n  Test-B RMSD: {avg_test:.2f}Å (n={n_test}) — held-out real")
    print(f"  Best val RMSD: {best_val:.2f}Å")
    return best_val, avg_test


def _eval_rmsd(model, loader, device):
    """Kabsch-aligned RMSD evaluation on a loader."""
    total_rmsd = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
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
                    total_rmsd += rmsd.item()
                    n += 1
    return total_rmsd, n


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Scheme 6 training on pipeline-exported data (three-layer weighted)'
    )
    parser.add_argument('--export-dir', required=True,
                        help='Pipeline output dir with coords.npy/confidences.npy/'
                             'sample_weights.npy/sources.npy')
    parser.add_argument('--output', default='models/torusfold_s6_pipeline')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--test-b-ratio', type=float, default=0.10)
    parser.add_argument('--max-len', type=int, default=None)
    parser.add_argument('--no-soft-noise', action='store_true',
                        help='Exclude B-layer soft noise (use A-layer only)')
    parser.add_argument('--curriculum-b', action='store_true',
                        help='Dynamic curriculum: ramp B-layer weight 0 -> 0.3 over warmup..full-at-epoch')
    parser.add_argument('--full-at-epoch', type=int, default=20,
                        help='Epoch when B-layer reaches full weight 0.3 (curriculum mode)')
    parser.add_argument('--seed', type=int, default=42)
    # P1: model capacity config
    parser.add_argument('--d-node', type=int, default=128, help='Node embedding dim')
    parser.add_argument('--d-edge', type=int, default=64, help='Edge embedding dim')
    parser.add_argument('--d-latent', type=int, default=256, help='Latent space dim')
    parser.add_argument('--n-encoder-layers', type=int, default=6)
    parser.add_argument('--n-decoder-layers', type=int, default=6)
    parser.add_argument('--n-diffusion-steps', type=int, default=100)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--use-ema', action='store_true', help='Enable EMA shadow params')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--self-distill', action='store_true',
                        help='P3: self-distillation (EMA teacher soft targets)')
    parser.add_argument('--distill-weight', type=float, default=2.0,
                        help='KD loss multiplier (self-distill mode)')
    parser.add_argument('--distill-start-epoch', type=int, default=10,
                        help='Epoch to activate self-distill')
    args = parser.parse_args()

    train_scheme6_pipeline(
        args.export_dir, args.output, args.epochs, args.batch_size, args.lr,
        args.device, args.val_ratio, args.test_b_ratio, args.max_len,
        include_soft_noise=not args.no_soft_noise, seed=args.seed,
        curriculum_b=args.curriculum_b, full_at_epoch=args.full_at_epoch,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        d_node=args.d_node, d_edge=args.d_edge, d_latent=args.d_latent,
        n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers,
        n_diffusion_steps=args.n_diffusion_steps, n_heads=args.n_heads,
        self_distill=args.self_distill, distill_weight=args.distill_weight,
        distill_start_epoch=args.distill_start_epoch,
    )
