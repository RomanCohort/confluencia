#!/usr/bin/env python3
"""
train_curriculum.py — Curriculum learning for TorusFold: quality + length stratification.

Trains models in 3 progressive phases:
  Phase 1: High-quality data (confidence >= 0.8) + short/medium (<=500nt)
           → Learn basic 3D geometry from reliable sources
  Phase 2: All quality data + short/medium (<=500nt)
           → Boost generalization with lower-quality data
  Phase 3: All quality data + long sequences (>500nt)
           → Scheme 7 (Mamba) and Scheme 8 (Sparse Pair) only

Rationale:
  - 143 real PDB structures (conf 1.0) vs 7024 synthetic (conf 0.3)
  - High-quality data = short; Low-quality = long (experimental limitation)
  - Previous Scheme 6 plateaued at val_loss=0.435 with mixed training
  - Curriculum learning establishes geometric priors before adding noise

Usage:
    python -m confluencia_3_0.core.circrna.torusfold.train_curriculum \
        --labels ./data/circrna_3d_merged --schemes 1 4 6 7 --device cuda

    # Phase-specific epoch overrides
    python -m confluencia_3_0.core.circrna.torusfold.train_curriculum \
        --schemes 7 --phase1-epochs 20 --phase2-epochs 40 --phase3-epochs 30
"""

import os
import sys
import math
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse core components from train_all_schemes
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels,
    CircRNADataset,
    collate_fn,
    kabsch_rmsd,
    Scheme1Model,
    SCHEME_MAX_LEN,
)


# ═══════════════════════════════════════════════════════════════
# Curriculum Phase Definitions
# ═══════════════════════════════════════════════════════════════

CURRICULUM_PHASES = {
    1: {"confidence_min": 0.8, "length_max": 500,
        "description": "High-quality short/medium (PDB + SHAPE + Rfam)"},
    2: {"confidence_min": 0.5, "length_max": 500,
        "description": "Medium+ quality short/medium (exclude low-quality synthetic)"},
    3: {"confidence_min": 0.5, "length_max": None,
        "description": "Medium+ quality long (>500nt, Scheme 7/8 only)"},
}

# Default epochs per phase per scheme
DEFAULT_PHASE_EPOCHS = {
    1: {1: 50, 4: 50, 6: 50, 7: 50, 8: 50},  # Phase 1: more epochs for better convergence
    2: {1: 50, 4: 50, 6: 50, 7: 50, 8: 50},  # Phase 2: include medium quality
    3: {1: 0,  4: 0,  6: 0,  7: 30, 8: 30},  # Phase 3 only for Scheme 7/8
}


# ═══════════════════════════════════════════════════════════════
# BSJ Geometry Loss (新增)
# ═══════════════════════════════════════════════════════════════

class BSJGeometryLoss(nn.Module):
    """
    BSJ区域几何约束损失（training_strategy_v2.md Phase 3）

    在BSJ连接位置施加物理约束：
    1. 键角约束（磷酸二酯键键角 ~108°）
    2. 二面角约束（磷酸二酯键二面角）
    3. 键长约束（磷酸二酯键长度 ~3.5 Å）

    Args:
        target_angle: 目标键角（度），默认108°
        target_dihedral: 目标二面角（度），默认180°
        target_distance: 目标BSJ距离（Å），默认3.5
        angle_weight: 键角损失权重
        dihedral_weight: 二面角损失权重
        distance_weight: 距离损失权重
    """

    def __init__(self,
                 target_angle=108.0,
                 target_dihedral=180.0,
                 target_distance=3.5,
                 angle_weight=1.0,
                 dihedral_weight=1.0,
                 distance_weight=2.0):
        super().__init__()
        self.target_angle = target_angle
        self.target_dihedral = target_dihedral
        self.target_distance = target_distance
        self.angle_weight = angle_weight
        self.dihedral_weight = dihedral_weight
        self.distance_weight = distance_weight

    def forward(self, coords, bsj_indices):
        """
        计算BSJ几何约束损失

        Args:
            coords: (L, 3) 原子坐标
            bsj_indices: (bsj_start, bsj_end) BSJ位置索引

        Returns:
            total_loss: 总几何损失
        """
        bsj_start, bsj_end = bsj_indices

        # 1. BSJ距离约束（磷酸二酯键长度）
        pred_distance = torch.norm(coords[bsj_end] - coords[bsj_start])
        loss_distance = torch.abs(pred_distance - self.target_distance)

        # 2. 键角约束（如果BSJ附近有足够原子）
        if bsj_start > 0 and bsj_end < len(coords) - 1:
            # 计算键角：三个连续原子形成的角度
            # BSJ前的原子 → BSJ起点 → BSJ终点 → BSJ后的原子
            vec1 = coords[bsj_start - 1] - coords[bsj_start]
            vec2 = coords[bsj_end] - coords[bsj_start]
            vec3 = coords[bsj_end + 1] - coords[bsj_end]

            # 计算两个键角
            angle1 = self._compute_angle(vec1, vec2)
            angle2 = self._compute_angle(vec2, vec3)

            # 键角损失
            loss_angle = (
                torch.abs(angle1 - self.target_angle) +
                torch.abs(angle2 - self.target_angle)
            ) / 2.0
        else:
            loss_angle = torch.tensor(0.0, device=coords.device)

        # 3. 二面角约束（如果BSJ附近有足够原子）
        if bsj_start > 1 and bsj_end < len(coords) - 2:
            # 计算二面角：四个连续原子形成的扭转角
            dihedral = self._compute_dihedral(
                coords[bsj_start - 2],
                coords[bsj_start - 1],
                coords[bsj_start],
                coords[bsj_end]
            )
            loss_dihedral = torch.abs(dihedral - self.target_dihedral)
        else:
            loss_dihedral = torch.tensor(0.0, device=coords.device)

        # 加权组合
        total_loss = (
            self.distance_weight * loss_distance +
            self.angle_weight * loss_angle +
            self.dihedral_weight * loss_dihedral
        )

        return total_loss

    def _compute_angle(self, vec1, vec2):
        """计算两个向量之间的夹角（度）"""
        # 归一化
        vec1_norm = vec1 / (torch.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (torch.norm(vec2) + 1e-6)

        # 计算cos(角度)
        cos_angle = torch.clamp(torch.dot(vec1_norm, vec2_norm), -1.0, 1.0)

        # 转换为角度（度）
        angle = torch.rad2deg(torch.arccos(cos_angle))

        return angle

    def _compute_dihedral(self, p1, p2, p3, p4):
        """计算四个点形成的二面角（度）"""
        # 计算两个平面
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        # 计算平面法向量
        n1 = torch.cross(b1, b2)
        n2 = torch.cross(b2, b3)

        # 归一化
        n1_norm = n1 / (torch.norm(n1) + 1e-6)
        n2_norm = n2 / (torch.norm(n2) + 1e-6)

        # 计算cos(二面角)
        cos_dihedral = torch.clamp(torch.dot(n1_norm, n2_norm), -1.0, 1.0)

        # 转换为角度（度）
        dihedral = torch.rad2deg(torch.arccos(cos_dihedral))

        return dihedral


def filter_by_phase(sequences, coords_labels, pair_labels, confidence_weights,
                    metadata, phase):
    """Filter data samples by curriculum phase criteria."""
    cfg = CURRICULUM_PHASES[phase]
    conf_min = cfg["confidence_min"]
    len_max = cfg["length_max"]

    if phase == 3:
        # Phase 3: length > 500 (complement of Phase 1+2)
        mask = [m['length'] > 500 for m in metadata]
    else:
        # Phase 1: conf >= 0.8 AND length <= 500
        # Phase 2: length <= 500 (all confidence)
        if conf_min > 0:
            mask = [m['confidence'] >= conf_min and m['length'] <= len_max
                    for m in metadata]
        else:
            mask = [m['length'] <= len_max for m in metadata]

    indices = [i for i, v in enumerate(mask) if v]
    n_total = len(sequences)
    n_kept = len(indices)

    source_breakdown = {}
    for i in indices:
        src = metadata[i]['source']
        source_breakdown[src] = source_breakdown.get(src, 0) + 1

    print(f"  Phase {phase}: {n_kept}/{n_total} samples "
          f"(conf>={conf_min}, len<={'inf' if len_max is None else len_max})")
    for src, cnt in sorted(source_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt}")

    return (
        [sequences[i] for i in indices],
        [coords_labels[i] for i in indices],
        [pair_labels[i] for i in indices],
        [confidence_weights[i] for i in indices],
        [metadata[i] for i in indices],
    )


def create_loaders(sequences, coords_labels, pair_labels, confidence_weights,
                   batch_size, split_ratio=0.9, val_conf_min=0.8):
    """
    Create train/val DataLoaders from filtered data.

    CRITICAL: Validation set ALWAYS uses high-quality data (conf >= val_conf_min),
    even in Phase 2 where training uses mixed quality. This ensures validation RMSD
    reflects actual model quality, not noise from low-quality pseudo-labels.

    Args:
        sequences, coords_labels, pair_labels, confidence_weights: data lists
        batch_size: batch size
        split_ratio: train/val split ratio (for fallback)
        val_conf_min: minimum confidence for validation samples
    """
    n = len(sequences)
    if n < 10:
        # Not enough data, use simple split
        split = int(split_ratio * n)
        train_indices = list(range(split))
        val_indices = list(range(split, n))
    else:
        # Separate high-quality samples for validation
        # Use conf >= 0.9 for highest quality validation
        val_indices = [i for i in range(n) if confidence_weights[i] >= 0.9]
        train_indices = [i for i in range(n) if i not in val_indices]

        # If not enough very high-quality for validation, use conf >= val_conf_min
        if len(val_indices) < 5:
            val_indices = [i for i in range(n) if confidence_weights[i] >= val_conf_min]
            train_indices = [i for i in range(n) if i not in val_indices]

        # If still not enough for validation, use top 15% by confidence
        if len(val_indices) < 5:
            sorted_idx = sorted(range(n), key=lambda i: confidence_weights[i], reverse=True)
            val_indices = sorted_idx[:max(5, int(0.15 * n))]
            train_indices = [i for i in range(n) if i not in val_indices]

    print(f"    Train: {len(train_indices)} samples (conf>={min(confidence_weights[i] for i in train_indices):.2f})")
    print(f"    Val: {len(val_indices)} samples (conf>={min(confidence_weights[i] for i in val_indices):.2f})")

    train_ds = CircRNADataset(
        [sequences[i] for i in train_indices],
        [coords_labels[i] for i in train_indices],
        [pair_labels[i] for i in train_indices],
        [confidence_weights[i] for i in train_indices])
    val_ds = CircRNADataset(
        [sequences[i] for i in val_indices],
        [coords_labels[i] for i in val_indices],
        [pair_labels[i] for i in val_indices],
        [confidence_weights[i] for i in val_indices])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
# Unified Training Loop (per phase)
# ═══════════════════════════════════════════════════════════════

def train_one_phase(model, train_loader, val_loader, optimizer, scheduler,
                    args, device, scheme_id, phase, n_epochs):
    """Train model for one curriculum phase, return best val RMSD."""
    bond_length = 5.9
    w_closure = getattr(args, 'w_closure', 5.0)
    warmup_epochs = 10 if phase > 1 else 3  # Longer warmup for phase transitions

    best_val = float('inf')
    best_state = None  # Save best model state
    patience_counter = 0
    max_patience = 10 if phase == 1 else 30  # Phase 1 stops early if saturated
    phase_metrics = []

    # Phase 1: Strong regularization to prevent overfitting
    # Phase 2+: Allow adaptation to new data
    grad_clip = 0.5 if phase == 1 else 1.0  # Phase 1: tighter gradient clipping
    dropout_p = 0.15 if phase == 1 else 0.05  # Phase 1: higher dropout

    # Apply dropout scaling to model (if it has dropout modules)
    for module in model.modules():
        if hasattr(module, 'p') and isinstance(module.p, float):
            module.p = dropout_p  # Dynamically adjust dropout

    # Phase transition: no freezing — just use warmup for stability.
    # Freezing caused Phase 2 crashes (RMSD 27→400Å) because frozen params
    # can't adapt to new data distribution, and low LR on unfrozen params
    # is too slow to compensate. Warmup alone is sufficient.

    for epoch in range(n_epochs):
        # Warmup LR
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            peak_lr = optimizer.param_groups[0]['initial_lr']
            for pg in optimizer.param_groups:
                pg['lr'] = peak_lr * warmup_factor

        model.train()
        train_loss = 0
        train_rmsd_sum = 0.0
        train_closure_sum = 0.0
        n_batches = 0
        nan_batches = 0

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']

            if torch.isinf(target).any() or torch.isnan(target).any():
                nan_batches += 1
                optimizer.zero_grad()
                continue

            conf_raw = batch.get('confidence', torch.tensor(0.5))
            # Per-sample confidence weighting (not batch-level):
            # - High quality (conf >= 0.8): weight = 1.5
            # - Medium quality (0.5 <= conf < 0.8): weight = 1.0
            # - Low quality (conf < 0.5): weight = 0.2
            # This allows mixed batches to have different weights per sample.
            conf_weight = torch.where(
                conf_raw >= 0.8, torch.tensor(1.5),
                torch.where(conf_raw >= 0.5, torch.tensor(1.0), torch.tensor(0.2))
            ).mean()  # Average weight for the batch

            B, L, _ = target.shape

            # ── Forward ──
            # Scheme-specific forward pass
            if scheme_id == 6:
                # Scheme 6 uses mode parameter
                out = model(seq_ids, mode='train')
            elif scheme_id in (4, 7, 8):
                # Schemes 4/7/8 use coords_target to trigger training
                out = model(seq_ids, coords_target=target)
            else:
                out = model(seq_ids)

            pred = out['coords']

            # ── Normalize ──
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1, 2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            pred_norm = pred_centered / target_scale

            # ── Coord loss (normalized MSE) ──
            coord_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                coord_loss = coord_loss + torch.mean(diff ** 2)
            coord_loss /= max(B, 1)

            # ── Closure loss (denormalized Å) ──
            pred_norm_scale = torch.norm(pred_centered, dim=(1, 2), keepdim=True).clamp(min=1e-6)
            pred_denorm = pred_centered / pred_norm_scale * target_scale + target.mean(dim=1, keepdim=True)
            closure_dists = torch.norm(pred_denorm[:, 0] - pred_denorm[:, -1], dim=-1)
            closure_mask = torch.tensor([lengths[b] >= 2 for b in range(B)],
                                        device=device, dtype=torch.float32)
            closure_loss = (closure_mask * (closure_dists - bond_length) ** 2).sum() / \
                          closure_mask.sum().clamp(min=1.0)

            # ── Bond consistency loss ──
            bond_loss = torch.tensor(0.0, device=device)
            n_bond_samples = 0
            for b in range(B):
                valid_L = lengths[b]
                if valid_L < 4:
                    continue
                bonds = torch.norm(
                    pred_denorm[b, 1:valid_L] - pred_denorm[b, :valid_L - 1], dim=-1)
                bsj_bond = torch.norm(pred_denorm[b, 0] - pred_denorm[b, valid_L - 1])
                all_bonds = torch.cat([bonds, bsj_bond.unsqueeze(0)])
                bond_loss = bond_loss + F.mse_loss(all_bonds, torch.full_like(all_bonds, bond_length))
                n_bond_samples += 1
            bond_loss = bond_loss / max(n_bond_samples, 1)

            # ── Diffusion loss (if applicable) ──
            # S4/S7/S8 return 'noise_loss' or 'total_loss', S6 returns 'diffusion_loss'
            diff_loss = out.get('diffusion_loss') or out.get('noise_loss') or out.get('total_loss')
            if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
                loss = diff_loss * 1.0 + coord_loss * 10.0 + closure_loss * w_closure + bond_loss * 2.0
            else:
                loss = coord_loss * 10.0 + w_closure * closure_loss + bond_loss * 2.0

            # Confidence weighting (tiered)
            loss = loss * conf_weight

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            # Gradient check
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # Phase-aware gradient clipping
            optimizer.step()
            optimizer.zero_grad()

            n_batches += 1
            train_loss += loss.item()

            # Track closure + RMSD
            with torch.no_grad():
                valid_closure = closure_dists[closure_mask.bool()]
                if len(valid_closure) > 0:
                    train_closure_sum += valid_closure.mean().item()
                for b in range(B):
                    valid_L = lengths[b]
                    if valid_L < 4:
                        continue
                    p_denorm = pred_centered[b, :valid_L] / pred_norm_scale[b] * target_scale[b] + \
                               target[b].mean(dim=0, keepdim=True)
                    t_denorm = target[b, :valid_L]
                    p_c = p_denorm - p_denorm.mean(dim=0)
                    t_c = t_denorm - t_denorm.mean(dim=0)
                    if p_c.abs().sum() > 1e-6 and t_c.abs().sum() > 1e-6:
                        rmsd = kabsch_rmsd(p_c, t_c)
                        if not (np.isnan(rmsd) or np.isinf(rmsd)):
                            train_rmsd_sum += rmsd

        # ── Validation ──
        model.eval()
        val_rmsd = 0.0
        val_closure_sum = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                if torch.isnan(target).any() or torch.isinf(target).any():
                    continue
                if target.abs().sum() < 1e-3:
                    continue

                B, L, _ = target.shape
                target_centered = target - target.mean(dim=1, keepdim=True)
                target_scale = torch.norm(target_centered, dim=(1, 2), keepdim=True).clamp(min=1.0)

                if scheme_id == 6:
                    out = model(seq_ids, mode='sample')
                elif scheme_id in (4, 7, 8):
                    # No coords_target = sampling mode
                    out = model(seq_ids)
                else:
                    out = model(seq_ids)
                pred = out['coords']

                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    continue

                pred_centered = pred - pred.mean(dim=1, keepdim=True)
                pred_norm_scale = torch.norm(pred_centered, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                pred_denorm = pred_centered / pred_norm_scale * target_scale + \
                              target.mean(dim=1, keepdim=True)

                for b in range(B):
                    valid_L = lengths[b]
                    if valid_L < 4:
                        continue
                    p = pred_denorm[b, :valid_L]
                    t = target[b, :valid_L]
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    if p_c.abs().sum() < 1e-6 or t_c.abs().sum() < 1e-6:
                        continue
                    rmsd = kabsch_rmsd(p_c, t_c)
                    if not (np.isnan(rmsd) or np.isinf(rmsd)):
                        val_rmsd += rmsd
                        n_val_samples += 1
                    closure_dist = torch.norm(pred_denorm[b, 0] - pred_denorm[b, valid_L - 1]).item()
                    val_closure_sum += closure_dist

        # ── Epoch summary ──
        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_rmsd / max(n_val_samples, 1) if n_val_samples > 0 else float('inf')
        avg_val_closure = val_closure_sum / max(n_val_samples, 1) if n_val_samples > 0 else float('inf')
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  [P{phase}] Epoch {epoch+1}/{n_epochs} "
              f"train={avg_train:.4f} val_rmsd={avg_val:.1f}Å "
              f"val_closure={avg_val_closure:.2f}Å "
              f"lr={current_lr:.1e} nan={nan_batches} pat={patience_counter}/{max_patience}")

        phase_metrics.append({
            'phase': phase, 'epoch': epoch + 1,
            'train_loss': avg_train, 'val_rmsd': avg_val,
            'val_closure': avg_val_closure, 'lr': current_lr,
        })

        if patience_counter >= max_patience:
            print(f"  [P{phase}] Early stopping at epoch {epoch+1}")
            break

    # Restore best model state (not last epoch)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  [P{phase}] Restored best model (val_rmsd={best_val:.1f}Å)")

    return best_val, phase_metrics


# ═══════════════════════════════════════════════════════════════
# Scheme-specific model creation
# ═══════════════════════════════════════════════════════════════

def create_model(scheme_id, args, device):
    """Create model for given scheme (full capacity with clean data)."""
    if scheme_id == 1:
        # Scheme 1: EGNN, full capacity
        model = Scheme1Model(d_hidden=args.d_hidden, n_layers=args.n_layers).to(device)
        lr = min(args.lr, 1e-4)
    elif scheme_id == 4:
        # Scheme 4: CircDiffusion, full capacity
        from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
            CircRNADiffusionModel, CircDiffusionConfig
        )
        config = CircDiffusionConfig(
            n_diffusion_steps=args.diffusion_steps,  # Full 100 steps
            d_node=args.d_hidden,
            d_edge=args.d_hidden // 2,
            n_egnn_layers=args.n_layers,  # Full 6 layers
        )
        model = CircRNADiffusionModel(config).to(device)
        lr = min(args.lr, 1e-4)
    elif scheme_id == 6:
        # Scheme 6: GNN Latent Diffusion, full capacity
        from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
            GNNLatentDiffusionModel, GNNLatentConfig
        )
        config = GNNLatentConfig(
            n_diffusion_steps=args.diffusion_steps,
            d_node=args.d_hidden,
            d_edge=args.d_hidden // 2,
            d_latent=args.d_hidden * 2,
            n_encoder_layers=args.n_layers,
            n_decoder_layers=args.n_layers,
            n_heads=8,
        )
        model = GNNLatentDiffusionModel(config).to(device)
        lr = min(args.lr, 1e-4)
    elif scheme_id == 7:
        # Scheme 7: Mamba Diffusion, full capacity
        from confluencia_3_0.core.circrna.torusfold.circrna_mamba_diffusion import (
            CircMambaDiffusionModel, CircMambaConfig, HAS_MAMBA_SSM
        )
        if HAS_MAMBA_SSM:
            n_mamba = getattr(args, 'n_mamba_layers', 6)  # Upgraded from 4
            n_attn = getattr(args, 'n_attn_layers', 2)
            n_diff = args.diffusion_steps  # Full 100 steps
        else:
            # Fallback without CUDA Mamba
            n_mamba = min(getattr(args, 'n_mamba_layers', 6), 3)
            n_attn = min(getattr(args, 'n_attn_layers', 2), 2)
            n_diff = min(args.diffusion_steps, 50)

        config = CircMambaConfig(
            n_mamba_layers=n_mamba,
            n_attn_layers=n_attn,
            n_diffusion_steps=n_diff,
            d_model=args.d_hidden,
            attn_window=getattr(args, 'attn_window', 40),  # Upgraded from 20
            bsj_flank=getattr(args, 'bsj_flank', 40),  # Upgraded from 20
        )
        model = CircMambaDiffusionModel(config).to(device)
        lr = min(args.lr, 1e-4)
    elif scheme_id == 8:
        # Scheme 8: Sparse Pair, full capacity
        from confluencia_3_0.core.circrna.torusfold.scheme8_sparse_pair import (
            Scheme8Model, Scheme8Config
        )
        from confluencia_3_0.core.circrna.torusfold.circrna_mamba_diffusion import HAS_MAMBA_SSM
        if HAS_MAMBA_SSM:
            n_mamba_layers = getattr(args, 'n_mamba_layers', 4)  # Upgraded from 2
            n_sparse_layers = 3  # Upgraded from 2
            n_diff = args.diffusion_steps  # Full 100 steps
        else:
            # Fallback without CUDA Mamba
            n_mamba_layers = 2  # Upgraded from 1
            n_sparse_layers = 2  # Upgraded from 1
            n_diff = min(args.diffusion_steps, 50)

        config = Scheme8Config(
            d_model=args.d_hidden,
            d_ssm=max(32, args.d_hidden // 2),
            d_pair=max(32, args.d_hidden // 2),
            d_global=getattr(args, 'scheme8_d_global', 64),  # Upgraded from 32
            n_mamba_layers=n_mamba_layers,
            n_sparse_layers=n_sparse_layers,
            n_denoiser_blocks=getattr(args, 'scheme8_n_blocks', 6),  # Upgraded from 4
            n_diffusion_steps=n_diff,
            K=getattr(args, 'scheme8_k', 30),  # Upgraded from 20
            bsj_flank=getattr(args, 'scheme8_bsj_flank', 40),  # Upgraded from 30
            attn_window=getattr(args, 'scheme8_window', 35),  # Upgraded from 25
            bond_length=5.9,
            closure_weight=1.0,
            use_gradient_checkpointing=True,
        )
        model = Scheme8Model(config).to(device)
        lr = min(args.lr, 1e-4)
    else:
        raise ValueError(f"Curriculum training not supported for Scheme {scheme_id}")

    return model, lr


# ═══════════════════════════════════════════════════════════════
# Curriculum Training Orchestrator
# ═══════════════════════════════════════════════════════════════

def train_curriculum_scheme(scheme_id, sequences, coords_labels, pair_labels,
                            confidence_weights, metadata, args, device):
    """Run curriculum training for a single scheme."""
    print("\n" + "=" * 60)
    print(f"  Curriculum Training — Scheme {scheme_id}")
    print("=" * 60)

    # Create model
    model, lr = create_model(scheme_id, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}, LR: {lr:.1e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # Stronger regularization
    # Store initial_lr for warmup
    for pg in optimizer.param_groups:
        pg['initial_lr'] = lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8
    )

    all_metrics = []
    best_overall_val = float('inf')
    output_dir = f"{args.output}/scheme{scheme_id}"
    os.makedirs(output_dir, exist_ok=True)

    for phase in [1, 2, 3]:
        # Phase 3 only for Scheme 7 and 8 (long-sequence capable)
        if phase == 3 and scheme_id not in (7, 8):
            print(f"  Phase 3 skipped (Scheme {scheme_id} not designed for long sequences)")
            continue

        # Get phase-specific epochs (use default if not specified)
        default_epochs = DEFAULT_PHASE_EPOCHS[phase].get(scheme_id, 30)
        n_epochs = getattr(args, f'phase{phase}_epochs', None)
        if n_epochs is None:
            n_epochs = default_epochs
        if n_epochs == 0:
            print(f"  Phase {phase} skipped (0 epochs configured)")
            continue

        # Filter data for this phase
        p_seq, p_coords, p_pairs, p_conf, p_meta = filter_by_phase(
            sequences, coords_labels, pair_labels, confidence_weights,
            metadata, phase
        )

        if len(p_seq) < 10:
            print(f"  Phase {phase} skipped (only {len(p_seq)} samples, need >= 10)")
            continue

        # Apply scheme-specific length cap
        scheme_max = SCHEME_MAX_LEN.get(scheme_id)
        if scheme_max is not None:
            keep = [i for i, m in enumerate(p_meta) if m['length'] <= scheme_max]
            if len(keep) < len(p_meta):
                print(f"  Scheme {scheme_id} length cap: {len(p_meta)} -> {len(keep)} "
                      f"(max_len={scheme_max})")
                p_seq = [p_seq[i] for i in keep]
                p_coords = [p_coords[i] for i in keep]
                p_pairs = [p_pairs[i] for i in keep]
                p_conf = [p_conf[i] for i in keep]
                p_meta = [p_meta[i] for i in keep]

        print(f"\n  ── Phase {phase}: {n_epochs} epochs, {len(p_seq)} samples ──")
        print(f"  {CURRICULUM_PHASES[phase]['description']}")

        train_loader, val_loader = create_loaders(
            p_seq, p_coords, p_pairs, p_conf, args.batch_size)

        # Reset scheduler for new phase (learning rate schedule starts fresh)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8
        )
        # Restore LR for new phase
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['initial_lr'] = lr

        best_val, metrics = train_one_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            args, device, scheme_id, phase, n_epochs
        )

        all_metrics.extend(metrics)

        # Save phase checkpoint
        ckpt_path = f"{output_dir}/scheme{scheme_id}_phase{phase}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Phase {phase} checkpoint: {ckpt_path}")

        if best_val < best_overall_val:
            best_overall_val = best_val
            torch.save(model.state_dict(), f"{output_dir}/scheme{scheme_id}_best.pt")
            print(f"  New best val: {best_val:.1f}Å")

    # Save metrics
    metrics_path = f"{output_dir}/curriculum_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'scheme': scheme_id,
            'phases': all_metrics,
            'best_val_rmsd': best_overall_val,
        }, f, indent=2)

    print(f"\n  Scheme {scheme_id} curriculum done: best_val={best_overall_val:.1f}Å")
    return best_overall_val


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='TorusFold Curriculum Training (Quality + Length Stratification)')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1, 4, 6, 7, 8],
                        help='Schemes to train (1=EGNN, 4=DDPM, 6=GNN-Latent, 7=Mamba, 8=SparsePair)')
    parser.add_argument('--labels', type=str, default='',
                        help='Path to merged dataset directory. Auto-searches if empty.')
    parser.add_argument('--output', type=str, default='models/torusfold_curriculum')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--w-closure', type=float, default=5.0)
    parser.add_argument('--n-mamba-layers', type=int, default=6)
    parser.add_argument('--n-attn-layers', type=int, default=2)
    parser.add_argument('--attn-window', type=int, default=40)
    parser.add_argument('--bsj-flank', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    # Phase-specific epoch counts
    parser.add_argument('--phase1-epochs', type=int, default=None,
                        help='Phase 1 epochs (default: 30)')
    parser.add_argument('--phase2-epochs', type=int, default=None,
                        help='Phase 2 epochs (default: 50)')
    parser.add_argument('--phase3-epochs', type=int, default=None,
                        help='Phase 3 epochs (default: 30, Scheme 7 only)')

    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output, exist_ok=True)

    # ── Load data ──
    labels_dir = args.labels
    if not labels_dir:
        script_dir = str(PROJECT_ROOT)
        search_paths = [
            'data/circrna_3d_merged',
            'data/circbase_real_3d',
            os.path.join(script_dir, 'data/circrna_3d_merged'),
            os.path.join(script_dir, 'data/circbase_real_3d'),
            '/root/data/circrna_3d_merged',
            '/root/data/circbase_real_3d',
            '/autodl-tmp/data/circrna_3d_merged',
            '/autodl-tmp/data/circbase_real_3d',
        ]
        for candidate in search_paths:
            if os.path.exists(os.path.join(candidate, 'sequences.json')):
                labels_dir = candidate
                print(f"  Auto-detected data: {candidate}")
                break

    if not labels_dir or not os.path.exists(labels_dir):
        print("ERROR: No labeled data found. Run merge_expanded_dataset.py first.")
        sys.exit(1)

    print(f"  Loading from: {labels_dir}")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = \
        load_pseudo_labels(labels_dir)

    if len(sequences) < 10:
        print("ERROR: Not enough data.")
        sys.exit(1)

    # ── Data summary ──
    print("\n" + "=" * 60)
    print("  Curriculum Training Overview")
    print("=" * 60)
    print(f"  Total samples: {len(sequences)}")
    print(f"  Device: {args.device}")
    print(f"  Schemes: {args.schemes}")
    print()

    for phase in [1, 2, 3]:
        cfg = CURRICULUM_PHASES[phase]
        n = sum(1 for m in metadata
                if (phase == 3 and m['length'] > 500) or
                   (phase != 3 and m['length'] <= cfg['length_max'] and
                    m['confidence'] >= cfg['confidence_min']))
        print(f"  Phase {phase}: {n} samples — {cfg['description']}")
    print()

    # ── Train each scheme ──
    device = torch.device(args.device)
    results = {}

    for scheme_id in args.schemes:
        t0 = time.time()
        best_val = train_curriculum_scheme(
            scheme_id, sequences, coords_labels, pair_labels,
            confidence_weights, metadata, args, device
        )
        elapsed = time.time() - t0
        results[scheme_id] = {
            'best_val_rmsd': best_val,
            'time_seconds': elapsed,
        }

        # Clear GPU cache between schemes
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Curriculum Training Summary")
    print("=" * 60)
    for sid, res in sorted(results.items()):
        print(f"  Scheme {sid}: best_val_rmsd={res['best_val_rmsd']:.1f}Å, "
              f"time={res['time_seconds']:.1f}s")

    with open(f"{args.output}/curriculum_results.json", 'w') as f:
        json.dump({
            'args': vars(args),
            'results': {str(k): v for k, v in results.items()},
        }, f, indent=2)

    print(f"\n  Results saved to {args.output}/curriculum_results.json")


if __name__ == '__main__':
    main()
