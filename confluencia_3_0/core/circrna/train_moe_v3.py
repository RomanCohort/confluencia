"""
train_moe_v3.py — TorusFold MOE V3 自举训练脚本

训练策略：
  1. 加载预训练的 TorusFold 模型
  2. 对 circBase 序列推理，提取结构信号
  3. 用启发式生成伪标签（免疫评分 + 目标权重）
  4. 训练 MOE Gate 和 Experts

Usage:
    python train_moe_v3.py \
        --torusfold-checkpoint models/torusfold_best.pt \
        --sequences data/circbase_130k.fa \
        --output models/moe_v3.pt \
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold_moe_v3 import (
    TorusFoldMOEIntegrated,
    MOEIntegratedConfig,
)
from confluencia_3_0.core.circrna.torusfold_scorer_v2 import extract_extended_signals


# ═══════════════════════════════════════════════════════════════
# Pseudo-label Generator
# ═══════════════════════════════════════════════════════════════

def generate_pseudo_labels(
    sequence: str,
    torusfold_signals: Optional[Dict],
) -> Dict:
    """从 TorusFold 信号生成伪标签。

    Args:
        sequence: circRNA 序列
        torusfold_signals: TorusFold 输出的结构信号

    Returns:
        {
            'immunogenicity': {'rig_i': 0.7, 'tlr7': 0.4, ...},
            'objective_weights': {'stability': 0.35, ...}
        }
    """
    L = len(sequence)
    gc = sum(1 for c in sequence.upper() if c in "GC") / max(L, 1)

    # === 提取 TorusFold 信号 ===
    if torusfold_signals and torusfold_signals.available:
        dsRNA_frac = torusfold_signals.dsRNA_fraction
        bsj_stab = torusfold_signals.bsj_stability
        sasa_mean = torusfold_signals.sasa_mean
        ires_access = torusfold_signals.motif_accessibility.get('ires', 0.5)
        m6a_access = torusfold_signals.motif_accessibility.get('m6a', 0.5)
    else:
        # 启发式兜底
        dsRNA_frac = 0.3
        bsj_stab = 0.5
        sasa_mean = 0.5
        ires_access = 0.5
        m6a_access = 0.5

    # === 免疫原性伪标签（启发式映射）===
    # RIG-I: dsRNA 高 → 激活强
    rig_i_label = dsRNA_frac * 0.7 + (1 - bsj_stab) * 0.3

    # TLR7: GU-rich motif（启发式）
    gu_count = sum(sequence.upper().count(m) for m in ["GU", "UG"])
    tlr7_label = min(gu_count / max(L / 50, 1), 1.0) * 0.6 + gc * 0.4

    # TLR8: AU-rich motif
    au_count = sum(sequence.upper().count(m) for m in ["AU", "UA"])
    tlr8_label = min(au_count / max(L / 50, 1), 1.0) * 0.6 + (1 - gc) * 0.4

    # PKR: long dsRNA
    pkr_label = dsRNA_frac * 0.8 + (1 - bsj_stab) * 0.2

    # === 目标权重伪标签（根据序列特征动态）===
    # 短序列 → stability 重要
    # 长序列 → delivery 重要
    # 高 dsRNA → immune_evasion 重要
    # 高 IRES 暴露 → translation 重要

    if L < 100:
        w_stability = 0.45
        w_translation = 0.25
        w_immune = 0.20
        w_delivery = 0.10
    elif L > 500:
        w_stability = 0.25
        w_translation = 0.20
        w_immune = 0.25
        w_delivery = 0.30
    else:
        w_stability = 0.30 + 0.1 * (1 - bsj_stab)
        w_translation = 0.25 + 0.1 * ires_access
        w_immune = 0.25 + 0.1 * dsRNA_frac
        w_delivery = 0.15

    # 归一化
    total = w_stability + w_translation + w_immune + w_delivery
    w_stability /= total
    w_translation /= total
    w_immune /= total
    w_delivery /= total

    return {
        'immunogenicity': {
            'rig_i': float(np.clip(rig_i_label, 0.0, 1.0)),
            'tlr7': float(np.clip(tlr7_label, 0.0, 1.0)),
            'tlr8': float(np.clip(tlr8_label, 0.0, 1.0)),
            'pkr': float(np.clip(pkr_label, 0.0, 1.0)),
        },
        'objective_weights': {
            'stability': float(w_stability),
            'translation': float(w_translation),
            'immune_evasion': float(w_immune),
            'delivery': float(w_delivery),
        },
    }


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class MOEDataset(Dataset):
    """MOE 训练数据集。

    每个样本：
      - sequence: circRNA 序列
      - torusfold_signals: TorusFold 提取的结构信号
      - pseudo_labels: 启发式生成的伪标签
    """

    def __init__(
        self,
        sequences: List[str],
        torusfold_outputs: Optional[List[Dict]] = None,
    ):
        self.sequences = sequences
        self.torusfold_outputs = torusfold_outputs or [None] * len(sequences)

        # 预计算伪标签
        print("Generating pseudo-labels...")
        self.pseudo_labels = []
        for i, (seq, tf_out) in enumerate(zip(sequences, self.torusfold_outputs)):
            label = generate_pseudo_labels(seq, tf_out)
            self.pseudo_labels.append(label)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'torusfold_signals': self.torusfold_outputs[idx],
            'pseudo_labels': self.pseudo_labels[idx],
        }


def collate_fn(batch):
    """自定义 collate（变长序列）。"""
    return {
        'sequences': [item['sequence'] for item in batch],
        'torusfold_signals': [item['torusfold_signals'] for item in batch],
        'pseudo_labels': [item['pseudo_labels'] for item in batch],
    }


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_epoch(
    model: TorusFoldMOEIntegrated,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    load_balance_weight: float = 0.01,
):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    total_imm_loss = 0.0
    total_weight_loss = 0.0
    total_lb_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequences']
        tf_signals = batch['torusfold_signals']
        labels = batch['pseudo_labels']

        batch_loss = 0.0

        for seq, tf_sig, label in zip(sequences, tf_signals, labels):
            # Forward
            pred = model(seq, tf_sig)

            # === Immunogenicity Loss ===
            pred_imm = pred['immunogenicity']['pathways']
            target_imm = label['immunogenicity']

            imm_loss = sum(
                (pred_imm.get(p, 0.0) - target_imm[p]) ** 2
                for p in ['rig_i', 'tlr7', 'tlr8', 'pkr']
            )

            # === Objective Weight Loss ===
            pred_weights = pred['objective_weights']
            target_weights = label['objective_weights']

            weight_loss = sum(
                (pred_weights.get(dim, 0.25) - target_weights[dim]) ** 2
                for dim in ['stability', 'translation', 'immune_evasion', 'delivery']
            )

            # === Load Balance Loss ===
            gate_logits = pred['gate_logits']

            # 免疫 gate 负载均衡
            imm_probs = F.softmax(torch.from_numpy(gate_logits['imm']), dim=0)
            imm_lb_loss = (imm_probs ** 2).sum()  # 鼓励均匀分布

            # 目标 gate 负载均衡
            obj_probs = F.softmax(torch.from_numpy(gate_logits['obj']), dim=0)
            obj_lb_loss = (obj_probs ** 2).sum()

            lb_loss = (imm_lb_loss + obj_lb_loss) * load_balance_weight

            # === Total Loss ===
            loss = imm_loss + weight_loss + lb_loss

            batch_loss += loss.item()
            total_imm_loss += imm_loss.item()
            total_weight_loss += weight_loss.item()
            total_lb_loss += lb_loss.item()

        # Average over batch
        batch_loss /= len(sequences)

        # Backward
        optimizer.zero_grad()
        batch_loss_tensor = torch.tensor(batch_loss, requires_grad=True)
        batch_loss_tensor.backward()
        optimizer.step()

        total_loss += batch_loss

    n = len(dataloader.dataset)
    return {
        'total_loss': total_loss / n,
        'imm_loss': total_imm_loss / n,
        'weight_loss': total_weight_loss / n,
        'lb_loss': total_lb_loss / n,
    }


def validate(
    model: TorusFoldMOEIntegrated,
    dataloader: DataLoader,
    device: str,
):
    """验证。"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences']
            tf_signals = batch['torusfold_signals']
            labels = batch['pseudo_labels']

            for seq, tf_sig, label in zip(sequences, tf_signals, labels):
                pred = model(seq, tf_sig)

                # Loss
                pred_imm = pred['immunogenicity']['pathways']
                target_imm = label['immunogenicity']

                imm_loss = sum(
                    (pred_imm.get(p, 0.0) - target_imm[p]) ** 2
                    for p in ['rig_i', 'tlr7', 'tlr8', 'pkr']
                )

                pred_weights = pred['objective_weights']
                target_weights = label['objective_weights']

                weight_loss = sum(
                    (pred_weights.get(dim, 0.25) - target_weights[dim]) ** 2
                    for dim in ['stability', 'translation', 'immune_evasion', 'delivery']
                )

                total_loss += (imm_loss + weight_loss).item()

    return {'val_loss': total_loss / len(dataloader.dataset)}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train MOE V3")
    parser.add_argument('--sequences', type=str, required=True,
                        help="FASTA file with circRNA sequences")
    parser.add_argument('--torusfold-checkpoint', type=str, default=None,
                        help="Pretrained TorusFold checkpoint (optional)")
    parser.add_argument('--torusfold-outputs', type=str, default=None,
                        help="Pre-computed TorusFold outputs JSON (optional)")
    parser.add_argument('--output', type=str, default='models/moe_v3.pt',
                        help="Output model path")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load-balance-weight', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-samples', type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # === Load sequences ===
    print(f"Loading sequences from {args.sequences}...")
    sequences = []
    with open(args.sequences, 'r') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)

    if args.n_samples:
        sequences = sequences[:args.n_samples]

    print(f"  Loaded {len(sequences)} sequences")

    # === Load TorusFold outputs (if available) ===
    torusfold_outputs = None
    if args.torusfold_outputs:
        print(f"Loading TorusFold outputs from {args.torusfold_outputs}...")
        with open(args.torusfold_outputs, 'r') as f:
            torusfold_outputs = json.load(f)
        print(f"  Loaded {len(torusfold_outputs)} TorusFold outputs")

    # === Create dataset ===
    dataset = MOEDataset(sequences, torusfold_outputs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Split train/val
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # === Create model ===
    print("Creating MOE V3 model...")
    config = MOEIntegratedConfig()
    model = TorusFoldMOEIntegrated(config).to(device)

    # === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # === Training loop ===
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.load_balance_weight
        )
        print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Imm: {train_metrics['imm_loss']:.4f}, "
              f"Weight: {train_metrics['weight_loss']:.4f}, "
              f"LB: {train_metrics['lb_loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"Val - Loss: {val_metrics['val_loss']:.4f}")

        # Save best
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best model to {args.output}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
