"""train_rhofold_cg.py — TorusFold + RhoFold+ CG decoder 训练脚本.

Phase 1: 冻结 backbone, 训 CG decoder + PairTrack
Phase 2: unfreeze 最后 3 层, 微调整个模型

用法:
  # Phase 1 (冻结 backbone)
  python train_rhofold_cg.py --labels ./data/circrna_3d_all --device cuda --phase 1

  # Phase 2 (微调)
  python train_rhofold_cg.py --labels ./data/circrna_3d_all --device cuda --phase 2 --checkpoint models/torusfold_rhofold_phase1.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 添加路径
DEPLOY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEPLOY_ROOT))

from torusfold.torusfold_rhofold import TorusFoldRhoFold, TorusFoldRhoFoldConfig


# ═══════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════

class CircRNADataset(Dataset):
    """circRNA 训练数据集 (scheme2 格式).

    数据格式:
      npz 文件包含:
        - ids: (N,) 序列 ID
        - sequences: (N,) 序列字符串
        - coords: (N, L_max, 3) CG 坐标 (P atom)
        - lengths: (N,) 有效长度
    """

    def __init__(self, npz_path: str, min_len: int = 30, max_len: int = 2000):
        data = np.load(npz_path, allow_pickle=True)
        self.ids = data['ids']
        self.sequences = data['sequences']
        self.coords = data['coords']
        self.lengths = data['lengths']

        # 过滤长度
        mask = (self.lengths >= min_len) & (self.lengths <= max_len)
        self.ids = self.ids[mask]
        self.sequences = self.sequences[mask]
        self.coords = self.coords[mask]
        self.lengths = self.lengths[mask]

        print(f"  Dataset: {len(self)} samples, len {min_len}-{max_len}nt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'coords': self.coords[idx],
            'length': self.lengths[idx],
        }


def sequence_to_tokens(seq: str) -> torch.Tensor:
    """序列 → token IDs (A=0, U=1, G=2, C=3, N=4)."""
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    tokens = [mapping.get(c, 4) for c in seq.upper()]
    return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """Collate: padding to max length."""
    max_len = max(b['length'] for b in batch)
    B = len(batch)

    seq_ids = torch.zeros(B, max_len, dtype=torch.long)
    coords = torch.zeros(B, max_len, 3)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = int(b['length'])
        seq_ids[i, :L] = sequence_to_tokens(b['sequence'][:L])
        coords[i, :L] = torch.tensor(b['coords'][:L], dtype=torch.float32)
        lengths[i] = L

    return {
        'seq_ids': seq_ids,
        'coords': coords,
        'lengths': lengths,
    }


# ═══════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════

def train_phase1(args):
    """Phase 1: 冻结 backbone, 训 CG decoder + PairTrack."""
    print("\n" + "="*60)
    print("  Phase 1: 冻结 backbone, 训 CG decoder + PairTrack")
    print("="*60)

    device = torch.device(args.device)

    # 数据
    dataset = CircRNADataset(args.labels, min_len=args.min_len, max_len=args.max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # 模型
    config = TorusFoldRhoFoldConfig(
        freeze_backbone=True,
        freeze_layers=9,
        use_pair_repr=True,
        d_node=256,
        d_pair=64,
        pair_track_layers=2,
        use_equivariant_decoder=False,
    )
    model = TorusFoldRhoFold(config).to(device)
    model.freeze_backbone()

    # 优化器 (只训可训练参数)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # 训练循环
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            coords = batch['coords'].to(device)

            result = model(seq_ids, coords_target=coords)
            loss = result['losses']['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if not torch.isnan(loss):
                train_loss += loss.item()
                n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                coords = batch['coords'].to(device)
                result = model(seq_ids, coords_target=coords)
                val_loss += result['losses']['total'].item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{args.output}/torusfold_rhofold_phase1.pt")

    print(f"  Best val loss: {best_val:.4f}")
    return best_val


def train_phase2(args):
    """Phase 2: unfreeze 最后 3 层, 微调."""
    print("\n" + "="*60)
    print("  Phase 2: unfreeze 最后 3 层, 微调")
    print("="*60)

    device = torch.device(args.device)

    # 数据
    dataset = CircRNADataset(args.labels, min_len=args.min_len, max_len=args.max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size // 2, shuffle=True,  # 小 batch 防 OOM
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # 加载 Phase 1 checkpoint
    config = TorusFoldRhoFoldConfig(
        freeze_backbone=False,  # 不冻结
        freeze_layers=9,
        use_pair_repr=True,
    )
    model = TorusFoldRhoFold(config).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # Unfreeze 最后 3 层
    model.unfreeze_backbone_last_n(3)

    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.rna_fm.layers[-3:].parameters(), 'lr': 1e-5},
        {'params': model.node_proj.parameters(), 'lr': 1e-3},
        {'params': model.pair_proj.parameters(), 'lr': 1e-3},
        {'params': model.pair_track.parameters(), 'lr': 1e-3},
        {'params': model.cg_decoder.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-3)

    # 训练循环 (同 Phase 1, 但 lr 更小)
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            coords = batch['coords'].to(device)

            result = model(seq_ids, coords_target=coords)
            loss = result['losses']['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if not torch.isnan(loss):
                train_loss += loss.item()
                n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                coords = batch['coords'].to(device)
                result = model(seq_ids, coords_target=coords)
                val_loss += result['losses']['total'].item()
                n_val += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{args.output}/torusfold_rhofold_phase2.pt")

    print(f"  Best val loss: {best_val:.4f}")
    return best_val


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TorusFold + RhoFold+ CG decoder 训练")
    parser.add_argument('--labels', type=str, required=True, help="训练数据 NPZ 路径")
    parser.add_argument('--output', type=str, default='models/torusfold_rhofold', help="输出目录")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help="训练阶段")
    parser.add_argument('--checkpoint', type=str, default=None, help="Phase 2 用的 Phase 1 checkpoint")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--min-len', type=int, default=30)
    parser.add_argument('--max-len', type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.phase == 1:
        train_phase1(args)
    else:
        train_phase2(args)


if __name__ == '__main__':
    main()
