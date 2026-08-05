"""train_hierarchical.py — 训练层次化预测系统.

Phase 1: 训练 overlap 一致性 + chunk 坐标预测
Phase 2: 训练 fusion GNN
Phase 3: 端到端微调

用法:
  python train_hierarchical.py --labels ../data/circrna_3d_all_consolidated --epochs 10 --device cuda
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from chunk_splitter import ChunkSplitter, ChunkInfo
from chunk_predictor import ChunkFeatureExtractor, ChunkPrediction
from chunk_fusion import ChunkFusionGNN
from overlap_loss import OverlapConsistencyLoss
from msa_cluster import MSACluster


# ================================================================
# Dataset
# ================================================================
class CircRNADataset(torch.utils.data.Dataset):
    """circRNA 3D 结构数据集 (CG 坐标 + 序列)."""

    def __init__(self, npz_path: str, max_len: int = 2000):
        data = np.load(npz_path, allow_pickle=True)
        self.ids = data['ids']
        self.lengths = data['lengths'].astype(int)
        self.coords = data['coords']  # (N, max_L, 3)
        self.max_len = max_len

        # 过滤
        mask = self.lengths <= max_len
        self.ids = self.ids[mask]
        self.lengths = self.lengths[mask]
        self.coords = self.coords[mask]
        print(f"  Dataset: {len(self)} samples, max_len={max_len}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        L = self.lengths[idx]
        coords = torch.tensor(self.coords[idx, :L], dtype=torch.float32)
        seq = torch.randint(0, 4, (L,))  # 随机序列 (暂无真实序列)
        return {
            'seq': seq,
            'coords': coords,
            'length': L,
            'circular': torch.tensor(1.0),  # 全部视为环状
        }


def collate_fn(batch):
    """动态 padding collate."""
    max_len = max(b['seq'].shape[0] for b in batch)
    B = len(batch)

    seqs = torch.zeros(B, max_len, dtype=torch.long)
    coords = torch.zeros(B, max_len, 3)
    lengths = torch.zeros(B, dtype=torch.long)
    circular = torch.zeros(B)

    for i, b in enumerate(batch):
        L = b['seq'].shape[0]
        seqs[i, :L] = b['seq']
        coords[i, :L] = b['coords']
        lengths[i] = L
        circular[i] = b['circular']

    return {
        'seq': seqs,
        'coords': coords,
        'lengths': lengths,
        'circular': circular,
    }


# ================================================================
# Simplified Predictor (no RNA FM backbone for quick training)
# ================================================================
class SimpleChunkPredictor(nn.Module):
    """简化版 chunk 预测器: 不用 RNA FM, 直接从 token embedding 预测."""

    def __init__(self, d_model: int = 256, d_pair: int = 64):
        super().__init__()
        # Token embedding
        self.token_emb = nn.Embedding(5, d_model)

        # 简单 self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pair: bilinear projection weights
        self.pair_proj_q = nn.Parameter(torch.randn(d_model, d_pair) * 0.02)
        self.pair_proj_k = nn.Parameter(torch.randn(d_model, d_pair) * 0.02)

        # PairTrack lite: 1层
        self.pair_update = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, d_pair),
        )

        # Fusion
        self.pair_to_node = nn.Sequential(
            nn.Linear(d_pair, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        # CG Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, seq_tokens):
        """(B, L) -> coords (B, L, 3), node_repr (B, L, d_model)"""
        B, L = seq_tokens.shape

        # Token embedding
        x = self.token_emb(seq_tokens)  # (B, L, d_model)

        # Self-attention
        x = self.encoder(x)  # (B, L, d_model)

        # Pair features: bilinear (轻量, O(B*L^2*d))
        # W_q, W_k: (d, d_pair), W_v: (d, d_pair)
        # pair_ij = (x_i W_q) * (x_j W_k) -> (B, L, L, d_pair)
        q = torch.einsum('bid,dj->bijd', x, self.pair_proj_q)  # (B, L, d_pair)
        k = torch.einsum('bid,dj->bijd', x, self.pair_proj_k)  # (B, L, d_pair)
        # Outer product: q_i * k_j -> (B, L, L, d_pair)
        pair = q.unsqueeze(2) * k.unsqueeze(1)  # (B, L, L, d_pair)

        # Top-K sparsification
        K = min(30, L)
        topk_vals, topk_idx = pair.norm(dim=-1).topk(K, dim=-1)  # (B, L, K)
        pair_topk = torch.gather(pair, 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, pair.shape[-1]))

        # Pair update
        pair_topk = pair_topk + self.pair_update(pair_topk)

        # Pair -> Node
        pair_mean = pair_topk.mean(dim=2)  # (B, L, d_pair)
        pair_enhance = self.pair_to_node(pair_mean)
        x = self.norm(x + pair_enhance)

        # Decode
        coords = self.decoder(x)

        return coords, x


class SimpleFusionGNN(nn.Module):
    """简化版 fusion GNN: global attention over chunks."""

    def __init__(self, d_chunk: int = 256, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.pos_embed = nn.Embedding(64, d_chunk)  # max 64 chunks

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_chunk, nhead=n_heads, dim_feedforward=d_chunk*2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.offset_head = nn.Linear(d_chunk, 3)
        self.refine = nn.Sequential(
            nn.Linear(3 + d_chunk, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, chunk_features, chunk_coords_list, chunk_infos, total_length):
        """融合 chunk 预测成全链坐标."""
        device = chunk_features.device
        N = chunk_features.shape[0]

        if N == 1:
            return chunk_coords_list[0]

        # Position encoding: (N, d) + (1, N, d) -> (1, N, d) broadcast
        pos = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
        x = chunk_features + self.pos_embed.weight[pos.squeeze(1)]  # (N, d_chunk)

        # Global attention: needs (batch, seq, d)
        x = x.unsqueeze(0)  # (1, N, d)
        x = self.transformer(x).squeeze(0)  # (N, d_chunk)

        # Offset per chunk
        offsets = self.offset_head(x)  # (N, 3)
        if offsets.dim() == 1:
            offsets = offsets.unsqueeze(0)  # ensure (N, 3)

        # Apply transforms
        global_coords = torch.zeros(total_length, 3, device=device)
        counts = torch.zeros(total_length, 1, device=device)

        for i in range(N):
            L_chunk = chunk_coords_list[i].shape[0]
            info = chunk_infos[i]
            offset_i = offsets[i].view(1, 3)  # (1, 3)
            transformed = chunk_coords_list[i] + offset_i  # (L_chunk, 3) + (1, 3)

            for j in range(L_chunk):
                pos_global = (info.start + j) % total_length
                global_coords[pos_global] += transformed[j]
                counts[pos_global] += 1

        counts = counts.clamp(min=1)
        global_coords = global_coords / counts

        # Refine
        for i in range(N):
            L_chunk = chunk_coords_list[i].shape[0]
            info = chunk_infos[i]
            for j in range(L_chunk):
                pos_global = (info.start + j) % total_length
                feat = x[i].view(-1)  # (d_chunk,)
                coord = global_coords[pos_global].view(-1)  # (3,)
                combined = torch.cat([coord, feat])
                delta = self.refine(combined)
                global_coords[pos_global] = global_coords[pos_global] + delta

        return global_coords


# ================================================================
# Training
# ================================================================
def train_one_epoch(
    model: SimpleChunkPredictor,
    fusion: SimpleFusionGNN,
    feature_extractor: ChunkFeatureExtractor,
    dataset: CircRNADataset,
    optimizer: torch.optim.Optimizer,
    overlap_criterion: OverlapConsistencyLoss,
    splitter: ChunkSplitter,
    device: str,
    chunk_size: int = 200,
    max_chunks: int = 8,
):
    """训练一个 epoch."""
    model.train()
    fusion.train()
    feature_extractor.train()

    total_loss = 0
    total_coord_loss = 0
    total_overlap_loss = 0
    n_batches = 0

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn,
        num_workers=0, drop_last=True,
    )

    for batch in dataloader:
        seq = batch['seq'].to(device)
        coords_gt = batch['coords'].to(device)
        lengths = batch['lengths'].to(device)
        circular = batch['circular'].to(device)

        B = seq.shape[0]
        batch_loss = 0
        batch_coord = 0
        batch_overlap = 0

        for b in range(B):
            L = int(lengths[b].item())
            seq_b = seq[b, :L]
            coords_b = coords_gt[b, :L]
            is_circ = circular[b] > 0

            # 切分 chunk
            chunks = splitter.split(seq_b, is_circular=is_circ, bsj_pos=0)

            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]

            chunk_coords = []
            chunk_losses = []

            # 每个 chunk 独立预测
            for chunk in chunks:
                tokens = chunk.seq_tokens.unsqueeze(0)  # (1, L_chunk)
                pred_coords, node_repr = model(tokens)
                pred_coords = pred_coords.squeeze(0)  # (L_chunk, 3)
                chunk_coords.append(pred_coords)

                # GT 坐标
                gt_chunk = coords_b[chunk.start:chunk.end]

                # 坐标 loss (如果 GT 足够长)
                if gt_chunk.shape[0] == pred_coords.shape[0]:
                    coord_loss = F.mse_loss(pred_coords, gt_chunk)
                else:
                    # 对齐长度
                    min_len = min(gt_chunk.shape[0], pred_coords.shape[0])
                    coord_loss = F.mse_loss(pred_coords[:min_len], gt_chunk[:min_len])

                chunk_losses.append(coord_loss)

            # Overlap 一致性 loss
            if len(chunk_coords) > 1:
                overlap_losses = overlap_criterion(chunk_coords, chunks, L)
                overlap_loss = overlap_losses['total']
            else:
                overlap_loss = torch.tensor(0.0, device=device)

            # 特征提取 + fusion
            chunk_features = []
            for i, chunk in enumerate(chunks):
                L_chunk = chunk.end - chunk.start
                pred = ChunkPrediction(
                    coords=chunk_coords[i].detach(),
                    node_repr=torch.randn(L_chunk, 256, device=device),
                    contact_map=torch.cdist(chunk_coords[i].unsqueeze(0), chunk_coords[i].unsqueeze(0)).squeeze(0),
                    bsj_confidence=0.85,
                    chunk_id=chunk.chunk_id,
                    start=chunk.start,
                    end=chunk.end,
                )
                feat = feature_extractor(pred)
                chunk_features.append(feat)

            features_tensor = torch.stack(chunk_features)

            # Fusion
            global_coords = fusion(features_tensor, chunk_coords, chunks, L)

            # 全链 coord loss
            global_coord_loss = F.mse_loss(global_coords, coords_b)

            # 总 loss
            loss = (
                sum(chunk_losses) * 1.0 +      # chunk coord loss
                overlap_loss * 1.0 +             # overlap consistency
                global_coord_loss * 2.0          # global coord loss
            )

            batch_loss += loss
            batch_coord += global_coord_loss.item()
            batch_overlap += overlap_loss.item()

        # Backward
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(fusion.parameters()) + list(feature_extractor.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += batch_loss.item()
        total_coord_loss += batch_coord
        total_overlap_loss += batch_overlap
        n_batches += 1

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'coord_loss': total_coord_loss / n,
        'overlap_loss': total_overlap_loss / n,
    }


@torch.no_grad()
def evaluate(
    model: SimpleChunkPredictor,
    fusion: SimpleFusionGNN,
    feature_extractor: ChunkFeatureExtractor,
    dataset: CircRNADataset,
    splitter: ChunkSplitter,
    device: str,
    chunk_size: int = 200,
    max_chunks: int = 8,
    n_eval: int = 100,
):
    """评估."""
    model.eval()
    fusion.eval()
    feature_extractor.eval()

    total_closure = 0
    total_coord = 0
    n = 0

    indices = np.random.choice(len(dataset), min(n_eval, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        seq = sample['seq'].unsqueeze(0).to(device)
        coords_gt = sample['coords'].unsqueeze(0).to(device)
        L = sample['length']

        chunks = splitter.split(sample['seq'], is_circular=True, bsj_pos=0)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        chunk_coords = []
        for chunk in chunks:
            tokens = chunk.seq_tokens.unsqueeze(0)
            pred_coords, _ = model(tokens)
            chunk_coords.append(pred_coords.squeeze(0))

        # Fusion
        if len(chunk_coords) > 1:
            features = []
            for i, chunk in enumerate(chunks):
                L_chunk = chunk.end - chunk.start
                pred = ChunkPrediction(
                    coords=chunk_coords[i].detach(),
                    node_repr=torch.randn(L_chunk, 256, device=device),
                    contact_map=torch.cdist(chunk_coords[i].unsqueeze(0), chunk_coords[i].unsqueeze(0)).squeeze(0),
                    bsj_confidence=0.85,
                    chunk_id=chunk.chunk_id,
                    start=chunk.start,
                    end=chunk.end,
                )
                feat = feature_extractor(pred)
                features.append(feat)

            features_tensor = torch.stack(features)
            global_coords = fusion(features_tensor, chunk_coords, chunks, L)
        else:
            global_coords = chunk_coords[0]

        # BSJ closure
        closure = torch.norm(global_coords[0] - global_coords[-1]).item()
        total_closure += closure

        # Coord MSE
        min_len = min(global_coords.shape[0], coords_gt.shape[1])
        coord_mse = F.mse_loss(global_coords[:min_len], coords_gt[0, :min_len]).item()
        total_coord += coord_mse

        n += 1

    return {
        'bsj_closure': total_closure / max(n, 1),
        'coord_mse': total_coord / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, help='Path to consolidated npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk_size', type=int, default=200)
    parser.add_argument('--overlap', type=int, default=30)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--output_dir', default='models/hierarchical')
    args = parser.parse_args()

    print("=" * 60)
    print("  Hierarchical Predictor Training")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    dataset = CircRNADataset(args.labels, max_len=args.max_len)

    # Models
    model = SimpleChunkPredictor(d_model=256, d_pair=64).to(args.device)
    fusion = SimpleFusionGNN(d_chunk=256, n_heads=8, n_layers=2).to(args.device)
    feature_extractor = ChunkFeatureExtractor(d_feature=256).to(args.device)

    n_params = sum(p.numel() for p in model.parameters()) + \
               sum(p.numel() for p in fusion.parameters()) + \
               sum(p.numel() for p in feature_extractor.parameters())
    print(f"  Total params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(fusion.parameters()) + list(feature_extractor.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    overlap_criterion = OverlapConsistencyLoss().to(args.device)

    # Splitter
    splitter = ChunkSplitter(chunk_size=args.chunk_size, overlap=args.overlap)

    # Training loop
    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}, device={args.device}")
    print(f"  Chunk: size={args.chunk_size}, overlap={args.overlap}")
    print()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()

        metrics = train_one_epoch(
            model, fusion, feature_extractor, dataset, optimizer,
            overlap_criterion, splitter, args.device,
            chunk_size=args.chunk_size,
        )

        scheduler.step()

        elapsed = time.time() - t0

        # Evaluate
        eval_metrics = evaluate(
            model, fusion, feature_extractor, dataset, splitter, args.device,
            chunk_size=args.chunk_size, n_eval=50,
        )

        print(f"  Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s)")
        print(f"    train: loss={metrics['loss']:.4f} coord={metrics['coord_loss']:.4f} overlap={metrics['overlap_loss']:.4f}")
        print(f"    eval:  bsj_closure={eval_metrics['bsj_closure']:.2f}A coord_mse={eval_metrics['coord_mse']:.4f}")

        # Save best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'model': model.state_dict(),
                'fusion': fusion.state_dict(),
                'feature_extractor': feature_extractor.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"    -> saved best (loss={best_loss:.4f})")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
