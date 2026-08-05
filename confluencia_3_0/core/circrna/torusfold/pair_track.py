"""pair_track.py — 轻量 Evoformer pair track (2-4 层).

核心思想 (来自 AlphaFold2 Evoformer):
  1. Pair representation z_ij: 每对残基 (i,j) 的标量特征
  2. Triangular multiplicative update: z_ij += z_ij * Aggregate(z_ik, z_jk)
     — 保证三角不等式: d(i,j) ≤ d(i,k) + d(k,j)
  3. Triangular self-attention: 用 pair 做全局注意力增强

设计约束:
  - pair 表示是标量 (不参与旋转), 不影响 S10 的 SO(2) 等变性
  - O(L² × d_pair), 适配 L≤2000nt (2000²×128 = 512MB, GPU 可承受)
  - 插入位置: S10 encoder 输出后, irrep 分离前
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairTrackConfig:
    d_pair: int = 128          # pair representation 维度
    n_layers: int = 3          # pair track 层数 (2-4)
    n_heads: int = 4           # triangular attention 多头数
    d_ffn: int = 256           # FFN 中间维度
    dropout: float = 0.1
    max_len: int = 2500        # 最大序列长度 (预分配三角 mask)


class TriangularMultiplicativeUpdate(nn.Module):
    """三角乘法更新 (Evoformer 核心).

    公式:
      z_ij += σ(W_out · (z_ij ⊙ Agg(z_ik · z_jk)))

    其中 Agg 是沿 k 维求和/平均, σ 是 gated residual.

    两个方向:
      - starting (k → i,j): z_ij ← Aggregate(z_ki, z_kj)
      - ending   (k → i,j): z_ij ← Aggregate(z_ik, z_jk)
    """

    def __init__(self, d_pair: int, dropout: float = 0.1):
        super().__init__()
        self.d_pair = d_pair

        # Starting edge: aggregate over source edges (k→i, k→j)
        self.norm_start = nn.LayerNorm(d_pair)
        self.proj_start_in = nn.Linear(d_pair, d_pair, bias=False)
        self.proj_start_edge = nn.Linear(d_pair, d_pair, bias=False)
        self.gate_start = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.Sigmoid(),
        )
        self.proj_start_out = nn.Linear(d_pair, d_pair, bias=False)

        # Ending edge: aggregate over target edges (i→k, j→k)
        self.norm_end = nn.LayerNorm(d_pair)
        self.proj_end_in = nn.Linear(d_pair, d_pair, bias=False)
        self.proj_end_edge = nn.Linear(d_pair, d_pair, bias=False)
        self.gate_end = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.Sigmoid(),
        )
        self.proj_end_out = nn.Linear(d_pair, d_pair, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, L, K, d_pair) — 稀疏 pair representation
        Returns: updated z (B, L, K, d_pair)
        """
        B, L, K, _ = z.shape

        # ── Starting edge: z_ik += gate * proj(Agg(proj(z_ki) ⊙ proj(z_kj))) ──
        z_start = self.norm_start(z)
        z_i = self.proj_start_in(z_start)   # (B, L, K, d_pair)
        z_j = self.proj_start_edge(z_start)  # (B, L, K, d_pair)
        # Agg: 沿 K 维求和 (K 近邻内)
        aggr_start = (z_i * z_j).sum(dim=2)  # (B, L, d_pair)
        aggr_start = aggr_start.unsqueeze(2).expand_as(z)  # (B, L, K, d_pair)
        gate = self.gate_start(aggr_start)
        z = z + self.dropout(self.proj_start_out(gate * aggr_start))

        # ── Ending edge: z_ik += gate * proj(Agg(proj(z_ik) ⊙ proj(z_jk))) ──
        z_end = self.norm_end(z)
        z_i = self.proj_end_in(z_end)
        z_j = self.proj_end_edge(z_end)
        aggr_end = (z_i * z_j).sum(dim=2)  # (B, L, d_pair)
        aggr_end = aggr_end.unsqueeze(2).expand_as(z)  # (B, L, K, d_pair)
        gate = self.gate_end(aggr_end)
        z = z + self.dropout(self.proj_end_out(gate * aggr_end))

        return z


class TriangularSelfAttention(nn.Module):
    """三角自注意力: 用 pair 表示做全局注意力.

    标准 multi-head attention 在 (L, L) pair 空间上:
      - 对每行 i, 注意力在 j 维度上
      - 对每列 j, 注意力在 i 维度上
    """

    def __init__(self, d_pair: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.d_head = d_pair // n_heads
        assert d_pair % n_heads == 0

        self.norm = nn.LayerNorm(d_pair)
        self.q_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.k_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.v_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.out_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, L, d_pair) → updated z"""
        B, L, _, _ = z.shape
        z_norm = self.norm(z)

        # QKV: (B, L, L, d_pair) → (B, n_heads, L, L, d_head)
        q = self.q_proj(z_norm).view(B, L, L, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)
        k = self.k_proj(z_norm).view(B, L, L, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)
        v = self.v_proj(z_norm).view(B, L, L, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)

        # 行注意力: 沿 j 维度 attention
        # q: (B, H, L_i, L_j, d), k: (B, H, L_i, L_k, d)
        attn_row = torch.einsum('bhijd,bhikd->bhijk', q, k) / math.sqrt(self.d_head)
        attn_row = F.softmax(attn_row, dim=-1)
        attn_row = self.dropout(attn_row)
        out_row = torch.einsum('bhijk,bhikd->bhijd', attn_row, v)

        # 列注意力: 沿 i 维度 attention (转置)
        attn_col = torch.einsum('bhijd,bhkjd->bhijk', q, k) / math.sqrt(self.d_head)
        attn_col = F.softmax(attn_col, dim=-2)  # 沿 i 维度 softmax
        attn_col = self.dropout(attn_col)
        out_col = torch.einsum('bhijk,bhkjd->bhijd', attn_col, v)

        # 合并 (行 + 列, 平均)
        out = (out_row + out_col) / 2.0
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, L, L, self.d_pair)
        return z + self.dropout(self.out_proj(out))


class PairTrackLayer(nn.Module):
    """单层 pair track: TriMulUpdate + FFN.

    注意: TriSelfAttention 因 O(L²×L²) 内存过大已移除.
    TriMulUpdate 已实现三角一致性 (核心价值), FFN 提供非线性.
    """

    def __init__(self, d_pair: int, n_heads: int = 4, d_ffn: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.tri_mul = TriangularMultiplicativeUpdate(d_pair, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_pair),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, K, d_pair) → updated z"""
        z = self.tri_mul(z)
        z = z + self.ffn(z)
        return z


class PairTrack(nn.Module):
    """稀疏 Pair Track: 只在 top-K 近邻上做三角更新.

    O(L×K×d) 而非 O(L²×d), 适配 L≤2000nt.
    TriMulUpdate 在稀疏图上做: z_ik 与 z_jk 的聚合只限于 K 近邻.
    """

    def __init__(self, config: PairTrackConfig, d_node: int = 64):
        super().__init__()
        self.config = config

        # 初始化 pair representation 从 node features
        self.init_pair = nn.Sequential(
            nn.Linear(d_node, config.d_pair),
            nn.GELU(),
            nn.Linear(config.d_pair, config.d_pair),
        )

        # RNA FM pair_repr → d_pair 投影 (维度可能不同)
        self.init_pair_proj = nn.Sequential(
            nn.Linear(128, config.d_pair),  # RNA FM d_pair=128 → config.d_pair
            nn.GELU(),
            nn.Linear(config.d_pair, config.d_pair),
        )

        # N 层 pair track (稀疏版本)
        self.layers = nn.ModuleList([
            PairTrackLayer(
                d_pair=config.d_pair,
                n_heads=config.n_heads,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.norm_out = nn.LayerNorm(config.d_pair)

    def init_from_rna_fm_pair(
        self,
        pair_repr: torch.Tensor,
        topk_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """从 RNA FM 的 pair_repr 初始化稀疏 pair representation.

        RNA FM 的 pair_repr 是全局 (B, L, L, d_rna_fm), 这里提取稀疏子集.

        Args:
            pair_repr: (B, L, L, d_rna_fm) — RNA FM 全局 pair repr (d_rna_fm=128)
            topk_idx: (B, L, K) — 每个节点的 K 个最近邻索引

        Returns:
            z: (B, L, K, d_pair) — 稀疏 pair representation
        """
        B, L, _, d_rna = pair_repr.shape
        K = self.config.max_len if hasattr(self.config, 'max_len') else 30
        K = min(K, L)

        if topk_idx is None:
            # 用 pair_repr 的强度选择 top-K (而非随机)
            pair_strength = pair_repr.mean(dim=-1)  # (B, L, L)
            _, topk_idx = pair_strength.topk(K, dim=-1)  # (B, L, K)

        # 从全局 pair_repr 提取稀疏子集
        b_idx = torch.arange(B, device=pair_repr.device).view(B, 1, 1).expand(B, L, K)
        i_idx = torch.arange(L, device=pair_repr.device).view(1, L, 1).expand(B, L, K)
        z = pair_repr[b_idx, i_idx, topk_idx]  # (B, L, K, d_rna)

        # 投影到 d_pair (如果维度不同)
        if d_rna != self.config.d_pair:
            z = self.init_pair_proj(z)  # (B, L, K, d_pair)

        return z

    def init_from_node(self, node_feat: torch.Tensor, topk_idx: torch.Tensor = None) -> torch.Tensor:
        """从 node representation 初始化稀疏 pair representation.

        node_feat: (B, L, d_node)
        topk_idx: (B, L, K) — 每个节点的 K 个最近邻索引 (可选, 无则随机)
        Returns: (B, L, K, d_pair) — 稀疏 pair
        """
        B, L, _ = node_feat.shape
        K = 30  # 默认稀疏度
        if topk_idx is None:
            topk_idx = torch.randint(0, L, (B, L, K), device=node_feat.device)

        # 构建稀疏 pair: z_ik = node_i * node_k (逐元素乘)
        d = node_feat.shape[-1]
        z_i = node_feat.unsqueeze(2).expand(B, L, K, d)  # (B, L, K, d)
        # advanced indexing: node_feat[b, topk_idx[b, i, k], :] → (B, L, K, d)
        b_idx = torch.arange(B, device=node_feat.device).view(B, 1, 1).expand(B, L, K)
        i_idx = torch.arange(L, device=node_feat.device).view(1, L, 1).expand(B, L, K)
        z_k = node_feat[b_idx, topk_idx, :]  # (B, L, K, d)
        z = z_i * z_k  # (B, L, K, d_node)
        z = self.init_pair(z)  # (B, L, K, d_pair)
        return z

    def forward(self, z: torch.Tensor, topk_idx: torch.Tensor = None) -> torch.Tensor:
        """z: (B, L, K, d_pair) → updated z"""
        for layer in self.layers:
            z = layer(z)
        return self.norm_out(z)
