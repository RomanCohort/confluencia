"""
equivariant_s8_refine.py — 稀疏等变 S8 Refine（O(L·K) 复杂度）

核心设计：
1. 用 ViennaRNA 配对概率选 Top-K 邻居
2. 只对邻居计算注意力（稀疏，O(L·K)）
3. 对邻居的 eq 特征做加权平均（保等变）
4. 循环连续（角度距离归一化）

输入：
  - node_repr_inv: (B, L, d_inv)  [degree 0]
  - node_repr_eq:  (B, L, d_eq, 2) [degree 1]
  - pair_probs: (B, L, L) 配对概率（可选）

输出：
  - refined_eq: (B, L, d_eq, 2)   [degree 1]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equivariant import SO2EquivariantLinear


class SparseEquivariantS8RefineLayer(nn.Module):
    """稀疏等变注意力层（O(L·K) 复杂度）"""

    def __init__(self, d_inv: int, d_eq: int, K: int = 40, dropout: float = 0.1):
        super().__init__()
        self.K = K

        # Q, K 投影（用 inv 算注意力）
        self.q_proj = nn.Linear(d_inv, d_inv, bias=False)
        self.k_proj = nn.Linear(d_inv, d_inv, bias=False)

        # V 投影（等变）
        self.v_proj = SO2EquivariantLinear(d_eq, d_eq, degree_in=1, degree_out=1, bias=False)

        # 输出投影（等变）
        self.out_proj = SO2EquivariantLinear(d_eq, d_eq, degree_in=1, degree_out=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_repr_inv: torch.Tensor, node_repr_eq: torch.Tensor,
                pair_probs: torch.Tensor = None, topk_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_repr_inv: (B, L, d_inv)
            node_repr_eq: (B, L, d_eq, 2)
            pair_probs: (B, L, L) 配对概率（可选）
            topk_idx: (B, L, K) 预计算的稀疏邻居索引（可选，避免 O(L²) 计算）

        Returns:
            refined_eq: (B, L, d_eq, 2)
        """
        B, L, _ = node_repr_inv.shape
        K = min(self.K, L)

        # 1. 选 Top-K 邻居
        # Fix 1: 优先用预计算的 topk_idx（来自 encoder 的 O(L·K) 计算）
        if topk_idx is not None:
            K = min(K, topk_idx.shape[2])  # 不能比预计算的邻居数多
            topk_idx = topk_idx[:, :, :K]
        elif pair_probs is not None:
            _, topk_idx = pair_probs.topk(K, dim=-1)
        else:
            # 默认用角度距离（循环连续）
            pos = torch.arange(L, device=node_repr_inv.device, dtype=torch.float32)
            theta = 2 * math.pi * pos / L
            theta_i = theta.view(L, 1)
            theta_j = theta.view(1, L)
            delta = (theta_i - theta_j + math.pi) % (2 * math.pi) - math.pi
            circ_dist = torch.min(delta.abs(), 2 * math.pi - delta.abs())
            pair_probs = -circ_dist.unsqueeze(0).expand(B, L, L)
            _, topk_idx = pair_probs.topk(K, dim=-1)

        # 2. 稀疏注意力
        q = self.q_proj(node_repr_inv)  # (B, L, d_inv)
        k = self.k_proj(node_repr_inv)  # (B, L, d_inv)

        # Gather 邻居
        b_idx = torch.arange(B, device=q.device).view(B, 1, 1).expand(B, L, K)
        l_idx = torch.arange(L, device=q.device).view(1, L, 1).expand(B, L, K)

        q_local = q.unsqueeze(2).expand(B, L, K, -1)  # (B, L, K, d_inv)
        k_neighbors = k[b_idx, topk_idx]  # (B, L, K, d_inv)

        # 注意力分数
        attn = torch.einsum('blkd,blkd->blk', q_local, k_neighbors) / math.sqrt(q_local.shape[-1])  # (B, L, K)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 3. 对邻居的 eq 做加权
        v = self.v_proj(node_repr_eq)  # (B, L, d_eq, 2)
        v_neighbors = v[b_idx, topk_idx]  # (B, L, K, d_eq, 2)

        # 加权平均（线性组合，保等变）
        out = torch.einsum('blk,blkeq->bleq', attn, v_neighbors)  # (B, L, d_eq, 2)

        # 4. 输出投影 + 残差
        out = self.out_proj(out)
        return node_repr_eq + self.dropout(out)


class SparseEquivariantS8Refine(nn.Module):
    """多层稀疏等变 S8 Refine"""

    def __init__(self, d_inv: int, d_eq: int, n_layers: int = 2, K: int = 40, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseEquivariantS8RefineLayer(d_inv, d_eq, K, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, node_repr_inv: torch.Tensor, node_repr_eq: torch.Tensor,
                pair_probs: torch.Tensor = None, topk_idx: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            node_repr_eq = layer(node_repr_inv, node_repr_eq, pair_probs=pair_probs, topk_idx=topk_idx)
        return node_repr_eq


# 兼容旧名
StrictlyEquivariantS8RefineLayer = SparseEquivariantS8RefineLayer
StrictlyEquivariantS8Refine = SparseEquivariantS8Refine


if __name__ == "__main__":
    # 快速测试
    print("稀疏等变 S8 Refine 测试")
    print("=" * 60)

    torch.manual_seed(42)
    B, L, d_inv, d_eq = 2, 100, 32, 16

    model = SparseEquivariantS8Refine(d_inv, d_eq, n_layers=2, K=20, dropout=0.0)
    model.eval()

    inv = torch.randn(B, L, d_inv)
    eq = torch.randn(B, L, d_eq, 2)

    print(f"输入: inv={inv.shape}, eq={eq.shape}")

    with torch.no_grad():
        out = model(inv, eq)

    print(f"输出: {out.shape}")
    print("[OK] 前向传播成功")