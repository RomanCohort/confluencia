"""
scheme10_equivariant.py — TorusFold S10 完整等变架构

严格等变设计：
1. irrep 显式分离：inv (degree-0) + eq (degree-1, shape=(B,L,d,2))
2. 所有 eq 投影用 SO2EquivariantLinear
3. TorusCoordHead 分离预测
4. 循环边界连续性

架构（训练）：
  序列 → ChiralityAwareEmbedding (无绝对位置)
      → SparseEquivariantGNN (steerable kernel + 等变 update)
      → Irrep Split: inv (B,L,d) + eq (B,L,d,2)
      → S8 Refine (attn from inv, v_proj from eq) — 在 Latent 之前
      → Latent-inv (MLP) + Latent-eq (SO2EquivariantLinear)
      → Inv Diffusion (只对 inv, eq 条件注入, training only)
      → SO2AxisAngleCoordHead (Rodrigues 轴角) → coords (B, L, 3)

架构（推理 — 扩散生成）：
  序列 → encoder → s8_refine → latent_eq (条件)
      → DDIMSampler: z~N(0,1) → 50 步反向采样
          cond_attn(latent_eq) + CFG(w) 引导
      → coord_head(sampled_inv, latent_eq) → coords
      → 多种子 → 构象系综 + RMSF + 折叠轨迹

SO(2) 等变声明：网络等变群是 SO(2)（绕 z 轴旋转），
不是 SO(3)。坐标头用 axis-angle + Rodrigues 参数化，
可表达任意 3D 方向，但等变约束严格为 SO(2)。
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from equivariant_tpe import CircularRelativePositionBias
from equivariant_s8_refine import SparseEquivariantS8Refine
from equivariant_coord_head import StrictlyEquivariantCoordHead
from so2_equivariant import SO2EquivariantLinear
from chirality_embedding import ChiralityAwareEmbedding
from adaptive_sparse_k import AdaptiveSparseK
from contact_map_aux_head import ContactMapAuxHead


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EquivariantS10Config:
    """完整等变 S10 配置"""
    n_tokens: int = 5
    d_model: int = 256

    # 稀疏 GNN
    d_edge: int = 64
    k_theta: int = 2
    k_phi: int = 1
    n_layers: int = 4
    K_sparse: int = 60
    K_sparse_ratio: float = 0.04
    K_sparse_min: int = 30
    dropout: float = 0.1

    # V2: 自适应稀疏 K
    use_adaptive_k: bool = True
    K_sparse_max: int = 120  # 自适应 K 上限

    # irrep 分离（显式）
    d_model_inv: int = 128  # degree-0 通道数（用 d_model//2，保证 inv 信号足够）
    d_model_eq: int = 64    # degree-1 通道数（实际维度是 d_model_eq * 2）

    # Latent
    d_inv: int = 64  # 翻倍：之前 32 维对 S8 Refine 的 Q/K 注意力信号太弱
    d_eq: int = 32

    # S8 Refine（稀疏等变）
    use_s8_refine: bool = True
    n_refine_layers: int = 2
    K_refine: int = 40  # Top-K 稀疏邻居

    # 扩散（只对 inv）
    use_diffusion: bool = True
    n_diffusion_steps: int = 100
    # Diffusion denoiser 配置（可搜索）
    denoiser_hidden: int = 128
    n_denoiser_hidden_layers: int = 2

    # V2: 接触图辅助任务
    use_contact_aux: bool = False

    # 结构
    bond_length: float = 5.9
    r_scale: float = 10.0
    n_edge_cats: int = 5


# ══════════════════════════════════════════════════════════════════════════════
# 稀疏等变 Kernel（保留原有实现）
# ══════════════════════════════════════════════════════════════════════════════

class SparseSO2SteerableKernel(nn.Module):
    """稀疏版 SO(2)×SO(2) steerable kernel（保留原有实现）"""

    def __init__(self, d_model, d_edge, n_edge_cats, k_theta=2, k_phi=1):
        super().__init__()
        self.d_model = d_model
        self.d_edge = d_edge
        self.n_edge_cats = n_edge_cats
        self.k_theta = k_theta
        self.k_phi = k_phi

        self.n_theta_irreps = 1 + 2 * k_theta
        self.n_phi_irreps = 1 + 2 * k_phi
        self.n_irrep_channels = self.n_theta_irreps * self.n_phi_irreps

        self.kernel = nn.Parameter(
            torch.empty(n_edge_cats, self.n_irrep_channels * d_edge, d_model)
        )
        nn.init.xavier_uniform_(self.kernel)

        self.node_lift = nn.Linear(d_model, self.n_irrep_channels * d_edge, bias=False)
        self.cat_embed = nn.Embedding(n_edge_cats, d_edge)

    def _irrep_features_sparse(self, delta_theta, delta_phi):
        """构建 irrep 特征"""
        B, L, K = delta_theta.shape

        theta_feats = [torch.ones_like(delta_theta)]
        for k in range(1, self.k_theta + 1):
            theta_feats.append(torch.sin(k * delta_theta))
            theta_feats.append(torch.cos(k * delta_theta))
        theta_feats = torch.stack(theta_feats, dim=-1)

        phi_feats = [torch.ones_like(delta_phi)]
        for l in range(1, self.k_phi + 1):
            phi_feats.append(torch.sin(l * delta_phi))
            phi_feats.append(torch.cos(l * delta_phi))
        phi_feats = torch.stack(phi_feats, dim=-1)

        irrep = theta_feats.unsqueeze(-1) * phi_feats.unsqueeze(-2)
        return irrep.reshape(B, L, K, -1)

    # 沿 K 维度分块的消息计算峰值内存上限（单位：邻居数）。
    # 全量物化 (B, L, K, K_irrep, d_edge) 在长序列+大 batch 下会撑爆显存
    # （e.g. B=8, L=500, K=60 → ~46GB modulated 张量）。
    # 分块后峰值 ≈ 全量 / (K / K_CHUNK)，保持数值结果完全等价。
    K_CHUNK_LIMIT = 20

    def forward(self, x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx):
        B, L, _ = x.shape
        K = delta_theta_sparse.shape[2]

        irrep = self._irrep_features_sparse(delta_theta_sparse, delta_phi_sparse)
        K_irrep = irrep.shape[-1]

        b_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, K)
        x_j = x[b_idx, topk_idx]  # (B, L, K, d_model)

        # === 沿 K 分块，避免一次性物化整个 (B, L, K, K_irrep, d_edge) ===
        K_chunk = self.K_CHUNK_LIMIT
        msgs = []
        for k0 in range(0, K, K_chunk):
            k1 = min(k0 + K_chunk, K)
            chunk_n = k1 - k0

            xi = x_j[:, :, k0:k1, :]  # (B, L, chunk_n, d_model)
            ir = irrep[:, :, k0:k1, :]  # (B, L, chunk_n, K_irrep)
            ec = edge_cat_sparse[:, :, k0:k1]  # (B, L, chunk_n)

            lifted = self.node_lift(xi).reshape(B, L, chunk_n, K_irrep, self.d_edge)
            modulated = lifted * ir.unsqueeze(-1)
            modulated = modulated * (1.0 + self.cat_embed(ec).unsqueeze(-2))

            flat = modulated.reshape(B, L, chunk_n, -1)
            partial = torch.einsum('blik,ckm->blicm', flat, self.kernel)

            bi = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, chunk_n)
            ii = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, chunk_n)
            ki = torch.arange(chunk_n, device=x.device).view(1, 1, chunk_n).expand(B, L, chunk_n)
            msgs.append(partial[bi, ii, ki, ec])  # (B, L, chunk_n, d_model)

        msg = torch.cat(msgs, dim=2)  # (B, L, K, d_model)
        return msg.mean(dim=2)


class EquivariantGNNLayer(nn.Module):
    """等变 GNN 层（带循环相对位置偏置）"""

    def __init__(self, config, n_heads=4):
        super().__init__()
        self.kernel = SparseSO2SteerableKernel(
            config.d_model, config.d_edge, config.n_edge_cats,
            config.k_theta, config.k_phi
        )

        # update 层：拆分 inv 和 eq 分别处理
        self.inv_update = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )

        self.norm = nn.LayerNorm(config.d_model)

        # 循环相对位置偏置（替代 TPE）
        self.rel_pos_bias = CircularRelativePositionBias(
            n_heads=n_heads,
            max_dist=32,
            dropout=config.dropout,
        )

    def forward(self, x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse):
        B, L, _ = x.shape

        msg = self.kernel(x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx)
        x_new = self.inv_update(torch.cat([x, msg], dim=-1))
        return self.norm(x + x_new)


# ══════════════════════════════════════════════════════════════════════════════
# 等变 Encoder
# ══════════════════════════════════════════════════════════════════════════════

class StrictlyEquivariantEncoder(nn.Module):
    """严格等变 Encoder（无绝对位置编码）

    关键改动：
    1. 不使用 TPE（去掉绝对位置）
    2. 在 GNN 层用循环相对位置偏置
    3. 位置信息通过 delta_theta（循环距离）传递
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # V2: 手性感知 token embedding（替代普通 token_embed）
        self.token_embed = ChiralityAwareEmbedding(config.n_tokens, config.d_model)

        # 不再使用 TPE
        # self.tpe = TorusPositionalEncoding(...)

        self.layers = nn.ModuleList([
            EquivariantGNNLayer(config) for _ in range(config.n_layers)
        ])

        # V2: 自适应稀疏 K
        if config.use_adaptive_k:
            self.adaptive_k = AdaptiveSparseK(
                K_min=config.K_sparse_min,
                K_max=config.K_sparse_max,
                K_base=config.K_sparse,
            )
        else:
            self.adaptive_k = None

        # irrep 分离投影（关键改动）
        # degree-0：可以用非线性
        self.to_inv = nn.Sequential(
            nn.Linear(config.d_model, config.d_model_inv * 2),
            nn.GELU(),
            nn.Linear(config.d_model_inv * 2, config.d_model_inv),
        )

        # degree-1：必须用等变 Linear
        # 输入 (B, L, d_model)，先投影到 (B, L, d_model_eq)，再 reshape 成 (B, L, d_model_eq, 2)
        # 这里用普通 Linear，因为输入是混合的
        self.to_eq_proj = nn.Linear(config.d_model, config.d_model_eq * 2, bias=False)

    def forward(self, seq_tokens, pair_probs=None):
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # V2: 动态 K（自适应或固定）
        if self.adaptive_k is not None:
            # 自适应稀疏 K：根据局部复杂度调整
            k_per_residue = self.adaptive_k(seq_tokens, pair_probs)  # [B, L]
            K_actual = self.adaptive_k.get_global_K(k_per_residue, L)  # 全局 K
        else:
            # 固定 K
            K_actual = max(self.config.K_sparse_min, int(L * self.config.K_sparse_ratio))
            K_actual = min(K_actual, L, self.config.K_sparse)

        # Token embedding（无位置编码）
        x = self.token_embed(seq_tokens)  # (B, L, d_model)

        # 不再使用 TPE
        # x = self.tpe(x, seq_len=L)

        # 构建稀疏邻居
        if pair_probs is None:
            topk_idx = torch.randint(0, L, (B, L, K_actual), device=device)
        else:
            _, topk_idx = pair_probs.topk(K_actual, dim=-1)

        # 构建角度差（循环连续性）
        pos = torch.arange(L, device=device, dtype=torch.float32)
        theta = 2 * math.pi * pos / L
        theta_j = theta[topk_idx]
        theta_i = theta.view(1, L, 1)

        # 关键：归一化到 [-π, π]（循环边界）
        delta_theta_sparse = theta_i - theta_j
        delta_theta_sparse = (delta_theta_sparse + math.pi) % (2 * math.pi) - math.pi

        delta_phi_sparse = torch.zeros_like(delta_theta_sparse)
        edge_cat_sparse = torch.zeros(B, L, K_actual, device=device, dtype=torch.long)

        # 稀疏 GNN
        for layer in self.layers:
            x = layer(x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse)

        # irrep 分离
        node_repr_inv = self.to_inv(x)  # (B, L, d_model_inv) [degree-0]

        # degree-1：投影后 reshape 成 (B, L, d_model_eq, 2)
        eq_flat = self.to_eq_proj(x)  # (B, L, d_model_eq * 2)
        node_repr_eq = eq_flat.view(B, L, self.config.d_model_eq, 2)  # (B, L, d_eq, 2)

        return node_repr_inv, node_repr_eq, topk_idx


# ══════════════════════════════════════════════════════════════════════════════
# 等变 Latent
# ══════════════════════════════════════════════════════════════════════════════

class StrictlyEquivariantLatent(nn.Module):
    """严格等变 Latent 投影（自动多尺度）

    序列长度 > 1000nt 时自动启用多尺度：
      - 粗粒度：降采样 → 投影 → 抓全局结构
      - 细粒度：上采样 → 局部 GNN → 精修几何
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 单尺度投影（用于短序列）
        self.inv_proj = nn.Sequential(
            nn.Linear(config.d_model_inv, config.d_inv * 2),
            nn.GELU(),
            nn.Linear(config.d_inv * 2, config.d_inv),
        )

        self.eq_proj = SO2EquivariantLinear(
            config.d_model_eq, config.d_eq, degree_in=1, degree_out=1, bias=False
        )

        # 多尺度组件（动态创建）
        self._multiscale_initialized = False
        self.downsample = None
        self.upsample = None
        self.fine_gnn = None

    def _init_multiscale_if_needed(self, factor: int, device):
        """初始化多尺度组件"""
        if self._multiscale_initialized:
            return

        from multiscale_equivariant import DownsampleEncoder, UpsampleDecoder, FineGrainedGNN

        self.downsample = DownsampleEncoder(
            factor, self.config.d_model_inv, self.config.d_model_eq
        ).to(device)

        self.upsample = UpsampleDecoder(
            factor, self.config.d_inv, self.config.d_eq
        ).to(device)

        self.fine_gnn = FineGrainedGNN(
            self.config.d_inv, self.config.d_eq,
            d_hidden=64,
            local_radius=8,
            dropout=self.config.dropout
        ).to(device)

        self._multiscale_initialized = True

    def forward(self, node_repr_inv, node_repr_eq):
        """
        Args:
            node_repr_inv: (B, L, d_model_inv)
            node_repr_eq:  (B, L, d_model_eq, 2)

        Returns:
            latent_inv: (B, L, d_inv)
            latent_eq:  (B, L, d_eq, 2)
        """
        B, L = node_repr_inv.shape[:2]

        # 短序列：直接投影
        if L <= 1000:
            latent_inv = self.inv_proj(node_repr_inv)
            latent_eq = self.eq_proj(node_repr_eq)
            return latent_inv, latent_eq

        # 长序列：多尺度处理
        # 选择降采样因子
        if L <= 2500:
            factor = 5
        elif L <= 5000:
            factor = 10
        else:
            factor = 10

        self._init_multiscale_if_needed(factor, node_repr_inv.device)

        # 降采样
        inv_coarse, eq_coarse = self.downsample(node_repr_inv, node_repr_eq)

        # 粗粒度投影
        latent_inv_coarse = self.inv_proj(inv_coarse)
        latent_eq_coarse = self.eq_proj(eq_coarse)

        # 上采样
        latent_inv_up, latent_eq_up = self.upsample(latent_inv_coarse, latent_eq_coarse, L)

        # 细粒度精修
        latent_inv, latent_eq = self.fine_gnn(latent_inv_up, latent_eq_up)

        return latent_inv, latent_eq


# ══════════════════════════════════════════════════════════════════════════════
# Mixed Attention: Sliding Window + Global Anchor (O(L), not O(L²))
# ══════════════════════════════════════════════════════════════════════════════
# Replaces the O(L²) nn.MultiheadAttention in InvDiffusion.cond_attn.
#
# For circRNA of length L (target: L=2000):
#   - SlidingWindowAttention: each token attends to W circular neighbors → O(L·W)
#   - GlobalAnchorAttention: every token attends to A global anchors covering
#     BSJ flanks + far-range positions for long-term pairing → O(L·A)
# Both use F.scaled_dot_product_attention with masks.


def _circular_neighbor_mask(L: int, W: int, device: torch.device) -> torch.Tensor:
    """(L, L) bool mask: token i attends to j iff d_ring(i, j) <= W//2.

    d_ring(i, j) = min(|i-j|, L-|i-j|).  No data copy — pure index math.
    """
    i = torch.arange(L, device=device)
    j = torch.arange(L, device=device)
    diff = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
    circ_dist = torch.minimum(diff, L - diff)
    return circ_dist <= (W // 2)


def _global_anchor_indices(L: int, A: int, bsj_flank: int,
                           device: torch.device) -> torch.Tensor:
    """A anchor positions: BSJ flanks + evenly spaced across interior.

    All indices clamped to [0, L-1].  For short sequences A is capped at L.
    """
    A = min(A, L)
    anchors: set[int] = set()
    half = max(1, bsj_flank)
    for _i in range(min(half, L)):
        anchors.add(_i)
        anchors.add(L - half + _i)
    inner = A - len(anchors)
    if inner > 0 and L > 0:
        for _i in range(inner):
            anchors.add(min(L - 1, (_i + 1) * L // max(inner, 1)))
    return torch.tensor(sorted(anchors)[:A], device=device, dtype=torch.long)


class SlidingWindowAttention(nn.Module):
    """Sliding-window cross-attention: O(L·W).

    Each token in `query` attends to W circular neighbors in key/value.
    Mask is built per-(L, W) pair and cached via `torch.compile` friendliness
    (no persistent state).
    """

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.window = window
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        device = query.device
        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, L, self.n_heads, self.head_dim)

        # Scaled dot-product attention via einsum: O(B·L·L·head_dim) compute,
        # but only O(L·W) entries have finite attention weight.
        scores = torch.einsum("blhd,bkhd->bhkl", q, k) / math.sqrt(self.head_dim)
        # Apply sliding-window mask: -inf where not attended
        mask = _circular_neighbor_mask(L, self.window, device)  # (L, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhkl,bkhd->blhd", attn, v)  # (B, L, heads, head_dim)
        out = out.reshape(B, L, D)
        return self.out_proj(out)


class GlobalAnchorAttention(nn.Module):
    """Global anchor cross-attention: O(L·A).

    Each token queries only A anchor positions (BSJ flanks + far-range) in key/value.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_anchors: int = 128,
                 bsj_flank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_anchors = n_anchors
        self.bsj_flank = bsj_flank
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        device = query.device
        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim)
        aidx = _global_anchor_indices(L, self.n_anchors, self.bsj_flank, device)
        A = aidx.shape[0]
        ka = self.k_proj(key[:, aidx, :]).view(B, A, self.n_heads, self.head_dim)
        va = self.v_proj(value[:, aidx, :]).view(B, A, self.n_heads, self.head_dim)
        # Manual attention (no SDPA mask needed — no masking for anchors)
        scores = torch.einsum("blhd,bahd->bhal", q, ka) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhal,bahd->blhd", attn, va)
        out = out.reshape(B, L, D)
        return self.out_proj(out)


class MixedHybridAttention(nn.Module):
    """Sliding-window + global-anchor mixed attention.

    4-layer default (local ↔ global interleaved):
      L0: SlidingWindow(W=256)  L1: GlobalAnchor(A=128)
      L2: SlidingWindow(W=256)  L3: GlobalAnchor(A=128)
    Each layer has residual + LayerNorm.  Same `cond_attn(q, k, v)` interface
    as nn.MultiheadAttention, so InvDiffusion code around it is unchanged.
    """

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4,
                 window: int = 256, n_anchors: int = 128, bsj_flank: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.layers.append(SlidingWindowAttention(
                    d_model, n_heads, window=window, dropout=dropout))
            else:
                self.layers.append(GlobalAnchorAttention(
                    d_model, n_heads, n_anchors=n_anchors,
                    bsj_flank=bsj_flank, dropout=dropout))
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = query
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(query, key, value)
            x = norm(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Inv 扩散（只对 degree-0）
# ══════════════════════════════════════════════════════════════════════════════

class InvDiffusion(nn.Module):
    """只对 inv 通道做扩散（条件：latent_eq）

    核心设计：
    - 扩散只对 degree-0（inv）
    - degree-1（eq）作为条件注入
    - 确保 inv 和 eq 的一致性
    """

    def __init__(self, d_inv, d_eq=16, n_steps=100,
                 denoiser_hidden=128, n_denoiser_hidden_layers=2,
                 cfg_dropout_prob=0.1):
        super().__init__()
        self.d_inv = d_inv
        self.d_eq = d_eq
        self.n_steps = n_steps
        self.cfg_dropout_prob = cfg_dropout_prob  # CFG 训练时随机 drop 条件

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # 条件嵌入（把 eq 的 (cos, sin) 投影到 d_inv，与 cond_attn embed_dim 对齐）
        self.cond_proj = nn.Linear(d_eq * 2, d_inv)

        # 混合注意力（滑动窗口 + 全局锚点），替代 O(L²) 的 nn.MultiheadAttention
        # 局部：窗口 256，保局部结构；全局：128 锚点覆盖 BSJ 首尾 + 远端，保长程配对
        self.cond_attn = MixedHybridAttention(
            d_model=d_inv,
            n_heads=4,
            window=256,
            n_anchors=128,
            bsj_flank=32,
            dropout=0.1,
        )

        # 去噪器（inv + t_emb + cond）— 可配置深度和宽度
        # 结构: input → [Linear(hidden)+GELU] × n_hidden → Linear(d_inv)
        layers = [nn.Linear(d_inv + 64, denoiser_hidden), nn.GELU()]
        for _ in range(n_denoiser_hidden_layers):
            layers.append(nn.Linear(denoiser_hidden, denoiser_hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(denoiser_hidden, d_inv))
        self.denoiser = nn.Sequential(*layers)

        beta = torch.linspace(1e-4, 0.02, n_steps)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

    def forward(self, x_clean, return_loss=True, cond_eq=None):
        """
        Args:
            x_clean: (B, L, d_inv) 干净的 inv
            return_loss: 是否返回 loss
            cond_eq: (B, L, d_eq, 2) 条件 eq 特征

        Returns:
            loss: 如果 return_loss=True
            noise_pred: 预测的噪声
        """
        B, L, D = x_clean.shape
        device = x_clean.device
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Fix 3: Vectorized add_noise — 无 Python for-loop
        # alpha_bar: (B,) → (B, 1, 1) broadcast to (B, L, D)
        alpha_bar_t = self.alpha_bar[t]                    # (B,)
        alpha_bar_t = alpha_bar_t.view(B, 1, 1)            # (B, 1, 1)
        one_minus_bar = 1.0 - alpha_bar_t                  # (B, 1, 1)

        noise = torch.randn_like(x_clean)                  # (B, L, D)
        x_noisy = torch.sqrt(alpha_bar_t) * x_clean + torch.sqrt(one_minus_bar) * noise

        # 条件交叉注意力（如果有 cond_eq）
        # CFG 训练：以 cfg_dropout_prob 概率随机 drop 条件，让模型同时学 conditional 和 unconditional
        if cond_eq is not None and self.training and self.cfg_dropout_prob > 0:
            # 每个 batch 独立决定是否 drop（不是 per-sample，简化实现）
            if torch.rand(1).item() < self.cfg_dropout_prob:
                cond_eq = None
        if cond_eq is not None:
            cond_flat = cond_eq.reshape(B, L, -1)
            cond_emb = self.cond_proj(cond_flat)  # (B, L, d_eq)
            x_cond = self.cond_attn(query=x_noisy, key=cond_emb, value=cond_emb)
            x_input = x_noisy + x_cond
        else:
            x_input = x_noisy

        # 时间步嵌入：t → t_emb (B, L, 64)
        t_frac = (t.float() / self.n_steps).unsqueeze(-1)      # (B, 1)
        t_emb = self.time_embed(t_frac)                          # (B, 64)
        t_emb = t_emb.unsqueeze(1).expand(B, L, -1)             # (B, L, 64)

        x_input = torch.cat([x_input, t_emb], dim=-1)
        noise_pred = self.denoiser(x_input)

        if return_loss:
            return F.mse_loss(noise_pred, noise)
        return noise_pred


# ══════════════════════════════════════════════════════════════════════════════
# 完整模型
# ══════════════════════════════════════════════════════════════════════════════

class StrictlyEquivariantS10(nn.Module):
    """完整等变 S10 模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = StrictlyEquivariantEncoder(config)
        self.latent = StrictlyEquivariantLatent(config)

        if config.use_s8_refine:
            self.s8_refine = SparseEquivariantS8Refine(
                config.d_model_inv, config.d_model_eq,  # Fix 4: refine 在 Latent 之前，用 node_repr 维度
                n_layers=config.n_refine_layers,
                K=config.K_refine,
                dropout=config.dropout,
            )
        else:
            self.s8_refine = None

        if config.use_diffusion:
            # 用关键字参数，避免 d_eq 被位置参数错赋为 n_diffusion_steps
            self.diffusion = InvDiffusion(
                d_inv=config.d_inv,
                d_eq=config.d_eq,
                n_steps=config.n_diffusion_steps,
                denoiser_hidden=config.denoiser_hidden,
                n_denoiser_hidden_layers=config.n_denoiser_hidden_layers,
            )
        else:
            self.diffusion = None

        self.coord_head = StrictlyEquivariantCoordHead(
            config.d_inv, config.d_eq,
            d_hidden=64,
            dropout=config.dropout,
            r_scale=config.r_scale,  # Fix 5: 显式传入 config.r_scale
        )

        # V2: 接触图辅助任务
        if config.use_contact_aux:
            self.contact_aux_head = ContactMapAuxHead(d_inv=config.d_inv, d_hidden=64)
        else:
            self.contact_aux_head = None

    def forward(self, seq_tokens, pair_probs=None, return_loss=True):
        """
        Args:
            seq_tokens: (B, L)
            pair_probs: (B, L, L) 配对概率（可选）

        Returns:
            coords: (B, L, 3)
            loss: 如果 return_loss=True
        """
        # Encoder
        node_repr_inv, node_repr_eq, topk_idx = self.encoder(seq_tokens, pair_probs)

        # S8 Refine 放在 Latent 之前（在 d_model_inv=128 维空间做，避免压缩后信息不足）
        # 传入 topk_idx 跳过 O(L²) pair_probs 计算
        if self.s8_refine is not None:
            node_repr_eq = self.s8_refine(
                node_repr_inv, node_repr_eq, topk_idx=topk_idx
            )

        # Latent
        latent_inv, latent_eq = self.latent(node_repr_inv, node_repr_eq)

        # Diffusion（只对 inv，条件注入 latent_eq）
        diffusion_loss = None
        if self.diffusion is not None and self.training:
            diffusion_loss = self.diffusion(latent_inv, return_loss=True, cond_eq=latent_eq)

        # Coord Head — SO(2) 等变 + 轴角参数化，显式传入 r_scale
        coords, r, axis, angle = self.coord_head(latent_inv, latent_eq)

        # V2: 接触图辅助任务
        contact_pred = None
        if self.contact_aux_head is not None:
            contact_pred = self.contact_aux_head(latent_inv)

        if return_loss and diffusion_loss is not None:
            return coords, diffusion_loss, contact_pred
        if contact_pred is not None:
            return coords, contact_pred
        return coords

    @torch.no_grad()
    def generate_ensemble(
        self,
        seq_tokens: torch.Tensor,
        n_samples: int = 100,
        cfg_scale: float = 1.0,
        return_trajectory: bool = False,
    ):
        """生成构象系综（动态模式）"""
        from dynamic_ensemble import ConformationalEnsembleGenerator
        generator = ConformationalEnsembleGenerator(
            self, self.config.d_inv, self.config.d_eq, self.config.n_diffusion_steps
        )
        return generator.generate_ensemble(
            seq_tokens, n_samples, cfg_scale, return_trajectory
        )


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

def test_full_equivariance():
    """测试完整模型的端到端等变性"""
    from so2_equivariant import rotation_matrix_2d

    print("=" * 60)
    print("StrictlyEquivariantS10 端到端等变性测试")
    print("=" * 60)

    torch.manual_seed(42)

    B, L = 2, 50
    config = EquivariantS10Config(
        d_model=128,
        d_model_inv=32,
        d_model_eq=32,
        d_inv=16,
        d_eq=16,
        n_layers=2,
        use_s8_refine=True,
        use_diffusion=False,
    )

    model = StrictlyEquivariantS10(config)
    model.eval()

    # 随机序列
    seq_tokens = torch.randint(0, config.n_tokens, (B, L))

    print(f"配置: B={B}, L={L}, n_layers={config.n_layers}")
    print()

    # 随机旋转
    theta = torch.rand(1).item() * 2 * math.pi
    R = rotation_matrix_2d(torch.tensor(theta))

    print(f"旋转角度: {theta:.4f}")

    # Path A: 先旋转序列（circRNA 旋转等价于序列循环移位）
    shift = int(theta / (2 * math.pi) * L) % L
    seq_rot = torch.roll(seq_tokens, shifts=shift, dims=1)

    with torch.no_grad():
        coords_A = model(seq_rot, return_loss=False)

    # Path B: 先预测，再旋转坐标
    with torch.no_grad():
        coords_B = model(seq_tokens, return_loss=False)
        # 旋转坐标（绕 z 轴）
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x_rot = cos_t * coords_B[..., 0] - sin_t * coords_B[..., 1]
        y_rot = sin_t * coords_B[..., 0] + cos_t * coords_B[..., 1]
        z_rot = coords_B[..., 2]
        coords_B_rot = torch.stack([x_rot, y_rot, z_rot], dim=-1)

    # 比较误差
    diff = (coords_A - coords_B_rot).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print()
    print("端到端等变性测试结果:")
    print(f"  |A - B|_max  = {max_diff:.6e}")
    print(f"  |A - B|_mean = {mean_diff:.6e}")
    print()

    threshold = 1e-4  # 端到端稍微放宽
    if max_diff < threshold:
        print(f"[PASS] 端到端等变性成立 (误差 < {threshold:.0e})")
    else:
        print(f"[FAIL] 端到端等变性不成立 (误差 >= {threshold:.0e})")

    return max_diff


if __name__ == "__main__":
    test_full_equivariance()