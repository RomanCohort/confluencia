"""
scheme10_equivariant.py — TorusFold S10 完整等变架构 (v4: 坐标扩散)

严格等变设计：
1. irrep 显式分离：inv (degree-0) + eq (degree-1, shape=(B,L,d,2))
2. 所有 eq 投影用 SO2EquivariantLinear

架构（训练 — AF3 式坐标扩散）：
  序列 → ChiralityAwareEmbedding (无绝对位置)
      → SparseEquivariantGNN (steerable kernel + 等变 update)
      → Irrep Split: inv (B,L,d) + eq (B,L,d,2)
      → S8 Refine (attn from inv, v_proj from eq) — 在 Latent 之前
      → Latent-inv (MLP) + Latent-eq (SO2EquivariantLinear)
      → CoordDiffusion: (B,L,3) 坐标直接去噪 (条件=inv+eq) → coords

架构（推理 — 坐标扩散 DDIM）：
  序列 → encoder → s8_refine → latent_inv + latent_eq
      → CoordDiffusion.generate: x_T~N(0,I) → DDIM (20 steps) → x_0 (B,L,3)
      → 多种子 → 构象系综 + RMSF + 折叠轨迹

关键变化（v3→v4）：
  - 废弃 InvDiffusion（特征空间扩散） + SO2AxisAngleCoordHead（Rodrigues 几何头）
  - 改为 CoordDiffusion（3D 坐标空间扩散）— 训练/推理靶子都是 (B,L,3) 坐标
  - denoiser 输出直接是 3D 坐标噪声预测，loss 是 MSE on (B,L,3)

SO(2) 等变声明：网络等变群是 SO(2)（绕 z 轴旋转），
不是 SO(3)。3D 坐标中 (x,y) 为 degree-1, z 为 degree-0，
diffusion 按此度分离处理以保持等变性。
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
from coord_diffusion import CoordDiffusion
from physics_refine import refine_coords
from so2_equivariant import SO2EquivariantLinear
from chirality_embedding import ChiralityAwareEmbedding
from adaptive_sparse_k import AdaptiveSparseK
from contact_map_aux_head import ContactMapAuxHead


# ══════════════════════════════════════════════════════════════════════════════
# CRBPSA hydrogen-bond weighting (A-U=2, G-C=3, G-U=0.8)
# ══════════════════════════════════════════════════════════════════════════════
# Indexed by the ACTUAL token encoding used in train_s10_curriculum.py collate:
#     A=0, U=1, G=2, C=3, N/pad=4   (n_tokens=5)
# Real Watson-Crick pairs (AU/GC) outrank the G-U wobble (0.8) and non-pairing
# (1.0), so the LAMA anchor hotspot concentrates on genuine stems instead of
# raw pairing-probability peaks. Symmetric matrix.
HBOND_MAP = torch.tensor([
    #            A(0)   U(1)   G(2)   C(3)   N(4)
    [1.0, 2.0, 1.0, 1.0, 1.0],  # A-*    (A-U=2)
    [2.0, 1.0, 0.8, 1.0, 1.0],  # U-*    (U-A=2, U-G=0.8)
    [1.0, 0.8, 1.0, 3.0, 1.0],  # G-*    (G-U=0.8, G-C=3)
    [1.0, 1.0, 3.0, 1.0, 1.0],  # C-*    (C-G=3)
    [1.0, 1.0, 1.0, 1.0, 1.0],  # N-*    (pad/unknown, neutral)
], dtype=torch.float32)


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

    # 坐标扩散（v4：直接在 3D 坐标空间做扩散，替代特征空间 InvDiffusion）
    use_coord_diffusion: bool = True
    n_diffusion_steps: int = 100
    d_coord_hidden: int = 128  # 坐标扩散内部特征维度
    cfg_dropout_prob: float = 0.1  # CFG 训练时 drop 条件的概率

    # v4.1: 动态锚点选择（AF3 思想，基于 pair_probs 热点）
    use_dynamic_anchors: bool = True  # 默认打开：DynamicGlobalAnchorAttention
    # [v5] 长序列动态扩展: A = max(n_anchors, L*ratio)。固定 128 在 L>1280nt
    # 时覆盖率不足，长 circRNA 会漏关键折叠信号。
    anchor_ratio: float = 0.1

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
            K_actual = min(K_actual, L)  # clamp: get_global_K 下限是 K_min，
                                         # L<K_min 的短序列会 topk 越界（真实 bug）
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
# Mixed Attention: now in mixed_attention.py (shared with coord_diffusion.py)
# ══════════════════════════════════════════════════════════════════════════════
# Moved out to avoid circular import: coord_diffusion imports MixedHybridAttention,
# and scheme10_equivariant imports CoordDiffusion from coord_diffusion.
# MixedHybridAttention is used internally by CoordDiffusion (in coord_diffusion.py).
from mixed_attention import (
    SlidingWindowAttention,
    GlobalAnchorAttention,
    MixedHybridAttention,
)

__all__ = ["MixedHybridAttention", "SlidingWindowAttention", "GlobalAnchorAttention"]


# ══════════════════════════════════════════════════════════════════════════════
# InvDiffusion (deprecated) — replaced by CoordDiffusion in v4.
#
# The old path denoised a (B,L,64) latent feature vector, then passed the
# result through a Rodrigues-axis-angle coord_head to recover 3D coords.
# This was a two-hop prediction: diffusion didn't see 3D geometry at all,
# so it couldn't be held accountable for coordinate errors.
#
# Replaced by CoordDiffusion which directly denoises (B,L,3) coordinates
# in the same way as AF3's structure module.
# ══════════════════════════════════════════════════════════════════════════════


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
                config.d_model_inv, config.d_model_eq,
                n_layers=config.n_refine_layers,
                K=config.K_refine,
                dropout=config.dropout,
            )
        else:
            self.s8_refine = None

        # v4: 坐标扩散直接对 (B,L,3) 做去噪，替代 InvDiffusion + coord_head
        if config.use_coord_diffusion:
            self.coord_diffusion = CoordDiffusion(
                d_inv=config.d_inv,
                d_eq=config.d_eq,
                d_coord_hidden=config.d_coord_hidden,
                n_steps=config.n_diffusion_steps,
                cfg_dropout_prob=config.cfg_dropout_prob,
                use_dynamic_anchors=config.use_dynamic_anchors,
                anchor_ratio=getattr(config, 'anchor_ratio', 0.1),
            )
        else:
            self.coord_diffusion = None

        # V2: 接触图辅助任务
        # Always instantiated (cheap). Used when use_contact_aux OR during the
        # Phase-1 stop-grad window — there all p_denorm-based supervision is cut,
        # so this latent-direct structural head is the Encoder's only geometry signal.
        self.contact_aux_head = ContactMapAuxHead(d_inv=config.d_inv, d_hidden=64)

        # [v4] Stop-Gradient toggle for latent→diffusion edge.
        # When True, geometric losses on x0_pred cannot backprop into the Encoder
        # via the diffusion path — Encoder learns structure from anchor_aux /
        # contact_aux self-supervision instead of being pulled by coordinate fits.
        # anchor_aux_loss still uses the un-detached latent (separate path) so the
        # scorer keeps learning. Toggled per-phase by the training loop.
        self.detach_latent = False

        # [v4] Kendall uncertainty weighting — learnable log σ² per conflicting loss term.
        # Uniform init (log_var=0 → σ²=1, weight=0.5) so the model discovers the balance
        # itself rather than inheriting the hand-tuned LOSS_WEIGHTS prior.
        # Applies to the 6 geometry+physics core terms that are prone to gradient conflict;
        # diffusion/torus/chirality/contrastive/distillation keep fixed weights.
        self.uncertainty_log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1)) for k in
            ['coord', 'bond', 'stereo', 'physics_pairing', 'contact_aux', 'physics_bridge']
        })

    def forward(
        self,
        seq_tokens: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
        pair_probs: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        return_coords: bool = False,
        lengths: Optional[torch.Tensor] = None,
        refine: bool = False,
        refine_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        v4: 训练时走坐标扩散，推理时走 DDIM 生成。

        Args:
            seq_tokens    : (B, L)
            target_coords : (B, L, 3) 训练时传入真实坐标（用于扩散 loss）
            pair_probs    : (B, L, L) 配对概率（可选）
            return_loss   : 训练时返回 diffusion loss
            return_coords : 训练时是否额外返回 coords（默认只返回 loss）
            lengths       : (B,) 有效长度。推理精修必需（padding 不动）
            refine        : 推理时是否跑 AF3 式轻量物理精修（键长/键角/位阻/二面角）
            refine_steps  : 精修步数（默认 100，比 20 更充分吸收局部 clash）

        Returns:
            - 训练: (diffusion_loss, pred_coords_or_None, contact_pred)
            - 推理: (pred_coords,)
        """
        # Encoder
        node_repr_inv, node_repr_eq, topk_idx = self.encoder(seq_tokens, pair_probs)

        # S8 Refine（在 Latent 之前）
        if self.s8_refine is not None:
            node_repr_eq = self.s8_refine(
                node_repr_inv, node_repr_eq, topk_idx=topk_idx,
            )

        # Latent
        latent_inv, latent_eq = self.latent(node_repr_inv, node_repr_eq)

        # ── 坐标扩散 ────────────────────────────────────────────
        diffusion_loss = None
        anchor_aux_loss = None
        pred_coords = None

        if self.coord_diffusion is not None:
            # [v4] Stop-Gradient: when detach_latent=True, the diffusion path sees a
            # detached latent so geometric losses on x0_pred cannot reach the Encoder.
            # anchor_aux_loss keeps the un-detached latent (independent scorer path).
            cond_inv_d = latent_inv.detach() if self.detach_latent else latent_inv
            cond_eq_d = latent_eq.detach() if self.detach_latent else latent_eq
            if self.training and target_coords is not None:
                diffusion_loss, noise_pred, pred_coords = self.coord_diffusion(
                    target_coords, cond_inv=cond_inv_d, cond_eq=cond_eq_d,
                    return_noise_pred=True, return_x0_pred=True,
                )
                # Dynamic anchor auxiliary loss (supervise scorer via pair_probs)
                # Uses the UN-detached latent so the scorer still learns even when
                # detach_latent=True (Encoder gets this signal, diffusion doesn't).
                if pair_probs is not None:
                    # CRBPSA: hydrogen-bond-weight the pairing signal before LAMA
                    # anchor supervision. Strong H-bond pairs (A-U=2, G-C=3) are real
                    # stem contacts; G-U wobble (0.8) and non-pairing stay at 1.0. This
                    # makes all 3 LAMA channels (pair_hotspot / neighbor_density /
                    # local_connectivity) stem-aware instead of raw-pairing-aware.
                    # Only the anchor scorer path is weighted — contact_aux and
                    # physics_pairing keep the original pair_probs.
                    # Dual advanced indexing (broadcast to [B,L,L]); the naive
                    # HBOND_MAP[seq][:,:,seq] would explode to [B,L,B,L,5].
                    # .to(seq_tokens.device): module-level const is on CPU, GPU
                    # indexing tensors must match the indexed tensor's device.
                    hbond_w = HBOND_MAP.to(seq_tokens.device)[
                        seq_tokens.unsqueeze(-1),
                        seq_tokens.unsqueeze(-2)]   # [B, L, L]
                    pair_hb = pair_probs * hbond_w.to(pair_probs.device)
                    anchor_aux_loss = self.coord_diffusion.anchor_aux_loss(
                        latent_inv, latent_eq, pair_hb,
                    )
                if return_coords:
                    pred_coords = self.coord_diffusion.generate(
                        cond_inv_d, cond_eq_d,
                    )
            elif not self.training or return_coords:
                # MC-Dropout (model.train() + return_loss=False + no target) also
                # needs generation — UQ estimation in train_s10_curriculum calls
                # model(seq_ids, return_loss=False) in train() mode with dropout
                # active. Without this, pred_coords stays None → UQ crashes.
                pred_coords = self.coord_diffusion.generate(
                    cond_inv_d, cond_eq_d,
                )

        # [v4] AF3-style lightweight physics refinement (inference only).
        # Minimizes stereochemistry energy (bond/angle/clash/dihedral) on the
        # predicted coords via short Adam descent + bond-length projection.
        # Guarantees chemically valid backbone geometry in the output.
        if refine and not self.training and pred_coords is not None and lengths is not None:
            pred_coords = refine_coords(
                pred_coords, lengths, n_steps=refine_steps, lr=0.5,
                project_bonds=True,
            )

        # V2: 接触图辅助任务 — predict only when it will be consumed
        # (use_contact_aux flag, or detach_latent so Phase-1 has a geometry signal)
        contact_pred = None
        if self.contact_aux_head is not None and (self.config.use_contact_aux or self.detach_latent):
            contact_pred = self.contact_aux_head(latent_inv)

        if return_loss and diffusion_loss is not None:
            # v4.1: return 4-tuple — (diff_loss, pred_coords, contact_pred, anchor_aux_loss)
            # anchor_aux_loss may be None (pair_probs not provided) or a scalar
            return diffusion_loss, pred_coords, contact_pred, anchor_aux_loss
        if contact_pred is not None:
            return pred_coords, contact_pred
        return pred_coords

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