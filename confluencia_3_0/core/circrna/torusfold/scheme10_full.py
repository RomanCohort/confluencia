"""
scheme10_full.py — S10完整版：稀疏等变encoder + S8 refine + inv扩散

架构：
  序列 → S10稀疏encoder → [inv | eq]分离 → inv扩散 + S8 refine(eq) → coords
           ↑                    ↑              ↑              ↑
       精确等变            等变latent      扩散不破坏     长程依赖增强
       O(L·K)                            等变性

v7 新增: MSA-Evoformer (AlphaFold2 风格 MSA+Pair 交替更新)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpe import TorusPositionalEncoding
from .pair_track import PairTrack, PairTrackConfig
from .msa_evoformer import MSAEvoformer, MSAEvoformerConfig
from .dynamic_trajectory import DynamicEnsembleGenerator, DynamicEnsembleConfig, DynamicEnsembleResult

# [v4] CRBPSA 氢键加权 (A-U=2, G-C=3, G-U=0.8)
HBOND_MAP = torch.tensor([
    [1.0, 2.0, 1.0, 1.0, 1.0],  # A-*    (A-U=2)
    [2.0, 1.0, 0.8, 1.0, 1.0],  # U-*    (U-A=2, U-G=0.8)
    [1.0, 0.8, 1.0, 3.0, 1.0],  # G-*    (G-U=0.8, G-C=3)
    [1.0, 1.0, 3.0, 1.0, 1.0],  # C-*    (C-G=3)
    [1.0, 1.0, 1.0, 1.0, 1.0],  # N-*    (pad/unknown)
], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FullS10Config:
    """完整版S10配置"""
    n_tokens: int = 5
    n_ss_tokens: int = 8      # .,(,),{,},<,> + padding(7)
    use_ss_embedding: bool = True  # SS token 融合到输入嵌入
    d_model: int = 256

    # 稀疏GNN
    d_edge: int = 64
    k_theta: int = 2
    k_phi: int = 1
    n_layers: int = 4
    K_sparse: int = 64       # K的最大值（√L缩放上限）
    dropout: float = 0.1
    use_weighted_pooling: bool = True  # 加权pooling替代mean
    use_log_distance_buckets: bool = True  # 对数距离分桶
    use_learnable_pair_scorer: bool = True  # 可学习pair scoring

    # irrep分离（新增）
    d_model_inv: int = 64    # encoder输出中degree 0的维度
    d_model_eq: int = 192    # encoder输出中degree 1+的维度
    # 注意：d_model_inv + d_model_eq 应该 <= d_model

    # Latent (CoordDiffusion 要求 d_inv >= 64, d_eq >= 64)
    d_inv: int = 64
    d_eq: int = 64

    # S8 Refine（只处理eq部分，增强长程依赖）
    use_s8_refine: bool = True
    n_refine_layers: int = 2
    refine_nhead: int = 4
    refine_dropout: float = 0.1

    # 扩散（v4: 坐标空间扩散，替代 InvDiffusion）
    use_coord_diffusion: bool = True          # True=CoordDiffusion(坐标空间), False=InvDiffusion(latent)
    n_diffusion_steps: int = 100
    d_coord_hidden: int = 128                 # CoordDiffusion 内部特征维度
    cfg_dropout_prob: float = 0.1             # CFG drop 条件概率
    use_dynamic_anchors: bool = True          # DynamicAnchorAttention
    anchor_ratio: float = 0.1                 # A = max(128, L*ratio)

    # Latent 多尺度（L>1000，在 EquivariantLatent 内做 flat↔2d reshape 适配）
    use_multiscale_latent: bool = True

    # Kendall uncertainty weighting
    use_kendall_uw: bool = True
    kendall_loss_keys: tuple = ('coord', 'bond', 'closure', 'stereo', 'physics_pairing', 'contact_aux', 'physics_bridge')

    # Contact auxiliary head
    use_contact_aux: bool = True

    # 结构引导微调 (structRFM 启发)
    use_structure_guided_mask: bool = False  # 训练时随机掩码 bpp 高概率区域
    structure_mask_prob: float = 0.15        # 掩码概率

    # 结构
    bond_length: float = 5.9
    r_scale: float = 0.5
    n_edge_cats: int = 9  # 对数距离分桶: 8个桶 + 1个默认

    # Pair Track (Evoformer lightweight)
    use_pair_track: bool = True
    pair_track_layers: int = 2       # 2 层 (轻量)
    d_pair: int = 64                 # pair representation 维度 (轻量)
    pair_n_heads: int = 4
    pair_d_ffn: int = 128
    pair_dropout: float = 0.1

    # 可学习 pair scoring（改进 D）
    pair_scorer_heads: int = 4       # 多头 attention
    pair_scorer_dropout: float = 0.1

    # 对数距离分桶（改进 C）
    log_distance_buckets: tuple = (1, 3, 7, 15, 31, 63, 127, 255)

    # MSA-Evoformer (AlphaFold2 风格)
    use_msa: bool = True
    d_msa: int = 128
    n_msa_layers: int = 4
    n_msa_heads: int = 4
    d_msa_hidden: int = 16   # 内部低维
    d_msa_ffn: int = 256
    n_msa_representatives: int = 64  # 聚类后代表性序列数
    msa_dropout: float = 0.1

    # RhoFold+ backbone (可选, 替代 token_embed)
    use_rhofold: bool = False
    rhofold_freeze_layers: int = 9   # 冻结前 N 层

    # 动态构象系综
    use_dynamic_ensemble: bool = True
    ensemble_temperatures: tuple = (0.0, 0.1, 0.3, 0.5, 1.0)
    n_samples_per_temp: int = 20
    use_potential_guidance: bool = True
    potential_weight: float = 0.1
    potential_refine_steps: int = 10
    use_markov_trajectory: bool = True
    n_trajectory_steps: int = 1000
    rmsd_kernel_sigma: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# bpp 先验模块（方案B：共享伴侣相似度）
# ══════════════════════════════════════════════════════════════════════════════

class BppPriorModule(nn.Module):
    """ViennaRNA bpp 先验：共享伴侣相似度

    核心洞察：如果 i 和 j 都与同一组位置配对（共享伴侣），
    则它们可能在同一结构域，有更高概率空间接近。

    输出：cooc (B, L, L) ∈ [0, 1]，共享伴侣的 Jaccard 相似度。
    """

    def __init__(self, d_model: int, bpp_threshold: float = 0.1):
        super().__init__()
        self.bpp_threshold = bpp_threshold
        # 可学习的先验权重
        self.prior_weight = nn.Parameter(torch.tensor(0.1))
        # 可选的非线性投影（增强表达能力）
        self.project = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor, bpp_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) 节点特征（用于形状参考）
            bpp_matrix: (L, L) 或 (B, L, L) ViennaRNA 配对概率矩阵

        Returns:
            cooc: (B, L, L) 共享伴侣相似度 × 可学习权重
        """
        B, L, D = x.shape
        device = x.device

        # 扩展 bpp_matrix 到 batch 维度
        if bpp_matrix.dim() == 2:
            bpp_matrix = bpp_matrix.unsqueeze(0).expand(B, -1, -1)
        bpp_matrix = bpp_matrix.to(device)

        # 二值化：找到每个位置的配对伙伴
        partner_mask = (bpp_matrix > self.bpp_threshold).float()  # (B, L, L)

        # 共享伴侣数：partner_mask @ partner_mask.T
        # shared_partners[i,j] = |partners(i) ∩ partners(j)|
        shared_partners = torch.bmm(partner_mask, partner_mask.transpose(1, 2))  # (B, L, L)

        # Jaccard 相似度：|A ∩ B| / |A ∪ B|
        # |A ∪ B| = |A| + |B| - |A ∩ B|
        partners_i = partner_mask.sum(dim=-1, keepdim=True)  # (B, L, 1)
        partners_j = partner_mask.sum(dim=-2, keepdim=True)  # (B, 1, L)
        union = partners_i + partners_j - shared_partners  # (B, L, L)
        cooc = shared_partners / (union + 1e-6)  # (B, L, L)

        # 可选非线性投影
        cooc_projected = self.project(cooc.unsqueeze(-1)).squeeze(-1)  # (B, L, L)

        return cooc_projected * self.prior_weight


# ══════════════════════════════════════════════════════════════════════════════
# 可学习 Pair Scorer（改进 D：端到端学习 pair 概率 + bpp 先验）
# ══════════════════════════════════════════════════════════════════════════════

class PairScorer(nn.Module):
    """可学习的 pair scoring 模块（多头点积注意力 + bpp 先验）

    双轨制：
    - Track 1（近端）：用 bpp 直接选择高概率配对
    - Track 2（远端）：用 learnable scores + bpp 共现正则

    输出 pair_probs: (B, L, L)，用于 Top-K 邻居选择。
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 use_bpp_prior: bool = True, bpp_threshold: float = 0.1,
                 local_distance: int = 100):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.local_distance = local_distance  # 近端/远端分界距离

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # bpp 先验模块
        self.use_bpp_prior = use_bpp_prior
        if use_bpp_prior:
            self.bpp_prior = BppPriorModule(d_model, bpp_threshold)

    def forward(self, x: torch.Tensor, bpp_matrix: torch.Tensor = None) -> torch.Tensor:
        """x: (B, L, d_model) -> pair_probs: (B, L, L)"""
        B, L, D = x.shape

        # 多头 Q/K 投影
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, d_head)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # 点积注意力（学习的 pair score）
        attn = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # (B, H, L, L)
        attn = self.dropout(attn)
        attn = attn.mean(dim=1)  # (B, L, L) 学习的 pair score

        # 加入 bpp 先验（方案B：共享伴侣相似度）
        if self.use_bpp_prior and bpp_matrix is not None:
            bpp_prior = self.bpp_prior(x, bpp_matrix)  # (B, L, L)
            attn = attn + bpp_prior

        pair_probs = F.softmax(attn, dim=-1)

        return pair_probs

    def get_dual_track_neighbors(self, x: torch.Tensor, K_local: int, K_global: int,
                                  bpp_matrix: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """双轨制邻居选择（方案C）

        Track 1（近端）：用 bpp 选择高概率配对
        Track 2（远端）：用 learnable scores + bpp 共现正则

        Returns:
            topk_idx: (B, L, K_local + K_global) 合并的邻居索引
            attn_weights: (B, L, K_local + K_global) 注意力权重
            track_mask: (B, L, K_local + K_global) True=近端, False=远端
        """
        B, L, D = x.shape
        device = x.device

        # 计算学习的 pair score
        pair_probs = self.forward(x, bpp_matrix)  # (B, L, L)

        # 距离矩阵
        pos = torch.arange(L, device=device).float()
        dist_matrix = torch.min(
            (pos.unsqueeze(0) - pos.unsqueeze(1)).abs(),
            L - (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        )  # (L, L)

        # Track 1：近端（距离 < local_distance），用 bpp
        local_mask = (dist_matrix < self.local_distance).unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        local_probs = pair_probs.clone()
        local_probs[~local_mask] = -1e9  # 屏蔽远端
        _, local_idx = local_probs.topk(K_local, dim=-1)  # (B, L, K_local)

        # Track 2：远端（距离 >= local_distance），用 learnable + bpp 共现正则
        global_mask = (dist_matrix >= self.local_distance).unsqueeze(0).expand(B, -1, -1)
        global_probs = pair_probs.clone()
        global_probs[~global_mask] = -1e9  # 屏蔽近端

        # 如果有 bpp，加入共现正则
        if bpp_matrix is not None:
            bpp_prior = self.bpp_prior(x, bpp_matrix)
            global_probs = global_probs + bpp_prior

        _, global_idx = global_probs.topk(K_global, dim=-1)  # (B, L, K_global)

        # 合并
        topk_idx = torch.cat([local_idx, global_idx], dim=-1)  # (B, L, K_local + K_global)

        # 提取对应的注意力权重
        attn_weights = torch.gather(pair_probs, 2, topk_idx)  # (B, L, K_local + K_global)

        # 标记哪些是近端（True），哪些是远端（False）
        track_mask = torch.cat([
            torch.ones(B, L, K_local, dtype=torch.bool, device=device),
            torch.zeros(B, L, K_global, dtype=torch.bool, device=device),
        ], dim=-1)

        return topk_idx, attn_weights, track_mask


# ══════════════════════════════════════════════════════════════════════════════
# 稀疏等变Kernel（正确的SO(2)版本）
# ══════════════════════════════════════════════════════════════════════════════

class SparseSO2SteerableKernel(nn.Module):
    """稀疏版SO(2)×SO(2) steerable kernel"""

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
        """构建irrep特征 (B, L, K, n_irrep)"""
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

    def forward(self, x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx, attn_weights=None):
        """稀疏 SO(2) steerable kernel 消息传递

        Args:
            x: (B, L, d_model) 节点特征
            delta_theta_sparse: (B, L, K) 角度差
            delta_phi_sparse: (B, L, K) φ角度差
            edge_cat_sparse: (B, L, K) 边类别
            topk_idx: (B, L, K) 邻居索引
            attn_weights: (B, L, K) 可选注意力权重（改进 B：加权 pooling）
        """
        B, L, _ = x.shape
        K = delta_theta_sparse.shape[2]

        irrep = self._irrep_features_sparse(delta_theta_sparse, delta_phi_sparse)
        K_irrep = irrep.shape[-1]

        # Gather邻居
        b_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, K)
        x_j = x[b_idx, topk_idx]

        # Lift
        lifted = self.node_lift(x_j).reshape(B, L, K, K_irrep, self.d_edge)

        # Modulate
        modulated = lifted * irrep.unsqueeze(-1)

        # Edge cat
        cat_emb = self.cat_embed(edge_cat_sparse)
        modulated = modulated * (1.0 + cat_emb.unsqueeze(-2))

        # Flatten
        flat = modulated.reshape(B, L, K, -1)

        # Contract
        C = self.n_edge_cats
        partial = torch.einsum('blik,ckm->blicm', flat, self.kernel)

        # Gather by category
        b_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, K)
        i_idx = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, K)
        k_idx = torch.arange(K, device=x.device).view(1, 1, K).expand(B, L, K)
        msg = partial[b_idx, i_idx, k_idx, edge_cat_sparse]

        # 改进 B：加权 pooling（替代 mean，保留邻居独特贡献）
        if attn_weights is not None:
            attn_weights = F.softmax(attn_weights / math.sqrt(self.d_model), dim=-1)
            return (msg * attn_weights.unsqueeze(-1)).sum(dim=2)
        else:
            return msg.mean(dim=2)


class SparseEquivariantGNNLayer(nn.Module):
    """稀疏等变GNN层"""

    def __init__(self, config):
        super().__init__()
        self.kernel = SparseSO2SteerableKernel(
            config.d_model, config.d_edge, config.n_edge_cats,
            config.k_theta, config.k_phi
        )
        self.update = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, attn_weights=None):
        msg = self.kernel(x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx, attn_weights)
        x_new = self.update(torch.cat([x, msg], dim=-1))
        return self.norm(x + x_new)


# ══════════════════════════════════════════════════════════════════════════════
# 等变Encoder（输出分离的inv和eq）
# ══════════════════════════════════════════════════════════════════════════════

class EquivariantEncoder(nn.Module):
    """等变encoder（稀疏steerable kernel）+ irrep分离

    改进:
    A. 动态K: √L缩放，符合图神经网络"有效直径"理论
    B. 加权pooling: 用pair_probs作权重，保留邻居独特贡献
    C. 对数距离分桶: 自适应长序列
    D. 可学习pair_probs: 端到端学习邻居选择
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_rhofold = getattr(config, 'use_rhofold', False)
        self.use_ss = getattr(config, 'use_ss_embedding', False)

        # SS token embedding (structRFM 启发)
        if self.use_ss:
            self.ss_embed = nn.Embedding(
                getattr(config, 'n_ss_tokens', 7), config.d_model, padding_idx=7
            )

        # 输入嵌入 (二选一)
        if self.use_rhofold:
            from .rhofold_backbone import RhoFoldBackbone
            self.backbone = RhoFoldBackbone(
                freeze_layers=getattr(config, 'rhofold_freeze_layers', 9),
                use_pair_repr=True,
            )
            # RhoFold+ (640d) → d_model (256d) 投影
            self.rhofold_proj = nn.Sequential(
                nn.Linear(640, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            # RhoFold+ pair_repr (128d) → 标量, 用于邻居选择
            self.rhofold_pair_proj = nn.Linear(128, 1, bias=False)
            self.token_embed = None
            self.tpe = None
        else:
            self.token_embed = nn.Embedding(config.n_tokens, config.d_model)
            self.tpe = TorusPositionalEncoding(config.d_model, n_harmonics=16, dropout=config.dropout)
            self.backbone = None
            self.rhofold_proj = None
            self.rhofold_pair_proj = None

        # MSA-Evoformer (v7 新增)
        if config.use_msa:
            msa_config = MSAEvoformerConfig(
                d_msa=config.d_msa,
                d_pair=64,  # 与 d_model_inv 匹配
                n_layers=config.n_msa_layers,
                n_heads=config.n_msa_heads,
                d_msa_hidden=config.d_msa_hidden,
                d_msa_ffn=config.d_msa_ffn,
                n_representatives=config.n_msa_representatives,
                dropout=config.msa_dropout,
            )
            self.msa_evoformer = MSAEvoformer(msa_config, n_tokens=config.n_tokens)
            self.msa_proj = nn.Sequential(
                nn.Linear(config.d_msa, config.d_model),
                nn.LayerNorm(config.d_model),
            )
        else:
            self.msa_evoformer = None
            self.msa_proj = None

        # 改进 D：可学习 pair scoring
        if config.use_learnable_pair_scorer:
            self.pair_scorer = PairScorer(
                config.d_model,
                n_heads=config.pair_scorer_heads,
                dropout=config.pair_scorer_dropout,
            )
        else:
            self.pair_scorer = None

        self.layers = nn.ModuleList([
            SparseEquivariantGNNLayer(config) for _ in range(config.n_layers)
        ])

        # irrep分离投影
        # degree 0部分：可以用非线性（因为后续是inv操作）
        self.to_inv = nn.Sequential(
            nn.Linear(config.d_model, config.d_model_inv * 2),
            nn.GELU(),
            nn.Linear(config.d_model_inv * 2, config.d_model_inv),
        )
        # degree 1+部分：只能用线性（保等变性）
        self.to_eq = nn.Linear(config.d_model, config.d_model_eq, bias=False)

        # S8 Refine 第1次（架构图: Encoder 内 SparseGNN×4 → S8 Refine → PairTrack）
        # 需要先做 irrep 分离再调用，然后 concat 回 d_model
        if getattr(config, 'use_s8_refine', True):
            self.encoder_s8_refine = EquivariantS8Refine(
                config.d_model_inv, config.d_model_eq,
                n_layers=1,  # encoder 内只跑 1 层（第2层在 PairTrack 之后）
                nhead=config.refine_nhead,
                dropout=config.refine_dropout,
            )
            # concat 后投影回 d_model
            self._s8_post_cat = nn.Sequential(
                nn.Linear(config.d_model_inv + config.d_model_eq, config.d_model),
                nn.LayerNorm(config.d_model),
            )
        else:
            self.encoder_s8_refine = None
            self._s8_post_cat = None

    @staticmethod
    def _encode_ss_tokens(ss_string: str, L: int, device):
        """二级结构字符串 → token ids (0-indexed)"""
        ss_map = {'.': 0, '(': 1, ')': 2, '{': 3, '}': 4, '<': 5, '>': 6}
        ids = [ss_map.get(ch, 7) for ch in ss_string]
        ids = ids[:L]
        ids += [7] * (L - len(ids))  # padding
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def forward(self, seq_tokens, pair_probs=None, bpp_matrix=None, msa_tokens=None, ss_tokens=None):
        """等变encoder前向传播

        Args:
            seq_tokens: (B, L) 序列token
            pair_probs: (B, L, L) 可选的预计算pair概率（兼容旧接口）
            bpp_matrix: (L, L) 或 (B, L, L) ViennaRNA bpp配对概率矩阵
            msa_tokens: (B, N_rep, L) MSA token（聚类后的代表性序列）

        Returns:
            node_repr_inv: (B, L, d_model_inv) 不变特征
            node_repr_eq: (B, L, d_model_eq) 等变特征
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # ── 输入嵌入 (RhoFold+ 或 token_embed) ──
        if self.use_rhofold and self.backbone is not None:
            rhofold_node, rhofold_pair = self.backbone(seq_tokens)
            x = self.rhofold_proj(rhofold_node)  # (B,L,640) → (B,L,256)
            # RhoFold+ pair → 邻居选择分数
            if pair_probs is None and rhofold_pair is not None:
                pair_probs = self.rhofold_pair_proj(rhofold_pair).squeeze(-1)  # (B,L,L,128)→(B,L,L)
        else:
            x = self.token_embed(seq_tokens)
            x = self.tpe(x, seq_len=L)

        # SS token 融合 (structRFM 启发)
        if self.use_ss and ss_tokens is not None:
            ss_emb = self.ss_embed(ss_tokens)  # (B, L, d_model)
            x = x + ss_emb  # 残差融合

        # ── MSA-Evoformer (v7 新增) ──
        if msa_tokens is not None and self.msa_evoformer is not None:
            msa_enhanced, pair_enhanced = self.msa_evoformer(msa_tokens, bpp_matrix)
            # 残差注入 MSA 增强特征
            x = x + self.msa_proj(msa_enhanced)

        # ── 改进 A：动态K（√L缩放）──
        K_actual = int(L ** 0.5)
        K_actual = max(8, min(K_actual, self.config.K_sparse))
        # 当 L<75 时自适应下调，避免 60% 稠密浪费计算
        if L < 75:
            K_actual = max(8, L // 2)

        # ── 结构引导掩码 (structRFM 启发，训练时随机掩码 bpp 高概率区域) ──
        if (self.training and bpp_matrix is not None and
                getattr(self.config, 'use_structure_guided_mask', False)):
            bpp_mask = torch.bernoulli(
                torch.full_like(bpp_matrix, 1.0 - self.config.structure_mask_prob)
            ).bool()
            # 只掩码高概率区域 (bpp > 0.5)
            high_prob = bpp_matrix > 0.5
            bpp_matrix = bpp_matrix.clone()
            bpp_matrix[high_prob & ~bpp_mask] = 0.0

        # ── 双轨制邻居选择（方案C + 方案B）──
        if self.pair_scorer is not None and bpp_matrix is not None:
            # 双轨制：近端用 bpp，远端用 learnable + bpp 共现正则
            K_local = K_actual // 2  # 近端占一半
            K_global = K_actual - K_local  # 远端占一半
            topk_idx, attn_weights, track_mask = self.pair_scorer.get_dual_track_neighbors(
                x, K_local, K_global, bpp_matrix
            )
        elif self.pair_scorer is not None:
            # 无 bpp 时，退化为纯 learnable
            pair_probs = self.pair_scorer(x, None)
            _, topk_idx = pair_probs.topk(K_actual, dim=-1)
            attn_weights = torch.gather(pair_probs, 2, topk_idx)
        elif pair_probs is not None:
            # 兼容旧接口
            _, topk_idx = pair_probs.topk(K_actual, dim=-1)
            attn_weights = torch.gather(pair_probs, 2, topk_idx)
        else:
            # Fallback：用距离衰减生成伪 pair_probs（近邻优先）
            pos = torch.arange(L, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            dist = torch.min(dist, L - dist)  # 循环距离
            pair_probs = torch.softmax(-dist / 10.0, dim=-1)  # (L, L)
            pair_probs = pair_probs.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
            _, topk_idx = pair_probs.topk(K_actual, dim=-1)
            attn_weights = torch.gather(pair_probs, 2, topk_idx)

        # ── 构建角度差（循环连续性）──
        pos = torch.arange(L, device=device, dtype=torch.float32)
        theta = 2 * math.pi * pos / L
        theta_j = theta[topk_idx]
        theta_i = theta.view(1, L, 1)
        # 归一化到 [-π, π]（循环边界）
        delta_theta_sparse = theta_i - theta_j
        delta_theta_sparse = (delta_theta_sparse + math.pi) % (2 * math.pi) - math.pi
        delta_phi_sparse = torch.zeros_like(delta_theta_sparse)

        # ── 改进 C：对数距离分桶 ──
        if self.config.use_log_distance_buckets:
            circular_dist = torch.min(
                (pos[topk_idx] - pos.unsqueeze(1)).abs(),
                L - (pos[topk_idx] - pos.unsqueeze(1)).abs()
            ).float()
            # 对数分桶
            buckets = self.config.log_distance_buckets
            edge_cat_sparse = torch.full(
                (B, L, K_actual), len(buckets), device=device, dtype=torch.long
            )
            for i, bucket in enumerate(buckets):
                edge_cat_sparse[circular_dist <= bucket] = i
        else:
            edge_cat_sparse = torch.zeros(B, L, K_actual, device=device, dtype=torch.long)

        # ── 稀疏 GNN（传递 attn_weights 用于加权 pooling）──
        for layer in self.layers:
            x = layer(x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, attn_weights)

        # ── S8 Refine 第1次（架构图: SparseGNN → S8 Refine → PairTrack）──
        if self.encoder_s8_refine is not None:
            _inv_pre = self.to_inv(x)
            _eq_pre = self.to_eq(x)
            _eq_refined = self.encoder_s8_refine(_inv_pre, _eq_pre)
            x = self._s8_post_cat(torch.cat([_inv_pre, _eq_refined], dim=-1))

        # irrep分离
        node_repr_inv = self.to_inv(x)   # (B, L, d_model_inv) - degree 0
        node_repr_eq = self.to_eq(x)     # (B, L, d_model_eq) - degree 1+

        # 返回 pair_probs 供 anchor_aux_loss / contact_aux 使用
        pair_probs_out = pair_probs if pair_probs is not None else torch.zeros(B, L, L, device=device)

        return node_repr_inv, node_repr_eq, pair_probs_out


# ══════════════════════════════════════════════════════════════════════════════
# Latent分离（接收已经分离的irrep）
# ══════════════════════════════════════════════════════════════════════════════

class EquivariantLatent(nn.Module):
    """等变latent：从分离的irrep投影到latent（支持多尺度 L>1000）

    eq 格式适配：encoder 输出 flat (B,L,d_eq)，Downsample/Upsample/FineGNN
    需要 (B,L,d_eq,2)。在 forward 内做 reshape 适配。
    """

    def __init__(self, d_model_inv, d_model_eq, d_inv, d_eq, use_multiscale=True):
        super().__init__()
        self.use_multiscale = use_multiscale

        self.inv_proj = nn.Sequential(
            nn.Linear(d_model_inv, d_inv * 2),
            nn.GELU(),
            nn.Linear(d_inv * 2, d_inv),
        )
        # flat eq 输出: (B,L,d_eq)
        self.eq_proj = nn.Linear(d_model_eq, d_eq, bias=False)

        # 多尺度组件（延迟初始化，按需加载）
        self._multiscale_initialized = False
        self.downsample = None
        self.upsample = None
        self.fine_gnn = None
        # 最小多尺度长度阈值
        self._multiscale_threshold = 1000

    def _init_multiscale_if_needed(self, factor, device):
        if self._multiscale_initialized:
            return
        from .multiscale_equivariant import DownsampleEncoder, UpsampleDecoder, FineGrainedGNN
        d_in_inv = self.inv_proj[0].in_features
        d_in_eq = self.eq_proj.in_features
        d_out_inv = self.inv_proj[-1].out_features
        d_out_eq = self.eq_proj.out_features
        self.downsample = DownsampleEncoder(factor, d_out_inv, d_out_eq // 2).to(device)
        self.upsample = UpsampleDecoder(factor, d_out_inv, d_out_eq // 2).to(device)
        self.fine_gnn = FineGrainedGNN(d_out_inv, d_out_eq // 2, d_hidden=64, local_radius=8, dropout=0.1).to(device)
        self._multiscale_initialized = True

    def forward(self, node_repr_inv, node_repr_eq):
        B, L = node_repr_inv.shape[:2]
        latent_inv = self.inv_proj(node_repr_inv)   # (B, L, d_inv)
        latent_eq = self.eq_proj(node_repr_eq)      # (B, L, d_eq) flat

        if (self.use_multiscale and L > self._multiscale_threshold
                and self.training):
            device = latent_inv.device
            d_eq = latent_eq.shape[-1]
            # flat → (B, L, d_eq//2, 2) 供 Downsample/Upsample/FineGNN
            eq_2d = latent_eq.reshape(B, L, d_eq // 2, 2)

            # 选择降采样因子
            from .multiscale_equivariant import get_downsample_factor, MultiScaleConfig
            ms_cfg = MultiScaleConfig()
            factor = get_downsample_factor(L, ms_cfg)
            self._init_multiscale_if_needed(factor, device)

            # Downsample: (B,L,d_inv) + (B,L,d_eq//2,2) → (B,Lc,d_inv) + (B,Lc,d_eq//2,2)
            inv_coarse, eq_coarse = self.downsample(latent_inv, eq_2d)
            Lc = inv_coarse.shape[1]

            # Fine-grained GNN on coarse
            inv_refined, eq_refined = self.fine_gnn(inv_coarse, eq_coarse)

            # Upsample: (B,Lc,d_inv) + (B,Lc,d_eq//2,2) → (B,L,d_inv) + (B,L,d_eq//2,2)
            inv_up, eq_up = self.upsample(inv_refined, eq_refined, L)

            # reshape 回 flat: (B,L,d_eq//2,2) → (B,L,d_eq)
            return inv_up, eq_up.reshape(B, L, d_eq)

        return latent_inv, latent_eq


# ══════════════════════════════════════════════════════════════════════════════
# S8 Refine（正确的等变版本：用node_repr_inv算注意力，对node_repr_eq加权）
# ══════════════════════════════════════════════════════════════════════════════

class EquivariantS8RefineLayer(nn.Module):
    """全局等变注意力层（严格保等变性）

    设计：
    - 注意力权重用 node_repr_inv（degree 0，不变量）计算
    - 对 node_repr_eq（degree 1+，等变）做加权平均

    因为注意力权重是标量（softmax后），对等变向量做加权平均是线性组合，保等变。
    """

    def __init__(self, d_model_inv, d_model_eq, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model_inv = d_model_inv
        self.d_model_eq = d_model_eq
        self.nhead = nhead
        self.d_head = d_model_inv // nhead

        assert d_model_inv % nhead == 0, f"d_model_inv={d_model_inv} must be divisible by nhead={nhead}"

        # QKV投影（只用inv部分计算注意力权重）
        self.q_proj = nn.Linear(d_model_inv, d_model_inv, bias=False)
        self.k_proj = nn.Linear(d_model_inv, d_model_inv, bias=False)
        self.v_proj = nn.Linear(d_model_eq, d_model_eq, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(d_model_eq, d_model_eq, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_repr_inv, node_repr_eq):
        """
        node_repr_inv: (B, L, d_model_inv) - degree 0，不变量
        node_repr_eq: (B, L, d_model_eq) - degree 1+，等变

        返回：refined node_repr_eq (B, L, d_model_eq)，严格保等变性
        """
        B, L, _ = node_repr_eq.shape

        # 1. 用inv部分计算注意力权重（softmax对不变量安全）
        q = self.q_proj(node_repr_inv)  # (B, L, d_model_inv)
        k = self.k_proj(node_repr_inv)  # (B, L, d_model_inv)

        # 全局注意力（不用multihead简化版本）
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model_inv)  # (B, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 2. 用注意力权重对eq部分做加权平均（线性组合，保等变）
        v = self.v_proj(node_repr_eq)  # (B, L, d_model_eq)
        out = torch.bmm(attn, v)  # (B, L, d_model_eq)
        out = self.out_proj(out)

        # 3. 残差（线性，保等变）
        return node_repr_eq + self.dropout(out)


class EquivariantS8Refine(nn.Module):
    """全局等变S8 refine：多层等变注意力

    用全局注意力覆盖长程依赖，同时严格保持等变性。
    """

    def __init__(self, d_model_inv, d_model_eq, n_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EquivariantS8RefineLayer(d_model_inv, d_model_eq, nhead, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, node_repr_inv, node_repr_eq):
        """
        node_repr_inv: (B, L, d_model_inv)
        node_repr_eq: (B, L, d_model_eq)

        返回：refined node_repr_eq (B, L, d_model_eq)
        """
        for layer in self.layers:
            node_repr_eq = layer(node_repr_inv, node_repr_eq)
        return node_repr_eq


# 保留旧版本（用于对比实验，不保等变性）
class S8RefineLayer(nn.Module):
    """旧版S8 refine：普通注意力（不保等变性）"""

    def __init__(self, d_eq, nhead=4, dropout=0.1):
        super().__init__()
        assert d_eq % nhead == 0, f"d_eq={d_eq} must be divisible by nhead={nhead}"
        self.d_eq = d_eq
        self.nhead = nhead
        self.d_head = d_eq // nhead

        self.q_proj = nn.Linear(d_eq, d_eq, bias=False)
        self.k_proj = nn.Linear(d_eq, d_eq, bias=False)
        self.v_proj = nn.Linear(d_eq, d_eq, bias=False)
        self.out_proj = nn.Linear(d_eq, d_eq, bias=False)
        self.norm = nn.LayerNorm(d_eq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.nhead, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.d_eq)
        out = self.out_proj(out)
        x = x + self.dropout(out)
        x = self.norm(x)
        return x


class S8Refine(nn.Module):
    """旧版S8 refine（对比用）"""

    def __init__(self, d_eq, n_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            S8RefineLayer(d_eq, nhead, dropout) for _ in range(n_layers)
        ])

    def forward(self, latent_eq):
        for layer in self.layers:
            latent_eq = layer(latent_eq)
        return latent_eq


# ══════════════════════════════════════════════════════════════════════════════
# Inv扩散
# ══════════════════════════════════════════════════════════════════════════════

class InvDiffusion(nn.Module):
    """只对inv通道做扩散"""

    def __init__(self, d_inv, n_steps=100):
        super().__init__()
        self.d_inv = d_inv
        self.n_steps = n_steps

        beta = torch.linspace(1e-4, 0.02, n_steps)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

        self.denoiser = nn.Sequential(
            nn.Linear(d_inv + 64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_inv),
        )

        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

    def add_noise(self, x_clean, t):
        noise = torch.randn_like(x_clean)
        alpha_bar_t = self.alpha_bar[t]
        return torch.sqrt(alpha_bar_t) * x_clean + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def denoise_step(self, x_noisy, t):
        t_emb = self.time_embed(torch.tensor([t / self.n_steps], device=x_noisy.device).float())
        t_emb = t_emb.unsqueeze(0).unsqueeze(0).expand(x_noisy.shape[0], x_noisy.shape[1], -1)
        x_input = torch.cat([x_noisy, t_emb], dim=-1)
        noise_pred = self.denoiser(x_input)

        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]
        x_prev = (1.0 / torch.sqrt(alpha_t)) * (x_noisy - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred)
        if t > 0:
            x_prev = x_prev + torch.sqrt(beta_t) * torch.randn_like(x_noisy)
        return x_prev

    def forward(self, x_clean, return_loss=True):
        B, L, D = x_clean.shape
        device = x_clean.device
        t = torch.randint(0, self.n_steps, (B,), device=device)

        x_noisy_list, noise_list = [], []
        for b in range(B):
            xn, n = self.add_noise(x_clean[b], t[b].item())
            x_noisy_list.append(xn)
            noise_list.append(n)

        x_noisy = torch.stack(x_noisy_list)
        noise_target = torch.stack(noise_list)

        t_emb = self.time_embed((t.float() / self.n_steps).unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).expand(B, L, -1)
        x_input = torch.cat([x_noisy, t_emb], dim=-1)
        noise_pred = self.denoiser(x_input)

        if return_loss:
            return x_noisy, F.mse_loss(noise_pred, noise_target)
        return noise_pred, None

    def sample(self, B, L, device):
        x = torch.randn(B, L, self.d_inv, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.denoise_step(x, t)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 结构头
# ══════════════════════════════════════════════════════════════════════════════

class TorusCoordHead(nn.Module):
    def __init__(self, d_latent, bond_length=5.9, r_scale=0.5):
        super().__init__()
        self.bond_length = bond_length
        self.r_scale = r_scale
        self.proj = nn.Sequential(nn.Linear(d_latent, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, latent):
        B, L, _ = latent.shape
        raw = self.proj(latent)
        theta = torch.tanh(raw[..., 0]) * math.pi
        phi = torch.tanh(raw[..., 1]) * math.pi
        r = self.r_scale * F.softplus(raw[..., 2])

        R = self.bond_length * L / (2 * math.pi)
        major_R = R + r * torch.cos(phi)
        x = major_R * torch.cos(theta)
        y = major_R * torch.sin(theta)
        z = r * torch.sin(phi)

        return torch.stack([x, y, z], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 完整模型
# ══════════════════════════════════════════════════════════════════════════════

class FullS10Model(nn.Module):
    """完整版S10：稀疏等变encoder + S8 refine + inv扩散"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or FullS10Config()
        c = self.config
        self.use_coord_diffusion = getattr(c, 'use_coord_diffusion', True)

        # Encoder输出分离的inv和eq
        self.encoder = EquivariantEncoder(c)

        # Latent从分离的irrep投影（支持多尺度）
        self.latent = EquivariantLatent(
            c.d_model_inv, c.d_model_eq, c.d_inv, c.d_eq,
            use_multiscale=getattr(c, 'use_multiscale_latent', True),
        )

        # Pair Track (Evoformer lightweight: triangular update + attention)
        if c.use_pair_track:
            pair_cfg = PairTrackConfig(
                d_pair=c.d_pair,
                n_layers=c.pair_track_layers,
                n_heads=c.pair_n_heads,
                d_ffn=c.pair_d_ffn,
                dropout=c.pair_dropout,
            )
            self.pair_track = PairTrack(pair_cfg, d_node=c.d_model_inv)
            # 从 pair → node 融合: 对每个 node i, 汇总所有 pair (i,j) 的信息
            self.pair_to_node = nn.Sequential(
                nn.Linear(c.d_pair, c.d_model_inv),
                nn.GELU(),
                nn.Linear(c.d_model_inv, c.d_model_inv),
            )
            self.node_norm_after_pair = nn.LayerNorm(c.d_model_inv)
        else:
            self.pair_track = None
            self.pair_to_node = None

        # S8 Refine（用node_repr_inv算注意力，对node_repr_eq加权）
        if c.use_s8_refine:
            self.s8_refine = EquivariantS8Refine(
                c.d_model_inv,
                c.d_model_eq,
                n_layers=c.n_refine_layers,
                nhead=c.refine_nhead,
                dropout=c.refine_dropout
            )
        else:
            self.s8_refine = None

        self.diffusion = InvDiffusion(c.d_inv, c.n_diffusion_steps)
        self.coord_head = TorusCoordHead(c.d_inv + c.d_eq, c.bond_length, c.r_scale)

        # [v4] CoordDiffusion (坐标空间扩散，替代 InvDiffusion)
        if c.use_coord_diffusion:
            from .coord_diffusion import CoordDiffusion
            self.coord_diffusion = CoordDiffusion(
                d_inv=c.d_inv,
                d_eq=c.d_eq // 2,  # flat eq: (B,L,d_eq) → reshape (B,L,d_eq//2,2)
                d_coord_hidden=c.d_coord_hidden,
                n_steps=c.n_diffusion_steps,
                cfg_dropout_prob=c.cfg_dropout_prob,
                use_dynamic_anchors=c.use_dynamic_anchors,
                anchor_ratio=c.anchor_ratio,
            )
        else:
            self.coord_diffusion = None

        # [v4] Contact auxiliary head
        if c.use_contact_aux:
            from .contact_map_aux_head import ContactMapAuxHead
            self.contact_aux_head = ContactMapAuxHead(d_inv=c.d_inv, d_hidden=64)
        else:
            self.contact_aux_head = None

        # [v4] Stop-Gradient toggle
        self.detach_latent = False

        # [v4] Kendall uncertainty weighting
        self.use_kendall_uw = c.use_kendall_uw
        if c.use_kendall_uw:
            self.uncertainty_log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in c.kendall_loss_keys
            })

        # 动态构象系综生成器
        if c.use_dynamic_ensemble:
            self.dynamic_config = DynamicEnsembleConfig(
                temperatures=c.ensemble_temperatures,
                n_samples_per_temp=c.n_samples_per_temp,
                use_potential_guidance=c.use_potential_guidance,
                potential_weight=c.potential_weight,
                potential_steps=c.potential_refine_steps,
                use_markov_trajectory=c.use_markov_trajectory,
                n_trajectory_steps=c.n_trajectory_steps,
                rmsd_kernel_sigma=c.rmsd_kernel_sigma,
            )
            # 注意：DynamicEnsembleGenerator 需要完整模型引用，延迟初始化
            self._dynamic_generator = None
        else:
            self._dynamic_generator = None

    @property
    def dynamic_generator(self):
        """延迟初始化动态构象系综生成器"""
        if self._dynamic_generator is None and self.config.use_dynamic_ensemble:
            self._dynamic_generator = DynamicEnsembleGenerator(self, self.dynamic_config)
        return self._dynamic_generator

    def forward(self, seq_tokens, coords_target=None, ss_tokens=None,
                bpp_matrix=None, msa_tokens=None, contact_target=None):
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 1. Encoder → 分离的irrep + pair_probs（SS token / bpp / MSA 在 encoder 内部处理）
        node_repr_inv, node_repr_eq, pair_probs = self.encoder(
            seq_tokens, bpp_matrix=bpp_matrix, msa_tokens=msa_tokens,
            ss_tokens=ss_tokens,
        )

        # 1.5. Pair Track: 构建稀疏 pair representation → 三角更新 → 融合回 node
        if self.pair_track is not None:
            z = self.pair_track.init_from_node(node_repr_inv)  # (B, L, K, d_pair)
            z = self.pair_track(z)
            pair_enhance = z.mean(dim=2)  # (B, L, d_pair)
            pair_enhance = self.pair_to_node(pair_enhance)  # (B, L, d_model)
            node_repr_inv = self.node_norm_after_pair(node_repr_inv + pair_enhance[:, :, :node_repr_inv.shape[-1]])
            node_repr_eq = self.s8_refine(node_repr_inv, node_repr_eq)

        # 2. Latent投影
        latent_inv, latent_eq = self.latent(node_repr_inv, node_repr_eq)

        # 3. 扩散：CoordDiffusion（坐标空间）或 InvDiffusion（latent 空间）
        diff_loss = None
        anchor_aux_loss = None
        pred_coords = None

        if self.use_coord_diffusion and self.coord_diffusion is not None:
            # [v4] CoordDiffusion: 直接在 (B,L,3) 坐标空间扩散
            cond_inv_d = latent_inv.detach() if self.detach_latent else latent_inv
            # flat eq (B,L,d_eq) → equivariant (B,L,d_eq//2,2)
            eq_flat = latent_eq.detach() if self.detach_latent else latent_eq
            cond_eq_d = eq_flat.reshape(B, L, -1, 2)
            if self.training and coords_target is not None:
                diff_loss, noise_pred, pred_coords = self.coord_diffusion(
                    coords_target, cond_inv=cond_inv_d, cond_eq=cond_eq_d,
                    return_noise_pred=True, return_x0_pred=True,
                )
                # anchor_aux_loss: 用 encoder 输出的 pair_probs 监督 dynamic anchor
                if pair_probs is not None:
                    anchor_aux_loss = self.coord_diffusion.anchor_aux_loss(
                        cond_inv_d, cond_eq_d, pair_probs,
                    )
                else:
                    anchor_aux_loss = None
            else:
                pred_coords = self.coord_diffusion.generate(
                    cond_inv_d, cond_eq_d, cfg_scale=1.0,
                )
        else:
            # Fallback: InvDiffusion + TorusCoordHead（旧路径）
            if self.training and coords_target is not None:
                latent_inv_noisy, diff_loss = self.diffusion(latent_inv, return_loss=True)
                latent_final = torch.cat([latent_inv_noisy, latent_eq], dim=-1)
            else:
                latent_inv_sampled = self.diffusion.sample(B, L, device)
                latent_final = torch.cat([latent_inv_sampled, latent_eq], dim=-1)
            pred_coords = self.coord_head(latent_final)

        # 4. Contact auxiliary head
        contact_pred = None
        if self.contact_aux_head is not None and (
            self.config.use_contact_aux or self.detach_latent
        ):
            contact_pred = self.contact_aux_head(latent_inv)

        # 5. Closure distance
        closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)

        result = {
            'coords': pred_coords,
            'latent_inv': latent_inv,
            'latent_eq': latent_eq,
            'closure_dist': closure_dist,
        }

        # 6. Losses
        if self.training and coords_target is not None:
            losses = self._compute_losses(
                pred_coords, coords_target, closure_dist,
                pair_probs=pair_probs, contact_target=contact_target,
            )
            if diff_loss is not None:
                losses['diffusion'] = diff_loss
            if anchor_aux_loss is not None:
                losses['anchor_aux'] = anchor_aux_loss

            # [v4] Kendall uncertainty weighting
            if self.use_kendall_uw:
                losses = self._kendall_weighted_loss(losses)

            result['losses'] = losses

        return result

    def _compute_losses(self, coords, target, closure_dist,
                         pair_probs=None, contact_target=None):
        """计算训练 loss

        Args:
            coords: (B, L, 3) 预测坐标
            target: (B, L, 3) 目标坐标
            closure_dist: (B,) 首尾距离
            pair_probs: (B, L, L) encoder 输出的配对概率（可选）
            contact_target: (B, L, L) 外部 contact map target（可选，None 时用 bpp 替代）
        """
        B, L, _ = coords.shape
        device = coords.device
        losses = {}

        # 1. Coord MSE
        losses['coord'] = F.mse_loss(coords, target)

        # 2. Bond loss (相邻残基距离趋向 bond_length)
        bond_dists = torch.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        losses['bond'] = F.mse_loss(bond_dists, torch.full_like(bond_dists, self.config.bond_length))

        # 3. Closure loss (首尾距离趋向 bond_length)
        losses['closure'] = F.mse_loss(closure_dist, torch.full_like(closure_dist, self.config.bond_length))

        # 4. Stereo loss — RNA 骨架角度约束
        #    RNA A-form: 键角 ~110-120°, 二面角分 6 个区域
        #    简化: 连续三原子键角 cos 应在 cos(120°)=-0.5 附近 (宽泛约束)
        if L >= 3:
            v1 = coords[:, :-2] - coords[:, 1:-1]  # (B, L-2, 3)
            v2 = coords[:, 2:] - coords[:, 1:-1]   # (B, L-2, 3)
            cos_angle = F.cosine_similarity(v1, v2, dim=-1)  # (B, L-2)
            # 软约束: 角度在 100°-140° 范围内 (cos ∈ [-0.77, -0.64])
            target_cos = -0.5  # 120°
            losses['stereo'] = F.mse_loss(cos_angle, torch.full_like(cos_angle, target_cos))

        # 5. Physics pairing — native pair 距离约束
        #    用 bpp_matrix 或 pair_probs 识别 native pairs，约束 predicted 距离
        if pair_probs is not None:
            # 取 top 配对（概率 > 0.3 的位置视为 native pair）
            with torch.no_grad():
                native_mask = (pair_probs > 0.3).float()  # (B, L, L)
                # 只取上三角避免重复
                tri_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
                native_mask = native_mask * tri_mask.unsqueeze(0)
                n_pairs = native_mask.sum(dim=(1, 2)).clamp(min=1)  # (B,)

            # 计算所有位置对距离
            coords_i = coords.unsqueeze(2)  # (B, L, 1, 3)
            coords_j = coords.unsqueeze(1)  # (B, 1, L, 3)
            dist_matrix = torch.norm(coords_i - coords_j, dim=-1)  # (B, L, L)

            # native pair 距离应接近 bond_length
            pair_dist = dist_matrix * native_mask
            target_dist = torch.full_like(pair_dist, self.config.bond_length)
            losses['physics_pairing'] = (pair_dist - target_dist).pow(2).sum(dim=(1, 2)) / n_pairs
            losses['physics_pairing'] = losses['physics_pairing'].mean()
        else:
            losses['physics_pairing'] = torch.tensor(0.0, device=device)

        # 6. Contact aux — bpp 做 soft contact target
        #    bpp > 0.5 视为 contact，用 BCE 损失
        if contact_target is not None:
            # 外部提供 contact target
            contact_score = torch.sigmoid(
                torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
            )  # (B, L, L), 距离越小 score 越高
            losses['contact_aux'] = F.binary_cross_entropy(
                contact_score, contact_target.float(),
            )
        elif pair_probs is not None:
            # 用 pair_probs 作为 soft contact target（bpp > 0.5 的区域）
            with torch.no_grad():
                contact_target = (pair_probs > 0.5).float()
            coord_dist = torch.norm(
                coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1,
            )  # (B, L, L)
            # 转为 contact score: 距离越近 → score 越高
            contact_pred = torch.exp(-coord_dist / self.config.bond_length)
            losses['contact_aux'] = F.binary_cross_entropy(
                contact_pred, contact_target,
            )
        else:
            losses['contact_aux'] = torch.tensor(0.0, device=device)

        # 总 loss
        losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))

        return losses

    def _kendall_weighted_loss(self, losses: dict) -> dict:
        """Kendall uncertainty weighting: L = Σ 0.5·exp(-logσ²)·L_i + 0.5·logσ²"""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        weighted = {}
        for key, log_var in self.uncertainty_log_vars.items():
            if key in losses:
                loss_val = losses[key]
                precision = torch.exp(-log_var)
                weighted[key] = 0.5 * precision * loss_val + 0.5 * log_var
                total = total + weighted[key]
        # 未在 Kendall 中的 loss 保持原始值直接相加
        for key, val in losses.items():
            if key not in self.uncertainty_log_vars:
                total = total + val
        weighted['total'] = total
        return weighted

    @torch.no_grad()
    def sample(self, seq_tokens, bpp_matrix=None, msa_tokens=None):
        """推理: 返回 coords + BSJ confidence.

        Args:
            seq_tokens: (B, L) 序列 token
            bpp_matrix: (L, L) 或 (B, L, L) ViennaRNA bpp（可选）
            msa_tokens: (B, N_rep, L) MSA tokens（可选）

        Returns:
            dict: {'coords': (B, L, 3), 'bsj_confidence': (B,), 'closure_dist': (B,)}
        """
        self.eval()
        result = self.forward(seq_tokens, bpp_matrix=bpp_matrix, msa_tokens=msa_tokens)
        coords = result['coords']

        # BSJ confidence: 从 closure distance 衍生
        closure_dist = result['closure_dist']  # (B,)
        bsj_confidence = torch.exp(-closure_dist / 10.0)  # 越近越好

        return {
            'coords': coords,
            'bsj_confidence': bsj_confidence,
            'closure_dist': closure_dist,
        }

    def generate_ensemble(self, seq_tokens, bpp_matrix=None, is_circular=True, seed=None):
        """生成动态构象系综

        Args:
            seq_tokens: (B, L) 序列 token
            bpp_matrix: (L, L) 或 (B, L, L) ViennaRNA bpp（可选）
            is_circular: 是否环化
            seed: 随机种子

        Returns:
            DynamicEnsembleResult:
                .conformations: (N, L, 3) 静态系综
                .physically_refined: (N, L, 3) 势能精炼构象
                .trajectory: (T, L, 3) 马尔可夫轨迹
                .transition_matrix: (N, N) 转移概率
                .free_energy_surface: (N, N) 自由能面
                .rmsf: (L,) 柔性分析
        """
        if not self.config.use_dynamic_ensemble:
            raise RuntimeError("use_dynamic_ensemble=False, cannot generate ensemble")

        return self.dynamic_generator.generate(
            seq_tokens,
            bpp_matrix=bpp_matrix,
            is_circular=is_circular,
            seed=seed,
        )

    def generate_trajectory(self, seq_tokens, bpp_matrix=None, n_steps=1000, is_circular=True, seed=None):
        """生成动态轨迹（快捷方法）

        Returns:
            trajectory: (T, L, 3) 平滑轨迹
            state_sequence: 状态序列
        """
        result = self.generate_ensemble(seq_tokens, bpp_matrix=bpp_matrix, is_circular=is_circular, seed=seed)
        return result.trajectory