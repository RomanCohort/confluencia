"""
scheme10_full.py — S10完整版：稀疏等变encoder + S8 refine + inv扩散

架构：
  序列 → S10稀疏encoder → [inv | eq]分离 → inv扩散 + S8 refine(eq) → coords
           ↑                    ↑              ↑              ↑
       精确等变            等变latent      扩散不破坏     长程依赖增强
       O(L·K)                            等变性
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


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FullS10Config:
    """完整版S10配置"""
    n_tokens: int = 5
    d_model: int = 256

    # 稀疏GNN
    d_edge: int = 64
    k_theta: int = 2
    k_phi: int = 1
    n_layers: int = 4
    K_sparse: int = 60       # K的最大值（kernel参数上限）
    K_sparse_ratio: float = 0.04  # 动态K：K = L * ratio
    K_sparse_min: int = 30   # 动态K：最小K=30
    dropout: float = 0.1

    # irrep分离（新增）
    d_model_inv: int = 64    # encoder输出中degree 0的维度
    d_model_eq: int = 192    # encoder输出中degree 1+的维度
    # 注意：d_model_inv + d_model_eq 应该 <= d_model

    # Latent
    d_inv: int = 32
    d_eq: int = 32

    # S8 Refine（只处理eq部分，增强长程依赖）
    use_s8_refine: bool = True
    n_refine_layers: int = 2
    refine_nhead: int = 4
    refine_dropout: float = 0.1

    # 扩散（只对inv）
    n_diffusion_steps: int = 100

    # 结构
    bond_length: float = 5.9
    r_scale: float = 0.5
    n_edge_cats: int = 5

    # Pair Track (Evoformer lightweight)
    use_pair_track: bool = True
    pair_track_layers: int = 2       # 2 层 (轻量)
    d_pair: int = 64                 # pair representation 维度 (轻量)
    pair_n_heads: int = 4
    pair_d_ffn: int = 128
    pair_dropout: float = 0.1


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

    def forward(self, x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx):
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

    def forward(self, x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse):
        msg = self.kernel(x, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse, topk_idx)
        x_new = self.update(torch.cat([x, msg], dim=-1))
        return self.norm(x + x_new)


# ══════════════════════════════════════════════════════════════════════════════
# 等变Encoder（输出分离的inv和eq）
# ══════════════════════════════════════════════════════════════════════════════

class EquivariantEncoder(nn.Module):
    """等变encoder（稀疏steerable kernel）+ irrep分离"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.n_tokens, config.d_model)
        self.tpe = TorusPositionalEncoding(config.d_model, n_harmonics=16, dropout=config.dropout)

        self.layers = nn.ModuleList([
            SparseEquivariantGNNLayer(config) for _ in range(config.n_layers)
        ])

        # irrep分离投影（新增）
        # degree 0部分：可以用非线性（因为后续是inv操作）
        self.to_inv = nn.Sequential(
            nn.Linear(config.d_model, config.d_model_inv * 2),
            nn.GELU(),
            nn.Linear(config.d_model_inv * 2, config.d_model_inv),
        )
        # degree 1+部分：只能用线性（保等变性）
        self.to_eq = nn.Linear(config.d_model, config.d_model_eq, bias=False)

    def forward(self, seq_tokens, pair_probs=None):
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 动态K：根据序列长度
        K_actual = max(self.config.K_sparse_min, int(L * self.config.K_sparse_ratio))
        K_actual = min(K_actual, L, self.config.K_sparse)

        x = self.token_embed(seq_tokens)
        x = self.tpe(x, seq_len=L)

        # 构建稀疏邻居
        if pair_probs is None:
            topk_idx = torch.randint(0, L, (B, L, K_actual), device=device)
        else:
            _, topk_idx = pair_probs.topk(K_actual, dim=-1)

        # 构建角度差
        pos = torch.arange(L, device=device, dtype=torch.float32)
        theta = 2 * math.pi * pos / L
        theta_j = theta[topk_idx]
        theta_i = theta.view(1, L, 1)
        delta_theta_sparse = theta_i - theta_j
        delta_phi_sparse = torch.zeros_like(delta_theta_sparse)
        edge_cat_sparse = torch.zeros(B, L, K_actual, device=device, dtype=torch.long)

        # 稀疏GNN
        for layer in self.layers:
            x = layer(x, topk_idx, delta_theta_sparse, delta_phi_sparse, edge_cat_sparse)

        # irrep分离（新增）
        node_repr_inv = self.to_inv(x)   # (B, L, d_model_inv) - degree 0
        node_repr_eq = self.to_eq(x)     # (B, L, d_model_eq) - degree 1+

        return node_repr_inv, node_repr_eq


# ══════════════════════════════════════════════════════════════════════════════
# Latent分离（接收已经分离的irrep）
# ══════════════════════════════════════════════════════════════════════════════

class EquivariantLatent(nn.Module):
    """等变latent：从分离的irrep投影到latent"""

    def __init__(self, d_model_inv, d_model_eq, d_inv, d_eq):
        super().__init__()
        # 从node_repr_inv投影到latent_inv（可以非线性）
        self.inv_proj = nn.Sequential(
            nn.Linear(d_model_inv, d_inv * 2),
            nn.GELU(),
            nn.Linear(d_inv * 2, d_inv),
        )
        # 从node_repr_eq投影到latent_eq（只能线性）
        self.eq_proj = nn.Linear(d_model_eq, d_eq, bias=False)

    def forward(self, node_repr_inv, node_repr_eq):
        """接收已经分离的irrep"""
        return self.inv_proj(node_repr_inv), self.eq_proj(node_repr_eq)


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

        # Encoder输出分离的inv和eq
        self.encoder = EquivariantEncoder(c)

        # Latent从分离的irrep投影
        self.latent = EquivariantLatent(c.d_model_inv, c.d_model_eq, c.d_inv, c.d_eq)

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

    def forward(self, seq_tokens, coords_target=None):
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 1. Encoder → 分离的irrep
        node_repr_inv, node_repr_eq = self.encoder(seq_tokens)

        # 1.5. Pair Track: 构建稀疏 pair representation → 三角更新 → 融合回 node
        if self.pair_track is not None:
            # 从 encoder 输出构建稀疏 pair representation (标量, 不影响等变性)
            # 用 inv 维度 (degree 0, 旋转不变) 构建 pair
            z = self.pair_track.init_from_node(node_repr_inv)  # (B, L, K, d_pair)
            z = self.pair_track(z)  # N 层三角更新
            # 融合回 node: 对 K 维度求平均 (每个 i 汇总 K 近邻的 pair 信息)
            pair_enhance = z.mean(dim=2)  # (B, L, d_pair)
            pair_enhance = self.pair_to_node(pair_enhance)  # (B, L, d_model)
            # 残差融合: pair_enhance 加到 node_repr_inv (不变量, 保等变性)
            node_repr_inv = self.node_norm_after_pair(node_repr_inv + pair_enhance[:, :, :node_repr_inv.shape[-1]])
            node_repr_eq = self.s8_refine(node_repr_inv, node_repr_eq)

        # 3. Latent投影
        latent_inv, latent_eq = self.latent(node_repr_inv, node_repr_eq)

        # 4. Diffusion：只对inv通道
        if self.training and coords_target is not None:
            latent_inv_noisy, diff_loss = self.diffusion(latent_inv, return_loss=True)
            latent_final = torch.cat([latent_inv_noisy, latent_eq], dim=-1)
        else:
            latent_inv_sampled = self.diffusion.sample(B, L, device)
            latent_final = torch.cat([latent_inv_sampled, latent_eq], dim=-1)
            diff_loss = None

        # 5. 结构头
        coords = self.coord_head(latent_final)

        result = {
            'coords': coords,
            'latent_inv': latent_inv,
            'latent_eq': latent_eq,
            'closure_dist': torch.norm(coords[:, 0] - coords[:, -1], dim=-1),
        }
        if diff_loss is not None:
            result['diff_loss'] = diff_loss
        return result

    @torch.no_grad()
    def sample(self, seq_tokens):
        """推理: 返回 coords + BSJ confidence.

        Returns:
            dict: {'coords': (B, L, 3), 'bsj_confidence': (B,), 'closure_dist': (B,)}
        """
        self.eval()
        result = self.forward(seq_tokens)
        coords = result['coords']

        # BSJ confidence: 从 closure distance 衍生
        closure_dist = result['closure_dist']  # (B,)
        bsj_confidence = torch.exp(-closure_dist / 10.0)  # 越近越好

        return {
            'coords': coords,
            'bsj_confidence': bsj_confidence,
            'closure_dist': closure_dist,
        }