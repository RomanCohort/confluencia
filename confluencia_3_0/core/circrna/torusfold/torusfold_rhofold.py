"""torusfold_rhofold.py — TorusFold + RhoFold+ 完整模型.

架构:
  circRNA 序列 → RhoFold+ RNA FM encoder (冻结) → node_repr + pair_repr
    ↓
  Projection Layer (可训练): 降维到 d_model
    ↓
  PairTrack (可训练): 三角一致性
    ↓
  CG decoder (可训练): → CG 3-bead 坐标
    ↓
  BSJ FAPE: 置信度 + 损失

用法:
  model = TorusFoldRhoFold(freeze_backbone=True)
  result = model(seq_tokens)
  coords = result['coords']  # (B, L, 3)
  bsj_conf = result['bsj_confidence']  # (B,)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rhofold_backbone import RhoFoldBackbone, RhoFoldBackboneLight
from .pair_track import PairTrack, PairTrackConfig
from .cg_decoder import CGDecoder, CGDecoderEquivariant
from .bsj_fape import BSJFAPELoss, BSJConfidence


@dataclass
class TorusFoldRhoFoldConfig:
    """TorusFold + RhoFold+ 配置."""
    # Backbone
    freeze_backbone: bool = True
    freeze_layers: int = 9  # 冻结前 N 层
    use_pair_repr: bool = True  # 是否用 E2Eformer pair_repr

    # Projection
    d_node_rna_fm: int = 640  # RNA FM 输出维度
    d_node: int = 256  # 投影后维度
    d_pair_rna_fm: int = 128  # RNA FM pair_repr 维度
    d_pair: int = 64  # 投影后维度

    # PairTrack
    pair_track_layers: int = 2
    pair_n_heads: int = 4
    pair_d_ffn: int = 128
    pair_dropout: float = 0.1
    k_neighbors: int = 30  # 稀疏近邻数

    # CG Decoder
    use_equivariant_decoder: bool = False  # Phase 2 用等变 decoder
    decoder_hidden: int = 256
    decoder_layers: int = 2

    # Loss
    w_bsj_fape: float = 2.0
    w_bond: float = 0.5
    w_closure: float = 5.0
    bond_length: float = 5.9


class TorusFoldRhoFold(nn.Module):
    """TorusFold + RhoFold+ 完整模型.

    Phase 1: 冻结 backbone, 训 CG decoder + PairTrack
    Phase 2: unfreeze 最后 3 层, 微调整个模型
    """

    def __init__(self, config: Optional[TorusFoldRhoFoldConfig] = None):
        super().__init__()
        self.config = config or TorusFoldRhoFoldConfig()
        c = self.config

        # 1. RhoFold+ backbone
        if c.use_pair_repr:
            self.backbone = RhoFoldBackbone(
                freeze_layers=c.freeze_layers,
                use_pair_repr=True,
            )
        else:
            self.backbone = RhoFoldBackboneLight(
                freeze_layers=c.freeze_layers,
            )

        # 2. Projection layers
        self.node_proj = nn.Sequential(
            nn.Linear(c.d_node_rna_fm, c.d_node),
            nn.LayerNorm(c.d_node),
            nn.GELU(),
        )

        if c.use_pair_repr:
            self.pair_proj = nn.Sequential(
                nn.Linear(c.d_pair_rna_fm, c.d_pair),
                nn.LayerNorm(c.d_pair),
                nn.GELU(),
            )

        # 3. PairTrack
        pair_cfg = PairTrackConfig(
            d_pair=c.d_pair,
            n_layers=c.pair_track_layers,
            n_heads=c.pair_n_heads,
            d_ffn=c.pair_d_ffn,
            dropout=c.pair_dropout,
        )
        self.pair_track = PairTrack(pair_cfg, d_node=c.d_node)

        # pair → node 融合: 对每个 node i, 汇总所有 pair (i,j) 的信息
        self.pair_to_node = nn.Sequential(
            nn.Linear(c.d_pair, c.d_node),
            nn.GELU(),
            nn.Linear(c.d_node, c.d_node),
        )
        self.node_norm_after_pair = nn.LayerNorm(c.d_node)

        # 4. CG Decoder
        if c.use_equivariant_decoder:
            self.cg_decoder = CGDecoderEquivariant(
                d_node=c.d_node,
                d_hidden=c.decoder_hidden,
                n_equiv_layers=c.decoder_layers,
            )
        else:
            self.cg_decoder = CGDecoder(
                d_node=c.d_node,
                d_hidden=c.decoder_hidden,
                n_layers=c.decoder_layers,
            )

        # 5. BSJ FAPE + Confidence
        self.bsj_fape = BSJFAPELoss(bsj_margin=10)
        self.bsj_confidence = BSJConfidence(temperature=5.0, bsj_margin=10)

        # 统计参数
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  TorusFoldRhoFold: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)")

    def forward(
        self,
        seq_tokens: torch.Tensor,
        coords_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            seq_tokens: (B, L) token IDs
            coords_target: (B, L, 3) 目标坐标 (训练时)

        Returns:
            dict: {
                'coords': (B, L, 3),
                'bsj_confidence': (B,),
                'closure_dist': (B,),
                'losses': dict (训练时),
            }
        """
        B, L = seq_tokens.shape

        # 1. RhoFold+ backbone
        node_repr, pair_repr = self.backbone(seq_tokens)
        # node_repr: (B, L, 640), pair_repr: (B, L, L, 128) 或 None

        # 2. Projection
        node_feat = self.node_proj(node_repr)  # (B, L, d_node)

        # 3. PairTrack
        if pair_repr is not None:
            pair_feat = self.pair_proj(pair_repr)  # (B, L, L, d_pair)
            # 从 RNA FM pair_repr 初始化稀疏 pair
            z = self.pair_track.init_from_rna_fm_pair(pair_feat)  # (B, L, K, d_pair)
        else:
            # fallback: 从 node 初始化
            z = self.pair_track.init_from_node(node_feat)  # (B, L, K, d_pair)

        z = self.pair_track(z)  # N 层 TriMulUpdate

        # 融合 pair → node
        pair_enhance = z.mean(dim=2)  # (B, L, d_pair)
        pair_enhance = self.pair_to_node(pair_enhance)  # (B, L, d_node)
        node_feat = self.node_norm_after_pair(node_feat + pair_enhance)

        # 4. CG Decoder
        coords = self.cg_decoder(node_feat)  # (B, L, 3)

        # 5. BSJ metrics
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)  # (B,)
        bsj_conf = torch.exp(-closure_dist / 10.0)  # (B,)

        result = {
            'coords': coords,
            'bsj_confidence': bsj_conf,
            'closure_dist': closure_dist,
        }

        # 6. Losses (训练时)
        if self.training and coords_target is not None:
            losses = self._compute_losses(coords, coords_target, closure_dist)
            result['losses'] = losses

        return result

    def _compute_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        closure_dist: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失."""
        c = self.config

        # Coord loss (L2)
        coord_loss = F.mse_loss(pred, target)

        # BSJ FAPE loss
        bsj_fape_loss = self.bsj_fape(pred, target)

        # Bond loss
        bonds = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1)
        bsj_bond = torch.norm(pred[:, 0] - pred[:, -1], dim=-1)
        all_bonds = torch.cat([bsj_bond.unsqueeze(1), bonds], dim=1)
        bond_loss = F.mse_loss(all_bonds, torch.full_like(all_bonds, c.bond_length))

        # Closure loss (标量距离)
        closure_loss = ((closure_dist - c.bond_length) ** 2).mean()

        # Total
        total_loss = (
            coord_loss
            + c.w_bsj_fape * bsj_fape_loss
            + c.w_bond * bond_loss
            + c.w_closure * closure_loss
        )

        return {
            'total': total_loss,
            'coord': coord_loss,
            'bsj_fape': bsj_fape_loss,
            'bond': bond_loss,
            'closure': closure_loss,
        }

    @torch.no_grad()
    def sample(self, seq_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理: 返回 coords + 置信度."""
        self.eval()
        result = self.forward(seq_tokens)
        return {
            'coords': result['coords'],
            'bsj_confidence': result['bsj_confidence'],
            'closure_dist': result['closure_dist'],
        }

    def freeze_backbone(self):
        """冻结 backbone (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen")

    def unfreeze_backbone_last_n(self, n: int = 3):
        """Unfreeze backbone 最后 N 层 (Phase 2)."""
        layers = list(self.backbone.rna_fm.layers)
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Backbone: unfrozen last {n} layers")
