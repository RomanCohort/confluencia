"""torusfold_rhofold.py — TorusFold + RhoFold+ 完整模型.

架构 (串联模式, 默认):
  circRNA 序列 → RhoFold+ RNA FM encoder (冻结) → node_repr + pair_repr
    ↓
  S10 等变解码器 (可训练):
    ChiralityAwareEmbedding → 4x SparseGNN → S8Refine → Latent
    → CoordDiffusion (DDIM 100步) → physics_refine 100步
    ↓
  CG 3-bead 坐标

架构 (原模式, use_s10_decoder=False):
  circRNA 序列 → RhoFold+ RNA FM encoder (冻结) → node_repr + pair_repr
    ↓
  Projection → PairTrack → CG Decoder → CG 3-bead 坐标

用法:
  # 串联模式 (默认)
  model = TorusFoldRhoFold(TorusFoldRhoFoldConfig(use_s10_decoder=True))
  result = model(seq_tokens, refine=True)
  coords = result['coords']  # (B, L, 3)

  # 原模式
  model = TorusFoldRhoFold(TorusFoldRhoFoldConfig(use_s10_decoder=False))
  result = model(seq_tokens)
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
from .scheme10_equivariant import StrictlyEquivariantS10, EquivariantS10Config


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

    # S10 串联解码器
    use_s10_decoder: bool = True  # True: RhoFold+ → S10 (串联) | False: RhoFold+ → CG Decoder


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

        # 4. 解码器: S10 或 CG Decoder
        if c.use_s10_decoder:
            # 串联模式: RhoFold+ → S10 等变解码器
            s10_config = EquivariantS10Config()
            self.s10_decoder = StrictlyEquivariantS10(s10_config, use_rhofold_input=True)
            self.cg_decoder = None  # 不用 CG Decoder
        else:
            # 原模式: RhoFold+ → CG Decoder
            self.s10_decoder = None
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
        refine: bool = False,
        refine_steps: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            seq_tokens: (B, L) token IDs
            coords_target: (B, L, 3) 目标坐标 (训练时)
            refine: 推理时是否跑 physics_refine
            refine_steps: 精修步数

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

        # 2. 解码: S10 串联 或 CG Decoder
        if self.s10_decoder is not None:
            # 串联模式: RhoFold+ → S10 等变解码器
            s10_input = {
                'seq_tokens': seq_tokens,
                'rhofold_node_repr': node_repr,
                'rhofold_pair_repr': pair_repr,
                'refine': refine,
                'refine_steps': refine_steps,
                'lengths': torch.full((B,), L, device=seq_tokens.device, dtype=torch.long),
            }
            if self.training and coords_target is not None:
                s10_input['target_coords'] = coords_target
                s10_input['return_loss'] = True
                s10_input['return_coords'] = True
            else:
                s10_input['return_loss'] = False
                s10_input['return_coords'] = True

            s10_output = self.s10_decoder(**s10_input)

            # S10 输出格式: (diffusion_loss, pred_coords, contact_pred) 或 (pred_coords,)
            if isinstance(s10_output, tuple):
                if len(s10_output) == 3:
                    diffusion_loss, coords, contact_pred = s10_output
                else:
                    diffusion_loss = None
                    coords = s10_output[0]
                    contact_pred = None
            else:
                diffusion_loss = None
                coords = s10_output
                contact_pred = None
        else:
            # 原模式: RhoFold+ → PairTrack → CG Decoder
            node_feat = self.node_proj(node_repr)  # (B, L, d_node)

            if pair_repr is not None:
                pair_feat = self.pair_proj(pair_repr)  # (B, L, L, d_pair)
                z = self.pair_track.init_from_rna_fm_pair(pair_feat)
            else:
                z = self.pair_track.init_from_node(node_feat)

            z = self.pair_track(z)
            pair_enhance = z.mean(dim=2)
            pair_enhance = self.pair_to_node(pair_enhance)
            node_feat = self.node_norm_after_pair(node_feat + pair_enhance)
            coords = self.cg_decoder(node_feat)
            diffusion_loss = None

        # 3. BSJ metrics
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)  # (B,)
        bsj_conf = torch.exp(-closure_dist / 10.0)  # (B,)

        result = {
            'coords': coords,
            'bsj_confidence': bsj_conf,
            'closure_dist': closure_dist,
        }

        # 4. Losses (训练时)
        if self.training and coords_target is not None:
            losses = self._compute_losses(coords, coords_target, closure_dist)
            if diffusion_loss is not None:
                losses['diffusion_loss'] = diffusion_loss
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
