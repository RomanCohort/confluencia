"""torusfold_rhofold.py — RhoFold+ → S10 串联架构.

架构:
  circRNA 序列 → RhoFold+ RNA FM encoder (冻结, 640d)
    ↓
  rhofold_proj (640→256, 可训练)
    ↓
  S10 等变解码器 (可训练):
    EquivariantGNNLayer ×4 → S8Refine → Latent
    → CoordDiffusion (DDIM 100步, DynamicAnchor)
    → physics_refine (Adam 100步)
    ↓
  CG 3-bead 坐标 (B, L, 3)

用法:
  model = TorusFoldRhoFold()
  result = model(seq_tokens, refine=True)
  coords = result['coords']  # (B, L, 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rhofold_backbone import RhoFoldBackbone
from .bsj_fape import BSJFAPELoss
from .scheme10_equivariant import StrictlyEquivariantS10, EquivariantS10Config


@dataclass
class TorusFoldRhoFoldConfig:
    """RhoFold+ → S10 串联配置."""
    # Backbone
    freeze_backbone: bool = True
    freeze_layers: int = 9  # 冻结前 N 层

    # Loss
    w_bsj_fape: float = 2.0
    w_bond: float = 0.5
    w_closure: float = 5.0
    bond_length: float = 5.9

    # S10 解码器配置
    s10_config: EquivariantS10Config = field(default_factory=EquivariantS10Config)


class TorusFoldRhoFold(nn.Module):
    """RhoFold+ → S10 串联模型.

    RhoFold+ RNA FM 作为冻结编码器, S10 等变解码器生成坐标.
    """

    def __init__(self, config: Optional[TorusFoldRhoFoldConfig] = None):
        super().__init__()
        self.config = config or TorusFoldRhoFoldConfig()
        c = self.config

        # 1. RhoFold+ backbone (冻结)
        self.backbone = RhoFoldBackbone(
            freeze_layers=c.freeze_layers,
            use_pair_repr=True,
        )

        # 2. S10 等变解码器 (RhoFold+ 特征作为输入)
        self.s10_decoder = StrictlyEquivariantS10(
            c.s10_config, use_rhofold_input=True,
        )

        # 3. BSJ FAPE
        self.bsj_fape = BSJFAPELoss(bsj_margin=10)

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

        # 1. RhoFold+ 编码
        node_repr, pair_repr = self.backbone(seq_tokens)
        # node_repr: (B, L, 640), pair_repr: (B, L, L, 128)

        # 2. S10 解码
        s10_kwargs = dict(
            seq_tokens=seq_tokens,
            rhofold_node_repr=node_repr,
            rhofold_pair_repr=pair_repr,
            refine=refine,
            refine_steps=refine_steps,
            lengths=torch.full((B,), L, device=seq_tokens.device, dtype=torch.long),
        )
        if self.training and coords_target is not None:
            s10_kwargs.update(target_coords=coords_target, return_loss=True, return_coords=True)
        else:
            s10_kwargs.update(return_loss=False, return_coords=True)

        s10_output = self.s10_decoder(**s10_kwargs)

        # 解析 S10 输出: (diffusion_loss, pred_coords, contact_pred) 或 (pred_coords,)
        if isinstance(s10_output, tuple):
            if len(s10_output) == 3:
                diffusion_loss, coords, _contact_pred = s10_output
            else:
                diffusion_loss = None
                coords = s10_output[0]
        else:
            diffusion_loss = None
            coords = s10_output

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

        # Closure loss
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
    def sample(self, seq_tokens: torch.Tensor, refine: bool = True) -> Dict[str, torch.Tensor]:
        """推理: 返回 coords + 置信度."""
        self.eval()
        result = self.forward(seq_tokens, refine=refine)
        return {
            'coords': result['coords'],
            'bsj_confidence': result['bsj_confidence'],
            'closure_dist': result['closure_dist'],
        }

    def freeze_backbone(self):
        """冻结 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen")

    def unfreeze_backbone_last_n(self, n: int = 3):
        """Unfreeze backbone 最后 N 层."""
        layers = list(self.backbone.rna_fm.layers)
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Backbone: unfrozen last {n} layers")
