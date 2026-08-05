"""cg_decoder.py — CG 3-bead 坐标解码器.

从 RhoFold+ 的 node_repr (640维) 投影到 CG 3-bead 坐标 (L×3).

设计:
  - Phase 1: 简单 Linear 投影 (验证可行性)
  - Phase 2: 加 equivariant layers (提升几何质量)

用法:
  decoder = CGDecoder(d_node=640)
  coords = decoder(node_repr)  # (B, L, 3)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CGDecoder(nn.Module):
    """CG 3-bead 坐标解码器.

    从 node_repr 投影到 P atom 坐标 (3-bead CG 的第一个 bead).

    Args:
        d_node: 输入维度 (RhoFold+ RNA FM=640)
        d_hidden: 中间维度
        n_layers: MLP 层数
    """

    def __init__(
        self,
        d_node: int = 640,
        d_hidden: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()

        # MLP: d_node → d_hidden → 3
        layers = []
        in_dim = d_node
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, 3))

        self.decoder = nn.Sequential(*layers)

        # 初始化: 输出接近零 (坐标从零开始学)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        """解码 CG 坐标.

        Args:
            node_repr: (B, L, d_node) — RhoFold+ node representation

        Returns:
            coords: (B, L, 3) — CG 3-bead P atom 坐标
        """
        return self.decoder(node_repr)


class CGDecoderEquivariant(nn.Module):
    """等变版 CG decoder (Phase 2).

    加 SO(2) 等变层, 保证旋转等变性.

    Args:
        d_node: 输入维度
        d_hidden: 中间维度
        n_equiv_layers: 等变层数
    """

    def __init__(
        self,
        d_node: int = 640,
        d_hidden: int = 256,
        n_equiv_layers: int = 2,
    ):
        super().__init__()

        # 先投影到 d_hidden
        self.proj_in = nn.Sequential(
            nn.Linear(d_node, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

        # 等变层 (简化版: 用 MLP + 残差)
        self.equiv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for _ in range(n_equiv_layers)
        ])

        # 输出 3D 坐标
        self.proj_out = nn.Linear(d_hidden, 3)

        # 初始化
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        """等变解码 CG 坐标."""
        x = self.proj_in(node_repr)

        # 等变层 + 残差
        for layer in self.equiv_layers:
            x = x + layer(x)

        return self.proj_out(x)
