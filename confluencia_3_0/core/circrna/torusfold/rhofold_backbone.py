"""rhofold_backbone.py — RhoFold+ RNA FM encoder 作为 TorusFold backbone.

核心思路:
  RhoFold+ 的 RNA FM encoder (12层 Transformer, 640维) 学到了:
  - 碱基配对倾向 (A-U, G-C, G-U)
  - 二级结构上下文 (stem/loop/bulge)
  - 序列保守性信号 (共进化)

  但 structure module 把这些特征"翻译"成了错误的坐标 (13 clash, 键角偏差大).
  因此:
  - ✅ 保留 RNA FM encoder (序列理解是它的强项)
  - ❌ 替换 structure module (全原子 decoder 不适用)
  - ✅ 加 CG decoder + PairTrack (物理合理性补丁)

架构:
  circRNA 序列 → RNA FM encoder (冻结) → node_repr + pair_repr
    ↓
  Projection Layer (可训练): 降维到 d_model
    ↓
  PairTrack (可训练): 三角一致性
    ↓
  CG decoder (可训练): → CG 3-bead 坐标

用法:
  backbone = RhoFoldBackbone(freeze_layers=9)
  node_repr, pair_repr = backbone(seq_tokens)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# RhoFold+ 路径
_RHOFOLD_ROOT = Path(__file__).resolve().parents[4] / "tools" / "RhoFold"
if str(_RHOFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_RHOFOLD_ROOT))


class RhoFoldBackbone(nn.Module):
    """RhoFold+ RNA FM encoder 作为 backbone.

    提取 RNA FM 的中间层输出 (node_repr) 和 E2Eformer 的 pair_repr.
    冻结/微调控制: freeze_layers 参数指定冻结前 N 层.

    Args:
        freeze_layers: 冻结前 N 层 (默认 9, 即最后 3 层可训练)
        use_pair_repr: 是否提取 E2Eformer 的 pair_repr (默认 True)
        device: 设备
    """

    def __init__(
        self,
        freeze_layers: int = 9,
        use_pair_repr: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.freeze_layers = freeze_layers
        self.use_pair_repr = use_pair_repr

        # 加载 RNA FM encoder
        from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
        self.rna_fm, self.alphabet = esm1b_rna_t12()
        self.rna_fm.eval()

        # 冻结前 N 层
        self._freeze_layers(freeze_layers)

        # 如果需要 pair_repr, 加载 E2Eformer
        if use_pair_repr:
            self._load_e2eformer()

        # RNA FM 输出维度
        self.d_node = 640  # RNA FM embed_dim
        self.d_pair = 128  # E2Eformer c_z

    def _freeze_layers(self, n_layers: int):
        """冻结 RNA FM 前 N 层."""
        # 冻结 embedding
        for param in self.rna_fm.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.rna_fm.embed_positions.parameters():
            param.requires_grad = False
        for param in self.rna_fm.emb_layer_norm_before.parameters():
            param.requires_grad = False
        for param in self.rna_fm.emb_layer_norm_after.parameters():
            param.requires_grad = False

        # 冻结前 N 层 Transformer
        for i, layer in enumerate(self.rna_fm.layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # 统计可训练参数
        trainable = sum(p.numel() for p in self.rna_fm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.rna_fm.parameters())
        print(f"  RNA FM: frozen={n_layers}/12 layers, trainable={trainable:,}/{total:,} params")

    def _load_e2eformer(self):
        """加载 E2Eformer 提取 pair_repr."""
        try:
            from rhofold.model.e2eformer import E2EformerStack
            from rhofold.model.embedders import MSANet, PairNet

            # 简化版: 只用 PairNet 提取 pair_repr
            # (不用完整 E2Eformer, 避免内存爆炸)
            self.pair_emb = PairNet(d_model=128, d_msa=5)
            print(f"  E2Eformer pair_repr: enabled (PairNet only)")
        except ImportError:
            print(f"  WARNING: E2Eformer not available, pair_repr disabled")
            self.use_pair_repr = False
            self.pair_emb = None

    def encode_sequence(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """RNA FM 编码序列.

        Args:
            seq_tokens: (B, L) token IDs

        Returns:
            node_repr: (B, L, 640) — RNA FM 隐藏层输出
        """
        with torch.no_grad() if self.freeze_layers > 0 else torch.enable_grad():
            results = self.rna_fm(
                seq_tokens,
                repr_layers=[12],
                return_contacts=False,
            )
            node_repr = results["representations"][12]  # (B, L, 640)
        return node_repr

    def encode_pair(self, seq_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """编码 pair representation.

        Args:
            seq_tokens: (B, L) token IDs

        Returns:
            pair_repr: (B, L, L, 128) — 残基对关系
        """
        if not self.use_pair_repr or self.pair_emb is None:
            return None

        with torch.no_grad() if self.freeze_layers > 0 else torch.enable_grad():
            # PairNet 期望 (B, K, L) 形状的 long tokens, K=1 (单序列)
            tokens_3d = seq_tokens.unsqueeze(1)  # (B, 1, L)
            pair_repr = self.pair_emb(tokens_3d)  # (B, L, L, 128)
        return pair_repr

    def forward(
        self,
        seq_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """RhoFold+ backbone forward.

        Args:
            seq_tokens: (B, L) token IDs (0=A, 1=U, 2=G, 3=C, 4=N)

        Returns:
            node_repr: (B, L, 640) — 每个残基的表征
            pair_repr: (B, L, L, 128) 或 None — 残基对关系
        """
        node_repr = self.encode_sequence(seq_tokens)
        pair_repr = self.encode_pair(seq_tokens)
        return node_repr, pair_repr

    def get_trainable_params(self) -> list:
        """获取可训练参数 (用于 optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


class RhoFoldBackboneLight(nn.Module):
    """轻量版: 只用 RNA FM, 不用 E2Eformer.

    更省内存, 适合 Phase 1 快速验证.
    """

    def __init__(self, freeze_layers: int = 9):
        super().__init__()
        self.d_node = 640

        from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
        self.rna_fm, self.alphabet = esm1b_rna_t12()
        self.rna_fm.eval()

        # 冻结前 N 层
        for i, layer in enumerate(self.rna_fm.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.rna_fm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.rna_fm.parameters())
        print(f"  RNA FM (light): frozen={freeze_layers}/12 layers, trainable={trainable:,}/{total:,} params")

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """只返回 node_repr (无 pair_repr)."""
        results = self.rna_fm(
            seq_tokens,
            repr_layers=[12],
            return_contacts=False,
        )
        return results["representations"][12]  # (B, L, 640)
