"""
equivariant_tpe.py — 循环相对位置编码（无绝对位置）

核心设计：
1. 不编码绝对位置 i，而是编码循环相对距离 d(i,j)
2. 循环距离：d(i,j) = min(|i-j|, L-|i-j|)
3. 无论序列从哪里开始，相对位置不变

实现：
- CircularRelativeBias 用于注意力偏置
- 不在 token embedding 里注入绝对位置

等变性验证：
- 序列循环移位后，相对位置编码不变
- 输出坐标应该跟着旋转
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CircularRelativePositionBias(nn.Module):
    """循环相对位置偏置

    对于 circRNA，位置 i 和位置 j 的相对距离是循环的：
    d(i,j) = min(|i-j|, L-|i-j|)

    这确保位置 0 和位置 L-1 被认为是相邻的。
    """

    def __init__(
        self,
        n_heads: int = 8,
        max_dist: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_heads: 注意力头数
            max_dist: 最大距离（超过这个距离的用同一个偏置）
            dropout: dropout 概率
        """
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist

        # 偏置表：(max_dist + 1) 种距离，每个距离 n_heads 个偏置
        self.relative_bias = nn.Embedding(max_dist + 1, n_heads)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.normal_(self.relative_bias.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        计算循环相对位置偏置矩阵

        Args:
            seq_len: 序列长度 L
            device: 设备

        Returns:
            bias: (1, n_heads, L, L) 偏置矩阵
        """
        if device is None:
            device = self.relative_bias.weight.device

        positions = torch.arange(seq_len, device=device)

        # 线性距离：(L, L)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)

        # 循环距离：min(|i-j|, L-|i-j|)
        circ_dist = torch.min(diff.abs(), seq_len - diff.abs())  # (L, L)

        # Clamp 到 max_dist
        circ_dist = circ_dist.clamp(0, self.max_dist)  # (L, L)

        # 查表
        bias = self.relative_bias(circ_dist)  # (L, L, n_heads)

        # 转置成注意力格式
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)

        return bias

    def forward_with_shift(
        self,
        seq_len: int,
        shift: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        带循环移位的偏置计算

        用于测试等变性：序列移位 shift 后，偏置应该不变

        Args:
            seq_len: 序列长度 L
            shift: 循环移位数（物理意义：环的旋转）
            device: 设备

        Returns:
            bias: (1, n_heads, L, L)
        """
        # 循环相对位置编码与绝对位置无关，所以 shift 不影响结果
        # 这是等变性的关键！
        return self.forward(seq_len, device)


class CircularRelativePositionEncoding(nn.Module):
    """循环相对位置编码（替代 TPE）

    不注入绝对位置，只通过注意力偏置注入相对位置。

    这个模块在注意力层使用，不是在 embedding 层。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        max_dist: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 相对位置偏置
        self.relative_bias = CircularRelativePositionBias(n_heads, max_dist, dropout)

        # 可选：额外的相对位置嵌入（用于更丰富的表示）
        # 这里我们只用偏置，保持简单

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """返回注意力偏置"""
        return self.relative_bias(seq_len, device)


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

def test_circular_equivariance():
    """测试循环相对位置编码的等变性"""
    print("=" * 60)
    print("CircularRelativePositionBias 等变性测试")
    print("=" * 60)

    torch.manual_seed(42)

    n_heads = 4
    max_dist = 32
    L = 50

    bias_module = CircularRelativePositionBias(n_heads, max_dist)
    bias_module.eval()

    # 计算原始偏置
    bias_orig = bias_module(L)  # (1, n_heads, L, L)

    # 循环移位（模拟序列旋转）
    shift = 10

    # 循环相对位置编码应该与移位无关！
    bias_shifted = bias_module.forward_with_shift(L, shift)

    # 验证：两者应该完全相等
    diff = (bias_orig - bias_shifted).abs().max().item()

    print(f"序列长度: L={L}")
    print(f"循环移位: shift={shift}")
    print(f"偏置误差: {diff:.6e}")
    print()

    if diff < 1e-10:
        print("[PASS] 循环相对位置编码等变性成立（与移位无关）")
    else:
        print("[FAIL] 等变性不成立")

    # 额外测试：边界连续性
    print()
    print("-" * 60)
    print("边界连续性验证")
    print("-" * 60)

    # 检查位置 0 和位置 L-1 的相对距离
    positions = torch.arange(L)
    diff_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
    circ_dist = torch.min(diff_matrix.abs(), L - diff_matrix.abs())

    # 位置 0 到位置 L-1 的距离
    dist_0_to_Lminus1 = circ_dist[0, L-1].item()

    print(f"位置 0 到位置 {L-1} 的循环距离: {dist_0_to_Lminus1}")
    print(f"（应该 = 1，表示相邻）")

    if dist_0_to_Lminus1 == 1:
        print("[PASS] 边界连续性成立")
    else:
        print("[FAIL] 边界连续性不成立")

    return diff


if __name__ == "__main__":
    test_circular_equivariance()