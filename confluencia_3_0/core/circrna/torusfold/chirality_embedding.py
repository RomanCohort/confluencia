"""
chirality_embedding.py — RNA 手性偏好 Embedding（float32-only）

关键修复：ROCm gfx1151 上 nn.Embedding 需要 int64 索引，
而 int64 在 ROCm gfx1151 上有系统性类型转换 bug。

修复方案：用 Linear 矩阵代替 Embedding，输入为 float32 one-hot，
完全避免 int64 路径。

输入：[B, L] float32，值在 0-4 范围（对应 A/U/C/G/N）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rocm_safe_parameter import SafeParameter


def _one_hot_float32(seq: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """创建 float32 one-hot，绕过 ROCm int64→float32 编译器 bug

    ROCm gfx1151 上 F.one_hot().float() 返回垃圾数据。
    替代方案：创建 float32 零矩阵，用 float 比较掩码直接赋值。
    """
    B, L = seq.shape
    oh = torch.zeros(B, L, n_tokens, dtype=torch.float32, device=seq.device)
    for t in range(n_tokens):
        mask = (seq >= t - 0.1) & (seq < t + 0.9)
        oh[mask, t] = 1.0
    return oh


class ChiralityEmbedding(nn.Module):
    """手性偏好 Embedding（float32-only，无 int64）

    用 Linear 矩阵代替 nn.Embedding，输入 float32 one-hot。
    """

    def __init__(self, n_tokens: int = 5, d_model: int = 128):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        # 用 Linear 代替 Embedding：[n_tokens, d_model]
        self.chirality_weight = nn.Linear(n_tokens, d_model, bias=False)
        nn.init.normal_(self.chirality_weight.weight, mean=0.0, std=0.01)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        B, L = seq_tokens.shape
        oh = _one_hot_float32(seq_tokens, self.n_tokens)  # [B, L, n_tokens]
        return self.chirality_weight(oh)  # [B, L, d_model]


class ChiralityAwareEmbedding(nn.Module):
    """手性感知 Token Embedding（float32-only）

    用两个 Linear 矩阵分别做 token embedding 和手性 embedding，
    输入为 float32 one-hot，完全避免 int64。
    """

    def __init__(self, n_tokens: int = 5, d_model: int = 128):
        super().__init__()
        self.n_tokens = n_tokens

        # token embedding: [n_tokens, d_model]
        self.token_weight = nn.Linear(n_tokens, d_model, bias=False)

        # 手性 embedding
        self.chirality = ChiralityEmbedding(n_tokens, d_model)

        # 混合权重：用 SafeParameter 绕开 gfx1151 上标量 backward 的 inf
        self.alpha = SafeParameter(0.5)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        B, L = seq_tokens.shape
        oh = _one_hot_float32(seq_tokens, self.n_tokens)  # [B, L, n_tokens]

        token_emb = self.token_weight(oh)
        chirality_emb = self.chirality(seq_tokens)

        # SafeParameter.forward(other) → _SafeMul.apply 走 matmul backward
        return token_emb + self.alpha(chirality_emb)