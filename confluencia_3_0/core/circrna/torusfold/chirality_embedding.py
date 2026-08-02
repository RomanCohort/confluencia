"""
chirality_embedding.py — RNA 手性感知 Token Embedding

用 Linear 矩阵代替 nn.Embedding（输入 float32 one-hot），
支持任意 dtypes / 后端（CUDA / ROCm / CPU）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(seq: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Create [B, L, n_tokens] float32 one-hot from integer seq."""
    return F.one_hot(seq.long(), num_classes=n_tokens).float()


class ChiralityEmbedding(nn.Module):
    """手性偏好 Embedding — Linear + one-hot."""

    def __init__(self, n_tokens: int = 5, d_model: int = 128):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.chirality_weight = nn.Linear(n_tokens, d_model, bias=False)
        nn.init.normal_(self.chirality_weight.weight, mean=0.0, std=0.01)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        oh = _one_hot(seq_tokens, self.n_tokens)  # [B, L, n_tokens]
        return self.chirality_weight(oh)  # [B, L, d_model]


class ChiralityAwareEmbedding(nn.Module):
    """手性感知 Token Embedding

    两个 Linear 矩阵分别做 token embedding 和手性 embedding，
    通过可学习的 alpha 权重混合。
    """

    def __init__(self, n_tokens: int = 5, d_model: int = 128):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_weight = nn.Linear(n_tokens, d_model, bias=False)
        self.chirality = ChiralityEmbedding(n_tokens, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        oh = _one_hot(seq_tokens, self.n_tokens)  # [B, L, n_tokens]
        token_emb = self.token_weight(oh)
        chirality_emb = self.chirality(seq_tokens)
        return token_emb + self.alpha * chirality_emb