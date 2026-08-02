"""
adaptive_sparse_k.py — 自适应稀疏邻居数

K 值由局部序列复杂度（滑动窗口频率熵）+ pair_probs 共同决定。
复杂度计算走 GPU 原生 Tensor ops，无 CPU 搬运。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_local_complexity(
    seq_tokens: torch.Tensor, window: int = 8, n_tokens: int = 5
) -> torch.Tensor:
    """计算局部序列复杂度（基于滑动窗口频率熵）

    完全在输入 tensor 所在设备上计算（GPU/CPU 均可）。
    复杂度 = normalized entropy of nucleotide frequencies in a window.

    Args:
        seq_tokens: [B, L] 整数 token（0..n_tokens-1）
        window: 滑动窗口大小
        n_tokens: 碱基数（默认 5: A/U/C/G/N）

    Returns:
        [B, L] float32，归一化熵 ∈ [0, 1]，同 device 作为输入。
    """
    B, L = seq_tokens.shape
    device = seq_tokens.device
    if n_tokens <= 1:
        return torch.zeros(B, L, dtype=torch.float32, device=device)

    seq = torch.clamp(seq_tokens, 0, n_tokens - 1).long()  # [B, L]

    # 每行 padding（edge mode）
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    seq_pad = F.pad(seq, (pad_left, pad_right), mode='constant', value=0)  # [B, L+window-1]

    # 滑动窗口展成 [B, window, L] 然后 transpose → [B, L, window]
    s = seq_pad.unfold(dimension=1, size=window, step=1)  # [B, L, window]

    # 每个窗口的碱基计数
    oh = F.one_hot(s, num_classes=n_tokens).float()  # [B, L, window, n_tokens]
    counts = oh.sum(dim=2)  # [B, L, n_tokens]

    # 频率 + 熵
    total = counts.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, L, 1]
    freq = counts / total  # [B, L, n_tokens]
    entropy = -(freq * torch.log(freq + 1e-8)).sum(dim=-1)  # [B, L]
    return (entropy / math.log(n_tokens)).float()


class AdaptiveSparseK(nn.Module):
    """自适应稀疏 K 模块"""

    def __init__(self, K_min: int = 20, K_max: int = 80, K_base: int = 40):
        super().__init__()
        self.K_min = K_min
        self.K_max = K_max
        self.K_base = K_base

    def forward(self, seq_tokens: torch.Tensor, pair_probs=None) -> torch.Tensor:
        complexity = compute_local_complexity(seq_tokens)
        k = self.K_min + complexity * (self.K_max - self.K_min)

        if pair_probs is not None:
            pair_strength = pair_probs.sum(dim=-1)
            pair_strength = (pair_strength - pair_strength.mean()) / (pair_strength.std() + 1e-8)
            pair_strength = torch.sigmoid(pair_strength)
            k = k * (0.7 + 0.3 * pair_strength)

        k_int = k.clamp(self.K_min, self.K_max)
        return k_int.to(torch.float32)

    def get_global_K(self, k_per_residue: torch.Tensor, L: int) -> int:
        # clamp + nan_to_num 防御性保护（避免极端序列产生 inf）
        k_safe = k_per_residue.clamp(self.K_min, self.K_max)
        K_avg = int(torch.nan_to_num(k_safe).mean().item())
        K_global = min(K_avg, L)
        return max(K_global, self.K_min)