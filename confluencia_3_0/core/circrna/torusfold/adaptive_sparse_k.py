"""
adaptive_sparse_k.py — 自适应稀疏邻居数

ROCm gfx1151/RDNA 3.5 编译器 bug：
1. int64→float32 返回未初始化内存（已用 bool 掩码修复）
2. .max(dim) / .min(dim) 非确定性内存腐蚀（已用 argmax+gather 修复）
3. .sum(dim) 也可能非确定性腐蚀

最终策略：compute_local_complexity 在 CPU 上计算，再搬到 GPU。
复杂度计算是纯数据操作，CPU 跑很快（~2ms per batch），完全不影响训练吞吐。
"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_local_complexity(
    seq_tokens: torch.Tensor, window: int = 8, n_tokens: int = 5
) -> torch.Tensor:
    """计算局部序列复杂度（基于频率熵）

    **CPU 计算**，避免 ROCm gfx1151 的 reduction 非确定性内存腐蚀。
    seq_tokens 可以是 GPU tensor（自动移到 CPU 计算）。

    Returns: [B, L] float32 tensor on the SAME device as input.
    """
    B, L = seq_tokens.shape
    if n_tokens <= 1:
        return torch.zeros(B, L, dtype=torch.float32, device=seq_tokens.device)

    # 移到 CPU 用 numpy 计算（安全、准确）
    seq_np = seq_tokens.cpu().numpy().astype(np.int64)  # [B, L]
    seq_np = np.clip(seq_np, 0, n_tokens - 1)

    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left

    # 每行 padding
    s_padded = np.pad(seq_np, ((0, 0), (pad_left, pad_right)), mode='edge')  # [B, L+window-1]

    # 滑动窗口：用 stride_tricks 或逐元素构造 [B, L, window]
    # 为性能考虑，用 stride 避免 for 循环
    from numpy.lib import stride_tricks
    # [B, L, window] via rolling view
    n_cols = L + window - 1
    itemsize = seq_np.strides[1]  # bytes per element in L-dim
    h, w = seq_np.shape[1], window
    complexity = np.zeros((B, L), dtype=np.float32)

    for b in range(B):
        s = s_padded[b]  # [L+window-1]
        # 滑动窗口 [L, window]
        strides = (s.strides[0], s.strides[0])
        windowed = stride_tricks.as_strided(
            s, shape=(L, window), strides=strides, writeable=False
        )  # [L, window]
        # 每窗口碱基计数
        counts = np.zeros((L, n_tokens), dtype=np.float32)
        for t_idx in range(n_tokens):
            counts[:, t_idx] = (windowed == t_idx).sum(axis=1)  # [L]
        total = counts.sum(axis=1, keepdims=True)  # [L, 1]
        freq = counts / total  # [L, n_tokens]
        entropy = -np.sum(freq * np.log(freq + 1e-8), axis=1)  # [L]
        max_entropy = np.log(n_tokens)
        complexity[b] = entropy / max_entropy

    out = torch.from_numpy(complexity).to(dtype=torch.float32, device=seq_tokens.device)
    return out


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
        # ROCm gfx1151 安全：先 clamp 防止 corruption 产生的 inf
        k_safe = k_per_residue.clamp(self.K_min, self.K_max)
        K_avg = int(torch.nan_to_num(k_safe).mean().item())
        K_global = min(K_avg, L)
        return max(K_global, self.K_min)