#!/usr/bin/env python3
"""
bias_annealing.py — 在线偏差退火数据增强

在训练最后 20% epoch 对 target coords 施加可控破坏，强迫模型学不变性规律而非过拟合坐标绝对值。

三种增强:
1. 高斯噪声: σ 从 0 线性升到 max_sigma(默认 2Å)
2. 配对翻转: 5% 的 pair_probs 随机翻转(0↔1)
3. BSJ 刚性旋转: 首尾 10nt 绕各自中心随机旋转 ±15°

用法:
    from bias_annealing import apply_bias_annealing

    # 在训练循环中
    epoch_frac = epoch / total_epochs
    if epoch_frac >= 0.8:  # 最后 20% epoch
        batch = apply_bias_annealing(batch, epoch_frac, max_sigma=2.0)
"""

import torch
import numpy as np
from typing import Dict


def rotation_matrix(axis: str, angle_deg: float) -> torch.Tensor:
    """生成绕指定轴的旋转矩阵 (3x3)。

    Args:
        axis: 'x', 'y', 或 'z'
        angle_deg: 旋转角度(度)

    Returns:
        (3, 3) 旋转矩阵
    """
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0, dtype=torch.float32)
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)

    if axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    elif axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    elif axis == 'z':
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def apply_gaussian_noise(coords: torch.Tensor, sigma: float) -> torch.Tensor:
    """对 coords 施加高斯噪声。

    Args:
        coords: (L, 3) 坐标张量
        sigma: 噪声标准差(Å)

    Returns:
        加噪后的 coords
    """
    if sigma <= 0:
        return coords

    noise = torch.randn_like(coords) * sigma
    return coords + noise


def apply_pair_flip(pair_probs: torch.Tensor, flip_ratio: float = 0.05) -> torch.Tensor:
    """随机翻转 pair_probs 中的元素 (0↔1)。

    Args:
        pair_probs: (L, L) 配对概率矩阵
        flip_ratio: 翻转比例(默认 5%)

    Returns:
        翻转后的 pair_probs
    """
    if flip_ratio <= 0 or pair_probs is None:
        return pair_probs

    # 确保 float (rand_like 不支持 bool)
    pp = pair_probs.float() if not pair_probs.is_floating_point() else pair_probs

    mask = torch.rand_like(pp) < flip_ratio
    flipped = 1.0 - pp  # 0→1, 1→0
    return torch.where(mask, flipped, pp)


def apply_bsj_rotation(coords: torch.Tensor, length: int, max_angle_deg: float = 15.0,
                       bsj_nt: int = 10) -> torch.Tensor:
    """对 BSJ 区域(首尾各 bsj_nt 个核苷酸)施加刚性旋转。

    Args:
        coords: (L, 3) 坐标张量
        length: 实际序列长度(不含 padding)
        max_angle_deg: 最大旋转角度(度)
        bsj_nt: BSJ 区域核苷酸数(默认 10)

    Returns:
        旋转后的 coords
    """
    if length < 2 * bsj_nt or max_angle_deg <= 0:
        return coords

    coords_rot = coords.clone()

    # 随机选择旋转轴和角度
    axes = ['x', 'y', 'z']
    angle_5prime = np.random.uniform(-max_angle_deg, max_angle_deg)
    angle_3prime = np.random.uniform(-max_angle_deg, max_angle_deg)

    # 5' 端旋转(前 bsj_nt 个)
    coords_5prime = coords[:bsj_nt].clone()
    center_5prime = coords_5prime.mean(dim=0, keepdim=True)
    coords_5prime_centered = coords_5prime - center_5prime

    axis = np.random.choice(axes)
    R_5prime = rotation_matrix(axis, angle_5prime)
    coords_5prime_rot = coords_5prime_centered @ R_5prime.T + center_5prime

    coords_rot[:bsj_nt] = coords_5prime_rot

    # 3' 端旋转(后 bsj_nt 个)
    coords_3prime = coords[-bsj_nt:].clone()
    center_3prime = coords_3prime.mean(dim=0, keepdim=True)
    coords_3prime_centered = coords_3prime - center_3prime

    axis = np.random.choice(axes)
    R_3prime = rotation_matrix(axis, angle_3prime)
    coords_3prime_rot = coords_3prime_centered @ R_3prime.T + center_3prime

    coords_rot[-bsj_nt:] = coords_3prime_rot

    return coords_rot


def apply_bias_annealing(batch: Dict[str, torch.Tensor],
                        epoch_frac: float,
                        max_sigma: float = 2.0,
                        pair_flip_ratio: float = 0.05,
                        bsj_max_angle: float = 15.0,
                        bsj_nt: int = 10) -> Dict[str, torch.Tensor]:
    """对 batch 施加偏差退火增强。

    Args:
        batch: 训练 batch (包含 'coords', 'lengths', 可选 'pair_probs')
        epoch_frac: 当前 epoch 占比 (0.0-1.0)
        max_sigma: 最大噪声标准差(Å)
        pair_flip_ratio: 配对翻转比例
        bsj_max_angle: BSJ 最大旋转角度(度)
        bsj_nt: BSJ 区域核苷酸数

    Returns:
        增强后的 batch
    """
    # 计算当前 sigma (线性递增)
    # epoch_frac=0.8 时 sigma=0, epoch_frac=1.0 时 sigma=max_sigma
    anneal_frac = (epoch_frac - 0.8) / 0.2  # 0.0 → 1.0
    sigma = anneal_frac * max_sigma

    batch_aug = {}
    coords_batch = batch['coords'].clone()  # (B, L, 3)
    lengths = batch['lengths']

    B = coords_batch.shape[0]

    for b in range(B):
        L = lengths[b]
        coords = coords_batch[b, :L]  # (L, 3)

        # 1. 高斯噪声
        coords = apply_gaussian_noise(coords, sigma)

        # 2. BSJ 刚性旋转
        coords = apply_bsj_rotation(coords, L, bsj_max_angle, bsj_nt)

        coords_batch[b, :L] = coords

    batch_aug['coords'] = coords_batch

    # 3. 配对翻转
    if 'pair_probs' in batch and batch['pair_probs'] is not None:
        pair_probs_batch = batch['pair_probs'].clone()  # (B, L, L)
        for b in range(B):
            L = lengths[b]
            pair_probs = pair_probs_batch[b, :L, :L]
            pair_probs_batch[b, :L, :L] = apply_pair_flip(pair_probs, pair_flip_ratio)
        batch_aug['pair_probs'] = pair_probs_batch

    # 复制其他字段
    for key in batch:
        if key not in batch_aug:
            batch_aug[key] = batch[key]

    return batch_aug


# ===== 测试 =====
if __name__ == '__main__':
    # 合成测试 batch
    B, L = 2, 100
    batch = {
        'coords': torch.randn(B, L, 3),  # 随机坐标
        'lengths': [L, L],
        'pair_probs': (torch.rand(B, L, L) > 0.5).float(),  # 随机配对(float32)
    }

    print("Original coords mean:", batch['coords'].mean().item())
    print("Original pair_probs mean:", batch['pair_probs'].float().mean().item())

    # epoch_frac=0.9 (最后 20% 的中点)
    batch_aug = apply_bias_annealing(batch, epoch_frac=0.9, max_sigma=2.0)

    print("\nAugmented (epoch_frac=0.9):")
    print("  Coords mean:", batch_aug['coords'].mean().item())
    print("  Coords diff from original:", (batch_aug['coords'] - batch['coords']).abs().mean().item())
    print("  Pair_probs diff from original:", (batch_aug['pair_probs'] - batch['pair_probs']).abs().mean().item())

    print("\n✓ Bias annealing augmentation works!")
