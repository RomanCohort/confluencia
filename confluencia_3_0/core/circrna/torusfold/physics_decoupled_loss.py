#!/usr/bin/env python3
"""
physics_decoupled_loss.py — Loss解耦v1 (ADR训练方案阶段0.2)

Loss = L_geo（配对距离MSE，权重1.0）+ L_phys（物理统计KL散度，权重0.1）

L_geo: 对配对约束的坐标距离回归
  - 从pair_probs提取高置信度配对（threshold=0.5）
  - 计算预测坐标间距离与目标距离的MSE

L_phys: 物理合理性统计损失
  - Rg分布（回转半径，衡量整体紧凑度）
  - clash密度（原子重叠率）
  - 键角分布（O-P-O角度退化率）
  - 与实验RNA统计先验的KL散度

实验RNA统计先验（短链RNA <500nt, PDB来源）:
  - Rg分布: log-normal, μ≈2.5Å×L^0.33, σ≈0.3
  - clash密度: <0.5% (原子间距离<2Å的比例)
  - 键角退化率: <5% (角度偏离理想值>30°的比例)

参考文献:
  - ADR训练方案v2, 阶段0.2
  - RNA 3D结构统计: PDB validation reports
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 实验RNA统计先验（硬编码，后续可替换为真实PDB统计）
# ═══════════════════════════════════════════════════════════════

class RNAStatsPrior:
    """实验RNA结构的统计先验（从PDB短链RNA提取）。"""

    # Rg分布参数（log-normal）
    # 典型RNA: Rg ≈ 2.5Å × L^0.33 (近似幂律)
    RG_SCALE = 2.5  # Å
    RG_POWER = 0.33
    RG_SIGMA = 0.3  # log-space标准差

    # clash密度阈值
    CLASH_DENSITY_MAX = 0.005  # <0.5%原子对距离<2Å

    # 键角退化率阈值
    BOND_ANGLE_DEGEN_MAX = 0.05  # <5%角度偏离>30°

    @classmethod
    def expected_rg(cls, length: int) -> float:
        """给定序列长度，返回期望的回转半径（Å）。"""
        return cls.RG_SCALE * (length ** cls.RG_POWER)

    @classmethod
    def rg_logprob(cls, rg: float, length: int) -> float:
        """给定Rg值，返回log-normal概率密度。"""
        mu = np.log(cls.expected_rg(length))
        sigma = cls.RG_SIGMA
        return -0.5 * ((np.log(rg) - mu) / sigma) ** 2


# ═══════════════════════════════════════════════════════════════
# 物理统计计算
# ═══════════════════════════════════════════════════════════════

def compute_rg(coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    计算回转半径（Radius of Gyration, Rg）。

    Rg = sqrt(1/N × Σ_i |r_i - r_center|²)

    Args:
        coords: (B, L, 3) 坐标张量（Å）
        lengths: (B,) 实际序列长度

    Returns:
        rg: (B,) 每个样本的Rg值（Å）
    """
    B, L, _ = coords.shape
    rg_list = []

    for b in range(B):
        valid_L = lengths[b]
        if valid_L < 4:
            rg_list.append(torch.tensor(0.0, device=coords.device))
            continue

        # 取有效坐标
        c = coords[b, :valid_L]  # (valid_L, 3)

        # 计算质心
        center = c.mean(dim=0)  # (3,)

        # 计算Rg
        dist_sq = torch.sum((c - center) ** 2, dim=1)  # (valid_L,)
        rg = torch.sqrt(dist_sq.mean())
        rg_list.append(rg)

    return torch.stack(rg_list)


def compute_clash_density(coords: torch.Tensor, lengths: torch.Tensor,
                          clash_thresh: float = 2.0) -> torch.Tensor:
    """
    计算clash密度（原子间距离<阈值的比例）。

    Args:
        coords: (B, L, 3) 坐标张量（Å）
        lengths: (B,) 实际序列长度
        clash_thresh: clash距离阈值（Å），默认2.0Å

    Returns:
        clash_density: (B,) 每个样本的clash密度
    """
    B, L, _ = coords.shape
    density_list = []

    for b in range(B):
        valid_L = lengths[b]
        if valid_L < 4:
            density_list.append(torch.tensor(0.0, device=coords.device))
            continue

        c = coords[b, :valid_L]  # (valid_L, 3)

        # 计算所有原子对距离（O(N²)，但对于短链OK）
        # 避免相邻原子的天然近距离
        n = valid_L
        dist_mat = torch.cdist(c, c, compute_mode='donot_use_mm_for_euclid_dist')  # (n, n)

        # 掩码：排除自身和相邻（距离<4）
        mask = torch.ones(n, n, device=coords.device, dtype=torch.bool)
        for i in range(n):
            for j in range(max(0, i-3), min(n, i+4)):
                mask[i, j] = False

        # 统计clash
        dist_masked = dist_mat[mask]
        if len(dist_masked) > 0:
            clash_count = (dist_masked < clash_thresh).sum().float()
            total_pairs = len(dist_masked)
            density = clash_count / total_pairs
        else:
            density = torch.tensor(0.0, device=coords.device)

        density_list.append(density)

    return torch.stack(density_list)


def compute_bond_angle_degen_rate(coords: torch.Tensor, lengths: torch.Tensor,
                                   ideal_angle: float = 109.5,
                                   angle_thresh: float = 30.0) -> torch.Tensor:
    """
    计算键角退化率（偏离理想值>阈值的比例）。

    RNA磷酸骨架: O-P-O理想角度≈109.5°（四面体）

    Args:
        coords: (B, L, 3) 坐标张量（Å）
        lengths: (B,) 实际序列长度
        ideal_angle: 理想键角（度）
        angle_thresh: 偏离阈值（度）

    Returns:
        degen_rate: (B,) 每个样本的键角退化率
    """
    B, L, _ = coords.shape
    rate_list = []

    for b in range(B):
        valid_L = lengths[b]
        if valid_L < 4:
            rate_list.append(torch.tensor(0.0, device=coords.device))
            continue

        c = coords[b, :valid_L]  # (valid_L, 3)
        n = valid_L

        # 计算相邻三元组的键角 (i, i+1, i+2)
        angles = []
        for i in range(n - 2):
            v1 = c[i+1] - c[i]    # (i, i+1) 向量
            v2 = c[i+2] - c[i+1]  # (i+1, i+2) 向量

            # 计算夹角
            v1_norm = v1 / (torch.norm(v1) + 1e-8)
            v2_norm = v2 / (torch.norm(v2) + 1e-8)

            cos_angle = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle_deg = torch.acos(cos_angle) * 180.0 / math.pi
            angles.append(angle_deg)

        if len(angles) > 0:
            angles_tensor = torch.stack(angles)
            # 统计偏离>阈值的比例
            deviations = torch.abs(angles_tensor - ideal_angle)
            degen_count = (deviations > angle_thresh).sum().float()
            rate = degen_count / len(angles)
        else:
            rate = torch.tensor(0.0, device=coords.device)

        rate_list.append(rate)

    return torch.stack(rate_list)


# ═══════════════════════════════════════════════════════════════
# 主Loss函数
# ═══════════════════════════════════════════════════════════════

class PhysicsDecoupledLoss(nn.Module):
    """
    Loss解耦v1: L_geo + L_phys

    L_geo: 配对距离MSE（权重1.0）
    L_phys: 物理统计KL散度（权重0.1）

    Args:
        w_geo: L_geo权重（默认1.0）
        w_phys: L_phys权重（默认0.1）
        pair_thresh: 配对概率阈值，高于此值才算配对（默认0.5）
        clash_thresh: clash距离阈值（Å，默认2.0）
        angle_thresh: 键角偏离阈值（度，默认30.0）
        use_rg_loss: 是否启用Rg损失（默认True）
        use_clash_loss: 是否启用clash损失（默认True）
        use_angle_loss: 是否启用键角损失（默认True）
    """

    def __init__(
        self,
        w_geo: float = 1.0,
        w_phys: float = 0.1,
        pair_thresh: float = 0.5,
        clash_thresh: float = 2.0,
        angle_thresh: float = 30.0,
        use_rg_loss: bool = True,
        use_clash_loss: bool = True,
        use_angle_loss: bool = True,
    ):
        super().__init__()
        self.w_geo = w_geo
        self.w_phys = w_phys
        self.pair_thresh = pair_thresh
        self.clash_thresh = clash_thresh
        self.angle_thresh = angle_thresh
        self.use_rg_loss = use_rg_loss
        self.use_clash_loss = use_clash_loss
        self.use_angle_loss = use_angle_loss

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        lengths: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算解耦损失。

        Args:
            pred_coords: (B, L, 3) 预测坐标（Å）
            target_coords: (B, L, 3) 目标坐标（Å）
            lengths: (B,) 实际序列长度
            pair_probs: (B, L, L) 可选的配对概率矩阵

        Returns:
            dict: {
                'loss': 总损失,
                'L_geo': 配对距离MSE,
                'L_phys': 物理统计KL散度,
                'rg': 平均Rg,
                'clash_density': 平均clash密度,
                'angle_degen_rate': 平均键角退化率,
            }
        """
        B, L, _ = pred_coords.shape
        device = pred_coords.device

        # ═════════════════════════════════════════════════════════
        # L_geo: 配对距离MSE
        # ═════════════════════════════════════════════════════════

        L_geo = torch.tensor(0.0, device=device)

        if pair_probs is not None:
            pair_mse_list = []

            for b in range(B):
                valid_L = lengths[b]
                if valid_L < 4:
                    continue

                # 提取高置信度配对
                pp = pair_probs[b, :valid_L, :valid_L]
                pairs = (pp > self.pair_thresh) & (pp > pp.T)  # 上三角，避免重复
                pair_indices = torch.nonzero(pairs, as_tuple=False)  # (n_pairs, 2)

                if len(pair_indices) == 0:
                    continue

                # 计算配对距离
                for i, j in pair_indices:
                    pred_dist = torch.norm(pred_coords[b, i] - pred_coords[b, j])
                    target_dist = torch.norm(target_coords[b, i] - target_coords[b, j])
                    pair_mse_list.append((pred_dist - target_dist) ** 2)

            if len(pair_mse_list) > 0:
                L_geo = torch.stack(pair_mse_list).mean()

        # ═════════════════════════════════════════════════════════
        # L_phys: 物理统计KL散度
        # ═════════════════════════════════════════════════════════

        L_phys = torch.tensor(0.0, device=device)

        # Rg损失
        if self.use_rg_loss:
            pred_rg = compute_rg(pred_coords, lengths)  # (B,)
            # KL散度（近似为偏离期望的平方）
            rg_loss_list = []
            for b in range(B):
                valid_L = lengths[b]
                if valid_L < 4:
                    continue
                expected_rg = RNAStatsPrior.expected_rg(valid_L)
                # 相对误差平方
                rg_loss_list.append(((pred_rg[b] - expected_rg) / expected_rg) ** 2)

            if len(rg_loss_list) > 0:
                L_phys = L_phys + torch.stack(rg_loss_list).mean()

        # Clash密度损失
        if self.use_clash_loss:
            clash_density = compute_clash_density(pred_coords, lengths, self.clash_thresh)
            # 超过阈值的惩罚
            clash_loss = F.relu(clash_density - RNAStatsPrior.CLASH_DENSITY_MAX).mean()
            L_phys = L_phys + clash_loss

        # 键角退化率损失
        if self.use_angle_loss:
            angle_degen = compute_bond_angle_degen_rate(pred_coords, lengths,
                                                        angle_thresh=self.angle_thresh)
            # 超过阈值的惩罚
            angle_loss = F.relu(angle_degen - RNAStatsPrior.BOND_ANGLE_DEGEN_MAX).mean()
            L_phys = L_phys + angle_loss

        # ═════════════════════════════════════════════════════════
        # 总损失
        # ═════════════════════════════════════════════════════════

        total_loss = self.w_geo * L_geo + self.w_phys * L_phys

        # 统计值（用于监控）
        with torch.no_grad():
            rg_mean = compute_rg(pred_coords, lengths).mean().item()
            clash_mean = compute_clash_density(pred_coords, lengths, self.clash_thresh).mean().item()
            angle_mean = compute_bond_angle_degen_rate(pred_coords, lengths, self.angle_thresh).mean().item()

        return {
            'loss': total_loss,
            'L_geo': L_geo,
            'L_phys': L_phys,
            'rg': rg_mean,
            'clash_density': clash_mean,
            'angle_degen_rate': angle_mean,
        }


# ═══════════════════════════════════════════════════════════════
# 便捷函数（兼容旧接口）
# ═══════════════════════════════════════════════════════════════

def compute_physics_decoupled_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    lengths: torch.Tensor,
    pair_probs: Optional[torch.Tensor] = None,
    w_geo: float = 1.0,
    w_phys: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：直接计算Loss解耦v1。

    Args:
        pred_coords: (B, L, 3) 预测坐标
        target_coords: (B, L, 3) 目标坐标
        lengths: (B,) 实际序列长度
        pair_probs: (B, L, L) 可选的配对概率
        w_geo: L_geo权重
        w_phys: L_phys权重

    Returns:
        dict: 损失字典
    """
    loss_fn = PhysicsDecoupledLoss(w_geo=w_geo, w_phys=w_phys)
    return loss_fn(pred_coords, target_coords, lengths, pair_probs)