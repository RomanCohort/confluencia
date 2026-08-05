"""bsj_fape.py — BSJ FAPE 式损失 + 置信度指标.

借鉴 AlphaFold3 FAPE (Frame Aligned Point Error) 的核心思想:
  - 在局部坐标系下比较 pred vs target 的原子位置
  - 旋转不变 (局部坐标系随结构旋转)
  - 对 BSJ (back-splicing junction) 区域专门监督

设计:
  1. 以 BSJ 位点 (P(L-1) 和 P(0)) 建立局部坐标系
  2. 在该坐标系下比较 pred vs target 的残基位置
  3. 惩罚 BSJ 附近的局部几何错误
  4. 可作为置信度指标 (BSJ FAPE 低 = 闭合质量高)

用途:
  - 训练损失: loss += λ_bsj * bsj_fape_loss(pred, target)
  - 置信度: confidence = exp(-bsj_fape(pred, target))
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_local_frame(
    coords: torch.Tensor,
    origin_idx: int,
    axis_end_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建局部坐标系.

    Args:
        coords: (B, L, 3) 原子坐标
        origin_idx: 原点原子索引 (BSJ 的 P(L-1))
        axis_end_idx: z 轴终点原子索引 (BSJ 的 P(0))

    Returns:
        origin: (B, 3) 原点坐标
        z_axis: (B, 3) z 轴方向 (单位向量)
        x_axis: (B, 3) x 轴方向 (单位向量)
    """
    B = coords.shape[0]
    origin = coords[:, origin_idx]  # (B, 3)
    axis_end = coords[:, axis_end_idx]  # (B, 3)

    # z 轴: origin → axis_end
    z_axis = axis_end - origin
    z_axis = F.normalize(z_axis, dim=-1)  # (B, 3)

    # x 轴: 用 Gram-Schmidt, 选一个不平行的参考向量
    # 用 (1,0,0) 作为参考, 如果太平行就用 (0,1,0)
    ref = torch.tensor([1.0, 0.0, 0.0], device=coords.device).expand(B, 3)
    dot = (ref * z_axis).sum(dim=-1, keepdim=True)  # (B, 1)
    # 如果 dot > 0.9, 用 (0,1,0)
    ref2 = torch.tensor([0.0, 1.0, 0.0], device=coords.device).expand(B, 3)
    ref = torch.where(dot.abs() > 0.9, ref2, ref)

    # Gram-Schmidt
    x_axis = ref - (ref * z_axis).sum(dim=-1, keepdim=True) * z_axis
    x_axis = F.normalize(x_axis, dim=-1)  # (B, 3)

    # y 轴 = z × x (不需要返回, 但确保正交)
    # y_axis = torch.cross(z_axis, x_axis, dim=-1)

    return origin, z_axis, x_axis


def to_local_frame(
    coords: torch.Tensor,
    origin: torch.Tensor,
    z_axis: torch.Tensor,
    x_axis: torch.Tensor,
) -> torch.Tensor:
    """将全局坐标变换到局部坐标系.

    Args:
        coords: (B, N, 3) 要变换的原子坐标
        origin: (B, 3) 局部坐标系原点
        z_axis: (B, 3) z 轴方向
        x_axis: (B, 3) x 轴方向

    Returns:
        local_coords: (B, N, 3) 局部坐标
    """
    B, N, _ = coords.shape

    # 平移到原点
    centered = coords - origin.unsqueeze(1)  # (B, N, 3)

    # 构建旋转矩阵 (B, 3, 3)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)  # (B, 3)
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (B, 3, 3)

    # 旋转: local = R^T @ centered
    # centered: (B, N, 3) → (B, 3, N)
    # R: (B, 3, 3)
    # local = R^T @ centered^T → (B, 3, N) → (B, N, 3)
    local_coords = torch.bmm(centered, R)  # (B, N, 3)

    return local_coords


class BSJFAPELoss(nn.Module):
    """BSJ FAPE 式损失.

    在 BSJ 局部坐标系下比较 pred vs target 的残基位置.
    旋转不变, 对 BSJ 闭合质量专门监督.

    Args:
        bsj_margin: BSJ 附近区域的残基数 (默认 L 的 5%, 最小 5, 最大 20)
        max_loss: 损失上限 (防止 outlier 爆炸)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        bsj_margin: int = 10,
        max_loss: float = 50.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.bsj_margin = bsj_margin
        self.max_loss = max_loss
        self.reduction = reduction

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        bsj_margin: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_coords: (B, L, 3) 预测坐标
            target_coords: (B, L, 3) 目标坐标 (detached, 不传梯度)
            bsj_margin: 覆盖默认 bsj_margin

        Returns:
            loss: scalar (或 (B,) 如果 reduction='none')
        """
        B, L, _ = pred_coords.shape
        margin = bsj_margin or self.bsj_margin
        target_coords = target_coords.detach()  # 不传梯度到 target

        # BSJ 位点: P(L-1) → origin, P(0) → z 轴终点
        origin_idx = L - 1
        axis_end_idx = 0

        # 构建局部坐标系 (用 pred 坐标)
        origin, z_axis, x_axis = build_local_frame(pred_coords, origin_idx, axis_end_idx)

        # BSJ 附近区域: 闭合点两侧各 margin 个残基
        # indices: [L-margin, ..., L-1, 0, 1, ..., margin]
        bsj_indices = []
        for i in range(L - margin, L):
            bsj_indices.append(i)
        for i in range(min(margin, L)):
            bsj_indices.append(i)
        bsj_indices = torch.tensor(bsj_indices, device=pred_coords.device, dtype=torch.long)
        bsj_indices = bsj_indices.clamp(0, L - 1)

        # 在局部坐标系下变换 pred 和 target
        pred_local = to_local_frame(
            pred_coords[:, bsj_indices], origin, z_axis, x_axis
        )  # (B, 2*margin, 3)
        target_local = to_local_frame(
            target_coords[:, bsj_indices], origin, z_axis, x_axis
        )  # (B, 2*margin, 3)

        # L2 距离损失 (FAPE 核心)
        per_residue_loss = torch.norm(pred_local - target_local, dim=-1)  # (B, 2*margin)

        # Clamp (防止 outlier)
        per_residue_loss = per_residue_loss.clamp(max=self.max_loss)

        if self.reduction == 'mean':
            return per_residue_loss.mean()
        elif self.reduction == 'sum':
            return per_residue_loss.sum()
        else:
            return per_residue_loss.mean(dim=-1)  # (B,)


class BSJConfidence(nn.Module):
    """BSJ 置信度指标.

    从 BSJ FAPE 衍生的置信度分数 (0-1):
      confidence = exp(-bsj_fape / temperature)

    BSJ FAPE 低 → confidence 高 → 闭合质量高 → 预测可信.

    用法:
      confidence = BSJConfidence()(pred_coords, target_coords)
      # 或推理时 (无 target):
      confidence = BSJConfidence()(pred_coords, pred_coords)  # 自一致性
    """

    def __init__(self, temperature: float = 5.0, bsj_margin: int = 10):
        super().__init__()
        self.temperature = temperature
        self.fape = BSJFAPELoss(bsj_margin=bsj_margin, reduction='none')

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_coords: (B, L, 3)
            target_coords: (B, L, 3), None 时用 pred 自一致性

        Returns:
            confidence: (B,) in [0, 1]
        """
        if target_coords is None:
            target_coords = pred_coords

        fape_score = self.fape(pred_coords, target_coords)  # (B,)
        confidence = torch.exp(-fape_score / self.temperature)
        return confidence
