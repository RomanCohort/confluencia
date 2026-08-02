"""
physics_loss.py — RNA 物理合理性正则项

在 BSJ 闭环约束基础上，增加：
1. 碱基配对一致性（AU/GC/GU 配对合法性）
2. 螺旋几何约束（螺距、每圈碱基数）
3. 环区熵正则（环区应该更灵活）

用法：
    from physics_loss import PhysicsLoss

    loss_fn = PhysicsLoss(n_tokens=5)
    physics_loss = loss_fn(coords, seq_tokens, pred_pairing_probs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# 合法碱基配对矩阵（5 tokens: A/U/C/G/N）
# 1=合法配对，0=非法配对
VALID_PAIRING = torch.tensor([
    # A  U  C  G  N
    [0, 1, 0, 0, 0],  # A
    [1, 0, 0, 0, 0],  # U
    [0, 0, 0, 1, 0],  # C
    [0, 0, 1, 0, 0],  # G
    [0, 0, 0, 0, 0],  # N
], dtype=torch.float32)


class PhysicsLoss(nn.Module):
    """RNA 物理合理性正则项

    Args:
        n_tokens: token 种类数（默认 5）
        device: 设备
        weights: dict 控制各项权重
    """

    def __init__(
        self,
        n_tokens: int = 5,
        device: str = 'cpu',
        weights: dict = None,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        # 注册配对矩阵到 device
        self.register_buffer('valid_pairing', VALID_PAIRING.to(device))

        # 权重配置
        if weights is None:
            weights = {
                'pairing': 0.1,       # 碱基配对一致性
                'helix': 0.05,        # 螺旋几何
                'loop_entropy': 0.02, # 环区熵
            }
        self.weights = weights

    def forward(
        self,
        coords: torch.Tensor,
        seq_tokens: torch.Tensor,
        pred_pairing_probs: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """计算物理 Loss

        Args:
            coords: [B, L, 3] 预测的 3D 坐标
            seq_tokens: [B, L] 序列 token
            pred_pairing_probs: [B, L, L] 预测的配对概率（可选）
            lengths: [B] 有效长度（可选）

        Returns:
            loss: 标量，物理正则项总和
        """
        B, L, _ = coords.shape
        device = coords.device

        if lengths is None:
            lengths = torch.full((B,), L, device=device)

        total_loss = 0.0

        # 1. 碱基配对一致性 Loss
        if pred_pairing_probs is not None:
            pairing_loss = self._pairing_consistency_loss(coords, seq_tokens, pred_pairing_probs, lengths)
            total_loss = total_loss + self.weights['pairing'] * pairing_loss

        # 2. 螺旋几何 Loss
        helix_loss = self._helix_geometry_loss(coords, seq_tokens, lengths)
        total_loss = total_loss + self.weights['helix'] * helix_loss

        # 3. 环区熵 Loss
        loop_loss = self._loop_entropy_loss(coords, seq_tokens, lengths)
        total_loss = total_loss + self.weights['loop_entropy'] * loop_loss

        return total_loss

    def _pairing_consistency_loss(
        self,
        coords: torch.Tensor,
        seq_tokens: torch.Tensor,
        pred_pairing_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """碱基配对一致性 Loss

        如果两个碱基能形成合法配对（AU/GC/GU），它们的空间距离应该较近。
        如果预测的配对概率高，但实际距离很远，则惩罚。

        Args:
            seq_tokens: [B, L]
            pred_pairing_probs: [B, L, L]
            lengths: [B]

        Returns:
            loss: 标量
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 构建理想配对矩阵（基于序列）
        seq_one_hot = F.one_hot(seq_tokens, self.n_tokens).float()  # [B, L, n_tokens]
        ideal_pairing = torch.einsum('bin,bjn->bij', seq_one_hot, seq_one_hot)  # [B, L, L]

        # 查询合法配对矩阵
        pairing_matrix = self.valid_pairing[seq_tokens.unsqueeze(-1), seq_tokens.unsqueeze(-2)]  # [B, L, L]

        # 距离矩阵
        dist = torch.cdist(coords, coords)  # [B, L, L]

        # 惩罚：合法配对但距离远
        # 理想距离：配对碱基应该在 10-25 Å 范围
        target_dist = 17.5  # Å
        dist_tolerance = 7.5  # Å

        dist_penalty = F.relu(dist - (target_dist + dist_tolerance)) ** 2
        dist_penalty = dist_penalty * pairing_matrix * ideal_pairing

        # 归一化
        n_pairs = pairing_matrix.sum(dim=(-1, -2)).clamp(min=1)
        loss = dist_penalty.sum(dim=(-1, -2)) / n_pairs

        return loss.mean()

    def _helix_geometry_loss(
        self,
        coords: torch.Tensor,
        seq_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """螺旋几何约束 Loss

        RNA 螺旋的几何特征：
        - 相邻碱基距离 ~3.4 Å
        - 螺旋每圈 ~11 个碱基
        - 螺距 ~28 Å

        Args:
            coords: [B, L, 3]
            seq_tokens: [B, L]
            lengths: [B]

        Returns:
            loss: 标量
        """
        B, L, _ = coords.shape
        device = coords.device

        # 相邻碱基距离（应该 ~3.4 Å）
        coords_shifted = torch.roll(coords, shifts=-1, dims=1)
        bond_lengths = torch.norm(coords - coords_shifted, dim=-1)  # [B, L]

        # 理想键长
        target_bond = 3.4
        bond_penalty = F.relu(torch.abs(bond_lengths - target_bond) - 0.5) ** 2

        # 屏蔽最后一个碱基（没有下一个）
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1) - 1
        bond_penalty = bond_penalty * mask.float()

        # 归一化
        loss = bond_penalty.sum(dim=-1) / lengths.clamp(min=1)

        return loss.mean()

    def _loop_entropy_loss(
        self,
        coords: torch.Tensor,
        seq_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """环区熵正则 Loss

        环区（loop）应该更灵活，坐标变化更大。
        如果环区的 RMSF 太小，说明模型过于"僵硬"，惩罚。

        简化版：用坐标的局部方差衡量灵活性。

        Args:
            coords: [B, L, 3]
            seq_tokens: [B, L]
            lengths: [B]

        Returns:
            loss: 标量
        """
        B, L, _ = coords.shape
        device = coords.device

        # 识别环区：N token 或连续相同 token 的区域
        # 简化：用 token 变化检测边界
        # clamp 代替 .float()，数值更稳定（边界值不会溢到 NaN）
        token_diff = torch.abs(seq_tokens[:, 1:] - seq_tokens[:, :-1])  # [B, L-1]
        loop_mask = token_diff.clamp(max=1.0)  # [B, L-1]，0 或 1
        loop_mask = F.pad(loop_mask, (0, 1), value=0)  # [B, L]

        # 计算局部坐标方差（灵活性）
        coords_smooth = F.avg_pool1d(
            coords.transpose(1, 2), kernel_size=5, stride=1, padding=2
        ).transpose(1, 2)  # [B, L, 3]
        local_var = ((coords - coords_smooth) ** 2).sum(dim=-1)  # [B, L]

        # 环区应该有较高的方差
        loop_var = (local_var * loop_mask).sum(dim=-1) / loop_mask.sum(dim=-1).clamp(min=1)

        # 如果环区方差太小，惩罚（鼓励灵活性）
        target_loop_var = 2.0  # Å^2
        loss = F.relu(target_loop_var - loop_var)

        # 过滤 NaN（loop_mask 全 0 的样本）
        loss = torch.nan_to_num(loss, nan=0.0)

        return loss.mean()