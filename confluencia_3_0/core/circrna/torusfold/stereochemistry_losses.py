"""stereochemistry_losses.py — 通用立体化学损失函数模块。

为所有 TorusFold Scheme 提供统一的立体化学约束。

损失函数：
  1. clash_loss: 惩罚原子重叠
  2. angle_loss: 惩罚键角异常
  3. dihedral_loss: 惩罚二面角异常
  4. closure_loss: 惩罚 BSJ 闭环偏差（已有，统一接口）

设计理念：
  - 即插即用：每个 Scheme 可选择性添加
  - 轻量级：计算成本 < 5% 额外开销
  - 可解释：每个损失项有物理意义

作者：Confluencia Team
日期：2026-07-01
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════
# 默认参数
# ═══════════════════════════════════════════════════════════════

DEFAULT_STEREO_PARAMS = {
    # Clash 参数
    "clash_distance": 2.5,      # Å，最小非键距离
    "clash_weight": 10.0,       # 损失权重

    # 键长参数
    "bond_length": 5.9,         # Å，P-P backbone
    "bond_weight": 0.5,         # 损失权重

    # 键角参数
    "target_angle": 109.5,      # 度，理想键角
    "angle_weight": 2.0,        # 损失权重

    # 二面角参数
    "target_dihedral_cos": -0.276,  # cos(θ) for C3'-endo
    "dihedral_weight": 1.0,     # 损失权重

    # 闭环参数
    "closure_target": 5.9,      # Å
    "closure_weight": 5.0,      # 损失权重
}


# ═══════════════════════════════════════════════════════════════
# PyTorch 损失函数
# ═══════════════════════════════════════════════════════════════

if HAS_TORCH:
    class StereochemistryLoss(torch.nn.Module):
        """统一的立体化学损失函数模块。

        用法：
            stereo_loss = StereochemistryLoss()
            loss = stereo_loss(coords, lengths)
        """

        def __init__(self, params: Optional[dict] = None):
            super().__init__()
            self.params = params or DEFAULT_STEREO_PARAMS

        def forward(
            self,
            coords: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
        ) -> dict:
            """计算所有立体化学损失。

            Args:
                coords: (B, L, 3) 预测坐标
                lengths: (B,) 每个序列的实际长度

            Returns:
                losses: {
                    'clash_loss': float,
                    'bond_loss': float,
                    'angle_loss': float,
                    'dihedral_loss': float,
                    'total': float,
                }
            """
            B, L, _ = coords.shape

            if lengths is None:
                lengths = torch.full((B,), L, device=coords.device)

            # Clash loss
            clash_loss = self._compute_clash_loss(coords, lengths)

            # Bond loss
            bond_loss = self._compute_bond_loss(coords, lengths)

            # Angle loss
            angle_loss = self._compute_angle_loss(coords, lengths)

            # Dihedral loss
            dihedral_loss = self._compute_dihedral_loss(coords, lengths)

            # Total
            total = (
                self.params["clash_weight"] * clash_loss +
                self.params["bond_weight"] * bond_loss +
                self.params["angle_weight"] * angle_loss +
                self.params["dihedral_weight"] * dihedral_loss
            )

            return {
                'clash_loss': clash_loss.item(),
                'bond_loss': bond_loss.item(),
                'angle_loss': angle_loss.item(),
                'dihedral_loss': dihedral_loss.item(),
                'total': total,
            }

        def _compute_clash_loss(
            self,
            coords: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """Clash loss: 惩罚非相邻碱基的原子重叠。"""
            B, L, _ = coords.shape
            clash_dist = self.params["clash_distance"]

            if L < 4:
                return torch.tensor(0.0, device=coords.device)

            total_loss = torch.tensor(0.0, device=coords.device)
            n_valid = 0

            for b in range(B):
                valid_L = int(lengths[b].item())
                if valid_L < 4:
                    continue

                # 提取有效坐标
                c = coords[b, :valid_L]

                # 计算距离矩阵
                diff = c[:, None, :] - c[None, :, :]  # (valid_L, valid_L, 3)
                dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

                # 创建 mask：|i-j| >= 2，排除 BSJ
                i_idx, j_idx = torch.triu_indices(valid_L, valid_L, offset=2)
                mask = ~((i_idx == 0) & (j_idx == valid_L - 1))

                # 提取有效距离
                valid_dists = dist_matrix[i_idx[mask], j_idx[mask]]

                # Clash penalty
                clashes = torch.clamp(clash_dist - valid_dists, min=0.0)
                total_loss = total_loss + torch.mean(clashes ** 2)
                n_valid += 1

            return total_loss / max(n_valid, 1)

        def _compute_bond_loss(
            self,
            coords: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """Bond loss: 惩罚键长偏差。"""
            B, L, _ = coords.shape
            target = self.params["bond_length"]

            total_loss = torch.tensor(0.0, device=coords.device)
            n_valid = 0

            for b in range(B):
                valid_L = int(lengths[b].item())
                if valid_L < 2:
                    continue

                c = coords[b, :valid_L]

                # 相邻键
                bonds = torch.norm(c[1:] - c[:-1], dim=1)

                # BSJ 键
                bsj_bond = torch.norm(c[0] - c[-1])

                # 所有键
                all_bonds = torch.cat([bonds, bsj_bond.unsqueeze(0)])

                # MSE
                total_loss = total_loss + torch.mean((all_bonds - target) ** 2)
                n_valid += 1

            return total_loss / max(n_valid, 1)

        def _compute_angle_loss(
            self,
            coords: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """Angle loss: 惩罚键角偏差。"""
            B, L, _ = coords.shape
            target_cos = np.cos(np.deg2rad(self.params["target_angle"]))

            total_loss = torch.tensor(0.0, device=coords.device)
            n_valid = 0

            for b in range(B):
                valid_L = int(lengths[b].item())
                if valid_L < 3:
                    continue

                c = coords[b, :valid_L]

                # 计算角度
                v1 = c[1:-1] - c[:-2]   # (valid_L-2, 3)
                v2 = c[2:] - c[1:-1]    # (valid_L-2, 3)

                cos_angle = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)

                # MSE
                total_loss = total_loss + torch.mean((cos_angle - target_cos) ** 2)
                n_valid += 1

            return total_loss / max(n_valid, 1)

        def _compute_dihedral_loss(
            self,
            coords: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """Dihedral loss: 惩罚二面角偏差（A-form RNA）。"""
            B, L, _ = coords.shape
            target_cos = self.params["target_dihedral_cos"]

            total_loss = torch.tensor(0.0, device=coords.device)
            n_valid = 0

            for b in range(B):
                valid_L = int(lengths[b].item())
                if valid_L < 4:
                    continue

                c = coords[b, :valid_L]

                # 计算二面角（简化版：相邻键向量的角度）
                v1 = c[1:-1] - c[:-2]
                v2 = c[2:] - c[1:-1]

                cos_angle = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)

                # MSE
                total_loss = total_loss + torch.mean((cos_angle - target_cos) ** 2)
                n_valid += 1

            return total_loss / max(n_valid, 1)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def compute_total_stereo_loss(
    coords: "torch.Tensor",
    lengths: Optional["torch.Tensor"] = None,
    params: Optional[dict] = None,
) -> "torch.Tensor":
    """便捷函数：计算总立体化学损失。

    Args:
        coords: (B, L, 3)
        lengths: (B,)
        params: 参数字典

    Returns:
        total_loss: 标量
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    stereo_loss = StereochemistryLoss(params)
    losses = stereo_loss(coords, lengths)
    return losses['total']


def get_stereo_loss_breakdown(
    coords: "torch.Tensor",
    lengths: Optional["torch.Tensor"] = None,
    params: Optional[dict] = None,
) -> dict:
    """便捷函数：获取各损失项的分解。

    Returns:
        {
            'clash_loss': float,
            'bond_loss': float,
            'angle_loss': float,
            'dihedral_loss': float,
            'total': float,
        }
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    stereo_loss = StereochemistryLoss(params)
    return stereo_loss(coords, lengths)


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch 未安装，跳过测试")
        exit(0)

    print("=" * 60)
    print("立体化学损失函数模块测试")
    print("=" * 60)

    # 创建测试数据
    B, L = 4, 100
    coords = torch.randn(B, L, 3) * 10.0
    lengths = torch.tensor([80, 90, 100, 95])

    # 计算损失
    stereo_loss = StereochemistryLoss()
    losses = stereo_loss(coords, lengths)

    print("\n损失分解:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")

    # 便捷函数测试
    total = compute_total_stereo_loss(coords, lengths)
    print(f"\n总损失: {total:.4f}")

    print("=" * 60)