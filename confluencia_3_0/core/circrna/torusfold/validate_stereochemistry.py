"""validate_stereochemistry.py — 立体化学验证模块。

检查 RNA 3D 结构的立体化学质量，防止类似 AlphaFold3 的失效。

检查项：
  1. Clash（原子重叠）
  2. 键长异常
  3. 键角异常
  4. 手性错误（可选）

参考：
  - AlphaFold3 stereochemistry issues (Stein et al., 2024)
  - Amber RNA OL3 parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════
# RNA 立体化学参数（来自 Amber RNA OL3）
# ═══════════════════════════════════════════════════════════════

RNA_STEREO_PARAMS = {
    # 键长参数（粗粒化：P-P 距离）
    "bond_length": 5.9,       # Å, P-P backbone
    "bond_tolerance": 0.5,    # Å, 允许偏差

    # Clash 参数
    "clash_distance": 2.5,    # Å, 最小非键距离
    "clash_tolerance": 0.1,   # Å, 允许偏差

    # 键角参数
    "bond_angle": 109.5,      # 度，理想键角
    "angle_tolerance": 10.0,  # 度，允许偏差

    # 二面角参数（A-form RNA）
    "dihedral_target": -0.276,  # cos(θ) for C3'-endo
    "dihedral_tolerance": 0.2,

    # BSJ 闭环参数
    "closure_target": 5.9,    # Å
    "closure_tolerance": 1.0, # Å
}


@dataclass
class StereochemistryReport:
    """立体化学验证报告。"""

    # === 总体状态 ===
    is_valid: bool                    # 是否通过所有检查
    score: float                      # 综合得分 [0, 1]

    # === Clash 检测 ===
    has_clashes: bool                 # 是否有原子重叠
    n_clashes: int                    # 重叠数量
    clash_pairs: List[Tuple[int, int]]  # 重叠的碱基对
    clash_distances: List[float]      # 重叠距离

    # === 键长检测 ===
    bond_mean_error: float            # 平均键长误差（Å）
    bond_max_error: float             # 最大键长误差（Å）
    bond_outliers: List[int]          # 键长异常的碱基索引

    # === 键角检测 ===
    angle_mean_error: float           # 平均键角误差（度）
    angle_max_error: float            # 最大键角误差（度）
    angle_outliers: List[int]         # 键角异常的碱基索引

    # === 闭环检测 ===
    closure_distance: float           # BSJ 闭环距离（Å）
    closure_error: float              # 闭环误差（Å）

    # === 警告 ===
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成摘要报告。"""
        lines = []
        lines.append("=" * 60)
        lines.append("立体化学验证报告")
        lines.append("=" * 60)

        # 总体状态
        status = "✅ 通过" if self.is_valid else "❌ 失败"
        lines.append(f"\n总体状态: {status} (得分: {self.score:.2f})")

        # Clash
        lines.append("\n原子重叠（Clash）:")
        if self.has_clashes:
            lines.append(f"  ⚠️ 发现 {self.n_clashes} 处重叠")
            for (i, j), d in zip(self.clash_pairs[:5], self.clash_distances[:5]):
                lines.append(f"    - 碱基 {i}-{j}: {d:.2f}Å")
        else:
            lines.append("  ✅ 无重叠")

        # 键长
        lines.append("\n键长检查:")
        lines.append(f"  平均误差: {self.bond_mean_error:.2f}Å")
        lines.append(f"  最大误差: {self.bond_max_error:.2f}Å")
        if self.bond_outliers:
            lines.append(f"  ⚠️ 异常键: {len(self.bond_outliers)} 个")

        # 键角
        lines.append("\n键角检查:")
        lines.append(f"  平均误差: {self.angle_mean_error:.1f}°")
        lines.append(f"  最大误差: {self.angle_max_error:.1f}°")
        if self.angle_outliers:
            lines.append(f"  ⚠️ 异常角: {len(self.angle_outliers)} 个")

        # 闭环
        lines.append("\nBSJ 闭环:")
        lines.append(f"  距离: {self.closure_distance:.2f}Å (目标: 5.9Å)")
        lines.append(f"  误差: {self.closure_error:.2f}Å")

        # 警告
        if self.warnings:
            lines.append("\n⚠️ 警告:")
            for w in self.warnings:
                lines.append(f"  {w}")

        lines.append("=" * 60)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 核心验证函数
# ═══════════════════════════════════════════════════════════════

def validate_stereochemistry(
    coords: np.ndarray,
    params: Optional[dict] = None,
) -> StereochemistryReport:
    """完整的立体化学验证。

    Args:
        coords: (L, 3) 坐标数组
        params: 立体化学参数（可选）

    Returns:
        StereochemistryReport: 验证报告
    """
    p = params or RNA_STEREO_PARAMS
    L = len(coords)

    # === 1. Clash 检测 ===
    clash_pairs, clash_distances = detect_clashes(coords, p["clash_distance"])
    has_clashes = len(clash_pairs) > 0
    n_clashes = len(clash_pairs)

    # === 2. 键长检测 ===
    bond_errors = compute_bond_errors(coords, p["bond_length"])
    bond_mean_error = float(np.mean(bond_errors))
    bond_max_error = float(np.max(bond_errors))
    bond_outliers = list(np.where(bond_errors > p["bond_tolerance"])[0])

    # === 3. 键角检测 ===
    if L >= 3:
        angle_errors = compute_angle_errors(coords, p["bond_angle"])
        angle_mean_error = float(np.mean(angle_errors))
        angle_max_error = float(np.max(angle_errors))
        angle_outliers = list(np.where(angle_errors > p["angle_tolerance"])[0])
    else:
        angle_mean_error = 0.0
        angle_max_error = 0.0
        angle_outliers = []

    # === 4. 闭环检测 ===
    closure_distance = float(np.linalg.norm(coords[0] - coords[-1]))
    closure_error = abs(closure_distance - p["closure_target"])

    # === 5. 综合得分 ===
    score = compute_stereo_score(
        has_clashes, n_clashes,
        bond_mean_error, angle_mean_error,
        closure_error, p
    )

    # === 6. 判断有效性 ===
    is_valid = (
        not has_clashes and
        bond_max_error < p["bond_tolerance"] * 2 and
        closure_error < p["closure_tolerance"]
    )

    # === 7. 警告 ===
    warnings = []
    if has_clashes:
        warnings.append(f"发现 {n_clashes} 处原子重叠")
    if bond_max_error > 1.0:
        warnings.append(f"键长最大误差 {bond_max_error:.2f}Å 超过 1.0Å")
    if closure_error > p["closure_tolerance"]:
        warnings.append(f"BSJ 闭环误差 {closure_error:.2f}Å 超过阈值")

    return StereochemistryReport(
        is_valid=is_valid,
        score=score,
        has_clashes=has_clashes,
        n_clashes=n_clashes,
        clash_pairs=clash_pairs,
        clash_distances=clash_distances,
        bond_mean_error=bond_mean_error,
        bond_max_error=bond_max_error,
        bond_outliers=bond_outliers,
        angle_mean_error=angle_mean_error,
        angle_max_error=angle_max_error,
        angle_outliers=angle_outliers,
        closure_distance=closure_distance,
        closure_error=closure_error,
        warnings=warnings,
    )


def detect_clashes(
    coords: np.ndarray,
    clash_distance: float = 2.5,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """检测原子重叠（Clash）。

    Args:
        coords: (L, 3) 坐标
        clash_distance: Å，最小允许距离

    Returns:
        clash_pairs: 重叠的碱基对列表
        clash_distances: 对应的距离
    """
    L = len(coords)

    if L < 4:
        return [], []

    # 计算距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-8)

    # 提取非相邻对（|i-j| >= 2），排除 BSJ（0, L-1）
    i_idx, j_idx = np.triu_indices(L, k=2)
    mask = ~((i_idx == 0) & (j_idx == L - 1))

    valid_i = i_idx[mask]
    valid_j = j_idx[mask]
    valid_dists = dist_matrix[valid_i, valid_j]

    # 找出 clash
    clash_mask = valid_dists < clash_distance
    clash_pairs = list(zip(valid_i[clash_mask], valid_j[clash_mask]))
    clash_distances = list(valid_dists[clash_mask])

    return clash_pairs, clash_distances


def compute_bond_errors(
    coords: np.ndarray,
    target_bond_length: float = 5.9,
) -> np.ndarray:
    """计算键长误差。

    Args:
        coords: (L, 3)
        target_bond_length: Å

    Returns:
        errors: (L,) 每个键的误差
    """
    L = len(coords)

    # 相邻键
    bonds = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

    # BSJ 键（首尾）
    bsj_bond = np.linalg.norm(coords[0] - coords[-1])

    # 所有键
    all_bonds = np.append(bonds, bsj_bond)

    # 误差
    errors = np.abs(all_bonds - target_bond_length)

    return errors


def compute_angle_errors(
    coords: np.ndarray,
    target_angle: float = 109.5,
) -> np.ndarray:
    """计算键角误差。

    Args:
        coords: (L, 3)
        target_angle: 度

    Returns:
        errors: (L-2,) 每个键角的误差（度）
    """
    L = len(coords)

    if L < 3:
        return np.array([])

    # 计算三个连续碱基的角度
    v1 = coords[1:-1] - coords[:-2]  # (L-2, 3)
    v2 = coords[2:] - coords[1:-1]   # (L-2, 3)

    # 计算角度
    cos_angles = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))

    # 误差
    errors = np.abs(angles - target_angle)

    return errors


def compute_stereo_score(
    has_clashes: bool,
    n_clashes: int,
    bond_mean_error: float,
    angle_mean_error: float,
    closure_error: float,
    params: dict,
) -> float:
    """计算综合立体化学得分。

    Returns:
        score: [0, 1]，1 表示完美
    """
    # Clash 惩罚
    clash_score = 0.0 if has_clashes else 1.0

    # 键长得分
    bond_score = max(0.0, 1.0 - bond_mean_error / params["bond_tolerance"])

    # 键角得分
    angle_score = max(0.0, 1.0 - angle_mean_error / params["angle_tolerance"])

    # 闭环得分
    closure_score = max(0.0, 1.0 - closure_error / params["closure_tolerance"])

    # 加权平均
    score = (
        0.40 * clash_score +
        0.30 * bond_score +
        0.15 * angle_score +
        0.15 * closure_score
    )

    return float(np.clip(score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# 损失函数（用于训练）
# ═══════════════════════════════════════════════════════════════

def compute_clash_loss_torch(coords, clash_distance=2.5):
    """PyTorch 版本的 clash_loss（用于训练）。

    Args:
        coords: (B, L, 3) torch.Tensor
        clash_distance: Å

    Returns:
        loss: 标量
    """
    import torch

    B, L, _ = coords.shape

    if L < 4:
        return torch.tensor(0.0, device=coords.device)

    # 计算距离矩阵
    diff = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]  # (B, L, L, 3)
    dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # (B, L, L)

    # 创建 mask
    i_idx, j_idx = torch.triu_indices(L, L, offset=2)
    mask = ~((i_idx == 0) & (j_idx == L - 1))

    # 提取有效距离
    valid_dists = dist_matrix[:, i_idx[mask], j_idx[mask]]  # (B, n_pairs)

    # Clash penalty: max(0, clash_dist - d)^2
    clashes = torch.clamp(clash_distance - valid_dists, min=0.0)
    loss = torch.mean(clashes ** 2)

    return loss


def compute_angle_loss_torch(coords, target_angle=109.5):
    """PyTorch 版本的 angle_loss（用于训练）。

    Args:
        coords: (B, L, 3) torch.Tensor
        target_angle: 度

    Returns:
        loss: 标量
    """
    import torch

    B, L, _ = coords.shape

    if L < 3:
        return torch.tensor(0.0, device=coords.device)

    # 计算角度
    v1 = coords[:, 2:] - coords[:, 1:-1]   # (B, L-2, 3)
    v2 = coords[:, 1:-1] - coords[:, :-2]  # (B, L-2, 3)

    cos_angle = torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8)

    # 目标角度
    target_cos = np.cos(np.deg2rad(target_angle))

    # 损失
    loss = torch.mean((cos_angle - target_cos) ** 2)

    return loss


# ═══════════════════════════════════════════════════════════════
# 示例
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("立体化学验证模块")
    print("=" * 60)

    # 创建测试坐标
    L = 50
    coords = np.random.rand(L, 3) * 10.0

    # 验证
    report = validate_stereochemistry(coords)

    print(report.summary())

    # PyTorch 版本测试
    try:
        import torch
        coords_torch = torch.from_numpy(coords).unsqueeze(0).float()

        clash_loss = compute_clash_loss_torch(coords_torch)
        angle_loss = compute_angle_loss_torch(coords_torch)

        print(f"\nPyTorch 损失函数测试:")
        print(f"  clash_loss: {clash_loss.item():.4f}")
        print(f"  angle_loss: {angle_loss.item():.4f}")
    except ImportError:
        print("\nPyTorch 未安装，跳过损失函数测试")

    print("=" * 60)