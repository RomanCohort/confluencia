"""p_to_3bead.py — 从 P-only CG 坐标重建 FebRNA 式 3-bead CG 表示。

输入: (L, 3) P-only 坐标
输出: (3L, 3) 3-bead CG 坐标 [P, C4', N1/N9] per nucleotide

使用 A-form RNA 的 canonical 偏移 (固定相对坐标)，让 cgRNASP 可以评分。
"""
from __future__ import annotations

import numpy as np

# A-form RNA canonical offsets: 每个核苷酸内 P → C4' → N 的相对坐标 (Å)
# 基于 1EHZ / 1M3N 晶体结构的平均偏移
# P at origin → C4' at ~ (1.85, 0.60, 0.30) → N at ~ (0.0, -0.85, 0.65) from C4'
# 旋转方向由 P-P 骨架方向确定

_OFFSET_P_TO_C4 = np.array([1.85, 0.60, 0.30], dtype=np.float64)
_OFFSET_C4_TO_N = np.array([-0.20, -0.85, 0.65], dtype=np.float64)


def _kabsch_rotation(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """返回把 unit vector v1 旋转到 unit vector v2 方向的 3x3 旋转矩阵。

    用 Rodrigues 旋转: axis = v1 × v2, angle = arccos(v1·v2)。
    退化 (v1≈v2 或 v1≈-v2) 时退化为 identity / 180° rotation。
    """
    a = v1 / (np.linalg.norm(v1) + 1e-8)
    b = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if abs(dot - 1.0) < 1e-6:
        # 同向: identity
        return np.eye(3, dtype=np.float64)
    if abs(dot + 1.0) < 1e-6:
        # 反向: 180° rotation around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = perp / (np.linalg.norm(perp) + 1e-8)
        c = -1.0
        s = 0.0
        cross = axis
    else:
        axis = cross / (np.linalg.norm(cross) + 1e-8)
        c = dot
        s = np.sqrt(1.0 - dot * dot + 1e-8)

    r = np.eye(3, dtype=np.float64)
    r[0, 1] = -axis[2]; r[0, 2] = axis[1]
    r[1, 0] = axis[2]; r[1, 2] = -axis[0]
    r[2, 0] = -axis[1]; r[2, 1] = axis[0]
    return r + np.outer(axis, axis) * (1.0 - c) + r * s


def p_to_3bead(p_coords: np.ndarray) -> np.ndarray:
    """P-only CG → 3-bead CG (FebRNA 格式)。

    Args:
        p_coords: (L, 3) P atom coordinates

    Returns:
        (3L, 3) 坐标, 顺序: [P_0, C4'_0, N_0, P_1, C4'_1, N_1, ...]
    """
    L = len(p_coords)
    out = np.zeros((3 * L, 3), dtype=np.float64)
    out[0::3, :] = p_coords  # P beads

    for i in range(L):
        p = p_coords[i]
        # 确定局部参考方向: 从 5'→3' 的骨架方向
        if i < L - 1:
            tangent = p_coords[i + 1] - p
        else:
            tangent = p - p_coords[i - 1]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent = tangent / tangent_norm

        # 默认 C4 偏移在 +x 方向 (RNA 螺旋外侧)
        ref_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        R = _kabsch_rotation(ref_dir, tangent)
        c4_offset = R @ _OFFSET_P_TO_C4
        n_offset = R @ _OFFSET_C4_TO_N

        out[3 * i + 1] = p + c4_offset
        out[3 * i + 2] = out[3 * i + 1] + n_offset

    return out


def split_3bead_coords(coords_3bead: np.ndarray) -> tuple:
    """拆分 3-bead CG 为 P, C4, N 三个数组。

    Returns:
        (P_coords, C4_coords, N_coords), each (L, 3)
    """
    L = coords_3bead.shape[0] // 3
    return (
        coords_3bead[0::3],
        coords_3bead[1::3],
        coords_3bead[2::3],
    )
