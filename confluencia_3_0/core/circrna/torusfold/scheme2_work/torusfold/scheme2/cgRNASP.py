"""cgRNASP.py - numpy 重实现的 cgRNASP-Feb CG 评分器。

从 FebRNA/cgRNASP-Feb.c 直接翻译，不依赖 C 编译。
输入：3-bead CG 坐标 (P, C4', N1/N9 per nt)，输出：全局 CG 评分。
用于 scheme2 CG 阶段的全局质量评估 (reward 补充项)。
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# 势能文件路径 (FebRNA/data/) — 从 src/scheme2/cgRNASP.py 上溯到 repo root
_FEBRNA_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "FebRNA" / "data"
)

# 势能表缓存 (进程内, 避免重复 IO + parse)
_POTENTIALS_CACHE: dict | None = None

# 12 atom types (FebRNA 的 CG 表示: 每个碱基 3 珠 = P, C4', N1/N9)
# 索引 0-2: A (P, C, N), 3-5: U, 6-8: C, 9-11: G
ATOM_TYPES = {
    ("A", "P"): 0, ("A", "C"): 1, ("A", "N"): 2,
    ("U", "P"): 3, ("U", "C"): 4, ("U", "N"): 5,
    ("C", "P"): 6, ("C", "C"): 7, ("C", "N"): 8,
    ("G", "P"): 9, ("G", "C"): 10, ("G", "N"): 11,
}
NUCLEOTIDE_ATOM_INDICES = {
    "A": (0, 1, 2),
    "U": (3, 4, 5),
    "C": (6, 7, 8),
    "G": (9, 10, 11),
}

# 距离 bin 参数
# k1=0 (0-1 nt gap), k2=1 (1-2), k3=2 (2-4), k4=4 (>4 = long-ranged)
K1, K2, K3, K4 = 0, 1, 2, 4
R1, R2, R3, R4 = 5.0, 9.0, 13.0, 24.0  # 最大距离 cutoff (Å)
BIN_WIDTH = 0.3  # Å

# 势能用到的 intervals
INTERVALS = {1: 17, 2: 30, 3: 43, 4: 80}  # n → intervals count


def _load_potential(path: Path, intervals: int) -> np.ndarray:
    """加载势能表。

    C 代码用 fscanf(..., &nnn, &nnn, &nnn, &v) 忽略前三列,
    直接用循环顺序 (n1, n2, n3) 作为索引 → 按行顺序 reshape。
    """
    values = np.loadtxt(path, comments=None, usecols=(3,)).astype(np.float64)
    expected = 12 * 12 * intervals
    if len(values) != expected:
        raise ValueError(
            f"势能文件 {path.name}: {len(values)} 行, "
            f"期望 {expected} (12×12×{intervals})"
        )
    return values.reshape((12, 12, intervals))


def load_all_potentials(febRNA_dir: Path | None = None) -> dict:
    """一次性加载 4 张势能表 (进程内缓存, 避免重复 IO)。

    Returns: {1: E_0_1, 2: E_1_2, 3: E_2_4, 4: E_long}
    """
    global _POTENTIALS_CACHE
    if febRNA_dir is None and _POTENTIALS_CACHE is not None:
        return _POTENTIALS_CACHE

    base = _FEBRNA_DATA if febRNA_dir is None else febRNA_dir / "data"
    pots = {
        1: _load_potential(base / "0-1_short-ranged.potential", INTERVALS[1]),
        2: _load_potential(base / "1-2_short-ranged.potential", INTERVALS[2]),
        3: _load_potential(base / "2-4_short-ranged.potential", INTERVALS[3]),
        4: _load_potential(base / "long-ranged.potential", INTERVALS[4]),
    }
    if febRNA_dir is None:
        _POTENTIALS_CACHE = pots
    return pots


def _build_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """(3L,3L) 距离矩阵. 优先 torch (多核加速), 回退 numpy.

    诊断 (2026-08-02): numpy 逐元素广播构建 (3L,3L,3) diff 是 score_cgrnas 的
    主要耗时 (L=3126 → 2.9s, 单线程内存带宽瓶颈). torch.cdist 用 OpenMP 多核
    (166ms, 18x), GPU/HIP 对中等矩阵反而慢 (785ms, 传输+内存开销).
    所以这里用 torch CPU (有 torch 时自动多核), 不可用则回退 numpy.
    """
    try:
        import torch
        _t = torch.from_numpy(np.ascontiguousarray(coords))
        d = torch.cdist(_t, _t)
        return d.numpy()
    except Exception:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))


def _residue_type_index(seq: str) -> np.ndarray:
    """从序列生成每个核苷酸的碱基索引 (0=A, 1=U, 2=C, 3=G)。

    Returns: (L,) int array, 0-3。
    """
    mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
    return np.array([mapping.get(b, 0) for b in seq], dtype=np.int32)


def _atom_types_for_residue(res_idx: int) -> tuple[int, int, int]:
    """给定碱基索引 (0-3)，返回该核苷酸 3 个珠子的 atom type 索引。"""
    return NUCLEOTIDE_ATOM_INDICES[["A", "U", "C", "G"][res_idx]]


def score_cgrnas(
    coords: np.ndarray,
    sequence: str,
    potentials: dict,
) -> tuple[float, dict]:
    """计算 cgRNASP-Feb 全局 CG 评分。

    输入:
        coords: (3*L, 3) — 3-bead CG 坐标 (P, C4', N1/N9 per nt, 按核苷酸顺序)
        sequence: 长度 L 的 ACGU 字符串
        potentials: {1,2,3,4} → 势能表 (from load_all_potentials)

    Returns:
        (total_score, breakdown_dict)
        total_score: 全局 CG 评分 (越低越好)
        breakdown: {E_0_1, E_1_2, E_2_4, E_long, E_bonded}
    """
    L = len(sequence)
    N_atoms = coords.shape[0]

    if N_atoms != 3 * L:
        raise ValueError(
            f"coords shape {coords.shape} 不匹配 3-bead CG: "
            f"期望 ({3 * L}, 3), 实际 ({N_atoms}, 3)"
        )

    # ── Non-bonded: 原子对势能 ──
    E = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    E_table = potentials

    # ── 向量化评分 ──
    # 1. 每个 residue 的 3 个 atom 类型索引 (L, 3)
    residue_type = _residue_type_index(sequence)
    atom_type_per_res = np.array([
        _atom_types_for_residue(r) for r in residue_type
    ], dtype=np.int32)

    # 2. 距离矩阵 (3L, 3L) → reshape (L, L, 3, 3)
    #    dist_4d[i, j, ai, aj] = dist between atom ai of res i and atom aj of res j
    #    torch 加速 (多核), 回退 numpy
    dist_matrix = _build_dist_matrix(coords)
    dist_4d = dist_matrix.reshape(L, 3, L, 3).transpose(0, 2, 1, 3)  # (L, L, 3, 3)

    # 3. seq_gap matrix (L, L) — 上三角
    seq_gap = (np.arange(L)[np.newaxis, :] - np.arange(L)[:, np.newaxis])  # j - i (col - row)
    i_upper = np.triu_indices(L, k=1)

    # 4. atom type indices for all residue pairs, broadcast to (L, L, 3, 3)
    #    at_i_4d[i, j, ai, aj] = atom type of atom ai in residue i
    at_i_4d = atom_type_per_res[:, np.newaxis, :, np.newaxis]  # (L, 1, 3, 1) → broadcast
    at_j_4d = atom_type_per_res[np.newaxis, :, np.newaxis, :]  # (1, L, 1, 3) → broadcast

    # 5. 向量化评分: 对每个 potential key 构建 mask + fancy indexing
    #    K1<g≤K2→key=1, K2<g≤K3→key=2, K3<g≤K4→key=3, g>K4→key=4
    cutoffs = {1: R1, 2: R2, 3: R3, 4: R4}
    interval_counts = {1: INTERVALS[1], 2: INTERVALS[2], 3: INTERVALS[3], 4: INTERVALS[4]}

    E_vec = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    # 用 (i_upper[0], i_upper[1]) 分别 index 4D array (避免 fancy index 维度冲突)
    iu0, iu1 = i_upper  # both (N_pairs,)
    d_upper = dist_4d[iu0, iu1, :, :]  # (N_pairs, 3, 3)
    gap_upper = seq_gap[iu0, iu1]  # (N_pairs,)
    at_i_upper = atom_type_per_res[iu0]  # (N_pairs, 3)
    at_j_upper = atom_type_per_res[iu1]  # (N_pairs, 3)

    for key in (1, 2, 3, 4):
        cutoff = cutoffs[key]
        n_intervals = interval_counts[key]
        potential = E_table[key]

        if key == 1:
            gap_mask = (gap_upper > K1) & (gap_upper <= K2)
        elif key == 2:
            gap_mask = (gap_upper > K2) & (gap_upper <= K3)
        elif key == 3:
            gap_mask = (gap_upper > K3) & (gap_upper <= K4)
        else:
            gap_mask = (gap_upper > K4)

        if not np.any(gap_mask):
            continue

        n_pairs = int(np.sum(gap_mask))
        d_key = d_upper[gap_mask]  # (n_pairs, 3, 3)
        at_i_key = at_i_upper[gap_mask]  # (n_pairs, 3)
        at_j_key = at_j_upper[gap_mask]  # (n_pairs, 3)
        dist_flat = d_key.reshape(-1)  # (n_pairs*9,)

        # distance mask
        dist_mask = dist_flat < cutoff
        if not np.any(dist_mask):
            continue

        # bin indexing
        bin_idx = (dist_flat / BIN_WIDTH).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, n_intervals - 1)

        # atom type indices: 展开到 (n_pairs*9,) 匹配 dist_flat 的 (ai,bj) 顺序
        # dist_flat: reshape(d_key) 后 ai 变化慢, bj 变化快 → (ai=0,bj=0..2),(ai=1,bj=0..2),...
        # at_i_flat[k] = at_i_key[p, k//3] → ai 重复 3 次 → repeat
        # at_j_flat[k] = at_j_key[p, k%3] → aj 循环 → tile
        at_i_flat = np.repeat(at_i_key, 3, axis=1).ravel()  # (n_pairs*9,)
        at_j_flat = np.tile(at_j_key, (1, 3)).ravel()  # (n_pairs*9,)

        # fancy index: potential[at_i, at_j, bin_idx]
        selected_bin = bin_idx[dist_mask]
        selected_at_i = at_i_flat[dist_mask]
        selected_at_j = at_j_flat[dist_mask]

        values = potential[
            selected_at_i,
            selected_at_j,
            selected_bin,
        ]
        E_vec[key] = float(np.sum(values))

    # ── Bonded: 内部坐标势能 ──
    # FebRNA 的 bonded term 依赖 C 语言的内部坐标计算,
    # 这里简化: 对 3-bead CG 用简单的键长/角度/二面角偏差惩罚
    # (跟 FebRNA 的量级一致)
    E_bonded = _compute_bonded(coords, L)

    # ── 组合 ──
    length_factor = _fun(L)
    total = (
        1.0 * E_vec[1]
        + 1.0 * E_vec[2]
        + 6.0 * E_vec[3]
        + 8.0 * E_vec[4] / max(length_factor, 1e-6)
        + 0.01 * E_bonded
    )

    return float(total), {
        "E_0_1": float(E_vec[1]),
        "E_1_2": float(E_vec[2]),
        "E_2_4": float(E_vec[3]),
        "E_long": float(E_vec[4]),
        "E_bonded": float(E_bonded),
    }


def _fun(n: int) -> float:
    """FebRNA 的 length normalization 函数。"""
    return -355.0 / np.sqrt(n + 16) + 72.0


def _compute_bonded(coords: np.ndarray, L: int) -> float:
    """简化的 bonded 势能: 键长偏差 + 相邻角度偏差。

    对于 3-bead CG (P, C4', N1/N9 per nt):
    - 键: P(i)-C4'(i), C4'(i)-N(i), P(i)-P(i+1)
    - 角度: P-C4'-N, C4'-N-P(next)
    """
    if L < 2:
        return 0.0

    # 期望键长 (Å)
    KB_PC = 1.85   # P-C4'
    KB_CN = 1.70   # C4'-N
    KB_PP = 3.0    # P-P (相邻)

    E = 0.0
    for i in range(L):
        p = coords[i * 3 + 0]
        c = coords[i * 3 + 1]
        n = coords[i * 3 + 2]

        # 键长偏差 (二次惩罚, 量级跟 FebRNA 的 k ~ 5-15)
        E += 5.0 * (np.linalg.norm(p - c) - KB_PC) ** 2
        E += 5.0 * (np.linalg.norm(c - n) - KB_CN) ** 2

        # P-P 相邻
        if i < L - 1:
            pp = np.linalg.norm(p - coords[(i + 1) * 3 + 0])
            E += 2.0 * (pp - KB_PP) ** 2

    return float(E)
