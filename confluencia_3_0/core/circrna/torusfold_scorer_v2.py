"""
torusfold_scorer.py 扩展 — 增加 SASA 和 motif_accessibility 计算

在现有 TorusFoldScorer 基础上增加：
  1. SASA (溶剂暴露度) 从 3D coords 计算
  2. motif_accessibility: 特殊位点的可及性
  3. dsRNA_mean_length: 配对链的平均长度

不修改原有代码，仅扩展 TorusFoldSignals 数据类。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import torch


@dataclass
class TorusFoldSignalsExtended:
    """扩展的 TorusFold 信号（V2 版本）。

    新增字段（原有字段保留）：
      - sasa_mean: 平均溶剂暴露度
      - sasa_bsj: BSJ 区域暴露度
      - sasa_per_nucleotide: 每个核苷酸的暴露度
      - motif_accessibility: 特殊位点可及性（IRES, m6A）
      - dsRNA_mean_length: dsRNA 链的平均长度
      - pair_chain_lengths: 配对链长度分布
    """
    # === 原有字段 ===
    available: bool
    method: str
    coords: Optional[np.ndarray] = None
    pair_probs: Optional[np.ndarray] = None
    bsj_closure: float = 0.0
    bond_rmsd: float = 0.0
    clash_count: int = 0
    confidence: float = 0.0

    # === V1 已有 ===
    dsRNA_fraction: float = 0.0
    bsj_stability: float = 0.0
    long_range_pair_fraction: float = 0.0

    # === V2 新增 ===
    sasa_mean: float = 0.5
    sasa_bsj: float = 0.5
    sasa_per_nucleotide: Optional[np.ndarray] = None
    motif_accessibility: Dict[str, float] = None  # {"ires": 0.7, "m6a": 0.6}
    dsRNA_mean_length: float = 20.0
    pair_chain_lengths: Optional[List[int]] = None

    # === 其他 ===
    immunogenicity_score: float = 0.0
    translation_efficiency: float = 0.0
    stability_score: float = 0.0


def compute_sasa_from_coords(
    coords: np.ndarray,
    probe_radius: float = 5.0,
    bond_length: float = 3.4,
) -> np.ndarray:
    """从 3D 坐标计算每个核苷酸的溶剂暴露度。

    使用简化滚动球近似（非完整 Shrake-Rupley）。

    Args:
        coords: (N, 3) 核苷酸坐标
        probe_radius: 探针半径
        bond_length: 平均键长

    Returns:
        (N,) 暴露度数组，1 = 完全暴露，0 = 完全埋藏
    """
    if coords is None or len(coords) == 0:
        return np.array([])

    coords = np.asarray(coords, dtype=np.float32)
    n_nuc = len(coords)
    if n_nuc == 1:
        return np.array([1.0])

    # 相互作用半径
    interaction_radius = 2 * probe_radius + bond_length

    # pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # 统计邻居数
    neighbor_mask = (distances > 0) & (distances < interaction_radius)
    neighbor_counts = np.sum(neighbor_mask, axis=1)

    # 埋藏核苷酸的最大邻居数
    max_buried_neighbors = 25.0

    # SASA
    sasa = 1.0 - np.clip(neighbor_counts / max_buried_neighbors, 0.0, 1.0)

    return sasa


def compute_bsj_sasa(coords: np.ndarray, bsj_window: int = 8) -> float:
    """计算 BSJ 区域的溶剂暴露度。

    Args:
        coords: (N, 3) 坐标
        bsj_window: BSJ 两侧窗口大小

    Returns:
        BSJ 区域平均暴露度
    """
    if coords is None or len(coords) < 2 * bsj_window:
        return 0.5

    sasa = compute_sasa_from_coords(coords)
    L = len(coords)

    # BSJ 区域：首尾各 bsj_window 个核苷酸
    bsj_indices = list(range(bsj_window)) + list(range(L - bsj_window, L))
    bsj_sasa = np.mean(sasa[bsj_indices])

    return float(bsj_sasa)


def compute_dsRNA_mean_length(pair_probs: np.ndarray, threshold: float = 0.8) -> float:
    """计算 dsRNA 链的平均长度。

    Args:
        pair_probs: (L, L) 配对概率矩阵
        threshold: 配对阈值

    Returns:
        平均 dsRNA 链长度
    """
    if pair_probs is None:
        return 20.0

    L = pair_probs.shape[0]

    # 找高置信配对
    high_prob_pairs = (pair_probs > threshold)

    # 统计连续配对链长度
    chain_lengths = []
    for i in range(L):
        for j in range(i + 4, L):  # 最小 hairpin loop = 4
            if high_prob_pairs[i, j]:
                # 向两边扩展，找配对链长度
                chain_len = 1
                for k in range(1, min(20, L - j)):  # 最大检查 20bp
                    if i + k < L and j - k >= 0 and high_prob_pairs[i + k, j - k]:
                        chain_len += 1
                    else:
                        break
                chain_lengths.append(chain_len)

    if len(chain_lengths) == 0:
        return 0.0

    return float(np.mean(chain_lengths))


def compute_motif_accessibility(
    sequence: str,
    coords: np.ndarray,
    sasa: np.ndarray,
    motifs: Dict[str, List[int]],
) -> Dict[str, float]:
    """计算特殊位点的可及性。

    Args:
        sequence: circRNA 序列
        coords: 3D 坐标
        sasa: 溶剂暴露度
        motifs: {"ires": [start, end], "m6a": [pos1, pos2], ...}

    Returns:
        {"ires": 0.7, "m6a": 0.6, ...}
    """
    if coords is None or sasa is None:
        return {k: 0.5 for k in motifs.keys()}

    result = {}
    for motif_name, positions in motifs.items():
        if len(positions) == 0:
            result[motif_name] = 0.5
            continue

        # 取位点平均暴露度
        motif_sasa = np.mean(sasa[positions])
        result[motif_name] = float(motif_sasa)

    return result


def detect_ires_region(sequence: str) -> List[int]:
    """启发式检测可能的 IRES 区域。

    简化版：检测 GC-rich 区域（IRES 常见特征）。

    Returns:
        可能的 IRES 核苷酸索引列表
    """
    L = len(sequence)
    window = 30
    gc_threshold = 0.6

    ires_candidates = []

    for i in range(L - window):
        region = sequence[i:i + window]
        gc = sum(1 for c in region.upper() if c in "GC") / window
        if gc > gc_threshold:
            ires_candidates.extend(range(i, i + window))

    # 去重
    ires_candidates = list(set(ires_candidates))

    # 如果未找到，取中间区域
    if len(ires_candidates) == 0:
        mid_start = L // 4
        mid_end = 3 * L // 4
        ires_candidates = list(range(mid_start, mid_end))

    return ires_candidates


def detect_m6a_sites(sequence: str) -> List[int]:
    """启发式检测可能的 m6A 位点。

    m6A motif: DRACH (D=A/G/U, R=A/G, H=A/C/U)
    """
    L = len(sequence)
    seq_upper = sequence.upper().replace("T", "U")

    m6a_candidates = []

    # DRACH motif
    d_bases = "AGU"
    r_bases = "AG"
    h_bases = "ACU"

    for i in range(L - 5):
        if seq_upper[i] in d_bases and seq_upper[i + 1] in r_bases and \
           seq_upper[i + 2] == "A" and seq_upper[i + 3] in h_bases:
            m6a_candidates.append(i + 2)  # A 是修饰位点

    return m6a_candidates


def extract_extended_signals(
    sequence: str,
    coords: Optional[np.ndarray],
    pair_probs: Optional[np.ndarray],
    gene_expr: Optional[Dict[str, float]] = None,
) -> TorusFoldSignalsExtended:
    """从 TorusFold 输出提取扩展信号。

    Args:
        sequence: circRNA 序列
        coords: 3D 坐标 (L, 3)
        pair_probs: 配对概率 (L, L)
        gene_expr: 基因表达

    Returns:
        TorusFoldSignalsExtended: 包含 SASA、motif 可及性等
    """
    L = len(sequence)

    # 检查可用性
    available = coords is not None and len(coords) > 0
    method = "torusfold" if available else "heuristic_fallback"

    # === 计算基础信号 ===
    if available:
        sasa = compute_sasa_from_coords(coords)
        sasa_mean = float(np.mean(sasa))
        sasa_bsj = compute_bsj_sasa(coords)
        sasa_per_nuc = sasa

        # BSJ closure
        bsj_closure = float(np.linalg.norm(coords[0] - coords[-1]))
        bsj_stability = float(1.0 / (1.0 + bsj_closure / 5.9))  # sigmoid

        # Bond RMSD
        bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        bond_rmsd = float(np.sqrt(np.mean((bond_lengths - 5.9) ** 2)))

        # dsRNA
        if pair_probs is not None:
            dsRNA_frac = float(np.mean(pair_probs > 0.8))
            dsRNA_mean_len = compute_dsRNA_mean_length(pair_probs)
        else:
            dsRNA_frac = 0.3
            dsRNA_mean_len = 20.0

        # Motif accessibility
        ires_pos = detect_ires_region(sequence)
        m6a_pos = detect_m6a_sites(sequence)
        motifs = {"ires": ires_pos, "m6a": m6a_pos}
        motif_access = compute_motif_accessibility(sequence, coords, sasa, motifs)

        confidence = 0.85

    else:
        # 启发式兜底
        sasa_mean = 0.5
        sasa_bsj = 0.5
        sasa_per_nuc = None
        bsj_closure = 5.9
        bsj_stability = 0.5
        bond_rmsd = 1.0
        dsRNA_frac = 0.3
        dsRNA_mean_len = 20.0
        motif_access = {"ires": 0.5, "m6a": 0.5}
        confidence = 0.3

    # === 免疫原性、翻译效率、稳定性评分 ===
    # 从结构信号推导
    immunogenicity = dsRNA_frac * 0.6 + (1 - sasa_bsj) * 0.4
    translation = motif_access.get("ires", 0.5) * 0.5 + sasa_mean * 0.3 + (1 - dsRNA_frac) * 0.2
    stability = bsj_stability * 0.7 + (1 - bond_rmsd / 5.0) * 0.3

    return TorusFoldSignalsExtended(
        available=available,
        method=method,
        coords=coords,
        pair_probs=pair_probs,
        bsj_closure=bsj_closure,
        bond_rmsd=bond_rmsd,
        clash_count=0,
        confidence=confidence,

        dsRNA_fraction=dsRNA_frac,
        bsj_stability=bsj_stability,
        long_range_pair_fraction=0.0,

        sasa_mean=sasa_mean,
        sasa_bsj=sasa_bsj,
        sasa_per_nucleotide=sasa_per_nuc,
        motif_accessibility=motif_access,
        dsRNA_mean_length=dsRNA_mean_len,
        pair_chain_lengths=None,

        immunogenicity_score=float(np.clip(immunogenicity, 0.0, 1.0)),
        translation_efficiency=float(np.clip(translation, 0.0, 1.0)),
        stability_score=float(np.clip(stability, 0.0, 1.0)),
    )


# === 集成到 TorusFoldScorer ===

def extend_torusfold_scorer_v2(scorer_class):
    """为 TorusFoldScorer 增加 V2 扩展方法。

    使用 monkey patching，不修改原有代码。
    """
    def extract_signals_extended(self, sequence: str, **kwargs):
        """V2 版本的信号提取（含 SASA）。"""
        # 先调用原有方法
        base_signals = self.extract_signals(sequence, **kwargs)

        # 转换为扩展版
        extended = extract_extended_signals(
            sequence,
            coords=base_signals.coords if base_signals.available else None,
            pair_probs=base_signals.pair_probs if base_signals.available else None,
        )

        return extended

    # Monkey patch
    scorer_class.extract_signals_extended = extract_signals_extended
    return scorer_class