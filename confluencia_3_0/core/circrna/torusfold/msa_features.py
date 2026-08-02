"""msa_features.py — MSA / 进化保守性 pair 特征通道（接口框架）。

回应评审点「缺失序列上下文与保守性信息的显式融合」：
AlphaFold 等结构预测依赖 MSA 共进化信号，本模型目前只有单条序列 +
ViennaRNA pair_probs。此模块预留**进化保守性先验**的 pair 特征通道。

设计：
  1. load_rfam_consensus()    — 解析 Rfam.seed (Stockholm) 得每条序列的
                                consensus SS（`#=GC SS_cons`，实验验证配对）
  2. consensus_to_pair(ss)    — 点括号 SS → [L,L] 配对矩阵
  3. match_family(seq)        — k-mer 相似度匹配 Rfam 家族
  4. fuse_pair_probs()        — 与 ViennaRNA pair_probs 融合 (α 混合)

状态：
  ⚠️ 接口 + 逻辑已实现，但【未接入训练】。需要下载 Rfam.seed:
     https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz
     (~50MB, 需外网)。A800 部署后跑 rfam_to_training.py 下载并测试接入。

这比完整 MSA 共变异 (JackHMMER+EVcouplings) 轻量：consensus SS 来自 Rfam
实验验证的家族共识配对，是"进化保守的碱基配对"的可靠代理，无需在线搜索。
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════
# 1. Rfam consensus SS 解析（复用 rfam_to_training.parse_stockholm）
# ═══════════════════════════════════════════════════════════════

def load_rfam_consensus(rfam_seed_path: str) -> List[Dict]:
    """从 Rfam.seed (Stockholm) 解析家族 consensus SS。

    Returns:
        [{family, sequence, secondary_structure, length}] — 每条序列的
        实验验证共识二级结构 (dot-bracket, 含 <> 等非规范配对)。
    """
    from rfam_to_training import parse_stockholm
    with open(rfam_seed_path, 'r', encoding='utf-8', errors='replace') as f:
        return parse_stockholm(f)


# ═══════════════════════════════════════════════════════════════
# 2. consensus SS → [L,L] 配对矩阵
# ═══════════════════════════════════════════════════════════════

def consensus_to_pair(ss: str, L: int) -> np.ndarray:
    """点括号二级结构 → [L,L] 0/1 配对矩阵。

    支持 `()`, `[]`, `{}`, `<>` 配对符号 (Rfam 非规范配对用方括号标注)。
    Args:
        ss: dot-bracket 字符串 (长度 = L)
        L: 序列长度 (用于 padding)
    Returns:
        [L,L] float32 矩阵, pair[i,j]=1 if i,j 配对.
    """
    pair = np.zeros((L, L), dtype=np.float32)
    stack: Dict[str, List[int]] = {')': [], ']': [], '}': [], '>': []}
    open_map = {')': '(', ']': '[', '}': '{', '>': '<'}
    close_map = {'(': ')', '[': ']', '{': '}', '<': '>'}
    n = min(len(ss), L)
    for i in range(n):
        c = ss[i]
        if c in '([{<':
            close_c = close_map[c]
            stack[close_c].append(i)
        elif c in ')]}>':
            if stack[c]:
                j = stack[c].pop()
                pair[i, j] = 1.0
                pair[j, i] = 1.0
    return pair


# ═══════════════════════════════════════════════════════════════
# 3. 家族匹配（k-mer 相似度，轻量代理 JackHMMER）
# ═══════════════════════════════════════════════════════════════

def match_family(seq: str, rfam_entries: List[Dict],
                 k: int = 6, top_n: int = 3) -> List[Dict]:
    """用 k-mer 相似度匹配序列到 Rfam 家族。

    完整方案是 JackHMMER (profile HMM 搜索)，但 k-mer 相似度是零依赖的
    轻量代理。返回 top_n 个候选家族。

    Args:
        seq: 查询序列 (ACGU)
        rfam_entries: load_rfam_consensus() 的输出
        k: k-mer 大小
        top_n: 返回候选数
    """
    query_kmers = set(seq[i:i + k] for i in range(len(seq) - k + 1))
    if not query_kmers:
        return []
    scored = []
    for e in rfam_entries:
        fam_seq = e['sequence']
        fam_kmers = set(fam_seq[i:i + k] for i in range(len(fam_seq) - k + 1))
        if not fam_kmers:
            continue
        # Jaccard 相似度
        inter = len(query_kmers & fam_kmers)
        union = len(query_kmers | fam_kmers)
        if union > 0:
            scored.append((inter / union, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_n]]


# ═══════════════════════════════════════════════════════════════
# 4. pair_probs 融合
# ═══════════════════════════════════════════════════════════════

def fuse_pair_probs(vienna_pp: np.ndarray, consensus_pp: np.ndarray,
                    alpha: float = 0.7) -> np.ndarray:
    """ViennaRNA pair_probs 与 consensus SS 配对融合。

    fused = α · vienna_pp + (1-α) · consensus_pp

    α=0.7 默认: 主要保留 ViennaRNA 热力学预测，掺入保守性先验。
    高保守区 (Rfam 家族明确配对) 由 consensus 兜底，低保守区靠 Vienna。

    Args:
        vienna_pp: [L,L] ViennaRNA 碱基配对概率
        consensus_pp: [L,L] 0/1 consensus 配对矩阵
        alpha: Vienna 权重 (0-1)
    """
    return alpha * vienna_pp + (1.0 - alpha) * consensus_pp


# ═══════════════════════════════════════════════════════════════
# 5. 高层接口：给定序列 + Rfam 数据 → 融合 pair 特征
# ═══════════════════════════════════════════════════════════════

def get_evolutionary_pair_features(
    seq: str,
    rfam_entries: List[Dict],
    vienna_pp: np.ndarray,
    alpha: float = 0.7,
    similarity_thresh: float = 0.15,
) -> Optional[np.ndarray]:
    """为单条序列计算融合的进化-pair 特征。

    流程: 匹配家族 → 取最佳共识 SS → 转配对矩阵 → 与 Vienna 融合。
    若没有家族相似度超过阈值，返回 None (调用方 fallback 到纯 Vienna)。

    Args:
        seq: ACGU 序列
        rfam_entries: load_rfam_consensus() 输出
        vienna_pp: [L,L] ViennaRNA pair_probs
        alpha: Vienna 权重
        similarity_thresh: 家族匹配最小 Jaccard (低于则视为无保守信号)
    """
    if not seq or not rfam_entries:
        return None
    matches = match_family(seq, rfam_entries, top_n=1)
    if not matches or matches[0]['family'] == 'unknown':
        return None
    best = matches[0]
    # Jaccard 阈值过滤弱匹配
    fam_seq = best['sequence']
    q = set(seq[i:i + 6] for i in range(max(0, len(seq) - 5)))
    f = set(fam_seq[i:i + 6] for i in range(max(0, len(fam_seq) - 5)))
    if not q or not f:
        return None
    jac = len(q & f) / len(q | f)
    if jac < similarity_thresh:
        return None
    ss = best['secondary_structure']
    consensus = consensus_to_pair(ss, len(seq))
    return fuse_pair_probs(vienna_pp, consensus, alpha=alpha)


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== msa_features 自测 ===')
    # consensus_to_pair
    ss = '..((..))..'
    pair = consensus_to_pair(ss, 10)
    print('pair[2,7] =', pair[2, 7], '(应 1.0)')
    assert pair[2, 7] == 1.0 and pair[7, 2] == 1.0

    # 非规范配对 (方括号)
    ss2 = '..[[..]]..'
    pair2 = consensus_to_pair(ss2, 10)
    print('pair2[2,7] =', pair2[2, 7], '(方括号也应 1.0)')
    assert pair2[2, 7] == 1.0

    # fuse
    vienna = np.full((10, 10), 0.05)
    fused = fuse_pair_probs(vienna, pair, alpha=0.7)
    print('fused[2,7] =', round(float(fused[2, 7]), 3), '(应 0.7*0.05+0.3*1.0=0.335)')
    assert abs(fused[2, 7] - 0.335) < 1e-6
    print('PASS')
