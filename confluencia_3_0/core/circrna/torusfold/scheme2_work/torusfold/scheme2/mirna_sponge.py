"""mirna_sponge.py - miRNA duplex 兼容性 (sponge 活性) 真版计算。

用 miRBase mature.fa 库 + ViennaRNA duplexfold, 对 circRNA 序列做
miRNA-mRNA duplex 评估:
  1. seed 区 (miRNA 2-8 位) 必须与靶点完全 Watson-Crick 互补 (硬门控, 哈希查)
  2. ViennaRNA RNA.duplexfold 算 duplex ΔG (kcal/mol, 越低越稳定)
  3. ΔG <= 阈值 (-10) 视为有效结合

生物学: circRNA 作 miRNA sponge = 能结合多个 miRNA 形成稳定 duplex,
        sequester 它们使其无法下调其他靶 mRNA。
        sponge_score = f(命中 miRNA 数, 平均 ΔG, 序列长度)

性能: seed (2-8 位) 反向互补哈希索引 O(L) 查表, 命中才调 duplexfold。
      hsa-miR ~2600 条, 60nt 序列通常命中 < 50 次 duplexfold。
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import gzip
import numpy as np

_COMP = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A"}
DEFAULT_MATURE_FA = r"D:\LENOVO\Documents\mature.fa"
_cache: Dict[str, object] = {}


def _load_mirna(mature_fa: str, species_prefix: str = "hsa-") -> List[Tuple[str, str]]:
    """加载 mature.fa, 筛指定物种, 返回 [(name, seq), ...]。缓存。"""
    key = (mature_fa, species_prefix)
    if key in _cache:
        return _cache[key]  # type: ignore

    p = Path(mature_fa)
    if not p.exists():
        raise FileNotFoundError(f"mature.fa 不存在: {mature_fa}")

    entries: List[Tuple[str, str]] = []
    cur_name: Optional[str] = None
    cur_seq: List[str] = []

    def _flush():
        if cur_name is not None and cur_seq:
            seq = "".join(cur_seq).upper().replace("T", "U")
            if all(c in "AUGC" for c in seq) and len(seq) >= 18:
                entries.append((cur_name, seq))

    opener = gzip.open if str(mature_fa).endswith(".gz") else open
    with opener(mature_fa, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                _flush()
                parts = line[1:].split()
                cur_name = parts[0] if parts else None
                cur_seq = []
            elif line:
                cur_seq.append(line)
    _flush()

    filtered = [(n, s) for (n, s) in entries if n.startswith(species_prefix)]
    _cache[key] = filtered
    return filtered


def _seed_index(mirna_list: List[Tuple[str, str]]) -> Dict[str, List[int]]:
    """建 seed (miRNA 2-8 位) 反向互补 → miRNA 索引列表 (哈希查表)。"""
    idx: Dict[str, List[int]] = {}
    for k, (_name, seq) in enumerate(mirna_list):
        if len(seq) < 9:
            continue
        seed = seq[1:8]  # 2-8 位 (0-indexed 1..7), 7nt
        target = "".join(_COMP[b] for b in reversed(seed))
        idx.setdefault(target, []).append(k)
    return idx


def _duplex_dg(mirna_seq: str, target_seq: str) -> float:
    """ViennaRNA duplexfold 算 miRNA-target duplex ΔG (kcal/mol)。"""
    import RNA  # type: ignore
    fc = RNA.duplexfold(mirna_seq, target_seq)
    return float(fc.energy)


def compute_sponge(
    sequence: str,
    mature_fa: str = DEFAULT_MATURE_FA,
    species: str = "hsa-",
    dg_threshold: float = -10.0,
    max_hits: int = 50,
) -> Dict:
    """算 circRNA 序列的 miRNA sponge 活性。

    Args:
        sequence: circRNA 序列 (ACGU, 也接受 ACTG)
        mature_fa: miRBase mature.fa 路径
        species: 物种前缀 (hsa- = 人类)
        dg_threshold: ΔG 阈值 (kcal/mol), 低于此视为有效结合
        max_hits: 最多记录的命中 miRNA 数

    Returns:
        dict: sponge_score [0,1], n_hits, mean_dg, hits [(name, dg, pos)]
    """
    seq = sequence.upper().replace("T", "U")
    L = len(seq)
    if L < 9:
        return {"sponge_score": 0.0, "n_hits": 0, "mean_dg": 0.0, "hits": []}

    try:
        mirna_list = _load_mirna(mature_fa, species)
    except FileNotFoundError:
        gu_frac = sum(1 for c in seq if c in "GU") / L
        return {"sponge_score": float(gu_frac * min(L / 200.0, 1.0)),
                "n_hits": 0, "mean_dg": 0.0, "hits": [], "fallback": "no_mirna_lib"}

    if not mirna_list:
        gu_frac = sum(1 for c in seq if c in "GU") / L
        return {"sponge_score": float(gu_frac * min(L / 200.0, 1.0)),
                "n_hits": 0, "mean_dg": 0.0, "hits": [], "fallback": "no_species"}

    seed_idx = _seed_index(mirna_list)

    # 扫 circRNA: 滑动 7nt 查 seed 索引, 命中后取 miRNA 长度的窗口调 duplexfold。
    # ViennaRNA 自动处理 5'/3' 对齐, 窗口取靶点周围 (靶点起点 pos, 往前往后各取一半)。
    hits: List[Tuple[str, float, int]] = []
    for pos in range(L - 6):
        window7 = seq[pos:pos + 7]
        if window7 not in seed_idx:
            continue
        for k in seed_idx[window7]:
            mirna_name, mirna_seq = mirna_list[k]
            ml = len(mirna_seq)
            # 靶点窗口: 以 seed 靶点为中心, 取 ml nt (靶点前 ml-7 + 靶点 7)
            win_start = max(0, pos - (ml - 7))
            win_end = min(L, pos + 7 + (ml - 7) // 2)
            target_win = seq[win_start:win_end]
            if len(target_win) < 10:
                continue
            dg = _duplex_dg(mirna_seq, target_win)
            if dg <= dg_threshold:
                hits.append((mirna_name, dg, pos))

    if not hits:
        return {"sponge_score": 0.0, "n_hits": 0, "mean_dg": 0.0, "hits": []}

    # 去重 (同 miRNA 多处命中只留最强 ΔG)
    best: Dict[str, Tuple[float, int]] = {}
    for name, dg, pos in hits:
        if name not in best or dg < best[name][0]:
            best[name] = (dg, pos)
    dedup = sorted(best.items(), key=lambda x: x[1][0])[:max_hits]

    n_hits = len(dedup)
    mean_dg = float(np.mean([dg for _, (dg, _) in dedup]))
    # sponge_score: 命中数 + ΔG 强度 + 长度因子 (饱和)
    hit_factor = min(n_hits / 20.0, 1.0)
    dg_factor = min(-mean_dg / 25.0, 1.0)
    len_factor = min(L / 200.0, 1.0)
    sponge_score = float(0.4 * hit_factor + 0.4 * dg_factor + 0.2 * len_factor)

    return {
        "sponge_score": sponge_score,
        "n_hits": n_hits,
        "mean_dg": mean_dg,
        "hits": [(name, dg, pos) for name, (dg, pos) in dedup],
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    # 含 let-7a seed 靶点 (CUACCUC) 的测试序列
    test = "ACGACGACGACGCUACCUCACGACGACGACGACGACGACGACGACG"
    r = compute_sponge(test)
    print(f"elapsed {time.time()-t0:.2f}s")
    print(f"sponge_score={r['sponge_score']:.3f} n_hits={r['n_hits']} mean_dg={r['mean_dg']:.1f}")
    for name, dg, pos in r["hits"][:6]:
        print(f"  {name} dg={dg:.1f} pos={pos}")

    print()
    # demo 60nt
    test2 = "AUGCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUACG"
    r2 = compute_sponge(test2)
    print(f"demo60 sponge_score={r2['sponge_score']:.3f} n_hits={r2['n_hits']} mean_dg={r2['mean_dg']:.1f}")
    for name, dg, pos in r2["hits"][:5]:
        print(f"  {name} dg={dg:.1f} pos={pos}")
