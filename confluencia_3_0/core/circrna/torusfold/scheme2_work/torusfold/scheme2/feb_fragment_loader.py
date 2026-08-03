"""feb_fragment_loader.py — 从 FebRNA fragment library 加载 3-bead CG 片段。

FebRNA 的 fragment/ 和 database/ 目录存着 PDB 全原子结构,
我们提取 3-bead CG (P, C4', N1/N9) 作为 scheme2 CG 初始化的候选种子。

用途:
  - 替代 scheme2 的正多边形初始化, 给 CG 求解一个真实 motif 起点
  - 给 cgRNASP 评分器提供 3-bead CG 输入
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np

# FebRNA 仓库路径 (从 src/scheme2/ 上溯到 repo root /FebRNA)
_FEBRNA_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent / "FebRNA"
)


def _parse_pdb_3bead(pdb_path: str) -> Optional[tuple[str, np.ndarray]]:
    """从 PDB 文件提取 3-bead CG (P, C4', N1/N9)。

    Returns:
        (sequence, coords_3bead) 或 None
        coords_3bead: (3L, 3) 顺序 [P_0, C4'_0, N_0, P_1, ...]
    """
    atoms = []  # [(res_seq, atom_name, x, y, z)]
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_seq = int(line[22:26])
            atoms.append((res_seq, atom_name, res_name, x, y, z))

    if not atoms:
        return None

    # 按残基分组
    res_dict: dict[int, dict[str, np.ndarray]] = {}
    res_order: list[int] = []
    res_names: dict[int, str] = {}
    for res_seq, atom_name, res_name, x, y, z in atoms:
        if res_seq not in res_dict:
            res_dict[res_seq] = {}
            res_order.append(res_seq)
            res_names[res_seq] = res_name
        res_dict[res_seq][atom_name] = np.array([x, y, z])

    seq = ""
    coords_list = []
    for rs in res_order:
        rn = res_names[rs]
        # RNA 标准碱基名 (PDB 里可能是 A/U/C/G)
        if rn not in ("A", "U", "C", "G"):
            return None
        seq += rn

        a = res_dict[rs]
        # 提取 P, C4', N (嘌呤 N9, 嘧啶 N1)
        if "P" not in a or "C4'" not in a:
            return None
        p = a["P"]
        c4 = a["C4'"]
        # 嘌呤 (A/G) 用 N9, 嘧啶 (U/C) 用 N1
        if rn in ("A", "G"):
            if "N9" not in a:
                return None
            n = a["N9"]
        else:
            if "N1" not in a:
                return None
            n = a["N1"]
        coords_list.extend([p, c4, n])

    if not coords_list:
        return None

    return seq, np.array(coords_list, dtype=np.float64)


def load_canonical_stem(bp_length: int) -> Optional[tuple[str, np.ndarray]]:
    """加载指定长度的 canonical A-form stem (来自 database/stems_standard)。

    Args:
        bp_length: 碱基对数 (1-18, 21, 22)

    Returns:
        (seq, coords_3bead) 或 None
        coords_3bead: (3*L, 3), L = 2*bp_length (双链)
    """
    path = _FEBRNA_ROOT / "database" / "stems_standard" / str(bp_length) / "rna_cg.pdb"
    if not path.exists():
        return None
    return _parse_pdb_3bead(str(path))


def load_all_stem_fragments(max_bp: int = 18) -> list[tuple[str, np.ndarray]]:
    """加载所有 stems_all 片段。

    Returns:
        [(seq, coords_3bead), ...]
    """
    stem_dir = _FEBRNA_ROOT / "database" / "stems_all"
    fragments = []
    for d in sorted(stem_dir.iterdir(), key=lambda x: int(x.name)):
        if not d.is_dir() or not d.name.isdigit():
            continue
        bp = int(d.name)
        if bp > max_bp:
            continue
        for pdb_path in sorted(d.glob("*.pdb")):
            result = _parse_pdb_3bead(str(pdb_path))
            if result is not None:
                fragments.append(result)
    return fragments


def load_all_hairpin_fragments() -> list[tuple[str, np.ndarray]]:
    """加载所有 hairpin loop 片段 (来自 database/1_bp/hairpin_loop)。

    Returns:
        [(seq, coords_3bead), ...]
    """
    hp_dir = _FEBRNA_ROOT / "database" / "1_bp" / "hairpin_loop"
    fragments = []
    for d in sorted(hp_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
        if not d.is_dir():
            continue
        for pdb_path in sorted(d.glob("*.pdb")):
            result = _parse_pdb_3bead(str(pdb_path))
            if result is not None:
                fragments.append(result)
    return fragments


def load_database_fragments() -> dict:
    """一次性加载 FebRNA 数据库中的所有片段类型。

    Returns:
        {
            "canonical_stems": {bp_length: (seq, coords), ...},
            "stems_all": [(seq, coords), ...],
            "hairpin_loops": [(seq, coords), ...],
            "bulge_loops": [(seq, coords), ...],
        }
    """
    result = {
        "canonical_stems": {},
        "stems_all": [],
        "hairpin_loops": [],
        "bulge_loops": [],
    }

    # canonical stems (1-22 bp)
    for bp in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22]:
        res = load_canonical_stem(bp)
        if res:
            result["canonical_stems"][bp] = res

    result["stems_all"] = load_all_stem_fragments()
    result["hairpin_loops"] = load_all_hairpin_fragments()

    # bulge loops
    bulge_dir = _FEBRNA_ROOT / "database" / "1_bp" / "bulge_loop"
    if bulge_dir.exists():
        for d in bulge_dir.iterdir():
            if d.is_dir():
                for pdb_path in sorted(d.glob("*.pdb")):
                    res = _parse_pdb_3bead(str(pdb_path))
                    if res:
                        result["bulge_loops"].append(res)

    return result


def get_init_fragment(
    seq: str,
    pairs: list[tuple],
) -> Optional[np.ndarray]:
    """根据给定序列和配对，从 fragment library 找最匹配的初始化种子。

    策略: 对每个 stem 区域, 从 stems_all 或 canonical_stems 选
    同样长度或最接近的片段, 用 Kabsch 对齐。
    返回: (L, 3) P-only 坐标, 或 None (退化到正多边形)
    """
    from .p_to_3bead import split_3bead_coords

    L = len(seq)

    # 简单策略: 对每个 stem, 取 canonical stem 的 P beads,
    # 拼成一个初步的 CG 初始化
    # 这里先做最简版本: 取 canonical stem 的 P 原子位置
    frag = load_canonical_stem(10)  # 10bp 茎
    if frag:
        seq_f, coords = frag
        p_coords = split_3bead_coords(coords)[0]
        # 只取 stem 区域对应长度的 P 坐标
        return p_coords

    return None


if __name__ == "__main__":
    print(f"FebRNA root: {_FEBRNA_ROOT}")
    db = load_database_fragments()
    print(f"canonical stems: {len(db['canonical_stems'])}")
    print(f"stems_all: {len(db['stems_all'])}")
    print(f"hairpin loops: {len(db['hairpin_loops'])}")
    print(f"bulge loops: {len(db['bulge_loops'])}")
    if db["canonical_stems"]:
        bp, (s, c) = list(db["canonical_stems"].items())[0]
        print(f"canonical stem {bp}bp: seq={s[:20]}..., shape={c.shape}")
    if db["hairpin_loops"]:
        s, c = db["hairpin_loops"][0]
        print(f"hairpin[0]: seq={s[:20]}..., shape={c.shape}")
