"""
structure_export.py — TorusFold 输出 → PDB 字符串 + 指纹旁路 JSON。

A1 档（coarse-grained，每核苷酸一个 P 原子）：
    TorusFold 的 coords(B,L,3) 是 C1'/P proxy 的单点坐标。
    本模块把每个点当作磷酸 P 原子，写成 PDB ATOM 记录，
    Mol* 读入后自动连 backbone、渲染为 cartoon tube。

    闭环（circRNA 的灵魂）：
        PDB 格式本身不支持"环"，线性 backbone 在末端会断开。
        这里用 CONECT 记录显式把最后一个残基的 P 连回第一个，
        Mol* 会画出闭合的 circRNA 环。BSJ 处的残基用特殊
        residue name (BSJ) 标记，前端可单独上色高亮。

A2 档（全原子，v2）：
    用 biotite + 理想化核苷酸模板补全碱基/糖环原子。
    接口不变，只是 _build_atom_records 从"只写 P"换成"拼模板"。

输入契约（来自 torus_coord_head.py 的 forward 返回 + torusfold.py 的 result）：
    coords:          (B, L, 3)  Cartesian, Å
    sequence:        str        长度 L，字母 ACGU
    confidence:      (B, L)     [0,100]，可选
    immune_fingerprints: dict   来自 ImmuneFingerprintHeads.forward
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# RNA 标准残基名（PDB 三字母码）
RES_NAME = {"A": "A", "U": "U", "G": "G", "C": "C"}
# BSJ 标记残基名（前端高亮用）
BSJ_RES_NAME = "BSJ"


def _to_numpy(x: Any) -> np.ndarray:
    """torch.Tensor / list / np.ndarray → np.ndarray。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _take_batch0(x: Any) -> np.ndarray:
    """取 batch=0，去 batch 维。输入 (B, L, ...) → (L, ...)。"""
    arr = _to_numpy(x)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _validate(seq: str, coords: np.ndarray) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 期望 (L, 3)，实际 {coords.shape}"
        )
    if len(seq) != coords.shape[0]:
        raise ValueError(
            f"sequence 长度 {len(seq)} != coords 残基数 {coords.shape[0]}"
        )
    bad = [c for c in seq if c not in RES_NAME]
    if bad:
        raise ValueError(f"sequence 含非法字母 {set(bad)}，只允许 ACGU")


def _atom_record(
    serial: int,
    res_seq: int,
    res_name: str,
    x: float,
    y: float,
    z: float,
    b_factor: float = 0.0,
) -> str:
    """
    PDB ATOM 记录（固定列格式，列号从 1 起）：
        1-6   "ATOM  "
        7-11  serial        右对齐
        13-16 atom name     左对齐（P 是单字符，第 14 列起）
        17    altLoc        空格
        18-20 resName       右对齐
        22    chainID       A
        23-26 resSeq        右对齐
        27    iCode         空格
        31-38 x             %8.3f
        39-46 y             %8.3f
        47-54 z             %8.3f
        55-60 occupancy     1.00
        61-66 tempFactor    %6.2f（这里放 confidence 当 B-factor）
        73-76 element       " P"
    """
    return (
        f"ATOM  "
        f"{serial:>5d} "
        f" P  "           # atom name（注意 PDB 对齐规则：单字符名从 14 列）
        f" "               # altLoc
        f"{res_name:>3s} "
        f"A"               # chainID
        f"{res_seq:>4d}"
        f"    "            # iCode + 3 spaces
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}"
        f"{b_factor:6.2f}"
        f"          "
        f" P"
    )


def coords_to_pdb(
    coords: Any,
    sequence: str,
    confidence: Optional[Any] = None,
    circular: bool = True,
    bsj_residue_names: bool = False,
) -> str:
    """
    A1 档主函数：coords + sequence → PDB 字符串。

    Args:
        coords: (B, L, 3) 或 (L, 3)，Å
        sequence: 长度 L 的 ACGU 字符串
        confidence: (B, L) 或 (L,)，可选；写入 B-factor 列（Mol* 可按它上色）
        circular: True=显式 CONECT 闭合最后一个残基回第一个（circRNA 闭环）
        bsj_residue_names: True=首尾残基 resName 改为 BSJ（前端高亮 back-splice junction）。
            默认 False —— Mol* 5.x 不认识 BSJ 这个非标准残基名，建 representation
            时会报 "Cannot read properties of undefined (reading 'data')"。
            BSJ 高亮改由前端按残基序号 (1 和 L) 单独着色，不再靠 resName。
    """
    coords_arr = _take_batch0(coords)
    _validate(sequence, coords_arr)

    conf_arr = None
    if confidence is not None:
        conf_arr = _take_batch0(confidence)
        if conf_arr.ndim == 0 or conf_arr.shape[0] != coords_arr.shape[0]:
            conf_arr = None  # 形状对不上就别写 B-factor

    L = len(sequence)
    lines: List[str] = []
    lines.append(
        f"REMARK   1 TorusFold circRNA structure (coarse-grained, P-only)"
    )
    lines.append(f"REMARK   2 length={L} circular={circular}")
    lines.append(f"HEADER    circRNA 3D STRUCTURE")

    atom_serial = 0
    res_indices: List[int] = []  # 记录每个残基的 P 原子 serial，给 CONECT 用

    for i in range(L):
        atom_serial += 1
        x, y, z = coords_arr[i]
        base = sequence[i]

        if bsj_residue_names and (i == 0 or i == L - 1):
            res_name = BSJ_RES_NAME  # 首尾标 BSJ
        else:
            res_name = RES_NAME[base]

        b_factor = float(conf_arr[i]) if conf_arr is not None else 0.0
        lines.append(
            _atom_record(atom_serial, i + 1, res_name, float(x), float(y), float(z), b_factor)
        )
        res_indices.append(atom_serial)

    # 闭环：显式 CONECT 最后一个 P → 第一个 P
    # Mol* 读 CONECT 会在两残基间画 bond，视觉上闭合 circRNA 环
    if circular and L >= 2:
        last_serial = res_indices[-1]
        first_serial = res_indices[0]
        lines.append(f"CONECT{last_serial:>5d}{first_serial:>5d}")

    # 相邻 backbone bond（可选，让 Mol* 显式连线，不依赖自动推断）
    # 只在 circular=False 或想要完整骨架时加；circRNA 时自动推断已够
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _clamp01(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(v)))


def fingerprints_to_json(
    coords: Any,
    sequence: str,
    immune_fingerprints: Optional[Dict[str, Any]] = None,
    confidence: Optional[Any] = None,
    per_residue_keys: Optional[Sequence[str]] = None,
    scalar_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    把免疫指纹 + confidence 整理成前端 coloring 用的 JSON 结构。

    返回 dict（序列化前），结构：
        {
          "sequence": "ACGU...",
          "length": L,
          "per_residue": {              # 每残基，长度 L 的数组
            "confidence": [...],
            "pkr_sasa": [...],
            "m6a_write_prob": [...],
            "tlr7_gu_density": [...],
            ...
          },
          "scalar": {                    # 整分子标量
            "nlrp3_persistence_length": ...,
            "sponge_score": ...,
            ...
          },
          "coloring_schemes": [         # 前端下拉框可选项
            {"key": "confidence", "label": "Confidence (pLDDT)", "type": "per_residue"},
            {"key": "pkr_sasa", "label": "PKR / SASA", "type": "per_residue"},
            ...
          ]
        }

    Args:
        per_residue_keys: 显式指定哪些 key 是 per-residue（默认按 shape 自动判断）
        scalar_keys: 显式指定哪些 key 是整分子标量
    """
    coords_arr = _take_batch0(coords)
    _validate(sequence, coords_arr)
    L = len(sequence)

    immune = immune_fingerprints or {}

    # 默认 per-residue / scalar 分类（基于 ImmuneFingerprintHeads.forward 的输出契约）
    default_per_res = {
        "pkr_stem_logit", "pkr_sasa",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density", "rigi_per_pos",
    }
    default_scalar = {
        "nlrp3_persistence_length", "sponge_score", "rigi_score",
    }

    per_res_keys = set(per_residue_keys) if per_residue_keys else default_per_res
    scal_keys = set(scalar_keys) if scalar_keys else default_scalar

    per_residue: Dict[str, List[float]] = {}

    # confidence 永远是 per-residue。原始值是 0~100（pLDDT 风格），
    # 归一到 [0,1] 跟其它 per-residue 指纹统一区间。前端会再 normalize 一次，
    # 但 JSON 里值域一致，将来读 JSON 不会困惑。
    if confidence is not None:
        conf = _take_batch0(confidence)
        if conf.ndim >= 1 and conf.shape[0] == L:
            per_residue["confidence"] = [_clamp01(float(v) / 100.0) for v in conf]
    # 没给 confidence 就合成一个中位值，前端总有这档可切
    if "confidence" not in per_residue:
        per_residue["confidence"] = [0.5] * L

    # 免疫指纹 per-residue
    # confidence 来自 build_synthetic_data / TorusFold 是 0~100（pLDDT 风格），
    # 单独按 /100 归一到 [0,1]；其它 per-residue 指纹本就是 [0,1] 概率/暴露度，
    # 直接 clamp01。原代码用嵌套三元，confidence 落在 0.7~0.95 区间梯度拉不开。
    for k in per_res_keys:
        if k in immune:
            arr = _take_batch0(immune[k])
            if arr.ndim >= 1 and arr.shape[0] == L:
                if k == "confidence":
                    per_residue[k] = [_clamp01(float(v) / 100.0) for v in arr]
                else:
                    per_residue[k] = [_clamp01(float(v)) for v in arr]
            # shape 对不上就跳过，不硬塞

    scalar: Dict[str, float] = {}
    for k in scal_keys:
        if k in immune:
            arr = _take_batch0(immune[k])
            try:
                scalar[k] = float(arr.item() if arr.size == 1 else arr.flat[0])
            except (ValueError, IndexError):
                pass

    # 构造前端下拉选项（只列实际存在的数据）
    schemes: List[Dict[str, str]] = []
    label_map = {
        "confidence": "Confidence (pLDDT)",
        "pkr_sasa": "PKR / SASA exposure",
        "pkr_stem_logit": "PKR stem",
        "m6a_write_prob": "m6A write probability",
        "drach_is_drach": "DRACH motif",
        "drach_in_loop": "in-loop",
        "tlr7_gu_density": "TLR7 GU density",
        "rigi_per_pos": "RIG-I (neg ctrl)",
        "nlrp3_persistence_length": "NLRP3 persistence length",
        "sponge_score": "miRNA sponge",
        "rigi_score": "RIG-I score (neg ctrl)",
    }
    for k in per_residue:
        schemes.append({
            "key": k,
            "label": label_map.get(k, k),
            "type": "per_residue",
        })
    for k in scalar:
        schemes.append({
            "key": k,
            "label": label_map.get(k, k) + " (scalar)",
            "type": "scalar",
        })

    return {
        "sequence": sequence,
        "length": L,
        "per_residue": per_residue,
        "scalar": scalar,
        "coloring_schemes": schemes,
    }


def export_circrna_structure(
    coords: Any,
    sequence: str,
    immune_fingerprints: Optional[Dict[str, Any]] = None,
    confidence: Optional[Any] = None,
    circular: bool = True,
) -> Dict[str, str]:
    """
    一站式：输入 TorusFold 输出，返回 {pdb, fingerprint_json}。
    供 html_renderer 直接注入 HTML 模板。

    Returns:
        {
          "pdb": "ATOM ...",
          "fingerprint_json": "{...}",   # 已 json.dumps 的字符串
        }
    """
    pdb = coords_to_pdb(coords, sequence, confidence=confidence, circular=circular)
    fp_dict = fingerprints_to_json(
        coords, sequence, immune_fingerprints, confidence=confidence
    )
    return {
        "pdb": pdb,
        "fingerprint_json": json.dumps(fp_dict, ensure_ascii=False),
    }
