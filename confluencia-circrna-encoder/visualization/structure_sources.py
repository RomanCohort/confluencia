"""
structure_sources.py — circRNA 3D 结构的数据源（统一入口）。

两条数据路径，studio (app.py) 和 demo 脚本都从这里取：
    1) build_synthetic_data: stem+loop 折叠合成结构（无需权重，演示用）
    2) run_torusfold_inference: 真实 TorusFold v2 forward → coords/immune
    3) get_structure_for_sequence: 统一入口，按 model='synthetic'/'torusfold' 分发

合成 vs 真实 必须透明：合成路径返回 source='synthetic'，绝不伪装成真实预测。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# 默认权重路径（本地训练产物；v1 残留，不兼容 v2）
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "torusfold_best.pt"


# ============================================================================
# 合成结构：stem + loop 折叠 circRNA（无需权重）
# ============================================================================

def _rna_stem_coords(
    n_res: int, start_pos: np.ndarray, start_dir: np.ndarray,
    helix_radius: float = 9.4, rise: float = 2.81, twist_deg: float = 32.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """A-form RNA 螺旋茎段：沿 start_dir 前进，P 原子绕轴画螺旋。"""
    axis = start_dir / (np.linalg.norm(start_dir) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(axis, u)

    coords = np.zeros((n_res, 3))
    for i in range(n_res):
        ang = np.deg2rad(twist_deg * i)
        coords[i] = (start_pos
                     + axis * rise * i
                     + u * helix_radius * np.cos(ang)
                     + v * helix_radius * np.sin(ang))
    end_dir = axis.copy()
    return coords, end_dir


def _rna_loop_coords(
    n_res: int, start_pos: np.ndarray, start_dir: np.ndarray,
    loop_radius: float = 8.0, turn_deg: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """发夹环段：垂直 start_dir 平面里走半圆弧，并把轴向转过 turn_deg。"""
    axis = start_dir / (np.linalg.norm(start_dir) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(axis, u)

    arc_span = np.pi
    coords = np.zeros((n_res, 3))
    for i in range(n_res):
        t = (i + 1) / n_res
        ang = arc_span * t
        coords[i] = (start_pos
                     + u * loop_radius * np.sin(ang)
                     + v * loop_radius * (1 - np.cos(ang)))

    turn_rad = np.deg2rad(turn_deg)
    new_axis = (axis * np.cos(turn_rad) + u * np.sin(turn_rad))
    return coords, new_axis


def _closing_loop_coords(
    n_res: int, start_pos: np.ndarray, target_pos: np.ndarray,
    cur_dir: np.ndarray, bond_length: float = 5.9,
) -> np.ndarray:
    """闭合 loop 段：平滑插值到 target_pos 附近，末位距 target ~bond_length（BSJ 闭合）。"""
    disp = target_pos - start_pos
    dist = np.linalg.norm(disp) + 1e-9
    main_dir = disp / dist

    ref = np.array([0.0, 0.0, 1.0]) if abs(main_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(main_dir, ref)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    arch = dist * 0.25

    coords = np.zeros((n_res, 3))
    for i in range(n_res):
        t = (i + 1) / n_res
        arc_offset = perp * arch * np.sin(np.pi * t)
        coords[i] = start_pos + main_dir * (dist * t) + arc_offset
    coords[-1] = target_pos - main_dir * bond_length
    return coords


def build_synthetic_data(
    L: int = 48, sequence: Optional[str] = None,
) -> Tuple[np.ndarray, str, Dict[str, Any], np.ndarray]:
    """
    合成折叠状 circRNA（演示用，不是真实预测）。

    结构 = 4 段 stem（A-form 螺旋）+ 4 段 loop（弧形回折）交替，首尾闭合。
    免疫指纹按结构位置赋值：stem 段 SASA 低/m6A 低；loop 段 SASA 高/GU 富集。
    切 coloring scheme 能看到 stem/loop 分布差异。

    Args:
        L: 目标残基数（实际会取整为 8 的倍数）
        sequence: 可选序列；不传则按 stem GC配对/loop GU-rich 自动生成

    Returns:
        coords (1, L, 3), sequence, immune_fingerprints dict, confidence (1, L)
    """
    if L < 12:
        L = 12

    n_segments = 4
    per_seg = max(3, L // (n_segments * 2))
    L = per_seg * n_segments * 2

    coords_list = []
    stem_mask = np.zeros(L, dtype=bool)

    cur_pos = np.array([0.0, 0.0, 0.0])
    cur_dir = np.array([1.0, 0.0, 0.0])

    idx = 0
    first_residue_pos = None
    for s in range(n_segments):
        stem_c, stem_end_dir = _rna_stem_coords(per_seg, cur_pos, cur_dir)
        if first_residue_pos is None:
            first_residue_pos = stem_c[0].copy()
        coords_list.append(stem_c)
        stem_mask[idx:idx + per_seg] = True
        idx += per_seg
        cur_pos = stem_c[-1]
        cur_dir = stem_end_dir

        if s < n_segments - 1:
            turn = 90.0 + 30.0 * np.sin(s)
            loop_c, loop_end_dir = _rna_loop_coords(
                per_seg, cur_pos, cur_dir, turn_deg=turn
            )
        else:
            loop_c = _closing_loop_coords(
                per_seg, cur_pos, first_residue_pos, cur_dir
            )
            loop_end_dir = (first_residue_pos - loop_c[-1])
            n = np.linalg.norm(loop_end_dir)
            loop_end_dir = loop_end_dir / (n + 1e-9)

        coords_list.append(loop_c)
        idx += per_seg
        cur_pos = loop_c[-1]
        cur_dir = loop_end_dir

    coords = np.concatenate(coords_list, axis=0)
    coords = coords - coords.mean(axis=0, keepdims=True)
    coords = coords[np.newaxis]  # (1, L, 3)

    # 序列：有传就用传的（截/补到 L）；没传就 stem GC/loop GU 自动生成
    if sequence is not None and len(sequence) >= L:
        seq = sequence[:L].upper().replace("T", "U")
    else:
        seq_chars = []
        for s in range(n_segments):
            stem_seq = "GCAUGC" * (per_seg // 6) + "GCAU"[:per_seg % 6]
            seq_chars.append(stem_seq)
            loop_seq = "GUGUUU" * (per_seg // 6) + "GUGU"[:per_seg % 6]
            seq_chars.append(loop_seq)
        seq = "".join(seq_chars)[:L]

    confidence = np.where(stem_mask, np.random.uniform(70, 95, L),
                          np.random.uniform(45, 75, L)).astype(np.float32)
    confidence = confidence[np.newaxis]

    loop_mask = ~stem_mask
    immune_fingerprints = {
        "pkr_sasa": np.where(stem_mask, np.random.uniform(0.1, 0.35, L),
                              np.random.uniform(0.6, 0.95, L))[np.newaxis],
        "pkr_stem_logit": np.where(stem_mask, np.random.uniform(1.0, 2.0, L),
                                    np.random.uniform(-2.0, 0.0, L))[np.newaxis],
        "drach_is_drach": np.where(loop_mask, np.random.uniform(0.3, 0.9, L),
                                    np.random.uniform(0.0, 0.1, L))[np.newaxis],
        "drach_in_loop": loop_mask.astype(float)[np.newaxis],
        "m6a_write_prob": np.where(loop_mask, np.random.uniform(0.1, 0.5, L),
                                    np.random.uniform(0.0, 0.05, L))[np.newaxis],
        "tlr7_gu_density": np.where(loop_mask, np.random.uniform(0.6, 0.95, L),
                                     np.random.uniform(0.2, 0.5, L))[np.newaxis],
        "rigi_per_pos": np.random.uniform(0.0, 0.15, (1, L)),
        "nlrp3_persistence_length": np.array([[5.8]]),
        "sponge_score": np.array([[0.42]]),
        "rigi_score": np.array([[0.12]]),
    }
    return coords, seq, immune_fingerprints, confidence


# ============================================================================
# 真实 TorusFold 推理
# ============================================================================

def run_torusfold_inference(
    sequence: str,
    weights: Path,
    device: str = "cpu",
) -> Tuple[np.ndarray, str, Dict[str, Any], np.ndarray]:
    """
    真实 TorusFold 推理：加载权重 → forward → 取 coords/confidence/immune。

    Raises:
        FileNotFoundError: 权重不存在
        RuntimeError: 权重不是 v2 格式 / forward 没产 coords
    """
    import torch
    from core.torusfold import TorusFold, TorusFoldConfig

    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    state = torch.load(str(weights), map_location=device, weights_only=False)
    if "config" in state:
        known_fields = {f.name for f in dataclasses.fields(TorusFoldConfig)}
        cfg_kwargs = {k: v for k, v in state["config"].items() if k in known_fields}
        dropped = set(state["config"].keys()) - known_fields
        if dropped:
            print(f"    [config] 丢弃历史遗留字段: {sorted(dropped)}")
        config = TorusFoldConfig(**cfg_kwargs)
    else:
        config = TorusFoldConfig()

    weights_state = state.get("model_state_dict", state)

    # v1 残留权重检测：缺 pairformer/structure_head 等 v2 必需 key 就拒绝
    v2_required_keys = {"backbone", "pairformer", "structure_head", "composite_head"}
    if not v2_required_keys.issubset(weights_state.keys()):
        missing = v2_required_keys - set(weights_state.keys())
        raise RuntimeError(
            f"权重不是 v2 格式（缺 {missing}）。"
            f"很可能是 TorusFold v1 残留（ESM backbone 版），"
            f"跟当前 v2（AF3-inspired）架构不兼容，无法 load。"
            f"请指向云端拉下来的 v2 权重，或重训。"
        )

    model = TorusFold(config).to(device)
    model.load(str(weights), device=device)
    model.eval()

    gene_expr = {
        "TROP2": 7.2, "NECTIN4": 5.1, "LIV-1": 3.5,
        "B7-H4": 6.0, "MKI67": 8.0, "MYC": 4.5,
    }
    gene_values = [gene_expr.get(g, 0.5) for g in config.gene_cols]
    gene_tensor = torch.tensor([gene_values], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model.forward(
            [sequence], gene_tensor, device=device, predict_structure=True
        )

    if "coords" not in outputs:
        raise RuntimeError(
            "TorusFold forward 没产出 coords —— structure_head 可能未启用 "
            f"(structure_mode={config.structure_mode})"
        )

    coords = outputs["coords"].cpu().numpy()
    confidence = outputs["confidence"].cpu().numpy()

    immune_fingerprints: Dict[str, Any] = {}
    if "immune_fingerprints" in outputs:
        for k, v in outputs["immune_fingerprints"].items():
            immune_fingerprints[k] = v.detach().cpu().numpy()

    print(f"    coords={coords.shape}, confidence={confidence.shape}, "
          f"immune_keys={len(immune_fingerprints)}, "
          f"closure_distance={outputs.get('closure_distance', 'n/a')}")

    return coords, sequence, immune_fingerprints, confidence


# ============================================================================
# 统一入口
# ============================================================================

def get_structure_for_sequence(
    sequence: str,
    model: str = "synthetic",
    weights: Optional[Path] = None,
    device: str = "cpu",
    L: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一入口：按 model 分发到合成或真实推理。

    Args:
        sequence: ACGU 字符串
        model: 'synthetic'（合成预览）或 'torusfold'（真实 v2 推理）
        weights: model='torusfold' 时的权重路径
        device: 推理设备
        L: 合成路径的目标残基数。默认 None=用 len(sequence)，
           这样长序列不会被 build_synthetic_data 的默认 L=48 截断。
           (torusfold 路径 L 由 forward 内部决定，忽略此参数)

    Returns:
        {
          "coords": (1, L, 3) np.ndarray,
          "sequence": str,
          "confidence": (1, L) np.ndarray,
          "immune_fingerprints": dict,
          "source": "synthetic" | "torusfold" | "synthetic-fallback",
          "fallback_reason": str | None,  # source=fallback 时有值
        }

    真实推理失败时自动 fallback 到合成，并在 source 字段透明标注，
    绝不把合成数据伪装成真实预测。
    """
    seq = sequence.upper().replace("T", "U")  # 容错 DNA→RNA
    # 合成路径目标长度：没指定就用序列本身长度，避免被截断
    synth_L = L if L is not None else len(seq)

    if model == "torusfold":
        if weights is None:
            weights = DEFAULT_WEIGHTS
        try:
            coords, seq, immune, conf = run_torusfold_inference(seq, Path(weights), device)
            return {
                "coords": coords, "sequence": seq,
                "confidence": conf, "immune_fingerprints": immune,
                "source": "torusfold", "fallback_reason": None,
            }
        except Exception as e:
            # 真实推理失败 → 透明 fallback 到合成
            reason = f"{type(e).__name__}: {e}"
            print(f"[get_structure] torusfold 失败，回退合成: {reason}")
            coords, seq, immune, conf = build_synthetic_data(L=synth_L, sequence=seq)
            return {
                "coords": coords, "sequence": seq,
                "confidence": conf, "immune_fingerprints": immune,
                "source": "synthetic-fallback", "fallback_reason": reason,
            }

    # 默认 synthetic
    coords, seq, immune, conf = build_synthetic_data(L=synth_L, sequence=seq)
    return {
        "coords": coords, "sequence": seq,
        "confidence": conf, "immune_fingerprints": immune,
        "source": "synthetic", "fallback_reason": None,
    }
