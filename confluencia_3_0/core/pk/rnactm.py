"""RNACTM 六室 PK 模型 — circRNA 专用

内化自 confluencia-2.0-drug/core/ctm.py (lines 132-491)。

六室模型: Inj(注射) → LNP(递送复合体) → Endo(内吞体) → Cyto(胞质RNA) → Trans(翻译蛋白) → Clear(累积清除)

文献参考:
- LNP 递送速率: Hassett et al. (2019) Mol Ther 27:1885-1897
- circRNA 稳定性/半衰期: Wesselhoeft et al. (2018) Nat Commun 9:2629
- 核苷酸修饰效果: Chen et al. (2019) Nature 586:651-655; Liu et al. (2023) Nat Commun 14:2548
- 内吞体逃逸效率: Gilleron et al. (2013) Nat Biotechnol 31:638-646
- 组织分布 (LNP): Paunovska et al. (2018) ACS Nano 12:8307-8320
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class RNACTMParams:
    """circRNA 六室药代动力学模型参数。

    室: Inj(jection) → LNP → Endo(some) → Cyto(plasmic RNA) → Trans(lated protein) → Clear
    """
    k_uptake: float        # Inj → LNP 摄取速率 (1/h)
    k_release: float       # LNP → 内吞体释放速率 (1/h)
    k_escape: float        # 内吞体逃逸效率 (1/h)
    k_translate: float     # 翻译起始速率 (1/h)
    k_degrade: float       # RNA 降解速率 (1/h)
    k_protein_half: float  # 产物蛋白半衰期 (h)
    k_immune_clear: float  # 免疫介导清除速率 (1/h)

    # 晚期加速清除参数
    k_protein_late_delay: float = 48.0     # 加速开始延迟 (h)
    k_protein_late_width: float = 12.0     # Sigmoid 转换宽度 (h)
    k_protein_late_factor: float = 3.0     # 最大加速倍数

    # 组织分布系数 (分数, 总和 ≈ 1)
    f_liver: float = 0.80
    f_spleen: float = 0.10
    f_muscle: float = 0.03
    f_other: float = 0.07


def infer_rna_ctm_params(
    modification: str = "none",
    delivery_vector: str = "LNP_standard",
    route: str = "IV",
    ires_score: float = 0.5,
    gc_content: float = 0.5,
    struct_stability: float = 0.5,
    innate_immune_score: float = 0.0,
) -> RNACTMParams:
    """从分子特征和递送配置推断 circRNA CTM 参数。

    使用文献推导的先验值，根据序列属性和递送系统特征调整。
    """
    mod = str(modification).lower().strip()
    vec = str(delivery_vector).strip()

    # 摄取速率: Inj → LNP，取决于给药途径
    base_uptake = {"IV": 0.80, "SC": 0.15, "IM": 0.20, "ID": 0.10}
    k_uptake = base_uptake.get(route.upper(), 0.30)

    # 释放速率: LNP → 内吞体，取决于递送系统
    base_release = {"LNP_standard": 0.12, "LNP_liver": 0.15, "LNP_spleen": 0.10,
                    "AAV": 0.005, "naked": 0.80}
    k_release = base_release.get(vec, 0.12)

    # 内吞体逃逸: 取决于递送系统和结构稳定性
    base_escape = {"LNP_standard": 0.025, "LNP_liver": 0.03, "LNP_spleen": 0.025,
                   "AAV": 0.95, "naked": 0.01}
    k_escape = base_escape.get(vec, 0.02)
    k_escape *= (0.8 + 0.4 * float(np.clip(struct_stability, 0.0, 1.0)))

    # 翻译速率: 取决于 IRES 强度
    k_translate = float(np.clip(0.02 + 0.30 * ires_score, 0.01, 0.50))

    # RNA 降解: 取决于修饰和 GC 含量
    mod_half_life_map = {"none": 1.0, "m6a": 1.8, "Ψ": 2.5, "ψ": 2.5,
                         "5mc": 2.0, "ms2m6a": 3.0}
    stability_factor = mod_half_life_map.get(mod, mod_half_life_map["none"])
    base_degrade = 0.12  # 未修饰 RNA 半衰期 ~6h (Wesselhoeft 2018) → k ≈ ln2/6
    k_degrade = base_degrade / stability_factor
    k_degrade *= (1.0 - 0.15 * float(np.clip(gc_content, 0.0, 1.0)))

    # 蛋白半衰期
    k_protein_half = 16.0

    # 免疫介导清除
    k_immune_clear = float(np.clip(0.01 + 0.15 * innate_immune_score, 0.005, 0.30))

    # 组织分布: 取决于递送系统
    del_params = {
        "LNP_standard": (0.80, 0.10, 0.03, 0.07),
        "LNP_liver":    (0.90, 0.05, 0.01, 0.04),
        "LNP_spleen":   (0.35, 0.50, 0.02, 0.13),
        "AAV":          (0.60, 0.15, 0.10, 0.15),
        "naked":        (0.20, 0.10, 0.05, 0.65),
    }
    f_liver, f_spleen, f_muscle, f_other = del_params.get(vec, (0.80, 0.10, 0.03, 0.07))

    return RNACTMParams(
        k_uptake=k_uptake,
        k_release=k_release,
        k_escape=k_escape,
        k_translate=k_translate,
        k_degrade=k_degrade,
        k_protein_half=k_protein_half,
        k_immune_clear=k_immune_clear,
        f_liver=f_liver,
        f_spleen=f_spleen,
        f_muscle=f_muscle,
        f_other=f_other,
    )


def simulate_rna_ctm(
    dose: float,
    freq: float,
    params: RNACTMParams,
    horizon: int = 168,
    dt: float = 1.0,
) -> pd.DataFrame:
    """模拟 circRNA 药代动力学 (六室模型)。

    使用 scipy.integrate.solve_ivp (RK45) 自适应步长求解。
    """
    horizon = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    pulse_every = max(int(round(24.0 / freq)), 1)
    k_protein_degrade_base = float(np.log(2.0) / max(params.k_protein_half, 1.0))

    dose_times = [float(t) for t in range(0, horizon, pulse_every)]

    def ode_rhs(t, y):
        Inj, LNP, Endo, Cyto, Trans, Clear = y

        # 晚期加速蛋白清除
        sigmoid_arg = (t - params.k_protein_late_delay) / max(params.k_protein_late_width, 1.0)
        acceleration = 1.0 + params.k_protein_late_factor / (1.0 + np.exp(-sigmoid_arg))
        k_protein_degrade = k_protein_degrade_base * acceleration

        dInj = -params.k_uptake * Inj
        dLNP = params.k_uptake * Inj - params.k_release * LNP
        dEndo = params.k_release * LNP - params.k_escape * Endo

        # 翻译是 Cyto 消除的一部分，不是额外损失路径
        translation_fraction = min(params.k_translate / max(params.k_degrade, 0.001), 0.8)
        k_total_out = params.k_degrade
        k_translation_flux = translation_fraction * k_total_out
        k_degradation_flux = (1.0 - translation_fraction) * k_total_out

        dCyto = params.k_escape * Endo - k_total_out * Cyto
        dTrans = k_translation_flux * Cyto - k_protein_degrade * Trans
        dClear = k_degradation_flux * Cyto + k_protein_degrade * Trans

        return [dInj, dLNP, dEndo, dCyto, dTrans, dClear]

    t_grid = np.arange(0, horizon + 1, dt, dtype=np.float64)
    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    segments = []
    current_y = np.array(y0, dtype=np.float64)

    all_events = sorted(set(dose_times + [0.0, float(horizon)]))
    grid_points = set(t_grid.tolist())
    event_points = set(all_events)
    boundaries = sorted(grid_points | event_points)

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]
        if t_end <= t_start:
            continue

        if t_start in event_points and any(abs(t_start - dt_ev) < 0.01 for dt_ev in dose_times):
            current_y[0] += dose

        t_eval = np.array([t for t in t_grid if t_start <= t <= t_end and t >= t_start],
                          dtype=np.float64)
        if t_eval.size == 0 or (t_eval.size == 1 and t_eval[0] == t_start):
            t_eval = np.array([t_start, t_end], dtype=np.float64)

        sol = solve_ivp(
            fun=ode_rhs,
            t_span=(t_start, t_end),
            y0=current_y,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        if sol.success and sol.y.shape[1] > 0:
            segments.append(sol)
            current_y = sol.y[:, -1].copy()
            current_y = np.maximum(current_y, 0.0)
        else:
            step_dt = min(dt, t_end - t_start)
            dy = np.array(ode_rhs(t_start, current_y))
            current_y = np.maximum(current_y + step_dt * dy, 0.0)
            single_t = np.array([t_end])
            single_y = current_y.reshape(6, 1)
            segments.append(type(sol)(t=single_t, y=single_y, success=True))

    rows: List[Dict[str, float]] = []
    for seg in segments:
        for j in range(seg.t.size):
            Inj_val = float(max(seg.y[0, j], 0.0))
            LNP_val = float(max(seg.y[1, j], 0.0))
            Endo_val = float(max(seg.y[2, j], 0.0))
            Cyto_val = float(max(seg.y[3, j], 0.0))
            Trans_val = float(max(seg.y[4, j], 0.0))
            Clear_val = float(max(seg.y[5, j], 0.0))

            circulating_rna = LNP_val + Endo_val + Cyto_val

            rows.append({
                "time_h": float(seg.t[j]),
                "rna_injected": Inj_val,
                "rna_lnp": LNP_val,
                "rna_endosomal": Endo_val,
                "rna_cytoplasmic": Cyto_val,
                "protein_translated": Trans_val,
                "cumulative_clearance": Clear_val,
                "tissue_liver": circulating_rna * params.f_liver,
                "tissue_spleen": circulating_rna * params.f_spleen,
                "tissue_muscle": circulating_rna * params.f_muscle,
                "tissue_other": circulating_rna * params.f_other,
                "rna_circulating_total": circulating_rna,
                "efficacy_signal": Trans_val,
                "toxicity_signal": 0.20 * Clear_val + 0.10 * params.k_immune_clear * Cyto_val,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["time_h"], keep="last").sort_values("time_h").reset_index(drop=True)
    return df


def summarize_rna_ctm_curve(curve: pd.DataFrame) -> Dict[str, float]:
    """总结 circRNA CTM 模拟结果。"""
    if curve.empty:
        return {
            "rna_ctm_auc_efficacy": 0.0,
            "rna_ctm_peak_protein": 0.0,
            "rna_ctm_peak_cytoplasmic_rna": 0.0,
            "rna_ctm_protein_expression_window_h": 0.0,
            "rna_ctm_protein_persistence_window_h": 0.0,
            "rna_ctm_rna_half_life_h": 0.0,
            "rna_ctm_bioavailability_frac": 0.0,
            "rna_ctm_peak_toxicity": 0.0,
        }

    protein = curve["protein_translated"].to_numpy(dtype=np.float64)
    rna_cyto = curve["rna_cytoplasmic"].to_numpy(dtype=np.float64)
    rna_circ = curve["rna_circulating_total"].to_numpy(dtype=np.float64)
    t = curve["time_h"].to_numpy(dtype=np.float64)

    trap = getattr(np, "trapezoid", None)
    _trapz = trap if callable(trap) else np.trapz

    auc_eff = float(_trapz(protein, t)) if t.size > 1 else 0.0
    peak_protein = float(np.max(protein)) if protein.size > 0 else 0.0
    peak_rna_cyto = float(np.max(rna_cyto)) if rna_cyto.size > 0 else 0.0
    peak_tox = float(curve["toxicity_signal"].max()) if curve["toxicity_signal"].size > 0 else 0.0

    # 蛋白表达窗口: 峰值 50% 以上的时间
    threshold = 0.5 * peak_protein if peak_protein > 0 else 0.0
    above = protein >= threshold
    window = 0.0
    if np.any(above):
        indices = np.where(above)[0]
        window = float(t[indices[-1]] - t[indices[0]]) if len(indices) > 1 else 1.0

    # 蛋白持久窗口: 峰值 10% 以上的时间 (更宽松的阈值，反映持续低水平表达)
    persistence_threshold = 0.1 * peak_protein if peak_protein > 0 else 0.0
    above_persist = protein >= persistence_threshold
    persistence_window = 0.0
    if np.any(above_persist):
        p_indices = np.where(above_persist)[0]
        persistence_window = float(t[p_indices[-1]] - t[p_indices[0]]) if len(p_indices) > 1 else 1.0

    # RNA 半衰期估计
    rna_half = 0.0
    if rna_circ.size > 4:
        pos = rna_circ > 1e-9
        if np.sum(pos) > 4:
            start = int(np.floor(0.7 * t.size))
            t_tail = t[start:][rna_circ[start:] > 1e-9]
            c_tail = rna_circ[start:][rna_circ[start:] > 1e-9]
            if t_tail.size > 3:
                y_log = np.log(np.clip(c_tail, 1e-12, None))
                slope, _ = np.polyfit(t_tail, y_log, 1)
                if slope < 0:
                    rna_half = float(np.log(2.0) / (-slope))

    # 生物利用度
    total_dose_injected = float(curve["rna_injected"].iloc[0]) if len(curve) > 0 else 0.0
    total_clearance = float(curve["cumulative_clearance"].iloc[-1]) if len(curve) > 0 else 0.0
    bioavail = total_clearance / max(total_dose_injected, 1e-6)
    bioavail = float(np.clip(bioavail, 0.0, 1.0))

    return {
        "rna_ctm_auc_efficacy": auc_eff,
        "rna_ctm_peak_protein": peak_protein,
        "rna_ctm_peak_cytoplasmic_rna": peak_rna_cyto,
        "rna_ctm_protein_expression_window_h": window,
        "rna_ctm_protein_persistence_window_h": persistence_window,
        "rna_ctm_rna_half_life_h": rna_half,
        "rna_ctm_bioavailability_frac": bioavail,
        "rna_ctm_peak_toxicity": peak_tox,
    }
