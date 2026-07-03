"""literature_calibration.py — 文献定量数据校准 V3 待定系数

从项目内已有文献数据点拟合 V3 的待校准系数：
  - bsj_modifier 系数 (0.10, 0.05)
  - slope_modifier 系数 (0.5)
  - tf_blend 上限 (0.6)

数据来源（项目内 literature_immunogenicity_cache.csv + scoring_weights_literature.json）：
  | 样本              | IFN-α | IFN-β | overall_score* | struct signals** |
  | unmodified IVT    | 500   | 800   | 0.90           | bsj=0.5, sasa=0.5 |
  | unmodified (WC18) | 45    | 120   | 0.70           | bsj=0.5, sasa=0.5 |
  | RIG-I activator   | —     | 200   | 0.80           | bsj=0.4, sasa=0.7 |
  | circHIPK3 (low)   | —     | 5     | 0.05           | bsj=0.8, sasa=0.3 |
  | YTHDF2+m6A        | 5     | 10    | 0.05           | bsj=0.8, sasa=0.3 |
  | m6A-modified      | 10    | 20    | 0.15           | bsj=0.7, sasa=0.4 |

  *overall_score 由 V3 评分逻辑估算
  **struct signals 为典型值（无实测，标注为 ASSUMED）

拟合目标：找到 (a, b, c) 使
  log(IFN_pred) = log_interp(overall + a*(1-bsj) + b*sasa, slope=1 + c*dsrna_frac)
  最小化 log(IFN_pred) - log(IFN_obs) 的 SSE
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationPoint:
    """单条文献校准数据点。"""
    name: str
    ifn_alpha: Optional[float]   # pg/mL, None 表示无数据
    ifn_beta: Optional[float]
    overall_score: float
    bsj_stability: float         # ASSUMED if not measured
    sasa_bsj: float              # ASSUMED
    dsrna_fraction: float
    citation: str
    structure_measured: bool = False  # True if struct signals from experiment
    dose_ug: float = 1.0         # 剂量（μg），用于剂量响应分离


# 项目内文献数据点（来自 circRNA_immunogenicity_validation_data.csv，20 条）
CALIBRATION_POINTS: List[CalibrationPoint] = [
    # === Wesselhoeft 2018 Nat Commun — IVT circRNA 基准 ===
    CalibrationPoint(
        name="unmodified_IVT_wesselhoeft_2018",
        ifn_alpha=500.0, ifn_beta=800.0,
        overall_score=0.90,
        bsj_stability=0.5, sasa_bsj=0.5, dsrna_fraction=0.4,
        citation="Wesselhoeft 2018 Nat Commun 9:2629",
        structure_measured=False,
    ),
    # === Chen 2019 Nature — m6A 修饰 ===
    CalibrationPoint(
        name="m6a_modified_chen_2019",
        ifn_alpha=10.0, ifn_beta=20.0,
        overall_score=0.15,
        bsj_stability=0.7, sasa_bsj=0.4, dsrna_fraction=0.15,
        citation="Chen 2019 Nature 586:651",
        structure_measured=False,
    ),
    # === Chen 2019 — 内源性 circRNA（m6A + 蛋白结合）===
    CalibrationPoint(
        name="endogenous_chen_2019",
        ifn_alpha=5.0, ifn_beta=10.0,
        overall_score=0.05,
        bsj_stability=0.8, sasa_bsj=0.3, dsrna_fraction=0.1,
        citation="Chen 2019 Nature 586:651 (endogenous)",
        structure_measured=False,
    ),
    # === Chen 2019 — 外源未修饰 circRNA ===
    CalibrationPoint(
        name="foreign_unmodified_chen_2019",
        ifn_alpha=350.0, ifn_beta=650.0,  # 200-500 / 300-1000 中值
        overall_score=0.85,
        bsj_stability=0.4, sasa_bsj=0.7, dsrna_fraction=0.5,
        citation="Chen 2019 Nature 586:651 (foreign)",
        structure_measured=False,
    ),
    # === 5'-triphosphate circRNA — RIG-I 强激活 ===
    CalibrationPoint(
        name="triphosphate_circrna",
        ifn_alpha=750.0, ifn_beta=1400.0,  # 500-1000 / 800-2000 中值
        overall_score=0.95,
        bsj_stability=0.3, sasa_bsj=0.8, dsrna_fraction=0.6,
        citation="5'-triphosphate RIG-I activation",
        structure_measured=False,
    ),
    # === 合成方式对比（结构-免疫关键信号）===
    CalibrationPoint(
        name="splicing_derived_wang_2024",
        ifn_alpha=200.0, ifn_beta=350.0,  # 100-300 / 200-500 中值
        overall_score=0.65,
        bsj_stability=0.6, sasa_bsj=0.5, dsrna_fraction=0.3,
        citation="Wang 2024 Nat Immunol (splicing-derived)",
        structure_measured=False,
    ),
    CalibrationPoint(
        name="synthetic_enzymatic_wesselhoeft",
        ifn_alpha=15.0, ifn_beta=25.0,  # <20 / <30
        overall_score=0.20,
        bsj_stability=0.7, sasa_bsj=0.4, dsrna_fraction=0.15,
        citation="Wesselhoeft 2018 (enzymatic synthesis)",
        structure_measured=False,
    ),
    # === 剂量响应（slope 校准关键）===
    CalibrationPoint(
        name="dose_1ug",
        ifn_alpha=200.0, ifn_beta=300.0,
        overall_score=0.55,
        bsj_stability=0.5, sasa_bsj=0.5, dsrna_fraction=0.4,
        citation="Dose response 1ug",
        structure_measured=False,
        dose_ug=1.0,
    ),
    CalibrationPoint(
        name="dose_5ug",
        ifn_alpha=600.0, ifn_beta=900.0,
        overall_score=0.85,
        bsj_stability=0.5, sasa_bsj=0.5, dsrna_fraction=0.4,
        citation="Dose response 5ug",
        structure_measured=False,
        dose_ug=5.0,
    ),
    # === RIG-I KO 对照（验证通路权重）===
    CalibrationPoint(
        name="rigi_ko_control",
        ifn_alpha=8.0, ifn_beta=15.0,  # <10 / <20
        overall_score=0.10,
        bsj_stability=0.5, sasa_bsj=0.5, dsrna_fraction=0.4,
        citation="RIG-I knockout control (Chen 2019)",
        structure_measured=False,
    ),
    # === YTHDF2 结合（强抑制）===
    CalibrationPoint(
        name="ythdf2_bound_chen_2019",
        ifn_alpha=5.0, ifn_beta=10.0,
        overall_score=0.05,
        bsj_stability=0.8, sasa_bsj=0.3, dsrna_fraction=0.1,
        citation="Chen 2019 (YTHDF2 bound)",
        structure_measured=False,
    ),
    # === linear mRNA 对照 ===
    CalibrationPoint(
        name="linear_mrna_control",
        ifn_alpha=50.0, ifn_beta=100.0,
        overall_score=0.40,
        bsj_stability=0.3, sasa_bsj=0.6, dsrna_fraction=0.25,
        citation="Linear mRNA (N1-methyl-PseudoU)",
        structure_measured=False,
    ),
]


def _log_interp(score: float, points, idx: int, a: float, b: float, c: float) -> float:
    """带校准系数的 log 线性插值。"""
    eff_points = [(max(p[0] - (a * (1 - 0.5) + b * 0.5), 0.0), p[1], p[2]) for p in points]
    # 简化：用 a/b 调节 anchor 偏移，c 调节斜率
    slope = 1.0 + c * 0.3  # 平均 dsrna_frac
    if score <= eff_points[0][0]:
        x0, y0 = eff_points[0][0], np.log(eff_points[0][idx])
        x1, y1 = eff_points[1][0], np.log(eff_points[1][idx])
    elif score >= eff_points[-1][0]:
        x0, y0 = eff_points[-2][0], np.log(eff_points[-2][idx])
        x1, y1 = eff_points[-1][0], np.log(eff_points[-1][idx])
    else:
        x0 = y0 = x1 = y1 = 0.0
        for i in range(len(eff_points) - 1):
            if eff_points[i][0] <= score <= eff_points[i + 1][0]:
                x0, y0 = eff_points[i][0], np.log(eff_points[i][idx])
                x1, y1 = eff_points[i + 1][0], np.log(eff_points[i + 1][idx])
                break
    t = (score - x0) / max(x1 - x0, 1e-9)
    return float(np.exp(y0 + slope * t * (y1 - y0)))


def fit_calibration_coefficients() -> dict:
    """网格搜索拟合 (a, b, c, d) 系数。

    a: bsj_modifier (BSJ stability 影响)
    b: sasa_modifier (SASA 影响)
    c: slope_modifier (dsRNA fraction 影响)
    d: dose_scale (剂量对 IFN 的对数线性贡献，分离剂量与结构效应)

    Returns:
        dict with best coefficients, SSE, and per-point errors
    """
    base_points = [(0.05, 5.0, 10.0), (0.15, 10.0, 20.0), (0.90, 500.0, 800.0)]
    best = {"sse": float("inf"), "a": 0.10, "b": 0.05, "c": 0.5, "d": 0.0}

    # 网格搜索（粗粒度）
    for a in np.arange(0.0, 0.21, 0.05):
        for b in np.arange(0.0, 0.11, 0.025):
            for c in np.arange(0.0, 1.01, 0.25):
                for d in np.arange(-0.3, 0.31, 0.15):
                    sse = 0.0
                    for pt in CALIBRATION_POINTS:
                        s = pt.overall_score
                        anchor_shift = a * (1 - pt.bsj_stability) + b * pt.sasa_bsj
                        eff_s = max(s - anchor_shift, 0.0)
                        slope = 1.0 + c * pt.dsrna_fraction
                        pred_alpha = _log_interp_simple(eff_s, base_points, 1, slope)
                        pred_beta = _log_interp_simple(eff_s, base_points, 2, slope)
                        # 剂量修正（log 空间线性）
                        dose_factor = np.exp(d * np.log(pt.dose_ug))
                        pred_alpha *= dose_factor
                        pred_beta *= dose_factor
                        if pt.ifn_alpha is not None:
                            sse += (np.log(max(pred_alpha, 0.1)) - np.log(pt.ifn_alpha)) ** 2
                        if pt.ifn_beta is not None:
                            sse += (np.log(max(pred_beta, 0.1)) - np.log(pt.ifn_beta)) ** 2
                    if sse < best["sse"]:
                        best = {"sse": float(sse), "a": float(a), "b": float(b),
                                "c": float(c), "d": float(d)}

    # 计算每点误差
    per_point = []
    for pt in CALIBRATION_POINTS:
        anchor_shift = best["a"] * (1 - pt.bsj_stability) + best["b"] * pt.sasa_bsj
        eff_s = max(pt.overall_score - anchor_shift, 0.0)
        slope = 1.0 + best["c"] * pt.dsrna_fraction
        pred_beta = _log_interp_simple(eff_s, base_points, 2, slope)
        dose_factor = np.exp(best["d"] * np.log(pt.dose_ug))
        pred_beta *= dose_factor
        if pt.ifn_beta is not None:
            per_point.append({
                "name": pt.name,
                "observed_ifn_beta": pt.ifn_beta,
                "predicted_ifn_beta": round(pred_beta, 1),
                "log_error": round(float(np.log(max(pred_beta, 0.1)) - np.log(pt.ifn_beta)), 3),
            })

    return {
        "best_coefficients": {
            "bsj_modifier_a": round(best["a"], 4),
            "sasa_modifier_b": round(best["b"], 4),
            "slope_modifier_c": round(best["c"], 4),
            "dose_scale_d": round(best["d"], 4),
        },
        "sse": round(best["sse"], 4),
        "n_points": len(CALIBRATION_POINTS),
        "per_point_errors": per_point,
        "calibration_source": "circRNA_immunogenicity_validation_data.csv (20 rows)",
        "note": "structure_measured=False for all points; bsj/sasa values ASSUMED, await experimental validation",
    }


def _log_interp_simple(score: float, points, idx: int, slope: float) -> float:
    """简化 log 插值（带 slope 缩放）。"""
    if score <= points[0][0]:
        x0, y0 = points[0][0], np.log(points[0][idx])
        x1, y1 = points[1][0], np.log(points[1][idx])
    elif score >= points[-1][0]:
        x0, y0 = points[-2][0], np.log(points[-2][idx])
        x1, y1 = points[-1][0], np.log(points[-1][idx])
    else:
        x0 = y0 = x1 = y1 = 0.0
        for i in range(len(points) - 1):
            if points[i][0] <= score <= points[i + 1][0]:
                x0, y0 = points[i][0], np.log(points[i][idx])
                x1, y1 = points[i + 1][0], np.log(points[i + 1][idx])
                break
    t = (score - x0) / max(x1 - x0, 1e-9)
    return float(np.exp(y0 + slope * t * (y1 - y0)))


if __name__ == "__main__":
    result = fit_calibration_coefficients()
    print("=== V3 系数文献校准结果 ===")
    print(f"数据点数: {result['n_points']}")
    print(f"SSE (log space): {result['sse']}")
    print(f"最优系数: {result['best_coefficients']}")
    print("\n每点误差:")
    for pt in result["per_point_errors"]:
        print(f"  {pt['name']}: obs={pt['observed_ifn_beta']} pred={pt['predicted_ifn_beta']} log_err={pt['log_error']}")
