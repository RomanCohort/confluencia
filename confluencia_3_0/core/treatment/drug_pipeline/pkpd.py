"""PK/PD 模拟 (Pharmacokinetic/Pharmacodynamic Simulation)

三室模型 + Hill 方程 PD 链接。

PK模型 (三室):
  中央室: dC/dt = -k10*C - k12*C + k21*P
  外周室: dP/dt = k12*C - k21*P
  效应室: dE/dt = ke0*(C - E)

PD模型 (Hill方程):
  E = Emax * C^Hill / (EC50^Hill + C^Hill)
"""
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PKPDParams:
    """PK/PD 参数"""
    # PK参数
    k10: float = 0.1      # 中央室消除率 (1/h)
    k12: float = 0.5      # 中央→外周分布率 (1/h)
    k21: float = 0.3      # 外周→中央分布率 (1/h)
    ke0: float = 0.5      # 效应室消除率 (1/h)
    vd: float = 10.0      # 分布体积 (L)

    # PD参数
    ec50: float = 1.0     # 半数有效浓度 (ng/mL)
    emax: float = 0.8     # 最大效应
    hill: float = 1.0     # Hill系数

    # 给药
    dose_mg: float = 100.0  # 剂量 (mg)
    freq_h: float = 24.0    # 给药间隔 (h)
    bioavailability: float = 1.0


@dataclass
class PKPDState:
    """PK/PD 运行时状态"""
    central_conc: float = 0.0    # 中央室浓度 (ng/mL)
    peripheral_conc: float = 0.0  # 外周室浓度 (ng/mL)
    effect_conc: float = 0.0      # 效应室浓度 (ng/mL)
    time_h: float = 0.0           # 当前时间 (h)
    auc: float = 0.0              # AUC
    cmax: float = 0.0             # Cmax
    tmax: float = 0.0             # Tmax


def hill_equation(concentration: float, ec50: float, emax: float, hill: float = 1.0) -> float:
    """Hill 方程计算 PD 效应

    E = Emax * C^Hill / (EC50^Hill + C^Hill)
    """
    if concentration <= 0:
        return 0.0
    c_hill = concentration ** hill
    ec50_hill = ec50 ** hill
    return emax * c_hill / (ec50_hill + c_hill)


def simulate_pkpd_step(
    pkpd_state: PKPDState,
    params: PKPDParams,
    dt_h: float = 1.0,
    is_dose_time: bool = False,
) -> PKPDState:
    """一步 PK/PD 模拟 (Euler法)

    Args:
        pkpd_state: 当前PK/PD状态
        params: PK/PD参数
        dt_h: 时间步长 (h)
        is_dose_time: 是否在给药时间点

    Returns:
        更新后的PK/PD状态
    """
    C = pkpd_state.central_conc
    P = pkpd_state.peripheral_conc
    E = pkpd_state.effect_conc

    # 给药（瞬时输入到中央室）
    if is_dose_time:
        dose_conc = params.dose_mg * params.bioavailability / params.vd * 1000  # mg→ng
        C += dose_conc

    # PK ODE (三室模型)
    dC = (-params.k10 * C - params.k12 * C + params.k21 * P) * dt_h
    dP = (params.k12 * C - params.k21 * P) * dt_h
    dE = params.ke0 * (C - E) * dt_h

    new_C = max(0.0, C + dC)
    new_P = max(0.0, P + dP)
    new_E = max(0.0, E + dE)

    # AUC 累积
    new_auc = pkpd_state.auc + new_C * dt_h

    # Cmax/Tmax
    new_cmax = pkpd_state.cmax
    new_tmax = pkpd_state.tmax
    if new_C > new_cmax:
        new_cmax = new_C
        new_tmax = pkpd_state.time_h + dt_h

    return PKPDState(
        central_conc=new_C,
        peripheral_conc=new_P,
        effect_conc=new_E,
        time_h=pkpd_state.time_h + dt_h,
        auc=new_auc,
        cmax=new_cmax,
        tmax=new_tmax,
    )


def simulate_pkpd_curve(
    params: PKPDParams,
    horizon_h: float = 168.0,  # 7天
    dt_h: float = 1.0,
) -> Dict[str, List[float]]:
    """模拟完整 PK/PD 曲线

    Returns:
        {"time": [...], "concentration": [...], "effect": [...]}
    """
    state = PKPDState()
    times, concentrations, effects = [], [], []

    n_steps = int(horizon_h / dt_h)
    for i in range(n_steps):
        t = i * dt_h
        is_dose_time = (i % int(params.freq_h / dt_h)) == 0

        state = simulate_pkpd_step(state, params, dt_h, is_dose_time)
        effect = hill_equation(state.effect_conc, params.ec50, params.emax, params.hill)

        times.append(t)
        concentrations.append(state.central_conc)
        effects.append(effect)

    return {"time": times, "concentration": concentrations, "effect": effects}


def infer_pkpd_params_from_drug(drug_def) -> PKPDParams:
    """从药物定义推断PK/PD参数"""
    return PKPDParams(
        k10=math.log(2) / drug_def.half_life_h,
        k12=0.5,
        k21=0.3,
        ke0=math.log(2) / drug_def.half_life_h * 0.5,
        vd=10.0,
        ec50=drug_def.ec50,
        emax=drug_def.emax,
        hill=drug_def.hill_coeff,
        dose_mg=drug_def.dose_mg_m2 * 1.7,  # 假设BSA=1.7m²
        freq_h=drug_def.frequency_days * 24.0,
    )