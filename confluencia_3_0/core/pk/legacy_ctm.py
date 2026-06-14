"""Legacy 四室 CTM 模型 — 通用药物 PK

内化自 confluencia-2.0-drug/core/ctm.py (lines 29-130)。

四室模型: A(吸收) → D(分布) → E(效应) → M(代谢)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CTMParams:
    ka: float       # 吸收速率
    kd: float       # 分布速率
    ke: float       # 效应消除速率
    km: float       # 代谢速率
    signal_gain: float  # 信号增益


def params_from_micro_scores(binding: float, immune: float, inflammation: float) -> CTMParams:
    """从微观预测分数映射 CTM 速率常数。"""
    b = float(np.clip(binding, 0.0, 1.0))
    i = float(np.clip(immune, 0.0, 1.0))
    inf = float(np.clip(inflammation, 0.0, 1.0))

    ka = 0.15 + 0.35 * b
    kd = 0.10 + 0.30 * i
    ke = 0.08 + 0.20 * (1.0 - inf)
    km = 0.06 + 0.30 * inf
    gain = 0.8 + 1.5 * (0.6 * b + 0.4 * i)
    return CTMParams(ka=ka, kd=kd, ke=ke, km=km, signal_gain=gain)


def simulate_ctm(
    dose: float,
    freq: float,
    params: CTMParams,
    horizon: int = 72,
    dt: float = 1.0,
) -> pd.DataFrame:
    """模拟四室 CTM (Euler 法)。"""
    steps = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    A = 0.0
    D = 0.0
    E = 0.0
    M = 0.0

    rows: List[Dict[str, float]] = []
    pulse_every = max(int(round(24.0 / freq)), 1)

    for t in range(steps):
        if t % pulse_every == 0:
            A += dose

        dA = -params.ka * A
        dD = params.ka * A - params.kd * D
        dE = params.kd * D - params.ke * E
        dM = params.ke * E + 0.2 * params.kd * D - params.km * M

        A = max(0.0, A + dt * dA)
        D = max(0.0, D + dt * dD)
        E = max(0.0, E + dt * dE)
        M = max(0.0, M + dt * dM)

        efficacy_signal = params.signal_gain * E / (1.0 + M)
        tox_signal = 0.35 * M + 0.15 * E

        rows.append({
            "time_h": float(t),
            "absorption_A": A,
            "distribution_D": D,
            "effect_E": E,
            "metabolism_M": M,
            "efficacy_signal": float(efficacy_signal),
            "toxicity_signal": float(tox_signal),
        })

    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame) -> Dict[str, float]:
    """总结四室 CTM 曲线。"""
    if curve.empty:
        return {"auc_efficacy": 0.0, "peak_efficacy": 0.0, "peak_toxicity": 0.0}
    y = curve["efficacy_signal"].to_numpy(dtype=np.float64)
    t = curve["time_h"].to_numpy(dtype=np.float64)
    trap = getattr(np, "trapezoid", None)
    auc = float(trap(y, t) if callable(trap) else np.trapz(y, t))
    return {
        "auc_efficacy": auc,
        "peak_efficacy": float(curve["efficacy_signal"].max()),
        "peak_toxicity": float(curve["toxicity_signal"].max()),
    }
