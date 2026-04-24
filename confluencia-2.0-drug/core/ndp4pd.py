from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class NDP4PDParams:
    alpha: float
    beta: float
    gamma: float
    delta: float
    coupling: float
    delay_gain: float
    saturation: float
    signal_gain: float


def ndp4pd_from_ctm_like(ka: float, kd: float, ke: float, km: float, signal_gain: float) -> NDP4PDParams:
    ka = float(np.clip(ka, 0.02, 1.2))
    kd = float(np.clip(kd, 0.02, 1.2))
    ke = float(np.clip(ke, 0.02, 1.2))
    km = float(np.clip(km, 0.01, 1.0))
    sg = float(np.clip(signal_gain, 0.1, 6.0))

    # Map CTM-like parameters into NDP4PD dynamic coefficients.
    alpha = 0.6 * ka + 0.08
    beta = 0.7 * kd + 0.06
    gamma = 0.5 * ke + 0.05
    delta = 0.7 * km + 0.03
    coupling = 0.15 + 0.25 * kd
    delay_gain = 0.04 + 0.12 * km
    saturation = 0.12 + 0.18 * km
    return NDP4PDParams(
        alpha=float(np.clip(alpha, 0.03, 1.3)),
        beta=float(np.clip(beta, 0.03, 1.3)),
        gamma=float(np.clip(gamma, 0.02, 1.0)),
        delta=float(np.clip(delta, 0.01, 0.9)),
        coupling=float(np.clip(coupling, 0.05, 0.8)),
        delay_gain=float(np.clip(delay_gain, 0.0, 0.6)),
        saturation=float(np.clip(saturation, 0.02, 0.8)),
        signal_gain=sg,
    )


def simulate_ndp4pd(
    dose: float,
    freq: float,
    params: NDP4PDParams,
    horizon: int = 72,
    dt: float = 1.0,
) -> pd.DataFrame:
    """NDP4PD-style nonlinear 4-phase pharmacodynamics simulation.

    Phases:
    - N: input/transport reservoir
    - D: distribution field
    - P: pharmacodynamic activity
    - R: regulatory/toxicity load
    """
    steps = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    N = 0.0
    D = 0.0
    P = 0.0
    R = 0.0

    pulse_every = max(int(round(24.0 / freq)), 1)
    delay_steps = max(int(round(6.0 / dt)), 1)
    p_hist = [0.0 for _ in range(delay_steps + 1)]

    rows: List[Dict[str, float]] = []
    for t in range(steps):
        if t % pulse_every == 0:
            N += dose

        p_delayed = float(p_hist[0])

        dN = -params.alpha * N
        dD = params.alpha * N - params.beta * D + params.coupling * np.tanh(P)
        dP = params.beta * D - params.gamma * P - params.delta * R
        dR = params.delta * P - 0.5 * params.gamma * R + params.delay_gain * p_delayed

        N = max(0.0, N + dt * dN)
        D = max(0.0, D + dt * dD)
        P = max(0.0, P + dt * dP)
        R = max(0.0, R + dt * dR)

        p_hist.append(P)
        p_hist.pop(0)

        efficacy_signal = params.signal_gain * P / (1.0 + params.saturation * R)
        toxicity_signal = 0.28 * R + 0.12 * P

        rows.append(
            {
                "time_h": float(t),
                "absorption_A": N,
                "distribution_D": D,
                "effect_E": P,
                "metabolism_M": R,
                "efficacy_signal": float(efficacy_signal),
                "toxicity_signal": float(toxicity_signal),
            }
        )

    return pd.DataFrame(rows)
