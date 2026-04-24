from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class PKPDParams:
    """Three-compartment PK model with sigmoid Emax PD link.

    Compartment model: Depot → Central ↔ Peripheral, with elimination from Central.
    PD model: E = Emax * C^Hill / (EC50^Hill + C^Hill)
    """
    ka: float       # Absorption rate constant (1/h): depot → central transfer rate.
    k12: float      # Central → peripheral distribution rate (1/h).
    k21: float      # Peripheral → central return rate (1/h).
    ke: float       # Elimination rate from central compartment (1/h).
    v1_l: float     # Central compartment volume (L); typical vaccine/therapeutic protein range 2-8 L.
    emax: float     # Maximum drug effect (dimensionless, normalized to 0-2.2).
    ec50_mg_per_l: float  # Concentration producing 50% of Emax (mg/L).
    hill: float = 1.2     # Hill coefficient for sigmoid PD curve; >1 indicates positive cooperativity.


def infer_pkpd_params(
    binding: float,
    immune: float,
    inflammation: float,
    dose_mg: float,
    freq_per_day: float,
) -> PKPDParams:
    b = float(np.clip(binding, 0.0, 1.0))
    i = float(np.clip(immune, 0.0, 1.0))
    inf = float(np.clip(inflammation, 0.0, 1.0))
    d = float(max(dose_mg, 0.0))
    f = float(max(freq_per_day, 0.01))

    # --- Absorption rate ---
    # Base 0.22/h (half-life ~3.2h, IM depot range) + binding-dependent acceleration.
    # log1p(f) term: more frequent dosing slightly increases absorption efficiency (tissue priming).
    # Bounds [0.05, 1.50]: physiologically plausible for IM/SC depot to rapid IV bolus.
    ka = float(np.clip(0.22 + 0.40 * b + 0.03 * np.log1p(f), 0.05, 1.50))
    # --- Central→Peripheral distribution ---
    # Base 0.08/h; immune activation increases vascular permeability and tissue uptake.
    k12 = float(np.clip(0.08 + 0.22 * i, 0.03, 0.60))
    # --- Peripheral→Central return ---
    # Base 0.06/h; lower inflammation allows more return (less tissue sequestration).
    k21 = float(np.clip(0.06 + 0.18 * (1.0 - inf), 0.03, 0.60))
    # --- Elimination ---
    # Base 0.04/h (half-life ~17h, consistent with therapeutic protein clearance);
    # inflammation accelerates via cytokine-mediated CYP upregulation.
    ke = float(np.clip(0.04 + 0.10 * inf + 0.01 * np.log1p(f), 0.02, 0.50))

    # Vaccines/therapeutic proteins often remain in low-liter central exposure space.
    # Base 2.8L (approx plasma volume); low binding increases distribution volume (less target-mediated
    # disposition); dose-proportional expansion at 0.06 L/mg accounts for non-linear distribution.
    v1_l = float(np.clip(2.8 + 2.2 * (1.0 - b) + 0.06 * d, 1.5, 10.0))
    # Emax: weighted 55% binding + 45% immune activation; reflects dual mechanism of action
    # for circRNA therapeutics (direct target engagement + immune stimulation).
    emax = float(np.clip(0.65 + 0.95 * (0.55 * b + 0.45 * i), 0.2, 2.2))
    # EC50: higher binding lowers EC50 (more potent); inflammation raises it (reduced sensitivity).
    ec50 = float(np.clip(0.18 + 0.90 * (1.0 - b) + 0.30 * inf, 0.05, 3.0))
    # Hill coefficient: base 1.0 (Michaelis-Menten) + immune-driven positive cooperativity.
    hill = float(np.clip(1.0 + 0.50 * i, 0.8, 2.5))

    return PKPDParams(
        ka=ka,
        k12=k12,
        k21=k21,
        ke=ke,
        v1_l=v1_l,
        emax=emax,
        ec50_mg_per_l=ec50,
        hill=hill,
    )


def _dose_schedule(horizon: int, freq_per_day: float) -> List[float]:
    period_h = max(24.0 / max(freq_per_day, 0.01), 0.5)
    times = [0.0]
    t = period_h
    while t <= float(horizon):
        times.append(float(t))
        t += period_h
    # Avoid duplicate timestamp from numerical edge cases.
    uniq = sorted(set(round(x, 6) for x in times))
    return [float(x) for x in uniq]


def _pk_ode(_: float, y: np.ndarray, p: PKPDParams) -> np.ndarray:
    depot, central, peripheral = y
    d_depot = -p.ka * depot
    d_central = p.ka * depot - (p.k12 + p.ke) * central + p.k21 * peripheral
    d_peripheral = p.k12 * central - p.k21 * peripheral
    return np.array([d_depot, d_central, d_peripheral], dtype=np.float64)


def simulate_pkpd(
    dose_mg: float,
    freq_per_day: float,
    params: PKPDParams,
    horizon: int = 72,
    dt: float = 1.0,
) -> pd.DataFrame:
    horizon = int(max(horizon, 2))
    dt = float(max(dt, 0.1))
    dose_mg = float(max(dose_mg, 0.0))
    freq_per_day = float(max(freq_per_day, 0.01))

    all_t: List[float] = []
    all_y: List[np.ndarray] = []

    y0 = np.array([dose_mg, 0.0, 0.0], dtype=np.float64)
    schedule = _dose_schedule(horizon=horizon, freq_per_day=freq_per_day)
    bounds = schedule + [float(horizon)]

    for i in range(len(bounds) - 1):
        t0 = float(bounds[i])
        t1 = float(bounds[i + 1])
        if t1 <= t0:
            continue

        grid = np.arange(t0, t1 + 1e-9, dt, dtype=np.float64)
        if grid.size == 0 or grid[-1] < t1:
            grid = np.append(grid, t1)

        sol = solve_ivp(
            fun=lambda t, y: _pk_ode(t, y, params),
            t_span=(t0, t1),
            y0=y0,
            t_eval=grid,
            method="RK45",
            vectorized=False,
            rtol=1e-5,   # Relative tolerance: 1e-5 provides ~5 significant digits for concentration,
                          # sufficient for pharmacological decision-making (PK analysis typically needs ~3 digits).
            atol=1e-7,   # Absolute tolerance: 1e-7 mg ensures accuracy even when concentrations approach
                          # zero (important for terminal elimination phase half-life estimation).
        )

        if sol.y.shape[1] == 0:
            continue

        for j in range(sol.y.shape[1]):
            t = float(sol.t[j])
            y = sol.y[:, j].astype(np.float64)
            if all_t and abs(t - all_t[-1]) < 1e-8:
                all_t[-1] = t
                all_y[-1] = y
            else:
                all_t.append(t)
                all_y.append(y)

        y0 = sol.y[:, -1].astype(np.float64)
        # Apply bolus to depot at the start of the next segment (except the last boundary).
        if i + 1 < len(schedule):
            y0 = y0.copy()
            y0[0] += dose_mg

    if not all_t:
        return pd.DataFrame(
            columns=[
                "time_h",
                "pkpd_depot_mg",
                "pkpd_central_mg",
                "pkpd_peripheral_mg",
                "pkpd_conc_mg_per_l",
                "pkpd_effect",
            ]
        )

    t = np.asarray(all_t, dtype=np.float64)
    y_mat = np.vstack(all_y)
    depot = np.clip(y_mat[:, 0], 0.0, None)
    central = np.clip(y_mat[:, 1], 0.0, None)
    peripheral = np.clip(y_mat[:, 2], 0.0, None)

    conc = np.clip(central / max(params.v1_l, 1e-6), 0.0, None)
    h = float(max(params.hill, 0.1))
    ec50_h = float(max(params.ec50_mg_per_l, 1e-6)) ** h
    effect = params.emax * (conc**h) / (ec50_h + conc**h)

    return pd.DataFrame(
        {
            "time_h": t,
            "pkpd_depot_mg": depot,
            "pkpd_central_mg": central,
            "pkpd_peripheral_mg": peripheral,
            "pkpd_conc_mg_per_l": conc,
            "pkpd_effect": effect,
        }
    )


def _estimate_terminal_half_life(time_h: np.ndarray, conc: np.ndarray) -> float:
    t = np.asarray(time_h, dtype=np.float64)
    c = np.asarray(conc, dtype=np.float64)

    pos = c > 1e-9
    t = t[pos]
    c = c[pos]
    if t.size < 4:
        return 0.0

    start = int(np.floor(0.7 * t.size))
    t_tail = t[start:]
    c_tail = c[start:]
    if t_tail.size < 3:
        return 0.0

    y = np.log(np.clip(c_tail, 1e-12, None))
    slope, _intercept = np.polyfit(t_tail, y, 1)
    if slope >= 0.0:
        return 0.0

    return float(np.log(2.0) / (-slope))


def summarize_pkpd_curve(curve: pd.DataFrame, params: PKPDParams) -> Dict[str, float]:
    if curve.empty:
        return {
            "pkpd_half_life_h": 0.0,
            "pkpd_vd_ss_l": 0.0,
            "pkpd_clearance_lph": 0.0,
            "pkpd_cmax_mg_per_l": 0.0,
            "pkpd_tmax_h": 0.0,
            "pkpd_auc_conc": 0.0,
            "pkpd_auc_effect": 0.0,
            "pkpd_effect_peak": 0.0,
            "pkpd_pk_effect_corr": 0.0,
        }

    t = curve["time_h"].to_numpy(dtype=np.float64)
    conc = curve["pkpd_conc_mg_per_l"].to_numpy(dtype=np.float64)
    eff = curve["pkpd_effect"].to_numpy(dtype=np.float64)

    cmax = float(np.max(conc)) if conc.size > 0 else 0.0
    tmax = float(t[int(np.argmax(conc))]) if conc.size > 0 else 0.0
    auc_c = float(np.trapezoid(conc, t)) if t.size > 1 else 0.0
    auc_e = float(np.trapezoid(eff, t)) if t.size > 1 else 0.0
    e_peak = float(np.max(eff)) if eff.size > 0 else 0.0

    corr = 0.0
    if conc.size > 2 and np.std(conc) > 1e-9 and np.std(eff) > 1e-9:
        corr = float(np.corrcoef(conc, eff)[0, 1])

    half_life = _estimate_terminal_half_life(t, conc)
    vd_ss = float(max(params.v1_l * (1.0 + params.k12 / max(params.k21, 1e-6)), 0.0))
    clearance = float(max(params.ke * params.v1_l, 0.0))

    return {
        "pkpd_half_life_h": half_life,
        "pkpd_vd_ss_l": vd_ss,
        "pkpd_clearance_lph": clearance,
        "pkpd_cmax_mg_per_l": cmax,
        "pkpd_tmax_h": tmax,
        "pkpd_auc_conc": auc_c,
        "pkpd_auc_effect": auc_e,
        "pkpd_effect_peak": e_peak,
        "pkpd_pk_effect_corr": corr,
    }
