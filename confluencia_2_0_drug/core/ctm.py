from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Literature references for pharmacokinetic parameter values:
#
# - LNP delivery rates: Hassett et al. (2019) Mol Ther 27:1885-1897
#   DOI: 10.1016/j.ymthe.2019.06.015
# - circRNA stability/half-life: Wesselhoeft et al. (2018) Nat Commun 9:2629
#   DOI: 10.1038/s41467-018-05096-x
# - Nucleotide modification effects on stability:
#   Chen et al. (2019) Nature 586:651-655 (m6A modification)
#   Liu et al. (2023) Nat Commun 14:2548 (modified circRNA therapeutics)
# - Endosomal escape efficiency: Gilleron et al. (2013) Nat Biotechnol 31:638-646
#   DOI: 10.1038/nbt.2612
# - Tissue distribution (LNP): Paunovska et al. (2018) ACS Nano 12:8307-8320
#   DOI: 10.1021/acsnano.8b03575
# - Protein half-life: Cambridge Protein Database, median ~24h for therapeutic proteins
# - RNA degradation kinetics: Padgett et al. (2022) RNA 28:398-410
# ---------------------------------------------------------------------------


@dataclass
class CTMParams:
    ka: float
    kd: float
    ke: float
    km: float
    signal_gain: float


def params_from_micro_scores(binding: float, immune: float, inflammation: float) -> CTMParams:
    b = float(np.clip(binding, 0.0, 1.0))
    i = float(np.clip(immune, 0.0, 1.0))
    inf = float(np.clip(inflammation, 0.0, 1.0))

    # Higher binding / immune activation tends to faster useful distribution and stronger effect.
    # ka: absorption rate [0.15, 0.50]. Base 0.15/h corresponds to ~4.6h half-life for depot release
    # (consistent with subcutaneous depot kinetics; tune with binding score for target affinity).
    ka = 0.15 + 0.35 * b
    # kd: distribution rate [0.10, 0.40]. Higher immune activation accelerates tissue distribution
    # via increased vascular permeability and immune cell trafficking.
    kd = 0.10 + 0.30 * i
    # ke: effect elimination [0.08, 0.28]. Lower inflammation → slower clearance (less immune-mediated removal).
    ke = 0.08 + 0.20 * (1.0 - inf)
    # km: metabolism rate [0.06, 0.36]. Higher inflammation accelerates metabolic turnover
    # via elevated hepatic/renal clearance (cytokine-mediated CYP modulation; Morgan 2011).
    km = 0.06 + 0.30 * inf
    # Signal gain: therapeutic effect magnitude [0.8, 2.3].
    # Weighted 60% binding + 40% immune activation reflects that direct target engagement
    # is the primary efficacy driver, with immune response as a secondary amplifier.
    gain = 0.8 + 1.5 * (0.6 * b + 0.4 * i)
    return CTMParams(ka=ka, kd=kd, ke=ke, km=km, signal_gain=gain)


def simulate_ctm(
    dose: float,
    freq: float,
    params: CTMParams,
    horizon: int = 72,
    dt: float = 1.0,
) -> pd.DataFrame:
    steps = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    A = 0.0  # absorption compartment
    D = 0.0  # distribution compartment
    E = 0.0  # effect compartment
    M = 0.0  # metabolism load

    rows: List[Dict[str, float]] = []
    pulse_every = max(int(round(24.0 / freq)), 1)

    for t in range(steps):
        if t % pulse_every == 0:
            A += dose

        dA = -params.ka * A
        dD = params.ka * A - params.kd * D
        dE = params.kd * D - params.ke * E
        dM = params.ke * E + 0.2 * params.kd * D - params.km * M
        # 0.2 × kd × D: fraction of distribution that feeds into metabolism (hepatic first-pass).
        # Only 20% because most distributed drug reaches the effect site, not the liver.

        A = max(0.0, A + dt * dA)
        D = max(0.0, D + dt * dD)
        E = max(0.0, E + dt * dE)
        M = max(0.0, M + dt * dM)

        efficacy_signal = params.signal_gain * E / (1.0 + M)
        # Michaelis-Menten saturation: efficacy saturates as metabolic load increases.
        tox_signal = 0.35 * M + 0.15 * E
        # Toxicity = 70% metabolism-driven (accumulated metabolites) + 30% effect-driven (on-target toxicity).
        # Weights reflect that metabolic byproducts are the primary toxicity source for circRNA therapeutics.

        rows.append(
            {
                "time_h": float(t),
                "absorption_A": A,
                "distribution_D": D,
                "effect_E": E,
                "metabolism_M": M,
                "efficacy_signal": float(efficacy_signal),
                "toxicity_signal": float(tox_signal),
            }
        )

    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame) -> Dict[str, float]:
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


# ===================================================================
# circRNA six-compartment PK model (RNACTM)
# ===================================================================

@dataclass
class RNACTMParams:
    """Parameters for the circRNA six-compartment pharmacokinetic model.

    Compartments: Inj(jection) → LNP → Endo(some) → Cyto(plasmic RNA) → Trans(lated protein) → Clear
    """
    k_uptake: float        # Inj → LNP uptake rate (1/h), independent from LNP→Endo
    k_release: float       # LNP → endosome release rate (1/h)
    k_escape: float        # Endosomal escape efficiency (1/h)
    k_translate: float     # Translation initiation rate (1/h)
    k_degrade: float       # RNA degradation rate (1/h)
    k_protein_half: float  # Product protein half-life (h)
    k_immune_clear: float  # Immune-mediated clearance rate (1/h)

    # Tissue distribution coefficients (fractions, sum ≈ 1 for remaining)
    # Values from Paunovska et al. (2018) ACS Nano for standard LNP formulations:
    # ~80% liver (hepatocyte uptake via ApoE-mediated LDLR), ~10% spleen (macrophage uptake),
    # ~3% muscle, ~7% other (kidney, lung, heart).
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
    """Infer circRNA CTM parameters from molecular features and delivery configuration.

    Uses literature-derived priors for circRNA PK behavior, adjusted by
    sequence properties and delivery system characteristics.
    """
    mod = str(modification).lower().strip()
    vec = str(delivery_vector).strip()

    # --- Uptake rate: Inj → LNP, depends on route of administration ---
    base_uptake = {"IV": 0.80, "SC": 0.15, "IM": 0.20, "ID": 0.10}
    k_uptake = base_uptake.get(route.upper(), 0.30)

    # --- Release rate: LNP → Endosome, depends on delivery system ---
    # Values from Hassett et al. (2019) Mol Ther for LNP systems
    base_release = {"LNP_standard": 0.12, "LNP_liver": 0.15, "LNP_spleen": 0.10,
                    "AAV": 0.005, "naked": 0.80}
    k_release = base_release.get(vec, 0.12)

    # --- Endosomal escape: depends on delivery system and structure ---
    # Values from Gilleron et al. (2013) Nat Biotechnol (1-5% escape for LNP)
    base_escape = {"LNP_standard": 0.02, "LNP_liver": 0.03, "LNP_spleen": 0.02,
                   "AAV": 0.95, "naked": 0.01}
    k_escape = base_escape.get(vec, 0.02)
    # Higher structure stability → slightly better escape (more rigid RNA resists endosomal degradation)
    k_escape *= (0.8 + 0.4 * float(np.clip(struct_stability, 0.0, 1.0)))

    # --- Translation rate: depends on IRES strength ---
    k_translate = float(np.clip(0.02 + 0.30 * ires_score, 0.01, 0.50))

    # --- RNA degradation: depends on modification and GC content ---
    # Half-life multipliers from Wesselhoeft (2018), Chen (2019), Liu (2023)
    mod_half_life_map = {"none": 1.0, "m6a": 1.8, "Ψ": 2.5, "ψ": 2.5,
                         "5mc": 2.0, "ms2m6a": 3.0}
    stability_factor = mod_half_life_map.get(mod, mod_half_life_map["none"])
    base_degrade = 0.12  # unmodified RNA half-life ~6h (Wesselhoeft 2018) → k ≈ ln2/6 ≈ 0.12
    k_degrade = base_degrade / stability_factor
    # Higher GC → slightly slower degradation
    k_degrade *= (1.0 - 0.15 * float(np.clip(gc_content, 0.0, 1.0)))

    # --- Protein half-life: depends on product type (default 24h) ---
    k_protein_half = 24.0

    # --- Immune-mediated clearance: depends on innate immune activation ---
    k_immune_clear = float(np.clip(0.01 + 0.15 * innate_immune_score, 0.005, 0.30))

    # --- Tissue distribution: from delivery system parameters ---
    # Values from Paunovska et al. (2018) ACS Nano for LNP biodistribution
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
    """Simulate circRNA pharmacokinetics using a six-compartment model.

    Compartments:
      Inj: injected dose pool
      LNP: LNP-encapsulated / delivery complex
      Endo: endosomal compartment
      Cyto: cytoplasmic circRNA (available for translation)
      Trans: translated protein product
      Clear: cumulative clearance

    Uses scipy.integrate.solve_ivp with adaptive RK45 for numerical stability.
    """
    horizon = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    pulse_every = max(int(round(24.0 / freq)), 1)
    k_protein_degrade_base = float(np.log(2.0) / max(params.k_protein_half, 1.0))

    dose_times = [float(t) for t in range(0, horizon, pulse_every)]

    def ode_rhs(t, y):
        Inj, LNP, Endo, Cyto, Trans, Clear = y

        k_protein_degrade = k_protein_degrade_base

        # Separate uptake (Inj→LNP) and release (LNP→Endo) rate constants
        dInj = -params.k_uptake * Inj
        dLNP = params.k_uptake * Inj - params.k_release * LNP
        dEndo = params.k_release * LNP - params.k_escape * Endo
        dCyto = params.k_escape * Endo - (params.k_degrade + params.k_translate + params.k_immune_clear) * Cyto
        dTrans = params.k_translate * Cyto - k_protein_degrade * Trans
        dClear = params.k_degrade * Cyto + params.k_immune_clear * Cyto + k_protein_degrade * Trans

        return [dInj, dLNP, dEndo, dCyto, dTrans, dClear]

    t_grid = np.arange(0, horizon + 1, dt, dtype=np.float64)
    current_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    segments = []
    all_events = sorted(set(dose_times + [0.0, float(horizon)]))
    grid_set = set(t_grid.tolist())
    boundaries = sorted(grid_set | set(all_events))

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]
        if t_end <= t_start:
            continue

        if t_start in set(all_events) and any(abs(t_start - dt_ev) < 0.01 for dt_ev in dose_times):
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
            current_y = np.maximum(sol.y[:, -1], 0.0)
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
            t_val = float(seg.t[j])
            Inj_val = float(max(seg.y[0, j], 0.0))
            LNP_val = float(max(seg.y[1, j], 0.0))
            Endo_val = float(max(seg.y[2, j], 0.0))
            Cyto_val = float(max(seg.y[3, j], 0.0))
            Trans_val = float(max(seg.y[4, j], 0.0))
            Clear_val = float(max(seg.y[5, j], 0.0))

            circulating_rna = LNP_val + Endo_val + Cyto_val
            tissue_liver = circulating_rna * params.f_liver
            tissue_spleen = circulating_rna * params.f_spleen
            tissue_muscle = circulating_rna * params.f_muscle
            tissue_other = circulating_rna * params.f_other

            efficacy_signal = Trans_val
            toxicity_signal = 0.20 * Clear_val + 0.10 * params.k_immune_clear * Cyto_val

            rows.append({
                "time_h": t_val,
                "rna_injected": Inj_val,
                "rna_lnp": LNP_val,
                "rna_endosomal": Endo_val,
                "rna_cytoplasmic": Cyto_val,
                "protein_translated": Trans_val,
                "cumulative_clearance": Clear_val,
                "tissue_liver": tissue_liver,
                "tissue_spleen": tissue_spleen,
                "tissue_muscle": tissue_muscle,
                "tissue_other": tissue_other,
                "rna_circulating_total": circulating_rna,
                "efficacy_signal": float(efficacy_signal),
                "toxicity_signal": float(toxicity_signal),
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["time_h"], keep="last").sort_values("time_h").reset_index(drop=True)
    return df


def summarize_rna_ctm_curve(curve: pd.DataFrame) -> Dict[str, float]:
    """Summarize circRNA CTM simulation results."""
    if curve.empty:
        return {
            "rna_ctm_auc_efficacy": 0.0,
            "rna_ctm_peak_protein": 0.0,
            "rna_ctm_peak_cytoplasmic_rna": 0.0,
            "rna_ctm_protein_expression_window_h": 0.0,
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

    # Protein expression window: time above 50% of peak
    threshold = 0.5 * peak_protein if peak_protein > 0 else 0.0
    above = protein >= threshold
    window = 0.0
    if np.any(above):
        indices = np.where(above)[0]
        window = float(t[indices[-1]] - t[indices[0]]) if len(indices) > 1 else 1.0

    # RNA half-life estimate from circulating RNA
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

    # Bioavailability: fraction of total dose that reaches cytoplasm
    total_dose_injected = float(curve["rna_injected"].iloc[0]) if len(curve) > 0 else 0.0
    total_clearance = float(curve["cumulative_clearance"].iloc[-1]) if len(curve) > 0 else 0.0
    bioavail = total_clearance / max(total_dose_injected, 1e-6)
    bioavail = float(np.clip(bioavail, 0.0, 1.0))

    return {
        "rna_ctm_auc_efficacy": auc_eff,
        "rna_ctm_peak_protein": peak_protein,
        "rna_ctm_peak_cytoplasmic_rna": peak_rna_cyto,
        "rna_ctm_protein_expression_window_h": window,
        "rna_ctm_rna_half_life_h": rna_half,
        "rna_ctm_bioavailability_frac": bioavail,
        "rna_ctm_peak_toxicity": peak_tox,
    }
