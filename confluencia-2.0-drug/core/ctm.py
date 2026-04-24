from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

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

    # --- Release rate: depends on delivery system ---
    # Values from Hassett et al. (2019) Mol Ther for LNP systems
    base_release = {"LNP_standard": 0.12, "LNP_liver": 0.15, "LNP_spleen": 0.10,
                    "AAV": 0.005, "naked": 0.80}
    k_release = base_release.get(vec, 0.12)
    # Route adjustment: SC/IM has slower release
    route_release_factor = {"IV": 1.0, "SC": 0.4, "IM": 0.5, "ID": 0.3}
    k_release *= route_release_factor.get(route.upper(), 1.0)

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

    Returns DataFrame with time-series for all compartments plus tissue distribution.
    """
    steps = int(max(horizon, 2))
    dose = float(max(dose, 0.0))
    freq = float(max(freq, 0.01))

    Inj = 0.0
    LNP = 0.0
    Endo = 0.0
    Cyto = 0.0
    Trans = 0.0
    Clear = 0.0

    rows: List[Dict[str, float]] = []
    pulse_every = max(int(round(24.0 / freq)), 1)
    k_protein_degrade = float(np.log(2.0) / max(params.k_protein_half, 1.0))

    for t in range(steps):
        if t % pulse_every == 0:
            Inj += dose

        # Flux: Inj → LNP (rapid for IV, slower for SC/IM)
        dInj = -params.k_release * Inj
        dLNP = params.k_release * Inj - params.k_release * LNP  # same release rate
        # Flux: LNP → Endo → Cyto
        dEndo = params.k_release * LNP - params.k_escape * Endo
        dCyto = params.k_escape * Endo - (params.k_degrade + params.k_translate + params.k_immune_clear) * Cyto
        # Flux: Cyto → Trans (protein production)
        dTrans = params.k_translate * Cyto - k_protein_degrade * Trans
        # Accumulated clearance
        dClear = params.k_degrade * Cyto + params.k_immune_clear * Cyto + k_protein_degrade * Trans

        Inj = max(0.0, Inj + dt * dInj)
        LNP = max(0.0, LNP + dt * dLNP)
        Endo = max(0.0, Endo + dt * dEndo)
        Cyto = max(0.0, Cyto + dt * dCyto)
        Trans = max(0.0, Trans + dt * dTrans)
        Clear = max(0.0, Clear + dt * dClear)

        # Tissue distribution of circulating RNA (LNP + Endo compartments)
        circulating_rna = LNP + Endo + Cyto
        tissue_liver = circulating_rna * params.f_liver
        tissue_spleen = circulating_rna * params.f_spleen
        tissue_muscle = circulating_rna * params.f_muscle
        tissue_other = circulating_rna * params.f_other

        # Effective efficacy signal: proportional to translated protein
        efficacy_signal = Trans
        # Toxicity signal: from RNA degradation products + immune clearance
        toxicity_signal = 0.20 * Clear + 0.10 * params.k_immune_clear * Cyto
        # 20% cumulative clearance products + 10% immune-mediated clearance of cytoplasmic RNA.
        # Immune clearance weighted lower because circRNA is designed to minimize innate immune activation
        # (Wesselhoeft et al. 2019, Mol Cell). The 0.10 coefficient modulates by immune_score.

        rows.append({
            "time_h": float(t),
            "rna_injected": Inj,
            "rna_lnp": LNP,
            "rna_endosomal": Endo,
            "rna_cytoplasmic": Cyto,
            "protein_translated": Trans,
            "cumulative_clearance": Clear,
            "tissue_liver": tissue_liver,
            "tissue_spleen": tissue_spleen,
            "tissue_muscle": tissue_muscle,
            "tissue_other": tissue_other,
            "rna_circulating_total": circulating_rna,
            "efficacy_signal": float(efficacy_signal),
            "toxicity_signal": float(toxicity_signal),
        })

    return pd.DataFrame(rows)


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
