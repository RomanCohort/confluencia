"""
circrna_ctm.py - circRNA Compartmental Transmission Model (CTM)

Uses drug 2.0's existing RNACTM six-compartment model.

Six compartments:
1. Inj (injection site)
2. LNP (lipid nanoparticle)
3. Endo (endosome)
4. Cyto (cytoplasmic RNA)
5. Trans (translated protein)
6. Clear (clearance)

This module wraps the existing RNACTM model from drug 2.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Import from drug 2.0's existing CTM module
_DRUG_CORE = Path(__file__).resolve().parents[3] / "confluencia-2.0-drug" / "core"
if str(_DRUG_CORE) not in sys.path:
    sys.path.insert(0, str(_DRUG_CORE))

try:
    from ctm import RNACTMParams, infer_rna_ctm_params, simulate_rna_ctm
    HAS_DRUG_CTM = True
except ImportError:
    HAS_DRUG_CTM = False
    # Fallback implementation

# Fallback if drug 2.0 not available
if not HAS_DRUG_CTM:

    @dataclass
    class RNACTMParams:
        """Fallback RNACTM parameters."""
        k_release: float = 0.12
        k_escape: float = 0.02
        k_translate: float = 0.15
        k_degrade: float = 0.08
        k_protein_half: float = 24.0
        k_immune_clear: float = 0.05
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
        """Fallback parameter inference."""
        mod_half_life_map = {"none": 1.0, "m6a": 1.8, "psi": 2.5, "5mc": 2.0, "ms2m6a": 3.0}
        stability_factor = mod_half_life_map.get(modification.lower(), 1.0)
        k_degrade = 0.12 / stability_factor
        k_degrade *= (1.0 - 0.15 * gc_content)
        k_translate = 0.02 + 0.30 * ires_score
        k_immune_clear = 0.01 + 0.15 * innate_immune_score

        return RNACTMParams(
            k_release=0.12,
            k_escape=0.02,
            k_translate=k_translate,
            k_degrade=k_degrade,
            k_protein_half=24.0,
            k_immune_clear=k_immune_clear,
        )


def simulate_circrna_ctm(
    sequence: str,
    dose: float = 100.0,
    route: str = "IV",
    modification: str = "none",
    delivery_vector: str = "LNP_standard",
    duration: float = 96.0,
) -> Dict:
    """
    Simulate circRNA CTM using the existing RNACTM model.

    Args:
        sequence: circRNA sequence
        dose: Dose in ng/kg
        route: Administration route (IV/SC/IM)
        modification: Modification type (none/m6a/psi/5mc/ms2m6a)
        delivery_vector: Delivery system (LNP_standard/LNP_liver/LNP_spleen/AAV/naked)
        duration: Simulation duration in hours

    Returns:
        CTM simulation results with compartment curves
    """
    # Calculate sequence features
    seq_len = len(sequence)
    seq = sequence.upper().replace('T', 'U')
    gc = sum(1 for c in seq if c in 'GC') / max(seq_len, 1)

    # Estimate IRES score (from GC and length)
    ires_score = gc * 0.5 + min(seq_len / 500, 0.5)

    # Estimate structure stability
    entropy = -sum(
        (sum(1 for c in seq if c == n) / seq_len) * np.log2(sum(1 for c in seq if c == n) / seq_len + 1e-10)
        for n in ['A', 'U', 'G', 'C']
    )
    struct_stability = entropy / 2.0  # Normalize

    # Innate immune score (placeholder)
    innate_score = 0.3

    # Infer parameters using existing model
    params = infer_rna_ctm_params(
        modification=modification,
        delivery_vector=delivery_vector,
        route=route,
        ires_score=ires_score,
        gc_content=gc,
        struct_stability=struct_stability,
        innate_immune_score=innate_score,
    )

    # Simulate compartments
    times = np.linspace(0, duration, 100)

    # Simple simulation of six compartments
    inj = np.zeros(len(times))
    lnp = np.zeros(len(times))
    endo = np.zeros(len(times))
    cyto = np.zeros(len(times))
    trans = np.zeros(len(times))
    clear = np.zeros(len(times))

    # Initial dose
    inj[0] = dose

    dt = times[1] - times[0] if len(times) > 1 else 1.0

    for i in range(1, len(times)):
        # Inj → LNP
        d_inj = -params.k_release * inj[i-1]
        d_lnp = params.k_release * inj[i-1] - params.k_release * lnp[i-1]
        d_endo = params.k_release * lnp[i-1] - params.k_escape * endo[i-1]
        d_cyto = params.k_escape * endo[i-1] - params.k_translate * cyto[i-1] - params.k_degrade * cyto[i-1]
        d_trans = params.k_translate * cyto[i-1] - np.log(2) / params.k_protein_half * trans[i-1]
        d_clear = params.k_degrade * cyto[i-1] + np.log(2) / params.k_protein_half * trans[i-1] + params.k_immune_clear * cyto[i-1]

        inj[i] = max(0, inj[i-1] + d_inj * dt)
        lnp[i] = max(0, lnp[i-1] + d_lnp * dt)
        endo[i] = max(0, endo[i-1] + d_endo * dt)
        cyto[i] = max(0, cyto[i-1] + d_cyto * dt)
        trans[i] = max(0, trans[i-1] + d_trans * dt)
        clear[i] = clear[i-1] + d_clear * dt

    compartments = {
        'inj': inj,
        'lnp': lnp,
        'endo': endo,
        'cyto': cyto,
        'trans': trans,
        'clear': clear,
    }

    # Calculate summary
    auc_cyto = np.trapz(cyto, times)
    auc_trans = np.trapz(trans, times)
    cmax_cyto = np.max(cyto)
    tmax_cyto = times[np.argmax(cyto)]
    cmax_trans = np.max(trans)

    # Half-life estimation
    peak_idx = np.argmax(cyto)
    if peak_idx < len(cyto) - 10:
        log_cyto = np.log(cyto[peak_idx:] + 1e-10)
        slope = np.polyfit(times[peak_idx:] - times[peak_idx], log_cyto, 1)[0]
        half_life = -np.log(2) / slope if slope < 0 else np.log(2) / params.k_degrade
    else:
        half_life = np.log(2) / params.k_degrade

    # Effect (protein translation)
    effect = trans / np.max(trans) if np.max(trans) > 0 else np.zeros_like(trans)
    max_effect = np.max(effect)

    return {
        'times': times,
        'compartments': compartments,
        'summary': {
            'half_life': half_life,
            'cmax_blood': cmax_cyto,  # Use cyto as "blood" equivalent
            'cmax_tumor': cmax_trans,  # Use trans as "tumor" equivalent (protein effect)
            'tmax_blood': tmax_cyto,
            'tmax_tumor': times[np.argmax(trans)],
            'auc_blood': auc_cyto,
            'auc_tumor': auc_trans,
            'tumor_exposure_ratio': auc_trans / auc_cyto if auc_cyto > 0 else 0,
            'total_exposure': auc_cyto + auc_trans,
            'max_effect': max_effect,
            'effect_duration_hours': np.sum(effect > 0.1) * (duration / len(effect)),
        },
        'max_effect': max_effect,
        'params': params,
        'modification': modification,
        'delivery_vector': delivery_vector,
        'stability_factor': 1.0 / (params.k_degrade / 0.12),
        'sequence_length': seq_len,
        'gc_content': gc,
    }
        """
        Simulate circRNA PK in multiple compartments.

        Args:
            dose: Dose in ng/kg
            route: Administration route
            duration: Simulation duration in hours
            n_steps: Number of time steps
            eta: Individual variability parameters

        Returns:
            Dict with concentration curves and summary metrics
        """
        times = np.linspace(0, duration, n_steps)

        # Apply variability
        if eta:
            ka = self.params.ka * np.exp(eta.get('ka', 0))
            ke = self.params.ke * np.exp(eta.get('ke', 0))
        else:
            ka = self.params.ka
            ke = self.params.ke

        # Initialize compartments
        if self.extended:
            # 6-compartment model
            compartments = self._simulate_6compartment(
                dose, route, times, ka, ke
            )
        else:
            # 3-compartment model (simplified)
            compartments = self._simulate_3compartment(
                dose, route, times, ka, ke
            )

        # Calculate summary metrics
        summary = self._calculate_summary(compartments, times)

        return {
            'times': times,
            'compartments': compartments,
            'summary': summary,
            'modification': self.modification,
            'route': route.name,
        }

    def _simulate_3compartment(
        self,
        dose: float,
        route: Route,
        times: np.ndarray,
        ka: float,
        ke: float,
    ) -> Dict:
        """3-compartment model: Depot, Blood, Tumor."""

        depot = np.zeros(len(times))
        blood = np.zeros(len(times))
        tumor = np.zeros(len(times))

        # Initial conditions
        if route == Route.IV:
            blood[0] = dose / self.params.v_blood * self.params.f
            depot[0] = 0
        else:
            depot[0] = dose * self.params.f
            blood[0] = 0

        tumor[0] = 0

        dt = times[1] - times[0] if len(times) > 1 else 0.1

        # Rate constants
        k_bt = self.params.k_blood_tumor
        k_tb = self.params.k_tumor_blood
        k_e = ke
        k_a = ka if route != Route.IV else 100  # Instant for IV

        for i in range(1, len(times)):
            # Depot -> Blood
            d_depot = -k_a * depot[i-1]

            # Blood dynamics
            d_blood = (
                k_a * depot[i-1]            # From depot
                - k_bt * blood[i-1]         # To tumor
                + k_tb * tumor[i-1]         # From tumor
                - k_e * blood[i-1]          # Elimination
            )

            # Tumor dynamics
            d_tumor = (
                k_bt * blood[i-1]           # From blood
                - k_tb * tumor[i-1]         # To blood
                - self.params.ke_tumor * tumor[i-1]  # Tumor elimination
            )

            depot[i] = depot[i-1] + d_depot * dt
            blood[i] = max(0, blood[i-1] + d_blood * dt)
            tumor[i] = max(0, tumor[i-1] + d_tumor * dt)

        return {
            'depot': depot,
            'blood': blood,
            'tumor': tumor,
        }

    def _simulate_6compartment(
        self,
        dose: float,
        route: Route,
        times: np.ndarray,
        ka: float,
        ke: float,
    ) -> Dict:
        """6-compartment model: Depot, Blood, Tumor, Immune, Lymph, Bone."""

        depot = np.zeros(len(times))
        blood = np.zeros(len(times))
        tumor = np.zeros(len(times))
        immune = np.zeros(len(times))
        lymph = np.zeros(len(times))
        bone = np.zeros(len(times))

        # Initial conditions
        if route == Route.IV:
            blood[0] = dose / self.params.v_blood * self.params.f
            depot[0] = 0
        elif route == Route.IT:
            tumor[0] = dose * self.params.f
            depot[0] = 0
        else:
            depot[0] = dose * self.params.f
            blood[0] = 0

        dt = times[1] - times[0] if len(times) > 1 else 0.1

        for i in range(1, len(times)):
            # Depot -> Blood
            d_depot = -ka * depot[i-1]

            # Blood dynamics (central compartment)
            d_blood = (
                ka * depot[i-1]
                - self.params.k_blood_tumor * blood[i-1]
                - self.params.k_blood_immune * blood[i-1]
                - self.params.k_blood_lymph * blood[i-1]
                - self.params.k_blood_bone * blood[i-1]
                + self.params.k_tumor_blood * tumor[i-1]
                + self.params.k_immune_blood * immune[i-1]
                + self.params.k_lymph_blood * lymph[i-1]
                + self.params.k_bone_blood * bone[i-1]
                - ke * blood[i-1]
            )

            # Tumor compartment
            d_tumor = (
                self.params.k_blood_tumor * blood[i-1]
                - self.params.k_tumor_blood * tumor[i-1]
                - self.params.ke_tumor * tumor[i-1]
            )

            # Immune cells compartment
            d_immune = (
                self.params.k_blood_immune * blood[i-1]
                - self.params.k_immune_blood * immune[i-1]
                - self.params.ke_immune * immune[i-1]
            )

            # Lymph node compartment
            d_lymph = (
                self.params.k_blood_lymph * blood[i-1]
                - self.params.k_lymph_blood * lymph[i-1]
            )

            # Bone marrow compartment
            d_bone = (
                self.params.k_blood_bone * blood[i-1]
                - self.params.k_bone_blood * bone[i-1]
            )

            depot[i] = depot[i-1] + d_depot * dt
            blood[i] = max(0, blood[i-1] + d_blood * dt)
            tumor[i] = max(0, tumor[i-1] + d_tumor * dt)
            immune[i] = max(0, immune[i-1] + d_immune * dt)
            lymph[i] = max(0, lymph[i-1] + d_lymph * dt)
            bone[i] = max(0, bone[i-1] + d_bone * dt)

        return {
            'depot': depot,
            'blood': blood,
            'tumor': tumor,
            'immune': immune,
            'lymph': lymph,
            'bone': bone,
        }

    def _calculate_summary(
        self,
        compartments: Dict,
        times: np.ndarray,
    ) -> Dict:
        """Calculate PK summary metrics."""

        blood = compartments['blood']
        tumor = compartments.get('tumor', np.zeros_like(blood))

        # Peak concentration
        cmax_blood = np.max(blood)
        tmax_blood = times[np.argmax(blood)]

        cmax_tumor = np.max(tumor)
        tmax_tumor = times[np.argmax(tumor)]

        # AUC (Area Under Curve)
        auc_blood = np.trapz(blood, times)
        auc_tumor = np.trapz(tumor, times)

        # Half-life (from elimination phase)
        # Find peak and estimate elimination
        peak_idx = np.argmax(blood)
        if peak_idx < len(blood) - 10:
            elimination_phase = blood[peak_idx:]
            elimination_times = times[peak_idx:] - times[peak_idx]
            if len(elimination_phase) > 2 and elimination_phase[-1] > 0:
                # Linear regression on log scale
                log_conc = np.log(elimination_phase + 1e-10)
                slope = np.polyfit(elimination_times, log_conc, 1)[0]
                half_life = -np.log(2) / slope if slope < 0 else self.params.get_half_life()
            else:
                half_life = self.params.get_half_life()
        else:
            half_life = self.params.get_half_life()

        # Mean residence time
        mrt = auc_blood / cmax_blood if cmax_blood > 0 else 0

        # Tumor exposure ratio
        tumor_ratio = auc_tumor / auc_blood if auc_blood > 0 else 0

        return {
            'cmax_blood': cmax_blood,
            'tmax_blood': tmax_blood,
            'cmax_tumor': cmax_tumor,
            'tmax_tumor': tmax_tumor,
            'auc_blood': auc_blood,
            'auc_tumor': auc_tumor,
            'half_life': half_life,
            'mrt': mrt,
            'tumor_exposure_ratio': tumor_ratio,
            'total_exposure': auc_blood + auc_tumor,
        }

    def predict_effect(
        self,
        concentration: np.ndarray,
        ec50: float = 50.0,
        hill_coeff: float = 1.5,
        max_effect: float = 1.0,
    ) -> np.ndarray:
        """
        Predict pharmacological effect using Hill equation.

        E = Emax * C^Hill / (EC50^Hill + C^Hill)
        """
        return max_effect * np.power(concentration, hill_coeff) / (
            np.power(ec50, hill_coeff) + np.power(concentration, hill_coeff)
        )


def simulate_circrna_ctm(
    sequence: str,
    dose: float = 100.0,
    route: str = "SC",
    duration: float = 96.0,
    modification: str = "none",
    extended: bool = True,
) -> Dict:
    """
    Quick CTM simulation for circRNA.

    Args:
        sequence: circRNA sequence
        dose: Dose in ng/kg
        route: Administration route (IV/IM/SC/IT)
        duration: Duration in hours
        modification: Modification type (none/m6A/psi/5mC/ms2m6A)
        extended: Use 6-compartment model

    Returns:
        CTM simulation results
    """
    # Parse route
    route_map = {
        'IV': Route.IV,
        'IM': Route.IM,
        'SC': Route.SC,
        'IT': Route.IT,
    }
    route_enum = route_map.get(route.upper(), Route.SC)

    # Adjust dose based on sequence length
    seq_len = len(sequence)
    gc = sum(1 for c in sequence.upper() if c in 'GC') / max(seq_len, 1)

    # GC content affects stability (higher GC = longer half-life)
    stability_factor = 1 + gc * 0.5

    # Create model
    params = CircRNAPKParameters()

    # Adjust based on modification and sequence
    ref = CircRNACTMModel.REFERENCE_PARAMS.get(modification, {'ke': 0.08})
    params.ke = ref['ke'] / stability_factor

    model = CircRNACTMModel(params, modification, extended)

    # Simulate
    result = model.simulate(dose, route_enum, duration)

    # Add effect prediction
    blood_conc = result['compartments']['blood']
    tumor_conc = result['compartments'].get('tumor', np.zeros_like(blood_conc))

    effect = model.predict_effect(tumor_conc, ec50=dose * 0.3)
    max_effect = np.max(effect)

    result['effect'] = effect
    result['max_effect'] = max_effect
    result['sequence_length'] = seq_len
    result['gc_content'] = gc
    result['stability_factor'] = stability_factor

    # Simplified output for integration
    result['summary']['max_effect'] = max_effect
    result['summary']['half_life_hours'] = result['summary']['half_life']
    result['summary']['peak_concentration'] = result['summary']['cmax_blood']
    result['summary']['peak_time_hours'] = result['summary']['tmax_blood']
    result['summary']['auc'] = result['summary']['auc_blood']
    result['summary']['effect_duration_hours'] = np.sum(effect > 0.1) * (duration / len(effect)) if len(effect) > 0 else 0

    return result


# Integration with multimodal predictor
def integrate_ctm_with_multimodal(
    sequence: str,
    dose: float = 100.0,
    modification: str = "none",
) -> Dict:
    """Integrate CTM results with multimodal prediction."""

    ctm_result = simulate_circrna_ctm(sequence, dose, modification=modification)

    return {
        'pk': {
            'half_life': ctm_result['summary']['half_life'],
            'cmax': ctm_result['summary']['cmax_blood'],
            'auc': ctm_result['summary']['auc_blood'],
            'tumor_exposure': ctm_result['summary']['auc_tumor'],
        },
        'pd': {
            'max_effect': ctm_result['max_effect'],
            'effect_duration': ctm_result['summary']['effect_duration_hours'],
        },
        'distribution': {
            'tumor_ratio': ctm_result['summary']['tumor_exposure_ratio'],
            'compartments': list(ctm_result['compartments'].keys()),
        },
        'modification': modification,
        'stability_factor': ctm_result['stability_factor'],
    }