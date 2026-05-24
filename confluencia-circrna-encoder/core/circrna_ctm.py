"""
circrna_ctm.py - circRNA Compartmental Transmission Model (CTM)

Similar to drug 2.0's RNACTMModel, adapted for circRNA therapeutics.

Models:
- OneCompartmentModel: Simple PK model
- CircRNACTMModel: Multi-compartment model for circRNA
- CircRNACTMExtended: Extended 6-compartment model

Compartments:
1. Depot (injection site)
2. Blood/Plasma
3. Tumor tissue
4. Immune cells
5. Lymph nodes
6. Bone marrow
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Route(Enum):
    """Administration routes."""
    IV = 0      # Intravenous
    IM = 1      # Intramuscular
    SC = 2      # Subcutaneous
    IT = 3      # Intratumoral


class Modification(Enum):
    """circRNA modifications affecting stability."""
    NONE = "none"
    M6A = "m6A"        # N6-methyladenosine
    PSI = "psi"        # Pseudouridine
    M5C = "5mC"        # 5-methylcytosine
    MS2M6A = "ms2m6A"  # MS2+m6A combo


@dataclass
class CircRNAPKParameters:
    """PK parameters for circRNA."""

    # Absorption
    ka: float = 0.5        # Absorption rate (1/h)

    # Elimination
    ke: float = 0.08       # Elimination rate (1/h)
    ke_tumor: float = 0.05  # Tumor elimination
    ke_immune: float = 0.03  # Immune cell elimination

    # Distribution
    v_blood: float = 3.0      # Blood volume (L)
    v_tumor: float = 0.5      # Tumor volume (L)
    v_immune: float = 1.0     # Immune cells volume (L)
    v_lymph: float = 0.3      # Lymph node volume (L)
    v_bone: float = 0.2       # Bone marrow volume (L)

    # Transfer rates
    k_blood_tumor: float = 0.1     # Blood to tumor
    k_blood_immune: float = 0.15   # Blood to immune
    k_blood_lymph: float = 0.05    # Blood to lymph
    k_blood_bone: float = 0.02     # Blood to bone marrow
    k_tumor_blood: float = 0.08    # Tumor to blood
    k_immune_blood: float = 0.12   # Immune to blood
    k_lymph_blood: float = 0.03    # Lymph to blood
    k_bone_blood: float = 0.01     # Bone to blood

    # Bioavailability
    f: float = 0.8          # Bioavailability

    # Variability (CV)
    omega_ka: float = 0.3
    omega_ke: float = 0.25
    omega_v: float = 0.2

    # Residual error
    sigma_prop: float = 0.15

    def to_dict(self) -> Dict:
        return {
            'ka': self.ka, 'ke': self.ke,
            'ke_tumor': self.ke_tumor, 'ke_immune': self.ke_immune,
            'v_blood': self.v_blood, 'v_tumor': self.v_tumor,
            'v_immune': self.v_immune, 'v_lymph': self.v_lymph,
            'v_bone': self.v_bone,
            'k_blood_tumor': self.k_blood_tumor,
            'k_blood_immune': self.k_blood_immune,
            'f': self.f,
        }

    def get_half_life(self) -> float:
        """Calculate half-life in hours."""
        return np.log(2) / self.ke


class CircRNACTMModel:
    """
    circRNA Compartmental Transmission Model.

    Multi-compartment PK model for circRNA therapeutics.
    """

    # Reference parameters from literature
    REFERENCE_PARAMS = {
        'none': {'ke': 0.1155, 'half_life': 6.0, 'cv': 0.25},
        'm6A': {'ke': 0.0642, 'half_life': 10.8, 'cv': 0.22},
        'psi': {'ke': 0.0462, 'half_life': 15.0, 'cv': 0.20},
        '5mC': {'ke': 0.0555, 'half_life': 12.5, 'cv': 0.22},
        'ms2m6A': {'ke': 0.0347, 'half_life': 20.0, 'cv': 0.18},
    }

    def __init__(
        self,
        params: Optional[CircRNAPKParameters] = None,
        modification: str = "none",
        extended: bool = False,
    ):
        self.params = params or CircRNAPKParameters()
        self.modification = modification
        self.extended = extended

        # Adjust params based on modification
        if modification in self.REFERENCE_PARAMS:
            ref = self.REFERENCE_PARAMS[modification]
            self.params.ke = ref['ke']

    def simulate(
        self,
        dose: float,
        route: Route = Route.SC,
        duration: float = 96.0,
        n_steps: int = 100,
        eta: Optional[Dict] = None,
    ) -> Dict:
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