"""
pkpd.py — Pharmacokinetic/Pharmacodynamic modeling for circRNA.

Adapted from drug 2.0's pkpd.py for circRNA context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class PKPDConfig:
    """Configuration for PK/PD modeling."""

    # PK parameters (hours)
    absorption_rate: float = 0.5  # circRNA uptake
    distribution_rate: float = 0.3  # circRNA distribution
    elimination_rate: float = 0.05  # circRNA degradation (slow)

    # circRNA specific
    stability_factor: float = 0.8  # circRNA stability advantage
    persistence_factor: float = 2.0  # circRNA persistence (days vs hours)

    # PD parameters
    effect_delay: float = 4.0  # Hours before effect
    max_effect: float = 1.0  # Maximum therapeutic effect
    ec50: float = 100.0  # ng/kg


class CircRNAPKModel:
    """
    Pharmacokinetic model for circRNA.

    Models:
    - circRNA uptake (IV/injection)
    - Distribution to tissues
    - Persistence (circRNA advantage: no linear RNA degradation)
    - Elimination (slow, days vs hours)
    """

    def __init__(self, config: Optional[PKPDConfig] = None):
        self.config = config or PKPDConfig()

    def simulate_pk(self, dose: float, duration: float = 168) -> Dict:
        """
        Simulate PK over time.

        Args:
            dose: Dose in ng/kg
            duration: Duration in hours (default: 7 days = 168h)

        Returns:
            PK profile
        """
        dt = 0.1  # Time step (hours)
        times = np.arange(0, duration, dt)

        # Initial concentration
        c0 = dose  # Initial

        # Compartment model
        # Central (blood) -> Peripheral (tissue) -> Elimination

        central = [c0]
        peripheral = [0]
        eliminated = [0]

        k_abs = self.config.absorption_rate
        k_dist = self.config.distribution_rate
        k_elim = self.config.elimination_rate / self.config.persistence_factor

        for t in times[1:]:
            # circRNA stability factor
            stability = self.config.stability_factor

            # Flow rates
            flow_to_peripheral = central[-1] * k_dist
            flow_back = peripheral[-1] * k_dist * 0.5  # Slow return
            flow_elim = central[-1] * k_elim * (1 - stability)

            # Updates
            c_central = central[-1] - flow_to_peripheral + flow_back - flow_elim
            c_peripheral = peripheral[-1] + flow_to_peripheral - flow_back
            c_elim = eliminated[-1] + flow_elim

            central.append(max(0, c_central))
            peripheral.append(max(0, c_peripheral))
            eliminated.append(c_elim)

        return {
            'times': times,
            'central_concentration': np.array(central),
            'peripheral_concentration': np.array(peripheral),
            'eliminated': np.array(eliminated),
            'half_life': np.log(2) / k_elim,
            'auc': np.sum(central) * dt,
            'peak': max(central),
            'peak_time': times[np.argmax(central)],
        }

    def estimate_half_life(self, sequence: str) -> float:
        """Estimate circRNA half-life based on sequence."""
        seq = sequence.upper()
        length = len(seq)

        # GC-rich circRNAs are more stable
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        # Length factor (longer circRNAs persist longer)
        length_factor = min(length / 300, 1.5)

        # Base half-life for circRNA: 24-48 hours (much longer than linear RNA)
        base_half_life = 24 * self.config.persistence_factor

        # Adjust by stability
        adjusted_half_life = base_half_life * (gc * 0.5 + length_factor * 0.5)

        return adjusted_half_life


class CircRNAPDModel:
    """
    Pharmacodynamic model for circRNA.

    Models:
    - Effect delay (immune activation time)
    - Emax model for efficacy
    - Dose-effect relationship
    """

    def __init__(self, config: Optional[PKPDConfig] = None):
        self.config = config or PKPDConfig()

    def simulate_pd(self, pk_profile: Dict, sequence: str) -> Dict:
        """
        Simulate PD based on PK profile.

        Args:
            pk_profile: PK simulation results
            sequence: circRNA sequence

        Returns:
            PD profile
        """
        from .innate_immune import quick_predict

        times = pk_profile['times']
        concentrations = pk_profile['central_concentration']

        # Get immune activation potential
        immune = quick_predict(sequence)
        immune_potential = immune['overall_score']

        # Effect model (Emax with delay)
        effects = []

        effect_delay = self.config.effect_delay

        for i, (t, c) in enumerate(zip(times, concentrations)):
            # Effect with delay
            delayed_index = max(0, i - int(effect_delay / 0.1))
            delayed_c = concentrations[delayed_index]

            # Emax model
            effect = self.config.max_effect * immune_potential * delayed_c / (self.config.ec50 + delayed_c)

            effects.append(effect)

        return {
            'times': times,
            'effect': np.array(effects),
            'max_effect': max(effects),
            'effect_duration': self._calc_effect_duration(times, effects),
            'effect_onset': times[np.argmax(effects)],
            'immune_potential': immune_potential,
        }

    def _calc_effect_duration(self, times: np.ndarray, effects: List[float]) -> float:
        """Calculate duration of significant effect."""
        threshold = max(effects) * 0.5

        above_threshold = [e > threshold for e in effects]

        if not any(above_threshold):
            return 0

        first_above = next(i for i, a in enumerate(above_threshold) if a)
        last_above = len(above_threshold) - next(i for i, a in enumerate(reversed(above_threshold)) if a)

        return times[last_above] - times[first_above]


class CircRNAPKPD:
    """Combined PK/PD model for circRNA."""

    def __init__(self, config: Optional[PKPDConfig] = None):
        self.config = config or PKPDConfig()
        self.pk_model = CircRNAPKModel(config)
        self.pd_model = CircRNAPDModel(config)

    def simulate(self, sequence: str, dose: float, duration: float = 168) -> Dict:
        """
        Simulate full PK/PD profile.

        Args:
            sequence: circRNA sequence
            dose: Dose in ng/kg
            duration: Duration in hours

        Returns:
            Complete PK/PD simulation
        """
        # PK simulation
        pk = self.pk_model.simulate_pk(dose, duration)

        # Adjust half-life based on sequence
        actual_half_life = self.pk_model.estimate_half_life(sequence)

        # PD simulation
        pd = self.pd_model.simulate_pd(pk, sequence)

        return {
            'pk': pk,
            'pd': pd,
            'sequence_properties': {
                'half_life': actual_half_life,
                'stability': self.config.stability_factor,
            },
            'summary': {
                'dose': dose,
                'peak_concentration': pk['peak'],
                'peak_time_hours': pk['peak_time'],
                'half_life_hours': actual_half_life,
                'auc': pk['auc'],
                'max_effect': pd['max_effect'],
                'effect_duration_hours': pd['effect_duration'],
            },
        }

    def find_optimal_dosing(self, sequence: str) -> Dict:
        """Find optimal dosing regimen."""
        best_dose = self.config.ec50
        best_score = 0

        for dose in [10, 50, 100, 200, 300, 500]:
            sim = self.simulate(sequence, dose)

            # Score: efficacy vs toxicity balance
            efficacy = sim['pd']['max_effect']
            auc = sim['pk']['auc']
            toxicity = auc / 1000  # Higher exposure = more toxicity risk

            score = efficacy - toxicity * 0.3

            if score > best_score:
                best_score = score
                best_dose = dose

        return {
            'optimal_dose': best_dose,
            'expected_half_life': self.pk_model.estimate_half_life(sequence),
            'expected_effect': self.simulate(sequence, best_dose)['pd']['max_effect'],
            'dosing_frequency': f"Every {int(self.pk_model.estimate_half_life(sequence) * 2)} hours",
        }


def simulate_pkpd(sequence: str, dose: float = 100) -> Dict:
    """Quick PK/PD simulation."""
    model = CircRNAPKPD()
    return model.simulate(sequence, dose)