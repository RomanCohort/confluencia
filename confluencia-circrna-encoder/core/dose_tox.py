"""
dose_tox.py — Dose-response and toxicity prediction for circRNA.

Adapted from drug 2.0's dose_tox.py for circRNA therapeutic context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class DoseToxConfig:
    """Configuration for dose-tox prediction."""

    # Dose ranges (ng/kg)
    min_dose: float = 1.0
    max_dose: float = 1000.0
    default_dose: float = 100.0

    # Toxicity thresholds
    max_safe_dose: float = 500.0  # Maximum safe dose
    toxicity_threshold: float = 0.7  # Score threshold for toxicity

    # Immune toxicity parameters
    immune_tox_weight: float = 0.4
    cytokine_weight: float = 0.3
    inflammation_weight: float = 0.3


class CircRNADoseResponse:
    """
    Predict dose-response for circRNA therapeutics.

    Considerations:
    - CircRNA stability and persistence
    - Immune activation dose-dependency
    - Cytokine storm potential
    - Therapeutic window estimation
    """

    def __init__(self, config: Optional[DoseToxConfig] = None):
        self.config = config or DoseToxConfig()

    def predict_response(self, sequence: str, dose: float) -> Dict:
        """
        Predict therapeutic response at given dose.

        Args:
            sequence: circRNA sequence
            dose: Dose in ng/kg

        Returns:
            Dict with efficacy and toxicity predictions
        """
        from .innate_immune import quick_predict

        # Get innate immune prediction
        innate = quick_predict(sequence)

        # Dose-response curve (Hill equation)
        efficacy_score = self._hill_equation(
            dose,
            innate['overall_score'],
            ec50=self.config.default_dose,
            hill_coeff=1.5
        )

        # Toxicity prediction
        toxicity_score = self._predict_toxicity(sequence, dose, innate)

        # Therapeutic window
        therapeutic_window = efficacy_score - toxicity_score

        return {
            'dose': dose,
            'efficacy_score': efficacy_score,
            'toxicity_score': toxicity_score,
            'therapeutic_window': therapeutic_window,
            'innate_activation': innate['overall_score'],
            'safe': toxicity_score < self.config.toxicity_threshold,
            'recommended_dose': self._find_optimal_dose(sequence),
        }

    def _hill_equation(self, dose: float, max_response: float, ec50: float, hill_coeff: float) -> float:
        """Hill equation for dose-response."""
        return max_response * (dose ** hill_coeff) / (ec50 ** hill_coeff + dose ** hill_coeff)

    def _predict_toxicity(self, sequence: str, dose: float, innate: Dict) -> float:
        """Predict toxicity based on dose and immune activation."""
        seq = sequence.upper()
        length = len(seq)

        # Base toxicity from immune activation
        immune_tox = innate['overall_score'] * self.config.immune_tox_weight

        # Cytokine storm potential (high dose + high immune activation)
        cytokine_risk = (
            min(dose / self.config.max_safe_dose, 1.0) *
            innate['rig_i']['score'] *
            self.config.cytokine_weight
        )

        # Inflammation risk
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)
        inflammation = gc * self.config.inflammation_weight

        # Overall toxicity
        toxicity = immune_tox + cytokine_risk + inflammation

        # Dose scaling
        dose_factor = min(dose / self.config.max_dose, 1.5)

        return min(toxicity * dose_factor, 1.0)

    def _find_optimal_dose(self, sequence: str) -> float:
        """Find optimal dose maximizing therapeutic window."""
        best_dose = self.config.default_dose
        best_window = 0

        # Search dose range
        for dose in [10, 50, 100, 200, 300, 500]:
            response = self.predict_response(sequence, dose)

            if response['therapeutic_window'] > best_window and response['safe']:
                best_window = response['therapeutic_window']
                best_dose = dose

        return best_dose

    def predict_dose_range(self, sequence: str) -> pd.DataFrame:
        """
        Predict response across dose range.

        Args:
            sequence: circRNA sequence

        Returns:
            DataFrame with dose-response curve
        """
        doses = [1, 10, 50, 100, 200, 300, 500, 750, 1000]

        results = []
        for dose in doses:
            response = self.predict_response(sequence, dose)
            results.append(response)

        return pd.DataFrame(results)

    def calculate_ld50(self, sequence: str) -> float:
        """Estimate LD50 ( lethal dose 50%)."""
        from .innate_immune import quick_predict

        innate = quick_predict(sequence)

        # Higher immune activation = lower LD50
        base_ld50 = self.config.max_dose * 10
        immune_factor = 1 - innate['overall_score'] * 0.8

        return base_ld50 * immune_factor

    def calculate_ed50(self, sequence: str) -> float:
        """Estimate ED50 (effective dose 50%)."""
        return self._find_optimal_dose(sequence)

    def calculate_therapeutic_index(self, sequence: str) -> float:
        """Calculate therapeutic index (LD50 / ED50)."""
        ld50 = self.calculate_ld50(sequence)
        ed50 = self.calculate_ed50(sequence)

        return ld50 / max(ed50, 1)


class CytokineStormPredictor:
    """Predict cytokine storm potential for circRNA."""

    # Cytokine thresholds
    CYTOKINE_THRESHOLDS = {
        'IL6': 100,  # pg/mL threshold for concern
        'TNF': 50,
        'IFN_alpha': 100,
        'IFN_beta': 50,
    }

    def predict(self, sequence: str, dose: float) -> Dict:
        """Predict cytokine levels."""
        from .innate_immune import quick_predict

        innate = quick_predict(sequence)

        # Base cytokine levels from immune activation
        rig_i = innate['rig_i']['score']
        tlr = innate['tlr']['score']

        # Estimate cytokine levels
        il6 = rig_i * 150 + tlr * 100 + dose / 10
        tnf = rig_i * 80 + tlr * 60 + dose / 20
        ifn_alpha = rig_i * 120 + dose / 5
        ifn_beta = rig_i * 80 + dose / 10

        # Risk assessment
        storm_risk = (
            (il6 > self.CYTOKINE_THRESHOLDS['IL6']) +
            (tnf > self.CYTOKINE_THRESHOLDS['TNF']) +
            (ifn_alpha > self.CYTOKINE_THRESHOLDS['IFN_alpha'])
        ) / 3

        return {
            'IL6': il6,
            'TNF': tnf,
            'IFN_alpha': ifn_alpha,
            'IFN_beta': ifn_beta,
            'storm_risk': storm_risk,
            'storm_level': "High" if storm_risk > 0.7 else ("Medium" if storm_risk > 0.4 else "Low"),
            'threshold_exceeded': {
                'IL6': il6 > self.CYTOKINE_THRESHOLDS['IL6'],
                'TNF': tnf > self.CYTOKINE_THRESHOLDS['TNF'],
                'IFN_alpha': ifn_alpha > self.CYTOKINE_THRESHOLDS['IFN_alpha'],
            },
        }


def quick_dose_predict(sequence: str, dose: float = 100.0) -> Dict:
    """Quick dose-response prediction."""
    predictor = CircRNADoseResponse()
    return predictor.predict_response(sequence, dose)