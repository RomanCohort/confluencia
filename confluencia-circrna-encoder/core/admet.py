"""
admet.py — ADMET-like properties for circRNA (Immunogenicity, Toxicity, Stability).

Adapted from drug 2.0's admet.py for circRNA context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class ADMETConfig:
    """Configuration for circRNA ADMET-like prediction."""

    # Thresholds
    immunogenicity_threshold: float = 0.6
    toxicity_threshold: float = 0.7
    stability_threshold: float = 0.5

    # Weights for overall score
    immunogenicity_weight: float = 0.35
    toxicity_weight: float = 0.25
    stability_weight: float = 0.20
    persistence_weight: float = 0.20


class CircRNAADMET:
    """
    ADMET-like property prediction for circRNA.

    Properties:
    - Immunogenicity (A-like): immune activation potential
    - Distribution (D-like): tissue distribution potential
    - Metabolism (M-like): circRNA stability/degradation
    - Excretion (E-like): circRNA persistence
    - Toxicity (T): cytokine storm/inflammation risk
    """

    def __init__(self, config: Optional[ADMETConfig] = None):
        self.config = config or ADMETConfig()

    def predict(self, sequence: str) -> Dict:
        """
        Predict all ADMET-like properties.

        Args:
            sequence: circRNA sequence

        Returns:
            ADMET property predictions
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        # 1. Immunogenicity (A)
        immunogenicity = self._predict_immunogenicity(seq, length)

        # 2. Distribution (D)
        distribution = self._predict_distribution(seq, length)

        # 3. Stability/Metabolism (M)
        stability = self._predict_stability(seq, length)

        # 4. Persistence/Excretion (E)
        persistence = self._predict_persistence(seq, length)

        # 5. Toxicity (T)
        toxicity = self._predict_toxicity(seq, length, immunogenicity)

        # Overall score
        overall = (
            immunogenicity['score'] * self.config.immunogenicity_weight +
            distribution['score'] * 0.1 +
            stability['score'] * self.config.stability_weight +
            persistence['score'] * self.config.persistence_weight -
            toxicity['score'] * self.config.toxicity_weight
        )

        return {
            'immunogenicity': immunogenicity,
            'distribution': distribution,
            'stability': stability,
            'persistence': persistence,
            'toxicity': toxicity,
            'overall_score': np.clip(overall, 0, 1),
            'pass': self._check_pass(immunogenicity, toxicity, stability),
            'recommendation': self._get_recommendation(immunogenicity, toxicity, stability),
        }

    def _predict_immunogenicity(self, seq: str, length: int) -> Dict:
        """Predict immunogenicity."""
        from .innate_immune import quick_predict

        immune = quick_predict(seq)

        return {
            'score': immune['overall_score'],
            'level': immune['overall_level'],
            'rig_i': immune['rig_i']['score'],
            'tlr': immune['tlr']['score'],
            'pkr': immune['pkr']['score'],
            'pass': immune['overall_score'] >= self.config.immunogenicity_threshold,
        }

    def _predict_distribution(self, seq: str, length: int) -> Dict:
        """Predict tissue distribution potential."""
        # Size affects distribution
        size_factor = min(length / 300, 1.5)

        # GC affects stability in tissues
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        # CircRNAs distribute broadly due to stability
        distribution_score = size_factor * 0.5 + gc * 0.3 + 0.2

        return {
            'score': min(distribution_score, 1),
            'size_factor': size_factor,
            'expected_tissues': ['Liver', 'Kidney', 'Tumor'],
            'blood_barrier_pass': distribution_score > 0.5,
        }

    def _predict_stability(self, seq: str, length: int) -> Dict:
        """Predict circRNA stability."""
        # GC content = stability
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        # Length = stability (longer circRNAs more stable)
        length_factor = min(length / 200, 1.5)

        # Entropy = complexity = more stable structure
        counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
        for c in seq:
            if c in counts:
                counts[c] += 1

        probs = [counts[n] / max(length, 1) for n in 'AUGC']
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

        stability_score = gc * 0.4 + length_factor * 0.3 + min(entropy / 2, 1) * 0.3

        return {
            'score': stability_score,
            'gc_content': gc,
            'length_factor': length_factor,
            'entropy': entropy,
            'half_life_hours': 24 * stability_score,
            'pass': stability_score >= self.config.stability_threshold,
        }

    def _predict_persistence(self, seq: str, length: int) -> Dict:
        """Predict circRNA persistence."""
        stability = self._predict_stability(seq, length)

        # circRNAs have inherent persistence advantage (no linear RNA degradation)
        persistence_score = stability['score'] * 0.8 + 0.2

        return {
            'score': persistence_score,
            'half_life_days': stability['half_life_hours'] / 24,
            'persistence_advantage': 'circRNA vs linear RNA: ~10x longer',
        }

    def _predict_toxicity(self, seq: str, length: int, immunogenicity: Dict) -> Dict:
        """Predict toxicity risk."""
        from .dose_tox import CytokineStormPredictor

        # Cytokine storm risk
        cytokine_pred = CytokineStormPredictor()

        # Use default dose for prediction
        cytokine = cytokine_pred.predict(seq, dose=100)

        toxicity_score = cytokine['storm_risk']

        return {
            'score': toxicity_score,
            'level': cytokine['storm_level'],
            'cytokine_risk': cytokine['storm_risk'],
            'IL6_estimate': cytokine['IL6'],
            'TNF_estimate': cytokine['TNF'],
            'pass': toxicity_score < self.config.toxicity_threshold,
        }

    def _check_pass(self, immunogenicity: Dict, toxicity: Dict, stability: Dict) -> bool:
        """Check if sequence passes ADMET criteria."""
        return (
            immunogenicity['pass'] and
            toxicity['pass'] and
            stability['pass']
        )

    def _get_recommendation(self, immunogenicity: Dict, toxicity: Dict, stability: Dict) -> str:
        """Get recommendation based on ADMET."""
        issues = []

        if not immunogenicity['pass']:
            issues.append("Low immunogenicity")

        if not toxicity['pass']:
            issues.append("High toxicity risk")

        if not stability['pass']:
            issues.append("Low stability")

        if not issues:
            return "Sequence passes ADMET criteria. Suitable for therapeutic development."

        return f"Issues: {', '.join(issues)}. Consider sequence optimization."

    def predict_batch(self, sequences: List[str]) -> List[Dict]:
        """Batch ADMET prediction."""
        return [self.predict(seq) for seq in sequences]


def quick_admet(sequence: str) -> Dict:
    """Quick ADMET prediction."""
    predictor = CircRNAADMET()
    return predictor.predict(sequence)