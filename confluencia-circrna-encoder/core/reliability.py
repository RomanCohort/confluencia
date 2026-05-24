"""
reliability.py — Reliability and confidence estimation for circRNA predictions.

Adapted from drug 2.0's reliability.py for circRNA context.
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
class ReliabilityConfig:
    """Configuration for reliability estimation."""

    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5

    # Uncertainty parameters
    ensemble_size: int = 10
    bootstrap_samples: int = 100

    # Feature-based reliability
    feature_stability_weight: float = 0.3
    prediction_variance_weight: float = 0.4
    domain_weight: float = 0.3


class ReliabilityEstimator:
    """
    Estimate prediction reliability and confidence.

    Methods:
    - Ensemble variance
    - Feature-based stability
    - Domain extrapolation detection
    - Calibrated confidence
    """

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()

    def estimate_confidence(self, sequence: str, prediction: float) -> Dict:
        """
        Estimate confidence for a prediction.

        Args:
            sequence: circRNA sequence
            prediction: Predicted score

        Returns:
            Dict with confidence metrics
        """
        # Feature-based reliability
        feature_reliability = self._feature_reliability(sequence)

        # Prediction stability (bootstrap)
        stability = self._prediction_stability(sequence)

        # Domain check (is sequence similar to training data?)
        domain_score = self._domain_check(sequence)

        # Combined confidence
        confidence = (
            feature_reliability * self.config.feature_stability_weight +
            stability * self.config.prediction_variance_weight +
            domain_score * self.config.domain_weight
        )

        # Calibrate to [0, 1]
        confidence = np.clip(confidence, 0, 1)

        # Confidence level
        if confidence >= self.config.high_confidence_threshold:
            level = "High"
        elif confidence >= self.config.medium_confidence_threshold:
            level = "Medium"
        else:
            level = "Low"

        return {
            'confidence_score': confidence,
            'confidence_level': level,
            'feature_reliability': feature_reliability,
            'prediction_stability': stability,
            'domain_score': domain_score,
            'prediction_adjusted': prediction * confidence,
            'uncertainty': 1 - confidence,
            'recommendation': self._get_recommendation(confidence),
        }

    def _feature_reliability(self, sequence: str) -> float:
        """Calculate feature-based reliability."""
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        # Length reliability (optimal: 100-500)
        if 100 <= length <= 500:
            length_reliability = 1.0
        elif 50 <= length <= 1000:
            length_reliability = 0.7
        else:
            length_reliability = 0.3

        # Composition reliability (balanced GC)
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        if 0.35 <= gc <= 0.65:
            gc_reliability = 1.0
        elif 0.25 <= gc <= 0.75:
            gc_reliability = 0.7
        else:
            gc_reliability = 0.4

        # Entropy reliability
        counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
        for c in seq:
            if c in counts:
                counts[c] += 1

        probs = [counts[n] / max(length, 1) for n in 'AUGC']
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

        if entropy >= 1.5:
            entropy_reliability = 1.0
        elif entropy >= 1.2:
            entropy_reliability = 0.7
        else:
            entropy_reliability = 0.4

        # Combined
        return (length_reliability + gc_reliability + entropy_reliability) / 3

    def _prediction_stability(self, sequence: str) -> float:
        """Calculate prediction stability via bootstrapping."""
        from .innate_immune import quick_predict

        # Bootstrap predictions with small mutations
        predictions = []

        for _ in range(self.config.bootstrap_samples):
            # Create perturbed sequence
            perturbed = self._perturb_sequence(sequence, rate=0.02)
            pred = quick_predict(perturbed)
            predictions.append(pred['overall_score'])

        variance = np.var(predictions)
        stability = 1 - min(variance * 5, 1)

        return stability

    def _perturb_sequence(self, sequence: str, rate: float = 0.02) -> str:
        """Create perturbed sequence."""
        seq = list(sequence)

        n_mutations = int(len(seq) * rate)

        for _ in range(n_mutations):
            pos = np.random.randint(0, len(seq))
            choices = ['A', 'U', 'G', 'C']
            seq[pos] = np.random.choice(choices)

        return ''.join(seq)

    def _domain_check(self, sequence: str) -> float:
        """Check if sequence is in known domain."""
        seq = sequence.upper()
        length = len(seq)

        # Check for unusual patterns
        max_repeat = 0
        for nuc in 'AUGC':
            count = 0
            max_c = 0
            for c in seq:
                if c == nuc:
                    count += 1
                    max_c = max(max_c, count)
                else:
                    count = 0
            max_repeat = max(max_repeat, max_c)

        # High repeats = lower domain score
        repeat_penalty = min(max_repeat / 50, 0.5)

        # Check for extreme composition
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)
        composition_penalty = 0 if 0.3 <= gc <= 0.7 else abs(gc - 0.5)

        domain_score = 1 - repeat_penalty - composition_penalty

        return max(domain_score, 0)

    def _get_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence."""
        if confidence >= 0.8:
            return "Prediction is reliable. Proceed with confidence."
        elif confidence >= 0.6:
            return "Prediction is moderately reliable. Consider additional validation."
        elif confidence >= 0.4:
            return "Prediction has significant uncertainty. Recommend experimental validation."
        else:
            return "Prediction is unreliable. Do not use for critical decisions."

    def batch_estimate(self, sequences: List[str], predictions: List[float]) -> pd.DataFrame:
        """Batch reliability estimation."""
        import pandas as pd

        results = []

        for seq, pred in zip(sequences, predictions):
            confidence = self.estimate_confidence(seq, pred)
            results.append(confidence)

        return pd.DataFrame(results)


class PredictionCalibrator:
    """Calibrate predictions for reliability."""

    def __init__(self):
        self.calibration_map = {}

    def calibrate(self, prediction: float, reliability: float) -> float:
        """Calibrate prediction based on reliability."""
        # Lower reliability = more conservative prediction
        calibrated = prediction * reliability

        # Add uncertainty bounds
        uncertainty = 1 - reliability
        lower_bound = calibrated - uncertainty * 0.2
        upper_bound = calibrated + uncertainty * 0.2

        return {
            'calibrated_prediction': calibrated,
            'lower_bound': max(0, lower_bound),
            'upper_bound': min(1, upper_bound),
            'uncertainty_range': upper_bound - lower_bound,
        }


def assess_prediction_reliability(sequence: str, prediction: float) -> Dict:
    """Quick reliability assessment."""
    estimator = ReliabilityEstimator()
    return estimator.estimate_confidence(sequence, prediction)