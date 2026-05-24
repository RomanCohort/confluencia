"""
innate_immune.py — Innate immune activation prediction for circRNA.

Predicts RIG-I, TLR, PKR activation based on circRNA sequence features.
Adapted from drug 2.0's innate_immune.py for circRNA context.
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
class InnateImmuneConfig:
    """Configuration for innate immune prediction."""

    # RIG-I activation thresholds
    rig_i_gc_threshold: float = 0.45  # GC content threshold
    rig_i_length_min: int = 100  # Minimum length
    rig_i_length_opt: int = 300  # Optimal length

    # TLR activation thresholds
    tlr_u_threshold: float = 0.30  # U content threshold
    tlr_repeat_threshold: float = 0.05  # Repeat threshold

    # PKR activation thresholds
    pkr_entropy_threshold: float = 1.5  # Entropy threshold

    # Scoring weights
    rig_i_weight: float = 0.40
    tlr_weight: float = 0.35
    pkr_weight: float = 0.25


class InnateImmunePredictor:
    """
    Predict innate immune activation for circRNA sequences.

    Pathways:
    - RIG-I: dsRNA sensor, activated by GC-rich sequences
    - TLR3/7/8: Toll-like receptors, activated by U-rich sequences
    - PKR: Protein kinase R, activated by structured RNAs
    """

    def __init__(self, config: Optional[InnateImmuneConfig] = None):
        self.config = config or InnateImmuneConfig()

    def predict_rig_i(self, sequence: str) -> Dict:
        """
        Predict RIG-I activation potential.

        RIG-I is activated by:
        - dsRNA structures (GC-rich sequences)
        - 5' triphosphate (not applicable for circRNA)
        - Length > 100bp optimal

        Args:
            sequence: circRNA sequence

        Returns:
            Dict with score, activation level, factors
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        # Calculate factors
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        # dsRNA potential (GC pairing)
        dsRNA_potential = gc * 0.5 + self._calc_pairing_potential(seq) * 0.5

        # Length factor
        if length < self.config.rig_i_length_min:
            length_factor = 0.2
        elif length > self.config.rig_i_length_opt:
            length_factor = 0.8
        else:
            length_factor = 0.5 + 0.3 * (length - 100) / 200

        # Overall score
        score = (
            dsRNA_potential * 0.6 +
            length_factor * 0.3 +
            gc * 0.1
        )

        # Activation level
        if score >= 0.7:
            level = "High"
        elif score >= 0.4:
            level = "Medium"
        else:
            level = "Low"

        return {
            'score': score,
            'level': level,
            'gc_content': gc,
            'dsRNA_potential': dsRNA_potential,
            'length_factor': length_factor,
            'activation_probability': min(score * 1.2, 1.0),
        }

    def predict_tlr(self, sequence: str) -> Dict:
        """
        Predict TLR3/7/8 activation potential.

        TLR7/8: activated by U-rich, GU-rich sequences
        TLR3: activated by dsRNA

        Args:
            sequence: circRNA sequence

        Returns:
            Dict with TLR activation scores
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        # Calculate factors
        u = sum(1 for c in seq if c == 'U') / max(length, 1)
        g = sum(1 for c in seq if c == 'G') / max(length, 1)
        gu = sum(1 for i in range(len(seq)-1) if seq[i:i+2] in ['GU', 'UG']) / max(length-1, 1)

        # Repeat content (U-repeats activate TLR7)
        u_repeat = self._calc_repeat(seq, 'U')

        # TLR7 score (U-rich, GU motifs)
        tlr7_score = (
            u * 0.4 +
            gu * 0.3 +
            min(u_repeat / 10, 0.3)
        )

        # TLR3 score (similar to RIG-I, dsRNA)
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)
        tlr3_score = gc * 0.5

        # Overall TLR score
        score = tlr7_score * 0.7 + tlr3_score * 0.3

        if score >= 0.6:
            level = "High"
        elif score >= 0.3:
            level = "Medium"
        else:
            level = "Low"

        return {
            'score': score,
            'tlr7_score': tlr7_score,
            'tlr3_score': tlr3_score,
            'level': level,
            'u_content': u,
            'gu_motifs': gu,
            'u_repeats': u_repeat,
        }

    def predict_pkr(self, sequence: str) -> Dict:
        """
        Predict PKR activation potential.

        PKR is activated by:
        - Structured RNAs (high entropy)
        - dsRNA regions
        - Length-dependent activation

        Args:
            sequence: circRNA sequence

        Returns:
            Dict with PKR activation scores
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        # Entropy
        counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
        for c in seq:
            if c in counts:
                counts[c] += 1

        probs = [counts[n] / max(length, 1) for n in 'AUGC']
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

        # Structure potential (entropy-based)
        structure_potential = min(entropy / 2.0, 1.0)

        # GC pairs
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

        # Overall score
        score = (
            structure_potential * 0.5 +
            gc * 0.3 +
            min(length / 500, 1.0) * 0.2
        )

        if score >= 0.6:
            level = "High"
        elif score >= 0.35:
            level = "Medium"
        else:
            level = "Low"

        return {
            'score': score,
            'level': level,
            'entropy': entropy,
            'structure_potential': structure_potential,
            'gc_content': gc,
        }

    def _calc_pairing_potential(self, sequence: str) -> float:
        """Calculate base pairing potential."""
        seq = sequence.upper()

        # Count potential pairs
        pairs = 0
        for i in range(len(seq) - 1):
            c1, c2 = seq[i], seq[i+1]
            if (c1, c2) in [('G', 'C'), ('C', 'G'), ('A', 'U'), ('U', 'A')]:
                pairs += 1

        return pairs / max(len(seq) - 1, 1)

    def _calc_repeat(self, sequence: str, nucleotide: str) -> int:
        """Calculate maximum repeat length for a nucleotide."""
        max_repeat = 0
        count = 0

        for c in sequence:
            if c == nucleotide:
                count += 1
                max_repeat = max(max_repeat, count)
            else:
                count = 0

        return max_repeat

    def predict_all(self, sequence: str) -> Dict:
        """
        Predict all innate immune activation scores.

        Args:
            sequence: circRNA sequence

        Returns:
            Dict with RIG-I, TLR, PKR predictions and overall score
        """
        rig_i = self.predict_rig_i(sequence)
        tlr = self.predict_tlr(sequence)
        pkr = self.predict_pkr(sequence)

        # Weighted overall score
        overall = (
            rig_i['score'] * self.config.rig_i_weight +
            tlr['score'] * self.config.tlr_weight +
            pkr['score'] * self.config.pkr_weight
        )

        return {
            'rig_i': rig_i,
            'tlr': tlr,
            'pkr': pkr,
            'overall_score': overall,
            'overall_level': "High" if overall >= 0.6 else ("Medium" if overall >= 0.4 else "Low"),
            'summary': f"RIG-I: {rig_i['level']}, TLR: {tlr['level']}, PKR: {pkr['level']}",
        }

    def predict_batch(self, sequences: List[str]) -> pd.DataFrame:
        """
        Batch prediction for multiple sequences.

        Args:
            sequences: List of circRNA sequences

        Returns:
            DataFrame with all predictions
        """
        results = []

        for i, seq in enumerate(sequences):
            pred = self.predict_all(seq)

            results.append({
                'sequence_id': i,
                'sequence_length': len(seq),
                'rig_i_score': pred['rig_i']['score'],
                'rig_i_level': pred['rig_i']['level'],
                'tlr_score': pred['tlr']['score'],
                'tlr7_score': pred['tlr']['tlr7_score'],
                'tlr_level': pred['tlr']['level'],
                'pkr_score': pred['pkr']['score'],
                'pkr_level': pred['pkr']['level'],
                'overall_innate_score': pred['overall_score'],
                'overall_level': pred['overall_level'],
            })

        return pd.DataFrame(results)


def quick_predict(sequence: str) -> Dict:
    """Quick innate immune prediction for a single sequence."""
    predictor = InnateImmunePredictor()
    return predictor.predict_all(sequence)