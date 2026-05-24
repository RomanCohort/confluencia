"""
Scoring modules for circRNA predictions.

Mirrors drug module's scoring.py structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


class CircRNAScorer:
    """
    Main scorer for circRNA immunogenicity predictions.
    """

    def score(self, predictions: Dict) -> Dict:
        """Calculate overall immunogenicity score."""
        # Base score from innate immune
        innate_score = predictions.get('innate_activation', 0)

        # Therapeutic potential
        therapeutic = predictions.get('therapeutic_window', 0)

        # Composite
        overall = innate_score * 0.5 + therapeutic * 0.3 + predictions.get('confidence', 0.5) * 0.2

        predictions['overall_score'] = overall
        predictions['tier'] = self._tier(overall)

        return predictions

    def _tier(self, score: float) -> str:
        """Classify into tier."""
        if score >= 0.7:
            return "Tier1_High"
        elif score >= 0.5:
            return "Tier2_Medium"
        elif score >= 0.3:
            return "Tier3_Low"
        else:
            return "Tier4_VeryLow"

    def batch_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring."""
        df['overall_score'] = (
            df.get('innate_activation', 0) * 0.5 +
            df.get('therapeutic_window', 0) * 0.3 +
            df.get('confidence', 0.5) * 0.2
        )
        df['tier'] = df['overall_score'].apply(self._tier)
        return df


class CompositeScorer:
    """
    Composite score calculator (mirrors drug scoring).

    Aggregates multiple predictions into unified scores.
    """

    # Score weights
    WEIGHTS = {
        "immunotherapy_score": 0.25,
        "tumor_killing_index": 0.20,
        "overall_immunogenicity": 0.15,
        "immune_cycle_score": 0.10,
        "tme_score": 0.10,
        "therapeutic_window": 0.10,
        "tide_score": 0.05,
        "ips": 0.05,
    }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scoring to predictions."""
        # Calculate weighted composite
        df["composite_score"] = 0.0
        for key, weight in self.WEIGHTS.items():
            if key in df.columns:
                df["composite_score"] += df[key] * weight

        # Normalize IPS (0-10 to 0-1)
        if "ips" in df.columns:
            df["ips_normalized"] = df["ips"] / 10.0

        # Tier classification
        df["tier"] = df["composite_score"].apply(self._tier)

        return df

    def _tier(self, score: float) -> str:
        """Classify into tier."""
        if score >= 0.7:
            return "Tier1_High"
        elif score >= 0.5:
            return "Tier2_Medium"
        elif score >= 0.3:
            return "Tier3_Low"
        else:
            return "Tier4_VeryLow"


class ReportScorer:
    """
    Report score calculator for innate immunity.

    Calculates RIG-I, TLR, PKR activation scores.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply report scoring."""
        # Innate activation score
        innate_cols = ["rig_i_score", "tlr_score", "pkr_score"]
        if all(c in df.columns for c in innate_cols):
            df["innate_activation"] = (
                df["rig_i_score"] * 0.4 +
                df["tlr_score"] * 0.3 +
                df["pkr_score"] * 0.3
            )

        # Risk assessment
        df["risk_level"] = df.apply(self._risk, axis=1)

        return df

    def _risk(self, row: pd.Series) -> str:
        """Assess risk level."""
        trained_risk = row.get("trained_model_risk", 0)
        tide = row.get("tide_score", 0)

        if trained_risk > 0.7 or tide > 0.7:
            return "HighRisk"
        elif trained_risk > 0.4 or tide > 0.4:
            return "MediumRisk"
        else:
            return "LowRisk"


def calculate_ips_score(predictions: Dict) -> float:
    """
    Calculate IPS (Immunotherapy Predictive Score).

    IPS = 0.3*immunogenicity + 0.2*tumor_killing + 0.15*cycle + 0.1*window
    """
    return (
        predictions.get("overall_immunogenicity", 0) * 0.3 +
        predictions.get("tumor_killing_index", 0) * 0.2 +
        predictions.get("immune_cycle_score", 0) * 0.15 +
        predictions.get("therapeutic_window", 0) * 0.1 +
        predictions.get("tme_score", 0) * 0.1 +
        predictions.get("immunotherapy_score", 0) * 0.15
    ) * 10  # Scale to 0-10