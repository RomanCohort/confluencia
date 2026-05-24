"""
multimodal_predictor.py — Multi-modal circRNA prediction.

Combines multiple modalities:
1. Sequence encoding (XGBoost/RF model)
2. Innate immune activation (RIG-I/TLR/PKR)
3. Dose-response prediction
4. PK/PD simulation
5. ADMET properties
6. Multi-scale modeling
7. Gene signature integration

Similar to drug 2.0's multi-modal approach.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import joblib

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal prediction."""

    # Modality weights
    sequence_weight: float = 0.35
    innate_weight: float = 0.25
    dose_weight: float = 0.15
    pkpd_weight: float = 0.10
    admet_weight: float = 0.10
    gene_weight: float = 0.05

    # Default parameters
    default_dose: float = 100.0
    gene_threshold: float = 0.5


class MultiModalCircRNAPredictor:
    """
    Multi-modal circRNA immunogenicity predictor.

    Integrates multiple prediction modalities for comprehensive analysis.
    """

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        config: Optional[MultiModalConfig] = None,
    ):
        self.config = config or MultiModalConfig()

        # Load model
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None

        # Load scaler
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
            except:
                self.scaler = None
        else:
            self.scaler = None

    def predict_full(self, sequence: str, dose: float = None, gene_signature: Dict = None) -> Dict:
        """
        Full multi-modal prediction.

        Args:
            sequence: circRNA sequence
            dose: Dose in ng/kg (optional)
            gene_signature: Gene expression dict (optional)

        Returns:
            Comprehensive prediction report
        """
        dose = dose or self.config.default_dose

        # Initialize modules
        from .features import CircRNAFeatureExtractor
        from .innate_immune import quick_predict
        from .dose_tox import quick_dose_predict
        from .admet import quick_admet
        from .multiscale import multiscale_simulation
        from .reliability import assess_prediction_reliability
        from .circrna_ctm import simulate_circrna_ctm

        # 1. Sequence-based prediction (ML model)
        seq_prediction = self._predict_sequence(sequence)

        # 2. Innate immune activation
        innate = quick_predict(sequence)

        # 3. Dose-response
        dose_response = quick_dose_predict(sequence, dose)

        # 4. CTM simulation (replaces simple PK/PD)
        ctm = simulate_circrna_ctm(sequence, dose, extended=True)

        # 5. ADMET properties
        admet = quick_admet(sequence)

        # 6. Multi-scale modeling
        multiscale = multiscale_simulation(sequence)

        # 7. Reliability assessment
        reliability = assess_prediction_reliability(sequence, seq_prediction['immunogenicity'])

        # 8. Gene signature integration (if provided)
        gene_score = self._calculate_gene_score(gene_signature)

        # Composite score (multi-modal fusion)
        composite = self._calculate_composite(
            seq_prediction, innate, dose_response, ctm, admet, gene_score
        )

        # Generate report
        report = {
            # Main prediction
            'immunogenicity': seq_prediction['immunogenicity'],
            'confidence': seq_prediction['confidence'],
            'level': seq_prediction['level'],
            'composite_score': composite['overall'],

            # Modality breakdown
            'modalities': {
                'sequence': seq_prediction,
                'innate_immune': innate,
                'dose_response': dose_response,
                'ctm': ctm['summary'],
                'admet': admet,
                'multiscale': multiscale['final_outcome'],
            },

            # Detailed innate immune
            'innate_details': {
                'rig_i': innate['rig_i'],
                'tlr': innate['tlr'],
                'pkr': innate['pkr'],
                'overall': innate['overall_score'],
            },

            # Dose optimization
            'dose_optimization': {
                'current_dose': dose,
                'efficacy': dose_response['efficacy_score'],
                'toxicity': dose_response['toxicity_score'],
                'therapeutic_window': dose_response['therapeutic_window'],
                'safe': dose_response['safe'],
            },

            # PK/PD summary (now from CTM)
            'pkpd_summary': {
                'half_life_hours': ctm['summary']['half_life'],
                'max_effect': ctm['max_effect'],
                'peak_concentration': ctm['summary']['cmax_blood'],
                'peak_time_hours': ctm['summary']['tmax_blood'],
                'auc': ctm['summary']['auc_blood'],
                'tumor_exposure': ctm['summary']['auc_tumor'],
                'tumor_ratio': ctm['summary']['tumor_exposure_ratio'],
                'effect_duration_hours': ctm['summary']['effect_duration_hours'],
            },

            # CTM compartments (new)
            'ctm_details': {
                'compartments': list(ctm['compartments'].keys()),
                'modification': ctm['modification'],
                'stability_factor': ctm['stability_factor'],
            },

            # ADMET assessment
            'admet': {
                'pass': admet['pass'],
                'recommendation': admet['recommendation'],
                'immunogenicity': admet['immunogenicity'],
                'toxicity': admet['toxicity'],
                'stability': admet['stability'],
            },

            # Multi-scale outcome
            'clinical_outcome': multiscale['final_outcome'],

            # Reliability
            'reliability': reliability,

            # Gene signature (if provided)
            'gene_signature': gene_score if gene_signature else None,

            # Recommendation
            'recommendation': self._generate_recommendation(composite, admet, dose_response),
        }

        return report

    def _predict_sequence(self, sequence: str) -> Dict:
        """Sequence-based ML prediction."""
        from .features import CircRNAFeatureExtractor

        extractor = CircRNAFeatureExtractor()
        features = extractor.extract(sequence).reshape(1, -1)

        if self.model:
            # Use scaler if compatible
            if self.scaler and hasattr(self.scaler, 'n_features_in_'):
                if self.scaler.n_features_in_ == features.shape[1]:
                    features = self.scaler.transform(features)

            prediction = self.model.predict(features)[0]

            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = max(proba)
            else:
                confidence = 0.8
        else:
            # Simple heuristic
            seq = sequence.upper().replace('T', 'U')
            gc = sum(1 for c in seq if c in 'GC') / max(len(seq), 1)
            prediction = gc * 0.5 + 0.3
            confidence = 0.6

        level = "High" if prediction > 0.6 else ("Medium" if prediction > 0.4 else "Low")

        return {
            'immunogenicity': float(prediction),
            'confidence': float(confidence),
            'level': level,
        }

    def _calculate_gene_score(self, gene_signature: Optional[Dict]) -> float:
        """Calculate gene signature score."""
        if not gene_signature:
            return 0.5

        # TROP2, NECTIN4, LIV-1, B7-H4, MKI67 are target genes
        target_genes = ['TROP2', 'NECTIN4', 'LIV-1', 'B7-H4', 'MKI67']

        scores = []
        for gene in target_genes:
            if gene in gene_signature:
                val = gene_signature[gene]
                # Normalize (assuming expression values)
                if val > self.config.gene_threshold:
                    scores.append(1.0)
                else:
                    scores.append(0.5)

        return np.mean(scores) if scores else 0.5

    def _calculate_composite(
        self,
        seq_pred: Dict,
        innate: Dict,
        dose: Dict,
        ctm: Dict,
        admet: Dict,
        gene_score: float,
    ) -> Dict:
        """Calculate composite multi-modal score."""

        # Weighted combination
        overall = (
            seq_pred['immunogenicity'] * self.config.sequence_weight +
            innate['overall_score'] * self.config.innate_weight +
            dose['therapeutic_window'] * self.config.dose_weight +
            ctm['max_effect'] * self.config.pkpd_weight +
            (1 if admet['pass'] else 0) * self.config.admet_weight +
            gene_score * self.config.gene_weight
        )

        # Tier classification
        if overall >= 0.7:
            tier = "Tier1_HighPotential"
        elif overall >= 0.5:
            tier = "Tier2_MediumPotential"
        elif overall >= 0.3:
            tier = "Tier3_LowPotential"
        else:
            tier = "Tier4_NotRecommended"

        return {
            'overall': overall,
            'tier': tier,
            'breakdown': {
                'sequence_contribution': seq_pred['immunogenicity'] * self.config.sequence_weight,
                'innate_contribution': innate['overall_score'] * self.config.innate_weight,
                'dose_contribution': dose['therapeutic_window'] * self.config.dose_weight,
                'ctm_contribution': ctm['max_effect'] * self.config.pkpd_weight,
                'admet_contribution': (1 if admet['pass'] else 0) * self.config.admet_weight,
                'gene_contribution': gene_score * self.config.gene_weight,
            },
        }

    def _generate_recommendation(self, composite: Dict, admet: Dict, dose: Dict) -> str:
        """Generate clinical recommendation."""

        tier = composite['tier']

        if tier == "Tier1_HighPotential":
            return f"Highly recommended for therapeutic development. Composite score: {composite['overall']:.2f}"

        elif tier == "Tier2_MediumPotential":
            issues = []
            if not admet['pass']:
                issues.append("ADMET concerns")
            if dose['toxicity'] > 0.5:
                issues.append("toxicity risk")
            return f"Promising candidate. Monitor: {', '.join(issues) if issues else 'dose optimization'}"

        elif tier == "Tier3_LowPotential":
            return f"Low therapeutic potential. Consider sequence optimization. Score: {composite['overall']:.2f}"

        else:
            return f"Not recommended. Major issues: low immunogenicity, safety concerns"


def predict_multimodal(
    sequence: str,
    model_path: str = "confluencia-circrna-encoder/data/models/finetune_xgb.joblib",
    dose: float = 100.0,
    gene_signature: Dict = None,
) -> Dict:
    """
    Quick multi-modal prediction.

    Args:
        sequence: circRNA sequence
        model_path: Trained model path
        dose: Dose in ng/kg
        gene_signature: Gene expression values

    Returns:
        Comprehensive prediction report
    """
    scaler_path = Path(model_path).parent / "finetune_scaler.joblib"
    predictor = MultiModalCircRNAPredictor(model_path, str(scaler_path))
    return predictor.predict_full(sequence, dose, gene_signature)