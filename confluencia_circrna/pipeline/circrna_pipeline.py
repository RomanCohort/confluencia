"""
circrna_pipeline.py — Complete circRNA analysis pipeline.

Provides unified interface for:
1. Sequence feature extraction (immune sensing + structure)
2. Gene expression integration
3. Scoring prediction
4. Integration with main Confluencia workflow

Usage:
    from confluencia_circrna.pipeline import CircRNAPipeline

    pipeline = CircRNAPipeline()
    result = pipeline.run(sequence="ACGU...", gene_expression={"TROP2": 7.0})
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path

from confluencia_circrna.core.features import CircRNAFeatureSpec, DEFAULT_FEATURE_SPEC
from confluencia_circrna.core.immune_sensing import predict_circrna_immunogenicity, ImmuneSensingConfig
from confluencia_circrna.core.structure_prediction import StructurePredictor, StructureFeatures, compute_pkr_score_from_structure
from confluencia_circrna.training.data_loader import normalize_gene_expression, GeneNormalizationConfig


# Load literature weights
_WEIGHTS_FILE = Path(__file__).parent.parent / "data" / "reference" / "scoring_weights_literature.json"


@dataclass
class CircRNAPipelineConfig:
    """Pipeline configuration."""
    enable_structure_prediction: bool = True
    use_literature_weights: bool = True
    min_dsrna_length: int = 33
    compute_mode: str = "auto"  # auto, low, medium, high
    gene_normalization_method: str = "minmax"


@dataclass
class CircRNAPipelineResult:
    """Pipeline output result."""
    # Immune pathway scores
    immune_scores: Dict[str, float]

    # Structure features
    structure_features: Optional[StructureFeatures]

    # Composite scores (13-key dict)
    composite_scores: Dict[str, float]

    # Uncertainty estimates
    uncertainty: Dict[str, float]

    # Design recommendations
    recommendations: List[str]

    # Metadata
    prediction_method: str = "rule_based"
    feature_spec_version: str = ""


class CircRNAPipeline:
    """
    circRNA analysis pipeline.

    Main entry point for circRNA immunogenicity analysis.

    Example:
        pipeline = CircRNAPipeline()
        result = pipeline.run(
            sequence="ACGUACGUACGU...",
            gene_expression={"TROP2": 7.0, "MKI67": 8.0}
        )
        print(result.composite_scores["ips"])
    """

    def __init__(self, config: Optional[CircRNAPipelineConfig] = None):
        self.config = config or CircRNAPipelineConfig()
        self.weights = self._load_weights()

        # Initialize components
        self.structure_predictor = None
        if self.config.enable_structure_prediction:
            self.structure_predictor = StructurePredictor(
                min_dsrna_length=self.config.min_dsrna_length
            )

        self.gene_norm_config = GeneNormalizationConfig(
            method=self.config.gene_normalization_method
        )

    def _load_weights(self) -> Dict[str, Any]:
        """Load scoring weights from config file."""
        if _WEIGHTS_FILE.exists():
            try:
                with open(_WEIGHTS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def run(
        self,
        sequence: str,
        gene_expression: Optional[Dict[str, float]] = None,
        clinical_data: Optional[Dict[str, Any]] = None,
    ) -> CircRNAPipelineResult:
        """
        Run complete circRNA analysis.

        Args:
            sequence: circRNA nucleotide sequence (ACGU format)
            gene_expression: Gene expression values {gene_name: value}
            clinical_data: Clinical context (dose, schedule, etc.)

        Returns:
            CircRNAPipelineResult with all scores and recommendations
        """
        # Step 1: Immune sensing (sequence-based)
        immune_config = ImmuneSensingConfig()
        immune_scores = predict_circrna_immunogenicity(sequence, immune_config)

        # Step 2: Structure prediction (optional)
        structure_features = None
        if self.structure_predictor:
            structure_features = self.structure_predictor.predict(sequence)
            # Update PKR score with real dsRNA data
            if structure_features.dsrna_fraction > 0:
                pkr_from_structure = compute_pkr_score_from_structure(structure_features)
                immune_scores["pkr_score"] = pkr_from_structure
                immune_scores["pkr_method"] = "structure_based"

        # Step 3: Gene expression integration
        norm_genes = {}
        if gene_expression:
            norm_genes = normalize_gene_expression(gene_expression, self.gene_norm_config)

        # Step 4: Compute composite scores
        composite_scores = self._compute_composite_scores(
            immune_scores=immune_scores,
            gene_expression=norm_genes,
            structure_features=structure_features,
        )

        # Step 5: Uncertainty quantification
        uncertainty = self._compute_uncertainty(
            immune_scores, norm_genes, structure_features
        )

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            immune_scores, composite_scores, structure_features, norm_genes
        )

        return CircRNAPipelineResult(
            immune_scores=immune_scores,
            structure_features=structure_features,
            composite_scores=composite_scores,
            uncertainty=uncertainty,
            recommendations=recommendations,
            prediction_method="rule_based" if not self.structure_predictor else "enhanced",
            feature_spec_version=DEFAULT_FEATURE_SPEC.schema_id(),
        )

    def _compute_composite_scores(
        self,
        immune_scores: Dict[str, float],
        gene_expression: Dict[str, float],
        structure_features: Optional[StructureFeatures],
    ) -> Dict[str, float]:
        """
        Compute composite scores based on literature weights.

        Returns 13-key dict compatible with JointScoringEngine.
        """
        scores = {}

        # Load weights
        weights = self.weights.get("composite_score_weights", {})

        # Individual pathway scores (already computed)
        scores["rig_i_score"] = immune_scores.get("rig_i_score", 0.0)
        scores["tlr_score"] = immune_scores.get("tlr_score", 0.0)
        scores["pkr_score"] = immune_scores.get("pkr_score", 0.0)

        # Overall immunogenicity
        overall_weights = self.weights.get("overall_pathway_weights", {
            "rig_i": 0.40, "tlr": 0.35, "pkr": 0.25
        })
        overall = (
            scores["rig_i_score"] * overall_weights.get("rig_i", 0.40) +
            scores["tlr_score"] * overall_weights.get("tlr", 0.35) +
            scores["pkr_score"] * overall_weights.get("pkr", 0.25)
        )
        scores["overall_immunogenicity"] = overall

        # Gene expression dependent scores
        trop2 = gene_expression.get("TROP2", 0.5)
        nectin4 = gene_expression.get("NECTIN4", 0.5)
        b7h4 = gene_expression.get("B7-H4", 0.5)
        mki67 = gene_expression.get("MKI67", 0.5)
        myc = gene_expression.get("MYC", 0.5)

        # TIDE score (inverse: high expression = low evasion)
        tide = np.clip(0.6 - 0.2 * trop2 - 0.15 * b7h4 - 0.1 * nectin4, 0.0, 1.0)
        scores["tide_score"] = tide

        # IPS (Immunotherapy Potential Score) [0, 10]
        ips = np.clip(
            3.0 + 2.0 * b7h4 + 1.5 * trop2 + overall * 2.0,
            0.0, 10.0
        )
        scores["ips"] = ips

        # Immunotherapy score
        imm_weights = weights.get("immunotherapy_score", {})
        immunotherapy = (
            overall * imm_weights.get("immunogenicity", 0.30) +
            (1 - tide) * imm_weights.get("tide_inverse", 0.25) +
            (ips / 10) * imm_weights.get("ips_fraction", 0.25) +
            (b7h4 + trop2) / 2 * imm_weights.get("immune_cycle", 0.20)
        )
        scores["immunotherapy_score"] = immunotherapy

        # Tumor killing index
        tk_weights = weights.get("tumor_killing_index", {})
        tumor_killing = (
            (1 - tide) * tk_weights.get("immune_cycle", 0.35) +
            mki67 * tk_weights.get("mki67_inverse", 0.25) +
            overall * tk_weights.get("immunogenicity", 0.25) +
            (ips / 10) * tk_weights.get("therapeutic_window", 0.15)
        )
        scores["tumor_killing_index"] = tumor_killing

        # Therapeutic window
        tw_weights = weights.get("therapeutic_window", {})
        therapeutic_window = (
            (1 - tide) * tw_weights.get("tide_inverse", 0.35) +
            (ips / 10) * tw_weights.get("ips_fraction", 0.30) +
            overall * tw_weights.get("immunogenicity", 0.20) +
            b7h4 * tw_weights.get("tme_score", 0.15)
        )
        scores["therapeutic_window"] = therapeutic_window

        # Immune cycle score
        scores["immune_cycle_score"] = overall * 0.6 + (ips / 10) * 0.4

        # TME score
        scores["tme_score"] = (b7h4 + nectin4) / 2

        # Trained model risk (inverse of immunotherapy score)
        scores["trained_model_risk"] = 1.0 - immunotherapy

        # Predicted response
        scores["predicted_response"] = self._classify_response(scores)

        return scores

    def _classify_response(self, scores: Dict[str, float]) -> str:
        """Classify predicted response based on thresholds."""
        thresholds = self.weights.get("response_classification_thresholds", {})

        ips = scores.get("ips", 0)
        tide = scores.get("tide_score", 0.5)
        immunogenicity = scores.get("overall_immunogenicity", 0.5)

        responder_thresholds = thresholds.get("likely_responder", {})
        non_responder_thresholds = thresholds.get("likely_non_responder", {})

        # Check responder conditions
        if (
            ips >= responder_thresholds.get("ips_min", 7.0) and
            tide <= responder_thresholds.get("tide_max", 0.3)
        ):
            return "likely_responder"

        # Check non-responder conditions
        if (
            ips <= non_responder_thresholds.get("ips_max", 3.0) or
            tide >= non_responder_thresholds.get("tide_min", 0.6)
        ):
            return "likely_non_responder"

        return "intermediate"

    def _compute_uncertainty(
        self,
        immune_scores: Dict[str, float],
        gene_expression: Dict[str, float],
        structure_features: Optional[StructureFeatures],
    ) -> Dict[str, float]:
        """
        Compute uncertainty estimates for predictions.

        Higher uncertainty when:
        - No structure prediction available
        - Gene expression values outside reference range
        - Scores at extreme values (close to 0 or 1)
        """
        uncertainty = {}

        # Structure prediction uncertainty
        if structure_features is None:
            uncertainty["structure"] = 0.3  # Higher uncertainty without structure
        else:
            uncertainty["structure"] = 0.1 if structure_features.prediction_method == "viennarna" else 0.2

        # Gene expression uncertainty
        gene_uncertainty = []
        for gene, val in gene_expression.items():
            if val < 0.1 or val > 0.9:
                gene_uncertainty.append(0.15)  # Extreme values
            else:
                gene_uncertainty.append(0.05)
        uncertainty["gene_expression"] = np.mean(gene_uncertainty) if gene_uncertainty else 0.1

        # Score uncertainty (based on score extremity)
        score_uncertainty = []
        for key in ["rig_i_score", "tlr_score", "pkr_score"]:
            val = immune_scores.get(key, 0.5)
            # Scores near 0.5 are more certain, extremes are less certain
            score_uncertainty.append(abs(val - 0.5) * 0.1)
        uncertainty["immune_scores"] = np.mean(score_uncertainty)

        # Total uncertainty
        uncertainty["total"] = (
            uncertainty["structure"] +
            uncertainty["gene_expression"] +
            uncertainty["immune_scores"]
        ) / 3

        return uncertainty

    def _generate_recommendations(
        self,
        immune_scores: Dict[str, float],
        composite_scores: Dict[str, float],
        structure_features: Optional[StructureFeatures],
        gene_expression: Dict[str, float],
    ) -> List[str]:
        """Generate design recommendations based on analysis."""
        recommendations = []

        # RIG-I recommendations
        rig_i = immune_scores.get("rig_i_score", 0.5)
        if rig_i > 0.6:
            recommendations.append("High RIG-I activation potential - strong innate immune response")
        elif rig_i < 0.3:
            recommendations.append("Low RIG-I score - consider adding GU-rich motifs at 5' end")

        # TLR recommendations
        tlr = immune_scores.get("tlr_score", 0.5)
        if tlr > 0.6:
            recommendations.append("High TLR7/8 activation - favorable for DC maturation")
        elif tlr < 0.3:
            recommendations.append("Low TLR score - consider increasing uridine content")

        # PKR recommendations
        pkr = immune_scores.get("pkr_score", 0.5)
        if pkr > 0.6:
            recommendations.append("WARNING: High PKR activation may inhibit protein expression")
            if structure_features:
                recommendations.append(f"  Consider reducing dsRNA regions (found {len(structure_features.dsrna_regions)} regions)")
        elif pkr < 0.3:
            recommendations.append("Low PKR score - minimal translation inhibition risk")

        # Structure recommendations
        if structure_features:
            if structure_features.structure_stability > 0.7:
                recommendations.append("High structure stability - may enhance immunogenicity")
            elif structure_features.structure_stability < 0.3:
                recommendations.append("Low structure stability - consider stabilizing modifications")

        # Gene expression recommendations
        for gene, val in gene_expression.items():
            if gene == "MKI67" and val > 0.8:
                recommendations.append("High MKI67 expression - high proliferation, may need higher dose")

        # Overall assessment
        ips = composite_scores.get("ips", 0)
        if ips >= 7.0:
            recommendations.append("CONCLUSION: Likely responder - recommend proceeding with design")
        elif ips < 3.0:
            recommendations.append("CONCLUSION: Likely non-responder - consider sequence redesign")
        else:
            recommendations.append("CONCLUSION: Intermediate response - further optimization recommended")

        return recommendations


def run_pipeline(
    sequence: str,
    gene_expression: Optional[Dict[str, float]] = None,
    enable_structure: bool = True,
) -> CircRNAPipelineResult:
    """Convenience function for quick pipeline execution."""
    config = CircRNAPipelineConfig(enable_structure_prediction=enable_structure)
    pipeline = CircRNAPipeline(config)
    return pipeline.run(sequence, gene_expression)


if __name__ == "__main__":
    # Demo
    test_sequence = "ACGUACGUACGUACGU" * 20  # 320 nt
    test_genes = {"TROP2": 8.0, "NECTIN4": 5.5, "B7-H4": 7.0, "MKI67": 6.0, "MYC": 4.0}

    print("CircRNA Pipeline Demo")
    print("=" * 60)

    pipeline = CircRNAPipeline()
    result = pipeline.run(sequence=test_sequence, gene_expression=test_genes)

    print("\nImmune Scores:")
    for key, val in result.immune_scores.items():
        print(f"  {key}: {val:.4f}")

    print("\nComposite Scores:")
    for key, val in result.composite_scores.items():
        if key != "predicted_response":
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print("\nUncertainty:")
    for key, val in result.uncertainty.items():
        print(f"  {key}: {val:.3f}")

    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")