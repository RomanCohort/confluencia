"""
integration.py — Integration with main Confluencia workflow.

Provides:
1. JointScoringEngine compatibility layer
2. Streamlit UI integration
3. Export to Confluencia Studio
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from confluencia_circrna.pipeline.circrna_pipeline import CircRNAPipeline, CircRNAPipelineResult, CircRNAPipelineConfig
from confluencia_circrna.core.features import CircRNAFeatureSpec, DEFAULT_FEATURE_SPEC


class CircRNAIntegration:
    """
    Integration layer between circRNA module and main Confluencia.

    Usage:
    1. From Streamlit UI: call predict_from_ui()
    2. From JointScoringEngine: call get_circrna_scores()
    3. From Studio: call export_to_studio()
    """

    def __init__(self, config: Optional[CircRNAPipelineConfig] = None):
        self.config = config or CircRNAPipelineConfig()
        self.pipeline = CircRNAPipeline(self.config)
        self.feature_spec = DEFAULT_FEATURE_SPEC

    def predict_from_ui(
        self,
        sequence: str,
        gene_expr_input: Dict[str, float],
        clinical_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process input from Streamlit UI and return formatted results.

        Args:
            sequence: circRNA sequence (ACGU format)
            gene_expr_input: Gene expression values {gene: value}
            clinical_context: Optional clinical data (dose, schedule, etc.)

        Returns:
            Dict compatible with Streamlit display components
        """
        result = self.pipeline.run(
            sequence=sequence,
            gene_expression=gene_expr_input,
            clinical_data=clinical_context,
        )

        return {
            "immune_scores": result.immune_scores,
            "composite_scores": result.composite_scores,
            "uncertainty": result.uncertainty,
            "recommendations": result.recommendations,
            "structure_summary": self._format_structure(result.structure_features),
            "predicted_response": result.composite_scores.get("predicted_response", "intermediate"),
            "confidence": 1.0 - result.uncertainty.get("total", 0.1),
        }

    def get_circrna_scores(
        self,
        sequence: str,
        gene_expr: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Get circRNA scores for JointScoringEngine fusion.

        Returns the 13-key dict expected by _score_circrna():
        {
            "immunotherapy_score": float,
            "tumor_killing_index": float,
            "overall_immunogenicity": float,
            "immune_cycle_score": float,
            "tme_score": float,
            "therapeutic_window": float,
            "tide_score": float,
            "ips": float,
            "rig_i_score": float,
            "tlr_score": float,
            "pkr_score": float,
            "trained_model_risk": float,
            "predicted_response": str,
        }
        """
        result = self.pipeline.run(sequence=sequence, gene_expression=gene_expr)
        return result.composite_scores

    def batch_predict(
        self,
        sequences: List[str],
        gene_exprs: List[Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple circRNA sequences.

        Args:
            sequences: List of circRNA sequences
            gene_exprs: List of gene expression dicts (or single dict for all)

        Returns:
            DataFrame with all scores per sequence
        """
        results = []

        # Handle single gene expression dict for all sequences
        if len(gene_exprs) == 1 and len(sequences) > 1:
            gene_exprs = gene_exprs * len(sequences)

        for i, (seq, genes) in enumerate(zip(sequences, gene_exprs)):
            result = self.pipeline.run(sequence=seq, gene_expression=genes)

            row = {
                "sequence_id": i,
                "sequence_length": len(seq),
            }

            # Add all scores
            row.update(result.immune_scores)
            row.update(result.composite_scores)
            row["uncertainty_total"] = result.uncertainty.get("total", 0)

            if result.structure_features:
                row["mfe"] = result.structure_features.mfe
                row["dsrna_fraction"] = result.structure_features.dsrna_fraction

            results.append(row)

        return pd.DataFrame(results)

    def export_to_studio(
        self,
        result: CircRNAPipelineResult,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export result for Confluencia Studio visualization.

        Args:
            result: Pipeline result
            format: Export format (json, dict)

        Returns:
            Studio-compatible dict
        """
        export_data = {
            "module": "circRNA",
            "version": "v3",
            "scores": {
                "immune": result.immune_scores,
                "composite": {
                    k: v for k, v in result.composite_scores.items()
                    if k != "predicted_response"
                },
            },
            "response_class": result.composite_scores.get("predicted_response"),
            "uncertainty": result.uncertainty,
            "recommendations": result.recommendations,
        }

        if result.structure_features:
            export_data["structure"] = {
                "mfe": result.structure_features.mfe,
                "mfe_normalized": result.structure_features.mfe_normalized,
                "dsrna_fraction": result.structure_features.dsrna_fraction,
                "stability": result.structure_features.structure_stability,
                "hairpins": result.structure_features.hairpin_count,
                "stems": result.structure_features.stem_count,
                "method": result.structure_features.prediction_method,
            }

        return export_data

    def _format_structure(
        self,
        features: Optional[Any],
    ) -> Dict[str, Any]:
        """Format structure features for display."""
        if features is None:
            return {"status": "not_predicted"}

        return {
            "status": "predicted",
            "method": features.prediction_method,
            "mfe_kcal_per_mol": features.mfe,
            "mfe_per_nt": features.mfe_normalized,
            "dsrna_fraction": f"{features.dsrna_fraction:.2%}",
            "stability_score": features.structure_stability,
            "hairpin_count": features.hairpin_count,
            "stem_count": features.stem_count,
        }


# UI helper functions for Streamlit
def create_streamlit_input_template() -> Dict[str, Any]:
    """Create input template for Streamlit UI."""
    return {
        "sequence": "",
        "gene_expression": {
            "TROP2": 7.0,
            "NECTIN4": 5.0,
            "LIV-1": 3.5,
            "B7-H4": 6.0,
            "MKI67": 8.0,
            "MYC": 4.5,
        },
        "options": {
            "enable_structure_prediction": True,
            "compute_mode": "auto",
        },
    }


def format_results_for_display(result_dict: Dict[str, Any]) -> str:
    """Format results for text display in Streamlit."""
    lines = ["## circRNA Analysis Results", ""]

    # Immune scores
    lines.append("### Immune Pathway Scores")
    for key, val in result_dict.get("immune_scores", {}).items():
        if isinstance(val, float):
            lines.append(f"- **{key}**: {val:.3f}")
    lines.append("")

    # Composite scores
    lines.append("### Composite Scores")
    composite = result_dict.get("composite_scores", {})
    important_keys = ["ips", "immunotherapy_score", "tumor_killing_index", "tide_score"]
    for key in important_keys:
        if key in composite:
            lines.append(f"- **{key}**: {composite[key]:.3f}")
    lines.append("")

    # Predicted response
    response = result_dict.get("predicted_response", "intermediate")
    lines.append(f"### Predicted Response: **{response}**")
    lines.append("")

    # Confidence
    confidence = result_dict.get("confidence", 0.9)
    lines.append(f"**Confidence**: {confidence:.2%}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    integration = CircRNAIntegration()

    test_seq = "GCGCGUGUGUACGUACGUACGUACGU" * 20
    test_genes = {"TROP2": 8.0, "MKI67": 6.0}

    print("CircRNA Integration Demo")
    print("=" * 60)

    # Test UI integration
    ui_result = integration.predict_from_ui(test_seq, test_genes)
    print("\nUI Result:")
    print(format_results_for_display(ui_result))

    # Test 13-key dict
    scores = integration.get_circrna_scores(test_seq, test_genes)
    print("\n13-key Dict for JointScoringEngine:")
    for key, val in scores.items():
        print(f"  {key}: {val if isinstance(val, str) else f'{val:.3f}'}")