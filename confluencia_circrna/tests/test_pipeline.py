"""
test_pipeline.py — Tests for circRNA analysis pipeline.

Tests:
1. Pipeline initialization
2. Complete analysis flow
3. Composite score calculation
4. Recommendation generation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add module path
_CIRCRNA_PATH = Path(__file__).parent.parent
if str(_CIRCRNA_PATH) not in sys.path:
    sys.path.insert(0, str(_CIRCRNA_PATH))

from confluencia_circrna.pipeline.circrna_pipeline import (
    CircRNAPipeline,
    CircRNAPipelineConfig,
    CircRNAPipelineResult,
    run_pipeline,
)


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_default_init(self):
        """Test default configuration."""
        pipeline = CircRNAPipeline()
        assert pipeline.config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircRNAPipelineConfig(
            enable_structure_prediction=False,
            compute_mode="high",
        )
        pipeline = CircRNAPipeline(config)

        assert not pipeline.config.enable_structure_prediction
        assert pipeline.config.compute_mode == "high"


class TestPipelineRun:
    """Tests for pipeline execution."""

    def test_basic_run(self):
        """Test basic pipeline run."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGUACGUACGUACGUACGUACGUACGU"
        result = pipeline.run(sequence=seq)

        assert isinstance(result, CircRNAPipelineResult)
        assert result.immune_scores is not None
        assert result.composite_scores is not None

    def test_with_gene_expression(self):
        """Test with gene expression input."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20
        genes = {"TROP2": 8.0, "MKI67": 6.0}

        result = pipeline.run(sequence=seq, gene_expression=genes)

        assert "ips" in result.composite_scores
        assert result.composite_scores["ips"] > 0

    def test_without_structure_prediction(self):
        """Test disabling structure prediction."""
        config = CircRNAPipelineConfig(enable_structure_prediction=False)
        pipeline = CircRNAPipeline(config)

        seq = "ACGUACGU" * 20
        result = pipeline.run(sequence=seq)

        assert result.structure_features is None
        assert result.prediction_method == "rule_based"

    def test_with_structure_prediction(self):
        """Test with structure prediction enabled."""
        config = CircRNAPipelineConfig(enable_structure_prediction=True)
        pipeline = CircRNAPipeline(config)

        seq = "ACGUACGU" * 50
        result = pipeline.run(sequence=seq)

        # Structure features may or may not be present depending on ViennaRNA
        if result.structure_features:
            assert result.structure_features.mfe is not None


class TestCompositeScores:
    """Tests for composite score calculation."""

    def test_all_13_keys_present(self):
        """Verify all 13 keys are present."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20
        genes = {"TROP2": 7.0, "MKI67": 6.0}

        result = pipeline.run(sequence=seq, gene_expression=genes)

        expected_keys = [
            "rig_i_score",
            "tlr_score",
            "pkr_score",
            "overall_immunogenicity",
            "immunotherapy_score",
            "tumor_killing_index",
            "therapeutic_window",
            "tide_score",
            "ips",
            "immune_cycle_score",
            "tme_score",
            "trained_model_risk",
            "predicted_response",
        ]

        for key in expected_keys:
            assert key in result.composite_scores

    def test_ips_range(self):
        """IPS should be in [0, 10] range."""
        pipeline = CircRNAPipeline()

        for _ in range(5):
            seq = "ACGU" * np.random.randint(10, 50)
            genes = {
                "TROP2": np.random.uniform(2, 12),
                "MKI67": np.random.uniform(4, 14),
            }
            result = pipeline.run(sequence=seq, gene_expression=genes)

            ips = result.composite_scores["ips"]
            assert 0.0 <= ips <= 10.0

    def test_score_ranges(self):
        """All numeric scores should be in valid ranges."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20
        genes = {"TROP2": 7.0}

        result = pipeline.run(sequence=seq, gene_expression=genes)

        for key, val in result.composite_scores.items():
            if key == "ips":
                assert 0.0 <= val <= 10.0
            elif key == "predicted_response":
                assert val in ["likely_responder", "intermediate", "likely_non_responder"]
            else:
                assert 0.0 <= val <= 1.0


class TestResponseClassification:
    """Tests for response classification."""

    def test_responder_classification(self):
        """High IPS + low TIDE should be responder."""
        pipeline = CircRNAPipeline()

        # Favorable conditions
        seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10  # High GC = high RIG-I
        genes = {"TROP2": 12.0, "B7-H4": 10.0}  # High target expression

        result = pipeline.run(sequence=seq, gene_expression=genes)

        # Should be favorable ( responder or intermediate)
        assert result.composite_scores["predicted_response"] in ["likely_responder", "intermediate"]

    def test_non_responder_classification(self):
        """Low IPS + high TIDE should be non-responder."""
        pipeline = CircRNAPipeline()

        # Unfavorable conditions
        seq = "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAU" * 10  # Low GC
        genes = {"TROP2": 2.0, "B7-H4": 2.0}  # Low target expression

        result = pipeline.run(sequence=seq, gene_expression=genes)

        # Should be unfavorable
        assert result.composite_scores["predicted_response"] in ["likely_non_responder", "intermediate"]


class TestUncertainty:
    """Tests for uncertainty quantification."""

    def test_uncertainty_keys(self):
        """Verify uncertainty keys."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20

        result = pipeline.run(sequence=seq)

        expected_keys = ["structure", "gene_expression", "immune_scores", "total"]
        for key in expected_keys:
            assert key in result.uncertainty

    def test_uncertainty_range(self):
        """Uncertainty should be in [0, 1]."""
        pipeline = CircRNAPipeline()

        for _ in range(5):
            seq = "ACGU" * np.random.randint(10, 50)
            result = pipeline.run(sequence=seq)

            for key, val in result.uncertainty.items():
                assert 0.0 <= val <= 1.0


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_present(self):
        """Recommendations should always be generated."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20

        result = pipeline.run(sequence=seq)
        assert len(result.recommendations) > 0

    def test_high_rig_i_recommendation(self):
        """High RIG-I should have specific recommendation."""
        pipeline = CircRNAPipeline()
        seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10  # High GC

        result = pipeline.run(sequence=seq)

        # Should mention RIG-I
        rig_i_rec = [r for r in result.recommendations if "RIG-I" in r]
        assert len(rig_i_rec) > 0

    def test_conclusion_present(self):
        """Recommendations should include conclusion."""
        pipeline = CircRNAPipeline()
        seq = "ACGUACGU" * 20

        result = pipeline.run(sequence=seq)

        conclusion = [r for r in result.recommendations if "CONCLUSION" in r]
        assert len(conclusion) > 0


class TestConvenienceFunction:
    """Tests for run_pipeline convenience function."""

    def test_convenience_function(self):
        """Test quick pipeline execution."""
        seq = "ACGUACGU" * 20
        genes = {"TROP2": 7.0}

        result = run_pipeline(sequence=seq, gene_expression=genes, enable_structure=False)

        assert isinstance(result, CircRNAPipelineResult)
        assert result.structure_features is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])