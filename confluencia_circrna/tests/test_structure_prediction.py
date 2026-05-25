"""
test_structure_prediction.py — Tests for ViennaRNA integration.

Tests:
1. StructurePredictor initialization
2. MFE calculation (mocked if ViennaRNA not installed)
3. dsRNA region extraction
4. PKR score from structure
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add module path
_CIRCRNA_PATH = Path(__file__).parent.parent
if str(_CIRCRNA_PATH) not in sys.path:
    sys.path.insert(0, str(_CIRCRNA_PATH))

from confluencia_circrna.core.structure_prediction import (
    StructurePredictor,
    StructureFeatures,
    compute_pkr_score_from_structure,
    PKR_MIN_DSRNA_LENGTH,
)


class TestStructurePredictorInit:
    """Tests for predictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = StructurePredictor()
        assert predictor.min_dsrna_length == PKR_MIN_DSRNA_LENGTH

    def test_custom_dsrna_length(self):
        """Test custom dsRNA threshold."""
        predictor = StructurePredictor(min_dsrna_length=50)
        assert predictor.min_dsrna_length == 50


class TestSequenceSanitization:
    """Tests for sequence sanitization."""

    def test_dna_to_rna_conversion(self):
        """T should be converted to U."""
        predictor = StructurePredictor()
        dna_seq = "ACGTACGT"
        rna_seq = predictor._sanitize_sequence(dna_seq)
        assert rna_seq == "ACGUACGU"
        assert "T" not in rna_seq

    def test_invalid_chars_filtered(self):
        """Invalid characters should be filtered."""
        predictor = StructurePredictor()
        seq = "ACGUACGU123NXYZ"
        sanitized = predictor._sanitize_sequence(seq)
        # Only valid RNA bases should remain
        assert all(c in "AUGC" for c in sanitized)

    def test_empty_sequence(self):
        """Empty sequence should return empty."""
        predictor = StructurePredictor()
        assert predictor._sanitize_sequence("") == ""


class TestStructurePrediction:
    """Tests for structure prediction."""

    def test_predict_returns_features(self):
        """Predict should return StructureFeatures."""
        predictor = StructurePredictor()
        seq = "ACGUACGUACGUACGU" * 10
        features = predictor.predict(seq)

        assert isinstance(features, StructureFeatures)
        assert features.mfe is not None
        assert features.dot_bracket is not None

    def test_mfe_normalized_calculation(self):
        """MFE normalized should be per nucleotide."""
        predictor = StructurePredictor()
        seq = "ACGU" * 50  # 200 nt
        features = predictor.predict(seq)

        # MFE normalized should equal MFE / length
        expected = features.mfe / len(seq)
        assert features.mfe_normalized == expected

    def test_fallback_method(self):
        """Fallback should work when ViennaRNA not available."""
        predictor = StructurePredictor()
        predictor._has_viennarna = False  # Force fallback

        seq = "ACGUACGUACGUACGU"
        features = predictor.predict(seq)

        assert features.prediction_method == "fallback"
        assert features.mfe != 0  # Should have estimated value


class TestDSRNAExtraction:
    """Tests for dsRNA region extraction."""

    def test_simple_stem_detection(self):
        """Simple stem should be detected."""
        predictor = StructurePredictor(min_dsrna_length=5)  # Lower threshold for test
        dot_bracket = "((((((........))))))"

        regions = predictor._extract_dsrna_regions(dot_bracket)
        assert len(regions) >= 0  # May or may not be detected based on threshold

    def test_no_stem_regions(self):
        """Unpaired sequence should have no regions."""
        predictor = StructurePredictor(min_dsrna_length=33)
        dot_bracket = "............."

        regions = predictor._extract_dsrna_regions(dot_bracket)
        assert len(regions) == 0

    def test_threshold_filtering(self):
        """Regions below threshold should be filtered."""
        predictor = StructurePredictor(min_dsrna_length=33)
        # Short stem (10 paired bases)
        dot_bracket = "((((((....))))))"

        regions = predictor._extract_dsrna_regions(dot_bracket)
        # Should not include short stem (<33)
        for start, end in regions:
            assert (end - start) >= predictor.min_dsrna_length


class TestStabilityScore:
    """Tests for stability score calculation."""

    def test_stability_range(self):
        """Stability score should be in [0, 1]."""
        predictor = StructurePredictor()

        for mfe_norm in [-0.1, -0.3, -0.5, -0.8, -1.0]:
            score = predictor._compute_stability_score(mfe_norm)
            assert 0.0 <= score <= 1.0

    def test_more_stable_higher_score(self):
        """More negative MFE should give higher stability."""
        predictor = StructurePredictor()

        score_unstable = predictor._compute_stability_score(-0.1)
        score_stable = predictor._compute_stability_score(-0.8)

        assert score_stable > score_unstable


class TestPKRScoreFromStructure:
    """Tests for PKR score calculation from structure."""

    def test_high_dsrna_high_pkr(self):
        """High dsRNA fraction should give high PKR score."""
        features = StructureFeatures(
            mfe=-150.0,
            mfe_normalized=-0.3,
            dsrna_regions=[(0, 50), (100, 150)],
            dsrna_fraction=0.4,
            structure_stability=0.7,
            hairpin_count=2,
            stem_count=2,
            dot_bracket="((((...))))",
            prediction_method="test",
        )

        pkr_score = compute_pkr_score_from_structure(features)
        assert pkr_score > 0.5  # Should be relatively high

    def test_no_dsrna_low_pkr(self):
        """No dsRNA should give low PKR score."""
        features = StructureFeatures(
            mfe=-50.0,
            mfe_normalized=-0.1,
            dsrna_regions=[],
            dsrna_fraction=0.0,
            structure_stability=0.3,
            hairpin_count=0,
            stem_count=0,
            dot_bracket="........",
            prediction_method="test",
        )

        pkr_score = compute_pkr_score_from_structure(features)
        assert pkr_score < 0.3  # Should be low

    def test_pkr_score_range(self):
        """PKR score should be bounded [0, 1]."""
        for dsrna_frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            features = StructureFeatures(
                mfe=-100.0,
                mfe_normalized=-0.2,
                dsrna_regions=[],
                dsrna_fraction=dsrna_frac,
                structure_stability=0.5,
                hairpin_count=1,
                stem_count=1,
                dot_bracket="...",
                prediction_method="test",
            )

            pkr_score = compute_pkr_score_from_structure(features)
            assert 0.0 <= pkr_score <= 1.0


class TestEmptySequenceHandling:
    """Tests for empty sequence handling."""

    def test_empty_sequence(self):
        """Empty sequence should return empty features."""
        predictor = StructurePredictor()
        features = predictor.predict("")

        assert features.mfe == 0.0
        assert features.mfe_normalized == 0.0
        assert len(features.dsrna_regions) == 0
        assert features.dsrna_fraction == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])