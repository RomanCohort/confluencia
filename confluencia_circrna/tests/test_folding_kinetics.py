#!/usr/bin/env python
"""
test_folding_kinetics.py — Test RNA folding kinetics prediction.

Usage:
    pytest confluencia_circrna/tests/test_folding_kinetics.py -v
"""

import pytest
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from confluencia_circrna.core.folding_kinetics import (
    FoldingKineticsPredictor,
    KineticsFeatures,
    SuboptimalStructure,
    predict_folding_kinetics,
    compute_kinetics_score,
)


class TestFoldingKineticsInit:
    """Test predictor initialization."""

    def test_default_init(self):
        """Default initialization should work."""
        predictor = FoldingKineticsPredictor()
        assert predictor.energy_window == 10.0
        assert predictor.max_suboptimal == 100

    def test_custom_params(self):
        """Custom parameters should be set."""
        predictor = FoldingKineticsPredictor(
            suboptimal_energy_window=15.0,
            max_suboptimal=50,
        )
        assert predictor.energy_window == 15.0
        assert predictor.max_suboptimal == 50


class TestSequenceSanitization:
    """Test sequence sanitization."""

    def test_dna_to_rna(self):
        """DNA T should convert to RNA U."""
        predictor = FoldingKineticsPredictor()
        result = predictor._sanitize_sequence("ACGTACGT")
        assert result == "ACGUACGU"

    def test_invalid_chars_filtered(self):
        """Invalid characters should be filtered."""
        predictor = FoldingKineticsPredictor()
        result = predictor._sanitize_sequence("ACGTXNACGT")
        assert result == "ACGUACGU"

    def test_empty_sequence(self):
        """Empty sequence should return empty."""
        predictor = FoldingKineticsPredictor()
        result = predictor._sanitize_sequence("")
        assert result == ""


class TestKineticsPrediction:
    """Test kinetics prediction."""

    def test_predict_returns_features(self):
        """Prediction should return KineticsFeatures."""
        predictor = FoldingKineticsPredictor()
        seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10
        result = predictor.predict(seq)

        assert isinstance(result, KineticsFeatures)
        assert result.kinetics_method in ["viennarna_kinetics", "fallback_kinetics"]
        assert 0.0 <= result.folding_rate <= 1e6
        assert 0.0 <= result.landscape_complexity <= 1.0

    def test_short_sequence_handling(self):
        """Short sequences should return empty features."""
        predictor = FoldingKineticsPredictor()
        result = predictor.predict("ACGU")

        assert result.kinetics_method == "sequence_too_short"
        assert result.folding_rate == 0.0

    def test_high_gc_sequence(self):
        """High GC sequences should have higher barriers."""
        predictor = FoldingKineticsPredictor()
        high_gc = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10
        low_gc = "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAU" * 10

        result_high = predictor.predict(high_gc)
        result_low = predictor.predict(low_gc)

        # High GC should have higher barrier (more stable)
        assert result_high.barrier_height >= result_low.barrier_height


class TestSuboptimalEstimation:
    """Test suboptimal structure estimation (fallback)."""

    def test_estimate_returns_list(self):
        """Estimation should return structure list."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGUACGU" * 10
        result = predictor._estimate_suboptimal(seq)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(s, SuboptimalStructure) for s in result)

    def test_native_structure_first(self):
        """Native (MFE) structure should be first."""
        predictor = FoldingKineticsPredictor()
        seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 5
        result = predictor._estimate_suboptimal(seq)

        assert result[0].energy_delta == 0.0
        assert result[0].probability >= result[1].probability if len(result) > 1 else True


class TestBarrierEstimation:
    """Test energy barrier estimation."""

    def test_barrier_range(self):
        """Barrier should be in reasonable range."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGUACGU" * 20

        barrier = predictor._estimate_barrier(seq)
        assert 0.0 <= barrier <= 30.0

    def test_high_gc_higher_barrier(self):
        """High GC should give higher barrier."""
        predictor = FoldingKineticsPredictor()

        high_gc = "GCGCGCGCGCGCGCGC" * 20
        low_gc = "AUAUAUAUAUAUAUAU" * 20

        barrier_high = predictor._estimate_barrier(high_gc)
        barrier_low = predictor._estimate_barrier(low_gc)

        assert barrier_high > barrier_low


class TestFoldingRate:
    """Test folding rate computation."""

    def test_rate_with_barrier(self):
        """Rate should decrease with higher barrier."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGU" * 10

        rate_low = predictor._compute_folding_rate(seq, 5.0)
        rate_high = predictor._compute_folding_rate(seq, 15.0)

        assert rate_low > rate_high

    def test_rate_range(self):
        """Rate should be in typical RNA folding range."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGU" * 20

        rate = predictor._compute_folding_rate(seq, 10.0)
        assert 1e-3 <= rate <= 1e6


class TestLandscapeComplexity:
    """Test landscape complexity computation."""

    def test_complexity_range(self):
        """Complexity should be in [0, 1]."""
        predictor = FoldingKineticsPredictor()

        structures = [
            SuboptimalStructure(mfe=-10.0, dot_bracket="((...))", energy_delta=0.0, probability=0.7),
            SuboptimalStructure(mfe=-8.0, dot_bracket=".(...).", energy_delta=2.0, probability=0.2),
            SuboptimalStructure(mfe=-6.0, dot_bracket=".....", energy_delta=4.0, probability=0.1),
        ]

        complexity = predictor._compute_landscape_complexity(structures)
        assert 0.0 <= complexity <= 1.0

    def test_more_structures_higher_complexity(self):
        """More structures should give higher complexity."""
        predictor = FoldingKineticsPredictor()

        few_structures = [
            SuboptimalStructure(mfe=-10.0, dot_bracket="((...))", energy_delta=0.0, probability=0.9),
            SuboptimalStructure(mfe=-8.0, dot_bracket=".(...).", energy_delta=2.0, probability=0.1),
        ]

        many_structures = few_structures + [
            SuboptimalStructure(mfe=-7.0, dot_bracket="....", energy_delta=3.0, probability=0.05),
            SuboptimalStructure(mfe=-6.0, dot_bracket="...", energy_delta=4.0, probability=0.03),
        ]

        complexity_few = predictor._compute_landscape_complexity(few_structures)
        complexity_many = predictor._compute_landscape_complexity(many_structures)

        assert complexity_many >= complexity_few


class TestCotransScore:
    """Test cotranscriptional folding score."""

    def test_cotrans_range(self):
        """Cotrans score should be in [0, 1]."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGU" * 10

        structures = predictor._estimate_suboptimal(seq)
        score = predictor._compute_cotrans_score(seq, structures)

        assert 0.0 <= score <= 1.0

    def test_low_gc_better_cotrans(self):
        """Low GC sequences should have higher cotrans score."""
        predictor = FoldingKineticsPredictor()

        high_gc = "GCGCGCGCGCGCGCGC" * 10
        low_gc = "AUAUAUAUAUAUAUAU" * 10

        struct_high = predictor._estimate_suboptimal(high_gc)
        struct_low = predictor._estimate_suboptimal(low_gc)

        score_high = predictor._compute_cotrans_score(high_gc, struct_high)
        score_low = predictor._compute_cotrans_score(low_gc, struct_low)

        assert score_low >= score_high  # Lower GC = easier cotrans folding


class TestDynamicStability:
    """Test dynamic stability computation."""

    def test_stability_range(self):
        """Stability should be in [0, 1]."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGU" * 10

        stability = predictor._compute_dynamic_stability(seq, 10.0, 0.3)
        assert 0.0 <= stability <= 1.0

    def test_high_barrier_more_stable(self):
        """Higher barrier should give higher stability."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGU" * 10

        stability_low = predictor._compute_dynamic_stability(seq, 5.0, 0.3)
        stability_high = predictor._compute_dynamic_stability(seq, 15.0, 0.3)

        assert stability_high >= stability_low


class TestKineticsScore:
    """Test kinetics score computation."""

    def test_score_dict_keys(self):
        """Score dict should have expected keys."""
        features = KineticsFeatures(
            folding_rate=100.0,
            barrier_height=10.0,
            metastable_count=5,
            landscape_complexity=0.3,
            cotrans_folding_score=0.5,
            stability_dynamic=0.6,
            suboptimal_structures=[],
            native_structure=None,
            kinetics_method="test",
        )

        scores = compute_kinetics_score(features)

        expected_keys = [
            "folding_rate_score",
            "barrier_score",
            "landscape_score",
            "cotrans_score",
            "dynamic_stability",
            "metastable_count",
            "immune_exposure_potential",
        ]

        for key in expected_keys:
            assert key in scores

    def test_score_ranges(self):
        """All scores should be in [0, 1]."""
        features = KineticsFeatures(
            folding_rate=1000.0,
            barrier_height=10.0,
            metastable_count=5,
            landscape_complexity=0.5,
            cotrans_folding_score=0.5,
            stability_dynamic=0.5,
            suboptimal_structures=[],
            native_structure=None,
            kinetics_method="test",
        )

        scores = compute_kinetics_score(features)

        for key, value in scores.items():
            if key != "metastable_count":
                assert 0.0 <= value <= 1.0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_predict_folding_kinetics(self):
        """Convenience function should work."""
        seq = "GCGCGCGCGCGCGCGC" * 5
        result = predict_folding_kinetics(seq)

        assert isinstance(result, KineticsFeatures)
        assert result.kinetics_method in ["viennarna_kinetics", "fallback_kinetics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])