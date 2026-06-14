"""
test_circrna_core.py — Basic unit tests for circRNA core modules.

Tests cover:
1. immune_sensing.py: predict_circrna_immunogenicity
2. rna_modifications.py: ModificationPredictor.analyze
3. folding_kinetics.py: KineticsFeatures dataclass structure
"""

import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from confluencia_circrna.core.immune_sensing import (
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
)
from confluencia_circrna.core.rna_modifications import (
    ModificationPredictor,
    ModificationFeatures,
)
from confluencia_circrna.core.folding_kinetics import (
    FoldingKineticsPredictor,
    KineticsFeatures,
)


class TestImmuneSensing:
    """Test predict_circrna_immunogenicity for circRNA immune sensing."""

    def test_gc_rich_high_rig_i(self):
        """GC-rich sequence should have high RIG-I via dsRNA structure."""
        gc_rich = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10
        result = predict_circrna_immunogenicity(gc_rich)
        assert result["rig_i_score"] > 0.3
        assert result["rig_i_dsRNA_structure"] > 0.2

    def test_au_rich_high_tlr8(self):
        """AU-rich sequence should have high TLR8 score."""
        au_rich = "AUUAUUAUUAUUAUUAUUAUUAUUAUUAUUAUUAUU" * 10
        result = predict_circrna_immunogenicity(au_rich)
        assert result["tlr8_score"] > result["tlr7_score"]

    def test_short_sequence_too_short(self):
        """Sequence <50nt should return 'too_short' sensing_method."""
        short_seq = "ACGUACGUACGU"
        result = predict_circrna_immunogenicity(short_seq)
        assert result["sensing_method"] == "too_short"
        assert result["overall_immunogenicity"] == 0.0

    def test_output_keys(self):
        """Verify output has required keys."""
        seq = "ACGUACGUACGUACGU" * 10
        result = predict_circrna_immunogenicity(seq)
        required_keys = [
            "rig_i_score",
            "tlr7_score",
            "tlr8_score",
            "pkr_score",
            "overall_immunogenicity",
        ]
        for key in required_keys:
            assert key in result
            assert isinstance(result[key], float)


class TestRNAModifications:
    """Test ModificationPredictor.analyze for modification prediction."""

    def test_drach_motif_m6a_sites(self):
        """Sequence with DRACH motifs should find m6A sites."""
        predictor = ModificationPredictor()
        # DRACH motif: GGACU, GAACU, AGACU are canonical
        seq_with_motifs = "GGACUGAACUAGACUAAACU" * 20  # Contains multiple DRACH
        result = predictor.analyze(seq_with_motifs)
        assert len(result.m6a_sites) > 0

    def test_output_metrics_in_range(self):
        """m6a_density, translation_potential, immunogenicity_modulation in [0,1]."""
        predictor = ModificationPredictor()
        seq = "ACGUACGUACGUACGUACGU" * 20
        result = predictor.analyze(seq)
        assert 0.0 <= result.m6a_density <= 1.0
        assert 0.0 <= result.translation_potential <= 1.0
        assert 0.0 <= result.immunogenicity_modulation <= 1.0


class TestFoldingKinetics:
    """Test KineticsFeatures dataclass structure."""

    def test_kinetics_features_structure(self):
        """Verify KineticsFeatures has expected fields."""
        predictor = FoldingKineticsPredictor()
        seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10
        result = predictor.predict(seq)
        assert isinstance(result, KineticsFeatures)
        assert hasattr(result, "folding_rate")
        assert hasattr(result, "barrier_height")
        assert hasattr(result, "landscape_complexity")

    def test_precision_tier_field(self):
        """precision_tier should be 'coarse_estimate' or 'viennarna_kinetics'."""
        predictor = FoldingKineticsPredictor()
        seq = "ACGUACGUACGUACGU" * 10
        result = predictor.predict(seq)
        assert result.precision_tier in ["coarse_estimate", "viennarna_kinetics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])