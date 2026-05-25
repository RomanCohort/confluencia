"""
test_immune_sensing.py — Unit tests for immune sensing functions.

Tests:
1. _detect_blunt_end() now returns varied scores, not always True
2. RIG-I scoring with different sequence compositions
3. TLR scoring with uridine-rich sequences
4. PKR scoring integration
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add module path
_CIRCRNA_PATH = Path(__file__).parent.parent
if str(_CIRCRNA_PATH) not in sys.path:
    sys.path.insert(0, str(_CIRCRNA_PATH))

from confluencia_circrna.core.immune_sensing import (
    _detect_blunt_end,
    _gc_content,
    _count_motifs,
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
)


class TestBluntEndDetection:
    """Tests for fixed blunt end detection."""

    def test_blunt_end_returns_float_not_bool(self):
        """Verify function returns float score, not True."""
        seq = "GCGCGCGCGCGC"
        result = _detect_blunt_end(seq)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_gu_rich_sequence_high_score(self):
        """GU-rich sequences should have higher blunt end scores."""
        gu_rich = "GUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGU"
        low_gu = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

        score_high = _detect_blunt_end(gu_rich)
        score_low = _detect_blunt_end(low_gu)

        assert score_high > score_low
        assert score_high > 0.3  # Should be relatively high

    def test_poly_u_penalty(self):
        """Poly-U tracts should reduce blunt end score."""
        with_poly_u = "GCGCGCUUUUUUAGAGAGCGCGCGCGCGCGCGCGC"
        without_poly_u = "GCGCGCGAGAGAGAGCGCGCGCGCGCGCGCGCGCGC"

        score_with = _detect_blunt_end(with_poly_u)
        score_without = _detect_blunt_end(without_poly_u)

        assert score_without > score_with

    def test_different_sequences_different_scores(self):
        """Verify different sequences get different scores."""
        seqs = [
            "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
            "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",
            "GUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGU",
            "ACACACACACACACACACACACACACACACACACACACAC",
        ]

        scores = [_detect_blunt_end(s) for s in seqs]

        # Not all scores should be identical (key fix from v2)
        assert len(set([round(s, 2) for s in scores])) > 1

    def test_gc_terminal_bonus(self):
        """G/C terminal bases should add bonus."""
        gc_start = "GGGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
        u_start = "UUUACGUACGUACGUACGUACGUACGUACGUACGU"

        score_gc = _detect_blunt_end(gc_start)
        score_u = _detect_blunt_end(u_start)

        assert score_gc > score_u


class TestGCContent:
    """Tests for GC content calculation."""

    def test_high_gc_sequence(self):
        """High GC sequence should have high GC content."""
        seq = "GCGCGCGCGCGCGCGC"
        gc = _gc_content(seq)
        assert gc == 1.0

    def test_low_gc_sequence(self):
        """Low GC sequence should have low GC content."""
        seq = "AUAUAUAUAUAUAUAU"
        gc = _gc_content(seq)
        assert gc == 0.0

    def test_mixed_sequence(self):
        """Mixed sequence should have correct GC content."""
        seq = "GCAU" * 10
        gc = _gc_content(seq)
        assert gc == 0.5

    def test_empty_sequence(self):
        """Empty sequence should return 0."""
        seq = ""
        gc = _gc_content(seq)
        assert gc == 0.0


class TestMotifCounting:
    """Tests for motif counting."""

    def test_count_rig_i_motifs(self):
        """Test RIG-I motif counting."""
        seq = "CCUCCACUCCGCUCCUCUCC"
        motifs = ["CCUCC", "UCUCC", "ACUCC", "GCUCC"]
        count = _count_motifs(seq, motifs)
        assert count >= 4  # Should find multiple motifs

    def test_count_tlr_motifs(self):
        """Test TLR motif counting."""
        seq = "GUUGUUGUUUGUUUUUGUU"
        motifs = ["GUUG", "UUGU", "UGUU", "GUUU", "GUU"]
        count = _count_motifs(seq, motifs)
        assert count >= 5

    def test_no_motifs(self):
        """Sequence without motifs should return 0."""
        seq = "ACACACACACACACAC"
        motifs = ["GUUG", "CCUCC"]
        count = _count_motifs(seq, motifs)
        assert count == 0


class TestOverallImmunogenicity:
    """Tests for complete immunogenicity prediction."""

    def test_high_gc_high_rig_i(self):
        """High GC sequences should have higher RIG-I scores."""
        high_gc = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 10
        low_gc = "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAU" * 10

        result_high = predict_circrna_immunogenicity(high_gc)
        result_low = predict_circrna_immunogenicity(low_gc)

        assert result_high["rig_i_score"] > result_low["rig_i_score"]

    def test_uridine_rich_high_tlr(self):
        """Uridine-rich sequences should have higher TLR scores."""
        u_rich = "UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU" + "GUUGUUGUU"  # 50 chars
        u_poor = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"  # 44 chars, will be too_short

        result_high = predict_circrna_immunogenicity(u_rich)
        result_low = predict_circrna_immunogenicity(u_poor)

        # u_poor is too short, so its score is 0
        # u_rich should have positive TLR score due to high U content
        assert result_high["tlr_score"] > 0.0
        assert result_high["tlr_score"] > result_low["tlr_score"]

    def test_output_format(self):
        """Verify output has all expected keys."""
        seq = "ACGUACGUACGUACGU"
        result = predict_circrna_immunogenicity(seq)

        expected_keys = [
            "rig_i_score",
            "tlr_score",
            "pkr_score",
            "overall_immunogenicity",
            "sensing_method",
        ]

        for key in expected_keys:
            assert key in result
            if key != "sensing_method":
                assert isinstance(result[key], float)
                assert 0.0 <= result[key] <= 1.0

    def test_short_sequence_handling(self):
        """Very short sequences should be handled gracefully."""
        seq = "ACGU"
        result = predict_circrna_immunogenicity(seq)

        # Should still have valid keys
        assert "rig_i_score" in result
        assert "overall_immunogenicity" in result

    def test_config_parameter(self):
        """Test custom configuration."""
        config = ImmuneSensingConfig(
            min_length=30,
            detect_blunt_end=True,
            detect_au_rich=True,
        )

        seq = "ACGUACGUACGUACGUACGUACGUACGUACGU"
        result = predict_circrna_immunogenicity(seq, config)

        assert result["sensing_method"] == "rule_based"


class TestLiteratureWeights:
    """Test that weights align with literature."""

    def test_rig_i_weight_distribution(self):
        """RIG-I should be weighted at ~40% of overall."""
        # This is a structural test of the weight constants
        # The actual weights should sum to 1.0
        seq = "ACGUACGUACGUACGU"
        result = predict_circrna_immunogenicity(seq)

        # Verify overall is a weighted combination
        rig_i = result["rig_i_score"]
        tlr = result["tlr_score"]
        pkr = result["pkr_score"]
        overall = result["overall_immunogenicity"]

        # Should approximately follow 0.4 * rig + 0.35 * tlr + 0.25 * pkr
        expected_overall = 0.40 * rig_i + 0.35 * tlr + 0.25 * pkr
        assert abs(overall - expected_overall) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])