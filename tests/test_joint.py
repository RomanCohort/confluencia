"""
Confluencia Joint Module Tests

Tests for the 5D joint evaluation system.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict

# Import from parent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from confluencia_joint.joint_input import (
    JointInput,
    validate_smiles,
    validate_amino_acid_sequence,
    validate_mhc_allele,
)
from confluencia_joint.scoring import (
    ClinicalScore,
    BindingScore,
    KineticsScore,
    JointScore,
    JointScoringEngine,
)


class TestJointInput:
    """Tests for JointInput validation."""

    def test_valid_input(self):
        """Test valid input passes validation."""
        inp = JointInput(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            epitope_seq="SLYNTVATL",
            mhc_allele="HLA-A*02:01",
            dose_mg=200.0,
            freq_per_day=2.0,
            treatment_time=72.0,
        )
        assert inp.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert inp.epitope_seq == "SLYNTVATL"
        assert inp.mhc_allele == "HLA-A*02:01"

    def test_invalid_smiles(self):
        """Test invalid SMILES raises error."""
        with pytest.raises(ValueError):
            JointInput(
                smiles="INVALID_SMILES_123",
                epitope_seq="SLYNTVATL",
                mhc_allele="HLA-A*02:01",
                dose_mg=200.0,
                freq_per_day=2.0,
                treatment_time=72.0,
            )

    def test_invalid_epitope(self):
        """Test invalid epitope sequence raises error."""
        with pytest.raises(ValueError):
            JointInput(
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                epitope_seq="INVALID123",
                mhc_allele="HLA-A*02:01",
                dose_mg=200.0,
                freq_per_day=2.0,
                treatment_time=72.0,
            )

    def test_negative_dose(self):
        """Test negative dose raises error."""
        with pytest.raises(ValueError):
            JointInput(
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                epitope_seq="SLYNTVATL",
                mhc_allele="HLA-A*02:01",
                dose_mg=-100.0,
                freq_per_day=2.0,
                treatment_time=72.0,
            )

    def test_to_drug_dataframe(self):
        """Test conversion to drug DataFrame."""
        inp = JointInput(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            epitope_seq="SLYNTVATL",
            mhc_allele="HLA-A*02:01",
            dose_mg=200.0,
            freq_per_day=2.0,
            treatment_time=72.0,
        )
        df = inp.to_drug_dataframe()
        assert "smiles" in df.columns
        assert "dose_mg" in df.columns
        assert len(df) == 1

    def test_to_epitope_dataframe(self):
        """Test conversion to epitope DataFrame."""
        inp = JointInput(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            epitope_seq="SLYNTVATL",
            mhc_allele="HLA-A*02:01",
            dose_mg=200.0,
            freq_per_day=2.0,
            treatment_time=72.0,
        )
        df = inp.to_epitope_dataframe()
        assert "epitope_seq" in df.columns
        assert "mhc_allele" in df.columns
        assert len(df) == 1

    def test_from_dataframe(self):
        """Test creating JointInput from DataFrame."""
        df = pd.DataFrame({
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
            "epitope_seq": ["SLYNTVATL"],
            "mhc_allele": ["HLA-A*02:01"],
            "dose_mg": [200.0],
            "freq_per_day": [2.0],
            "treatment_time": [72.0],
        })
        inputs = JointInput.from_dataframe(df)
        assert len(inputs) == 1
        assert inputs[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"


class TestScoring:
    """Tests for scoring system."""

    def test_clinical_score(self):
        """Test clinical score computation."""
        score = ClinicalScore(
            efficacy=0.8,
            target_binding=0.7,
            immune_activation=0.6,
            safety_penalty=0.1,
        )
        assert 0.0 <= score.overall <= 1.0
        assert score.efficacy == 0.8

    def test_binding_score(self):
        """Test binding score computation."""
        score = BindingScore(
            efficacy=0.75,
            uncertainty=0.2,
        )
        assert 0.0 <= score.overall <= 1.0
        # Higher uncertainty should lower overall
        assert score.overall < score.efficacy

    def test_kinetics_score(self):
        """Test kinetics score computation."""
        score = KineticsScore(
            half_life=24.0,
            auc=500.0,
            cmax=10.0,
            therapeutic_index=5.0,
        )
        assert 0.0 <= score.overall <= 1.0

    def test_joint_score(self):
        """Test joint score computation."""
        clinical = ClinicalScore(
            efficacy=0.8, target_binding=0.7, immune_activation=0.6, safety_penalty=0.1
        )
        binding = BindingScore(efficacy=0.75, uncertainty=0.2)
        kinetics = KineticsScore(half_life=24.0, auc=500.0, cmax=10.0, therapeutic_index=5.0)

        joint = JointScore(
            clinical=clinical,
            binding=binding,
            kinetics=kinetics,
        )
        assert 0.0 <= joint.composite <= 1.0
        assert joint.recommendation in ["Go", "Conditional", "No-Go"]

    def test_joint_scoring_engine(self):
        """Test full scoring engine."""
        engine = JointScoringEngine(
            clinical_weight=0.4,
            binding_weight=0.35,
            kinetics_weight=0.25,
        )

        clinical = ClinicalScore(
            efficacy=0.8, target_binding=0.7, immune_activation=0.6, safety_penalty=0.1
        )
        binding = BindingScore(efficacy=0.75, uncertainty=0.2)
        kinetics = KineticsScore(half_life=24.0, auc=500.0, cmax=10.0, therapeutic_index=5.0)

        joint = engine.compute_joint_score(clinical, binding, kinetics)
        assert joint is not None
        assert hasattr(joint, "composite")


class TestValidation:
    """Tests for input validation functions."""

    def test_validate_smiles_valid(self):
        """Test valid SMILES validation."""
        # Basic SMILES format check
        result = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result is True

    def test_validate_smiles_empty(self):
        """Test empty SMILES raises error."""
        with pytest.raises(ValueError):
            validate_smiles("")

    def test_validate_amino_acid_valid(self):
        """Test valid amino acid sequence."""
        result = validate_amino_acid_sequence("SLYNTVATL")
        assert result is True

    def test_validate_amino_acid_invalid(self):
        """Test invalid amino acid sequence."""
        with pytest.raises(ValueError):
            validate_amino_acid_sequence("SLYNTVATX")  # X is not standard AA

    def test_validate_mhc_allele_valid(self):
        """Test valid MHC allele."""
        result = validate_mhc_allele("HLA-A*02:01")
        assert result is True

    def test_validate_mhc_allele_class_ii(self):
        """Test valid MHC class II allele."""
        result = validate_mhc_allele("HLA-DRB1*04:01")
        assert result is True


class TestIntegration:
    """Integration tests for joint evaluation pipeline."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(
            "confluencia-2.0-drug/data"
        ).exists(),
        reason="Drug data not available"
    )
    def test_full_pipeline_smoke(self):
        """Smoke test for full joint evaluation pipeline."""
        # This test requires actual model files
        # Skip if not available
        pytest.skip("Requires model files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
