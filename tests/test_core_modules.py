"""
Core Module Unit Tests
=======================
Tests for core computational modules in Confluencia.

Run:
    python -m pytest tests/test_core_modules.py -v
    or
    python tests/test_core_modules.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Epitope and Drug modules both have core/ packages.
# We manage sys.path carefully so the correct one is first for each import group.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPITOPE_ROOT = str(PROJECT_ROOT / "confluencia-2.0-epitope")
DRUG_ROOT = str(PROJECT_ROOT / "confluencia-2.0-drug")
SHARED_ROOT = str(PROJECT_ROOT)

# Add shared root to path first (for confluencia_shared imports)
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)


def _import_from_epitope(module_name: str):
    """Import a module from the epitope package (adds epitope root first)."""
    # Clear any cached core modules to ensure fresh import
    for key in list(sys.modules.keys()):
        if key.startswith("core"):
            del sys.modules[key]

    if EPITOPE_ROOT in sys.path:
        sys.path.remove(EPITOPE_ROOT)
    if DRUG_ROOT in sys.path:
        sys.path.remove(DRUG_ROOT)
    sys.path.insert(0, EPITOPE_ROOT)

    return importlib.import_module(module_name)


def _import_from_drug(module_name: str):
    """Import a module from the drug package (adds drug root first)."""
    # Clear any cached core modules to ensure fresh import
    for key in list(sys.modules.keys()):
        if key.startswith("core"):
            del sys.modules[key]

    if EPITOPE_ROOT in sys.path:
        sys.path.remove(EPITOPE_ROOT)
    if DRUG_ROOT in sys.path:
        sys.path.remove(DRUG_ROOT)
    sys.path.insert(0, DRUG_ROOT)

    return importlib.import_module(module_name)


# ============================================================================
# Test: core/features.py (Epitope)
# ============================================================================

class TestFeatures:
    """Tests for confluencia-2.0-epitope/core/features.py"""

    def _features(self):
        return _import_from_epitope("core.features")

    def test_clean_seq_removes_whitespace(self):
        features = self._features()
        assert features._clean_seq("  A C D  ") == "ACD"
        assert features._clean_seq("siinfekl") == "SIINFEKL"
        assert features._clean_seq("") == ""
        assert features._clean_seq(None) == ""

    def test_clean_seq_uppercase(self):
        features = self._features()
        assert features._clean_seq("acdefg") == "ACDEFG"

    def test_stable_hash_u64_deterministic(self):
        features = self._features()
        v1 = features._stable_hash_u64("test_sequence")
        v2 = features._stable_hash_u64("test_sequence")
        assert v1 == v2, "Hash should be deterministic"
        assert isinstance(v1, int)
        assert v1 >= 0, "Hash should be unsigned"

    def test_stable_hash_u64_different_inputs(self):
        features = self._features()
        v1 = features._stable_hash_u64("SIINFEKL")
        v2 = features._stable_hash_u64("GILGFVFTL")
        assert v1 != v2, "Different inputs should produce different hashes"

    def test_hash_kmer_dim_correct(self):
        features = self._features()
        for dim in [32, 64, 128]:
            result = features._hash_kmer("SIINFEKL", k=2, dim=dim)
            assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"

    def test_hash_kmer_dtype(self):
        features = self._features()
        result = features._hash_kmer("SIINFEKL", k=2, dim=64)
        assert result.dtype == np.float32

    def test_hash_kmer_normalized(self):
        features = self._features()
        result = features._hash_kmer("SIINFEKL", k=2, dim=64)
        norm = np.linalg.norm(result)
        assert norm <= 1.0 + 1e-6, f"Norm should be <= 1.0, got {norm}"

    def test_hash_kmer_short_sequence(self):
        features = self._features()
        # Sequence shorter than k
        result = features._hash_kmer("AB", k=3, dim=64)
        assert np.allclose(result, 0.0), "Short sequence should produce zero vector"

    def test_biochem_stats_shape(self):
        features = self._features()
        result = features._biochem_stats("SIINFEKL")
        assert result.shape == (16,), f"Expected shape (16,), got {result.shape}"
        assert result.dtype == np.float32

    def test_biochem_stats_empty(self):
        features = self._features()
        result = features._biochem_stats("")
        assert result.shape == (16,)
        assert np.allclose(result, 0.0)

    def test_biochem_stats_length(self):
        features = self._features()
        result = features._biochem_stats("SIINFEKL")
        assert result[0] == 8.0, f"Length should be 8, got {result[0]}"

    def test_biochem_stats_fractions_sum(self):
        features = self._features()
        result = features._biochem_stats("ACDEFGHIKLMNPQRSTVWY")
        # Hydrophobic + polar + acidic + basic fractions should be reasonable
        total = result[1] + result[2] + result[3] + result[4]
        assert 0 < total <= 1.2, f"Sum of AA type fractions should be positive, got {total}"

    def test_feature_schema_id_format(self):
        features = self._features()
        schema = features.feature_schema_id()
        assert "epitope-feature-schema" in schema
        assert "mamba_d=" in schema
        assert "kmer_dim=" in schema

    def test_build_feature_matrix_output_shape(self):
        features = self._features()
        df = pd.DataFrame({
            "epitope_seq": ["SIINFEKL", "GILGFVFTL", "NLVPMVATV"],
            "dose": [1.0, 2.0, 3.0],
            "freq": [1.0, 1.0, 1.0],
        })
        X, names, env_cols = features.build_feature_matrix(df)
        assert X.shape[0] == 3, f"Expected 3 rows, got {X.shape[0]}"
        assert X.shape[1] == len(names), f"Feature count mismatch: {X.shape[1]} vs {len(names)}"
        assert X.dtype == np.float32

    def test_build_feature_matrix_no_seq(self):
        features = self._features()
        df = pd.DataFrame({
            "dose": [1.0],
            "freq": [1.0],
        })
        X, names, env_cols = features.build_feature_matrix(df)
        assert X.shape[0] == 1
        assert not np.any(np.isnan(X)), "Features should not contain NaN"

    def test_ensure_columns(self):
        features = self._features()
        df = pd.DataFrame({"epitope_seq": ["SIINFEKL"]})
        result = features.ensure_columns(df)
        assert "dose" in result.columns
        assert "freq" in result.columns
        assert "treatment_time" in result.columns


# ============================================================================
# Test: core/moe.py (Epitope)
# ============================================================================

class TestMOE:
    """Tests for confluencia-2.0-epitope/core/moe.py"""

    def _moe(self):
        return _import_from_epitope("core.moe")

    def test_choose_compute_profile_low(self):
        moe = self._moe()
        profile = moe.choose_compute_profile(50)
        assert profile.level == "low"
        assert profile.folds == 3
        assert "ridge" in profile.enabled_experts
        assert "hgb" in profile.enabled_experts

    def test_choose_compute_profile_medium(self):
        moe = self._moe()
        profile = moe.choose_compute_profile(100)
        assert profile.level == "medium"
        assert profile.folds == 4
        assert "rf" in profile.enabled_experts

    def test_choose_compute_profile_high(self):
        moe = self._moe()
        profile = moe.choose_compute_profile(500)
        assert profile.level == "high"
        assert profile.folds == 5
        assert "mlp" in profile.enabled_experts

    def test_choose_compute_profile_explicit(self):
        moe = self._moe()
        profile = moe.choose_compute_profile(500, requested="low")
        assert profile.level == "low"

    def test_make_expert_valid_names(self):
        moe = self._moe()
        for name in ["ridge", "hgb", "rf", "mlp"]:
            expert = moe._make_expert(name, random_state=42)
            assert hasattr(expert, "fit")
            assert hasattr(expert, "predict")

    def test_make_expert_invalid_name(self):
        moe = self._moe()
        try:
            moe._make_expert("invalid_model", random_state=42)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_moe_regressor_fit_predict(self):
        moe = self._moe()
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 10)).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] * 0.5 + rng.normal(0, 0.1, size=100)).astype(np.float32)

        estimator = moe.MOERegressor(expert_names=["ridge", "hgb"], folds=3, random_state=42)
        estimator.fit(X, y)
        preds = estimator.predict(X)

        assert preds.shape == (100,), f"Expected shape (100,), got {preds.shape}"
        assert not np.any(np.isnan(preds)), "Predictions should not contain NaN"

    def test_moe_regressor_weights(self):
        moe = self._moe()
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 10)).astype(np.float32)
        y = (X[:, 0] * 2 + rng.normal(0, 0.1, size=100)).astype(np.float32)

        estimator = moe.MOERegressor(expert_names=["ridge", "hgb", "rf"], folds=3, random_state=42)
        estimator.fit(X, y)

        # Weights should be defined for each expert
        assert len(estimator.global_weights) == 3, f"Expected 3 weights, got {len(estimator.global_weights)}"
        # All weights should be positive
        for w in estimator.global_weights.values():
            assert w > 0, f"Weights should be positive, got {w}"

    def test_moe_regressor_single_expert(self):
        moe = self._moe()
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(50, 5)).astype(np.float32)
        y = (X[:, 0] + rng.normal(0, 0.1, size=50)).astype(np.float32)

        estimator = moe.MOERegressor(expert_names=["ridge"], folds=2, random_state=42)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert preds.shape == (50,)

    def test_compute_profile_dataclass(self):
        moe = self._moe()
        profile = moe.ComputeProfile(level="low", folds=3, enabled_experts=["ridge"])
        assert profile.level == "low"
        assert profile.folds == 3
        assert "ridge" in profile.enabled_experts


# ============================================================================
# Test: core/mamba3.py (Epitope)
# ============================================================================

class TestMamba3:
    """Tests for confluencia-2.0-epitope/core/mamba3.py"""

    def _mamba3(self):
        return _import_from_epitope("core.mamba3")

    def test_mamba3_config_defaults(self):
        mamba3 = self._mamba3()
        config = mamba3.Mamba3Config()
        assert config.d_model == 24
        assert config.local_window == 3
        assert config.meso_window == 11
        assert config.global_window == 33
        assert config.seed == 42

    def test_mamba3_encoder_output_keys(self):
        mamba3 = self._mamba3()
        encoder = mamba3.Mamba3LiteEncoder(mamba3.Mamba3Config())
        result = encoder.encode("SIINFEKL")

        assert "summary" in result
        assert "local_pool" in result
        assert "meso_pool" in result
        assert "global_pool" in result
        assert "token_hidden" in result

    def test_mamba3_encoder_output_shape(self):
        mamba3 = self._mamba3()
        config = mamba3.Mamba3Config(d_model=24)
        encoder = mamba3.Mamba3LiteEncoder(config)
        result = encoder.encode("SIINFEKL")

        d = config.d_model
        assert result["summary"].shape == (d * 4,), f"Summary: expected ({d*4},), got {result['summary'].shape}"
        assert result["local_pool"].shape == (d,)
        assert result["meso_pool"].shape == (d,)
        assert result["global_pool"].shape == (d,)
        assert result["token_hidden"].shape[1] == d

    def test_mamba3_encode_empty(self):
        mamba3 = self._mamba3()
        encoder = mamba3.Mamba3LiteEncoder(mamba3.Mamba3Config())
        result = encoder.encode("")
        assert result["summary"].shape[0] == mamba3.Mamba3Config().d_model * 4
        assert result["token_hidden"].shape[0] == 0

    def test_mamba3_tokenize_basic(self):
        mamba3 = self._mamba3()
        encoder = mamba3.Mamba3LiteEncoder(mamba3.Mamba3Config())
        ids = encoder._tokenize("ACD")
        assert len(ids) == 3
        assert ids.dtype == np.int64

    def test_mamba3_feature_names(self):
        mamba3 = self._mamba3()
        encoder = mamba3.Mamba3LiteEncoder(mamba3.Mamba3Config())
        names = encoder.feature_names()
        d = mamba3.Mamba3Config().d_model
        # 4 * d (summary) + 3 * d (pools) = 7 * d
        assert len(names) == 7 * d

    def test_mamba3_deterministic(self):
        mamba3 = self._mamba3()
        config = mamba3.Mamba3Config(seed=42)
        enc1 = mamba3.Mamba3LiteEncoder(config)
        enc2 = mamba3.Mamba3LiteEncoder(config)
        r1 = enc1.encode("SIINFEKL")
        r2 = enc2.encode("SIINFEKL")
        assert np.allclose(r1["summary"], r2["summary"]), "Same seed should produce same results"


# ============================================================================
# Test: core/ctm.py (Drug)
# ============================================================================

class TestCTM:
    """Tests for confluencia-2.0-drug/core/ctm.py"""

    def _ctm(self):
        return _import_from_drug("core.ctm")

    def test_params_from_micro_scores_defaults(self):
        ctm = self._ctm()
        params = ctm.params_from_micro_scores(0.5, 0.5, 0.5)
        assert params.ka > 0
        assert params.kd > 0
        assert params.ke > 0
        assert params.km > 0
        assert params.signal_gain > 0

    def test_params_from_micro_scores_bounds(self):
        ctm = self._ctm()
        # Test clipping: values outside [0,1] should be clipped
        params = ctm.params_from_micro_scores(-1.0, 2.0, 0.0)
        assert params.ka >= 0.15  # 0.15 + 0.35 * 0.0
        assert params.ka <= 0.50  # 0.15 + 0.35 * 1.0

    def test_params_from_micro_scores_monotonic_binding(self):
        ctm = self._ctm()
        p_low = ctm.params_from_micro_scores(0.1, 0.5, 0.5)
        p_high = ctm.params_from_micro_scores(0.9, 0.5, 0.5)
        # Higher binding -> higher ka and higher signal_gain
        assert p_high.ka > p_low.ka
        assert p_high.signal_gain > p_low.signal_gain

    def test_simulate_ctm_output_shape(self):
        ctm = self._ctm()
        params = ctm.CTMParams(ka=0.3, kd=0.2, ke=0.15, km=0.2, signal_gain=1.5)
        df = ctm.simulate_ctm(dose=5.0, freq=1.0, params=params, horizon=72)

        assert len(df) == 72
        expected_cols = ["time_h", "absorption_A", "distribution_D", "effect_E",
                         "metabolism_M", "efficacy_signal", "toxicity_signal"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_simulate_ctm_non_negative(self):
        ctm = self._ctm()
        params = ctm.CTMParams(ka=0.3, kd=0.2, ke=0.15, km=0.2, signal_gain=1.5)
        df = ctm.simulate_ctm(dose=5.0, freq=1.0, params=params, horizon=72)

        for col in ["absorption_A", "distribution_D", "effect_E", "metabolism_M"]:
            assert (df[col] >= 0).all(), f"{col} should be non-negative"

    def test_simulate_ctm_monotonic_dose(self):
        ctm = self._ctm()
        params = ctm.CTMParams(ka=0.3, kd=0.2, ke=0.15, km=0.2, signal_gain=1.5)
        df_low = ctm.simulate_ctm(dose=1.0, freq=1.0, params=params, horizon=72)
        df_high = ctm.simulate_ctm(dose=10.0, freq=1.0, params=params, horizon=72)

        # Higher dose should produce higher peak efficacy signal
        peak_low = df_low["efficacy_signal"].max()
        peak_high = df_high["efficacy_signal"].max()
        assert peak_high > peak_low, "Higher dose should produce higher efficacy"

    def test_simulate_ctm_custom_horizon(self):
        ctm = self._ctm()
        params = ctm.CTMParams(ka=0.3, kd=0.2, ke=0.15, km=0.2, signal_gain=1.5)
        df = ctm.simulate_ctm(dose=5.0, freq=1.0, params=params, horizon=48)
        assert len(df) == 48

    def test_simulate_ctm_pulsed_dosing(self):
        ctm = self._ctm()
        params = ctm.CTMParams(ka=0.3, kd=0.2, ke=0.15, km=0.2, signal_gain=1.5)
        df = ctm.simulate_ctm(dose=5.0, freq=2.0, params=params, horizon=72)
        # With freq=2.0 (twice daily), absorption should increase at pulse times
        assert df["absorption_A"].sum() > 0, "Some absorption should occur"


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests() -> None:
    """Run all test classes and report results."""
    test_classes = [
        ("Features", TestFeatures()),
        ("MOE", TestMOE()),
        ("Mamba3", TestMamba3()),
        ("CTM", TestCTM()),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for class_name, instance in test_classes:
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n--- {class_name} ({len(methods)} tests) ---")

        for method_name in methods:
            full_name = f"{class_name}.{method_name}"
            try:
                getattr(instance, method_name)()
                print(f"  [PASS] {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {method_name}: {e}")
                total_failed += 1
                failures.append((full_name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print(f"{'=' * 60}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
