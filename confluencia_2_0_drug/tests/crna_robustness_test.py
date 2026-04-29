"""Robustness regression tests for circRNA modules (v2.2).

Tests circRNA feature engineering, RNA CTM, innate immune, evolution,
and trial simulation under edge cases and stress conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.legacy_algorithms import LegacyAlgorithmConfig
from core.training import train_and_predict_drug
from core.features import (
    encode_cirrna_sequence, encode_cirrna_structure, encode_cirrna_functional,
    encode_cirrna_modification, encode_cirrna_delivery,
    build_cirrna_feature_vector, build_cirrna_feature_matrix,
    ensure_cirrna_columns,
)
from core.ctm import infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve
from core.innate_immune import assess_innate_immune, batch_assess_innate_immune
from core.evolution import CircRNAEvolutionConfig, evolve_cirrna_sequences
from core.reliability import credible_eval_cirrna


def _make_crna_df(n: int = 36, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seqs = [
        "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGC",
        "GCUAUGCGCUAUGGCUAGCUAUGCGCUAUGG",
        "AUGCGCUAUGGC" * 3,
        "GCGCGCGCGCGCGCGCGCGCGC",
    ]
    mods = ["m6A", "Psi", "5mC", "none"]
    vecs = ["LNP_standard", "LNP_liver", "AAV"]
    routes = ["IV", "SC", "IM"]

    df = pd.DataFrame({
        "smiles": [["CCO", "CCN(CC)CC", "CC(=O)OC1=CC=CC=C1C(=O)O"][i % 3] for i in range(n)],
        "epitope_seq": ["SLYNTVATL"] * n,
        "dose": rng.uniform(0.2, 8.0, size=n),
        "freq": rng.uniform(0.5, 3.0, size=n),
        "treatment_time": rng.uniform(0, 72, size=n),
        "group_id": rng.choice(["G1", "G2"], size=n),
        "circrna_seq": [seqs[i % len(seqs)] for i in range(n)],
        "modification": [mods[i % len(mods)] for i in range(n)],
        "delivery_vector": [vecs[i % len(vecs)] for i in range(n)],
        "route": [routes[i % len(routes)] for i in range(n)],
        "ires_type": [["EMCV", "HCV", "", ""][i % 4] for i in range(n)],
        "efficacy": 0.5 * rng.uniform(0.5, 5.0, size=n) + rng.normal(0, 0.3, size=n),
    })
    return df


def _assert_no_nan_inf(arr: np.ndarray, label: str) -> None:
    assert bool(np.all(np.isfinite(arr))), f"{label}: contains NaN or Inf"
    assert bool(np.all(arr >= -1e6)), f"{label}: contains extreme negative values"
    assert bool(np.all(arr <= 1e6)), f"{label}: contains extreme positive values"


def test_crna_features_basic():
    """Test basic circRNA feature extraction."""
    seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGCGCUAUGGCUAGCUAUGCGC"
    feats = {
        "sequence": encode_cirrna_sequence(seq),
        "structure": encode_cirrna_structure(seq),
        "functional": encode_cirrna_functional(seq, "EMCV"),
        "modification": encode_cirrna_modification("m6A"),
        "delivery": encode_cirrna_delivery("LNP_liver", "IV"),
    }
    for name, arr in feats.items():
        _assert_no_nan_inf(arr, f"crna_{name}")
        assert arr.ndim == 1, f"crna_{name}: expected 1D, got {arr.ndim}D"
        assert arr.size > 0, f"crna_{name}: empty"

    full = build_cirrna_feature_vector(seq, "m6A", "LNP_liver", "IV", "EMCV")
    _assert_no_nan_inf(full, "crna_full_vector")
    assert full.size == 23, f"expected 23 features, got {full.size}"


def test_crna_features_edge_cases():
    """Test edge cases: empty, short, invalid sequences."""
    assert encode_cirrna_sequence("").sum() == 0.0
    assert encode_cirrna_structure("").sum() == 0.0
    assert encode_cirrna_functional("").sum() == 0.0

    # Very short sequence
    short = encode_cirrna_sequence("AUG")
    _assert_no_nan_inf(short, "short_seq")
    assert short[0] == 3.0  # seq_len

    # Invalid bases (should produce len but zero composition)
    invalid = encode_cirrna_sequence("XYZ123")
    _assert_no_nan_inf(invalid, "invalid_seq")
    assert invalid[0] == 6.0  # seq_len still counts

    # Unknown modification
    unk_mod = encode_cirrna_modification("unknown_chem")
    assert unk_mod[0] == 1.0  # default stability
    assert unk_mod[1] == 0.0  # no evasion

    # Unknown delivery
    unk_del = encode_cirrna_delivery("unknown_vector")
    assert unk_del.sum() == 0.0


def test_crna_features_matrix():
    """Test feature matrix from DataFrame."""
    df = _make_crna_df(10)
    matrix, names = build_cirrna_feature_matrix(df)
    assert matrix.shape[0] == 10, f"expected 10 rows, got {matrix.shape[0]}"
    assert matrix.shape[1] == len(names), f"cols mismatch: {matrix.shape[1]} vs {len(names)}"
    _assert_no_nan_inf(matrix, "crna_matrix")


def test_rna_ctm_configs():
    """Test RNA CTM across all delivery/modification configurations."""
    configs = [
        ("m6A", "LNP_liver", "IV", 0.7, 0.55, 0.8, 0.1),
        ("none", "LNP_standard", "SC", 0.5, 0.5, 0.5, 0.3),
        ("Psi", "AAV", "IM", 0.8, 0.6, 0.9, 0.05),
        ("none", "naked", "IV", 0.4, 0.45, 0.3, 0.5),
    ]
    for mod, vec, route, ires, gc, stab, immune in configs:
        params = infer_rna_ctm_params(mod, vec, route, ires, gc, stab, immune)
        curve = simulate_rna_ctm(2.0, 1.0, params, 168)
        summary = summarize_rna_ctm_curve(curve)
        assert len(curve) == 168, f"{mod}/{vec}: expected 168 time points"
        assert summary["rna_ctm_peak_protein"] >= 0.0
        assert 0.0 <= summary["rna_ctm_bioavailability_frac"] <= 1.0
        _assert_no_nan_inf(curve.values, f"rna_ctm_{mod}_{vec}")


def test_rna_ctm_edge_cases():
    """Test RNA CTM with zero dose and extreme parameters."""
    params = infer_rna_ctm_params("none", "LNP_standard", "IV")
    curve_zero = simulate_rna_ctm(0.0, 1.0, params, 24)
    assert curve_zero["protein_translated"].sum() < 1e-6, "zero dose should produce no protein"

    curve_high = simulate_rna_ctm(100.0, 10.0, params, 72)
    _assert_no_nan_inf(curve_high.values, "high_dose")


def test_innate_immune_all_mods():
    """Test innate immune assessment for all modification types."""
    seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGCGCUAUGGCUAGCUAUGCGC"
    mods = ["none", "m6A", "Psi", "5mC", "ms2m6A"]
    for mod in mods:
        result = assess_innate_immune(seq, mod, "LNP_standard")
        assert 0.0 <= result.innate_immune_score <= 1.0
        assert 0.0 <= result.net_safety_score <= 1.0
        assert 0.0 <= result.interferon_storm_risk <= 1.0
        assert result.interferon_storm_level in ("low", "medium", "high", "critical")
        # Modification should help evasion
        if mod != "none":
            assert result.modification_evasion > 0.0, f"{mod}: should have evasion > 0"


def test_innate_immune_edge_cases():
    """Test innate immune with edge case sequences."""
    # Empty
    r = assess_innate_immune("", "none", "LNP_standard")
    assert r.innate_immune_score == 0.0

    # Very short
    r = assess_innate_immune("AUG", "none", "LNP_standard")
    assert r.innate_immune_score == 0.0

    # dsRNA-rich sequence (high immune activation expected)
    dsrna_seq = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 3
    r_dsrna = assess_innate_immune(dsrna_seq, "none", "naked")
    r_safe = assess_innate_immune(dsrna_seq, "Psi", "LNP_standard")
    assert r_dsrna.tlr3 > 0.1, "dsRNA should activate TLR3"
    assert r_safe.innate_immune_score < r_dsrna.innate_immune_score, "Psi should reduce activation"


def test_batch_innate_immune():
    """Test batch assessment."""
    df = _make_crna_df(20)
    results = batch_assess_innate_immune(df)
    assert len(results) == 20
    for r in results:
        assert "innate_immune_score" in r
        assert "innate_safety_score" in r


def test_crna_evolution():
    """Test circRNA evolution runs to completion."""
    cfg = CircRNAEvolutionConfig(
        rounds=2, candidates_per_round=8, top_k=3, seed=42,
        seed_seq="AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC",
        modification="none", delivery_vector="LNP_liver",
    )
    result_df, artifacts = evolve_cirrna_sequences(cfg)
    assert result_df is not None and not result_df.empty, "evolution should produce results"
    assert len(result_df) == 16, f"expected 2*8=16 rows, got {len(result_df)}"
    assert "reward" in result_df.columns
    assert "innate_safety_score" in result_df.columns
    assert artifacts["rounds_ran"] <= 2
    assert all(r >= 0.0 for r in result_df["reward"].tolist()), "rewards should be non-negative"


def test_crna_credible_eval():
    """Test circRNA credibility evaluation."""
    df = _make_crna_df(80, seed=42)
    result = credible_eval_cirrna(df, backend="hgb", seed=42)
    assert result["enabled"], "evaluation should be enabled for 80 samples"
    assert "feature_importance_pct" in result
    assert "innate_immune_cv_mean" in result
    assert result["has_cirrna_data"]


def test_crna_credible_eval_few_samples():
    """Test that too few samples returns disabled."""
    df = _make_crna_df(10, seed=42)
    result = credible_eval_cirrna(df, backend="hgb", seed=42)
    assert not result["enabled"]


def test_pipeline_compatibility():
    """Ensure circRNA columns don't break existing pipeline."""
    from core.pipeline import run_pipeline
    df = _make_crna_df(40, seed=42)
    # Pipeline should still work even with circRNA columns present
    result_df, curve_df, artifacts, report = train_and_predict_drug(
        df, compute_mode="low", model_backend="hgb",
        legacy_cfg=LegacyAlgorithmConfig(epochs=3, batch_size=16),
    )
    assert len(result_df) == 40
    assert "efficacy_pred" in result_df.columns
    assert len(curve_df) > 0


def main() -> None:
    test_crna_features_basic()
    print("  PASS crna_features_basic")
    test_crna_features_edge_cases()
    print("  PASS crna_features_edge_cases")
    test_crna_features_matrix()
    print("  PASS crna_features_matrix")
    test_rna_ctm_configs()
    print("  PASS rna_ctm_configs")
    test_rna_ctm_edge_cases()
    print("  PASS rna_ctm_edge_cases")
    test_innate_immune_all_mods()
    print("  PASS innate_immune_all_mods")
    test_innate_immune_edge_cases()
    print("  PASS innate_immune_edge_cases")
    test_batch_innate_immune()
    print("  PASS batch_innate_immune")
    test_crna_evolution()
    print("  PASS crna_evolution")
    test_crna_credible_eval()
    print("  PASS crna_credible_eval")
    test_crna_credible_eval_few_samples()
    print("  PASS crna_credible_eval_few_samples")
    test_pipeline_compatibility()
    print("  PASS pipeline_compatibility")
    print("drug circRNA robustness test passed")


if __name__ == "__main__":
    main()
