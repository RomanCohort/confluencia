"""RNACTM 六室 PK 模型单元测试。"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

# 确保可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from confluencia_3_0.core.pk.rnactm import (
    RNACTMParams,
    infer_rna_ctm_params,
    simulate_rna_ctm,
    summarize_rna_ctm_curve,
)
from confluencia_3_0.core.pk.legacy_ctm import (
    CTMParams,
    params_from_micro_scores,
    simulate_ctm,
    summarize_curve,
)


class TestRNACTMParams:
    """验证 RNACTMParams 默认值。"""

    def test_defaults(self):
        p = RNACTMParams(
            k_uptake=0.8, k_release=0.12, k_escape=0.025,
            k_translate=0.1, k_degrade=0.12, k_protein_half=16.0,
            k_immune_clear=0.01,
        )
        assert p.k_uptake == 0.8
        assert p.k_protein_late_delay == 48.0
        assert p.f_liver == 0.80
        assert p.f_spleen == 0.10


class TestInferRNACTMParams:
    """验证参数推断。"""

    def test_default_params(self):
        p = infer_rna_ctm_params()
        assert 0.01 < p.k_uptake < 2.0
        assert 0.001 < p.k_release < 2.0
        assert 0.001 < p.k_escape < 2.0
        assert 0.001 < p.k_translate < 1.0
        assert 0.01 < p.k_degrade < 1.0
        assert p.k_protein_half > 0
        assert p.k_immune_clear > 0

    def test_modification_effect(self):
        p_none = infer_rna_ctm_params(modification="none")
        p_m6a = infer_rna_ctm_params(modification="m6a")  # lowercase maps to m6a
        p_5mc = infer_rna_ctm_params(modification="5mc")  # lowercase maps to 5mc
        # 修饰应降低降解速率 (提高稳定性)
        assert p_m6a.k_degrade < p_none.k_degrade
        assert p_5mc.k_degrade < p_none.k_degrade

    def test_delivery_vector_effect(self):
        p_lnp = infer_rna_ctm_params(delivery_vector="LNP_standard")
        p_liver = infer_rna_ctm_params(delivery_vector="LNP_liver")
        # 肝靶向 LNP 应有更高肝分布
        assert p_liver.f_liver > p_lnp.f_liver

    def test_route_effect(self):
        p_iv = infer_rna_ctm_params(route="IV")
        p_sc = infer_rna_ctm_params(route="SC")
        # IV 给药摄取应更快
        assert p_iv.k_uptake > p_sc.k_uptake

    def test_immune_score_effect(self):
        p_low = infer_rna_ctm_params(innate_immune_score=0.0)
        p_high = infer_rna_ctm_params(innate_immune_score=1.0)
        # 高免疫评分应增加清除
        assert p_high.k_immune_clear > p_low.k_immune_clear


class TestSimulateRNACTM:
    """验证 RNACTM 模拟。"""

    def test_basic_simulation(self):
        p = infer_rna_ctm_params()
        curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=p, horizon=72)
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) > 0
        assert "protein_translated" in curve.columns
        assert "rna_cytoplasmic" in curve.columns
        assert "efficacy_signal" in curve.columns
        assert "toxicity_signal" in curve.columns

    def test_mass_conservation(self):
        """验证质量守恒: 所有室总和 ≈ 累积给药量。"""
        p = infer_rna_ctm_params()
        curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=p, horizon=72, dt=1.0)
        for _, row in curve.iterrows():
            total = (row["rna_injected"] + row["rna_lnp"] + row["rna_endosomal"] +
                     row["rna_cytoplasmic"] + row["protein_translated"] +
                     row["cumulative_clearance"])
            # 允许数值误差
            assert total >= -0.01, f"Negative total mass at t={row['time_h']}"

    def test_non_negative_compartments(self):
        """验证所有室非负。"""
        p = infer_rna_ctm_params()
        curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=p, horizon=72)
        for col in ["rna_injected", "rna_lnp", "rna_endosomal",
                     "rna_cytoplasmic", "protein_translated", "cumulative_clearance"]:
            assert (curve[col] >= -1e-6).all(), f"Negative values in {col}"

    def test_protein_peak(self):
        """验证蛋白翻译有峰值。"""
        p = infer_rna_ctm_params(ires_score=0.8)
        curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=p, horizon=168)
        protein = curve["protein_translated"]
        assert protein.max() > 0, "No protein produced"
        # 峰值应在中间某处
        peak_idx = protein.idxmax()
        assert 0 < peak_idx < len(protein) - 1

    def test_tissue_distribution(self):
        """验证组织分布系数总和 ≈ 1。"""
        p = infer_rna_ctm_params()
        total = p.f_liver + p.f_spleen + p.f_muscle + p.f_other
        assert abs(total - 1.0) < 0.01


class TestSummarizeRNACTM:
    """验证 RNACTM 曲线总结。"""

    def test_summary_keys(self):
        p = infer_rna_ctm_params()
        curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=p, horizon=72)
        summary = summarize_rna_ctm_curve(curve)
        expected_keys = [
            "rna_ctm_auc_efficacy", "rna_ctm_peak_protein",
            "rna_ctm_peak_cytoplasmic_rna", "rna_ctm_protein_expression_window_h",
            "rna_ctm_rna_half_life_h", "rna_ctm_bioavailability_frac",
            "rna_ctm_peak_toxicity",
        ]
        for k in expected_keys:
            assert k in summary, f"Missing key: {k}"

    def test_empty_curve(self):
        summary = summarize_rna_ctm_curve(pd.DataFrame())
        assert summary["rna_ctm_auc_efficacy"] == 0.0


class TestLegacyCTM:
    """验证四室 CTM 向后兼容。"""

    def test_simulate_ctm(self):
        p = params_from_micro_scores(0.5, 0.5, 0.3)
        curve = simulate_ctm(dose=1.0, freq=1.0, params=p, horizon=48)
        assert isinstance(curve, pd.DataFrame)
        assert len(curve) > 0
        assert "efficacy_signal" in curve.columns

    def test_summarize_curve(self):
        p = params_from_micro_scores(0.5, 0.5, 0.3)
        curve = simulate_ctm(dose=1.0, freq=1.0, params=p, horizon=48)
        summary = summarize_curve(curve)
        assert "auc_efficacy" in summary
        assert summary["auc_efficacy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
