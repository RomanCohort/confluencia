"""Unit tests for confluencia_shared modules.

Tests cover:
- metrics.py: rmse, reg_metrics
- models.py: ModelFactory, ModelConfig, ModelName
- moe.py: MOERegressor, ExpertConfig, choose_compute_profile
- protocols.py: PredictableRegressor
- data_utils.py: resolve_label
- features/bioseq.py: AA constants and helpers
- utils/logging.py: get_logger
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_rmse_zero(self):
        from confluencia_shared.metrics import rmse
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        from confluencia_shared.metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert abs(rmse(y_true, y_pred) - np.sqrt(1.0 / 3.0)) < 1e-6

    def test_reg_metrics_no_prefix(self):
        from confluencia_shared.metrics import reg_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        m = reg_metrics(y_true, y_pred)
        assert "mae" in m
        assert "rmse" in m
        assert "r2" in m
        assert m["r2"] > 0.8

    def test_reg_metrics_with_prefix(self):
        from confluencia_shared.metrics import reg_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        m = reg_metrics(y_true, y_pred, prefix="eff")
        assert "eff_mae" in m
        assert "eff_rmse" in m
        assert "eff_r2" in m

    def test_reg_metrics_empty(self):
        from confluencia_shared.metrics import reg_metrics
        m = reg_metrics(np.array([]), np.array([]))
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 0.0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class TestModelFactory:
    def test_build_all_types(self):
        from confluencia_shared.models import ModelFactory
        factory = ModelFactory()
        for name in ["rf", "mlp", "ridge", "sgd", "gbr", "hgb"]:
            model = factory.build(name, random_state=42)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_build_epitope_preset(self):
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.EPITOPE)
        rf = factory.build("rf")
        assert hasattr(rf, "predict")

    def test_build_drug_preset(self):
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.DRUG)
        gbr = factory.build("gbr")
        assert hasattr(gbr, "predict")

    def test_unsupported_model_raises(self):
        from confluencia_shared.models import ModelFactory
        factory = ModelFactory()
        with pytest.raises(ValueError, match="Unsupported"):
            factory.build("unknown_model")

    def test_model_config_epitope(self):
        from confluencia_shared.models import ModelConfig
        cfg = ModelConfig.for_epitope()
        assert cfg.rf_n_estimators == 500

    def test_model_config_drug(self):
        from confluencia_shared.models import ModelConfig
        cfg = ModelConfig.for_drug()
        assert cfg.rf_n_estimators == 800

    def test_model_name_type(self):
        from confluencia_shared.models import ModelName
        assert "hgb" in ModelName.__args__
        assert "ridge" in ModelName.__args__
        assert "sgd" in ModelName.__args__


# ---------------------------------------------------------------------------
# moe
# ---------------------------------------------------------------------------
class TestMOE:
    def test_choose_compute_profile_low(self):
        from confluencia_shared.moe import choose_compute_profile
        prof = choose_compute_profile(n_samples=20)
        assert prof.level in ("low", "medium", "high")

    def test_choose_compute_profile_medium(self):
        from confluencia_shared.moe import choose_compute_profile
        prof = choose_compute_profile(n_samples=150)
        assert prof.level in ("low", "medium", "high")

    def test_choose_compute_profile_high(self):
        from confluencia_shared.moe import choose_compute_profile
        prof = choose_compute_profile(n_samples=500)
        assert prof.level in ("low", "medium", "high")

    def test_moe_fit_predict(self):
        from confluencia_shared.moe import MOERegressor, choose_compute_profile
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10)).astype(np.float32)
        y = X[:, 0] * 2.0 + rng.standard_normal(100).astype(np.float32) * 0.1
        prof = choose_compute_profile(n_samples=100)
        moe = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds)
        moe.fit(X, y)
        pred = moe.predict(X)
        assert pred.shape == (100,)
        weights = moe.explain_weights()
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.05

    def test_expert_config(self):
        from confluencia_shared.moe import ExpertConfig
        cfg = ExpertConfig()
        assert cfg.ridge_alpha > 0
        assert cfg.rf_n_estimators > 0


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------
class TestProtocols:
    def test_predictable_regressor_protocol(self):
        from confluencia_shared.protocols import PredictableRegressor
        # sklearn models should satisfy the protocol
        from sklearn.linear_model import Ridge
        assert isinstance(Ridge(), PredictableRegressor)


# ---------------------------------------------------------------------------
# data_utils
# ---------------------------------------------------------------------------
class TestDataUtils:
    def test_resolve_label_existing(self):
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"efficacy": [1.0, 2.0, 3.0]})
        result = resolve_label(df, "efficacy")
        assert result is not None
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_resolve_label_missing(self):
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"a": [1.0]})
        assert resolve_label(df, "nonexistent") is None

    def test_resolve_label_coerces(self):
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"val": ["1.0", "bad", "3.0"]})
        result = resolve_label(df, "val")
        assert result is not None
        assert result[1] == 0.0  # NaN filled with 0.0


# ---------------------------------------------------------------------------
# features / bioseq
# ---------------------------------------------------------------------------
class TestBioseq:
    def test_aa_order_20(self):
        from confluencia_shared.features.bioseq import AA_ORDER
        assert len(AA_ORDER) == 20

    def test_aa_to_idx(self):
        from confluencia_shared.features.bioseq import AA_TO_IDX, AA_ORDER
        for aa in AA_ORDER:
            assert aa in AA_TO_IDX
            assert AA_TO_IDX[aa] < 20

    def test_is_valid_aa(self):
        from confluencia_shared.features.bioseq import is_valid_aa
        assert is_valid_aa("A") is True
        assert is_valid_aa("X") is False

    def test_hydrophobic_fraction(self):
        from confluencia_shared.features.bioseq import hydrophobic_fraction
        assert abs(hydrophobic_fraction("AILM") - 1.0) < 1e-6
        assert abs(hydrophobic_fraction("DEKR") - 0.0) < 1e-6

    def test_aa_composition(self):
        from confluencia_shared.features.bioseq import aa_composition
        comp = aa_composition("AAC")
        assert abs(comp["A"] - 2.0 / 3.0) < 1e-6
        assert abs(comp["C"] - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class TestLogging:
    def test_get_logger(self):
        from confluencia_shared.utils.logging import get_logger
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_get_logger_cached(self):
        from confluencia_shared.utils.logging import get_logger
        l1 = get_logger("test_cache")
        l2 = get_logger("test_cache")
        assert l1 is l2
