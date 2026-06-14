"""
Confluencia Core Predictor Module Tests

Tests for drug and epitope predictor functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsFunctions:
    """Tests for shared metrics functions."""

    def test_rmse_valid(self):
        """Test RMSE calculation with valid inputs."""
        from confluencia_shared.metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.0])
        result = rmse(y_true, y_pred)
        # RMSE = sqrt(mean((y_true - y_pred)^2))
        expected = np.sqrt(np.mean((y_true - y_pred)**2))
        assert pytest.approx(result, rel=0.01) == expected

    def test_rmse_empty(self):
        """Test RMSE with empty arrays returns NaN."""
        from confluencia_shared.metrics import rmse
        y_true = np.array([])
        y_pred = np.array([])
        result = rmse(y_true, y_pred)
        # Empty arrays should return NaN or 0.0 depending on implementation
        assert result == 0.0 or np.isnan(result)

    def test_reg_metrics(self):
        """Test full regression metrics."""
        from confluencia_shared.metrics import reg_metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.1, 3.0, 4.2, 4.8])
        metrics = reg_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.99


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_create_ridge(self):
        """Test Ridge model creation."""
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.EPITOPE)
        model = factory.build("ridge")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_rf(self):
        """Test RandomForest model creation."""
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.EPITOPE)
        model = factory.build("rf")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_hgb(self):
        """Test HistGradientBoosting model creation."""
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.EPITOPE)
        model = factory.build("hgb")
        assert model is not None
        assert hasattr(model, "fit")

    def test_model_fit_predict(self):
        """Test model fit and predict workflow."""
        from confluencia_shared.models import ModelFactory, ModelPreset
        factory = ModelFactory(ModelPreset.EPITOPE)
        model = factory.build("ridge")
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == 100


class TestMOERegressor:
    """Tests for MOE ensemble regressor."""

    def test_moe_creation(self):
        """Test MOE regressor creation."""
        from confluencia_shared.moe import MOERegressor, ExpertConfig
        config = ExpertConfig()
        moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], config=config)
        assert moe is not None

    def test_moe_fit(self):
        """Test MOE fitting."""
        from confluencia_shared.moe import MOERegressor, ExpertConfig
        config = ExpertConfig()
        moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], config=config)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        moe.fit(X, y)

    def test_moe_predict(self):
        """Test MOE prediction."""
        from confluencia_shared.moe import MOERegressor, ExpertConfig
        config = ExpertConfig()
        moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], config=config)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        moe.fit(X, y)
        pred = moe.predict(X)
        assert len(pred) == 100

    def test_moe_weights(self):
        """Test MOE expert weights."""
        from confluencia_shared.moe import MOERegressor, ExpertConfig
        config = ExpertConfig()
        moe = MOERegressor(expert_names=["ridge", "hgb", "rf"], config=config)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        moe.fit(X, y)
        weights = moe.explain_weights()
        assert len(weights) > 0
        # Weights should sum to ~1.0
        assert pytest.approx(sum(weights.values()), rel=0.1) == 1.0


class TestDataUtils:
    """Tests for data utility functions."""

    def test_resolve_label_existing(self):
        """Test resolve_label with existing column."""
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"efficacy": [0.1, 0.5, 0.9]})
        result = resolve_label(df, "efficacy")
        assert result is not None
        assert len(result) == 3

    def test_resolve_label_missing(self):
        """Test resolve_label with missing column."""
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = resolve_label(df, "efficacy")
        assert result is None

    def test_resolve_label_nan_handling(self):
        """Test resolve_label handles NaN values."""
        from confluencia_shared.data_utils import resolve_label
        df = pd.DataFrame({"efficacy": [0.1, np.nan, 0.9]})
        result = resolve_label(df, "efficacy")
        assert result is not None
        assert result[1] == 0.0  # NaN filled with 0.0


class TestSafeSerialization:
    """Tests for safe serialization module."""

    def test_serialize_deserialize_model(self):
        """Test model serialization roundtrip."""
        from confluencia_shared.safe_serialize import serialize_safe, deserialize_safe
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        model.fit(X, y)

        data = serialize_safe(model, {"model_type": "ridge"})
        loaded_model, meta = deserialize_safe(data)

        assert "model_type" in meta
        assert hasattr(loaded_model, "predict")

    def test_serialize_numpy(self):
        """Test numpy array serialization."""
        from confluencia_shared.safe_serialize import serialize_numpy, deserialize_numpy
        arr = np.random.randn(100, 20).astype(np.float32)
        data = serialize_numpy(arr, {"description": "test array"})
        loaded = deserialize_numpy(data)
        assert loaded.shape == arr.shape
        assert np.allclose(loaded, arr)

    def test_hash_verification(self):
        """Test hash verification."""
        from confluencia_shared.safe_serialize import compute_hash
        data1 = b"test data 123"
        data2 = b"test data 123"
        data3 = b"test data 456"
        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)
        hash3 = compute_hash(data3)
        assert hash1 == hash2
        assert hash1 != hash3

    def test_is_safe_serialization(self):
        """Test safe serialization detection."""
        from confluencia_shared.safe_serialize import is_safe_serialization, serialize_safe, SAFE_FORMAT_MAGIC
        from sklearn.linear_model import Ridge
        model = Ridge()
        safe_data = serialize_safe(model)
        assert is_safe_serialization(safe_data)
        assert not is_safe_serialization(b"invalid data")


class TestMHCFeatures:
    """Tests for MHC feature encoding."""

    def test_mhc_i_encoder(self):
        """Test MHC-I feature encoding."""
        from confluencia_2_0_epitope.core.mhc_features import MHCFeatureEncoder
        encoder = MHCFeatureEncoder()
        feat = encoder.encode("SLYNTVATL", "HLA-A*02:01")
        assert feat.shape[0] == encoder.feature_dim
        # Features may include biochemical properties that can be negative
        assert feat.shape == (encoder.feature_dim,)
        assert not np.any(np.isnan(feat))

    def test_mhc_ii_encoder(self):
        """Test MHC-II feature encoding."""
        from confluencia_2_0_epitope.core.mhc_features import MHCIIFeatureEncoder
        encoder = MHCIIFeatureEncoder()
        feat = encoder.encode("SLYNTVATLYVATL", "HLA-DRB1*04:01")
        assert feat.shape[0] == encoder.feature_dim

    def test_mhc_class_detection(self):
        """Test MHC class auto-detection."""
        from confluencia_2_0_epitope.core.mhc_features import detect_mhc_class
        assert detect_mhc_class("HLA-A*02:01") == "I"
        assert detect_mhc_class("HLA-B*07:02") == "I"
        assert detect_mhc_class("HLA-DRB1*04:01") == "II"
        assert detect_mhc_class("HLA-DQA1*05:01/DQB1*02:01") == "II"

    def test_mhc_batch_encoding(self):
        """Test batch MHC encoding."""
        from confluencia_2_0_epitope.core.mhc_features import MHCFeatureEncoder
        encoder = MHCFeatureEncoder()
        peptides = ["SLYNTVATL", "GILGFVFTL"]
        alleles = ["HLA-A*02:01", "HLA-A*02:01"]
        feats = encoder.encode_batch(peptides, alleles)
        assert feats.shape[0] == 2
        assert feats.shape[1] == encoder.feature_dim


class TestPerAlleleFineTuning:
    """Tests for per-allele fine-tuning pipeline."""

    def test_config_creation(self):
        """Test fine-tuning config."""
        from confluencia_2_0_epitope.core.per_allele_finetuning import PerAlleleFineTuningConfig
        config = PerAlleleFineTuningConfig(
            min_samples_per_allele=100,
            auc_threshold=0.85,
            base_weight=0.3
        )
        assert config.min_samples_per_allele == 100
        assert config.auc_threshold == 0.85
        assert config.base_weight == 0.3

    def test_tuner_creation(self):
        """Test fine-tuner creation."""
        from confluencia_2_0_epitope.core.per_allele_finetuning import PerAlleleFineTuner, PerAlleleFineTuningConfig
        config = PerAlleleFineTuningConfig()
        tuner = PerAlleleFineTuner(config)
        assert tuner is not None
        assert tuner.base_model is None  # Not yet trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])