"""
Unified Model Factory for Confluencia.

Provides a single source of truth for model instantiation across
epitope and drug prediction modules.

Usage:
    from confluencia_shared.models import ModelFactory, ModelPreset

    # Use preset for epitope
    factory = ModelFactory(ModelPreset.EPITOPE)
    rf_model = factory.build("rf", random_state=42)

    # Use preset for drug
    factory = ModelFactory(ModelPreset.DRUG)
    mlp_model = factory.build("mlp", random_state=42)

    # Custom configuration
    config = ModelConfig(rf_n_estimators=1000, mlp_hidden=(512, 256))
    factory = ModelFactory(config)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Union

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Unified model name type - supports all model types across epitope and drug modules
ModelName = Literal["hgb", "gbr", "rf", "mlp", "ridge", "sgd"]


class ModelPreset(Enum):
    """Presets for different prediction tasks."""
    EPITOPE = "epitope"
    DRUG = "drug"
    DEFAULT = "default"


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters.

    Attributes:
        rf_n_estimators: Number of trees in Random Forest.
        rf_max_depth: Maximum depth of RF trees (None for unlimited).
        rf_n_jobs: Number of parallel jobs for RF.
        mlp_hidden: Hidden layer sizes for MLP.
        mlp_alpha: L2 regularization for MLP.
        mlp_max_iter: Maximum iterations for MLP.
        mlp_early_stopping: Whether to use early stopping for MLP.
        ridge_alpha: L2 regularization for Ridge.
        sgd_alpha: Regularization for SGD.
        sgd_l1_ratio: L1/L2 ratio for SGD elastic net.
        hgb_l2: L2 regularization for HGB.
        hgb_max_iter: Maximum iterations for HGB.
        gbr_n_estimators: Number of boosting stages for GBR.
    """
    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_n_jobs: int = -1

    # MLP
    mlp_hidden: Tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 2000
    mlp_early_stopping: bool = True
    mlp_patience: int = 10

    # Ridge
    ridge_alpha: float = 1.0

    # SGD
    sgd_alpha: float = 1e-4
    sgd_l1_ratio: float = 0.15
    sgd_early_stopping: bool = True

    # HGB
    hgb_l2: float = 0.0
    hgb_max_iter: int = 100
    hgb_early_stopping: bool = True
    hgb_validation_fraction: float = 0.1

    # GBR
    gbr_n_estimators: int = 100
    gbr_max_depth: int = 3

    @classmethod
    def for_epitope(cls) -> "ModelConfig":
        """Return configuration optimized for epitope prediction."""
        return cls(
            rf_n_estimators=500,
            rf_n_jobs=-1,
            mlp_hidden=(128, 64),
            mlp_max_iter=2000,
        )

    @classmethod
    def for_drug(cls) -> "ModelConfig":
        """Return configuration optimized for drug prediction."""
        return cls(
            rf_n_estimators=800,
            rf_n_jobs=-1,
            mlp_hidden=(256, 128),
            mlp_max_iter=4000,
            ridge_alpha=1.0,
        )


# Preset configurations
PRESET_CONFIGS: Dict[ModelPreset, ModelConfig] = {
    ModelPreset.EPITOPE: ModelConfig.for_epitope(),
    ModelPreset.DRUG: ModelConfig.for_drug(),
    ModelPreset.DEFAULT: ModelConfig(),
}


class ModelFactory:
    """Factory for creating sklearn-compatible regressors.

    Provides a unified interface for model creation with configurable
    hyperparameters and presets for different prediction tasks.

    Example:
        >>> factory = ModelFactory(ModelPreset.EPITOPE)
        >>> model = factory.build("rf", random_state=42)
        >>> model.fit(X, y)
    """

    SUPPORTED_MODELS = {"rf", "mlp", "ridge", "sgd", "gbr", "hgb"}

    def __init__(
        self,
        config_or_preset: Union[ModelConfig, ModelPreset, None] = None,
    ) -> None:
        """Initialize factory with configuration.

        Args:
            config_or_preset: ModelConfig instance, ModelPreset enum, or None.
                If None, uses ModelPreset.DEFAULT.
        """
        if config_or_preset is None:
            self.config = PRESET_CONFIGS[ModelPreset.DEFAULT]
        elif isinstance(config_or_preset, ModelPreset):
            self.config = PRESET_CONFIGS[config_or_preset]
        else:
            self.config = config_or_preset

    def build(self, model_name: str, random_state: int = 42) -> Any:
        """Build a regressor by name.

        Args:
            model_name: One of "rf", "mlp", "ridge", "sgd", "gbr", "hgb".
            random_state: Random seed for reproducibility.

        Returns:
            sklearn-compatible regressor.

        Raises:
            ValueError: If model_name is not supported.
        """
        model_name = model_name.lower()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        builder = {
            "rf": self._build_rf,
            "mlp": self._build_mlp,
            "ridge": self._build_ridge,
            "sgd": self._build_sgd,
            "gbr": self._build_gbr,
            "hgb": self._build_hgb,
        }[model_name]

        return builder(random_state)

    def _build_rf(self, random_state: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=random_state,
            n_jobs=self.config.rf_n_jobs,
        )

    def _build_mlp(self, random_state: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=self.config.mlp_hidden,
                        activation="relu",
                        alpha=self.config.mlp_alpha,
                        max_iter=self.config.mlp_max_iter,
                        early_stopping=self.config.mlp_early_stopping,
                        n_iter_no_change=self.config.mlp_patience,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def _build_ridge(self, random_state: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.config.ridge_alpha, random_state=random_state)),
            ]
        )

    def _build_sgd(self, random_state: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "sgd",
                    SGDRegressor(
                        loss="squared_error",
                        penalty="elasticnet",
                        alpha=self.config.sgd_alpha,
                        l1_ratio=self.config.sgd_l1_ratio,
                        max_iter=self.config.mlp_max_iter,
                        tol=1e-4,
                        early_stopping=self.config.sgd_early_stopping,
                        n_iter_no_change=self.config.mlp_patience,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def _build_gbr(self, random_state: int) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=self.config.gbr_n_estimators,
            max_depth=self.config.gbr_max_depth,
            random_state=random_state,
        )

    def _build_hgb(self, random_state: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.config.hgb_max_iter,
            early_stopping=self.config.hgb_early_stopping,
            validation_fraction=self.config.hgb_validation_fraction,
            l2_regularization=self.config.hgb_l2,
            random_state=random_state,
        )


# Convenience functions for backwards compatibility
def build_epitope_model(model_name: str, random_state: int = 42) -> Any:
    """Build a model with epitope-optimized settings."""
    factory = ModelFactory(ModelPreset.EPITOPE)
    return factory.build(model_name, random_state)


def build_drug_model(model_name: str, random_state: int = 42) -> Any:
    """Build a model with drug-optimized settings."""
    factory = ModelFactory(ModelPreset.DRUG)
    return factory.build(model_name, random_state)


__all__ = [
    "ModelFactory",
    "ModelConfig",
    "ModelName",
    "ModelPreset",
    "PRESET_CONFIGS",
    "build_epitope_model",
    "build_drug_model",
]
