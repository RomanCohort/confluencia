"""Minimal ModelConfig/ModelFactory shim used by predictor.build_model()."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = str


@dataclass
class ModelConfig:
    mlp_alpha: float = 1e-4
    mlp_early_stopping: bool = True
    mlp_patience: int = 10
    sgd_alpha: float = 1e-4
    sgd_l1_ratio: float = 0.15
    sgd_early_stopping: bool = True
    hgb_l2: float = 0.0

    @staticmethod
    def for_epitope() -> "ModelConfig":
        return ModelConfig()


class ModelFactory:
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()

    def build(self, name: ModelName, random_state: int = 42) -> Any:
        if name == "hgb":
            return HistGradientBoostingRegressor(random_state=random_state)
        if name == "gbr":
            return GradientBoostingRegressor(random_state=random_state)
        if name == "rf":
            return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=1)
        if name == "ridge":
            return Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=random_state))])
        if name == "mlp":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            early_stopping=bool(self.cfg.mlp_early_stopping),
                            max_iter=1200,
                            random_state=random_state,
                            alpha=float(self.cfg.mlp_alpha),
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unsupported model name: {name}")
