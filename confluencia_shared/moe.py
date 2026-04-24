"""
Unified Mixture-of-Experts (MOE) Regressor for Confluencia.

This module provides a sample-size-adaptive MOE ensemble that automatically
selects and weights regression experts based on data availability.

Used by both epitope and drug prediction modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable, Optional, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class ComputeProfile:
    """Configuration for compute intensity level."""
    level: str
    folds: int
    enabled_experts: List[str]


@dataclass
class ExpertConfig:
    """Hyperparameters for individual experts.

    Allows customization of expert behavior for different modules.
    """
    ridge_alpha: float = 1.0        # L2 regularization strength; 1.0 is sklearn default, suitable for standardized features.
    hgb_max_depth: int = 6          # Limits tree depth to prevent overfitting; 6 ≈ log2(64) captures interactions without memorization.
    hgb_learning_rate: float = 0.05 # Conservative shrinkage; slower learning with more trees reduces variance in small samples.
    rf_n_estimators: int = 240      # Sufficient for stable ensemble estimates; diminishing returns beyond 200 (Probst et al. 2019).
    rf_max_depth: int = 12          # Allows deeper trees than HGB since RF's bagging provides additional regularization.
    rf_n_jobs: int = 1              # Single-thread default for reproducibility; set to -1 for production.
    mlp_hidden_layers: tuple = (128, 64)  # Two-layer bottleneck: 128 captures interactions, 64 compresses before output.
    mlp_max_iter: int = 400         # Enough for convergence on N<300 with early stopping enabled.
    mlp_early_stopping: bool = True # Essential for small samples; prevents overfitting by monitoring validation loss.


# Default configs for each module
EXPERT_CONFIG_EPITOPE = ExpertConfig(
    ridge_alpha=1.2,
    rf_n_estimators=220,
    rf_n_jobs=-1,
    mlp_max_iter=450,
)

EXPERT_CONFIG_DRUG = ExpertConfig(
    ridge_alpha=1.0,
    rf_n_estimators=240,
    rf_n_jobs=1,
    mlp_max_iter=400,
)


def choose_compute_profile(n_samples: int, requested: str = "auto") -> ComputeProfile:
    """Select compute profile based on sample size.

    Args:
        n_samples: Number of training samples.
        requested: "auto" for automatic selection, or "low"/"medium"/"high".

    Returns:
        ComputeProfile with level, folds, and enabled experts.
    """
    if requested != "auto":
        if requested == "low":
            return ComputeProfile(level="low", folds=3, enabled_experts=["ridge", "hgb"])
        if requested == "medium":
            return ComputeProfile(level="medium", folds=4, enabled_experts=["ridge", "hgb", "rf"])
        return ComputeProfile(level="high", folds=5, enabled_experts=["ridge", "hgb", "rf", "mlp"])

    if n_samples < 80:
        # Low profile: below ~80 samples, RF and MLP exhibit high variance (overfit).
        # Ridge (parametric) + HGB (boosting with depth limit) are the safest choices.
        return ComputeProfile(level="low", folds=3, enabled_experts=["ridge", "hgb"])
    if n_samples < 300:
        # Medium profile: enough data for RF's bagging to be effective (O(n_estimators) variance reduction),
        # but MLP still risks overfitting. 80-300 is the typical circRNA wet-lab range.
        return ComputeProfile(level="medium", folds=4, enabled_experts=["ridge", "hgb", "rf"])
    return ComputeProfile(level="high", folds=5, enabled_experts=["ridge", "hgb", "rf", "mlp"])


@runtime_checkable
class RegressorLike(Protocol):
    """Protocol for sklearn-compatible regressors."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressorLike":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def _make_expert(
    name: str,
    random_state: int,
    config: Optional[ExpertConfig] = None,
) -> RegressorLike:
    """Create an expert regressor by name.

    Args:
        name: Expert name ("ridge", "hgb", "rf", "mlp").
        random_state: Random seed for reproducibility.
        config: Expert hyperparameters. Uses defaults if None.

    Returns:
        Instantiated regressor.
    """
    cfg = config or ExpertConfig()

    if name == "ridge":
        return Ridge(alpha=cfg.ridge_alpha)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=cfg.hgb_max_depth,
            learning_rate=cfg.hgb_learning_rate,
            random_state=random_state,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            random_state=random_state,
            n_jobs=cfg.rf_n_jobs,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=cfg.mlp_hidden_layers,
            early_stopping=cfg.mlp_early_stopping,
            max_iter=cfg.mlp_max_iter,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported expert: {name}")


class MOERegressor:
    """Transparent MOE regressor with data-dependent expert weighting.

    The ensemble combines predictions from multiple expert regressors
    weighted by their out-of-fold RMSE performance (inverse weighting).

    Attributes:
        expert_names: List of expert names to include.
        folds: Number of CV folds for OOF predictions.
        random_state: Random seed.
        config: Expert hyperparameters.
        scaler: Fitted StandardScaler.
        experts: Dictionary of fitted experts.
        global_weights: Expert weights (sum to 1.0).
        metrics: Training metrics for each expert.

    Example:
        >>> moe = MOERegressor(["ridge", "hgb", "rf"], folds=4)
        >>> moe.fit(X_train, y_train)
        >>> predictions = moe.predict(X_test)
        >>> uncertainty = moe.predict_uncertainty(X_test)
    """

    def __init__(
        self,
        expert_names: List[str],
        folds: int = 4,
        random_state: int = 42,
        config: Optional[ExpertConfig] = None,
    ) -> None:
        self.expert_names = list(expert_names)
        self.folds = int(max(folds, 2))
        self.random_state = int(random_state)
        self.config = config or ExpertConfig()
        self.scaler = StandardScaler()
        self.experts: Dict[str, RegressorLike] = {}
        self.global_weights: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MOERegressor":
        """Fit the MOE ensemble.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,).

        Returns:
            self
        """
        Xs = self.scaler.fit_transform(X)
        n = len(y)
        split_n = min(self.folds, max(2, n // 10)) if n >= 20 else 2
        kf = KFold(n_splits=split_n, shuffle=True, random_state=self.random_state)

        scores: Dict[str, float] = {}
        for name in self.expert_names:
            oof = np.zeros(n, dtype=np.float32)
            for tr, va in kf.split(Xs):
                m = _make_expert(name, self.random_state, self.config)
                m.fit(Xs[tr], y[tr])
                oof[va] = m.predict(Xs[va]).astype(np.float32)

            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            scores[name] = rmse
            self.metrics[f"{name}_rmse"] = rmse

            final_m = _make_expert(name, self.random_state, self.config)
            final_m.fit(Xs, y)
            self.experts[name] = final_m

        inv = np.array([1.0 / max(scores[k], 1e-6) for k in self.expert_names], dtype=np.float64)
        inv = inv / max(inv.sum(), 1e-8)
        self.global_weights = {k: float(w) for k, w in zip(self.expert_names, inv)}
        self._fitted = True
        return self

    def predict_experts(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each expert separately.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping expert name to predictions.
        """
        Xs = self.scaler.transform(X)
        out: Dict[str, np.ndarray] = {}
        for name, model in self.experts.items():
            out[name] = model.predict(Xs).astype(np.float32)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average predictions.
        """
        expert_pred = self.predict_experts(X)
        y = np.zeros((X.shape[0],), dtype=np.float32)
        for name, pred in expert_pred.items():
            y += float(self.global_weights.get(name, 0.0)) * pred
        return y

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty from expert disagreement.

        Higher values indicate greater disagreement among experts.

        Args:
            X: Feature matrix.

        Returns:
            Standard deviation of expert predictions.
        """
        expert_pred = self.predict_experts(X)
        if not expert_pred:
            return np.zeros((X.shape[0],), dtype=np.float32)
        arr = np.vstack([p for p in expert_pred.values()]).astype(np.float32)
        return arr.std(axis=0).astype(np.float32)

    def explain_weights(self) -> Dict[str, float]:
        """Get the learned expert weights.

        Returns:
            Dictionary mapping expert name to weight.
        """
        return dict(self.global_weights)

    def __repr__(self) -> str:
        return (
            f"MOERegressor(experts={self.expert_names}, folds={self.folds}, "
            f"weights={self.global_weights if self._fitted else 'not fitted'})"
        )


class GatedMOERegressor:
    """Gated MOE regressor with sample-dependent expert weighting.

    Instead of global static weights (OOF-RMSE inverse weighting), this model
    uses a lightweight gating network that outputs per-expert weights conditioned
    on input features (dose/freq context). This captures the fact that Ridge may
    be better for low-dose samples while HGB excels at high-dose regimes.

    Attributes:
        expert_names: List of expert names.
        folds: CV folds for OOF training of base experts.
        gating_hidden: Hidden layer sizes for the gating MLP.
        random_state: Random seed.
        config: Expert hyperparameters.
        scaler: Fitted StandardScaler for features.
        gating_scaler: StandardScaler for gating inputs.
        experts: Dictionary of fitted base experts.
        gating_net: Fitted MLPRegressor for per-sample weights.
        metrics: Training metrics.
    """

    def __init__(
        self,
        expert_names: List[str],
        folds: int = 4,
        gating_hidden: tuple = (32, 16),
        random_state: int = 42,
        config: Optional[ExpertConfig] = None,
    ) -> None:
        self.expert_names = list(expert_names)
        self.folds = int(max(folds, 2))
        self.gating_hidden = gating_hidden
        self.random_state = int(random_state)
        self.config = config or ExpertConfig()
        self.scaler = StandardScaler()
        self.gating_scaler = StandardScaler()
        self.experts: Dict[str, RegressorLike] = {}
        self.gating_net: Optional[Any] = None
        self.metrics: Dict[str, float] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GatedMOERegressor":
        """Fit base experts (OOF) and gating network.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,).

        Returns:
            self
        """
        Xs = self.scaler.fit_transform(X)
        n = len(y)
        split_n = min(self.folds, max(2, n // 10)) if n >= 20 else 2
        kf = KFold(n_splits=split_n, shuffle=True, random_state=self.random_state)

        # Step 1: Train base experts OOF and get out-of-fold predictions
        oof_preds: Dict[str, np.ndarray] = {}
        scores: Dict[str, float] = {}
        for name in self.expert_names:
            oof = np.zeros(n, dtype=np.float32)
            for tr, va in kf.split(Xs):
                m = _make_expert(name, self.random_state, self.config)
                m.fit(Xs[tr], y[tr])
                oof[va] = m.predict(Xs[va]).astype(np.float32)
            oof_preds[name] = oof
            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            scores[name] = rmse
            self.metrics[f"{name}_rmse"] = rmse
            final_m = _make_expert(name, self.random_state, self.config)
            final_m.fit(Xs, y)
            self.experts[name] = final_m

        # Step 2: Train gating network on (X, oof_weights)
        # Gating input: use a subset of X (dose/freq context columns) or all features
        # Gate target: per-sample softmax-normalized inverse-RMSE weights
        k = len(self.expert_names)
        # Compute sample-dependent inverse-RMSE weights (use OOF predictions as signal)
        # For each sample, the "best" expert gets weight 1, others 0, then smooth
        best_per_sample = np.argmax(np.stack([oof_preds[name] for name in self.expert_names], axis=1), axis=1)
        # Smooth: use distance from best expert's prediction as weight signal
        oof_stack = np.stack([oof_preds[name] for name in self.expert_names], axis=1)  # (n, k)
        residuals = np.abs(oof_stack - y.reshape(-1, 1))  # (n, k)
        inv_res = 1.0 / (residuals + 1e-4)
        gate_targets = inv_res / inv_res.sum(axis=1, keepdims=True)  # (n, k), sum=1

        # Gating network: small MLP on X → k softmax weights
        Xg = self.gating_scaler.fit_transform(Xs)
        self.gating_net = MLPRegressor(
            hidden_layer_sizes=self.gating_hidden,
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.random_state,
        )
        self.gating_net.fit(Xg, gate_targets)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get gated ensemble predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average predictions with sample-dependent gating.
        """
        Xs = self.scaler.transform(X)
        k = len(self.expert_names)

        if self.gating_net is None:
            # Fallback: uniform weights
            weights = np.ones(k, dtype=np.float32) / k
            return self._weighted_predict(Xs, weights)

        Xg = self.gating_scaler.transform(Xs)
        logits = self.gating_net.predict(Xg).astype(np.float32)  # (n, k)
        # Manual softmax for numerical stability
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        gate_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return self._weighted_predict(Xs, gate_weights)

    def _weighted_predict(self, Xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply per-sample weights to expert predictions."""
        n = Xs.shape[0]
        k = len(self.expert_names)
        out = np.zeros(n, dtype=np.float32)
        for j, name in enumerate(self.expert_names):
            pred = self.experts[name].predict(Xs).astype(np.float32)
            out += weights[:, j] * pred
        return out

    def predict_experts(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each expert separately."""
        Xs = self.scaler.transform(X)
        out: Dict[str, np.ndarray] = {}
        for name, model in self.experts.items():
            out[name] = model.predict(Xs).astype(np.float32)
        return out

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty from expert disagreement."""
        expert_pred = self.predict_experts(X)
        if not expert_pred:
            return np.zeros((X.shape[0],), dtype=np.float32)
        arr = np.vstack([p for p in expert_pred.values()]).astype(np.float32)
        return arr.std(axis=0).astype(np.float32)

    def explain_weights(self) -> Dict[str, float]:
        """Return static weights (for compatibility)."""
        # Gated MOE has sample-dependent weights — return uniform for summary
        k = len(self.expert_names)
        return {name: 1.0 / k for name in self.expert_names}

    def __repr__(self) -> str:
        return (
            f"GatedMOERegressor(experts={self.expert_names}, folds={self.folds}, "
            f"gating_hidden={self.gating_hidden}, fitted={self._fitted})"
        )

