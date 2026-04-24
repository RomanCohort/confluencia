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

    Supports two gating modes:
    - "static": Global OOF-RMSE inverse weights (default, original behavior)
    - "gating": Input-dependent gating network that learns per-sample weights

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
        gating: str = "static",
    ) -> None:
        self.expert_names = list(expert_names)
        self.folds = int(max(folds, 2))
        self.random_state = int(random_state)
        self.config = config or ExpertConfig()
        self.gating = gating  # "static" or "gating"
        self.scaler = StandardScaler()
        self.experts: Dict[str, RegressorLike] = {}
        self.global_weights: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
        self._fitted = False
        # Gating network parameters (used when gating="gating")
        self._gate_w: Optional[np.ndarray] = None
        self._gate_b: Optional[np.ndarray] = None
        self._oof_predictions: Optional[np.ndarray] = None

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
        oof_all = np.zeros((n, len(self.expert_names)), dtype=np.float32)

        for idx, name in enumerate(self.expert_names):
            oof = np.zeros(n, dtype=np.float32)
            for tr, va in kf.split(Xs):
                m = _make_expert(name, self.random_state, self.config)
                m.fit(Xs[tr], y[tr])
                oof[va] = m.predict(Xs[va]).astype(np.float32)

            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            scores[name] = rmse
            self.metrics[f"{name}_rmse"] = rmse
            oof_all[:, idx] = oof

            final_m = _make_expert(name, self.random_state, self.config)
            final_m.fit(Xs, y)
            self.experts[name] = final_m

        # Static weights (OOF-RMSE inverse)
        inv = np.array([1.0 / max(scores[k], 1e-6) for k in self.expert_names], dtype=np.float64)
        inv = inv / max(inv.sum(), 1e-8)
        self.global_weights = {k: float(w) for k, w in zip(self.expert_names, inv)}

        # Train gating network if requested
        if self.gating == "gating":
            self._train_gating_network(Xs, y, oof_all)

        self._oof_predictions = oof_all
        self._fitted = True
        return self

    def _train_gating_network(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oof_predictions: np.ndarray,
    ) -> None:
        """Train an input-dependent gating network using OOF predictions.

        The gating network learns to assign per-sample expert weights based
        on input features. It minimizes the squared error of the weighted
        combination of OOF predictions vs. true targets.

        Uses a simple softmax regression: gate_weights = softmax(X @ W + b)
        Trained via gradient descent on the MSE of the gated ensemble.
        """
        n_experts = len(self.expert_names)
        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Initialize gating network weights
        self._gate_w = rng.normal(0, 0.01, size=(n_features, n_experts)).astype(np.float32)
        self._gate_b = np.zeros(n_experts, dtype=np.float32)

        # Gradient descent training
        lr = 0.01
        n_epochs = 100
        best_loss = float("inf")
        patience = 10
        no_improve = 0

        for epoch in range(n_epochs):
            # Forward: compute gating weights
            logits = X @ self._gate_w + self._gate_b  # (n, n_experts)
            # Softmax with numerical stability
            logits_shifted = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            gate_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (n, n_experts)

            # Weighted predictions
            gated_pred = np.sum(gate_weights * oof_predictions, axis=1)  # (n,)

            # Loss
            error = gated_pred - y
            loss = float(np.mean(error ** 2))

            if loss < best_loss - 1e-6:
                best_loss = loss
                no_improve = 0
                # Save best weights
                best_w = self._gate_w.copy()
                best_b = self._gate_b.copy()
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            # Backward: gradient of MSE w.r.t. gating weights
            # d_loss/d_gate_weights = 2/n * sum(error * d_gated_pred/d_gate_weights)
            # gated_pred = sum_k gate_weights_k * oof_k
            # gate_weights_k = softmax_k(logits)
            # d_softmax_k/d_logits_j = gate_weights_k * (delta_kj - gate_weights_j)
            d_pred_d_gate = oof_predictions  # (n, n_experts)
            d_loss_d_gated = 2.0 / len(y) * error  # (n,)
            d_loss_d_gate_weights = d_loss_d_gated[:, None] * d_pred_d_gate  # (n, n_experts)

            # Jacobian of softmax: d_softmax/d_logits
            # d_gate_weights_k / d_logits_j = gate_weights_k * (delta_kj - gate_weights_j)
            # So d_loss/d_logits_j = sum_k d_loss/d_gate_weights_k * gate_weights_k * (delta_kj - gate_weights_j)
            # = d_loss/d_gate_weights_j * gate_weights_j - gate_weights_j * sum_k d_loss/d_gate_weights_k * gate_weights_k
            sum_term = (d_loss_d_gate_weights * gate_weights).sum(axis=1, keepdims=True)  # (n, 1)
            d_loss_d_logits = gate_weights * (d_loss_d_gate_weights - sum_term)  # (n, n_experts)

            # Gradient for W and b
            d_loss_d_W = X.T @ d_loss_d_logits  # (n_features, n_experts)
            d_loss_d_b = d_loss_d_logits.sum(axis=0)  # (n_experts,)

            # Clip gradients
            d_loss_d_W = np.clip(d_loss_d_W, -1.0, 1.0)
            d_loss_d_b = np.clip(d_loss_d_b, -1.0, 1.0)

            # Update
            self._gate_w -= lr * d_loss_d_W
            self._gate_b -= lr * d_loss_d_b

        # Restore best weights
        if best_w is not None:
            self._gate_w = best_w
            self._gate_b = best_b

        self.metrics["gating_network_loss"] = best_loss

    def _compute_gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample gating weights.

        Returns:
            Array of shape (n_samples, n_experts) with weights summing to 1.
        """
        if self._gate_w is None:
            # Fallback to static weights
            w = np.array([self.global_weights.get(k, 0.0) for k in self.expert_names])
            return np.tile(w, (X.shape[0], 1))

        logits = X @ self._gate_w + self._gate_b
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

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

        Uses static global weights by default, or per-sample gating weights
        when gating="gating" was specified during initialization.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average predictions.
        """
        expert_pred = self.predict_experts(X)

        if self.gating == "gating" and self._gate_w is not None:
            # Input-dependent gating
            Xs = self.scaler.transform(X)
            gate_weights = self._compute_gate_weights(Xs)
            y = np.zeros((X.shape[0],), dtype=np.float32)
            for idx, name in enumerate(self.expert_names):
                y += gate_weights[:, idx] * expert_pred[name]
        else:
            # Static global weights
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

