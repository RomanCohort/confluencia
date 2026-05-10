"""
Unified Mixture-of-Experts (MOE) Regressor for Confluencia.

This module provides a sample-size-adaptive MOE ensemble that automatically
selects and weights regression experts based on data availability.

Used by both epitope and drug prediction modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable, Optional, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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
    # ── XGBoost parameters ─────────────────────────────────────────────────────
    xgb_n_estimators: int = 300     # Number of boosting rounds.
    xgb_max_depth: int = 6          # Tree depth (same as HGB for consistency).
    xgb_learning_rate: float = 0.05 # Step size shrinkage.
    xgb_subsample: float = 0.8      # Row subsampling ratio for regularization.
    xgb_colsample_bytree: float = 0.8  # Column subsampling ratio.
    xgb_reg_lambda: float = 1.0     # L2 regularization on weights.
    xgb_n_jobs: int = 1             # Parallel threads.
    # ── LightGBM parameters ────────────────────────────────────────────────────
    lgb_n_estimators: int = 300     # Number of boosting iterations.
    lgb_max_depth: int = 6          # Tree depth limit.
    lgb_learning_rate: float = 0.05 # Learning rate.
    lgb_num_leaves: int = 31        # Max leaves (2^depth - 1 for depth=6 is 63, use 31 for regularization).
    lgb_subsample: float = 0.8      # Bagging fraction.
    lgb_colsample_bytree: float = 0.8  # Feature fraction.
    lgb_reg_lambda: float = 1.0     # L2 regularization.
    lgb_n_jobs: int = 1             # Parallel threads.
    # ── Extra Trees parameters ─────────────────────────────────────────────────
    et_n_estimators: int = 240      # Number of trees.
    et_max_depth: int = 12          # Tree depth.
    et_n_jobs: int = 1              # Parallel threads.


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

EXPERT_CONFIG_DRUG_ULTRA = ExpertConfig(
    ridge_alpha=1.0,
    rf_n_estimators=240,
    rf_n_jobs=1,
    mlp_max_iter=400,
    # XGBoost
    xgb_n_estimators=300,
    xgb_max_depth=6,
    xgb_learning_rate=0.05,
    xgb_subsample=0.8,
    xgb_colsample_bytree=0.8,
    xgb_reg_lambda=1.0,
    xgb_n_jobs=-1,
    # LightGBM
    lgb_n_estimators=300,
    lgb_max_depth=6,
    lgb_learning_rate=0.05,
    lgb_num_leaves=31,
    lgb_subsample=0.8,
    lgb_colsample_bytree=0.8,
    lgb_reg_lambda=1.0,
    lgb_n_jobs=-1,
    # Extra Trees
    et_n_estimators=240,
    et_max_depth=12,
    et_n_jobs=-1,
)


def choose_compute_profile(n_samples: int, requested: str = "auto") -> ComputeProfile:
    """Select compute profile based on sample size.

    Args:
        n_samples: Number of training samples.
        requested: "auto" for automatic selection, or "low"/"medium"/"high"/"ultra".

    Returns:
        ComputeProfile with level, folds, and enabled experts.
    """
    if requested != "auto":
        if requested == "low":
            return ComputeProfile(level="low", folds=3, enabled_experts=["ridge", "hgb"])
        if requested == "medium":
            return ComputeProfile(level="medium", folds=4, enabled_experts=["ridge", "hgb", "rf"])
        if requested == "high":
            return ComputeProfile(level="high", folds=5, enabled_experts=["ridge", "hgb", "rf", "mlp"])
        if requested == "ultra":
            # Ultra profile: all experts including XGBoost and LightGBM
            return ComputeProfile(
                level="ultra",
                folds=5,
                enabled_experts=["ridge", "hgb", "rf", "mlp", "xgb", "lgb", "et"],
            )
        # Default to high for unknown levels
        return ComputeProfile(level="high", folds=5, enabled_experts=["ridge", "hgb", "rf", "mlp"])

    if n_samples < 80:
        # Low profile: below ~80 samples, RF and MLP exhibit high variance (overfit).
        # Ridge (parametric) + HGB (boosting with depth limit) are the safest choices.
        return ComputeProfile(level="low", folds=3, enabled_experts=["ridge", "hgb"])
    if n_samples < 300:
        # Medium profile: enough data for RF's bagging to be effective (O(n_estimators) variance reduction),
        # but MLP still risks overfitting. 80-300 is the typical circRNA wet-lab range.
        return ComputeProfile(level="medium", folds=4, enabled_experts=["ridge", "hgb", "rf"])
    if n_samples < 1000:
        # High profile: enough for MLP to be stable.
        return ComputeProfile(level="high", folds=5, enabled_experts=["ridge", "hgb", "rf", "mlp"])
    # Ultra profile: large datasets benefit from diverse expert pool including XGBoost/LightGBM.
    return ComputeProfile(
        level="ultra",
        folds=5,
        enabled_experts=["ridge", "hgb", "rf", "mlp", "xgb", "lgb", "et"],
    )


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
        name: Expert name ("ridge", "hgb", "rf", "mlp", "xgb", "lgb", "et").
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
    if name == "xgb":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample_bytree,
                reg_lambda=cfg.xgb_reg_lambda,
                random_state=random_state,
                n_jobs=cfg.xgb_n_jobs,
                verbosity=0,
            )
        except ImportError:
            logger.warning(
                f"XGBoost not available; skipping expert '{name}'. "
                "Install with: pip install xgboost"
            )
            raise ValueError(f"Expert '{name}' requires xgboost: pip install xgboost")
    if name == "lgb":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=cfg.lgb_n_estimators,
                max_depth=cfg.lgb_max_depth,
                learning_rate=cfg.lgb_learning_rate,
                num_leaves=cfg.lgb_num_leaves,
                subsample=cfg.lgb_subsample,
                colsample_bytree=cfg.lgb_colsample_bytree,
                reg_lambda=cfg.lgb_reg_lambda,
                random_state=random_state,
                n_jobs=cfg.lgb_n_jobs,
                verbosity=-1,
            )
        except ImportError:
            logger.warning(
                f"LightGBM not available; skipping expert '{name}'. "
                "Install with: pip install lightgbm"
            )
            raise ValueError(f"Expert '{name}' requires lightgbm: pip install lightgbm")
    if name == "et":
        try:
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=cfg.et_n_estimators,
                max_depth=cfg.et_max_depth,
                random_state=random_state,
                n_jobs=cfg.et_n_jobs,
            )
        except ImportError:
            raise ValueError(f"Expert '{name}' requires sklearn.ensemble.ExtraTreesRegressor")
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


class StackingMOERegressor:
    """Stacking ensemble with meta-learner on top of base experts.

    Unlike MOERegressor which uses fixed OOF-RMSE weights, StackingMOERegressor
    trains a meta-learner (Ridge for regression, LogisticRegression for classification)
    on the out-of-fold predictions from base experts.

    The meta-learner learns WHEN each expert is better, rather than just HOW MUCH
    weight to give them globally.

    For classification tasks (binder prediction), uses meta_learner_type="logistic".
    For regression tasks (efficacy prediction), uses meta_learner_type="ridge".

    Enhanced with:
    - Multiple meta-learner options: ridge, xgb_meta, lgb_meta, hgb_meta
    - Residual features: |oof_pred - y| per expert
    - Rank features: rank within expert predictions per sample
    - Cross-validated meta-learner to prevent overfitting

    Attributes:
        expert_names: List of base expert names.
        folds: CV folds for OOF predictions.
        meta_learner_type: "ridge" (regression) or "logistic" (classification).
        meta_learner: Fitted meta-learner.
        experts: Dictionary of fitted base experts.
        include_original_features: Whether to pass subset of original features to meta-learner.
    """

    def __init__(
        self,
        expert_names: List[str],
        folds: int = 5,
        meta_learner_type: str = "ridge",  # "ridge", "xgb_meta", "lgb_meta", "hgb_meta", "logistic"
        include_original_features: bool = False,
        n_meta_features: int = 20,
        random_state: int = 42,
        config: Optional[ExpertConfig] = None,
        use_residual_features: bool = True,   # NEW: include |oof - y| per expert
        use_rank_features: bool = True,       # NEW: include rank per expert prediction
        meta_cv_folds: int = 3,               # NEW: CV folds for meta-learner regularization
    ) -> None:
        self.expert_names = list(expert_names)
        self.folds = int(max(folds, 2))
        self.meta_learner_type = meta_learner_type
        self.include_original_features = include_original_features
        self.n_meta_features = n_meta_features
        self.random_state = int(random_state)
        self.config = config or ExpertConfig()
        self.scaler = StandardScaler()
        self.meta_scaler = StandardScaler()
        self.experts: Dict[str, RegressorLike] = {}
        self.meta_learner: Optional[Any] = None
        self.metrics: Dict[str, float] = {}
        self._fitted = False
        self._n_classes: int = 2  # Default for binary classification
        self.use_residual_features = use_residual_features
        self.use_rank_features = use_rank_features
        self.meta_cv_folds = meta_cv_folds

    def _build_meta_learner(self, is_classification: bool):
        """Build meta-learner based on type."""
        if is_classification:
            try:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=1000, random_state=self.random_state
                )
            except ImportError:
                return Ridge(alpha=1.0)

        # Regression meta-learners
        if self.meta_learner_type == "xgb_meta":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    verbosity=0,
                )
            except ImportError:
                logger.warning("XGBoost not available for meta-learner, falling back to Ridge")
                return Ridge(alpha=1.0)
        elif self.meta_learner_type == "lgb_meta":
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    verbosity=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available for meta-learner, falling back to Ridge")
                return Ridge(alpha=1.0)
        elif self.meta_learner_type == "hgb_meta":
            return HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.1,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
        else:
            # Default: Ridge
            return Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingMOERegressor":
        """Fit base experts OOF, then train meta-learner on OOF predictions.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,).

        Returns:
            self
        """
        Xs = self.scaler.fit_transform(X)
        n = len(y)
        k = len(self.expert_names)

        # Determine if regression or classification based on y values
        is_binary_class = set(y).issubset({0.0, 1.0}) or set(np.round(y)).issubset({0, 1})
        is_classification = is_binary_class and len(np.unique(y)) == 2

        # Adaptive fold count
        split_n = min(self.folds, max(2, n // 10)) if n >= 20 else 2
        kf = KFold(n_splits=split_n, shuffle=True, random_state=self.random_state)

        # Step 1: Generate OOF predictions from each expert
        oof_preds: Dict[str, np.ndarray] = {}
        for name in self.expert_names:
            oof = np.zeros(n, dtype=np.float64)
            for tr, va in kf.split(Xs):
                m = _make_expert(name, self.random_state, self.config)
                m.fit(Xs[tr], y[tr])
                oof[va] = m.predict(Xs[va])
            oof_preds[name] = oof

            rmse = float(np.sqrt(mean_squared_error(y, oof)))
            self.metrics[f"{name}_rmse"] = rmse

            # Fit final expert on all data
            final_m = _make_expert(name, self.random_state, self.config)
            final_m.fit(Xs, y)
            self.experts[name] = final_m

        # Step 2: Build meta-features
        meta_feats_list = [oof_preds[name].reshape(-1, 1) for name in self.expert_names]

        # Residual features: |oof_pred - y| per expert
        if self.use_residual_features:
            for name in self.expert_names:
                residual = np.abs(oof_preds[name] - y).reshape(-1, 1)
                meta_feats_list.append(residual)

        # Rank features: rank of each expert's prediction within sample
        if self.use_rank_features:
            oof_stack = np.stack([oof_preds[name] for name in self.expert_names], axis=1)  # (n, k)
            ranks = np.argsort(np.argsort(oof_stack, axis=1), axis=1) + 1  # (n, k), 1-indexed
            for j in range(k):
                meta_feats_list.append(ranks[:, j].reshape(-1, 1))

        # Expert weight features: softmax-normalized inverse-RMSE weights
        oof_stack = np.stack([oof_preds[name] for name in self.expert_names], axis=1)  # (n, k)
        residuals = np.abs(oof_stack - y.reshape(-1, 1))
        inv_res = 1.0 / (residuals + 1e-4)
        soft_weights = inv_res / inv_res.sum(axis=1, keepdims=True)  # (n, k)
        for j in range(k):
            meta_feats_list.append(soft_weights[:, j].reshape(-1, 1))

        if self.include_original_features and X.shape[1] >= self.n_meta_features:
            # Include subset of original features (dose/freq context)
            meta_feats_list.append(Xs[:, :self.n_meta_features])

        meta_features = np.concatenate(meta_feats_list, axis=1)
        meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 3: Fit meta-learner with optional CV regularization
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)

        if is_classification:
            y_binary = (y >= 0.5).astype(int)
            self.meta_learner = self._build_meta_learner(is_classification=True)
            self.meta_learner.fit(meta_features_scaled, y_binary)
            self._n_classes = 2
        else:
            self.meta_learner = self._build_meta_learner(is_classification=False)
            # Apply CV-regularized meta-learner if enabled
            if self.meta_cv_folds > 1 and n >= 50:
                meta_oof = np.zeros(n, dtype=np.float64)
                meta_kf = KFold(n_splits=self.meta_cv_folds, shuffle=True, random_state=self.random_state + 100)
                for tr, va in meta_kf.split(meta_features_scaled):
                    meta_clf = self._build_meta_learner(is_classification=False)
                    meta_clf.fit(meta_features_scaled[tr], y[tr])
                    meta_oof[va] = meta_clf.predict(meta_features_scaled[va])
                self.metrics["meta_oof_rmse"] = float(np.sqrt(mean_squared_error(y, meta_oof)))
                logger.info(
                    f"Stacking meta-learner ({self.meta_learner_type}) CV RMSE: "
                    f"{self.metrics['meta_oof_rmse']:.4f}"
                )
            self.meta_learner.fit(meta_features_scaled, y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions via stacking.

        Args:
            X: Feature matrix.

        Returns:
            Predictions from meta-learner.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        Xs = self.scaler.transform(X)
        n = X.shape[0]
        k = len(self.expert_names)

        # Get base expert predictions
        base_preds = np.column_stack([self.experts[name].predict(Xs) for name in self.expert_names])

        # Build meta-features
        meta_feats_list = [base_preds[:, j].reshape(-1, 1) for j in range(k)]

        # Residual features (use zero since we don't have y)
        if self.use_residual_features:
            for j in range(k):
                meta_feats_list.append(np.zeros((n, 1), dtype=np.float32))

        # Rank features
        if self.use_rank_features:
            ranks = np.argsort(np.argsort(base_preds, axis=1), axis=1) + 1
            for j in range(k):
                meta_feats_list.append(ranks[:, j].reshape(-1, 1))

        # Softmax weights (use uniform since we don't have y)
        for j in range(k):
            meta_feats_list.append(np.ones((n, 1), dtype=np.float32) / k)

        if self.include_original_features and X.shape[1] >= self.n_meta_features:
            meta_feats_list.append(Xs[:, :self.n_meta_features])

        meta_feats = np.concatenate(meta_feats_list, axis=1)
        meta_feats = np.nan_to_num(meta_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply meta-scaler
        meta_feats_scaled = self.meta_scaler.transform(meta_feats)

        # Apply meta-learner
        if hasattr(self.meta_learner, "predict_proba") and self._n_classes == 2:
            # For LogisticRegression, predict_proba gives class probabilities
            proba = self.meta_learner.predict_proba(meta_feats_scaled)
            # Return probability of class 1
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        elif hasattr(self.meta_learner, "predict"):
            return self.meta_learner.predict(meta_feats_scaled)
        else:
            raise RuntimeError("Meta-learner must have predict() method")

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
        """Return uniform weights (for compatibility)."""
        k = len(self.expert_names)
        return {name: 1.0 / k for name in self.expert_names}

    def __repr__(self) -> str:
        return (
            f"StackingMOERegressor(experts={self.expert_names}, folds={self.folds}, "
            f"meta={self.meta_learner_type}, fitted={self._fitted})"
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


# ═══════════════════════════════════════════════════════════════════════════════════════════
# 类生物门控机制 (Bio-Mimetic Gating) - Drug MOE
# ═══════════════════════════════════════════════════════════════════════════════════════════


class MembranePotential:
    """膜电位系统 - 模拟神经元的膜电位机制

    用于追踪每个Expert的激活历史:
    - 高电位 = 该Expert近期表现好
    - 低电位 = 该Expert近期表现差/长期未使用

    不应期机制:
    - 强激活后进入不应期，暂时降低被选中概率
    - 避免某些Expert被过度使用
    """

    def __init__(
        self,
        n_experts: int,
        decay: float = 0.85,
        boost: float = 0.35,
        refractory_duration: int = 3,
        rest_potential: float = 0.5,
    ) -> None:
        self.n_experts = n_experts
        self.decay = decay       # 电位衰减系数
        self.boost = boost   # 激活强化系数
        self.refractory_duration = refractory_duration  # 不应期长度

        # 膜电位: 每个Expert的电位值
        self.potential = np.full(n_experts, rest_potential, dtype=np.float32)

        # 不应期计数器
        self.refractory = np.zeros(n_experts, dtype=np.int32)

        # 历史记录
        self.history: list = []  # [(expert_id, score, is_strong)]

    def update(self, expert_id: int, score: float, is_strong: bool = False) -> None:
        """更新Expert电位

        Args:
            expert_id: Expert索引
            score: 预测分数 [0, 1]，越高表示表现越好
            is_strong: 是否为强激活 (触发不应期)
        """
        # 电位衰减
        self.potential[expert_id] *= self.decay

        # 激活提升
        self.potential[expert_id] += score * self.boost

        # 强激活触发不应期
        if is_strong:
            self.refractory[expert_id] = self.refractory_duration

        # 所有Expert不应期递减
        self.refractory = np.maximum(self.refractory - 1, 0)

        # 不应期惩罚
        if self.refractory[expert_id] > 0:
            self.potential[expert_id] *= 0.2

        # 限制在 [0, 1]
        self.potential[expert_id] = np.clip(self.potential[expert_id], 0.0, 1.0)

        # 记录历史
        self.history.append((expert_id, score, is_strong))
        if len(self.history) > 100:  # 保留最近100条
            self.history.pop(0)

    def get_state(self) -> np.ndarray:
        """获取当前膜电位状态"""
        return self.potential.copy()

    def get_refractory_penalty(self) -> np.ndarray:
        """获取不应期惩罚因子"""
        # 不应期内权重×0.2，否则×1.0
        return np.where(self.refractory > 0, 0.2, 1.0).astype(np.float32)


class EmotionalState:
    """情绪状态系统 - Drug-specific

    用于追踪当前药物分子的状态特征
    决定路由策略:
    - high novelty → exploration mode (愿意尝试新Expert)
    - low novelty + high risk → caution mode (保守策略)
    - good Lipinski → 倾向于"已验证"的Expert
    """

    def __init__(self) -> None:
        # 药物属性状态
        self.state = {
            "lipinski_score": 0.5,        # Lipinski规则符合度
            "structural_novelty": 0.5,   # 结构新颖度
            "admet_risk": 0.5,          # ADMET风险
            "synthetic_access": 0.5,   # 合成可达性
        }

        # 全局"情绪" - 控制路由策略
        self.mood = {
            "exploration": 0.5,         # 探索意愿
            "caution": 0.5,             # 谨慎程度
            "novelty_seeking": 0.3,     # 新靶点偏好
        }

    def update_from_molecule(self, lipinski_score: float, novelty: float,
                          admet_risk: float = 0.5, synthetic: float = 0.5) -> None:
        """根据分子特征更新状态

        Args:
            lipinski_score: Lipinski符合度 [0, 1]
            novelty: 结构新颖度 [0, 1]
            admet_risk: ADMET风险 [0, 1]
            synthetic: 合成可达性 [0, 1]
        """
        self.state["lipinski_score"] = lipinski_score
        self.state["structural_novelty"] = novelty
        self.state["admet_risk"] = admet_risk
        self.state["synthetic_access"] = synthetic

        # 更新情绪
        # 高新颖度 → 高探索欲
        self.mood["exploration"] = novelty

        # 低符合度 or 高风险 → 高谨慎
        self.mood["caution"] = min(1.0, (1 - lipinski_score) * 0.5 + admet_risk * 0.5)

    def get_routing_bias(self) -> dict:
        """获取路由偏置"""
        return {
            "exploration_bonus": (1 - self.mood["exploration"]) * 0.2,
            "caution_penalty": self.mood["caution"] * 0.15,
            "novelty_seeking": self.mood["novelty_seeking"],
        }

    def get_state_vector(self) -> np.ndarray:
        """获取状态向量用于特征"""
        return np.array([
            self.state["lipinski_score"],
            self.state["structural_novelty"],
            self.state["admet_risk"],
            self.state["synthetic_access"],
            self.mood["exploration"],
            self.mood["caution"],
            self.mood["novelty_seeking"],
        ], dtype=np.float32)


class BioGatedMOERegressor:
    """类生物门控MOE回归器

    在GatedMOE基础上引入:
    1. 膜电位系统 - 追踪Expert历史激活
    2. 不应期机制 - 防止Expert过度使用
    3. 情绪状态 - 根据输入分子特性动态调整路由
    4. 侧抑制 - 强化Winner，降低其他候选

    这模拟了"情绪影响认知" - 相同的输入在不同系统状态下可能产生不同的路由

    Attributes:
        expert_names: Expert名称列表
        folds: CV折数
        random_state: 随机种子
        config: Expert超参数
        membrane: 膜电位系统
        emotional: 情绪状态系统
        scaler: 特征标准化
        experts: 训练的Expert模型
    """

    def __init__(
        self,
        expert_names: List[str],
        folds: int = 4,
        gating_hidden: tuple = (32, 16),
        random_state: int = 42,
        config: Optional[ExpertConfig] = None,
        # 膜电位参数
        membrane_decay: float = 0.85,
        membrane_boost: float = 0.35,
        refractory_duration: int = 3,
        # 侧抑制参数
        lateral_enhance: float = 1.3,
        strong_threshold: float = 0.8,
    ) -> None:
        self.expert_names = list(expert_names)
        self.folds = int(max(folds, 2))
        self.gating_hidden = gating_hidden
        self.random_state = int(random_state)
        self.config = config or ExpertConfig()

        # 膜电位系统
        self.membrane = MembranePotential(
            n_experts=len(expert_names),
            decay=membrane_decay,
            boost=membrane_boost,
            refractory_duration=refractory_duration,
        )

        # 情绪状态系统
        self.emotional = EmotionalState()

        # 标准化
        self.scaler = StandardScaler()
        self.gating_scaler = StandardScaler()

        # Expert模型
        self.experts: Dict[str, RegressorLike] = {}
        self.gating_net: Optional[Any] = None
        self.metrics: Dict[str, float] = {}

        # 侧抑制参数
        self.lateral_enhance = lateral_enhance
        self.strong_threshold = strong_threshold

        self._fitted = False

    def set_molecule_state(self, lipinski_score: float, novelty: float,
                      admet_risk: float = 0.5, synthetic: float = 0.5) -> None:
        """设置输入分子的状态

        用于推理时指定分子的特性，系统会根据这些特性调整路由
        """
        self.emotional.update_from_molecule(lipinski_score, novelty, admet_risk, synthetic)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BioGatedMOERegressor":
        """训练Bio-Gated MOE

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)

        Returns:
            self
        """
        Xs = self.scaler.fit_transform(X)
        n = len(y)
        k = len(self.expert_names)

        # 自适应折数
        split_n = min(self.folds, max(2, n // 10)) if n >= 20 else 2
        kf = KFold(n_splits=split_n, shuffle=True, random_state=self.random_state)

        # Step 1: 训练基础Expert (OOF)
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

            # 训练最终模型
            final_m = _make_expert(name, self.random_state, self.config)
            final_m.fit(Xs, y)
            self.experts[name] = final_m

        # Step 2: 训练门控网络
        # 目标: 每个样本的per-expert权重
        oof_stack = np.stack([oof_preds[name] for name in self.expert_names], axis=1)
        residuals = np.abs(oof_stack - y.reshape(-1, 1))
        inv_res = 1.0 / (residuals + 1e-4)
        gate_targets = inv_res / inv_res.sum(axis=1, keepdims=True)

        # 门控网络输入
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

    def _apply_bio_gating(
        self,
        base_weights: np.ndarray,
        membrane_state: np.ndarray,
        emotional_bias: dict,
    ) -> np.ndarray:
        """应用生物门控调制

        Args:
            base_weights: 基础门控权重 (n_samples, n_experts)
            membrane_state: 膜电位状态 (n_experts,)
            emotional_bias: 情绪偏置

        Returns:
            调制后的权重
        """
        n, k = base_weights.shape

        # 1. 膜电位调制
        # 高电位 → 倾向于已验证的Expert (电位×权重)
        # 低电位 → 更愿意尝试新Expert (探索模式)
        membrane_mod = membrane_state[np.newaxis, :]  # (1, k)
        电位_modifier = 0.7 + 0.5 * membrane_mod  # [0.7, 1.2]

        # 2. 情绪调制 - 探索模式加成
        exploration = emotional_bias.get("exploration_bonus", 0.0)
        # 对低电位(未充分探索)的Expert加分
        unexplored_bonus = (1 - membrane_mod) * exploration

        # 3. 不应期惩罚
        refractory_penalty = self.membrane.get_refractory_penalty()[np.newaxis, :]  # (1, k)

        # 4. 侧抑制 - 强化Top-1
        top_idx = np.argmax(base_weights, axis=1)  # (n,)
        lateral = np.ones_like(base_weights)
        for i in range(n):
            lateral[i, top_idx[i]] = self.lateral_enhance

        # 综合计算
        final_weights = base_weights * 电位_modifier * refractory_penalty * lateral
        final_weights += unexplored_bonus

        # Softmax归一化
        w_max = final_weights.max(axis=1, keepdims=True)
        exp_w = np.exp(final_weights - w_max)
        final_weights = exp_w / exp_w.sum(axis=1, keepdims=True)

        return final_weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """获取生物门控集成预测

        Args:
            X: 特征矩阵

        Returns:
            预测值
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        Xs = self.scaler.transform(X)
        n = X.shape[0]
        k = len(self.expert_names)

        # 基础门控权重
        Xg = self.gating_scaler.transform(Xs)
        logits = self.gating_net.predict(Xg).astype(np.float32)

        # Softmax
        logits_max = logits.max(axis=1, keepdims=True)
        base_weights = np.exp(logits - logits_max)
        base_weights = base_weights / base_weights.sum(axis=1, keepdims=True)

        # 获取膜电位状态和情绪偏置
        membrane_state = self.membrane.get_state()
        emotional_bias = self.emotional.get_routing_bias()

        # 应用生物门控
        gate_weights = self._apply_bio_gating(base_weights, membrane_state, emotional_bias)

        # 加权预测
        out = np.zeros(n, dtype=np.float32)
        for j, name in enumerate(self.expert_names):
            pred = self.experts[name].predict(Xs).astype(np.float32)
            out += gate_weights[:, j] * pred

        return out

    def predict_with_routing(self, X: np.ndarray) -> tuple:
        """预测并返回路由信息

        Returns:
            (predictions, routing_weights, membrane_state)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        Xs = self.scaler.transform(X)
        n = X.shape[0]
        k = len(self.expert_names)

        # 基础门控
        Xg = self.gating_scaler.transform(Xs)
        logits = self.gating_net.predict(Xg).astype(np.float32)
        logits_max = logits.max(axis=1, keepdims=True)
        base_weights = np.exp(logits - logits_max)
        base_weights = base_weights / base_weights.sum(axis=1, keepdims=True)

        # 生物门控
        membrane_state = self.membrane.get_state()
        emotional_bias = self.emotional.get_routing_bias()
        gate_weights = self._apply_bio_gating(base_weights, membrane_state, emotional_bias)

        # 预测
        out = np.zeros(n, dtype=np.float32)
        for j, name in enumerate(self.expert_names):
            pred = self.experts[name].predict(Xs).astype(np.float32)
            out += gate_weights[:, j] * pred

        return out, gate_weights, membrane_state

    def update_membrane(self, expert_predictions: np.ndarray, target: np.ndarray) -> None:
        """根据预测结果更新膜电位

        在推理后调用，根据每个Expert的误差更新电位

        Args:
            expert_predictions: 各Expert的预测 (n_experts,)
            target: 真实值 (scalar)
        """
        if not self._fitted:
            return

        # 找最佳Expert
        errors = np.abs(expert_predictions - target)
        best_idx = int(np.argmin(errors))
        best_error = errors[best_idx]

        # 判断是否为"强激活"
        is_strong = best_error < (1 - self.strong_threshold)

        # 分数 = 1 - error (越接近1越好)
        score = max(0.0, 1.0 - best_error)

        # 更新膜电位
        self.membrane.update(best_idx, score, is_strong)

    def predict_experts(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """获取各Expert的独立预测"""
        Xs = self.scaler.transform(X)
        out: Dict[str, np.ndarray] = {}
        for name, model in self.experts.items():
            out[name] = model.predict(Xs).astype(np.float32)
        return out

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """从Expert分歧估计不确定性"""
        expert_pred = self.predict_experts(X)
        if not expert_pred:
            return np.zeros((X.shape[0],), dtype=np.float32)
        arr = np.vstack([p for p in expert_pred.values()]).astype(np.float32)
        return arr.std(axis=0).astype(np.float32)

    def explain_weights(self) -> Dict[str, float]:
        """返回当前膜电位作为权重"""
        membrane = self.membrane.get_state()
        return {name: float(w) for name, w in zip(self.expert_names, membrane)}

    def get_emotion_state(self) -> dict:
        """获取当前情绪状态"""
        return {
            "state": dict(self.emotional.state),
            "mood": dict(self.emotional.mood),
        }

    def get_membrane_state(self) -> dict:
        """获取膜电位状态"""
        membrane = self.membrane.get_state()
        refractory = self.membrane.refractory.copy()
        return {
            name: {"potential": float(membrane[i]), "refractory": int(refractory[i])}
            for i, name in enumerate(self.expert_names)
        }

    def __repr__(self) -> str:
        return (
            f"BioGatedMOERegressor(experts={self.expert_names}, folds={self.folds}, "
            f"membrane_decay={self.membrane.decay}, fitted={self._fitted})"
        )


# ═══════════════════════════════════════════════════════════════════════════════════
# Export aliases for drug module
# ═══════════════════════════════════════════════════════════════════════════════════════════

# Re-export all classes
__all__ = [
    "ComputeProfile",
    "ExpertConfig",
    "MOERegressor",
    "StackingMOERegressor",
    "GatedMOERegressor",
    "BioGatedMOERegressor",  # NEW
    "MembranePotential",      # NEW
    "EmotionalState",       # NEW
    "choose_compute_profile",
    "EXPERT_CONFIG_DRUG",
    "EXPERT_CONFIG_DRUG_ULTRA",
    "EXPERT_CONFIG_EPITOPE",
]

