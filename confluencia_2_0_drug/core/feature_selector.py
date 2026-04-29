"""
Feature selection / pruning utilities.
Three-stage pipeline: variance threshold → RF importance top-k → correlation pruning.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class FeatureSelector:
    """Three-stage feature selection.

    1. Variance threshold: remove near-constant features (var < threshold)
    2. RF importance top-k: train RF, keep top-k features by importance
    3. Correlation pruning: remove one of each pair with |r| > threshold
    """

    def __init__(
        self,
        top_k: int = 512,
        var_thresh: float = 1e-6,
        corr_thresh: float = 0.95,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._var_selector = None
        self._keep_mask: Optional[np.ndarray] = None

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply selection pipeline and return pruned matrix + names.

        Args:
            X: Feature matrix (n, d).
            y: Target vector (n,).
            feature_names: Optional feature names.

        Returns:
            (X_pruned, names_pruned).
        """
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.ensemble import RandomForestRegressor

        if X.shape[1] == 0:
            return X, feature_names or []

        names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        n_original = X.shape[1]

        # Stage 1: Variance threshold
        self._var_selector = VarianceThreshold(threshold=self.var_thresh)
        try:
            X_vt = self._var_selector.fit_transform(X)
        except ValueError:
            # All features have zero variance — skip selection
            self._var_selector = None
            self._keep_mask = np.arange(X.shape[1])
            return X, names
        vt_mask = self._var_selector.get_support()
        vt_names = [n for n, keep in zip(names, vt_mask) if keep]

        if X_vt.shape[1] == 0:
            return X_vt, vt_names

        # Stage 2: RF importance top-k
        k = min(self.top_k, X_vt.shape[1])
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_vt, y)
        importances = rf.feature_importances_
        top_k_idx = np.argsort(importances)[::-1][:k]
        X_topk = X_vt[:, top_k_idx]
        topk_names = [vt_names[i] for i in top_k_idx]

        if X_topk.shape[1] <= 1:
            # Map top_k_idx back to original var-support indices
            vt_support = np.where(vt_mask)[0]
            self._keep_mask = vt_support[top_k_idx]
            return X_topk, topk_names

        # Stage 3: Correlation pruning
        corr_mat = np.corrcoef(X_topk.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        n_topk = X_topk.shape[1]
        to_remove: set = set()
        for i in range(n_topk):
            if i in to_remove:
                continue
            for j in range(i + 1, n_topk):
                if j not in to_remove and abs(corr_mat[i, j]) > self.corr_thresh:
                    to_remove.add(j)

        keep_idx = [i for i in range(n_topk) if i not in to_remove]
        X_pruned = X_topk[:, keep_idx]
        pruned_names = [topk_names[i] for i in keep_idx]

        # Store mask for transform()
        vt_support = np.where(vt_mask)[0]
        self._keep_mask = vt_support[top_k_idx[keep_idx]]

        return X_pruned, pruned_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply previously fitted selection to new data."""
        if self._var_selector is None or self._keep_mask is None:
            raise RuntimeError("fit_transform must be called before transform")
        X_vt = self._var_selector.transform(X)
        return X_vt[:, self._keep_mask]
