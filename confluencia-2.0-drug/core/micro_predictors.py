from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


MICRO_TARGETS: List[str] = [
    "target_binding",
    "immune_activation",
    "immune_cell_activation",
    "inflammation_risk",
    "toxicity_risk",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def proxy_micro_labels(X: np.ndarray) -> np.ndarray:
    # Deterministic weak labels for unlabeled demo mode.
    if X.size == 0:
        return np.zeros((0, len(MICRO_TARGETS)), dtype=np.float32)

    n = X.shape[0]
    a = X[:, : min(32, X.shape[1])].mean(axis=1)
    b = X[:, min(32, X.shape[1] // 3): min(96, X.shape[1])].mean(axis=1)
    c = X[:, -min(16, X.shape[1]):].mean(axis=1)

    binding = _sigmoid(3.0 * a + 0.8 * c)
    immune = _sigmoid(2.2 * b + 0.5 * a)
    immune_cell = _sigmoid(1.8 * b + 0.9 * a - 0.3 * c)
    inflammation = _sigmoid(1.5 * c - 0.7 * b)
    toxicity = _sigmoid(1.7 * c + 0.6 * a - 0.4 * b)

    y = np.stack([binding, immune, immune_cell, inflammation, toxicity], axis=1).astype(np.float32)
    noise = np.random.default_rng(42).normal(0, 0.03, size=y.shape).astype(np.float32)
    return np.clip(y + noise, 0.0, 1.0)


@dataclass
class MicroPredictorBundle:
    model: MultiOutputRegressor
    used_proxy_labels: bool


class MicroPredictor:
    def __init__(self, random_state: int = 42) -> None:
        # Keep RF single-threaded to avoid joblib ThreadPool issues in some Python 3.13 environments.
        base = RandomForestRegressor(n_estimators=220, max_depth=12, random_state=random_state, n_jobs=1)
        self.model = MultiOutputRegressor(base)
        self.used_proxy_labels = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MicroPredictor":
        if y is None:
            y = proxy_micro_labels(X)
            self.used_proxy_labels = True
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        pred = np.asarray(self.model.predict(X), dtype=np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        return {name: pred[:, i] for i, name in enumerate(MICRO_TARGETS)}

    def export_bundle(self) -> MicroPredictorBundle:
        return MicroPredictorBundle(model=self.model, used_proxy_labels=self.used_proxy_labels)
