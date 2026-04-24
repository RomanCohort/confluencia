from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from .ctm import CTMParams

PARAM_NAMES = ["ka", "kd", "ke", "km", "signal_gain"]


def _clip_params(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=np.float32).copy()
    if out.ndim == 1:
        out = out[None, :]
    # Keep parameters in physically plausible ranges.
    out[:, 0] = np.clip(out[:, 0], 0.02, 0.9)  # ka
    out[:, 1] = np.clip(out[:, 1], 0.02, 0.9)  # kd
    out[:, 2] = np.clip(out[:, 2], 0.02, 0.9)  # ke
    out[:, 3] = np.clip(out[:, 3], 0.01, 0.8)  # km
    out[:, 4] = np.clip(out[:, 4], 0.2, 4.0)   # signal_gain
    return out


def heuristic_param_targets(
    binding: np.ndarray,
    immune: np.ndarray,
    inflammation: np.ndarray,
    dose: np.ndarray,
    freq: np.ndarray,
) -> np.ndarray:
    b = np.clip(np.asarray(binding, dtype=np.float32), 0.0, 1.0)
    i = np.clip(np.asarray(immune, dtype=np.float32), 0.0, 1.0)
    inf = np.clip(np.asarray(inflammation, dtype=np.float32), 0.0, 1.0)
    d = np.clip(np.asarray(dose, dtype=np.float32), 0.0, None)
    f = np.clip(np.asarray(freq, dtype=np.float32), 0.0, None)

    ka = 0.14 + 0.26 * b + 0.015 * np.log1p(d)
    kd = 0.11 + 0.24 * i + 0.02 * np.log1p(f)
    ke = 0.09 + 0.17 * (1.0 - inf) + 0.01 * np.log1p(f)
    km = 0.05 + 0.24 * inf + 0.01 * np.log1p(d)
    gain = 0.75 + 1.25 * (0.6 * b + 0.4 * i) - 0.1 * inf

    Y = np.stack([ka, kd, ke, km, gain], axis=1).astype(np.float32)
    return _clip_params(Y)


@dataclass
class CTMParamBundle:
    base_model: MultiOutputRegressor
    residual_model: MultiOutputRegressor
    source: str
    group_bias: Dict[str, np.ndarray]


class CTMParamModel:
    """Hierarchical CTM parameter learner.

    Structure:
    - Global base model predicts shared kinetics trends.
    - Group-level bias captures cohort/patient-stratum shifts.
    - Residual model captures individual-level deviation.
    """

    def __init__(self, random_state: int = 42) -> None:
        # Keep RF single-threaded to avoid joblib ThreadPool issues in some Python 3.13 environments.
        base = RandomForestRegressor(n_estimators=180, max_depth=10, random_state=random_state, n_jobs=1)
        residual = RandomForestRegressor(n_estimators=140, max_depth=8, random_state=random_state + 1, n_jobs=1)
        self.base_model = MultiOutputRegressor(base)
        self.residual_model = MultiOutputRegressor(residual)
        self.group_bias: Dict[str, np.ndarray] = {}
        self.group_encoder = LabelEncoder()
        self._fit_group_ids: List[str] = []
        self.source = "learned"

    def fit(self, X_ctm: np.ndarray, Y_params: np.ndarray, group_ids: np.ndarray | None = None) -> "CTMParamModel":
        Y = _clip_params(Y_params)
        n = X_ctm.shape[0]
        if group_ids is None:
            g = np.array(["G0"] * n, dtype=object)
        else:
            g = np.asarray(group_ids, dtype=object)

        self._fit_group_ids = [str(v) for v in g.tolist()]
        _ = self.group_encoder.fit(np.array(self._fit_group_ids, dtype=object))

        self.base_model.fit(X_ctm, Y)
        y_base = np.asarray(self.base_model.predict(X_ctm), dtype=np.float32)

        # Estimate group bias in parameter space.
        residual_1 = Y - y_base
        self.group_bias = {}
        for grp in sorted(set(self._fit_group_ids)):
            mask = np.array([str(x) == grp for x in self._fit_group_ids], dtype=bool)
            if mask.any():
                self.group_bias[str(grp)] = residual_1[mask].mean(axis=0).astype(np.float32)

        bias_mat = np.stack([self.group_bias.get(str(x), np.zeros((5,), dtype=np.float32)) for x in g], axis=0)
        residual_2 = residual_1 - bias_mat
        self.residual_model.fit(X_ctm, residual_2)
        return self

    def _group_bias_matrix(self, group_ids: np.ndarray) -> np.ndarray:
        g = np.asarray(group_ids, dtype=object)
        return np.stack([self.group_bias.get(str(x), np.zeros((5,), dtype=np.float32)) for x in g], axis=0)

    def predict_decomposed(self, X_ctm: np.ndarray, group_ids: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = X_ctm.shape[0]
        if group_ids is None:
            g = np.array(["G0"] * n, dtype=object)
        else:
            g = np.asarray(group_ids, dtype=object)

        y_base = np.asarray(self.base_model.predict(X_ctm), dtype=np.float32)
        y_group = self._group_bias_matrix(g)
        y_resid = np.asarray(self.residual_model.predict(X_ctm), dtype=np.float32)
        y_final = _clip_params(y_base + y_group + y_resid)
        return y_final, y_base, y_group + y_resid

    def predict_params(self, X_ctm: np.ndarray, group_ids: np.ndarray | None = None) -> np.ndarray:
        y, _, _ = self.predict_decomposed(X_ctm, group_ids=group_ids)
        return y

    def to_params(self, arr: np.ndarray) -> CTMParams:
        v = _clip_params(arr)[0]
        return CTMParams(
            ka=float(v[0]),
            kd=float(v[1]),
            ke=float(v[2]),
            km=float(v[3]),
            signal_gain=float(v[4]),
        )

    def export_bundle(self) -> CTMParamBundle:
        return CTMParamBundle(
            base_model=self.base_model,
            residual_model=self.residual_model,
            source=self.source,
            group_bias=self.group_bias,
        )
