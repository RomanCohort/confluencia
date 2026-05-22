"""A tiny, dependency-free differential-evolution-like optimizer used as a fallback.

This implementation is intentionally simple and deterministic enough for small
searches used by the UI (suggestion search). It returns (best_vector, best_value).
"""
from __future__ import annotations

from typing import Sequence, Tuple, Callable
import numpy as np


def de_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    *,
    maximize: bool = True,
    pop_size: int = 50,
    max_iter: int = 100,
    F: float = 0.8,
    CR: float = 0.9,
) -> Tuple[np.ndarray, float]:
    dims = len(bounds)
    rng = np.random.default_rng()

    best_vec = None
    best_val = -np.inf if maximize else np.inf

    # Simple random-search baseline (sufficient for UI suggestions)
    n_samples = max(1000, pop_size * max_iter)
    for _ in range(n_samples):
        vec = np.array([rng.uniform(l, h) for (l, h) in bounds], dtype=float)
        try:
            val = float(objective(vec))
        except Exception:
            continue
        if maximize:
            if val > best_val:
                best_val = val
                best_vec = vec
        else:
            if val < best_val:
                best_val = val
                best_vec = vec

    if best_vec is None:
        best_vec = np.array([(l + h) / 2.0 for (l, h) in bounds], dtype=float)
        best_val = float(objective(best_vec))

    return best_vec, float(best_val)
