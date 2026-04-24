"""
Differential evolution optimizer (numpy implementation).
Provides a general-purpose DE solver for continuous inputs and a small helper
that optimizes a numeric input vector to maximize/minimize a scalar objective.

API:
- differential_evolution(objective, bounds, pop_size=20, F=0.8, CR=0.9, max_iter=100, minimize=True)
- de_optimize(objective, dim, bounds, **kwargs)

The implementation is intentionally minimal and dependency-free (NumPy only).
"""
from typing import Callable, Sequence, Tuple

import numpy as np


def _ensure_bounds(bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    b = np.array(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must be sequence of (low, high) pairs")
    return b


def differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    pop_size: int = 20,
    F: float = 0.8,
    CR: float = 0.9,
    max_iter: int = 100,
    minimize: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Run differential evolution to optimize a continuous objective.

    Parameters
    - objective: callable(x) -> scalar. x is 1D numpy array of length D.
    - bounds: sequence of (low, high) for each dimension (length D).
    - pop_size: population size (recommended >= 5*D).
    - F: differential weight.
    - CR: crossover probability.
    - max_iter: number of generations.
    - minimize: whether to minimize (True) or maximize (False).
    - seed: random seed.

    Returns: (best_x, best_value)
    """
    rng = np.random.default_rng(seed)
    b = _ensure_bounds(bounds)
    D = b.shape[0]
    # population size at least 4
    NP = max(int(pop_size), 4)
    # initialize population uniformly in bounds
    pop = rng.random((NP, D)) * (b[:, 1] - b[:, 0])[None, :] + b[:, 0][None, :]

    # evaluate
    vals = np.array([objective(ind) for ind in pop], dtype=float)
    # convert to minimization internally
    if not minimize:
        vals = -vals

    best_idx = int(np.argmin(vals))
    best = pop[best_idx].copy()
    best_val = float(vals[best_idx])

    for gen in range(int(max_iter)):
        for i in range(NP):
            idxs = [idx for idx in range(NP) if idx != i]
            a, b_, c = pop[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b_ - c), b[:, 0], b[:, 1])
            cross = rng.random(D) < CR
            if not np.any(cross):
                cross[rng.integers(0, D)] = True
            trial = np.where(cross, mutant, pop[i])
            val = objective(trial)
            v = -val if not minimize else val
            if v < vals[i]:
                pop[i] = trial
                vals[i] = v
                if v < best_val:
                    best_val = v
                    best = trial.copy()
        # (optional) early exit could be added here
    final_val = -best_val if not minimize else best_val
    return best, float(final_val)


def de_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    maximize: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    Convenience wrapper: maximize objective if maximize=True.
    Returns (best_x, best_obj_value)
    """
    best_x, val = differential_evolution(objective, bounds, minimize=not maximize, **kwargs)
    return best_x, val
