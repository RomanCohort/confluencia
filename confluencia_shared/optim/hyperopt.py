from typing import Any, Dict, Tuple, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def run_hyper_search(
    estimator,
    param_grid: Dict[str, Any],
    X,
    y,
    strategy: str = "grid",
    n_iter: int = 20,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
):
    """
    Run GridSearchCV or RandomizedSearchCV on given estimator.

    Returns (best_estimator, best_params, cv_results_)
    """
    strategy = str(strategy).lower()
    if strategy == "grid":
        gs = GridSearchCV(estimator, param_grid, cv=int(cv), scoring=scoring, n_jobs=int(n_jobs), refit=True)
        gs.fit(X, y)
    elif strategy in {"random", "randomized"}:
        rs = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=int(n_iter),
            cv=int(cv),
            scoring=scoring,
            n_jobs=int(n_jobs),
            random_state=random_state,
            refit=True,
        )
        rs.fit(X, y)
        gs = rs
    else:
        raise ValueError("Unknown strategy, choose 'grid' or 'random'")

    return gs.best_estimator_, gs.best_params_, gs.cv_results_
