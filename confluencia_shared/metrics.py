"""
Shared regression metrics for Confluencia.

Provides unified metric computation functions used across epitope and drug modules.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def reg_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R2).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        prefix: Optional prefix for metric keys (e.g. "eff_" -> "eff_mae").

    Returns:
        Dict with keys like "{prefix}mae", "{prefix}rmse", "{prefix}r2".
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    sep = "" if not prefix or prefix.endswith("_") else "_"
    key = lambda name: f"{prefix}{sep}{name}"

    if y_true.size == 0:
        return {key("mae"): 0.0, key("rmse"): 0.0, key("r2"): 0.0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse_val = float(np.sqrt(np.mean(err * err)))

    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom < 1e-8:
        r2_val = 0.0
    else:
        r2_val = float(1.0 - np.sum(err * err) / denom)

    return {key("mae"): mae, key("rmse"): rmse_val, key("r2"): r2_val}


__all__ = ["rmse", "reg_metrics"]
