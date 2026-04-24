"""Minimal regression metric helpers expected by the project."""
from __future__ import annotations

from typing import Sequence, Dict
import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_a = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred, dtype=np.float64)
    if y_true_a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))


def reg_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_a = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred, dtype=np.float64)
    if y_true_a.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    mae = float(np.mean(np.abs(y_true_a - y_pred_a)))
    rm = float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))
    denom = float(np.sum((y_true_a - float(np.mean(y_true_a))) ** 2))
    if denom == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - float(np.sum((y_true_a - y_pred_a) ** 2)) / denom
    return {"mae": mae, "rmse": rm, "r2": r2}
