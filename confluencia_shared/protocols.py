"""
Shared protocol definitions for Confluencia.

Provides common Protocol classes used across epitope and drug modules.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PredictableRegressor(Protocol):
    """Protocol for sklearn-compatible regressors."""

    def fit(self, X, y, *args, **kwargs) -> Any:  # pragma: no cover
        ...

    def predict(self, X, *args, **kwargs) -> Any:  # pragma: no cover
        ...


__all__ = ["PredictableRegressor"]
