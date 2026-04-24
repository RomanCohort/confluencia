"""Small typing protocols to satisfy imports used in the project."""
from __future__ import annotations

from typing import Protocol, Any


class PredictableRegressor(Protocol):
    def predict(self, X: Any) -> Any: ...


__all__ = ["PredictableRegressor"]
