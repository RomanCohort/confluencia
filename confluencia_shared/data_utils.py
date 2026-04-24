"""
Shared data utilities for Confluencia.

Common DataFrame and array manipulation helpers used across modules.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def resolve_label(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    """Extract a numeric label column from a DataFrame.

    Args:
        df: Source DataFrame.
        name: Column name to resolve.

    Returns:
        Float32 numpy array if column exists, None otherwise.
        NaN values are filled with 0.0.
    """
    if name not in df.columns:
        return None
    vals = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return vals


__all__ = ["resolve_label"]
