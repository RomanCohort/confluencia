"""Small utilities for resolving labels from dataframes."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def resolve_label(df: pd.DataFrame, label: str) -> Optional[np.ndarray]:
    """Return a numeric numpy array for `label` if present and has values,
    otherwise return None.
    """
    if label not in df.columns:
        return None
    col = pd.to_numeric(df[label], errors="coerce")
    if col.isna().all():
        return None
    return col.to_numpy(dtype=np.float32)
