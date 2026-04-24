"""
src.common.dataset_autofetch — auto-detect and fetch training-ready datasets.

Re-exports from ``confluencia_shared.utils.dataset_autofetch``.
"""

from confluencia_shared.utils.dataset_autofetch import (  # noqa: F401
    FetchedDataset,
    fetch_datasets,
)

__all__ = ["FetchedDataset", "fetch_datasets"]
