"""Minimal shim for `confluencia_shared` used by this repository.

This package provides small, self-contained implementations of the
symbols the project imports from `confluencia_shared`. It is intended
as a compatibility layer for running the app without the external
`confluencia_shared` distribution.
"""

__all__ = [
    "features",
    "metrics",
    "data_utils",
    "utils",
    "training",
    "models",
    "protocols",
    "optim",
]
