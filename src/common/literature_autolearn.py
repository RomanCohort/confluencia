"""
src.common.literature_autolearn — automated literature search & dataset hinting.

Re-exports from ``confluencia_shared.utils.literature_autolearn``.
"""

from confluencia_shared.utils.literature_autolearn import (  # noqa: F401
    LiteratureItem,
    literature_autolearn,
)

__all__ = ["LiteratureItem", "literature_autolearn"]
