"""Canonical amino-acid constants (minimal, local shim).

These values approximate what the main `confluencia_shared` package
would provide; they are sufficient for the in-repo feature code to run.
"""
from __future__ import annotations

from typing import List, Dict, Set

AA_ORDER: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(AA_ORDER)}

# Simple biochemical groupings (reasonable defaults)
HYDROPHOBIC: Set[str] = set(list("AILMFWVY"))
POLAR: Set[str] = set(list("STNQCYH"))
ACIDIC: Set[str] = set(list("DE"))
BASIC: Set[str] = set(list("KRH"))

__all__ = ["AA_ORDER", "AA_TO_IDX", "HYDROPHOBIC", "POLAR", "ACIDIC", "BASIC"]
