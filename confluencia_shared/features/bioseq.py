"""
Shared biochemical sequence constants and utilities.

This module provides canonical definitions for amino acid properties,
eliminating duplication across epitope and drug modules.
"""
from __future__ import annotations

from typing import Set, Dict, Tuple

# Canonical amino acid order (20 standard amino acids)
AA_ORDER: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

# Mapping from amino acid to index
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}

# Amino acid property categories
HYDROPHOBIC: Set[str] = set("AVILMFWY")  # Aliphatic + Aromatic
POLAR: Set[str] = set("STNQCY")  # Polar uncharged
ACIDIC: Set[str] = set("DE")  # Negatively charged
BASIC: Set[str] = set("KRH")  # Positively charged
AROMATIC: Set[str] = set("FWY")  # Aromatic ring
SMALL: Set[str] = set("AGST")  # Small side chains
TINY: Set[str] = set("AGS")  # Very small

# Proline and Glycine (flexibility markers)
PROLINE: Set[str] = set("P")
GLYCINE: Set[str] = set("G")

# Combined categories
CHARGED: Set[str] = ACIDIC | BASIC
SPECIAL: Set[str] = PROLINE | GLYCINE


def is_valid_aa(ch: str) -> bool:
    """Check if character is a standard amino acid."""
    return ch in AA_TO_IDX


def aa_composition(seq: str) -> Dict[str, float]:
    """Calculate amino acid composition as fractions.

    Args:
        seq: Amino acid sequence.

    Returns:
        Dictionary mapping AA to fraction in sequence.
    """
    seq = str(seq or "").upper().replace(" ", "")
    if not seq:
        return {aa: 0.0 for aa in AA_ORDER}

    counts = {aa: 0 for aa in AA_ORDER}
    total = 0
    for ch in seq:
        if ch in AA_TO_IDX:
            counts[ch] += 1
            total += 1

    if total == 0:
        return {aa: 0.0 for aa in AA_ORDER}
    return {aa: counts[aa] / total for aa in AA_ORDER}


def hydrophobic_fraction(seq: str) -> float:
    """Calculate fraction of hydrophobic residues."""
    seq = str(seq or "").upper()
    if not seq:
        return 0.0
    count = sum(1 for ch in seq if ch in HYDROPHOBIC)
    return count / len(seq)


def polar_fraction(seq: str) -> float:
    """Calculate fraction of polar residues."""
    seq = str(seq or "").upper()
    if not seq:
        return 0.0
    count = sum(1 for ch in seq if ch in POLAR)
    return count / len(seq)


def charge_fraction(seq: str) -> Tuple[float, float]:
    """Calculate fractions of acidic and basic residues.

    Returns:
        Tuple of (acidic_fraction, basic_fraction).
    """
    seq = str(seq or "").upper()
    if not seq:
        return (0.0, 0.0)
    acidic = sum(1 for ch in seq if ch in ACIDIC)
    basic = sum(1 for ch in seq if ch in BASIC)
    return (acidic / len(seq), basic / len(seq))


__all__ = [
    "AA_ORDER",
    "AA_TO_IDX",
    "HYDROPHOBIC",
    "POLAR",
    "ACIDIC",
    "BASIC",
    "AROMATIC",
    "SMALL",
    "TINY",
    "PROLINE",
    "GLYCINE",
    "CHARGED",
    "SPECIAL",
    "is_valid_aa",
    "aa_composition",
    "hydrophobic_fraction",
    "polar_fraction",
    "charge_fraction",
]
