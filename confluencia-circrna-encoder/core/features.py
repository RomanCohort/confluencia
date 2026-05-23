"""
Feature construction for circRNA analysis.

Mirrors drug module's features.py structure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Default gene columns
DEFAULT_GENE_COLS = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"]


def build_gene_features(
    gene_expr: Dict[str, float],
    gene_cols: List[str] = DEFAULT_GENE_COLS,
) -> np.ndarray:
    """
    Build gene expression feature vector.

    Args:
        gene_expr: Dict of gene expression values
        gene_cols: Gene column names

    Returns:
        Feature vector (len(gene_cols))
    """
    return np.array([gene_expr.get(g, 0.5) for g in gene_cols])


def build_sequence_features(
    sequence: str,
) -> Dict[str, float]:
    """
    Build sequence-derived features (non-encoded).

    Args:
        sequence: RNA sequence

    Returns:
        Dict with sequence statistics
    """
    seq = sequence.upper()
    length = len(seq)

    # Base composition
    a_count = sum(1 for c in seq if c == "A")
    u_count = sum(1 for c in seq if c == "U")
    g_count = sum(1 for c in seq if c == "G")
    c_count = sum(1 for c in seq if c == "C")

    # Ratios
    gc_content = (g_count + c_count) / max(length, 1)
    au_content = (a_count + u_count) / max(length, 1)
    purine_content = (a_count + g_count) / max(length, 1)

    # Complexity (entropy-like)
    bases = {"A": a_count, "U": u_count, "G": g_count, "C": c_count}
    probs = [bases[b] / max(length, 1) for b in bases]
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

    return {
        "length": length,
        "gc_content": gc_content,
        "au_content": au_content,
        "purine_content": purine_content,
        "entropy": entropy,
        "a_count": a_count,
        "u_count": u_count,
        "g_count": g_count,
        "c_count": c_count,
    }


def build_feature_matrix(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    gene_cols: List[str] = DEFAULT_GENE_COLS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        sequence_col: Column containing sequences
        gene_cols: Gene expression columns

    Returns:
        Feature matrix, feature names
    """
    features = []
    feature_names = []

    for idx, row in df.iterrows():
        row_features = []

        # Gene features
        gene_values = [row.get(g, 0.5) for g in gene_cols]
        row_features.extend(gene_values)
        feature_names.extend([f"gene_{g}" for g in gene_cols])

        # Sequence features
        seq = str(row.get(sequence_col, ""))
        seq_feats = build_sequence_features(seq)
        row_features.extend([seq_feats.get(k, 0) for k in seq_feats])
        feature_names.extend(list(seq_feats.keys()))

        features.append(row_features)

    return np.array(features), feature_names


def get_default_gene_expression() -> Dict[str, float]:
    """Get default gene expression values."""
    return {
        "TROP2": 7.2,
        "NECTIN4": 5.1,
        "LIV-1": 3.5,
        "B7-H4": 6.0,
        "MKI67": 8.0,
        "MYC": 4.5,
    }