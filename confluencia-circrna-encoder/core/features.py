"""
Feature construction for circRNA analysis.

Mirrors drug module's features.py structure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CircRNAFeatureExtractor:
    """
    Feature extractor for circRNA sequences.

    Extracts sequence-based features for immunogenicity prediction.
    """

    NUCS = ['A', 'U', 'G', 'C']

    def __init__(self):
        self.feature_names = []

    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract all features from a circRNA sequence.

        Args:
            sequence: circRNA sequence string

        Returns:
            Feature vector (numpy array)
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        features = []
        self.feature_names = []

        counts = Counter(seq)

        # 1. Nucleotide frequencies (4)
        for nuc in self.NUCS:
            features.append(counts.get(nuc, 0) / max(length, 1))
            self.feature_names.append(f'{nuc}_freq')

        # 2. GC content
        gc = (counts.get('G', 0) + counts.get('C', 0)) / max(length, 1)
        features.append(gc)
        self.feature_names.append('gc_content')

        # 3. AU content
        au = (counts.get('A', 0) + counts.get('U', 0)) / max(length, 1)
        features.append(au)
        self.feature_names.append('au_content')

        # 4. Purine (AG) content
        purine = (counts.get('A', 0) + counts.get('G', 0)) / max(length, 1)
        features.append(purine)
        self.feature_names.append('purine_content')

        # 5. Entropy
        probs = [counts.get(n, 0) / max(length, 1) for n in self.NUCS]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        features.append(entropy)
        self.feature_names.append('entropy')

        # 6. Length normalized
        features.append(min(length / 1000.0, 2.0))
        self.feature_names.append('length_normalized')

        # 7. Log length
        features.append(np.log1p(length))
        self.feature_names.append('log_length')

        # 8. Di-nucleotide frequencies (16)
        dinucs = ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC',
                  'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']
        for dinuc in dinucs:
            count = sum(1 for k in range(len(seq)-1) if seq[k:k+2] == dinuc)
            features.append(count / max(length-1, 1))
            self.feature_names.append(f'dinuc_{dinuc}')

        # 9. Important tri-nucleotides (16)
        trinucs = ['AUU', 'AGU', 'ACU', 'UAU', 'UGU', 'UCU', 'GAU', 'GUU',
                   'UUU', 'AAA', 'GGG', 'CCC', 'AUG', 'UAG', 'GAC', 'CAG']
        for trinuc in trinucs:
            count = sum(1 for k in range(len(seq)-2) if seq[k:k+3] == trinuc)
            features.append(count / max(length-2, 1))
            self.feature_names.append(f'trinuc_{trinuc}')

        # 10. Repeat content
        max_repeat = 0
        for nuc in self.NUCS:
            count = 0
            max_c = 0
            for c in seq:
                if c == nuc:
                    count += 1
                    max_c = max(max_c, count)
                else:
                    count = 0
            max_repeat = max(max_repeat, max_c)
        features.append(max_repeat / max(length, 1))
        self.feature_names.append('max_repeat_ratio')

        # 11. Complexity (unique 4-mers)
        if length >= 4:
            unique_4mers = len(set(seq[i:i+4] for i in range(length-3)))
            features.append(unique_4mers / max(length-3, 1))
        else:
            features.append(0)
        self.feature_names.append('complexity')

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names

    def batch_extract(self, sequences: List[str]) -> np.ndarray:
        """Batch feature extraction."""
        features = []
        for seq in sequences:
            features.append(self.extract(seq))
        return np.array(features)


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