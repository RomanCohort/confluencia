"""
Sequence-Aware Data Splitting Patch
=====================================
Fixes data leakage by ensuring no sequence/SMILES appears in both train and test.

Drop-in replacement for random train_test_split when working with sequence data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def sequence_split(
    df: pd.DataFrame,
    seq_col: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by unique sequences.

    Guarantees that the same sequence (e.g., same epitope_seq or SMILES)
    does NOT appear in both train and test sets.

    Args:
        df: Input DataFrame
        seq_col: Column name containing sequence/SMILES identifiers
        test_ratio: Fraction of unique sequences to reserve for testing
        seed: Random seed for reproducibility

    Returns:
        (train_df, test_df) pair
    """
    unique_seqs = df[seq_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)

    n_test_seqs = max(1, int(len(unique_seqs) * test_ratio))
    test_seqs = set(unique_seqs[:n_test_seqs])

    is_test = df[seq_col].isin(test_seqs)
    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)

    return train_df, test_df


def stratified_sequence_split(
    df: pd.DataFrame,
    seq_col: str,
    label_col: str,
    n_bins: int = 5,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified sequence split: ensures label distribution is preserved.

    Sequences are grouped by their mean label value (binned), then
    a proportional number of sequences from each bin are assigned to test.
    """
    # Compute per-sequence mean label
    seq_stats = df.groupby(seq_col)[label_col].mean().reset_index()
    seq_stats.columns = [seq_col, "_mean_label"]

    # Bin labels
    try:
        seq_stats["_bin"] = pd.qcut(seq_stats["_mean_label"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        seq_stats["_bin"] = 0

    rng = np.random.default_rng(seed)
    test_seqs = set()

    for bin_id, group in seq_stats.groupby("_bin"):
        seqs = group[seq_col].values
        rng.shuffle(seqs)
        n_test = max(1, int(len(seqs) * test_ratio))
        test_seqs.update(seqs[:n_test])

    is_test = df[seq_col].isin(test_seqs)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def leave_one_sequence_out_cv(
    df: pd.DataFrame,
    seq_col: str,
    max_folds: int = 20,
    seed: int = 42,
):
    """Generate LOCO (Leave-One-Cluster-Out) CV splits by sequence.

    Yields (train_indices, test_indices) tuples.
    Each fold holds out one unique sequence as the test set.
    Limits to max_folds to avoid excessive computation.
    """
    unique_seqs = df[seq_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)

    for seq in unique_seqs[:max_folds]:
        test_idx = df.index[df[seq_col] == seq].values
        train_idx = df.index[df[seq_col] != seq].values
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_col: str,
) -> bool:
    """Verify that no sequence appears in both train and test."""
    train_seqs = set(train_df[seq_col].dropna().unique())
    test_seqs = set(test_df[seq_col].dropna().unique())
    overlap = train_seqs & test_seqs
    if overlap:
        print(f"WARNING: Data leakage detected! {len(overlap)} sequences appear in both train and test.")
        print(f"Leaking sequences: {list(overlap)[:10]}...")
        return False
    print(f"No leakage detected. Train: {len(train_seqs)} unique sequences, Test: {len(test_seqs)} unique sequences.")
    return True
