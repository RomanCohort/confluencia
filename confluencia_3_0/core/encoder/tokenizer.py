"""
tokenizer.py — RNA sequence tokenizer for circRNA encoder.

Handles RNA nucleotide sequences (A/U/C/G) using ESM-2's tokenizer
with RNA-specific preprocessing. Supports sliding window encoding
for sequences longer than max_seq_len.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


# RNA nucleotide alphabet (T → U conversion)
_RNA_MAP = {"T": "U", "t": "u"}
_VALID_RNA = set("AUGCaugc")


def sanitize_rna_sequence(seq: str) -> str:
    """Convert DNA bases to RNA and filter invalid characters.

    Parameters
    ----------
    seq : str
        Raw nucleotide sequence (may contain DNA or RNA bases).

    Returns
    -------
    str
        Sanitized RNA sequence (A/U/C/G only).
    """
    # Convert T to U (DNA → RNA)
    result = []
    for ch in seq.upper():
        mapped = _RNA_MAP.get(ch, ch)
        if mapped in _VALID_RNA:
            result.append(mapped)
        # Skip invalid characters silently
    return "".join(result)


def tokenize_rna_sequence(
    sequence: str,
    tokenizer,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize an RNA sequence using ESM-2 tokenizer.

    ESM-2 tokenizer uses 4 nucleotide tokens (A, U, C, G) plus
    special tokens <cls>, <eos>, <pad>, <mask>.

    Parameters
    ----------
    sequence : str
        RNA nucleotide sequence.
    tokenizer
        ESM-2 / RNA-FM tokenizer (from transformers).
    max_length : int
        Maximum tokenized length (including special tokens).

    Returns
    -------
    token_ids : np.ndarray
        Token ID array (1D).
    attention_mask : np.ndarray
        Binary mask array (1D), 1 for real tokens, 0 for padding.
    """
    sanitized = sanitize_rna_sequence(sequence)
    if not sanitized:
        # Empty sequence: return padding only
        ids = np.full(max_length, tokenizer.pad_token_id or 1, dtype=np.int64)
        mask = np.zeros(max_length, dtype=np.int64)
        return ids, mask

    encoded = tokenizer(
        sanitized,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


def tokenize_rna_batch(
    sequences: List[str],
    tokenizer,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize a batch of RNA sequences.

    Parameters
    ----------
    sequences : list[str]
        List of RNA nucleotide sequences.
    tokenizer
        ESM-2 / RNA-FM tokenizer.
    max_length : int
        Maximum tokenized length per sequence.

    Returns
    -------
    token_ids : np.ndarray
        Token ID array (batch_size, max_length).
    attention_mask : np.ndarray
        Binary mask array (batch_size, max_length).
    """
    sanitized = [sanitize_rna_sequence(s) for s in sequences]
    # Replace empty strings with a single A to avoid tokenizer errors
    sanitized = [s if s else "A" for s in sanitized]

    encoded = tokenizer(
        sanitized,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def sliding_window_encode(
    sequence: str,
    window_size: int = 512,
    stride: int = 256,
) -> List[str]:
    """Split a long RNA sequence into overlapping windows.

    For sequences longer than max_seq_len, split into windows and
    encode each separately, then mean-pool the embeddings.

    Parameters
    ----------
    sequence : str
        RNA nucleotide sequence.
    window_size : int
        Window length in nucleotides.
    stride : int
        Step size between windows.

    Returns
    -------
    list[str]
        List of windowed subsequences.
    """
    sanitized = sanitize_rna_sequence(sequence)
    if len(sanitized) <= window_size:
        return [sanitized]

    windows = []
    for start in range(0, len(sanitized), stride):
        end = min(start + window_size, len(sanitized))
        window = sanitized[start:end]
        if len(window) >= 50:  # Skip very short fragments
            windows.append(window)
    return windows if windows else [sanitized[:window_size]]


def encode_gene_expression(
    gene_expr: Dict[str, float],
    gene_cols: List[str],
    default_value: float = 0.5,
) -> np.ndarray:
    """Convert gene expression dict to ordered numpy array.

    Parameters
    ----------
    gene_expr : dict
        {gene_name: expression_value} mapping.
    gene_cols : list[str]
        Ordered list of gene column names.
    default_value : float
        Default value for missing genes.

    Returns
    -------
    np.ndarray
        (len(gene_cols),) float32 array.
    """
    arr = np.array(
        [float(gene_expr.get(g, default_value)) for g in gene_cols],
        dtype=np.float32,
    )
    return arr


def encode_gene_batch(
    gene_exprs: List[Dict[str, float]],
    gene_cols: List[str],
    default_value: float = 0.5,
) -> np.ndarray:
    """Convert batch of gene expression dicts to numpy array.

    Parameters
    ----------
    gene_exprs : list[dict]
        List of {gene_name: expression_value} mappings.
    gene_cols : list[str]
        Ordered gene column names.
    default_value : float
        Default for missing genes.

    Returns
    -------
    np.ndarray
        (batch_size, len(gene_cols)) float32 array.
    """
    return np.stack([
        encode_gene_expression(g, gene_cols, default_value)
        for g in gene_exprs
    ], axis=0)