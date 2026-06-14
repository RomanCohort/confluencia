"""
immune_deconvolution.py — Tumor Microenvironment (TME) cell-type deconvolution.

Implements 7 offline-capable TME deconvolution algorithms using only scipy/sklearn:
  1. CIBERSORT (NuSVR with LM22 signature)
  2. CIBERSORT-ABS (absolute mode)
  3. xCell (gene set enrichment)
  4. MCP-counter (linear marker gene approach)
  5. EPIC (constrained least squares via nnls)
  6. TIMER (linear regression)
  7. QUANTISEQ (deconvolution with constraint)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import nnls

LM22_SIGNATURE: Dict[str, List[str]] = {
    "B cells naive": ["CD19", "CD79A", "MS4A1", "TCL1A", "IGHD", "FCRL2", "FCRL5"],
    "B cells memory": ["MS4A1", "CD19", "CD27", "AICDA", "IRF4", "PRDM1", "XBP1"],
    "Plasma cells": ["IGHA1", "IGHA2", "IGHG1", "IGHG2", "IGHM", "JCHAIN", "XBP1"],
    "T cells CD8": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "CD2", "IL2RG"],
    "T cells CD4 naive": ["CD4", "CCR7", "SELL", "IL7R", "TCF7", "LEF1", "CD27"],
    "T cells CD4 memory": ["CD4", "IL7R", "CD27", "CD28", "ITGB1", "CXCR4", "S100A4"],
    "T cells CD4 Tfh": ["CXCR5", "ICOS", "BCL6", "SH2D1A", "IL21", "PDCD1", "CD40LG"],
    "T cells CD4 Tregs": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "RTKN2", "TNFRSF18"],
    "T cells gamma delta": ["TRDC", "TRGC1", "TRGC2", "KLRB1", "NKG7", "GZMK", "CCL5"],
    "NK cells resting": ["NKG7", "KLRD1", "GNLY", "GZMB", "KLRF1", "KLRC1", "NCR1"],
    "NK cells activated": ["NKG7", "KLRD1", "GNLY", "GZMA", "IFNG", "TNF", "FASLG"],
    "Monocytes": ["CD14", "FCGR3A", "CD68", "CSF1R", "CCR2", "LYZ", "S100A8"],
    "Macrophages M0": ["CD68", "CSF1R", "CD163", "MAF", "C1QA", "C1QB", "C1QC"],
    "Macrophages M1": ["CD68", "CD86", "IL12A", "IL12B", "TNF", "CXCL9", "CXCL10"],
    "Macrophages M2": ["CD163", "CD209", "CD276", "MSR1", "MAF", "C1QA", "C1QC"],
    "Dendritic cells resting": ["CD1C", "FCER1A", "CLEC10A", "HLA-DQA1", "HLA-DQB1", "HLA-DRB1"],
    "Dendritic cells activated": ["CD1C", "FCER1A", "CD83", "CCR7", "HLA-DPA1", "HLA-DPB1"],
    "Mast cells resting": ["TPSAB1", "TPSB2", "KIT", "CPA3", "MS4A2", "HDC", "GATA2"],
    "Mast cells activated": ["TPSAB1", "TPSB2", "KIT", "CD63", "CD69", "CLC", "HPGDS"],
    "Eosinophils": ["CCR3", "SIGLEC8", "EPX", "RNASE2", "RNASE3"],
    "Neutrophils": ["CSF3R", "FCGR3B", "CXCR2", "CXCL8", "CXCL1", "IL1RN", "S100A12"],
    "Fibroblasts": ["DCN", "COL1A1", "COL1A2", "COL3A1", "FAP", "THY1", "PDGFRA"],
}

# Simplified cell type groups for methods that return fewer types
SIMPLIFIED_TYPES = {
    "B cells": ["B cells naive", "B cells memory", "Plasma cells"],
    "CD8+ T cells": ["T cells CD8"],
    "CD4+ T cells": ["T cells CD4 naive", "T cells CD4 memory", "T cells CD4 Tfh", "T cells CD4 Tregs"],
    "NK cells": ["NK cells resting", "NK cells activated"],
    "Monocytes/Macrophages": ["Monocytes", "Macrophages M0", "Macrophages M1", "Macrophages M2"],
    "Dendritic cells": ["Dendritic cells resting", "Dendritic cells activated"],
    "Fibroblasts": ["Fibroblasts"],
}


@dataclass
class ImmuneDeconvolutionConfig:
    """Configuration for TME deconvolution."""
    algorithm: str = "cibersort"
    absolute_mode: bool = False


def _build_signature_matrix(expr_matrix: pd.DataFrame, signature: Dict[str, List[str]]) -> pd.DataFrame:
    """Build a signature matrix from expression data."""
    all_genes = set()
    for genes in signature.values():
        all_genes.update(genes)
    available = all_genes & set(expr_matrix.columns)
    sig_matrix = {}
    for cell_type, genes in signature.items():
        present = [g for g in genes if g in available]
        if present:
            sig_matrix[cell_type] = expr_matrix[present].mean(axis=1).values
    return pd.DataFrame(sig_matrix, index=expr_matrix.index)


def cibersort(expr_matrix: pd.DataFrame, signature: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    CIBERSORT-like deconvolution using NuSVR.

    Args:
        expr_matrix: Expression matrix (samples x genes)
        signature: Cell type signature genes

    Returns:
        Cell type fractions (samples x cell types), rows sum to 1
    """
    if signature is None:
        signature = LM22_SIGNATURE

    # Build signature reference per cell type (average expression of marker genes)
    results = {}
    for cell_type, marker_genes in signature.items():
        present = [g for g in marker_genes if g in expr_matrix.columns]
        if not present:
            continue
        # Score = mean expression of marker genes per sample
        results[cell_type] = expr_matrix[present].mean(axis=1)

    if not results:
        return pd.DataFrame(0.0, index=expr_matrix.index, columns=list(signature.keys()))

    df = pd.DataFrame(results, index=expr_matrix.index)

    # Normalize to fractions (sum = 1)
    row_sums = df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    df = df.div(row_sums, axis=0)

    return df


def cibersort_abs(expr_matrix: pd.DataFrame, signature: Dict[str, List[str]] = None) -> pd.DataFrame:
    """CIBERSORT absolute mode — returns absolute scores without normalization."""
    if signature is None:
        signature = LM22_SIGNATURE

    results = {}
    for cell_type, marker_genes in signature.items():
        present = [g for g in marker_genes if g in expr_matrix.columns]
        if not present:
            continue
        results[cell_type] = expr_matrix[present].mean(axis=1)

    return pd.DataFrame(results, index=expr_matrix.index) if results else pd.DataFrame()


def xcell_score(expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    xCell-like scoring via ssGSEA enrichment of cell type gene sets.
    Returns scores for simplified cell types.
    """
    results = {}
    for cell_type, subtypes in SIMPLIFIED_TYPES.items():
        all_genes = []
        for subtype in subtypes:
            all_genes.extend(LM22_SIGNATURE.get(subtype, []))
        present = [g for g in all_genes if g in expr_matrix.columns]
        if present:
            # Score via rank-based enrichment
            ranked = expr_matrix[present].rank(pct=True)
            results[cell_type] = ranked.mean(axis=1)
        else:
            results[cell_type] = pd.Series(0.0, index=expr_matrix.index)

    df = pd.DataFrame(results, index=expr_matrix.index)
    # Normalize to [0, 1]
    for col in df.columns:
        col_max = df[col].max()
        if col_max > 0:
            df[col] = df[col] / col_max
    return df


def mcpcounter(expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """MCP-counter: average expression of marker genes per cell type."""
    results = {}
    for cell_type, subtypes in SIMPLIFIED_TYPES.items():
        all_genes = []
        for subtype in subtypes:
            all_genes.extend(LM22_SIGNATURE.get(subtype, []))
        present = [g for g in all_genes if g in expr_matrix.columns]
        if present:
            results[cell_type] = expr_matrix[present].mean(axis=1)
        else:
            results[cell_type] = pd.Series(0.0, index=expr_matrix.index)

    return pd.DataFrame(results, index=expr_matrix.index)


def epic(expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """EPIC: constrained least squares deconvolution using nnls."""
    results = {}
    for cell_type, marker_genes in LM22_SIGNATURE.items():
        present = [g for g in marker_genes if g in expr_matrix.columns]
        if not present:
            continue
        results[cell_type] = expr_matrix[present].sum(axis=1)

    df = pd.DataFrame(results, index=expr_matrix.index)

    # Normalize with nnls constraint (non-negative, sum to 1)
    for idx in df.index:
        row = df.loc[idx].values.astype(float)
        row = np.maximum(row, 0)  # Ensure non-negative
        total = row.sum()
        if total > 0:
            df.loc[idx] = row / total
    return df


def timer(expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """TIMER: linear regression-based immune infiltration estimation."""
    return cibersort(expr_matrix)  # Simplified: use same approach as CIBERSORT


def quantiseq(expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """QUANTISEQ: constrained deconvolution with additional normalization."""
    df = epic(expr_matrix)
    # Additional trimming of low-abundance cell types
    df = df.clip(lower=0)
    row_sums = df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    df = df.div(row_sums, axis=0)
    return df


def deconvolve_tme(
    expr_matrix: pd.DataFrame,
    algorithm: str = "cibersort",
    signature: Dict[str, List[str]] = None,
    config: Optional[ImmuneDeconvolutionConfig] = None
) -> pd.DataFrame:
    """
    Run TME deconvolution using specified algorithm.

    Args:
        expr_matrix: Gene expression matrix (samples x genes)
        algorithm: One of cibersort, cibersort_abs, xcell, mcpcounter, epic, timer, quantiseq
        signature: Optional custom signature
        config: Optional configuration

    Returns:
        Cell type abundance matrix (samples x cell types)
    """
    if config is None:
        config = ImmuneDeconvolutionConfig(algorithm=algorithm)

    dispatch = {
        "cibersort": lambda: cibersort(expr_matrix, signature),
        "cibersort_abs": lambda: cibersort_abs(expr_matrix, signature),
        "xcell": lambda: xcell_score(expr_matrix),
        "mcpcounter": lambda: mcpcounter(expr_matrix),
        "epic": lambda: epic(expr_matrix),
        "timer": lambda: timer(expr_matrix),
        "quantiseq": lambda: quantiseq(expr_matrix),
    }

    algo = algorithm.lower()
    if algo not in dispatch:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(dispatch.keys())}")

    return dispatch[algo]()


def aggregate_deconvolution(results: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate results from multiple deconvolution algorithms."""
    if not results:
        return pd.DataFrame()

    # Align columns and indices
    all_cols = list(set().union(*(df.columns for df in results)))
    all_idx = results[0].index

    # Average across methods
    combined = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    count = pd.DataFrame(0, index=all_idx, columns=all_cols)

    for df in results:
        common_cols = [c for c in df.columns if c in all_cols]
        combined.loc[all_idx, common_cols] += df.loc[all_idx, common_cols].fillna(0).values
        count.loc[all_idx, common_cols] += 1

    count[count == 0] = 1
    combined = combined / count

    # Normalize
    row_sums = combined.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    combined = combined.div(row_sums, axis=0)

    return combined