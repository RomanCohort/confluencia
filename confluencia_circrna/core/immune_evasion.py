"""
immune_evasion.py — Immunotherapy response prediction via TIDE and IPS scoring.

Implements:
  - TIDE (Tumor Immune Dysfunction and Exclusion) score (Jiang et al., Nat Cancer 2018)
  - IPS (Immunophenoscore) for ICI response prediction

TIDE evaluates ICB efficacy by simulating T cell dysfunction and exclusion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Gene sets
DEFAULT_DYSFUNCTION_GENES = ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD96", "ENTPD1"]
DEFAULT_EXCLUSION_GENES = ["VEGFA", "CXCL12", "TGFB1", "CXCL8", "IL10", "ARG1", "ADORA2A", "CD274"]

IPS_EFFECTOR = ["CD8A", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10", "TBX21", "EOMES", "GNLY"]
IPS_SUPPRESSOR = ["FOXP3", "TGFB1", "IL10", "ARG1", "VEGFA", "CD274", "PDCD1", "CTLA4"]
IPS_MHC = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "HLA-DMA", "HLA-DMB", "HLA-DPA1", "HLA-DPB1"]
IPS_CHECKPOINT = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD160"]

IPS_DEFAULT_WEIGHTS = {
    "effector": 1.0,
    "suppressor": -1.0,
    "mhc": 0.5,
    "checkpoint": -0.5,
}


@dataclass
class ImmuneEvasionConfig:
    """Configuration for immune evasion analysis."""
    dysfunction_genes: List[str] = field(default_factory=lambda: DEFAULT_DYSFUNCTION_GENES.copy())
    exclusion_genes: List[str] = field(default_factory=lambda: DEFAULT_EXCLUSION_GENES.copy())
    ips_weights: Dict[str, float] = field(default_factory=lambda: IPS_DEFAULT_WEIGHTS.copy())


def _normalize_expr(expr_values: np.ndarray) -> np.ndarray:
    """Min-max normalize expression values to [0, 1]."""
    vmin, vmax = np.nanmin(expr_values), np.nanmax(expr_values)
    if vmax == vmin:
        return np.zeros_like(expr_values, dtype=float)
    return (expr_values - vmin) / (vmax - vmin)


def _mean_expression(expr_matrix: pd.DataFrame, genes: List[str]) -> pd.Series:
    """Compute mean normalized expression of a gene set across samples."""
    present = [g for g in genes if g in expr_matrix.columns]
    if not present:
        return pd.Series(0.0, index=expr_matrix.index)
    return expr_matrix[present].mean(axis=1)


def compute_tide_score(
    expr_matrix: pd.DataFrame,
    config: Optional[ImmuneEvasionConfig] = None
) -> pd.DataFrame:
    """
    Compute TIDE scores for each sample.

    Args:
        expr_matrix: Gene expression matrix (samples x genes)
        config: Optional configuration

    Returns:
        DataFrame with: tide_score, dysfunction_score, exclusion_score per sample
    """
    if config is None:
        config = ImmuneEvasionConfig()

    # Dysfunction score: mean expression of dysfunction genes (normalized)
    dysf = _mean_expression(expr_matrix, config.dysfunction_genes)

    # Exclusion score: mean expression of exclusion genes (normalized)
    excl = _mean_expression(expr_matrix, config.exclusion_genes)

    # Normalize both to [0, 1]
    dysf_norm = _normalize_expr(dysf.values)
    excl_norm = _normalize_expr(excl.values)

    # TIDE = dysfunction + exclusion (higher = worse ICI response)
    tide = dysf_norm + excl_norm

    return pd.DataFrame({
        "tide_score": tide,
        "dysfunction_score": dysf_norm,
        "exclusion_score": excl_norm,
    }, index=expr_matrix.index)


def compute_ips(
    expr_matrix: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Compute Immunophenoscore (IPS) for each sample.

    IPS = w1 * effector - w2 * suppressor + w3 * MHC - w4 * checkpoint
    Normalized to 0-10 scale.

    Args:
        expr_matrix: Gene expression matrix (samples x genes)
        weights: Optional weights for each component

    Returns:
        Series with IPS scores (0-10 range) per sample
    """
    if weights is None:
        weights = IPS_DEFAULT_WEIGHTS

    effector = _mean_expression(expr_matrix, IPS_EFFECTOR)
    suppressor = _mean_expression(expr_matrix, IPS_SUPPRESSOR)
    mhc = _mean_expression(expr_matrix, IPS_MHC)
    checkpoint = _mean_expression(expr_matrix, IPS_CHECKPOINT)

    # Weighted combination
    ips = (
        weights["effector"] * effector +
        weights["suppressor"] * suppressor +
        weights["mhc"] * mhc +
        weights["checkpoint"] * checkpoint
    )

    # Normalize to 0-10
    ips_min = ips.min()
    ips_max = ips.max()
    if ips_max > ips_min:
        ips = (ips - ips_min) / (ips_max - ips_min) * 10.0
    else:
        ips = pd.Series(5.0, index=expr_matrix.index)

    return ips


def immune_evasion_report(
    expr_matrix: pd.DataFrame,
    config: Optional[ImmuneEvasionConfig] = None
) -> Dict:
    """
    Generate full immune evasion report.

    Returns:
        Dict with TIDE, IPS, dysfunction, exclusion, predicted_response
    """
    tide_df = compute_tide_score(expr_matrix, config)
    ips_series = compute_ips(expr_matrix)

    # Predicted response based on IPS
    def categorize(ips_val):
        if ips_val > 6:
            return "likely_responder"
        elif ips_val < 4:
            return "likely_non_responder"
        return "intermediate"

    predicted_response = ips_series.apply(categorize)

    return {
        "tide": tide_df,
        "ips": ips_series,
        "predicted_response": predicted_response,
        "mean_tide": float(tide_df["tide_score"].mean()),
        "mean_ips": float(ips_series.mean()),
        "n_responders": int((predicted_response == "likely_responder").sum()),
        "n_non_responders": int((predicted_response == "likely_non_responder").sum()),
    }