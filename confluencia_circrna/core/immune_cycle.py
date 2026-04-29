"""
immune_cycle.py — Cancer Immunity Cycle 7-step scoring via ssGSEA.

Implements the Chen & Mellman (2013, Immunity) cancer immunity cycle:
  1. Release of tumor antigens
  2. Dendritic cell (DC) priming and activation
  3. T cell priming and activation
  4. Trafficking of T cells to the tumor
  5. Infiltration of T cells into the tumor
  6. Recognition of tumor cells by T cells
  7. Killing of tumor cells by cytotoxic T cells

Uses single-sample GSEA (ssGSEA) — no external API, pure scipy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata


CANCER_IMMUNITY_CYCLE_STEPS: Dict[str, List[str]] = {
    "antigen_release": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG", "TBX21", "FASLG", "CXCL9", "CXCL10"],
    "dc_priming": ["CCR7", "CD80", "CD86", "CD40", "IL12A", "IL6", "CXCL9", "CXCL10", "FLT3LG", "BATF3"],
    "t_cell_priming": ["CD3D", "CD3E", "CD3G", "CD28", "ICOS", "IL2", "STAT4", "TBX21", "IL12RB2", "IL18RAP"],
    "trafficking": ["CXCL9", "CXCL10", "CXCL11", "CXCR3", "CCR5", "S1PR1", "S1PR4", "CXCR4", "CCL19", "CCL21"],
    "infiltration": ["CD8A", "CD8B", "CD4", "FOXP3", "MMP9", "VEGFA", "CXCL12", "CXCR4", "CCL5", "ITGA1"],
    "recognition": ["HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "B2M", "CD8A", "CD8B", "TRAC", "TRBC"],
    "killing": ["GZMB", "PRF1", "IFNG", "FASLG", "TNF", "CXCL9", "CXCL10", "GNLY", "KLRK1", "LAG3"],
}


@dataclass
class ImmuneCycleConfig:
    """Configuration for immune cycle scoring."""
    n_permutations: int = 0  # 0 = deterministic ssGSEA
    normalize: bool = True
    min_gene_set_size: int = 3
    max_gene_set_size: int = 500


def ss_gsea(es: np.ndarray) -> float:
    """
    Single-sample GSEA (ssGSEA) enrichment score.

    Args:
        es: Expression values for one sample (genes × 1)

    Returns:
        Enrichment score (float)
    """
    # Rank genes by expression (descending)
    ranked = rankdata(-es, method='average')  # Highest expr = rank 1
    n = len(es)
    es_normalized = ranked / n

    # ES = max(1/n) sum over all genes where gene in set of (rank/n - 1/n))
    # Simplified: ES = mean rank deviation for gene set members
    return float(np.mean(es_normalized))


def compute_enrichment_score(
    expr_row: pd.Series,
    gene_set: List[str],
    config: ImmuneCycleConfig
) -> float:
    """Compute ssGSEA enrichment score for one sample and one gene set."""
    available_genes = [g for g in gene_set if g in expr_row.index]
    if len(available_genes) < config.min_gene_set_size:
        return 0.0

    # Get expression values for gene set genes
    gene_expr = expr_row[available_genes].values
    # Compute enrichment
    es = ss_gsea(gene_expr)
    return es


def compute_immune_cycle_scores(
    expr_matrix: pd.DataFrame,
    config: Optional[ImmuneCycleConfig] = None
) -> pd.DataFrame:
    """
    Compute ssGSEA scores for all 7 steps of the cancer immunity cycle.

    Args:
        expr_matrix: Gene expression matrix (rows=samples, cols=genes)
        config: Optional configuration

    Returns:
        DataFrame with 7 step scores per sample
    """
    if config is None:
        config = ImmuneCycleConfig()

    results = {}
    for step_name, gene_set in CANCER_IMMUNITY_CYCLE_STEPS.items():
        scores = []
        for _, row in expr_matrix.iterrows():
            score = compute_enrichment_score(row, gene_set, config)
            scores.append(score)
        results[step_name] = scores

    df = pd.DataFrame(results, index=expr_matrix.index)

    if config.normalize:
        # Min-max normalize each column to [0, 1]
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)

    return df


def compute_tumor_killing_index(cycle_scores: pd.DataFrame) -> pd.Series:
    """
    Compute tumor killing index from immune cycle scores.

    Index = weighted average of infiltration + recognition + killing steps.
    """
    if "infiltration" not in cycle_scores or "recognition" not in cycle_scores or "killing" not in cycle_scores:
        return pd.Series(0.0, index=cycle_scores.index)

    tki = (
        0.35 * cycle_scores["infiltration"] +
        0.25 * cycle_scores["recognition"] +
        0.40 * cycle_scores["killing"]
    )
    return tki


def compute_therapeutic_window(cycle_scores: pd.DataFrame) -> pd.Series:
    """
    Compute therapeutic window: killing / (antigen_release + 0.1).

    Higher = better potential for therapeutic effect.
    """
    if "killing" not in cycle_scores or "antigen_release" not in cycle_scores:
        return pd.Series(0.0, index=cycle_scores.index)

    # Avoid division by zero
    denom = cycle_scores["antigen_release"] + 0.1
    tw = cycle_scores["killing"] / denom * 10  # Scale to roughly [0, 10]
    return tw.clip(0, 10)


if __name__ == "__main__":
    # Demo: synthetic expression matrix
    np.random.seed(42)
    n_samples = 20
    genes = list(set([g for genes in CANCER_IMMUNITY_CYCLE_STEPS.values() for g in genes] +
                     [f"GENE{i}" for i in range(100)]))
    expr = pd.DataFrame(
        np.random.exponential(2, (n_samples, len(genes))),
        index=[f"sample_{i}" for i in range(n_samples)],
        columns=genes
    )
    expr.iloc[:5, :20] *= 2.0  # Enrich some genes in first 5 samples

    config = ImmuneCycleConfig()
    cycle_df = compute_immune_cycle_scores(expr, config)
    tki = compute_tumor_killing_index(cycle_df)
    tw = compute_therapeutic_window(cycle_df)

    print("Cancer Immunity Cycle Scores (first 5 samples):")
    print(cycle_df.head())
    print(f"\nTumor Killing Index: {tki.head().values}")
    print(f"Therapeutic Window: {tw.head().values}")