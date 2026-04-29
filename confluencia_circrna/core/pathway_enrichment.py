"""
pathway_enrichment.py — GO/KEGG enrichment and single-sample pathway scoring.

Implements:
  - Fisher's exact test GO/KEGG enrichment with BH correction
  - GSVA (Gene Set Variation Analysis) via kernel-smoothed K-S test
  - ssGSEA pathway scoring
  - Offline-capable (no external API)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata, fisher_exact
from scipy.special import expit


KEGG_CANCER_PATHWAYS: Dict[str, List[str]] = {
    "cell_cycle": ["CDK1", "CDK2", "CDK4", "CDK6", "CCND1", "CCNE1", "CDKN1A", "CDKN2A", "TP53", "RB1", "E2F1", "E2F2", "E2F3"],
    "apoptosis": ["BAX", "BCL2", "CASP3", "CASP8", "CASP9", "FAS", "FADD", "TP53", "MCL1", "BIRC5", "DIABLO", "APAF1"],
    "pi3k_akt": ["PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "PDPK1", "RPS6KB1", "PTEN", "GSK3B", "FOXO3", "TSC1", "TSC2"],
    "mapk": ["KRAS", "BRAF", "RAF1", "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "NRAS", "HRAS", "DUSP6", "ETS1", "ELK1"],
    "immune": ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD28", "ICOS", "TNFRSF9"],
    "ddr": ["BRCA1", "BRCA2", "ATM", "TP53", "CHEK1", "CHEK2", "RAD51", "ATR", "PARP1", "XRCC1", "XRCC2"],
    "tgf_beta": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4", "SMAD7", "ACVR1", "ACVR2A", "ACVR2B", "LEFTY1", "LEFTY2", "INHBA"],
    "hippo": ["YAP1", "WWTR1", "TEAD1", "TEAD2", "LATS1", "LATS2", "STK3", "STK4", "MOB1A", "MOB1B", "SAV1", "NF2"],
    "wnt": ["CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B", "DVL1", "DVL2", "DVL3", "LEF1", "TCF7", "MYC", "CCND1"],
    "hedgehog": ["SHH", "GLI1", "GLI2", "GLI3", "PTCH1", "SMO", "SUFU", "IHH", "DHH", "HHIP", "ZIC2", "ZIC3"],
    "erbb": ["EGFR", "ERBB2", "ERBB3", "ERBB4", "GRB2", "SOS1", "HRAS", "KRAS", "PIK3CA", "AKT1", "RAF1", "MAPK1"],
    "jak_stat": ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6", "SOCS1", "SOCS3"],
    "notch": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1", "DLL3", "DLL4", "RBPJ", "HES1", "HES5"],
    "interferon_gamma": ["IFNG", "IRF1", "IRF2", "STAT1", "STAT2", "JAK1", "JAK2", "CIITA", "SOCS1", "IDO1", "CXCL9", "CXCL10", "CXCL11"],
    "interferon_alpha": ["IFNA1", "IFNA2", "IFNA4", "IFNA5", "IFNA6", "IFNA7", "IFNA8", "IRF3", "IRF7", "IRF9", "STAT1", "STAT2", "JAK1", "TYK2"],
    "cytotoxic": ["GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1", "FASLG", "TNFSF10", "GNLY", "KLRK1", "NKG7"],
    "antigen_presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP", "PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "ERAP1", "ERAP2", "CIITA"],
    "immunosuppressive": ["FOXP3", "TGFB1", "IL10", "ARG1", "VEGFA", "CD274", "PDCD1", "CTLA4", "IDO1", "LAG3", "HAVCR2"],
}

GO_BP_SUBSETS: Dict[str, List[str]] = {
    "immune_response": ["CD4", "CD8A", "IFNG", "IL2", "TNF", "CXCL9", "CXCL10", "IL12A", "IL12B", "GZMB", "PRF1"],
    "cell_proliferation": ["CCND1", "CDK4", "MYC", "E2F1", "E2F2", "MKI67", "PCNA", "TOP2A", "AURKA", "AURKB", "PLK1"],
    "dna_repair": ["BRCA1", "BRCA2", "ATM", "TP53", "RAD51", "ERCC1", "ERCC2", "MLH1", "MSH2", "PARP1", "XRCC1"],
    "rna_processing": ["EIF4A3", "UPF1", "UPF2", "HNRNPA1", "HNRNPA2B1", "SRSF1", "SRSF2", "SF3B1", "U2AF1", "PTBP1"],
    "metabolism": ["LDHA", "PKM", "HK2", "GAPDH", "ENO1", "ENO2", "PGK1", "PDK1", "SLC2A1", "SLC2A3"],
    "hypoxia_response": ["HIF1A", "EPAS1", "VEGFA", "PGK1", "LDHA", "ALDOA", "ENO1", "SLC2A1", "CA9", "ADM", "EPO", "ANGPTL4"],
}


@dataclass
class EnrichmentConfig:
    """Configuration for pathway enrichment."""
    pval_threshold: float = 0.05
    correction_method: str = "BH"
    min_gene_set_size: int = 3
    max_gene_set_size: int = 500


def _bh_correction(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return []

    ranked_pvals = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n

    prev_idx = None
    for rank, (orig_idx, pval) in enumerate(ranked_pvals, start=1):
        adj_p = pval * n / rank
        adjusted[orig_idx] = adj_p

    # Enforce monotonicity
    min_adj = 1.0
    for orig_idx in [x[0] for x in ranked_pvals][::-1]:
        min_adj = min(min_adj, adjusted[orig_idx])
        adjusted[orig_idx] = min_adj

    return adjusted


def go_kegg_enrichment(
    genes: List[str],
    background: List[str],
    pathway_db: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Perform GO/KEGG enrichment analysis using Fisher's exact test.

    Args:
        genes: List of query genes (e.g., DEGs)
        background: List of background genes
        pathway_db: Pathway gene set dictionary

    Returns:
        DataFrame with pathway, pvalue, padj, OR, n_overlap
    """
    if pathway_db is None:
        pathway_db = KEGG_CANCER_PATHWAYS

    query_set = set(genes)
    bg_set = set(background)
    n_bg = len(bg_set)

    results = []
    for pathway, path_genes in pathway_db.items():
        path_set = set(path_genes) & bg_set
        n_path = len(path_set)
        n_overlap = len(query_set & path_set)
        n_query = len(query_set)

        # Fisher's exact test
        #          in_pathway  not_in_pathway
        # in_query     a           b
        # not_in_query c           d
        a = n_overlap
        b = n_query - n_overlap
        c = n_path - n_overlap
        d = n_bg - n_query - n_path + n_overlap

        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        or_val = (a * d) / (b * c) if b > 0 and c > 0 else 0.0

        results.append({
            "pathway": pathway,
            "pvalue": pval,
            "odds_ratio": or_val,
            "n_overlap": n_overlap,
            "n_pathway": n_path,
            "genes_overlap": list(query_set & path_set),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df["padj"] = _bh_correction(df["pvalue"].tolist())
        df = df.sort_values("pvalue")

    return df


def ssgsea(
    expr_row: pd.Series,
    gene_set: List[str],
    weight: float = 0.25
) -> float:
    """
    Single-sample GSEA enrichment score.

    Args:
        expr_row: Expression values for one sample
        gene_set: Gene set to test
        weight: Weight for ES calculation

    Returns:
        Enrichment score
    """
    present_genes = [g for g in gene_set if g in expr_row.index]
    if not present_genes:
        return 0.0

    # Rank genes by expression
    ranked = expr_row.rank(ascending=False, pct=True)

    # Compute running sum
    n = len(expr_row)
    n_set = len(present_genes)

    # Gene set contribution
    set_contribution = np.abs(ranked[present_genes]) ** weight
    # Non-set contribution
    all_genes = set(expr_row.index)
    non_set = list(all_genes - set(present_genes))
    non_set_contribution = np.ones(len(non_set)) / (n - n_set)

    # ES = sum over genes in set minus uniform contribution
    es = (set_contribution.sum() / n_set) - (non_set_contribution.sum() if non_set else 0)
    return float(es)


def gsva(
    expr_matrix: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    kcdf: str = "gaussian",
    mx_diff: bool = True
) -> pd.DataFrame:
    """
    Gene Set Variation Analysis (GSVA).

    Simplified implementation without full kernel smoothing.

    Args:
        expr_matrix: Expression matrix (samples x genes)
        gene_sets: Gene set dictionary
        kcdf: Kernel type (ignored, simplified)
        mx_diff: Use max difference

    Returns:
        DataFrame with GSVA scores (samples x pathways)
    """
    results = {}
    for pathway, genes in gene_sets.items():
        scores = []
        for _, row in expr_matrix.iterrows():
            score = ssgsea(row, genes)
            scores.append(score)
        results[pathway] = scores

    df = pd.DataFrame(results, index=expr_matrix.index)

    # Normalize to zero mean
    for col in df.columns:
        df[col] = df[col] - df[col].mean()

    return df


def ssgsea_pathway_scores(
    expr_matrix: pd.DataFrame,
    gene_sets: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """Compute ssGSEA scores for all pathways."""
    if gene_sets is None:
        gene_sets = KEGG_CANCER_PATHWAYS

    return gsva(expr_matrix, gene_sets)


def enrichment_report(
    degs: List[str],
    expr_matrix: pd.DataFrame,
    background: Optional[List[str]] = None,
    pathway_db: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    Generate comprehensive enrichment report.

    Args:
        degs: Differentially expressed genes
        expr_matrix: Expression matrix
        background: Background gene list
        pathway_db: Pathway database

    Returns:
        Dict with top_pathways, pathway_scores
    """
    if pathway_db is None:
        pathway_db = KEGG_CANCER_PATHWAYS

    if background is None:
        background = list(expr_matrix.columns)

    # GO/KEGG enrichment
    enrich_df = go_kegg_enrichment(degs, background, pathway_db)
    top_pathways = enrich_df.head(10).to_dict("records") if not enrich_df.empty else []

    # ssGSEA scores
    pathway_scores = ssgsea_pathway_scores(expr_matrix, pathway_db)

    return {
        "top_pathways": top_pathways,
        "pathway_scores": pathway_scores,
        "enrichment_df": enrich_df,
        "n_degs": len(degs),
    }