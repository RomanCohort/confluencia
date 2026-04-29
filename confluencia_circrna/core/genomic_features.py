"""
genomic_features.py — Genomic feature quantification for immunotherapy prediction.

Computes:
  - TMB (Tumor Mutational Burden) — non-synonymous mutations / genome size
  - CNV (Copy Number Variation) burden and arm-level scores
  - DDR (DNA Damage Repair) pathway mutation burden

References:
  - Lawrence et al., Nature 2013 — TMB and neoantigen load
  - Beroukhim et al., Nature 2010 — CNV landscape across cancers
  - Knijnenburg et al., Cell Reports 2018 — DDR pathway alterations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd


DDR_GENES: Dict[str, Set[str]] = {
    "NER": {"ERCC1", "ERCC2", "XPA", "XPC", "DDB1", "DDB2", "ERCC3", "ERCC4", "ERCC5", "ERCC6", "ERCC8"},
    "HRR": {"BRCA1", "BRCA2", "RAD51", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D", "NBN", "MRE11", "RAD50"},
    "BER": {"MPG", "UNG", "OGG1", "MUTYH", "APEX1", "LIG3", "POLB", "PCNA", "RFC1", "RFC2"},
    "MMR": {"MLH1", "MSH2", "MSH6", "PMS2", "APC", "MSH3", "PMS1", "MLH3"},
    "FA": {"FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FANCB", "FANCL", "FANCM", "PALB2"},
    "NHEJ": {"XRCC4", "XRCC5", "LIG4", "PRKDC", "DCLRE1C", "XRCC6", "XRCC7"},
    "TLS": {"POLH", "POLI", "REV1", "REV3L", "POLQ", "POLK"},
}

NONSYNONYMOUS_TYPES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
    "Nonstop_Mutation", "Silent_Mutation"  # some include silent
}

SYNONYMOUS_TYPES = {
    "Silent", "Splice_Region", "Intron", "3'UTR", "5'UTR",
    "3'Flank", "5'Flank", "IGR", "RNA"
}


@dataclass
class GenomicConfig:
    """Configuration for genomic feature computation."""
    genome_size_mb: float = 30.0  # ~30 Mb exome
    cnv_threshold_amplification: float = 0.5  # log2 ratio > 0.5
    cnv_threshold_deletion: float = -0.5   # log2 ratio < -0.5


def compute_tmb(
    mutations: pd.DataFrame,
    genome_size: float = 30.0
) -> Optional[float]:
    """
    Compute Tumor Mutational Burden (mutations/Mb).

    Args:
        mutations: DataFrame with columns [sample_id, gene, variant_type]
        genome_size: Exome size in Mb (default 30 Mb)

    Returns:
        TMB value (mutations per Mb)
    """
    if mutations.empty:
        return None

    # Count non-synonymous mutations
    nonsyn = 0
    if "variant_type" in mutations.columns:
        nonsyn = mutations[mutations["variant_type"].isin(NONSYNONYMOUS_TYPES)].shape[0]
    elif "Variant_Classification" in mutations.columns:
        nonsyn = mutations[mutations["Variant_Classification"].isin(NONSYNONYMOUS_TYPES)].shape[0]
    else:
        nonsyn = mutations.shape[0]

    tmb = nonsyn / genome_size
    return round(tmb, 4)


def compute_cnv_scores(
    cnv_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute CNV burden from CNV data.

    Args:
        cnv_data: DataFrame with [sample_id, gene, chromosome, start, end, copy_number, segment_mean]

    Returns:
        Dict with: cnv_burden, amplification_score, deletion_score, focal_events
    """
    if cnv_data.empty:
        return {
            "cnv_burden": 0.0,
            "amplification_score": 0.0,
            "deletion_score": 0.0,
            "focal_events": 0,
        }

    # CNV burden: fraction of genome covered by CNV segments
    if "start" in cnv_data.columns and "end" in cnv_data.columns:
        cnv_data = cnv_data.copy()
        cnv_data["length"] = cnv_data["end"] - cnv_data["start"]
        total_cnv_length = cnv_data["length"].sum()
        # Approximate genome size 3 Gb
        cnv_burden = total_cnv_length / 3e9
    else:
        cnv_burden = cnv_data.shape[0] / 1000  # fallback: per-gene count

    # Amplification score (segments with log2 > threshold)
    if "segment_mean" in cnv_data.columns:
        amplifications = (cnv_data["segment_mean"] > 0.5).sum()
        deletions = (cnv_data["segment_mean"] < -0.5).sum()
        focal_amplifications = (cnv_data["segment_mean"] > 1.0).sum()
        focal_deletions = (cnv_data["segment_mean"] < -1.0).sum()
        focal_events = focal_amplifications + focal_deletions
        amplification_score = amplifications / max(cnv_data.shape[0], 1)
        deletion_score = deletions / max(cnv_data.shape[0], 1)
    else:
        amplification_score = deletion_score = focal_events = 0.0

    return {
        "cnv_burden": round(cnv_burden, 6),
        "amplification_score": round(amplification_score, 4),
        "deletion_score": round(deletion_score, 4),
        "focal_events": int(focal_events),
    }


def compute_cnv_arm_level(cnv_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-chromosome-arm CNV scores."""
    if cnv_data.empty or "chromosome" not in cnv_data.columns:
        return pd.DataFrame()

    arms = ["p", "q"]
    results = {}
    for chrom in cnv_data["chromosome"].unique():
        chrom_data = cnv_data[cnv_data["chromosome"] == chrom]
        for arm in arms:
            key = f"{chrom}{arm}"
            if "segment_mean" in chrom_data.columns:
                results[key] = {
                    "mean_segment_mean": chrom_data["segment_mean"].mean(),
                    "amplification_fraction": (chrom_data["segment_mean"] > 0.5).mean(),
                    "deletion_fraction": (chrom_data["segment_mean"] < -0.5).mean(),
                }

    return pd.DataFrame(results).T if results else pd.DataFrame()


def compute_ddr_mutation_burden(
    mutations: pd.DataFrame,
    ddr_genes: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, float]:
    """
    Compute DDR pathway mutation burden.

    Args:
        mutations: DataFrame with [sample_id, gene, variant_type]
        ddr_genes: DDR pathway gene sets

    Returns:
        Dict with per-pathway mutation counts and burden
    """
    if ddr_genes is None:
        ddr_genes = DDR_GENES

    if mutations.empty:
        return {path: 0.0 for path in ddr_genes}

    results = {}
    for pathway, genes in ddr_genes.items():
        pathway_genes = [g for g in genes]
        if "gene" in mutations.columns:
            count = mutations[mutations["gene"].isin(pathway_genes)].shape[0]
        elif "Hugo_Symbol" in mutations.columns:
            count = mutations[mutations["Hugo_Symbol"].isin(pathway_genes)].shape[0]
        else:
            count = 0
        results[f"{pathway.lower()}_burden"] = count

    return results


def compute_genomic_features(
    mutations: Optional[pd.DataFrame] = None,
    cnv_data: Optional[pd.DataFrame] = None,
    expr_matrix: Optional[pd.DataFrame] = None,
    config: Optional[GenomicConfig] = None
) -> pd.DataFrame:
    """
    Compute combined genomic feature matrix.

    Args:
        mutations: Mutation data
        cnv_data: CNV data
        expr_matrix: Gene expression (for MSI proxy)
        config: Configuration

    Returns:
        DataFrame with genomic features per sample
    """
    if config is None:
        config = GenomicConfig()

    features = {}

    # TMB
    if mutations is not None and not mutations.empty:
        if "sample_id" in mutations.columns:
            sample_tmb = {}
            for sid in mutations["sample_id"].unique():
                sample_mut = mutations[mutations["sample_id"] == sid]
                sample_tmb[sid] = compute_tmb(sample_mut, config.genome_size_mb)
            features["tmb"] = sample_tmb
        else:
            features["tmb"] = {"sample_1": compute_tmb(mutations, config.genome_size_mb)}

    # CNV
    if cnv_data is not None and not cnv_data.empty:
        cnv_scores = compute_cnv_scores(cnv_data)
        for k, v in cnv_scores.items():
            features[k] = v

    # DDR
    if mutations is not None and not mutations.empty:
        ddr = compute_ddr_mutation_burden(mutations)
        features.update(ddr)

    # Aneuploidy score (simple proxy from CNV data)
    if cnv_data is not None and not cnv_data.empty and "segment_mean" in cnv_data.columns:
        features["aneuploidy_score"] = (
            (cnv_data["segment_mean"].abs() > 0.5).sum() /
            max(cnv_data.shape[0], 1)
        )
    else:
        features["aneuploidy_score"] = 0.0

    # MSI proxy (from expression of mismatch repair genes)
    if expr_matrix is not None:
        msi_genes = ["MLH1", "MSH2", "MSH6", "PMS2"]
        present = [g for g in msi_genes if g in expr_matrix.columns]
        if present:
            features["msi_proxy"] = expr_matrix[present].mean(axis=1).to_dict()
        else:
            features["msi_proxy"] = {sid: 0.0 for sid in expr_matrix.index}
    else:
        features["msi_proxy"] = {}

    # Build DataFrame
    all_samples = set()
    for v in features.values():
        if isinstance(v, dict):
            all_samples.update(v.keys())

    rows = []
    for sid in all_samples:
        row = {"sample_id": sid}
        for fname, fval in features.items():
            if isinstance(fval, dict):
                row[fname] = fval.get(sid, 0.0)
            else:
                row[fname] = fval
        rows.append(row)

    return pd.DataFrame(rows).set_index("sample_id")