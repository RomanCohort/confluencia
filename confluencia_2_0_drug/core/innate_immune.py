"""Innate immune sensing module for circRNA therapeutics.

Models three key RNA-sensing pathways that determine the immunogenicity
of circRNA drugs:
  1. TLR3/7/8 (endosomal RNA sensing) → NF-κB → pro-inflammatory cytokines
  2. RIG-I/MDA5 (cytosolic RNA sensing) → IFN-I production
  3. PKR (Protein Kinase R) → translation inhibition

All models use literature-derived parameters and sequence-based heuristics,
avoiding heavy external dependencies (no structural RNA folding tools required).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ===================================================================
# TLR3/7/8 pathway model
# ===================================================================

# TLR3 prefers dsRNA > 40 bp; TLR7/8 prefer ssRNA with GU-rich motifs.
# Activation scores are sequence-composition based heuristics calibrated
# against published immunogenicity data.

# GU-rich motif weights (TLR7/8 ligands)
_GU_MOTIFS: List[Tuple[str, float]] = [
    ("GU", 1.0), ("UG", 0.9), ("GUU", 1.3), ("UGU", 1.2),
    ("GUG", 1.1), ("UGUG", 1.4), ("GUUG", 1.3),
]

# dsRNA motif indicators (TLR3 ligands)
_DSRNA_MOTIFS: List[Tuple[str, float]] = [
    ("GCGC", 0.8), ("CGCG", 0.9), ("GCGCGC", 1.2),
    ("CGCGCG", 1.3), ("CCGG", 0.7),
]


def _compute_gu_richness(seq: str) -> float:
    """Compute GU-richness score (0-1) for TLR7/8 activation."""
    s = seq.upper()
    n = max(len(s), 1)
    score = 0.0
    for motif, weight in _GU_MOTIFS:
        count = 0
        pos = 0
        while True:
            idx = s.find(motif, pos)
            if idx < 0:
                break
            count += 1
            pos = idx + 1
        score += weight * count
    # Normalize by sequence length; cap at 1.0
    return float(np.clip(score / (n * 0.3), 0.0, 1.0))


def _compute_dsrna_content(seq: str) -> float:
    """Estimate dsRNA content score (0-1) based on complementary motifs."""
    s = seq.upper()
    n = max(len(s), 1)
    score = 0.0
    for motif, weight in _DSRNA_MOTIFS:
        count = s.count(motif)
        if count > 0:
            score += weight * min(count, 5)
    return float(np.clip(score / (n * 0.1), 0.0, 1.0))


def _compute_seq_length_factor(seq: str) -> float:
    """TLR activation scales with RNA length (longer = more immunogenic)."""
    n = len(seq)
    if n < 20:
        return 0.2
    elif n < 100:
        return 0.2 + 0.4 * (n - 20) / 80.0
    elif n < 500:
        return 0.6 + 0.3 * (n - 100) / 400.0
    else:
        return 0.9 + 0.1 * min((n - 500) / 1000.0, 1.0)


def compute_tlr_activation(seq: str) -> Dict[str, float]:
    """Compute TLR3/7/8 pathway activation scores.

    Returns dict with:
      - tlr3_score: TLR3 (dsRNA) activation (0-1)
      - tlr7_score: TLR7 (ssRNA GU-rich) activation (0-1)
      - tlr8_score: TLR8 (ssRNA GU-rich) activation (0-1)
      - tlr_combined: weighted combination (0-1)
      - nfkb_activation: NF-κB downstream signal (0-1)
      - pro_inflammatory: pro-inflammatory cytokine estimate (0-1)
    """
    s = seq.upper().replace(" ", "")
    if not s or len(s) < 10:
        return {
            "tlr3_score": 0.0, "tlr7_score": 0.0, "tlr8_score": 0.0,
            "tlr_combined": 0.0, "nfkb_activation": 0.0, "pro_inflammatory": 0.0,
        }

    length_factor = _compute_seq_length_factor(s)
    gu_score = _compute_gu_richness(s)
    dsrna_score = _compute_dsrna_content(s)

    # TLR3: primarily dsRNA-sensing
    tlr3 = float(np.clip(0.3 * dsrna_score + 0.2 * length_factor + 0.1 * gu_score, 0.0, 1.0))
    # TLR7: prefers GU-rich ssRNA, inhibited by dsRNA structure
    tlr7 = float(np.clip(0.5 * gu_score + 0.2 * length_factor - 0.1 * dsrna_score, 0.0, 1.0))
    # TLR8: similar to TLR7 but less sensitive to GU, more to AU
    au_frac = (s.count("A") + s.count("U")) / max(len(s), 1)
    tlr8 = float(np.clip(0.3 * gu_score + 0.2 * au_frac + 0.15 * length_factor, 0.0, 1.0))

    tlr_combined = float(np.clip(0.4 * tlr3 + 0.35 * tlr7 + 0.25 * tlr8, 0.0, 1.0))
    # NF-κB activation scales with combined TLR signal
    nfkb = float(np.clip(0.7 * tlr_combined + 0.15 * tlr3, 0.0, 1.0))
    # Pro-inflammatory cytokine output (TNF-α, IL-6, IL-12)
    pro_inf = float(np.clip(0.6 * nfkb + 0.2 * tlr7 + 0.15 * tlr8, 0.0, 1.0))

    return {
        "tlr3_score": tlr3, "tlr7_score": tlr7, "tlr8_score": tlr8,
        "tlr_combined": tlr_combined, "nfkb_activation": nfkb, "pro_inflammatory": pro_inf,
    }


# ===================================================================
# RIG-I/MDA5 pathway model
# ===================================================================

def _compute_5ppp_score(seq: str) -> float:
    """Estimate 5'-triphosphate signature (RIG-I ligand).

    For circRNA, the backsplice junction creates a covalently closed loop
    without free 5' ends, so RIG-I activation is inherently low.
    Partial linear contaminants may still carry 5'ppp.
    """
    # circRNA: inherently low 5'ppp (closed loop)
    # We assume a small fraction of linear contaminant (~2-5%)
    contaminant_frac = 0.03
    # Linear RNA with 5'ppp would score ~0.7
    return float(np.clip(contaminant_frac * 0.7, 0.0, 1.0))


def _compute_dsrna_rigi_score(seq: str) -> float:
    """Compute dsRNA features relevant for RIG-I/MDA5.

    RIG-I: prefers short blunt-end dsRNA (< 300 bp)
    MDA5: prefers long dsRNA (> 1 kbp)
    """
    s = seq.upper()
    n = max(len(s), 1)
    dsrna = _compute_dsrna_content(s)

    # RIG-I: short dsRNA motifs
    rigi = float(np.clip(0.6 * dsrna * min(n / 300.0, 1.0) + 0.1 * _compute_5ppp_score(s), 0.0, 1.0))
    # MDA5: long dsRNA motifs
    mda5 = float(np.clip(0.7 * dsrna * min(n / 1000.0, 1.0), 0.0, 1.0))

    return rigi, mda5


def compute_rigi_mda5_activation(seq: str) -> Dict[str, float]:
    """Compute RIG-I/MDA5 pathway activation scores.

    Returns dict with:
      - rigi_score: RIG-I activation (0-1)
      - mda5_score: MDA5 activation (0-1)
      - cytosolic_rna_sensing: combined cytosolic sensor score (0-1)
      - ifn_alpha_beta: type-I interferon production estimate (0-1)
    """
    s = seq.upper().replace(" ", "")
    if not s or len(s) < 10:
        return {
            "rigi_score": 0.0, "mda5_score": 0.0,
            "cytosolic_rna_sensing": 0.0, "ifn_alpha_beta": 0.0,
        }

    rigi, mda5 = _compute_dsrna_rigi_score(s)

    # Combined cytosolic sensing
    cytosolic = float(np.clip(0.55 * rigi + 0.45 * mda5, 0.0, 1.0))

    # IFN-I production: driven by RIG-I/MDA5 via MAVS pathway
    ifn = float(np.clip(0.6 * cytosolic + 0.2 * rigi + 0.15 * mda5, 0.0, 1.0))

    return {
        "rigi_score": rigi, "mda5_score": mda5,
        "cytosolic_rna_sensing": cytosolic, "ifn_alpha_beta": ifn,
    }


# ===================================================================
# PKR pathway model
# ===================================================================

def compute_pkr_activation(seq: str) -> Dict[str, float]:
    """Compute PKR (Protein Kinase R) activation score.

    PKR is activated by long (> 30 bp) dsRNA stretches.
    Activated PKR phosphorylates eIF2α, inhibiting translation.

    Returns dict with:
      - pkr_score: PKR activation level (0-1)
      - translation_inhibition: estimated translation inhibition fraction (0-1)
    """
    s = seq.upper().replace(" ", "")
    if not s or len(s) < 30:
        return {"pkr_score": 0.0, "translation_inhibition": 0.0}

    # Estimate longest dsRNA stretch using complementary pairing
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    n = len(s)
    max_dsrna_stretch = 0
    current_stretch = 0

    for i in range(n - 1):
        if s[i + 1] == comp.get(s[i], ""):
            current_stretch += 1
            max_dsrna_stretch = max(max_dsrna_stretch, current_stretch)
        else:
            current_stretch = 0

    # PKR requires > ~15 bp perfect dsRNA for activation, strong activation at > 30 bp
    if max_dsrna_stretch < 5:
        pkr = 0.05
    elif max_dsrna_stretch < 15:
        pkr = 0.05 + 0.15 * (max_dsrna_stretch - 5) / 10.0
    elif max_dsrna_stretch < 30:
        pkr = 0.20 + 0.40 * (max_dsrna_stretch - 15) / 15.0
    else:
        pkr = 0.60 + 0.40 * min((max_dsrna_stretch - 30) / 30.0, 1.0)

    pkr = float(np.clip(pkr, 0.0, 1.0))
    # Translation inhibition: PKR phosphorylates eIF2α
    trans_inhib = float(np.clip(0.8 * pkr, 0.0, 1.0))

    return {"pkr_score": pkr, "translation_inhibition": trans_inhib}


# ===================================================================
# Comprehensive innate immune scoring
# ===================================================================

@dataclass
class InnateImmuneResult:
    """Complete innate immune assessment result."""
    # Individual pathway scores
    tlr3: float
    tlr7: float
    tlr8: float
    rigi: float
    mda5: float
    pkr: float

    # Downstream signals
    nfkb: float
    pro_inflammatory: float
    ifn_alpha_beta: float
    translation_inhibition: float

    # Composite scores
    innate_immune_score: float
    interferon_storm_risk: float
    interferon_storm_level: str
    modification_evasion: float
    net_safety_score: float  # 0 = dangerous, 1 = safe


def assess_innate_immune(
    seq: str,
    modification: str = "none",
    delivery_vector: str = "LNP_standard",
) -> InnateImmuneResult:
    """Perform comprehensive innate immune assessment for a circRNA sequence.

    Combines TLR, RIG-I/MDA5, and PKR pathway scores with modification-based
    immune evasion to produce an overall safety assessment.

    Args:
        seq: circRNA sequence (AUGC)
        modification: modification type (m6A, Psi, 5mC, etc.)
        delivery_vector: delivery system type

    Returns:
        InnateImmuneResult with all pathway scores and composite assessments
    """
    s = seq.upper().replace(" ", "")
    if not s or len(s) < 10:
        return InnateImmuneResult(
            tlr3=0.0, tlr7=0.0, tlr8=0.0,
            rigi=0.0, mda5=0.0, pkr=0.0,
            nfkb=0.0, pro_inflammatory=0.0, ifn_alpha_beta=0.0,
            translation_inhibition=0.0, innate_immune_score=0.0,
            interferon_storm_risk=0.0, interferon_storm_level="low",
            modification_evasion=0.0, net_safety_score=1.0,
        )

    # Get pathway scores
    tlr = compute_tlr_activation(s)
    rigi_mda5 = compute_rigi_mda5_activation(s)
    pkr = compute_pkr_activation(s)

    # Modification immune evasion
    mod = str(modification).lower().strip()
    mod_evasion_map = {
        "none": 0.0, "m6a": 0.25, "ψ": 0.55, "\u03c8": 0.55,
        "5mc": 0.35, "ms2m6a": 0.50, "psi": 0.55,
    }
    mod_evasion = mod_evasion_map.get(mod, mod_evasion_map["none"])

    # Delivery system factor: LNP can have innate immune effects itself
    vec = str(delivery_vector).strip()
    lnp_immune_factor = {"LNP_standard": 0.05, "LNP_liver": 0.04,
                         "LNP_spleen": 0.06, "AAV": 0.02, "naked": 0.0}
    lnp_bonus = lnp_immune_factor.get(vec, 0.05)

    # Combined innate immune score (before modification evasion)
    raw_score = float(np.clip(
        0.30 * tlr["tlr_combined"]
        + 0.25 * rigi_mda5["cytosolic_rna_sensing"]
        + 0.15 * pkr["pkr_score"]
        + 0.15 * tlr["pro_inflammatory"]
        + 0.15 * rigi_mda5["ifn_alpha_beta"]
        + lnp_bonus,
        0.0, 1.0,
    ))

    # Apply modification evasion (reduces immune activation)
    effective_score = float(np.clip(raw_score * (1.0 - mod_evasion), 0.0, 1.0))

    # Interferon storm risk: driven primarily by IFN-I and pro-inflammatory signals
    ifn_risk = float(np.clip(
        0.5 * rigi_mda5["ifn_alpha_beta"] * (1.0 - mod_evasion * 0.5)
        + 0.3 * tlr["pro_inflammatory"] * (1.0 - mod_evasion * 0.3)
        + 0.2 * tlr["nfkb_activation"] * (1.0 - mod_evasion * 0.3),
        0.0, 1.0,
    ))

    # Risk level classification
    if ifn_risk < 0.15:
        level = "low"
    elif ifn_risk < 0.35:
        level = "medium"
    elif ifn_risk < 0.60:
        level = "high"
    else:
        level = "critical"

    # Net safety score: 1 = very safe, 0 = dangerous
    # Accounts for innate activation, PKR translation inhibition, and modification benefit
    safety = float(np.clip(
        1.0 - 0.4 * effective_score
            - 0.2 * pkr["translation_inhibition"]
            - 0.1 * ifn_risk
            + 0.3 * mod_evasion,
        0.0, 1.0,
    ))

    return InnateImmuneResult(
        tlr3=tlr["tlr3_score"],
        tlr7=tlr["tlr7_score"],
        tlr8=tlr["tlr8_score"],
        rigi=rigi_mda5["rigi_score"],
        mda5=rigi_mda5["mda5_score"],
        pkr=pkr["pkr_score"],
        nfkb=tlr["nfkb_activation"],
        pro_inflammatory=tlr["pro_inflammatory"],
        ifn_alpha_beta=rigi_mda5["ifn_alpha_beta"],
        translation_inhibition=pkr["translation_inhibition"],
        innate_immune_score=effective_score,
        interferon_storm_risk=ifn_risk,
        interferon_storm_level=level,
        modification_evasion=mod_evasion,
        net_safety_score=safety,
    )


def innate_immune_result_to_dict(result: InnateImmuneResult) -> Dict[str, float]:
    """Convert InnateImmuneResult to a flat dictionary for DataFrame integration."""
    return {
        "innate_tlr3": result.tlr3,
        "innate_tlr7": result.tlr7,
        "innate_tlr8": result.tlr8,
        "innate_rigi": result.rigi,
        "innate_mda5": result.mda5,
        "innate_pkr": result.pkr,
        "innate_nfkb": result.nfkb,
        "innate_pro_inflammatory": result.pro_inflammatory,
        "innate_ifn_alpha_beta": result.ifn_alpha_beta,
        "innate_translation_inhibition": result.translation_inhibition,
        "innate_immune_score": result.innate_immune_score,
        "innate_ifn_storm_risk": result.interferon_storm_risk,
        "innate_ifn_storm_level": result.interferon_storm_level,
        "innate_mod_evasion": result.modification_evasion,
        "innate_safety_score": result.net_safety_score,
    }


def batch_assess_innate_immune(
    df,
    seq_col: str = "circrna_seq",
    mod_col: str = "modification",
    vec_col: str = "delivery_vector",
) -> List[Dict[str, float]]:
    """Batch assess innate immune responses for a DataFrame."""
    results = []
    for _, row in df.iterrows():
        r = assess_innate_immune(
            seq=str(row.get(seq_col, "")),
            modification=str(row.get(mod_col, "none")),
            delivery_vector=str(row.get(vec_col, "LNP_standard")),
        )
        results.append(innate_immune_result_to_dict(r))
    return results
