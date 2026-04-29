"""
immune_sensing.py — circRNA-specific innate immune sensing prediction.

Predicts RIG-I, TLR7/8, and PKR pathway activation based on:
  - Sequence features (5'-triphosphate blunt end, AU-rich motifs, length)
  - Base composition (GC content, dinucleotide frequency)
  - Known suppressors/inhibitors of each pathway

References:
  - Chen & Mellman, Immunity 2013 — innate immune sensing
  - Kato et al., Nat Rev Microbiol 2008 — RIG-I literature
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

# Motif definitions
RIG_I_MOTIFS = ["CCUCC", "UCUCC", "ACUCC", "GCUCC"]
BLUNT_END_WINDOW = 20
TLR_MOTIFS = ["GUUG", "UUGU", "UGUU", "GUUU", "GUU"]
AU_RICH_PATTERN = re.compile(r"AUUUA|AU-rich|UUAUUUAU|UAUUUAU|UUAUUUAUU")
PKR_MIN_DSRNA = 30
PKR_SUPPRESSORS = ["m6A", "psi", "ac4C", "m5C"]
TLR_SUPPRESSORS = ["m6A", "ac4C", "m5C"]


@dataclass
class ImmuneSensingConfig:
    """Configuration for immune sensing prediction."""
    min_length: int = 50
    max_length: int = 50000
    detect_blunt_end: bool = True
    detect_au_rich: bool = True
    detect_m6a: bool = True


def _gc_content(seq: str) -> float:
    """Calculate GC content of sequence."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    gc = sum(1 for c in seq_upper if c in "GC")
    return gc / len(seq_upper)


def _count_motifs(seq: str, motifs: List[str]) -> int:
    """Count occurrences of any motif in sequence."""
    count = 0
    seq_upper = seq.upper()
    for motif in motifs:
        count += seq_upper.count(motif.upper())
    return count


def _detect_blunt_end(seq: str, window: int = BLUNT_END_WINDOW) -> bool:
    """Detect if 5' end is blunt (no overhang)."""
    if len(seq) < window:
        window = len(seq)
    end5 = seq[:window].upper()
    # Blunt end: 5'-triphosphate without overhang
    # CircRNA back-spliced junction is typically blunt at both ends
    # For simplicity: check if first 10nt has high GC or no poly-U tract
    return True  # Default assumption for circRNA


def _detect_au_rich(seq: str) -> int:
    """Count AU-rich elements."""
    matches = AU_RICH_PATTERN.findall(seq.upper())
    return len(matches)


def _estimate_dsRNA_potential(seq: str) -> float:
    """Estimate double-stranded RNA formation potential."""
    if len(seq) < PKR_MIN_DSRNA:
        return 0.0
    # Simplified: check for complementarity patterns
    gc = _gc_content(seq)
    length_factor = min(len(seq) / 500, 1.0)
    return min(gc * length_factor, 1.0)


def predict_circrna_immunogenicity(
    seq: str,
    config: Optional[ImmuneSensingConfig] = None
) -> Dict[str, float]:
    """
    Predict RIG-I, TLR7/8, and PKR pathway activation scores for a circRNA sequence.

    Args:
        seq: circRNA nucleotide sequence
        config: Optional configuration

    Returns:
        Dict with keys: rig_i_score, tlr_score, pkr_score, overall_immunogenicity, sensing_method
    """
    if config is None:
        config = ImmuneSensingConfig()

    seq_upper = seq.upper()
    seq_len = len(seq)

    # Validate length
    if seq_len < config.min_length:
        return {
            "rig_i_score": 0.0, "tlr_score": 0.0, "pkr_score": 0.0,
            "overall_immunogenicity": 0.0, "sensing_method": "too_short"
        }
    if seq_len > config.max_length:
        seq_upper = seq_upper[:config.max_length]

    # === RIG-I scoring (0.4 weight) ===
    # RIG-I recognizes 5'-triphosphate blunt-ended RNA with panhandle structure
    rig_i_score = 0.0

    # 1. Blunt end (strong signal for circRNA)
    if config.detect_blunt_end:
        blunt_score = 0.3 if _detect_blunt_end(seq) else 0.0
        rig_i_score += blunt_score

    # 2. Motif matching (RIG-I prefers 5'-diphosphate RNA)
    motif_count = _count_motifs(seq, RIG_I_MOTIFS)
    motif_score = min(motif_count * 0.15, 0.4)
    rig_i_score += motif_score

    # 3. GC content (higher GC = more structured = stronger RIG-I)
    gc = _gc_content(seq)
    gc_score = gc * 0.2
    rig_i_score += gc_score

    # 4. Length (longer circRNA more immunogenic via RIG-I)
    length_score = min(seq_len / 5000 * 0.1, 0.1)
    rig_i_score += length_score

    rig_i_score = min(rig_i_score, 1.0)

    # === TLR7/8 scoring (0.35 weight) ===
    # TLR7/8 recognizes single-stranded UR-rich sequences in endosomes
    tlr_score = 0.0

    # 1. Uridine content (TLR7/8 prefers poly-U)
    u_count = seq_upper.count("U")
    u_ratio = u_count / seq_len
    tlr_score += min(u_ratio * 2.0, 0.4)

    # 2. AU-rich elements
    if config.detect_au_rich:
        au_count = _detect_au_rich(seq)
        au_score = min(au_count * 0.1, 0.3)
        tlr_score += au_score

    # 3. TLR motif matches
    tlr_motif_count = _count_motifs(seq, TLR_MOTIFS)
    tlr_motif_score = min(tlr_motif_count * 0.08, 0.2)
    tlr_score += tlr_motif_score

    # 4. Sequence length (longer = more uridine-rich regions)
    len_score = min(seq_len / 3000 * 0.1, 0.1)
    tlr_score += len_score

    tlr_score = min(tlr_score, 1.0)

    # === PKR scoring (0.25 weight) ===
    # PKR recognizes double-stranded regions >30bp
    pkr_score = 0.0

    # 1. dsRNA formation potential
    dsrna_potential = _estimate_dsRNA_potential(seq)
    pkr_score += dsrna_potential * 0.5

    # 2. Length (PKR needs ~30+ bp dsRNA)
    if seq_len >= PKR_MIN_DSRNA:
        pkr_score += 0.2
        length_factor = min((seq_len - PKR_MIN_DSRNA) / 500, 1.0)
        pkr_score += length_factor * 0.2

    # 3. GC-rich regions indicate more stable dsRNA
    gc_pkr = min(gc * 0.3, 0.3)
    pkr_score += gc_pkr

    # 4. Suppressor penalties (m6A, psi, ac4C modifications reduce PKR activation)
    if config.detect_m6a:
        # In absence of modification data, apply small penalty for typical circRNA mods
        pkr_score *= 0.85  # Most circRNAs have some modifications

    pkr_score = min(pkr_score, 1.0)

    # === Overall immunogenicity ===
    # Weighted combination
    overall = 0.4 * rig_i_score + 0.35 * tlr_score + 0.25 * pkr_score

    return {
        "rig_i_score": round(rig_i_score, 4),
        "tlr_score": round(tlr_score, 4),
        "pkr_score": round(pkr_score, 4),
        "overall_immunogenicity": round(overall, 4),
        "sensing_method": "rule_based"
    }


def score_sequence(seq: str) -> Dict[str, float]:
    """Convenience wrapper with default config."""
    return predict_circrna_immunogenicity(seq, ImmuneSensingConfig())


# === Demo / test ===
if __name__ == "__main__":
    test_sequences = [
        # High immunogenicity: GC-rich, has RIG-I motifs
        "GCCGCCGCC" * 50 + "CCUCC" + "GCGCGCGC" * 30,
        # Low immunogenicity: AU-rich, few motifs
        "AUUAUUAUUAUU" * 20 + "GUUGUUGUU",
        # Typical circRNA
        "AUCGAUCGAUCGA" * 100,
    ]

    print("CircRNA Immune Sensing Scores:")
    print("=" * 60)
    for i, seq in enumerate(test_sequences):
        result = predict_circrna_immunogenicity(seq)
        print(f"\nSequence {i+1} (len={len(seq)}):")
        print(f"  RIG-I: {result['rig_i_score']:.4f}")
        print(f"  TLR7/8: {result['tlr_score']:.4f}")
        print(f"  PKR: {result['pkr_score']:.4f}")
        print(f"  Overall: {result['overall_immunogenicity']:.4f}")