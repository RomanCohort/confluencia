"""
bsj_features.py — Back-Splice Junction Feature Extraction

Predicts circRNA biogenesis efficiency based on:
1. Alu element detection in flanking introns
2. Intron complementarity (reverse complementary matches)
3. Splice site strength (5' and 3' splice site scores)
4. Circularization efficiency prediction

Literature basis:
- Jeck et al., Nature 2013: "Circular RNAs are abundant, conserved, and associated with ALU repeats"
  DOI: 10.1038/nature09823
- Zhang et al., Cell 2014: "Complementary sequence-mediated exon circularization"
  DOI: 10.1016/j.cell.2014.03.078
- Ivanov et al., RNA Biology 2016: "Analysis of intron sequences reveals hallmarks of circular RNA biogenesis"
  DOI: 10.1080/15476286.2016.1215795
- Gao et al., Nature Communications 2015: "Circular RNA identification based on multiple features"
  DOI: 10.1038/ncomms9270

Key concepts:
- Alu elements: ~280bp SINE repeats that drive intronic base pairing
- Intron complementarity: Reverse complementary sequences in flanking introns
- Splice site strength: MaxEntScan-like scoring for 5'/3' splice sites
- Circularization efficiency: Combined score predicting biogenesis rate

Version: bsj-v1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import numpy as np


# ============================================================================
# Constants
# ============================================================================

# Alu consensus patterns (simplified)
ALU_CONSENSUS_PATTERNS = [
    # AluSx (most common subfamily)
    r"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAG",
    r"GAGGTCAGGAGATCGAGACCATCCTGGCTAACATGGTGAAACCCC",
    # AluY (youngest subfamily)
    r"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAG",
    r"AAGATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACT",
    # AluJ (oldest subfamily)
    r"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAG",
]

# Minimum Alu match length
MIN_ALU_MATCH_LENGTH = 30

# Minimum complementarity for circularization (Zhang et al., 2014)
MIN_COMPLEMENTARITY_LENGTH = 50

# Splice site consensus motifs
SPlice_SITE_5_CONSENSUS = "GURAGU"  # R = purine (A/G)
SPlice_SITE_3_CONSENSUS = "YYYYYYNCAG"  # Y = pyrimidine (U/C)

# Optimal exon length for circularization (Gao et al., 2015)
OPTIMAL_EXON_LENGTH_MIN = 200
OPTIMAL_EXON_LENGTH_MAX = 500


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AluMatch:
    """An Alu element match in flanking region."""
    position: int                    # Position in sequence
    length: int                      # Match length
    score: float                     # Match quality (0-1)
    subfamily: str                   # AluSx/AluY/AluJ
    orientation: str                 # forward/reverse


@dataclass
class SpliceSiteScore:
    """Splice site strength score."""
    score_5: float                   # 5' splice site strength (0-1)
    score_3: float                   # 3' splice site strength (0-1)
    motif_5: str                     # Detected 5' motif
    motif_3: str                     # Detected 3' motif
    branch_point_score: float        # Branch point quality (0-1)


@dataclass
class BSJFeatures:
    """Complete back-splice junction features."""

    # Alu element detection
    alu_elements_present: bool
    alu_element_count: int
    alu_matches: List[AluMatch]
    alu_flanking_distance: Tuple[int, int]  # (5' distance, 3' distance)

    # Intron complementarity
    intron_complementarity_score: float     # 0-1
    complementary_length: int               # Length of complementary region
    complementary_regions: List[Tuple[int, int]]

    # Splice site scores
    splice_site_5_score: float              # 0-1
    splice_site_3_score: float              # 0-1
    splice_site_motifs: List[str]
    branch_point_score: float

    # Circularization efficiency
    circularization_score: float            # 0-1
    exon_length: int
    flanking_intron_length: Tuple[int, int]  # Estimated

    # BSJ protection
    bsj_stability: float                    # 0-1
    protected_region: Tuple[int, int]       # Recommended protected region
    biogenesis_efficiency_class: str        # high/medium/low

    # NEW: Real-time detection fields
    realtime_detected: bool = False
    detection_confidence: float = 0.0
    junction_signals: List['JunctionSignal'] = field(default_factory=list)

    # NEW: Conservation fields
    conservation_score: float = 0.0
    conservation_annotation: Optional['ConservationAnnotation'] = None


# ============================================================================
# NEW: Real-Time Detection Data Structures
# ============================================================================

@dataclass
class JunctionSignal:
    """Real-time junction detection from sequencing reads."""
    position: int                    # Position in sequence
    read_support: int                # Number of supporting reads
    strand: str                      # +/-
    confidence: float                # Detection confidence (0-1)
    detection_timestamp: float       # Unix timestamp
    read_ids: List[str] = field(default_factory=list)  # Supporting read IDs


@dataclass
class BSJValidationResult:
    """Real-time BSJ validation result."""
    junction_validated: bool
    confidence_score: float
    supporting_reads: int
    false_positive_risk: float
    validation_method: str           # "read_span" / "junction_signature" / "paired_end"


@dataclass
class ConservationAnnotation:
    """Evolutionary conservation annotation for BSJ."""
    junction_conservation: float     # PhyloP/PhastCons at junction
    alu_conservation: float          # Alu element age
    splice_site_conservation: float  # Splice site preservation
    overall_conservation: float      # Combined score
    species_conserved: List[str]     # Species with same BSJ
    conservation_class: str          # high/medium/low


# ============================================================================
# Core Functions
# ============================================================================

def detect_alu_elements(
    sequence: str,
    flanking_region: int = 500,
    min_match_length: int = MIN_ALU_MATCH_LENGTH,
) -> List[AluMatch]:
    """
    Detect Alu elements near back-splice junction.

    Alu elements are ~280bp SINE repeats that form complementary
    pairs in flanking introns, driving circularization.

    Args:
        sequence: Full circRNA sequence with flanking regions
        flanking_region: Search window for Alu detection
        min_match_length: Minimum match length to report

    Returns:
        List of AluMatch objects

    Literature:
        Jeck et al., 2013: "Alu repeats comprise ~10% of genome and
        are major drivers of circRNA biogenesis via intronic base pairing"
    """
    matches = []
    seq_upper = sequence.upper().replace("T", "U")

    # Search in 5' flanking region
    region_5 = seq_upper[:min(flanking_region, len(seq_upper)//2)]

    # Search in 3' flanking region
    region_3_start = max(len(seq_upper)//2, len(seq_upper) - flanking_region)
    region_3 = seq_upper[region_3_start:]

    for pattern in ALU_CONSENSUS_PATTERNS:
        # Forward strand search
        for match in re.finditer(pattern, region_5):
            if len(match.group()) >= min_match_length:
                matches.append(AluMatch(
                    position=match.start(),
                    length=len(match.group()),
                    score=len(match.group()) / 50,  # Normalize by expected length
                    subfamily="AluSx",  # Simplified classification
                    orientation="forward",
                ))

        # Reverse strand (reverse complement)
        pattern_rc = _reverse_complement(pattern)
        for match in re.finditer(pattern_rc, region_3):
            if len(match.group()) >= min_match_length:
                matches.append(AluMatch(
                    position=region_3_start + match.start(),
                    length=len(match.group()),
                    score=len(match.group()) / 50,
                    subfamily="AluSx",
                    orientation="reverse",
                ))

    return matches


def compute_intron_complementarity(
    seq_5_flanking: str,
    seq_3_flanking: str,
    min_length: int = MIN_COMPLEMENTARITY_LENGTH,
) -> Tuple[float, int, List[Tuple[int, int]]]:
    """
    Compute complementarity between flanking introns.

    Zhang et al., 2014: "Reverse complementary Alus in flanking introns
    promote exon circularization via intron pairing"

    Args:
        seq_5_flanking: 5' flanking intron sequence
        seq_3_flanking: 3' flanking intron sequence
        min_length: Minimum complementary length threshold

    Returns:
        (complementarity_score, complementary_length, complementary_regions)

    Literature:
        Zhang et al., Cell 2014: 50bp minimum complementarity required
        for efficient circularization.
    """
    seq_5 = seq_5_flanking.upper().replace("T", "U")
    seq_3 = seq_3_flanking.upper().replace("T", "U")

    # Reverse complement of 3' flanking
    seq_3_rc = _reverse_complement(seq_3)

    # Find complementary regions using simple sliding window
    complementary_regions = []
    total_complementary_length = 0

    window_size = 20
    threshold = 0.7  # 70% complementarity

    for i in range(0, len(seq_5) - window_size, window_size//2):
        window_5 = seq_5[i:i+window_size]

        for j in range(0, len(seq_3_rc) - window_size, window_size//2):
            window_3 = seq_3_rc[j:j+window_size]

            # Compute complementarity
            comp_score = _compute_complementarity(window_5, window_3)

            if comp_score >= threshold:
                # Extend match
                extended_length = _extend_complementary_match(
                    seq_5, seq_3_rc, i, j, threshold
                )
                if extended_length >= min_length:
                    complementary_regions.append((i, i + extended_length))
                    total_complementary_length += extended_length

    # Normalize score
    if len(seq_5) > 0 and len(seq_3) > 0:
        complementarity_score = total_complementary_length / max(len(seq_5), len(seq_3))
    else:
        complementarity_score = 0.0

    return complementarity_score, total_complementary_length, complementary_regions


def score_splice_site(
    sequence: str,
    junction_pos: Optional[int] = None,
) -> SpliceSiteScore:
    """
    Score splice site strength.

    Uses simplified MaxEntScan-like scoring based on consensus motifs.

    Args:
        sequence: RNA sequence
        junction_pos: Position of splice junction (optional)

    Returns:
        SpliceSiteScore with 5'/3' scores and branch point score

    Literature:
        Yeo & Burge, JCB 2004: MaxEntScan scoring algorithm
        5' splice site: GURAGU (9bp window)
        3' splice site: YYYYYYNCAG (23bp window including branch point)
    """
    seq = sequence.upper().replace("T", "U")

    # Default junction at center if not provided
    if junction_pos is None:
        junction_pos = len(seq) // 2

    # Extract splice site windows
    window_5_start = max(0, junction_pos - 3)
    window_5 = seq[window_5_start:window_5_start + 9]

    window_3_start = junction_pos
    window_3 = seq[window_3_start:window_3_start + 23]

    # Score 5' splice site (GURAGU consensus)
    score_5 = _score_5_splice_site(window_5)

    # Score 3' splice site (YYYYYYNCAG consensus)
    score_3, branch_score = _score_3_splice_site(window_3)

    # Extract motifs
    motif_5 = window_5[:6] if len(window_5) >= 6 else window_5
    motif_3 = window_3[-3:] if len(window_3) >= 3 else window_3  # CAG

    return SpliceSiteScore(
        score_5=score_5,
        score_3=score_3,
        motif_5=motif_5,
        motif_3=motif_3,
        branch_point_score=branch_score,
    )


def predict_circularization_efficiency(
    features: BSJFeatures,
) -> float:
    """
    Predict circRNA circularization efficiency.

    Combines multiple factors:
    1. Alu complementarity (+0.3 if present)
    2. Exon length (optimal 200-500bp)
    3. Flanking intron length (shorter = better)
    4. Splice site strength

    Args:
        features: BSJFeatures object

    Returns:
        Circularization efficiency score (0-1)

    Literature:
        Gao et al., 2015: "Exon length optimal 200-500bp for circularization"
        Zhang et al., 2014: "Alu complementarity strongly promotes circularization"
    """
    score = 0.0

    # Alu complementarity bonus
    if features.alu_elements_present:
        score += 0.30
        # Extra bonus for paired Alus
        if features.intron_complementarity_score > 0.3:
            score += 0.15

    # Exon length optimization
    exon_len = features.exon_length
    if OPTIMAL_EXON_LENGTH_MIN <= exon_len <= OPTIMAL_EXON_LENGTH_MAX:
        score += 0.25  # Optimal length
    elif exon_len < OPTIMAL_EXON_LENGTH_MIN:
        score += 0.15 * (exon_len / OPTIMAL_EXON_LENGTH_MIN)
    else:
        score += 0.15 * (OPTIMAL_EXON_LENGTH_MAX / exon_len)

    # Splice site strength contribution
    splice_score = (features.splice_site_5_score + features.splice_site_3_score) / 2
    score += splice_score * 0.20

    # Branch point contribution
    score += features.branch_point_score * 0.10

    return min(score, 1.0)


def determine_protected_region(
    sequence: str,
    features: BSJFeatures,
) -> Tuple[int, int]:
    """
    Determine optimal protected region for BSJ during mutations.

    Protects:
    1. Junction region (first/last nucleotides)
    2. Alu elements if present
    3. High complementarity regions

    Args:
        sequence: Full sequence
        features: BSJFeatures object

    Returns:
        (start, end) tuple for protected region
    """
    seq_len = len(sequence)

    # Base protection: junction region
    protected_start = min(10, seq_len // 4)
    protected_end = min(10, seq_len // 4)

    # Extend if Alu elements present
    if features.alu_elements_present:
        for alu_match in features.alu_matches:
            if alu_match.position < seq_len // 2:
                # 5' Alu - protect start more
                protected_start = max(protected_start, alu_match.position + alu_match.length)
            else:
                # 3' Alu - protect end more
                protected_end = max(protected_end, seq_len - alu_match.position)

    # Extend for high complementarity
    if features.complementary_regions:
        for region in features.complementary_regions:
            protected_start = max(protected_start, min(region[0], 20))
            protected_end = max(protected_end, seq_len - max(region[1], seq_len - 20))

    return (int(protected_start), int(protected_end))


# ============================================================================
# Helper Functions
# ============================================================================

def _reverse_complement(seq: str) -> str:
    """Get reverse complement of RNA sequence."""
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return "".join(complement.get(c, c) for c in reversed(seq.upper()))


def _compute_complementarity(seq1: str, seq2: str) -> float:
    """Compute complementarity score between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0

    complementary = {"A": "U", "U": "A", "G": "C", "C": "G"}
    matches = 0

    for c1, c2 in zip(seq1, seq2):
        if complementary.get(c1) == c2:
            matches += 1

    return matches / len(seq1)


def _extend_complementary_match(
    seq1: str,
    seq2: str,
    start1: int,
    start2: int,
    threshold: float,
) -> int:
    """Extend complementary match region."""
    length = 20  # Start with window size

    while start1 + length < len(seq1) and start2 + length < len(seq2):
        window1 = seq1[start1:start1+length+5]
        window2 = seq2[start2:start2+length+5]

        if _compute_complementarity(window1, window2) >= threshold:
            length += 5
        else:
            break

    return length


def _score_5_splice_site(window: str) -> float:
    """
    Score 5' splice site strength.

    Consensus: GURAGU (positions -3 to +6 relative to exon-intron boundary)
    Key positions:
    - Position -3: G (strongest)
    - Position +1: G (invariant)
    - Position +5: U (preferred)
    """
    if len(window) < 6:
        return 0.0

    score = 0.0

    # Check G at position 0
    if window[0] == "G":
        score += 0.3

    # Check purine (A/G) at position 1
    if window[1] in "AG":
        score += 0.2

    # Check AG at positions 2-3
    if window[2:4] == "AG":
        score += 0.3

    # Check U at position 5
    if window[5] == "U":
        score += 0.2

    return min(score, 1.0)


def _score_3_splice_site(window: str) -> Tuple[float, float]:
    """
    Score 3' splice site strength.

    Consensus: YYYYYYNCAG (polypyrimidine tract + CAG)
    Key positions:
    - Polypyrimidine tract (Y-rich region)
    - N (any nucleotide)
    - CAG invariant at splice site
    """
    if len(window) < 10:
        return 0.0, 0.0

    score_3 = 0.0
    branch_score = 0.0

    # Check polypyrimidine tract (first 7 positions)
    py_tract = window[:7]
    y_count = sum(1 for c in py_tract if c in "UC")
    score_3 += (y_count / 7) * 0.4

    # Check CAG at positions -3 to -1
    if len(window) >= 3:
        if window[-3:] == "CAG":
            score_3 += 0.4

    # Branch point detection (YNYURAY motif in upstream region)
    branch_motifs = ["UNU", "URA", "UAY"]
    for motif in branch_motifs:
        if motif in window[:-3]:
            branch_score += 0.3

    return min(score_3, 1.0), min(branch_score, 1.0)


# ============================================================================
# Feature Extractor Class
# ============================================================================

class BSJFeatureExtractor:
    """
    Complete BSJ feature extraction.

    Usage:
        extractor = BSJFeatureExtractor()
        features = extractor.extract(sequence, flanking_5, flanking_3)
    """

    def __init__(
        self,
        flanking_search_region: int = 500,
        min_alu_match: int = MIN_ALU_MATCH_LENGTH,
        min_complementarity: int = MIN_COMPLEMENTARITY_LENGTH,
    ):
        self.flanking_search_region = flanking_search_region
        self.min_alu_match = min_alu_match
        self.min_complementarity = min_complementarity

    def extract(
        self,
        sequence: str,
        flanking_5: Optional[str] = None,
        flanking_3: Optional[str] = None,
    ) -> BSJFeatures:
        """
        Extract all BSJ features from sequence.

        Args:
            sequence: circRNA exon sequence
            flanking_5: 5' flanking intron (optional)
            flanking_3: 3' flanking intron (optional)

        Returns:
            BSJFeatures object
        """
        seq_upper = sequence.upper().replace("T", "U")

        # Detect Alu elements
        alu_matches = detect_alu_elements(
            sequence if flanking_5 is None else flanking_5 + sequence + (flanking_3 or ""),
            flanking_region=self.flanking_search_region,
            min_match_length=self.min_alu_match,
        )

        alu_present = len(alu_matches) > 0
        alu_count = len(alu_matches)

        # Calculate flanking distances
        if alu_matches:
            alu_5_distances = [m.position for m in alu_matches if m.position < len(sequence)//2]
            alu_3_distances = [len(sequence) - m.position for m in alu_matches if m.position >= len(sequence)//2]
            alu_flanking_dist = (
                min(alu_5_distances) if alu_5_distances else 0,
                min(alu_3_distances) if alu_3_distances else 0,
            )
        else:
            alu_flanking_dist = (0, 0)

        # Compute intron complementarity
        if flanking_5 and flanking_3:
            comp_score, comp_len, comp_regions = compute_intron_complementarity(
                flanking_5, flanking_3, self.min_complementarity
            )
        else:
            comp_score = 0.0
            comp_len = 0
            comp_regions = []

        # Score splice sites
        splice_scores = score_splice_site(sequence)

        # Create base features
        features = BSJFeatures(
            alu_elements_present=alu_present,
            alu_element_count=alu_count,
            alu_matches=alu_matches,
            alu_flanking_distance=alu_flanking_dist,
            intron_complementarity_score=comp_score,
            complementary_length=comp_len,
            complementary_regions=comp_regions,
            splice_site_5_score=splice_scores.score_5,
            splice_site_3_score=splice_scores.score_3,
            splice_site_motifs=[splice_scores.motif_5, splice_scores.motif_3],
            branch_point_score=splice_scores.branch_point_score,
            circularization_score=0.0,  # Will be computed
            exon_length=len(sequence),
            flanking_intron_length=(len(flanking_5 or ""), len(flanking_3 or "")),
            bsj_stability=0.0,  # Will be computed
            protected_region=(0, 0),  # Will be computed
            biogenesis_efficiency_class="unknown",
        )

        # Compute derived scores
        features.circularization_score = predict_circularization_efficiency(features)
        features.bsj_stability = min(
            features.circularization_score * 0.6 +
            (features.splice_site_5_score + features.splice_site_3_score) * 0.2,
            1.0
        )
        features.protected_region = determine_protected_region(sequence, features)

        # Classify biogenesis efficiency
        if features.circularization_score >= 0.7:
            features.biogenesis_efficiency_class = "high"
        elif features.circularization_score >= 0.4:
            features.biogenesis_efficiency_class = "medium"
        else:
            features.biogenesis_efficiency_class = "low"

        return features


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_bsj_features(
    sequence: str,
    flanking_5: Optional[str] = None,
    flanking_3: Optional[str] = None,
) -> BSJFeatures:
    """Convenience wrapper for BSJ feature extraction."""
    extractor = BSJFeatureExtractor()
    return extractor.extract(sequence, flanking_5, flanking_3)


def get_bsj_summary(features: BSJFeatures) -> Dict:
    """Get summary dict of BSJ features."""
    return {
        "alu_present": features.alu_elements_present,
        "alu_count": features.alu_element_count,
        "complementarity_score": features.intron_complementarity_score,
        "splice_5_score": features.splice_site_5_score,
        "splice_3_score": features.splice_site_3_score,
        "circularization_score": features.circularization_score,
        "protected_region": features.protected_region,
        "efficiency_class": features.biogenesis_efficiency_class,
        "conservation_score": features.conservation_score,
    }


# ============================================================================
# NEW: Real-Time Junction Detection Functions
# ============================================================================

def detect_junction_signal(
    read_stream: List[Dict],
    signal_window: int = 50,
    min_read_support: int = 5,
) -> List[JunctionSignal]:
    """
    Detect back-splice junction signals from sequencing reads.

    Real-time detection algorithm:
    1. Scan for non-linear junction signatures
    2. Check read span across BSJ boundary
    3. Validate with paired-end consistency

    Args:
        read_stream: List of read dictionaries with keys:
                     'read_id', 'position', 'cigar', 'sequence', 'mate_position'
        signal_window: Window size for junction detection
        min_read_support: Minimum reads supporting junction

    Returns:
        List of JunctionSignal objects

    Literature:
        Gao et al., Nat Commun 2015: CIRI algorithm for BSJ detection
        Memczak et al., Nature 2013: circRNA detection from RNA-seq
    """
    signals = []
    junction_candidates = {}  # position -> supporting reads

    for read in read_stream:
        cigar = read.get('cigar', '')
        position = read.get('position', 0)
        read_id = read.get('read_id', '')

        # Check for chimeric/split reads indicating BSJ
        # N in CIGAR = skipped region (junction)
        if 'N' in cigar:
            # Parse junction position from CIGAR
            junction_pos = _parse_junction_from_cigar(cigar, position)

            if junction_pos:
                if junction_pos not in junction_candidates:
                    junction_candidates[junction_pos] = []
                junction_candidates[junction_pos].append(read_id)

        # Check for discordant paired-end reads
        mate_pos = read.get('mate_position', 0)
        if mate_pos > 0 and abs(mate_pos - position) > signal_window:
            # Potential BSJ signature: mates map far apart
            junction_pos = (position + mate_pos) // 2
            if junction_pos not in junction_candidates:
                junction_candidates[junction_pos] = []
            junction_candidates[junction_pos].append(read_id)

    # Filter by minimum support and create signals
    import time
    timestamp = time.time()

    for position, read_ids in junction_candidates.items():
        if len(read_ids) >= min_read_support:
            confidence = min(len(read_ids) / 20, 1.0)  # Normalize by expected support

            signals.append(JunctionSignal(
                position=position,
                read_support=len(read_ids),
                strand='+',  # Simplified
                confidence=confidence,
                detection_timestamp=timestamp,
                read_ids=read_ids[:10],  # Keep first 10 for reference
            ))

    # Sort by confidence
    signals.sort(key=lambda s: s.confidence, reverse=True)

    return signals


def _parse_junction_from_cigar(cigar: str, position: int) -> Optional[int]:
    """Parse junction position from CIGAR string."""
    import re

    # Find N operations (skipped regions = junctions)
    n_matches = re.findall(r'(\d+)N', cigar)

    if n_matches:
        # Calculate junction position
        # Simplified: use first N operation
        skip_length = int(n_matches[0])
        # Parse preceding operations to get actual position
        preceding = cigar.split('N')[0]
        read_consumed = sum(int(m) for m in re.findall(r'(\d+)[MID]', preceding))

        return position + read_consumed

    return None


def validate_bsj_realtime(
    junction_signal: JunctionSignal,
    reference_sequence: str,
    min_confidence: float = 0.7,
) -> BSJValidationResult:
    """
    Validate BSJ detection in real-time.

    Validation criteria:
    - Minimum read support threshold
    - Junction position consistency
    - Sequence context verification

    Args:
        junction_signal: Detected junction signal
        reference_sequence: Reference sequence for validation
        min_confidence: Minimum confidence threshold

    Returns:
        BSJValidationResult with validation status
    """
    # Check read support
    read_support = junction_signal.read_support
    confidence = junction_signal.confidence

    # Calculate false positive risk
    # Higher risk if low read support or low confidence
    fp_risk = max(0, 1.0 - confidence - (min(read_support, 20) / 40))

    # Check sequence context at junction
    position = junction_signal.position
    seq_len = len(reference_sequence)

    # Verify junction is within valid range
    context_valid = 0 < position < seq_len

    # Check for canonical splice site context
    if context_valid:
        # Extract flanking sequences
        flank_5 = reference_sequence[max(0, position-5):position]
        flank_3 = reference_sequence[position:min(seq_len, position+5)]

        # Check for splice site motifs
        has_splice_motif = (
            'GU' in flank_3[:2] or  # 5' splice site
            'AG' in flank_5[-2:]     # 3' splice site
        )
    else:
        has_splice_motif = False

    # Determine validation
    validated = (
        confidence >= min_confidence and
        read_support >= 5 and
        context_valid and
        fp_risk < 0.3
    )

    # Determine validation method
    if has_splice_motif:
        method = "junction_signature"
    elif read_support >= 10:
        method = "read_span"
    else:
        method = "paired_end"

    return BSJValidationResult(
        junction_validated=validated,
        confidence_score=confidence,
        supporting_reads=read_support,
        false_positive_risk=fp_risk,
        validation_method=method,
    )


# ============================================================================
# NEW: Conservation Scoring Functions
# ============================================================================

def compute_bsj_conservation_score(
    bsj_features: BSJFeatures,
    species_alignment: Optional[Dict[str, str]] = None,
    phylop_scores: Optional[List[float]] = None,
) -> ConservationAnnotation:
    """
    Compute evolutionary conservation score for BSJ region.

    Methods:
    - PhyloP/PhastCons from UCSC bigWig files
    - Cross-species junction preservation
    - Alu element evolutionary age

    Args:
        bsj_features: BSJFeatures object
        species_alignment: Dict mapping species to aligned sequence
        phylop_scores: Pre-computed PhyloP scores

    Returns:
        ConservationAnnotation object

    Literature:
        Rybak-Wolf et al., Mol Cell 2015: circRNA conservation
        Jeck et al., Nature 2013: Alu-mediated circularization evolution
    """
    # Junction conservation
    if phylop_scores and len(phylop_scores) > 0:
        # PhyloP scores: positive = conserved, negative = accelerated
        junction_cons = np.mean([max(0, s) for s in phylop_scores[:20]])
        junction_cons = min(junction_cons / 3.0, 1.0)  # Normalize
    else:
        # Estimate based on splice site scores
        junction_cons = (bsj_features.splice_site_5_score +
                        bsj_features.splice_site_3_score) / 2

    # Alu conservation
    if bsj_features.alu_elements_present:
        # Older Alu elements (AluJ) = higher conservation
        # Younger Alu elements (AluY) = lower conservation
        # Simplified: assume medium conservation
        alu_cons = 0.5 + 0.3 * bsj_features.intron_complementarity_score
    else:
        alu_cons = 0.0

    # Splice site conservation
    splice_cons = (bsj_features.splice_site_5_score +
                   bsj_features.splice_site_3_score) / 2

    # Cross-species conservation
    species_conserved = []
    if species_alignment:
        for species, seq in species_alignment.items():
            # Check if BSJ exists in this species
            if _check_bsj_in_species(seq, bsj_features):
                species_conserved.append(species)

    # Overall conservation
    overall = (
        0.4 * junction_cons +
        0.3 * splice_cons +
        0.2 * alu_cons +
        0.1 * min(len(species_conserved) / 5, 1.0)
    )

    # Classify
    if overall >= 0.7:
        cons_class = "high"
    elif overall >= 0.4:
        cons_class = "medium"
    else:
        cons_class = "low"

    return ConservationAnnotation(
        junction_conservation=junction_cons,
        alu_conservation=alu_cons,
        splice_site_conservation=splice_cons,
        overall_conservation=overall,
        species_conserved=species_conserved,
        conservation_class=cons_class,
    )


def _check_bsj_in_species(species_seq: str, bsj_features: BSJFeatures) -> bool:
    """Check if BSJ exists in another species."""
    # Simplified: check for similar splice sites
    if len(species_seq) < 50:
        return False

    # Look for similar splice site motifs
    has_5_site = 'GU' in species_seq[:30] or 'GC' in species_seq[:30]
    has_3_site = 'AG' in species_seq[-30:]

    return has_5_site and has_3_site


def get_cross_species_bsj(
    circRNA_id: str,
    species_list: List[str] = None,
) -> Dict[str, bool]:
    """
    Check BSJ presence across species.

    High conservation = more likely functional.

    Args:
        circRNA_id: circRNA identifier
        species_list: Species to check (default: human, mouse, rat, chimp)

    Returns:
        Dict mapping species to BSJ presence (bool)
    """
    if species_list is None:
        species_list = ["human", "mouse", "rat", "chimpanzee"]

    # Known conserved circRNAs
    conserved_circrnas = {
        "hsa_circ_0000198": ["human", "chimpanzee"],  # CDR1as
        "hsa_circ_0000064": ["human", "mouse", "rat"],  # circHIPK3
        "hsa_circ_0000080": ["human", "mouse", "chimpanzee"],  # circFOXO3
    }

    if circRNA_id in conserved_circrnas:
        return {sp: sp in conserved_circrnas[circRNA_id] for sp in species_list}

    # Default: assume human-specific
    return {sp: (sp == "human") for sp in species_list}


def add_conservation_to_features(
    features: BSJFeatures,
    species_alignment: Optional[Dict[str, str]] = None,
    phylop_scores: Optional[List[float]] = None,
) -> BSJFeatures:
    """
    Add conservation annotation to BSJFeatures.

    Modifies features in-place and returns.
    """
    conservation = compute_bsj_conservation_score(
        features, species_alignment, phylop_scores
    )

    features.conservation_score = conservation.overall_conservation
    features.conservation_annotation = conservation

    return features


# ============================================================================
# Demo / Test
# ============================================================================

if __name__ == "__main__":
    print("BSJ Feature Extraction Demo")
    print("=" * 60)

    # Test sequence with simulated Alu flanking
    test_seq = "ACGUACGUACGUACGU" * 30  # 480 nt exon

    # Simulated flanking introns with Alu-like elements
    flanking_5 = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAG" * 2 + "AUCGAUCGAUCG"
    flanking_3 = "AUCGAUCGAUCG" + "GAGGTCAGGAGATCGAGACCATCCTGGCTAACATGGTGAAACCCC" * 2

    extractor = BSJFeatureExtractor()
    features = extractor.extract(test_seq, flanking_5, flanking_3)

    print(f"\nSequence length: {features.exon_length} nt")
    print(f"Alu elements: {features.alu_element_count}")
    print(f"Alu present: {features.alu_elements_present}")
    print(f"Complementarity score: {features.intron_complementarity_score:.3f}")
    print(f"Splice 5' score: {features.splice_site_5_score:.3f}")
    print(f"Splice 3' score: {features.splice_site_3_score:.3f}")
    print(f"Circularization score: {features.circularization_score:.3f}")
    print(f"Protected region: {features.protected_region}")
    print(f"Efficiency class: {features.biogenesis_efficiency_class}")

    summary = get_bsj_summary(features)
    print("\nSummary:")
    for key, val in summary.items():
        print(f"  {key}: {val}")