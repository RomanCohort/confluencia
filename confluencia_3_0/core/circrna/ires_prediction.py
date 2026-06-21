"""
IRES 3D Pocket Scoring Module

Evaluates IRES (Internal Ribosome Entry Site) elements from a 3D structural perspective.
When TorusFold 3D coordinates are available, determines whether IRES motifs are physically
accessible to ribosomes based on solvent-accessible surface area (SASA) calculations.

Author: Confluencia Project
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# IRES Motif Catalog
# =============================================================================

IRES_MOTIFS: Dict[str, List[str]] = {
    "EMCV": ["GCGCC", "GGGG", "CCUG", "GGAAGG"],
    "HCV": ["UUGU", "AUGG", "GGUG"],
    "generic": ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"],
}


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class IRESPredictionResult:
    """
    Result of IRES prediction analysis.

    Attributes:
        ires_type: Type of IRES motif set used (EMCV, HCV, generic)
        motif_positions: Dictionary mapping each motif to list of start positions
        motif_accessibility: Dictionary mapping each motif to SASA-based accessibility [0,1]
        overall_ires_accessibility: Average accessibility across all found motifs [0,1]
        pocket_depth: Average distance from motif centers to surface [0,1] normalized
        method: "3d_structure" or "sequence_heuristic"
    """
    ires_type: str
    motif_positions: Dict[str, List[int]] = field(default_factory=dict)
    motif_accessibility: Dict[str, float] = field(default_factory=dict)
    overall_ires_accessibility: float = 0.0
    pocket_depth: float = 0.0
    method: str = "sequence_heuristic"


# =============================================================================
# Core Functions
# =============================================================================

def find_ires_motifs(sequence: str, ires_type: str = "generic") -> Dict[str, List[int]]:
    """
    Scan a nucleotide sequence for IRES motif patterns.

    Args:
        sequence: RNA nucleotide sequence (uppercase)
        ires_type: Key into IRES_MOTIFS catalog ("EMCV", "HCV", "generic")

    Returns:
        Dictionary mapping each motif to list of start positions where found.
        Empty dict if ires_type not recognized or sequence empty.
    """
    if not sequence or ires_type not in IRES_MOTIFS:
        return {}

    sequence = sequence.upper().replace("T", "U")  # Normalize to RNA
    motifs = IRES_MOTIFS[ires_type]
    result: Dict[str, List[int]] = {}

    for motif in motifs:
        positions = []
        motif_upper = motif.upper()
        start = 0
        while True:
            pos = sequence.find(motif_upper, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        if positions:
            result[motif] = positions

    return result


def _compute_motif_sasa(
    coords: np.ndarray,
    motif_positions: Dict[str, List[int]],
    sasa: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute average SASA for each motif based on nucleotide positions.

    Args:
        coords: (N, 3) array of nucleotide coordinates
        motif_positions: Motif -> list of start positions
        sasa: Optional pre-computed SASA array of length N

    Returns:
        Dictionary mapping motif -> average SASA value
    """
    if sasa is None:
        # Simple distance-based SASA approximation if not provided
        sasa = _estimate_sasa_from_coords(coords)

    motif_sasa: Dict[str, float] = {}

    for motif, positions in motif_positions.items():
        motif_len = len(motif)
        sasa_values = []
        for start_pos in positions:
            # Collect SASA for all nucleotides in this motif instance
            end_pos = min(start_pos + motif_len, len(sasa))
            if start_pos < len(sasa):
                sasa_values.extend(sasa[start_pos:end_pos])

        if sasa_values:
            motif_sasa[motif] = float(np.mean(sasa_values))

    return motif_sasa


def _estimate_sasa_from_coords(coords: np.ndarray) -> np.ndarray:
    """
    Estimate SASA from coordinates using a simple distance-based approach.

    Nucleotides with many nearby neighbors are considered buried.

    Args:
        coords: (N, 3) array of nucleotide coordinates

    Returns:
        Array of estimated SASA values [0, 1] for each nucleotide.
    """
    n = len(coords)
    if n == 0:
        return np.array([])

    # Count neighbors within cutoff distance for each nucleotide
    cutoff = 15.0  # Angstroms, typical RNA interaction distance

    sasa = np.ones(n)
    for i in range(n):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        # Count neighbors (excluding self) within cutoff
        neighbors = np.sum((distances < cutoff) & (distances > 0.1))
        # Normalize: more neighbors = lower SASA
        sasa[i] = 1.0 / (1.0 + 0.1 * neighbors)

    return sasa


def _compute_pocket_depth(
    coords: np.ndarray,
    motif_positions: Dict[str, List[int]],
    sasa: Optional[np.ndarray] = None
) -> float:
    """
    Compute average pocket depth for IRES motifs.

    Pocket depth is the average distance from motif centers to the nearest
    "surface" nucleotide (one with high SASA).

    Args:
        coords: (N, 3) array of nucleotide coordinates
        motif_positions: Motif -> list of start positions
        sasa: Optional pre-computed SASA array

    Returns:
        Normalized pocket depth [0, 1]
    """
    if sasa is None:
        sasa = _estimate_sasa_from_coords(coords)

    # Identify surface nucleotides (high SASA)
    surface_threshold = 0.5
    surface_indices = np.where(sasa >= surface_threshold)[0]

    if len(surface_indices) == 0:
        return 0.0  # No surface defined

    surface_coords = coords[surface_indices]

    depths = []
    for motif, positions in motif_positions.items():
        motif_len = len(motif)
        for start_pos in positions:
            end_pos = min(start_pos + motif_len, len(coords))
            if start_pos >= len(coords):
                continue

            # Compute motif center
            motif_coords = coords[start_pos:end_pos]
            if len(motif_coords) == 0:
                continue
            motif_center = np.mean(motif_coords, axis=0)

            # Distance to nearest surface nucleotide
            distances = np.linalg.norm(surface_coords - motif_center, axis=1)
            min_dist = np.min(distances)
            depths.append(min_dist)

    if not depths:
        return 0.0

    # Normalize: typical RNA radius ~20-50 A
    avg_depth = float(np.mean(depths))
    normalized_depth = min(avg_depth / 30.0, 1.0)  # 30 A as reference

    return normalized_depth


def compute_ires_3d_score(
    sequence: str,
    coords: np.ndarray,
    ires_type: str = "generic",
    sasa: Optional[np.ndarray] = None
) -> IRESPredictionResult:
    """
    Compute IRES accessibility score using 3D structural information.

    Args:
        sequence: RNA nucleotide sequence
        coords: (N, 3) array of 3D coordinates from TorusFold
        ires_type: Key into IRES_MOTIFS catalog
        sasa: Optional pre-computed SASA array (length N, values [0, 1])

    Returns:
        IRESPredictionResult with 3D-based accessibility scores
    """
    # Find motif positions
    motif_positions = find_ires_motifs(sequence, ires_type)

    if not motif_positions or len(coords) == 0:
        return IRESPredictionResult(
            ires_type=ires_type,
            method="3d_structure"
        )

    # Compute SASA for each motif
    motif_accessibility = _compute_motif_sasa(coords, motif_positions, sasa)

    # Overall accessibility
    if motif_accessibility:
        overall_accessibility = float(np.mean(list(motif_accessibility.values())))
    else:
        overall_accessibility = 0.0

    # Pocket depth
    pocket_depth = _compute_pocket_depth(coords, motif_positions, sasa)

    return IRESPredictionResult(
        ires_type=ires_type,
        motif_positions=motif_positions,
        motif_accessibility=motif_accessibility,
        overall_ires_accessibility=overall_accessibility,
        pocket_depth=pocket_depth,
        method="3d_structure"
    )


def compute_ires_heuristic_score(
    sequence: str,
    ires_type: str = "generic",
    bsj_position: Optional[int] = None
) -> IRESPredictionResult:
    """
    Compute IRES accessibility score using sequence-based heuristics.

    Estimates accessibility from:
    - GC content (high GC = more likely structured/buried)
    - Position relative to back-splice junction (motifs near BSJ more buried)

    Args:
        sequence: RNA nucleotide sequence
        ires_type: Key into IRES_MOTIFS catalog
        bsj_position: Optional position of back-splice junction

    Returns:
        IRESPredictionResult with heuristic accessibility scores
    """
    # Find motif positions
    motif_positions = find_ires_motifs(sequence, ires_type)

    if not motif_positions:
        return IRESPredictionResult(
            ires_type=ires_type,
            method="sequence_heuristic"
        )

    sequence = sequence.upper().replace("T", "U")
    seq_len = len(sequence)

    # Compute GC content around each motif
    motif_accessibility: Dict[str, float] = {}
    window = 10  # Nucleotides on each side for GC content

    for motif, positions in motif_positions.items():
        motif_len = len(motif)
        access_values = []

        for start_pos in positions:
            # Local GC content
            local_start = max(0, start_pos - window)
            local_end = min(seq_len, start_pos + motif_len + window)
            local_seq = sequence[local_start:local_end]

            gc_count = local_seq.count("G") + local_seq.count("C")
            gc_content = gc_count / len(local_seq) if local_seq else 0.5

            # High GC = more structured = less accessible
            gc_factor = 1.0 - gc_content

            # Position factor: motifs near BSJ more likely buried
            if bsj_position is not None:
                dist_to_bsj = abs(start_pos - bsj_position)
                # Normalize by sequence length
                pos_factor = min(dist_to_bsj / (seq_len * 0.5), 1.0)
            else:
                pos_factor = 0.5  # Neutral if no BSJ info

            # Combine factors
            accessibility = gc_factor * 0.6 + pos_factor * 0.4
            access_values.append(accessibility)

        motif_accessibility[motif] = float(np.mean(access_values))

    # Overall accessibility
    overall_accessibility = float(np.mean(list(motif_accessibility.values())))

    # Estimate pocket depth from GC content (heuristic)
    global_gc = (sequence.count("G") + sequence.count("C")) / seq_len if seq_len > 0 else 0.5
    pocket_depth = global_gc  # High GC = deeper pockets (heuristic)

    return IRESPredictionResult(
        ires_type=ires_type,
        motif_positions=motif_positions,
        motif_accessibility=motif_accessibility,
        overall_ires_accessibility=overall_accessibility,
        pocket_depth=pocket_depth,
        method="sequence_heuristic"
    )


def quick_ires_score(
    sequence: str,
    coords: Optional[np.ndarray] = None,
    ires_type: str = "generic",
    sasa: Optional[np.ndarray] = None
) -> float:
    """
    Convenience function returning overall IRES accessibility [0, 1].

    Uses 3D structure when coordinates are available, otherwise falls back
    to sequence-based heuristics.

    Args:
        sequence: RNA nucleotide sequence
        coords: Optional (N, 3) array of 3D coordinates
        ires_type: Key into IRES_MOTIFS catalog
        sasa: Optional pre-computed SASA array

    Returns:
        Overall IRES accessibility score in [0, 1]
    """
    if coords is not None and len(coords) > 0:
        result = compute_ires_3d_score(sequence, coords, ires_type, sasa)
    else:
        result = compute_ires_heuristic_score(sequence, ires_type)

    return result.overall_ires_accessibility


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "IRES_MOTIFS",
    "IRESResult",
    "IRESPredictionResult",
    "find_ires_motifs",
    "compute_ires_3d_score",
    "compute_ires_heuristic_score",
    "quick_ires_score",
]

# Alias for convenience
IRESResult = IRESPredictionResult
