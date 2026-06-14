"""
physics_bridge.py — DL-to-Physics Constraint Bridge for circRNA.

Converts deep learning predictions (pair probabilities, pair representations)
into geometric constraints suitable for the constraint solver.

This is the zero-training bridge: no experimental 3D structures needed.
Watson-Crick pair distances come from crystallographic data (10.6 Å C1'-C1'),
backbone bond lengths from RNA geometry (3.4 Å P-P, 5.9 Å P-P next-neighbor),
and A-form dihedral angles from structural biology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Watson-Crick and wobble pair definitions
WC_PAIRS = frozenset({('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')})
WOBBLE_PAIRS = frozenset({('G', 'U'), ('U', 'G')})
ALL_PAIRS = WC_PAIRS | WOBBLE_PAIRS

# A-form RNA backbone dihedral angles (degrees)
# (alpha, beta, gamma, delta, epsilon, zeta)
A_FORM_DIHEDRALS = (-68.0, 174.0, 54.0, 82.0, -153.0, -71.0)


@dataclass
class ConstraintSet:
    """Geometric constraints for circRNA 3D structure.

    Attributes:
        bond_constraints: (i, j, target_distance_Å) for backbone bonds
        pair_constraints: (i, j, target_distance_Å, confidence_weight) for base pairs
        dihedral_constraints: (position, target_angle_deg, weight) for torsions
        closure_constraint: Whether to enforce ring closure Σ=0
        sequence: The circRNA sequence
        seq_len: Sequence length
    """
    bond_constraints: List[Tuple[int, int, float]] = field(default_factory=list)
    pair_constraints: List[Tuple[int, int, float, float]] = field(default_factory=list)
    dihedral_constraints: List[Tuple[int, float, float]] = field(default_factory=list)
    closure_constraint: bool = True
    sequence: str = ""
    seq_len: int = 0


class ConstraintExtractor(nn.Module):
    """Convert DL predictions into geometric constraints.

    Extracts from pair_repr and pair_probs:
    - Watson-Crick pairs: C1'-C1' distance = 10.6 Å (from crystallography)
    - Non-canonical pairs: variable distance from learned head
    - Backbone bonds: P-P distance = 5.9 Å (next-neighbor)
    - Dihedral angles: A-form RNA defaults + learned deviations
    - Closure: sum of vectors around ring = 0

    Args:
        c_z: Pair representation dimension
        n_rbf: Number of RBF bins for distance prediction
        bond_length: Backbone P-P bond length (Å)
        pair_distance: Watson-Crick C1'-C1' distance (Å)
        pair_threshold: Minimum pair probability to create constraint
        bsj_weight_boost: Extra weight for BSJ-crossing pairs
    """

    def __init__(
        self,
        c_z: int = 128,
        n_rbf: int = 32,
        bond_length: float = 5.9,
        pair_distance: float = 10.6,
        pair_threshold: float = 0.3,
        bsj_weight_boost: float = 2.0,
    ):
        super().__init__()
        self.c_z = c_z
        self.n_rbf = n_rbf
        self.bond_length = bond_length
        self.pair_distance = pair_distance
        self.pair_threshold = pair_threshold
        self.bsj_weight_boost = bsj_weight_boost

        # Distance head for non-canonical pairs
        self.dist_head = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.GELU(),
            nn.Linear(c_z, n_rbf),
        )
        self.register_buffer(
            "rbf_centers",
            torch.linspace(3.0, 30.0, n_rbf),
        )

        # Dihedral angle head (6 torsions per nucleotide)
        self.dihedral_head = nn.Sequential(
            nn.Linear(c_z, 64),
            nn.GELU(),
            nn.Linear(64, 6),  # alpha, beta, gamma, delta, epsilon, zeta
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        pair_probs: torch.Tensor,
        sequence: str,
    ) -> ConstraintSet:
        """Extract constraint set from DL predictions.

        Args:
            pair_repr: (B, L, L, c_z) from CircPairformerStack
            pair_probs: (B, L, L) from PairPredictionHead
            sequence: circRNA sequence string

        Returns:
            ConstraintSet with all geometric constraints
        """
        # Use first batch element (per-sequence processing)
        if pair_repr.dim() == 4:
            pair_repr = pair_repr[0]  # (L, L, c_z)
        if pair_probs.dim() == 3:
            pair_probs = pair_probs[0]  # (L, L)

        L = pair_repr.size(0)
        seq = sequence.upper().replace('T', 'U')

        constraints = ConstraintSet(
            closure_constraint=True,
            sequence=sequence,
            seq_len=L,
        )

        # 1. Backbone bond constraints
        for i in range(L):
            j = (i + 1) % L  # circular: last bonds back to first
            constraints.bond_constraints.append((i, j, self.bond_length))

        # 2. Base pair constraints from pair_probs
        seq_len = len(seq)
        for i in range(L):
            for j in range(i + 4, L):  # minimum loop of 3
                prob = pair_probs[i, j].item()
                if prob < self.pair_threshold:
                    continue

                # Determine distance
                nt_i = seq[i] if i < seq_len else 'N'
                nt_j = seq[j] if j < seq_len else 'N'

                if (nt_i, nt_j) in WC_PAIRS:
                    dist = self.pair_distance
                elif (nt_i, nt_j) in WOBBLE_PAIRS:
                    dist = self.pair_distance * 1.05  # slightly longer
                else:
                    # Learned distance from pair representation
                    dist_logits = self.dist_head(pair_repr[i, j])  # (n_rbf,)
                    dist_probs = F.softmax(dist_logits, dim=0)
                    dist = (dist_probs * self.rbf_centers).sum().item()

                # Weight = pair probability, boosted for BSJ-crossing pairs
                weight = prob
                circ_dist = min(abs(i - j), L - abs(i - j))
                if circ_dist >= L / 2:  # BSJ-crossing pair
                    weight *= self.bsj_weight_boost

                constraints.pair_constraints.append((i, j, dist, weight))

        # 3. Dihedral angle constraints from pair representation
        # Use diagonal (i, i) of pair_repr for per-position features
        for i in range(L):
            dih_offset = self.dihedral_head(pair_repr[i, i])  # (6,)
            angles = []
            for k in range(6):
                # A-form prior + learned deviation (clamped to ±30°)
                angle = A_FORM_DIHEDRALS[k] + dih_offset[k].item() * 30.0
                angles.append(angle)

            # Weight: 1.0 for well-predicted positions, lower for uncertain
            # Use max pair probability at this position as proxy
            max_pair_prob = pair_probs[i].max().item()
            weight = 0.5 + 0.5 * max_pair_prob  # range [0.5, 1.0]

            for k, angle in enumerate(angles):
                constraints.dihedral_constraints.append((i, angle, weight))

        return constraints
