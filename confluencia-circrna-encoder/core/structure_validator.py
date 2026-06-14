"""
structure_validator.py — Validate circRNA 3D structures against physical constraints.

Checks:
1. Closure: ||x[0] - x[-1]|| ≈ bond_length
2. Bond lengths: all adjacent distances ≈ target
3. Clash: no non-bonded pairs closer than clash_distance
4. Pair distances: predicted pairs at expected distances
5. Energy: simple CG energy estimate

Produces ValidationMetrics that feed into TorusFoldScorer's 4D objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ValidationMetrics:
    """Metrics from structure validation.

    Attributes:
        closure_distance: ||x[0] - x[-1]|| in Å
        closure_score: Normalized [0,1], 1 = perfect closure
        bond_rmsd: RMS deviation from ideal bond length
        clash_count: Number of steric clashes
        pair_satisfaction: Fraction of predicted pairs at correct distance
        energy_score: Simple CG energy (kcal/mol, lower = better)
        stability_score: Combined stability [0,1], 1 = most stable
    """
    closure_distance: float = 0.0
    closure_score: float = 0.0
    bond_rmsd: float = 0.0
    clash_count: int = 0
    pair_satisfaction: float = 0.0
    energy_score: float = 0.0
    stability_score: float = 0.0


class StructureValidator:
    """Validate circRNA 3D structure against physical constraints.

    Args:
        bond_length: Expected backbone bond length (Å)
        pair_tolerance: Å tolerance for pair distance satisfaction
        clash_distance: Minimum non-bonded distance (Å)
    """

    def __init__(
        self,
        bond_length: float = 5.9,
        pair_tolerance: float = 2.0,
        clash_distance: float = 3.0,
    ):
        self.bond_length = bond_length
        self.pair_tolerance = pair_tolerance
        self.clash_distance = clash_distance

    def validate(
        self,
        coords: np.ndarray,
        constraint_set,
    ) -> ValidationMetrics:
        """Validate a single conformation.

        Args:
            coords: (L, 3) numpy array of nucleotide positions
            constraint_set: ConstraintSet with expected constraints

        Returns:
            ValidationMetrics with all quality metrics
        """
        L = len(coords)
        metrics = ValidationMetrics()

        # 1. Closure distance
        metrics.closure_distance = float(np.linalg.norm(coords[0] - coords[-1]))
        # Closure score: 1.0 at perfect, 0.0 at 10Å deviation
        closure_dev = abs(metrics.closure_distance - self.bond_length)
        metrics.closure_score = max(0.0, 1.0 - closure_dev / 10.0)

        # 2. Bond RMSD
        bond_deviations = []
        for i in range(L):
            j = (i + 1) % L
            d = np.linalg.norm(coords[j] - coords[i])
            bond_deviations.append(d - self.bond_length)
        metrics.bond_rmsd = float(np.sqrt(np.mean(np.array(bond_deviations) ** 2)))

        # 3. Clash count
        metrics.clash_count = 0
        for i in range(L):
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue  # closure bond
                d = np.linalg.norm(coords[j] - coords[i])
                if d < self.clash_distance:
                    metrics.clash_count += 1

        # 4. Pair satisfaction
        if constraint_set.pair_constraints:
            satisfied = 0
            for (i, j, target_d, weight) in constraint_set.pair_constraints:
                d = np.linalg.norm(coords[j] - coords[i])
                if abs(d - target_d) < self.pair_tolerance:
                    satisfied += 1
            metrics.pair_satisfaction = satisfied / len(constraint_set.pair_constraints)

        # 5. Simple CG energy
        energy = 0.0
        # Bond energy
        for i in range(L):
            j = (i + 1) % L
            d = np.linalg.norm(coords[j] - coords[i])
            energy += (d - self.bond_length) ** 2
        # Pair energy
        for (i, j, target_d, weight) in constraint_set.pair_constraints:
            d = np.linalg.norm(coords[j] - coords[i])
            energy += 0.5 * weight * (d - target_d) ** 2
        # Clash energy
        for i in range(L):
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue
                d = np.linalg.norm(coords[j] - coords[i])
                if d < self.clash_distance:
                    energy += 10.0 * (self.clash_distance - d) ** 2
        metrics.energy_score = float(energy)

        # 6. Combined stability score
        # Higher = more stable. Based on closure, bonds, clashes, pairs.
        bond_score = max(0.0, 1.0 - metrics.bond_rmsd / 5.0)
        clash_score = max(0.0, 1.0 - metrics.clash_count / max(1, L))
        metrics.stability_score = (
            0.3 * metrics.closure_score +
            0.3 * bond_score +
            0.2 * clash_score +
            0.2 * metrics.pair_satisfaction
        )

        return metrics

    def validate_best(
        self,
        conformations: list,
        constraint_set,
    ) -> tuple:
        """Validate multiple conformations and return the best.

        Args:
            conformations: List of (L, 3) numpy arrays
            constraint_set: ConstraintSet

        Returns:
            (best_coords, best_metrics) tuple
        """
        if not conformations:
            L = constraint_set.seq_len
            empty = np.zeros((L, 3))
            return empty, ValidationMetrics()

        best_coords = None
        best_metrics = None
        best_energy = float('inf')

        for coords in conformations:
            metrics = self.validate(coords, constraint_set)
            if metrics.energy_score < best_energy:
                best_energy = metrics.energy_score
                best_coords = coords
                best_metrics = metrics

        return best_coords, best_metrics
