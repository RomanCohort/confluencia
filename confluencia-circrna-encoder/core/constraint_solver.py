"""
constraint_solver.py — Geometric Constraint Solver for circRNA.

Implements Plan B: zero-training 3D structure prediction from geometric constraints.
Uses a combination of:
1. Regular polygon initialization (closure guaranteed by construction)
2. Iterative perturbation to satisfy pair constraints
3. Shapiro-Barnes algorithm for closure correction
4. Simple coarse-grained energy ranking

No experimental data needed. All physics comes from:
- Backbone bond geometry (5.9 Å P-P distance)
- Watson-Crick pair geometry (10.6 Å C1'-C1' distance)
- A-form RNA dihedral preferences
- Steric exclusion (3.0 Å minimum non-bonded distance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SolverConfig:
    """Configuration for the constraint solver."""
    bond_length: float = 5.9        # Å, P-P backbone distance
    pair_distance: float = 10.6     # Å, WC C1'-C1' distance
    clash_distance: float = 3.0     # Å, minimum non-bonded distance
    n_samples: int = 20             # Number of conformations to sample
    closure_tolerance: float = 0.5  # Å, max allowed closure deviation
    max_iterations: int = 100       # Max perturbation iterations
    perturbation_scale: float = 0.5 # Å, initial perturbation amplitude


class GeometricConstraintSolver:
    """Solve circRNA 3D structure from geometric constraints.

    Core algorithm:
    1. Initialize nucleotides on regular polygon (radius = L * bond_length / 2π)
       → closure is guaranteed: x[L-1] connects to x[0] exactly
    2. Iteratively perturb positions to satisfy pair constraints
       → for each pair (i, j, d), move j toward/away from i to achieve d
    3. After perturbation, apply Shapiro-Barnes closure correction
       → redistribute closure error evenly across all bonds
    4. Filter out conformations with steric clashes
    5. Rank remaining by simple CG energy

    Args:
        config: SolverConfig with geometric parameters
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    def solve(self, constraint_set) -> List[np.ndarray]:
        """Sample conformations satisfying all constraints.

        Args:
            constraint_set: ConstraintSet from ConstraintExtractor

        Returns:
            List of (L, 3) numpy arrays, each a valid conformation
        """
        L = constraint_set.seq_len
        if L < 3:
            return [self._single_point(L)]

        conformations = []

        for sample_idx in range(self.config.n_samples):
            # 1. Initialize on regular polygon (closure guaranteed)
            coords = self._regular_polygon(L, self.config.bond_length)

            # 2. Perturb to satisfy pair constraints
            coords = self._satisfy_pair_constraints(coords, constraint_set)

            # 3. Correct closure if needed
            coords = self._closure_correction(coords)

            # 4. Check for steric clashes
            if self._has_clashes(coords):
                continue

            # 5. Additional local relaxation (optional)
            coords = self._local_relax(coords, constraint_set)

            conformations.append(coords)

        # Rank by simple CG energy
        if len(conformations) > 1:
            energies = [self._compute_cg_energy(c, constraint_set) for c in conformations]
            sorted_idx = np.argsort(energies)
            conformations = [conformations[i] for i in sorted_idx]

        return conformations

    def _regular_polygon(self, L: int, bond_length: float) -> np.ndarray:
        """Place nucleotides on a regular polygon (ring).

        Radius R = L * bond_length / (2π) ensures circumference = L * bond_length.
        This guarantees closure: the last nucleotide connects to the first.

        Returns:
            (L, 3) coordinates with positions on the ring
        """
        R = L * bond_length / (2 * math.pi)
        coords = np.zeros((L, 3), dtype=np.float64)

        for i in range(L):
            angle = 2 * math.pi * i / L
            coords[i, 0] = R * math.cos(angle)
            coords[i, 1] = R * math.sin(angle)
            coords[i, 2] = 0.0  # Start flat (z=0)

        return coords

    def _satisfy_pair_constraints(
        self,
        coords: np.ndarray,
        constraint_set,
        max_iter: Optional[int] = None,
    ) -> np.ndarray:
        """Iteratively perturb positions to satisfy pair constraints.

        For each pair (i, j, target_d, weight):
        - Compute current distance d_curr
        - If d_curr differs from target_d, move j toward/away from i
        - Weight determines how much to move (higher = more movement)

        This is a local optimization, not global. Each pair is treated
        independently, which can lead to conflicts. The closure correction
        and energy ranking handle the global consistency.
        """
        max_iter = max_iter or self.config.max_iterations
        coords = coords.copy()

        pair_constraints = constraint_set.pair_constraints
        if not pair_constraints:
            return coords

        for iteration in range(max_iter):
            max_error = 0.0

            for (i, j, target_d, weight) in pair_constraints:
                # Current distance
                d_curr = np.linalg.norm(coords[j] - coords[i])

                # Error
                error = abs(d_curr - target_d)
                max_error = max(max_error, error)

                if error < 0.1:  # Within tolerance
                    continue

                # Direction vector from i to j
                direction = coords[j] - coords[i]
                if d_curr < 0.01:  # Avoid division by zero
                    direction = np.random.randn(3)
                    d_curr = np.linalg.norm(direction)

                direction = direction / d_curr  # normalize

                # Movement amount: proportional to error and weight
                move_amount = (target_d - d_curr) * weight * 0.3
                move_amount = np.clip(move_amount, -2.0, 2.0)  # Limit max movement

                # Move j toward/away from i
                coords[j] += move_amount * direction

            if max_error < 0.5:  # All constraints satisfied
                break

        return coords

    def _closure_correction(self, coords: np.ndarray) -> np.ndarray:
        """Shapiro-Barnes algorithm for ring closure.

        After perturbation, the closure constraint ||x[0] - x[-1]|| ≈ bond_length
        may be violated. This algorithm redistributes the closure error
        evenly across all nucleotides by adjusting bond vectors.

        Algorithm:
        1. Compute closure error: e = x[-1] - x[0] (should be ~bond_length vector)
        2. Compute target closure vector: v = x[0] + bond_length * unit(e)
        3. Distribute error: each bond vector gets a correction = -e/L
        4. Apply corrections to all positions

        This preserves the relative geometry while fixing closure.
        """
        coords = coords.copy()
        L = len(coords)
        bond_length = self.config.bond_length

        # Current closure vector (should connect x[-1] back to x[0])
        closure_vec = coords[0] - coords[-1]
        closure_dist = np.linalg.norm(closure_vec)

        # Check if closure is already satisfied
        if abs(closure_dist - bond_length) < self.config.closure_tolerance:
            return coords

        # Target: we want ||x[0] - x[-1]|| = bond_length
        # Current: ||closure_vec|| = closure_dist

        # Distribute error across all bonds
        error_vec = closure_vec - np.array([bond_length, 0, 0])  # reference direction
        correction = -error_vec / L

        # Apply correction cumulatively
        for i in range(L):
            coords[i] += i * correction

        # Final closure check: explicitly set last position to connect to first
        # Place x[-1] at distance bond_length from x[0]
        final_vec = coords[0] - coords[-2]
        final_dist = np.linalg.norm(final_vec)
        if final_dist > 0.01:
            final_dir = final_vec / final_dist
            coords[-1] = coords[0] - bond_length * final_dir

        return coords

    def _has_clashes(self, coords: np.ndarray) -> bool:
        """Check for steric clashes (non-bonded atoms too close).

        Clash: any pair (i, j) where |i-j| > 1 and distance < clash_distance.
        (Adjacent pairs are bonded, so allowed to be close.)

        Returns:
            True if any clash found
        """
        L = len(coords)
        clash_dist = self.config.clash_distance

        for i in range(L):
            for j in range(i + 2, L):  # Skip adjacent (|i-j|=1)
                # Also skip closure pair (i=0, j=L-1)
                if i == 0 and j == L - 1:
                    continue

                d = np.linalg.norm(coords[j] - coords[i])
                if d < clash_dist:
                    return True

        return False

    def _local_relax(self, coords: np.ndarray, constraint_set) -> np.ndarray:
        """Local relaxation to reduce energy.

        Simple iterative adjustment:
        - For each position, move toward the mean of its neighbors
        - This reduces bond strain while preserving overall shape
        """
        coords = coords.copy()
        L = len(coords)

        for iteration in range(20):
            for i in range(L):
                # Neighbors: (i-1) and (i+1), circularly
                prev = (i - 1) % L
                next = (i + 1) % L

                # Target: midpoint of neighbors + bond_length direction
                midpoint = (coords[prev] + coords[next]) / 2

                # Move slightly toward midpoint (relaxation)
                coords[i] = 0.9 * coords[i] + 0.1 * midpoint

        return coords

    def _compute_cg_energy(self, coords: np.ndarray, constraint_set) -> float:
        """Compute simple coarse-grained energy.

        Energy components:
        1. Bond energy: k_bond * Σ(d - bond_length)²
        2. Pair energy: k_pair * Σ(d - target)² for satisfied pairs
        3. Clash energy: k_clash * Σmax(0, clash_dist - d)²

        This is a "rough physics" energy for ranking, not a real force field.
        """
        L = len(coords)
        bond_length = self.config.bond_length
        clash_dist = self.config.clash_distance

        # Force constants (arbitrary units, just for ranking)
        k_bond = 1.0
        k_pair = 0.5
        k_clash = 10.0

        energy = 0.0

        # Bond energy
        for i in range(L):
            j = (i + 1) % L
            d = np.linalg.norm(coords[j] - coords[i])
            energy += k_bond * (d - bond_length) ** 2

        # Pair energy
        for (i, j, target_d, weight) in constraint_set.pair_constraints:
            d = np.linalg.norm(coords[j] - coords[i])
            energy += k_pair * weight * (d - target_d) ** 2

        # Clash energy
        for i in range(L):
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue
                d = np.linalg.norm(coords[j] - coords[i])
                if d < clash_dist:
                    energy += k_clash * (clash_dist - d) ** 2

        return energy

    def _single_point(self, L: int) -> np.ndarray:
        """Degenerate case: place all nucleotides at same point."""
        return np.zeros((L, 3), dtype=np.float64)


def circular_distance_matrix_np(L: int) -> np.ndarray:
    """Compute circular distance matrix (NumPy version).

    d_circ(i, j) = min(|i - j|, L - |i - j|)

    Args:
        L: Sequence length

    Returns:
        (L, L) numpy array with circular distances
    """
    positions = np.arange(L)
    diff = np.abs(positions[:, None] - positions[None, :])
    return np.minimum(diff, L - diff)


def is_bsj_crossing_np(i: int, j: int, L: int) -> bool:
    """Check if pair (i, j) crosses the BSJ.

    BSJ-crossing: the shorter arc between i and j contains the BSJ.
    Equivalent to: |i - j| > L/2 in linear distance.
    """
    return abs(i - j) > L / 2