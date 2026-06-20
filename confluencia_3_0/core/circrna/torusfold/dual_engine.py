"""
dual_engine.py — Dual-engine iterative evolution for circRNA 3D structure.

Scheme 3: Generator (G) + Selector (S) iterative refinement.

Two implementation paths:
  Path 1 (Macro): Genetic algorithm on structure level
    G generates → S scores → top-K selection → crossover/mutation → repeat

  Path 2 (Micro): Physics-as-loss training (CS-Fold inspired)
    BSJ closure as hard penalty in training → G learns to generate closed structures
    S provides gradient feedback → G's parameters are updated

The micro path is inspired by CS-Fold (2025 bioRxiv): evolutionary/physics
constraints are enforced as gradient penalties during training, not just
post-hoc scoring. For circRNA, the BSJ closure constraint ("first and last
nucleotide must connect") becomes a hard training penalty.

PaxNet-inspired: S engine uses multiplex graph scoring with:
  - Local layer: bond + angle interactions
  - Non-local layer: van der Waals + electrostatic
  - BSJ edge: explicit 5'-3' connection edge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DualEngineConfig:
    """Configuration for dual-engine architecture."""
    d_model: int = 256
    n_harmonics: int = 16
    n_blocks: int = 4
    n_heads: int = 8
    n_iterations: int = 3          # G↔S iteration rounds
    n_candidates: int = 20         # Candidates per G round
    feedback_top_k: int = 5        # Top-K residues to feedback
    feedback_mode: str = "strain"  # "strain", "constraint", "combined"
    bond_length: float = 5.9       # Å
    pair_distance: float = 10.6    # Å
    clash_distance: float = 3.0    # Å


class PhysicsSelector:
    """S engine: scores candidates and generates feedback.

    Scoring uses extended CG energy (bond + pair + clash + stacking + electrostatic).
    Feedback identifies residues with highest strain for G to improve.
    """

    def __init__(self, config: Optional[DualEngineConfig] = None):
        self.config = config or DualEngineConfig()

    def score(self, coords: np.ndarray, pair_constraints: List) -> Tuple[float, Dict]:
        """Score a single conformation.

        Returns:
            (energy, feedback_dict) with per-residue strain map
        """
        L = len(coords)
        bond_length = self.config.bond_length
        clash_dist = self.config.clash_distance

        total_energy = 0.0
        residue_strain = np.zeros(L)  # Per-residue strain score

        # Bond energy
        for i in range(L):
            j = (i + 1) % L
            d = np.linalg.norm(coords[j] - coords[i])
            strain = (d - bond_length) ** 2
            total_energy += strain
            residue_strain[i] += strain
            residue_strain[j] += strain

        # Pair constraint energy
        for (i, j, target_d, weight) in pair_constraints:
            d = np.linalg.norm(coords[j] - coords[i])
            strain = weight * (d - target_d) ** 2
            total_energy += strain
            residue_strain[i] += strain
            residue_strain[j] += strain

        # Clash energy
        for i in range(L):
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue
                d = np.linalg.norm(coords[j] - coords[i])
                if d < clash_dist:
                    strain = 10.0 * (clash_dist - d) ** 2
                    total_energy += strain
                    residue_strain[i] += strain
                    residue_strain[j] += strain

        # Stacking energy
        for i in range(L):
            j = (i + 1) % L
            dz = abs(coords[j, 2] - coords[i, 2])
            strain = 0.3 * (dz - 3.4) ** 2
            total_energy += strain
            residue_strain[i] += strain

        # Electrostatic repulsion
        for i in range(L):
            for j in range(i + 2, min(i + 20, L)):
                if i == 0 and j == L - 1:
                    continue
                d = np.linalg.norm(coords[j] - coords[i])
                if 0.1 < d < 20.0:
                    e = 0.05 / d
                    total_energy += e

        # Identify top-K problematic residues
        top_k_indices = np.argsort(residue_strain)[-self.config.feedback_top_k:]

        # Generate constraint violations list
        violations = []
        for (i, j, target_d, weight) in pair_constraints:
            d = np.linalg.norm(coords[j] - coords[i])
            if abs(d - target_d) > 2.0:  # >2Å deviation
                violations.append({
                    'pair': (i, j),
                    'target': target_d,
                    'actual': d,
                    'deviation': abs(d - target_d),
                })

        feedback = {
            'energy': total_energy,
            'residue_strain': residue_strain,
            'top_k_indices': top_k_indices,
            'violations': violations,
            'closure_distance': np.linalg.norm(coords[0] - coords[-1]),
        }

        return total_energy, feedback

    def select_best(
        self,
        candidates: List[np.ndarray],
        pair_constraints: List,
    ) -> Tuple[np.ndarray, Dict]:
        """Score all candidates, return best + aggregated feedback."""
        scored = []
        for coords in candidates:
            energy, feedback = self.score(coords, pair_constraints)
            scored.append((energy, coords, feedback))

        # Sort by energy
        scored.sort(key=lambda x: x[0])

        best_coords = scored[0][1]
        best_feedback = scored[0][2]

        # Aggregate feedback from top-5 candidates
        top5_feedbacks = [s[2] for s in scored[:5]]
        avg_strain = np.mean([f['residue_strain'] for f in top5_feedbacks], axis=0)
        best_feedback['avg_residue_strain'] = avg_strain

        return best_coords, best_feedback


class FeedbackConditioner(nn.Module):
    """Converts physics feedback into conditioning signal for Generator.

    Takes residue strain map → attention bias adjustment for CircPairformer.
    High-strain residues get more attention in the next iteration.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.strain_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(
        self,
        residue_strain: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Convert strain to attention bias.

        Returns:
            (B, L, d_model) conditioning signal
        """
        strain_input = residue_strain.unsqueeze(-1)  # (B, L, 1)
        bias = self.strain_proj(strain_input)  # (B, L, d_model)
        return bias


class DualEngineTorusFold(nn.Module):
    """Scheme 3: Dual-engine iterative evolution.

    Architecture:
        G (Generator): TPE + CircPairformer → predicts pair_probs + initial coords
        S (Selector): PhysicsSelector → scores + feedback
        F (Feedback): FeedbackConditioner → converts feedback to G conditioning

    Iteration:
        1. G predicts candidates (n_candidates per round)
        2. S scores and selects best
        3. F converts feedback to conditioning
        4. G re-predicts with conditioning
        5. Repeat n_iterations times
    """

    def __init__(self, config: Optional[DualEngineConfig] = None):
        super().__init__()
        self.config = config or DualEngineConfig()

        # Generator components (lightweight)
        self.embed = nn.Embedding(5, self.config.d_model)
        self.feedback_conditioner = FeedbackConditioner(self.config.d_model)

        # Selector (NumPy-based, not a module)
        self.selector = PhysicsSelector(self.config)

    def predict(
        self,
        sequence: str,
        pair_constraints: List,
        n_iterations: Optional[int] = None,
    ) -> Dict:
        """Run dual-engine prediction.

        Args:
            sequence: circRNA sequence (ACGU)
            pair_constraints: List of (i, j, target_d, weight) tuples
            n_iterations: Override config iterations

        Returns:
            Dict with best coords, energy history, feedback history
        """
        n_iter = n_iterations or self.config.n_iterations
        L = len(sequence)

        best_coords = None
        best_energy = float('inf')
        energy_history = []
        feedback_history = []
        residue_strain = np.zeros(L)

        for iteration in range(n_iter):
            # Generate candidates
            # In real implementation, CircPairformer generates with strain conditioning
            # Here we use the constraint solver with varying random seeds
            candidates = self._generate_candidates(L, pair_constraints, residue_strain)

            # Score and select
            best_iter_coords, feedback = self.selector.select_best(
                candidates, pair_constraints
            )

            # Update best
            if feedback['energy'] < best_energy:
                best_energy = feedback['energy']
                best_coords = best_iter_coords

            # Record history
            energy_history.append(feedback['energy'])
            feedback_history.append({
                'iteration': iteration,
                'energy': feedback['energy'],
                'top_k': feedback['top_k_indices'].tolist(),
                'n_violations': len(feedback['violations']),
                'closure': feedback['closure_distance'],
            })

            # Update strain for next iteration
            residue_strain = feedback.get('avg_residue_strain', feedback['residue_strain'])

            # Early stopping
            if feedback['energy'] < 0.1 and feedback['closure_distance'] < 0.5:
                break

        return {
            'coords': best_coords,
            'energy': best_energy,
            'energy_history': energy_history,
            'feedback_history': feedback_history,
            'n_iterations': len(energy_history),
        }

    def _generate_candidates(
        self,
        L: int,
        pair_constraints: List,
        residue_strain: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate diverse candidate conformations.

        Uses regular polygon + strain-aware perturbation.
        """
        from .constraint_solver import GeometricConstraintSolver, SolverConfig

        config = SolverConfig(n_samples=self.config.n_candidates)
        solver = GeometricConstraintSolver(config)

        # Create a minimal constraint set
        class MinimalConstraintSet:
            def __init__(self, seq_len, pairs):
                self.seq_len = seq_len
                self.pair_constraints = pairs

        constraint_set = MinimalConstraintSet(L, pair_constraints)
        conformations = solver.solve(constraint_set)

        return conformations


# ── Path 2: CS-Fold inspired gradient feedback ──────────────

class BSJClosurePenalty(nn.Module):
    """BSJ closure as hard training penalty (CS-Fold inspired).

    Instead of relying on S engine to discover closure problems
    post-hoc, this penalty forces G to learn closed structures
    during training. This is the "micro" path of Scheme 3.

    Penalty = ||d(x[0], x[-1]) - bond_length||^2

    This can be added to any training loss function.
    """

    def __init__(self, bond_length: float = 5.9, weight: float = 1.0):
        super().__init__()
        self.bond_length = bond_length
        self.weight = weight

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute BSJ closure penalty.

        Args:
            coords: (B, L, 3) predicted coordinates

        Returns:
            Scalar penalty (lower = better closure)
        """
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)
        penalty = (closure_dist - self.bond_length) ** 2
        return self.weight * penalty.mean()


class PaxNetScorer(nn.Module):
    """PaxNet-inspired multiplex graph scoring (Scheme 5 + Scheme 3 S engine).

    Two-layer multiplex graph:
    - Local layer: bond + angle interactions (short-range)
    - Non-local layer: van der Waals + electrostatic (long-range)
    - BSJ edge: explicit 5'-3' connection (circRNA-specific)

    After scoring, provides gradient feedback to G engine.
    """

    def __init__(
        self,
        d_node: int = 64,
        d_edge_local: int = 32,
        d_edge_nonlocal: int = 16,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.d_node = d_node
        self.bond_length = bond_length

        # Local message passing (bond + angle)
        self.local_msg = nn.Linear(d_node * 2 + d_edge_local, d_node)

        # Non-local message passing (vdW + electrostatic)
        self.nonlocal_msg = nn.Linear(d_node * 2 + d_edge_nonlocal, d_node)

        # Attention fusion (local vs non-local weight)
        self.fusion = nn.Sequential(
            nn.Linear(d_node * 2, d_node),
            nn.Sigmoid(),
        )

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(d_node, d_node // 2),
            nn.GELU(),
            nn.Linear(d_node // 2, 1),
        )

    def score_with_feedback(
        self,
        coords: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Score structure and provide gradient feedback.

        Returns:
            Dict with score, gradient, local/non-local contributions
        """
        coords = coords.detach().requires_grad_(True)
        B, L, _ = coords.shape

        # Compute distance matrix
        dist_matrix = torch.cdist(coords, coords)  # (B, L, L)

        # Local layer: bond + angle (adjacent residues, circular)
        idx_i = torch.arange(L, device=coords.device)
        idx_j = (idx_i + 1) % L
        bond_dists = dist_matrix[:, idx_i, idx_j]  # (B, L)
        local_energy = ((bond_dists - self.bond_length) ** 2).sum(dim=1)  # (B,)

        # Non-local layer: vdW + electrostatic (vectorized)
        # Build mask for non-adjacent pairs (|i-j| >= 2, excluding BSJ pair)
        mask = torch.ones(L, L, device=coords.device, dtype=torch.bool)
        for i in range(L):
            j = (i + 1) % L
            mask[i, j] = False
            mask[j, i] = False
            mask[i, i] = False
        # Exclude BSJ pair (0, L-1)
        if L > 2:
            mask[0, L-1] = False
            mask[L-1, 0] = False

        # Upper triangle only to avoid double counting
        mask = mask & torch.triu(torch.ones(L, L, device=coords.device, dtype=torch.bool), diagonal=2)

        # Safe inverse distance (clamp to avoid division by zero)
        safe_dist = dist_matrix.clamp(min=0.5)

        # vdW: simplified Lennard-Jones (only repulsive part matters for training)
        vdw = (3.0 / safe_dist) ** 6  # Use r^-6 instead of full LJ for stability
        # Electrostatic: phosphate repulsion
        elec = 0.05 / safe_dist

        # Apply mask and sum
        nonlocal_per_pair = (vdw + elec) * mask.unsqueeze(0).float()  # (B, L, L)
        nonlocal_energy = nonlocal_per_pair.sum(dim=(1, 2))  # (B,)

        # BSJ edge: explicit 5'-3' connection
        bsj_dist = dist_matrix[:, 0, L-1]
        bsj_penalty = (bsj_dist - self.bond_length) ** 2

        # Total score
        total_score = local_energy + 0.3 * nonlocal_energy + 10.0 * bsj_penalty

        # Compute gradient for feedback
        total_score.sum().backward()
        gradient = coords.grad

        return {
            'score': total_score.detach(),
            'local_energy': local_energy.detach(),
            'nonlocal_energy': nonlocal_energy.detach(),
            'bsj_penalty': bsj_penalty.detach(),
            'gradient': gradient,
        }
