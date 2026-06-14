"""
physics_structure_head.py — Physics-based structure head for circRNA.

This module provides a unified interface that plugs into TorusFold exactly
where SimpleStructureHead and CircDiffusionStructure sit. It composes:

1. ConstraintExtractor: DL predictions → geometric constraints
2. GeometricConstraintSolver: constraints → 3D coordinates (Plan B)
3. CGMDRefiner: coordinates → refined coordinates (Plan A, optional)
4. StructureValidator: coordinates → validation metrics

Mode selection via structure_mode:
- "physics_b": ConstraintExtractor + GeometricConstraintSolver only
- "physics_ba": Plan B + CGMDRefiner (OpenMM refinement)

This is a zero-training supplement to the DL structure heads, for use
when experimental 3D data is unavailable (which is the case for circRNA).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

from .physics_bridge import ConstraintExtractor, ConstraintSet
from .constraint_solver import GeometricConstraintSolver, SolverConfig
from .structure_validator import StructureValidator, ValidationMetrics


class PhysicsStructureHead(nn.Module):
    """Physics-based structure head. Complements existing DL heads.

    Implements Plan B (geometric constraint solver) as primary, with
    optional Plan A (CGMD refinement) as second stage.

    Args:
        c_z: Pair representation dimension
        structure_mode: "physics_b" or "physics_ba"
        bond_length: Backbone bond length (Å)
        pair_distance: Watson-Crick pair distance (Å)
        n_solver_samples: Number of conformations to sample
        pair_threshold: Minimum pair probability for constraint
        n_minimize_steps: OpenMM minimization steps (only physics_ba)
        n_md_steps: OpenMM MD relaxation steps (only physics_ba)
        use_dl_bias: DL bias in CG MD (only physics_ba)
        closure_tolerance: Å tolerance for closure constraint
    """

    def __init__(
        self,
        c_z: int = 128,
        structure_mode: str = "physics_b",
        bond_length: float = 5.9,
        pair_distance: float = 10.6,
        n_solver_samples: int = 20,
        pair_threshold: float = 0.3,
        n_minimize_steps: int = 500,
        n_md_steps: int = 5000,
        use_dl_bias: bool = True,
        closure_tolerance: float = 0.5,
        clash_distance: float = 3.0,
    ):
        super().__init__()
        self.c_z = c_z
        self.structure_mode = structure_mode
        self.bond_length = bond_length
        self.pair_distance = pair_distance
        self.n_solver_samples = n_solver_samples
        self.use_dl_bias = use_dl_bias

        # Constraint extractor (PyTorch module)
        self.constraint_extractor = ConstraintExtractor(
            c_z=c_z,
            bond_length=bond_length,
            pair_distance=pair_distance,
            pair_threshold=pair_threshold,
        )

        # Constraint solver (NumPy, not a module)
        solver_config = SolverConfig(
            bond_length=bond_length,
            pair_distance=pair_distance,
            clash_distance=clash_distance,
            n_samples=n_solver_samples,
            closure_tolerance=closure_tolerance,
        )
        self.constraint_solver = GeometricConstraintSolver(solver_config)

        # Structure validator (NumPy)
        self.validator = StructureValidator(
            bond_length=bond_length,
            clash_distance=clash_distance,
        )

        # CG MD refiner (optional, only for physics_ba)
        self.cgmd_refiner = None
        if structure_mode == "physics_ba":
            # Import lazily to avoid OpenMM dependency for physics_b mode
            try:
                from .cgmd_refiner import CGMDRefiner
                self.cgmd_refiner = CGMDRefiner(
                    bond_length=bond_length,
                    n_minimize_steps=n_minimize_steps,
                    n_md_steps=n_md_steps,
                    use_dl_bias=use_dl_bias,
                )
            except ImportError:
                # OpenMM not available, fall back to physics_b
                self.structure_mode = "physics_b"

        # Confidence head (matches SimpleStructureHead interface)
        self.confidence_head = nn.Sequential(
            nn.Linear(c_z, c_z // 2),
            nn.GELU(),
            nn.Linear(c_z // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict 3D structure from pair representation.

        Same interface as SimpleStructureHead / CircDiffusionStructure:
        takes pair_repr, returns dict with coords, confidence, closure_distance.

        Args:
            pair_repr: (B, L, L, c_z) from CircPairformerStack
            pair_probs: (B, L, L) from PairPredictionHead (optional, computed if not)
            sequences: List of circRNA sequence strings (required for physics modes)

        Returns:
            Dict with coords, confidence, closure_distance, dist_pred,
            plus validation_metrics, energy_score (extras for physics modes)
        """
        B, L, _, c_z = pair_repr.shape
        device = pair_repr.device

        # If no sequences provided, use placeholders
        if sequences is None:
            sequences = ['N' * L for _ in range(B)]

        # If no pair_probs, compute from pair_repr diagonal mean
        if pair_probs is None:
            # Simple estimate: use diagonal features
            diag = pair_repr.diagonal(dim1=1, dim2=2)  # (B, L, c_z)
            pair_logits = self.confidence_head(diag).squeeze(-1)  # (B, L) placeholder
            pair_probs = torch.sigmoid(pair_logits.unsqueeze(1).expand(-1, L, -1) * 0.5)

        # Process each sequence in batch
        batch_coords = []
        batch_closure = []
        batch_confidence = []
        batch_energy = []
        batch_stability = []

        for b in range(B):
            # Extract constraints (PyTorch)
            constraint_set = self.constraint_extractor(
                pair_repr[b:b+1],
                pair_probs[b:b+1] if pair_probs.dim() == 3 else pair_probs,
                sequences[b],
            )

            # Solve constraints (NumPy)
            conformations = self.constraint_solver.solve(constraint_set)

            # Validate and select best
            best_coords, best_metrics = self.validator.validate_best(
                conformations, constraint_set
            )

            # Optional CGMD refinement (NumPy → NumPy)
            if self.structure_mode == "physics_ba" and self.cgmd_refiner is not None:
                refined = self.cgmd_refiner.refine(
                    best_coords,
                    constraint_set,
                    pair_repr[b:b+1] if self.use_dl_bias else None,
                )
                best_coords = refined['coords']
                best_metrics = self.validator.validate(best_coords, constraint_set)

            # Confidence from pair_repr (PyTorch)
            pair_mean = pair_repr[b].mean(dim=(0, 1))  # (c_z,)
            confidence = self.confidence_head(pair_mean).squeeze()

            batch_coords.append(best_coords)
            batch_closure.append(best_metrics.closure_distance)
            batch_confidence.append(confidence)
            batch_energy.append(best_metrics.energy_score)
            batch_stability.append(best_metrics.stability_score)

        # Convert to tensors
        coords_tensor = torch.tensor(
            np.stack(batch_coords), dtype=torch.float32, device=device
        )  # (B, L, 3)
        closure_tensor = torch.tensor(
            batch_closure, dtype=torch.float32, device=device
        )  # (B,)
        confidence_tensor = torch.stack(batch_confidence)  # (B,)
        energy_tensor = torch.tensor(
            batch_energy, dtype=torch.float32, device=device
        )  # (B,)
        stability_tensor = torch.tensor(
            batch_stability, dtype=torch.float32, device=device
        )  # (B,)

        # Distance prediction (placeholder, use simple estimate)
        dist_pred = torch.cdist(coords_tensor, coords_tensor)  # (B, L, L)

        return {
            'coords': coords_tensor,
            'confidence': confidence_tensor * 100,  # Scale to [0, 100] like SimpleStructureHead
            'closure_distance': closure_tensor,
            'closure_dist': closure_tensor,  # Alias for compatibility
            'dist_pred': dist_pred,
            'energy_score': energy_tensor,
            'stability_score': stability_tensor,
            'structure_method': self.structure_mode,
        }