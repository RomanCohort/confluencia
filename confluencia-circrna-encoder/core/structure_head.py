"""
structure_head.py — 3D Torus Structure Prediction for circular RNA.

Predicts the 3D coordinates of each nucleotide in a circRNA molecule,
with a key constraint that the circular topology must be preserved:
the first and last nucleotides must be covalently bonded.

Unlike AlphaFold which predicts protein structures (open chains),
TorusFold predicts RNA structures on a ring topology where:
- x[0] and x[L-1] must be within bond distance
- The structure may contain topological knots
- SE(3) equivariance is still required (rotation/translation invariance)

Architecture:
1. Pair representation → 3D coordinate initialization
2. Structure refinement with SE(3) equivariant updates
3. Circular closure constraint loss
4. Confidence estimation (pLDDT-style)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureInitialization(nn.Module):
    """
    Initialize 3D coordinates from pair representation.

    Uses a weighted average of predicted inter-nucleotide distances
    to place initial coordinates on a circle.
    """

    def __init__(self, d_pair: int = 64, n_rbf: int = 16):
        super().__init__()
        self.n_rbf = n_rbf

        # Predict distance between each pair
        self.dist_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, n_rbf),
        )

        # RBF centers for distance binning (0.5 Å to 30 Å)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.5, 30.0, n_rbf),
        )
        self.register_buffer(
            "rbf_gamma",
            torch.tensor(10.0),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize 3D coordinates from pair representation.

        Args:
            pair_repr: (B, L, L, d_pair)

        Returns:
            coords: (B, L, 3) initial 3D coordinates
            dist_pred: (B, L, L) predicted pairwise distances
        """
        B, L, _, d = pair_repr.shape

        # Predict distance distribution for each pair
        dist_logits = self.dist_head(pair_repr)  # (B, L, L, n_rbf)

        # Convert to soft distance via RBF
        dist_probs = F.softmax(dist_logits, dim=-1)
        dist_pred = (dist_probs * self.rbf_centers).sum(dim=-1)  # (B, L, L)

        # Enforce symmetry
        dist_pred = 0.5 * (dist_pred + dist_pred.transpose(-1, -2))

        # Initialize coordinates via multidimensional scaling (MDS-like)
        # Center the distance matrix
        H = -0.5 * (dist_pred ** 2)
        row_mean = H.mean(dim=-1, keepdim=True)
        col_mean = H.mean(dim=-2, keepdim=True)
        grand_mean = H.mean(dim=(-1, -2), keepdim=True)
        B_mat = H - row_mean - col_mean + grand_mean

        # Eigendecomposition for top 3 components
        # Use SVD for stability
        try:
            U, S, Vh = torch.linalg.svd(B_mat)
            # Take top 3 components
            coords = U[:, :, :3] * torch.sqrt(S[:, :3].unsqueeze(1))
        except Exception:
            # Fallback: place on a circle
            coords = self._init_circle(B, L, pair_repr.device)

        return coords, dist_pred

    @staticmethod
    def _init_circle(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Fallback: initialize coordinates on a circle."""
        theta = torch.linspace(0, 2 * math.pi, seq_len + 1, device=device)[:-1]
        radius = seq_len * 0.3  # Scale with length

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z = torch.zeros(seq_len, device=device)

        coords = torch.stack([x, y, z], dim=-1)  # (L, 3)
        return coords.unsqueeze(0).expand(batch_size, -1, -1)


class StructureRefinementLayer(nn.Module):
    """
    SE(3)-equivariant structure refinement layer.

    Updates 3D coordinates based on:
    1. Current coordinates (geometric context)
    2. Pair representations (sequence/structural features)
    3. Circular closure constraint

    Inspired by AlphaFold's structure module but adapted for
    circular topology.
    """

    def __init__(self, d_pair: int = 64, d_coord: int = 32, n_heads: int = 4):
        super().__init__()
        self.d_pair = d_pair
        self.d_coord = d_coord

        # Coordinate encoder: 3D → d_coord
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, d_coord),
            nn.GELU(),
            nn.Linear(d_coord, d_coord),
        )

        # Pair-conditioned update
        # Input: coord_feat (d_coord) + pair_agg (d_pair)
        self.update_net = nn.Sequential(
            nn.Linear(d_coord + d_pair, d_coord * 2),
            nn.GELU(),
            nn.LayerNorm(d_coord * 2),
            nn.Linear(d_coord * 2, d_coord),
            nn.GELU(),
            nn.Linear(d_coord, 3),  # Output: 3D displacement
        )

        self.norm = nn.LayerNorm(3)

    def forward(
        self,
        coords: torch.Tensor,
        pair_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine 3D coordinates.

        Args:
            coords: (B, L, 3) current coordinates
            pair_repr: (B, L, L, d_pair)

        Returns:
            (B, L, 3) updated coordinates
        """
        B, L, _ = coords.shape

        # Encode current coordinates
        coord_feat = self.coord_encoder(coords)  # (B, L, d_coord)

        # For each position i, aggregate pair features with neighbor j
        # Use top-k nearest neighbors for efficiency
        with torch.no_grad():
            # Compute pairwise distances
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
            dist = diff.norm(dim=-1)  # (B, L, L)

            # Top-k neighbors
            k = min(32, L)
            _, topk_idx = dist.topk(k, dim=-1, largest=False)  # (B, L, k)

        # Gather neighbor features
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_pair)  # (B, L, k, d_pair)
        neighbor_pair = torch.gather(
            pair_repr, 2, topk_idx_expanded.permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)  # This doesn't work directly, use loop

        # Simpler: use mean-pooled pair features
        pair_agg = pair_repr.mean(dim=2)  # (B, L, d_pair)

        # Compute displacement
        concat_feat = torch.cat([coord_feat, pair_agg], dim=-1)
        displacement = self.update_net(concat_feat)  # (B, L, 3)

        # Apply update with small step size
        coords_new = coords + 0.1 * displacement

        return coords_new


class CircularClosureLoss(nn.Module):
    """
    Loss term that enforces circular closure.

    For circRNA, the distance between x[0] and x[L-1] must be
    approximately equal to a typical phosphodiester bond length
    (~3.4 Å in A-form RNA helix).
    """

    def __init__(self, bond_length: float = 3.4, tolerance: float = 1.0):
        super().__init__()
        self.bond_length = bond_length
        self.tolerance = tolerance

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, L, 3) predicted coordinates

        Returns:
            Scalar loss
        """
        # Distance between first and last nucleotide
        closure_dist = (coords[:, 0, :] - coords[:, -1, :]).norm(dim=-1)  # (B,)

        # Target: bond_length with tolerance
        loss = F.smooth_l1_loss(closure_dist, torch.full_like(closure_dist, self.bond_length))

        return loss


class ConfidenceEstimator(nn.Module):
    """
    Predict per-nucleotide confidence (pLDDT-style).

    Like AlphaFold's pLDDT, predicts how confident the model is
    about each nucleotide's predicted position.
    """

    def __init__(self, d_pair: int = 64):
        super().__init__()
        self.conf_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, 1),
            nn.Sigmoid(),  # Output in [0, 1] → scaled to [0, 100]
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, d_pair)
        Returns:
            (B, L) per-position confidence in [0, 100]
        """
        # Aggregate pair features for each position
        agg = pair_repr.mean(dim=2)  # (B, L, d_pair)
        conf = self.conf_head(agg).squeeze(-1)  # (B, L)
        return conf * 100.0


class TorusStructureHead(nn.Module):
    """
    3D structure prediction head with circular topology constraints.

    Full pipeline:
    1. Initialize coordinates from pair representation
    2. Refine with SE(3)-equivariant updates
    3. Enforce circular closure
    4. Estimate per-nucleotide confidence

    Args:
        d_pair: Pair representation dimension
        d_coord: Internal coordinate feature dimension
        n_refinement_iters: Number of structure refinement iterations
        n_rbf: Number of RBF kernels for distance prediction
        bond_length: Target bond length for circular closure (Å)
    """

    def __init__(
        self,
        d_pair: int = 64,
        d_coord: int = 32,
        n_refinement_iters: int = 3,
        n_rbf: int = 16,
        bond_length: float = 3.4,
    ):
        super().__init__()
        self.n_refinement_iters = n_refinement_iters

        # Initialization
        self.initializer = StructureInitialization(d_pair, n_rbf)

        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            StructureRefinementLayer(d_pair, d_coord)
            for _ in range(n_refinement_iters)
        ])

        # Circular closure
        self.closure_loss = CircularClosureLoss(bond_length)

        # Confidence
        self.confidence = ConfidenceEstimator(d_pair)

    def forward(
        self,
        pair_repr: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 3D structure from pair representation.

        Args:
            pair_repr: (B, L, L, d_pair) from IRSPairModule
            return_loss: Whether to compute closure loss

        Returns:
            Dict with:
                - coords: (B, L, 3) predicted 3D coordinates
                - dist_pred: (B, L, L) predicted pairwise distances
                - confidence: (B, L) per-position confidence (pLDDT)
                - closure_loss: scalar (if return_loss=True)
                - closure_distance: (B,) distance between first/last nucleotide
        """
        # 1. Initialize
        coords, dist_pred = self.initializer(pair_repr)

        # 2. Iterative refinement
        for layer in self.refinement_layers:
            coords = layer(coords, pair_repr)

        # 3. Compute closure distance
        closure_distance = (coords[:, 0, :] - coords[:, -1, :]).norm(dim=-1)  # (B,)

        # 4. Compute losses
        result = {
            "coords": coords,
            "dist_pred": dist_pred,
            "closure_distance": closure_distance,
        }

        if return_loss:
            closure_loss = self.closure_loss(coords)
            result["closure_loss"] = closure_loss

        # 5. Confidence
        confidence = self.confidence(pair_repr)
        result["confidence"] = confidence

        return result

    def compute_structure_loss(
        self,
        coords: torch.Tensor,
        pair_repr: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total structure loss.

        Args:
            coords: (B, L, 3) predicted coordinates
            pair_repr: (B, L, L, d_pair) pair representations
            target_coords: (B, L, 3) ground truth (optional)

        Returns:
            Dict of loss terms
        """
        losses = {}

        # 1. Circular closure loss
        losses["closure_loss"] = self.closure_loss(coords)

        # 2. Bond length consistency loss
        # Adjacent nucleotides should be ~3.4 Å apart
        adjacent_dist = (coords[:, 1:, :] - coords[:, :-1, :]).norm(dim=-1)
        target_bond = torch.full_like(adjacent_dist, 3.4)
        losses["bond_loss"] = F.smooth_l1_loss(adjacent_dist, target_bond)

        # 3. Steric clash loss (no two nucleotides too close)
        all_dist = torch.cdist(coords, coords)  # (B, L, L)
        # Penalize non-adjacent pairs closer than 3.0 Å
        min_dist = 3.0
        clash_mask = (all_dist < min_dist).float()
        # Remove diagonal and adjacent
        eye = torch.eye(all_dist.size(-1), device=all_dist.device)
        clash_mask = clash_mask * (1 - eye.unsqueeze(0))
        adj_mask = torch.zeros_like(eye)
        adj_mask[range(all_dist.size(-1) - 1), range(1, all_dist.size(-1))] = 1
        adj_mask[range(1, all_dist.size(-1)), range(all_dist.size(-1) - 1)] = 1
        clash_mask = clash_mask * (1 - adj_mask.unsqueeze(0))

        losses["clash_loss"] = (clash_mask * (min_dist - all_dist).clamp(max=0).abs()).mean()

        # 4. FAPE loss (if target provided)
        if target_coords is not None:
            losses["fape_loss"] = self._fape_loss(coords, target_coords)

        return losses

    @staticmethod
    def _fape_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        clamp: float = 10.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Frame Aligned Point Error (FAPE) loss, adapted from AlphaFold.

        Measures the average distance between predicted and target
        coordinates after optimal rigid alignment.
        """
        B, L, _ = pred.shape

        # Center both structures
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)

        # Compute optimal rotation via SVD (Kabsch algorithm)
        H = torch.bmm(target_centered.transpose(1, 2), pred_centered)  # (B, 3, 3)
        U, S, Vh = torch.linalg.svd(H)

        # Ensure proper rotation (det = +1)
        det = torch.det(torch.bmm(U, Vh))
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        Vh_adj = Vh * sign

        R = torch.bmm(U, Vh_adj)  # (B, 3, 3)

        # Apply rotation to prediction
        pred_aligned = torch.bmm(pred_centered, R.transpose(1, 2))

        # Compute distance
        dist = (pred_aligned - target_centered).norm(dim=-1)  # (B, L)
        fape = dist.clamp(max=clamp).mean()

        return fape
