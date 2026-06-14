"""
diffusion_structure.py — Diffusion-based 3D structure prediction for circRNA.

Inspired by AlphaFold3's diffusion module, adapted for circRNA's torus topology.

AF3 replaces the Structure Module (IPA) from AF2 with a diffusion model that:
1. Operates directly on 3D atom coordinates
2. Conditioned on pair representation from Pairformer
3. Generates diverse structures through denoising
4. Can model side chains, ligands, and nucleic acids

For circRNA, we adapt this:
1. Diffusion on backbone 3D coordinates (P atom positions)
2. Conditioned on pair representation + TPE
3. Circular closure constraint as additional loss
4. Can model IRS elements and back-splice junction geometry

The diffusion process:
- T=0: Pure noise coordinates
- T→T-1: Denoise using pair representation conditioning
- T=0: Final structure with circular closure

This is simpler than AF3's full atom diffusion since we focus on
circRNA backbone + major structural features (stem-loops, IRS pairs).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timestep.

    Following the standard diffusion model time encoding.
    """

    def __init__(self, d_time: int = 64):
        super().__init__()
        self.d_time = d_time

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep values in [0, T_max]

        Returns:
            (B, d_time) time embeddings
        """
        device = t.device
        half_dim = self.d_time // 2

        # Sinusoidal encoding
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class DiffusionConditioning(nn.Module):
    """
    Conditioning network for diffusion model.

    Takes pair representation and time embedding, outputs conditioning
    features for the diffusion denoiser.
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_time: int = 64,
        d_cond: int = 256,
    ):
        super().__init__()
        self.d_pair = d_pair
        self.d_time = d_time
        self.d_cond = d_cond

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_time),
            nn.Linear(d_time, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        # Pair aggregation: z[i,j] → per-position features
        self.pair_agg = nn.Sequential(
            nn.Linear(d_pair, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        # Combine time + pair conditioning
        self.cond_combine = nn.Sequential(
            nn.Linear(d_cond * 2, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, d_pair) pair representation
            t: (B,) timestep

        Returns:
            (B, L, d_cond) per-position conditioning
        """
        B, L, _, d_pair = pair_repr.shape

        # Time embedding
        time_emb = self.time_mlp(t)  # (B, d_cond)

        # Aggregate pair features per position
        # Mean pool over j dimension: z[i,:] → z_i
        pair_agg = pair_repr.mean(dim=2)  # (B, L, d_pair)
        pair_emb = self.pair_agg(pair_agg)  # (B, L, d_cond)

        # Combine time + pair
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_cond)
        cond = torch.cat([time_emb_expanded, pair_emb], dim=-1)
        cond = self.cond_combine(cond)  # (B, L, d_cond)

        return cond


class DiffusionDenoiser(nn.Module):
    """
    Denoiser network for diffusion process.

    Takes noisy coordinates + conditioning, predicts clean coordinates.
    Architecture inspired by AF3's diffusion module but simplified for
    circRNA backbone.
    """

    def __init__(
        self,
        d_cond: int = 256,
        d_coord: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.d_cond = d_cond
        self.d_coord = d_coord

        # Coordinate encoder: (x,y,z) → d_coord features
        self.coord_encoder = nn.Linear(3, d_coord)

        # Denoising MLP layers
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_cond + d_coord, d_cond + d_coord))
            layers.append(nn.LayerNorm(d_cond + d_coord))
            layers.append(nn.GELU())

        self.denoiser = nn.Sequential(*layers)

        # Output: predict displacement (not full coords)
        self.coord_out = nn.Linear(d_cond + d_coord, 3)

        # Circular closure head (optional)
        self.closure_head = nn.Sequential(
            nn.Linear(d_cond, d_cond // 2),
            nn.GELU(),
            nn.Linear(d_cond // 2, 1),
        )

    def forward(
        self,
        coords_noisy: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords_noisy: (B, L, 3) noisy coordinates
            cond: (B, L, d_cond) conditioning from pair repr + time

        Returns:
            coords_clean: (B, L, 3) predicted clean coordinates
            closure_pred: (B,) predicted closure distance
        """
        B, L, _ = coords_noisy.shape

        # Encode coordinates
        coord_feat = self.coord_encoder(coords_noisy)  # (B, L, d_coord)

        # Concatenate with conditioning
        x = torch.cat([cond, coord_feat], dim=-1)  # (B, L, d_cond + d_coord)

        # Denoise
        x = self.denoiser(x)

        # Predict displacement
        displacement = self.coord_out(x)  # (B, L, 3)

        # Clean coords = noisy - displacement (denoising direction)
        coords_clean = coords_noisy - displacement

        # Predict closure distance
        # Use first and last position conditioning
        closure_cond = cond[:, 0, :] + cond[:, -1, :]  # (B, d_cond)
        closure_pred = self.closure_head(closure_cond).squeeze(-1)  # (B,)

        return coords_clean, closure_pred

    # ============== Flexible Structure Head ==============

class FlexibleStructureHead(nn.Module):
    """
    Multi-conformation structure prediction head for flexible circRNA.

    Unlike SimpleStructureHead which outputs a single structure,
    this module predicts multiple possible conformations (an ensemble)
    along with mixture weights and flexibility indices.

    RNA is inherently flexible, and circRNA is even more flexible
    (it's a "spring coil" with no free ends). A single structure
    prediction fails to capture this flexibility.

    Output:
    - M conformations: coords_0, coords_1, ..., coords_M-1
    - Mixture weights: w_0, w_1, ..., w_M (softmax over conformations)
    - Flexibility index: flex_i for each position (high = flexible)
    - Cross-conformation correlation: how correlated are positions across conformations

    Args:
        d_pair: Pair representation dimension
        d_coord: Coordinate dimension (3 for 3D)
        d_cond: Conditioning dimension for diffusion
        n_rbf: Number of RBF features for distance encoding
        num_conformations: Number of conformations to predict (default: 8)
        use_diffusion: Whether to use diffusion for each conformation (default: False)
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_coord: int = 3,
        d_cond: int = 128,
        n_rbf: int = 16,
        num_conformations: int = 8,
        use_diffusion: bool = False,
    ):
        super().__init__()
        self.d_pair = d_pair
        self.d_coord = d_coord
        self.d_cond = d_cond
        self.num_confs = num_conformations
        self.use_diffusion = use_diffusion

        # Per-conformation structure predictors
        self.conf_heads = nn.ModuleList([
            SimpleStructureHead(d_pair, d_coord, n_rbf)
            for _ in range(num_conformations)
        ])

        # Mixture weight predictor
        self.mixture_weights = nn.Sequential(
            nn.Linear(d_pair, 64),
            nn.GELU(),
            nn.Linear(64, num_conformations),
        )

        # Flexibility index predictor (per-position)
        self.flexibility_head = nn.Sequential(
            nn.Linear(d_pair, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Cross-conformation correlation (learnable bias matrix)
        self.conf_bias = nn.Parameter(torch.randn(num_conformations, num_conformations) * 0.1)

    def forward(self, pair_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: (B, L, L, d_pair) pair representation

        Returns:
            Dict with:
                conformations: List of (B, L, d_coord) tensors
                weights: (B, num_confs) mixture weights
                flexibility: (B, L) flexibility indices
                correlation: (B, L, L) cross-conformation correlation
        """
        B, L, _, _ = pair_repr.shape

        # Predict each conformation
        conformations = []
        for head in self.conf_heads:
            out = head(pair_repr)
            conformations.append(out['coords'])

        # Predict mixture weights (condition on pair representation mean)
        pair_mean = pair_repr.mean(dim=(1, 2))  # (B, d_pair)
        weights = F.softmax(self.mixture_weights(pair_mean), dim=-1)  # (B, num_confs)

        # Predict flexibility index
        # Use row-wise mean of pair representation as position features
        pos_feat = pair_repr.mean(dim=2)  # (B, L, d_pair)
        flexibility = self.flexibility_head(pos_feat).squeeze(-1)  # (B, L)

        # Compute cross-conformation correlation
        # For each position, measure variance across conformations
        # High variance = flexible, Low variance = rigid
        stacked_coords = torch.stack(conformations, dim=1)  # (B, num_confs, L, d_coord)

        # Variance across conformations (normalized)
        coord_var = stacked_coords.var(dim=1)  # (B, L, d_coord)
        flex_from_var = coord_var.mean(dim=-1)  # (B, L)

        # Combine explicit flexibility prediction with empirical variance
        flexibility = 0.5 * flexibility + 0.5 * torch.sigmoid(flex_from_var)

        # Weighted average conformation (the "representative" structure)
        weighted_coords = torch.einsum('bm,bmlc->blc', weights, stacked_coords)

        # Add bias-weighted conformation interactions
        # This captures correlations between conformations (e.g., if conf_0 moves, conf_1 moves)
        conf_bias_exp = torch.exp(self.conf_bias)  # (num_confs, num_confs)
        conf_corr = F.softmax(conf_bias_exp, dim=-1)  # Soft correlation matrix

        return {
            'conformations': conformations,
            'weights': weights,
            'flexibility': flexibility,
            'weighted_coords': weighted_coords,
            'conf_correlation': conf_corr,
            'stacked_coords': stacked_coords,
        }

    def sample(self, pair_repr: torch.Tensor, n_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Sample conformations according to mixture weights.

        Args:
            pair_repr: (B, L, L, d_pair)
            n_samples: Number of samples (default: 1)

        Returns:
            Dict with sampled coordinates and metadata
        """
        out = self.forward(pair_repr)
        weights = out['weights']  # (B, num_confs)
        conformations = out['conformations']
        B = weights.size(0)

        # Sample from mixture
        sampled_indices = torch.multinomial(weights, n_samples, replacement=True)  # (B, n_samples)

        # Get sampled conformations
        sampled_coords = []
        for b in range(B):
            for s in range(n_samples):
                idx = sampled_indices[b, s]
                sampled_coords.append(conformations[idx][b])

        sampled_coords = torch.stack(sampled_coords).view(B, n_samples, -1, self.d_coord)

        return {
            'coords': sampled_coords[:, 0],  # First sample
            'all_samples': sampled_coords,
            'weights': weights,
            'flexibility': out['flexibility'],
        }


class ClosureConstrainedDiffusion(nn.Module):
    """
    Diffusion model with explicit closure constraint for circRNA.

    Extends CircDiffusionStructure with:
    1. Multiple diffusion trajectories (conformation ensemble)
    2. Per-position flexibility-aware closure tolerance
    3. B-factor-like confidence from diffusion noise

    Args:
        Same as CircDiffusionStructure plus:
        num_conformations: Number of diffusion trajectories
        flex_tolerance_scale: How much flexibility affects closure tolerance
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_time: int = 64,
        d_cond: int = 128,
        d_coord: int = 3,
        n_layers: int = 3,
        n_steps: int = 10,
        bond_length: float = 3.4,
        num_conformations: int = 4,
        flex_tolerance_scale: float = 0.5,
    ):
        super().__init__()
        self.num_confs = num_conformations
        self.flex_tolerance_scale = flex_tolerance_scale
        self.bond_length = bond_length
        self.n_steps = n_steps

        # Multiple denoisers for each conformation
        self.denoisers = nn.ModuleList([
            DenoiseNetwork(d_coord, d_time, d_cond, n_layers)
            for _ in range(num_conformations)
        ])

        # Mixture weights (which conformation is most likely)
        self.mixture_weights = nn.Sequential(
            nn.Linear(d_cond, 32),
            nn.GELU(),
            nn.Linear(32, num_conformations),
        )

        # Flexibility predictor
        self.flexibility_head = nn.Sequential(
            nn.Linear(d_pair, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time),
        )

    def forward(self, coords_noisy: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor, flexibility: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one denoising step."""
        # Simplified: use first denoiser
        t_emb = self.time_embed(t.unsqueeze(-1))
        return self.denoisers[0](coords_noisy, t_emb, cond), torch.zeros(coords_noisy.size(0))

    def sample(self, pair_repr: torch.Tensor, n_samples: int = 1,
               return_trajectory: bool = False, flexibility: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Sample multiple conformations with flexibility-aware closure.

        Args:
            pair_repr: (B, L, L, d_pair)
            n_samples: Samples per conformation
            return_trajectory: Whether to return full diffusion trajectory
            flexibility: (B, L) optional precomputed flexibility indices

        Returns:
            Dict with conformations, weights, closure distances, flexibility
        """
        B, L, _, d_pair = pair_repr.shape
        device = pair_repr.device

        # Compute flexibility if not provided
        if flexibility is None:
            pos_feat = pair_repr.mean(dim=2)
            flexibility = self.flexibility_head(pos_feat).squeeze(-1)  # (B, L)

        # Condition from pair representation
        cond = pair_repr.mean(dim=(1, 2))  # (B, d_pair)

        # Predict mixture weights
        weights = F.softmax(self.mixture_weights(cond), dim=-1)  # (B, num_confs)

        # Sample each conformation
        conformations = []
        closure_dists = []

        for conf_idx in range(self.num_confs):
            # Initialize with noise
            coords = torch.randn(B, L, 3, device=device) * 10.0

            # Diffusion steps
            for step_idx in range(self.n_steps):
                t = torch.ones(B, device=device) * (self.n_steps - step_idx) / self.n_steps
                coords_clean, _ = self.denoisers[conf_idx](coords, self.time_embed(t.unsqueeze(-1)), cond)

                # Apply closure constraint with flexibility tolerance
                if step_idx > self.n_steps // 2:
                    # Higher flexibility = more tolerance for imperfect closure
                    tol = self.bond_length * (1.0 + self.flex_tolerance_scale * flexibility.mean(dim=-1))
                    closure_vec = coords_clean[:, -1] - coords_clean[:, 0]
                    closure_dist = closure_vec.norm(dim=-1)

                    # Soft constraint: move towards closure if too far
                    too_far = closure_dist > tol
                    if too_far.any():
                        target = coords_clean[:, 0] + closure_vec / closure_dist.unsqueeze(-1) * tol.unsqueeze(-1)
                        coords_clean[:, -1] = torch.where(too_far.unsqueeze(-1), target, coords_clean[:, -1])

                coords = coords_clean

            # Final closure distance
            final_closure = (coords[:, 0] - coords[:, -1]).norm(dim=-1)
            conformations.append(coords)
            closure_dists.append(final_closure)

        # Stack results
        stacked_coords = torch.stack(conformations, dim=1)  # (B, num_confs, L, 3)
        closure_dists = torch.stack(closure_dists, dim=1)  # (B, num_confs)

        # Weighted average conformation
        weighted_coords = torch.einsum('bm,bmlc->blc', weights, stacked_coords)

        return {
            'coords': weighted_coords,
            'conformations': conformations,
            'weights': weights,
            'closure_dists': closure_dists,
            'flexibility': flexibility,
            'stacked_coords': stacked_coords,
        }


class CircDiffusionStructure(nn.Module):
    """
    Full diffusion-based structure prediction for circRNA.

    Inspired by AlphaFold3's diffusion module, with key adaptations:
    1. Circular topology: closure constraint as diffusion endpoint
    2. IRS conditioning: pair representation includes IRS pairs
    3. TPE conditioning: torus positional encoding guides structure
    4. Multiple sampling: generate diverse structures

    Args:
        d_pair: Pair representation dimension
        d_time: Time embedding dimension
        d_cond: Conditioning dimension
        d_coord: Coordinate feature dimension
        n_layers: Number of denoiser layers
        n_steps: Number of diffusion steps (inference)
        schedule: Noise schedule ("linear", "cosine")
        bond_length: Target bond length for closure (Å)
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_time: int = 64,
        d_cond: int = 256,
        d_coord: int = 64,
        n_layers: int = 4,
        n_steps: int = 100,
        schedule: str = "cosine",
        bond_length: float = 3.4,
    ):
        super().__init__()
        self.d_pair = d_pair
        self.n_steps = n_steps
        self.schedule = schedule
        self.bond_length = bond_length

        # Conditioning network
        self.conditioning = DiffusionConditioning(d_pair, d_time, d_cond)

        # Denoiser
        self.denoiser = DiffusionDenoiser(d_cond, d_coord, n_layers)

        # Confidence estimator (like AF3's confidence head)
        self.confidence = nn.Sequential(
            nn.Linear(d_pair, d_cond // 2),
            nn.GELU(),
            nn.Linear(d_cond // 2, 1),
            nn.Sigmoid(),
        )

    def _get_sigma_schedule(self, device: torch.device) -> torch.Tensor:
        """Compute noise schedule (sigma_t for each timestep)."""
        t = torch.arange(self.n_steps, device=device, dtype=torch.float32)

        if self.schedule == "linear":
            # Linear: sigma_t = sigma_max * (1 - t/T)
            sigma = 1.0 * (1 - t / self.n_steps)
        elif self.schedule == "cosine":
            # Cosine: smoother schedule
            sigma = torch.cos(0.5 * math.pi * t / self.n_steps)
        else:
            sigma = 1.0 * (1 - t / self.n_steps)

        return sigma

    def sample(
        self,
        pair_repr: torch.Tensor,
        n_samples: int = 1,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample structure from pair representation via diffusion.

        Args:
            pair_repr: (B, L, L, d_pair) pair representation from Pairformer
            n_samples: Number of diverse samples (uncertainty estimation)
            return_trajectory: Whether to return full denoising trajectory

        Returns:
            Dict with:
                - coords: (B*n_samples, L, 3) final coordinates
                - confidence: (B*n_samples, L) per-position confidence
                - closure_dist: (B*n_samples,) closure distances
                - trajectory: List of coords at each step (if return_trajectory)
        """
        B, L, _, _ = pair_repr.shape
        device = pair_repr.device

        # Expand for multiple samples
        if n_samples > 1:
            pair_repr = pair_repr.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
            pair_repr = pair_repr.reshape(B * n_samples, L, L, self.d_pair)
            B = B * n_samples

        # Initialize: random noise coordinates
        coords = torch.randn(B, L, 3, device=device) * 10.0  # Large initial variance

        # Get sigma schedule
        sigmas = self._get_sigma_schedule(device)

        trajectory = [] if return_trajectory else None

        # Diffusion denoising loop
        for t_idx in range(self.n_steps - 1, -1, -1):  # T-1 → 0
            t = torch.full((B,), t_idx, device=device, dtype=torch.float32)

            # Get conditioning
            cond = self.conditioning(pair_repr, t)

            # Denoise step
            coords, closure_pred = self.denoiser(coords, cond)

            # Apply closure constraint progressively
            # In later steps, enforce x[0] ≈ x[L-1] more strongly
            closure_weight = (self.n_steps - t_idx) / self.n_steps
            if closure_weight > 0.5:
                # Partial closure enforcement: move x[0] and x[-1] towards bond_length distance
                closure_vec = coords[:, -1] - coords[:, 0]
                closure_dist = closure_vec.norm(dim=-1)
                if closure_dist > self.bond_length * 2:
                    # Too far: pull towards bond_length
                    target_vec = closure_vec / closure_dist * self.bond_length
                    coords[:, -1] = coords[:, 0] + target_vec

            if return_trajectory:
                trajectory.append(coords.clone())

            # Add noise for next step (except at t=0)
            if t_idx > 0:
                noise = torch.randn_like(coords) * sigmas[t_idx] * 0.1
                coords = coords + noise

        # Final closure enforcement
        closure_dist = (coords[:, 0] - coords[:, -1]).norm(dim=-1)

        # Confidence estimation
        pair_agg = pair_repr.mean(dim=2)  # (B, L, d_pair)
        confidence = self.confidence(pair_agg).squeeze(-1) * 100.0  # (B, L)

        return {
            "coords": coords,
            "confidence": confidence,
            "closure_dist": closure_dist,
            "trajectory": trajectory,
        }

    def compute_loss(
        self,
        coords_noisy: torch.Tensor,
        coords_clean: torch.Tensor,
        pair_repr: torch.Tensor,
        t: torch.Tensor,
        coords_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for diffusion model.

        Args:
            coords_noisy: Noisy coordinates
            coords_clean: Predicted clean coordinates
            pair_repr: Pair representation
            t: Current timestep
            coords_target: Ground truth (optional)

        Returns:
            Dict of loss terms
        """
        losses = {}

        # Closure loss (always computed)
        closure_dist = (coords_clean[:, 0] - coords_clean[:, -1]).norm(dim=-1)
        losses["closure_loss"] = F.smooth_l1_loss(
            closure_dist, torch.full_like(closure_dist, self.bond_length)
        )

        # Bond consistency loss
        bond_dists = (coords_clean[:, 1:] - coords_clean[:, :-1]).norm(dim=-1)
        losses["bond_loss"] = F.smooth_l1_loss(
            bond_dists, torch.full_like(bond_dists, self.bond_length)
        )

        # Steric clash loss
        all_dists = torch.cdist(coords_clean, coords_clean)
        min_dist = 3.0
        eye = torch.eye(coords_clean.size(1), device=coords_clean.device)
        adj_mask = torch.zeros_like(eye)
        adj_mask[range(coords_clean.size(1) - 1), range(1, coords_clean.size(1))] = 1
        adj_mask[range(1, coords_clean.size(1)), range(coords_clean.size(1) - 1)] = 1
        clash_mask = (all_dists < min_dist) * (1 - eye.unsqueeze(0)) * (1 - adj_mask.unsqueeze(0))
        losses["clash_loss"] = (clash_mask * (min_dist - all_dists).clamp(max=0)).mean()

        # Target loss (if provided)
        if coords_target is not None:
            losses["coord_loss"] = F.mse_loss(coords_clean, coords_target)

            # FAPE loss (like AF3)
            losses["fape_loss"] = self._compute_fape(coords_clean, coords_target)

        return losses

    def _compute_fape(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        clamp: float = 10.0,
    ) -> torch.Tensor:
        """Frame Aligned Point Error (FAPE) from AF2/AF3."""
        B, L, _ = pred.shape

        # Center both structures
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)

        # Kabsch alignment
        H = torch.bmm(target_centered.transpose(1, 2), pred_centered)
        U, S, Vh = torch.linalg.svd(H)

        det = torch.det(torch.bmm(U, Vh))
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        Vh_adj = Vh * sign
        R = torch.bmm(U, Vh_adj)

        pred_aligned = torch.bmm(pred_centered, R.transpose(1, 2))

        dist = (pred_aligned - target_centered).norm(dim=-1)
        fape = dist.clamp(max=clamp).mean()

        return fape


class SimpleStructureHead(nn.Module):
    """
    Simple structure head for non-diffusion inference.

    When diffusion is too slow (inference), use this lightweight
    head for quick structure prediction.
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_coord: int = 64,
        n_rbf: int = 16,
    ):
        super().__init__()
        self.n_rbf = n_rbf

        # Distance prediction head
        self.dist_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, n_rbf),
        )

        # RBF centers
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.5, 30.0, n_rbf),
        )

        # Confidence head
        self.confidence = nn.Sequential(
            nn.Linear(d_pair, d_pair // 2),
            nn.GELU(),
            nn.Linear(d_pair // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: (B, L, L, d_pair)

        Returns:
            Dict with coords, dist_pred, confidence, closure_dist
        """
        B, L, _, d = pair_repr.shape

        # Predict distance distribution
        dist_logits = self.dist_head(pair_repr)  # (B, L, L, n_rbf)
        dist_probs = F.softmax(dist_logits, dim=-1)
        dist_pred = (dist_probs * self.rbf_centers).sum(dim=-1)  # (B, L, L)

        # Enforce symmetry
        dist_pred = 0.5 * (dist_pred + dist_pred.transpose(-1, -2))

        # Initialize coordinates via MDS
        coords = self._init_coords_mds(dist_pred)

        # Closure distance
        closure_dist = (coords[:, 0] - coords[:, -1]).norm(dim=-1)

        # Confidence
        pair_agg = pair_repr.mean(dim=2)
        confidence = self.confidence(pair_agg).squeeze(-1) * 100.0

        return {
            "coords": coords,
            "dist_pred": dist_pred,
            "confidence": confidence,
            "closure_dist": closure_dist,
        }

    def _init_coords_mds(self, dist_pred: torch.Tensor) -> torch.Tensor:
        """Initialize coordinates via MDS from predicted distances."""
        B, L = dist_pred.shape[:2]

        # Center distance matrix
        H = -0.5 * (dist_pred ** 2)
        row_mean = H.mean(dim=-1, keepdim=True)
        col_mean = H.mean(dim=-2, keepdim=True)
        grand_mean = H.mean(dim=(-1, -2), keepdim=True)
        B_mat = H - row_mean - col_mean + grand_mean

        # SVD for top 3 components
        try:
            U, S, Vh = torch.linalg.svd(B_mat)
            coords = U[:, :, :3] * torch.sqrt(S[:, :3].unsqueeze(1))
        except Exception:
            # Fallback: circle
            theta = torch.linspace(0, 2 * math.pi, L, device=dist_pred.device)[:-1]
            radius = L * 0.3
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)
            z = torch.zeros(L - 1, device=dist_pred.device)
            coords = torch.stack([x, y, z], dim=-1).unsqueeze(0).expand(B, -1, -1)

        return coords