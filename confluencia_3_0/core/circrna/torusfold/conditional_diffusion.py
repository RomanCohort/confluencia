"""
conditional_diffusion.py — CircRNA-conditional diffusion for 3D structure.

Scheme 4: Diffusion model with circRNA-specific conditioning.

Conditions:
- Sequence (ACGU) → ESM2/RNA-FM embedding
- Secondary structure → ViennaRNA pseudo-labels
- Circular topology → TPE encoding + BSJ closure constraint
- Pair constraints → from CircPairformer

The diffusion generates 3D coordinates satisfying all constraints.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import to avoid circular dependency
# from .experimental_conditions import ExperimentalConditions, ConditionEncoder


class CircRNADiffusion(nn.Module):
    """Diffusion model with circRNA-specific conditioning.

    Architecture:
        Input: sequence, pair_repr, secondary_structure (optional)
        ↓
        ConditionEncoder: sequence + ss + topology → conditioning
        ↓
        DiffusionDenoiser: noisy_coords + conditioning → clean_coords
        ↓
        Output: 3D coordinates satisfying circular closure
    """

    def __init__(
        self,
        d_model: int = 256,
        d_pair: int = 128,
        d_time: int = 64,
        n_diffusion_steps: int = 100,
        n_denoiser_layers: int = 4,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.n_steps = n_diffusion_steps
        self.bond_length = bond_length

        # Sequence encoder (lightweight, or use frozen ESM2)
        self.seq_embed = nn.Embedding(5, d_model)  # A,C,G,U,unk

        # Secondary structure encoder (dot-bracket → features)
        self.ss_embed = nn.Embedding(3, d_model // 4)  # (, ), .

        # Circular topology encoder (TPE-like)
        self.topo_embed = CircTopologyEncoder(d_model)

        # Pair representation projection
        self.pair_proj = nn.Linear(d_pair, d_model, bias=False)

        # Experimental condition encoder (pH, Mg²⁺, Na⁺, temperature)
        self.cond_proj = nn.Linear(d_model, d_model)  # Projects condition embedding

        # Combine all conditions (now includes experimental conditions)
        self.condition_merger = nn.Sequential(
            nn.Linear(d_model * 2 + d_model // 4 + d_model, d_model),  # +d_model for exp conditions
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_time),
            nn.Linear(d_time, d_model),
            nn.GELU(),
        )

        # Denoiser
        self.denoiser = CircDenoiser(d_model, n_layers=n_denoiser_layers)

        # Closure predictor
        self.closure_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Noise schedule
        self.register_buffer(
            'betas',
            torch.linspace(1e-4, 0.02, n_diffusion_steps)
        )
        self.register_buffer(
            'alphas',
            1.0 - self.betas
        )
        self.register_buffer(
            'alpha_bars',
            torch.cumprod(self.alphas, dim=0)
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,      # (B, L) tokenized sequence
        pair_repr: torch.Tensor,        # (B, L, L, d_pair)
        ss_tokens: Optional[torch.Tensor] = None,  # (B, L) secondary structure
        coords_target: Optional[torch.Tensor] = None,  # (B, L, 3) for training
        exp_condition_emb: Optional[torch.Tensor] = None,  # (B, d_model) experimental conditions
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        If coords_target provided: train diffusion
        Else: sample new structure

        Args:
            exp_condition_emb: (B, d_model) from ConditionEncoder, injects
                experimental conditions (pH, Mg²⁺, temperature, ionic strength)
                into the diffusion process as classifier-free guidance.
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Encode conditions
        seq_emb = self.seq_embed(seq_tokens)  # (B, L, d_model)

        ss_emb = torch.zeros(B, L, self.d_model // 4, device=device)
        if ss_tokens is not None:
            ss_emb = self.ss_embed(ss_tokens)

        topo_emb = self.topo_embed(L, device)  # (L, d_model)
        topo_emb = topo_emb.unsqueeze(0).expand(B, -1, -1)

        # Experimental condition embedding (pH, Mg²⁺, temperature)
        exp_emb = torch.zeros(B, L, self.d_model, device=device)
        if exp_condition_emb is not None:
            # Project and broadcast to all positions
            exp_projected = self.cond_proj(exp_condition_emb)  # (B, d_model)
            exp_emb = exp_projected.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_model)

        # Merge conditions (now includes experimental conditions)
        cond_input = torch.cat([seq_emb, topo_emb, ss_emb, exp_emb], dim=-1)
        cond = self.condition_merger(cond_input)  # (B, L, d_model)

        # Add pair representation
        pair_agg = pair_repr.mean(dim=2)  # (B, L, d_pair)
        cond = cond + self.pair_proj(pair_agg)

        if coords_target is not None:
            # Training: add noise and denoise
            return self._train_step(coords_target, cond)
        else:
            # Inference: sample from noise
            return self._sample(B, L, cond, device)

    def _train_step(
        self,
        coords_target: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Single training step with random timestep."""
        B, L, _ = coords_target.shape
        device = coords_target.device

        # Sample random timestep
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(coords_target)
        alpha_bar = self.alpha_bars[t].view(B, 1, 1)
        coords_noisy = torch.sqrt(alpha_bar) * coords_target + \
                       torch.sqrt(1 - alpha_bar) * noise

        # Time embedding
        t_emb = self.time_embed(t.float())  # (B, d_model)

        # Denoise
        coords_pred = self.denoiser(coords_noisy, cond, t_emb)

        # Losses
        noise_loss = F.mse_loss(coords_pred, noise)

        # Closure loss (first and last should be ~bond_length apart)
        closure_dist = torch.norm(coords_pred[:, 0] - coords_pred[:, -1], dim=-1)
        closure_target = torch.full_like(closure_dist, self.bond_length)
        closure_loss = F.mse_loss(closure_dist, closure_target)

        return {
            'noise_loss': noise_loss,
            'closure_loss': closure_loss,
            'total_loss': noise_loss + 0.1 * closure_loss,
        }

    def _sample(
        self,
        B: int,
        L: int,
        cond: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.5,
    ) -> Dict[str, torch.Tensor]:
        """Sample from diffusion model with classifier-free guidance.

        Args:
            guidance_scale: >1.0 amplifies condition influence.
                1.0 = standard conditional sampling
                0.0 = unconditional sampling
                3.0+ = strong condition guidance (more pH/Mg/T effect)
        """
        # Start from pure noise
        coords = torch.randn(B, L, 3, device=device)

        # Iteratively denoise
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float)
            t_emb = self.time_embed(t_tensor)

            if guidance_scale != 1.0 and self.training:
                # Classifier-free guidance: interpolate conditional and unconditional
                noise_cond = self.denoiser(coords, cond, t_emb)

                # Unconditional (zero out condition)
                cond_null = torch.zeros_like(cond)
                noise_uncond = self.denoiser(coords, cond_null, t_emb)

                # Guided prediction
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard conditional sampling
                noise_pred = self.denoiser(coords, cond, t_emb)

            # Denoise step
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            if t > 0:
                noise = torch.randn_like(coords)
                sigma = self.betas[t] ** 0.5
            else:
                noise = 0
                sigma = 0

            coords = (1 / alpha.sqrt()) * (coords - \
                     (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) + \
                     sigma * noise

        # Enforce circular closure (post-hoc)
        coords = self._enforce_closure(coords)

        return {'coords': coords}

    def _enforce_closure(self, coords: torch.Tensor) -> torch.Tensor:
        """Post-hoc enforcement of circular closure."""
        # Move last residue to be bond_length away from first
        B, L, _ = coords.shape

        # Direction from second-to-last to first
        direction = coords[:, 0] - coords[:, -2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Place last residue at bond_length from first
        coords[:, -1] = coords[:, 0] - self.bond_length * direction

        return coords


class CircTopologyEncoder(nn.Module):
    """Encode circular topology for diffusion conditioning.

    Similar to TPE but designed for diffusion conditioning.
    """

    def __init__(self, d_model: int, n_harmonics: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics

        self.harmonic_weights = nn.Parameter(
            torch.randn(n_harmonics, d_model // 2) * 0.02
        )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate topology encoding.

        Returns:
            (L, d_model) topology encoding
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        encoding = torch.zeros(seq_len, self.d_model, device=device)

        for h in range(self.n_harmonics):
            omega = 2.0 * math.pi * (h + 1) / seq_len
            angles = omega * positions

            encoding[:, 0::2] += torch.outer(
                torch.sin(angles), self.harmonic_weights[h]
            )
            encoding[:, 1::2] += torch.outer(
                torch.cos(angles), self.harmonic_weights[h]
            )

        return encoding


class CircDenoiser(nn.Module):
    """Denoiser for circRNA diffusion.

    Takes noisy coords + conditions, predicts noise.
    """

    def __init__(self, d_model: int, n_layers: int = 4):
        super().__init__()

        self.coord_proj = nn.Linear(3, d_model)

        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(d_model * 2, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
            ])
        self.layers = nn.Sequential(*layers)

        self.out_proj = nn.Linear(d_model * 2, 3)

    def forward(
        self,
        coords: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise.

        Args:
            coords: (B, L, 3) noisy coordinates
            cond: (B, L, d_model) conditioning
            t_emb: (B, d_model) time embedding

        Returns:
            (B, L, 3) predicted noise
        """
        coord_feat = self.coord_proj(coords)  # (B, L, d_model)

        # Add time embedding to each position
        t_expanded = t_emb.unsqueeze(1).expand(-1, coords.shape[1], -1)
        x = torch.cat([coord_feat, cond + t_expanded], dim=-1)

        x = self.layers(x)
        noise_pred = self.out_proj(x)

        return noise_pred


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep."""

    def __init__(self, d_time: int = 64):
        super().__init__()
        self.d_time = d_time

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_time // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding
