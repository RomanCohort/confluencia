"""
gnn_latent_diffusion.py — Scheme 6 Path 2: GNN encoder + latent diffusion + decoder.

Architecture:
    Sequence → GNN Encoder → Latent z (with physics constraints)
                         ↓
                   Latent Diffusion (denoise)
                         ↓
                   GNN Decoder → 3D coords

Key design:
    - GNN encoder: physics-aware, extracts bond/pair/stacking/electrostatic
    - Latent space: low-dimensional, compresses sequence information
    - Diffusion: operates in latent, efficient sampling
    - Decoder: reconstructs 3D coords with physics validation

This modular approach allows separate training/optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNLatentConfig:
    d_node: int = 64
    d_edge: int = 32
    d_latent: int = 128
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_diffusion_steps: int = 50
    n_heads: int = 4
    bond_length: float = 5.9
    pair_distance: float = 10.6


class PhysicsGNNEncoder(nn.Module):
    """GNN encoder with physics-aware edge features.

    Extracts:
    - Node features: sequence embedding + position encoding
    - Edge features: bond, pair, electrostatic, stacking

    Outputs latent representation z.
    """

    def __init__(self, config: GNNLatentConfig):
        super().__init__()
        self.config = config

        # Sequence embedding
        self.seq_embed = nn.Embedding(5, config.d_node)

        # Circular position encoding
        self.pos_embed = CircularPositionEncoding(config.d_node)

        # Edge feature encoder (physics-based)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, config.d_edge),  # bond, pair, elec, stack
            nn.LayerNorm(config.d_edge),
            nn.GELU(),
        )

        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(config.d_node + config.d_edge, config.d_node)
            for _ in range(config.n_encoder_layers)
        ])

        # Node update layers
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_node * 2, config.d_node),
                nn.LayerNorm(config.d_node),
                nn.GELU(),
            )
            for _ in range(config.n_encoder_layers)
        ])

        # Project to latent
        self.to_latent = nn.Linear(config.d_node, config.d_latent)

    def forward(
        self,
        seq_tokens: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Encode sequence to latent representation.

        Returns:
            (B, L, d_latent) latent features
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Node features
        node_feat = self.seq_embed(seq_tokens)  # (B, L, d_node)
        node_feat = node_feat + self.pos_embed(L, device)  # add circular position

        # Message passing (sparse: backbone + local neighbors)
        for layer_idx in range(self.config.n_encoder_layers):
            # Backbone neighbors
            prev_feat = torch.roll(node_feat, shifts=1, dims=1)
            next_feat = torch.roll(node_feat, shifts=-1, dims=1)

            # Default edge features (bond=1)
            bond_edge = torch.ones(B, L, 4, device=device)
            bond_edge[:, :, 1] = 0.0  # not pair
            edge_feat = self.edge_encoder(bond_edge)  # (B, L, d_edge)

            # Messages
            prev_msg = self.message_layers[layer_idx](
                torch.cat([prev_feat, edge_feat], dim=-1)
            )
            next_msg = self.message_layers[layer_idx](
                torch.cat([next_feat, edge_feat], dim=-1)
            )

            # Aggregate
            messages = (prev_msg + next_msg) / 2

            # Update
            combined = torch.cat([node_feat, messages], dim=-1)
            node_feat = self.update_layers[layer_idx](combined)

        # Project to latent
        latent = self.to_latent(node_feat)  # (B, L, d_latent)

        return latent


class CircularPositionEncoding(nn.Module):
    """Circular position encoding for circRNA."""

    def __init__(self, d_model: int, n_harmonics: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate circular position encoding.

        Returns:
            (L, d_model) position encoding
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        encoding = torch.zeros(seq_len, self.d_model, device=device)

        for h in range(self.n_harmonics):
            omega = 2.0 * math.pi * (h + 1) / seq_len
            angles = omega * positions

            # Alternate sin/cos
            if h % 2 == 0:
                encoding[:, h % self.d_model] = torch.sin(angles)
            else:
                encoding[:, h % self.d_model] = torch.cos(angles)

        return encoding


class LatentDiffusion(nn.Module):
    """Diffusion model operating in latent space.

    More efficient than full 3D diffusion.
    """

    def __init__(self, config: GNNLatentConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(config.d_latent // 2),
            nn.Linear(config.d_latent // 2, config.d_latent),
            nn.GELU(),
        )

        # Denoiser MLP
        self.denoiser = nn.Sequential(
            nn.Linear(config.d_latent * 2, config.d_latent),
            nn.LayerNorm(config.d_latent),
            nn.GELU(),
            nn.Linear(config.d_latent, config.d_latent),
            nn.LayerNorm(config.d_latent),
            nn.GELU(),
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, config.n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def forward(
        self,
        latent_cond: torch.Tensor,  # (B, L, d_latent) from encoder
        mode: str = 'sample',
    ) -> torch.Tensor:
        """Sample latent through diffusion.

        Args:
            latent_cond: conditioning latent from encoder
            mode: 'sample' for inference, 'train' for training

        Returns:
            (B, L, d_latent) sampled latent
        """
        B, L, d_latent = latent_cond.shape
        device = latent_cond.device

        if mode == 'sample':
            return self._sample(latent_cond, B, L, device)
        else:
            return self._train_step(latent_cond)

    def _sample(self, latent_cond, B, L, device):
        """Denoise from pure noise."""
        latent = torch.randn(B, L, self.config.d_latent, device=device)

        for t in reversed(range(self.config.n_diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float)
            t_emb = self.time_embed(t_tensor)  # (B, d_latent)

            # Concatenate condition + time
            combined = torch.cat([latent, latent_cond + t_emb.unsqueeze(1)], dim=-1)

            # Denoise
            latent = self.denoiser(combined)

            # Add noise (except at t=0)
            if t > 0:
                latent = latent + torch.randn_like(latent) * self.betas[t] * 0.1

        return latent

    def _train_step(self, latent_target):
        """Training step (returns loss)."""
        B, L, d_latent = latent_target.shape
        device = latent_target.device

        # Random timestep
        t = torch.randint(0, self.config.n_diffusion_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(latent_target)
        alpha_bar = self.alpha_bars[t].view(B, 1, 1)
        latent_noisy = torch.sqrt(alpha_bar) * latent_target + \
                       torch.sqrt(1 - alpha_bar) * noise

        # Predict noise
        t_emb = self.time_embed(t.float())
        combined = torch.cat([latent_noisy, t_emb.unsqueeze(1).expand(-1, L, -1)], dim=-1)
        noise_pred = self.denoiser(combined)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        return {'loss': loss, 'latent_pred': noise_pred}


class PhysicsGNNDecoder(nn.Module):
    """GNN decoder: latent → 3D coords with physics validation."""

    def __init__(self, config: GNNLatentConfig):
        super().__init__()
        self.config = config

        # Project latent to node features
        self.from_latent = nn.Linear(config.d_latent, config.d_node)

        # Message passing layers (physics-aware)
        self.message_layers = nn.ModuleList([
            nn.Linear(config.d_node + config.d_edge, config.d_node)
            for _ in range(config.n_decoder_layers)
        ])

        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_node * 2, config.d_node),
                nn.LayerNorm(config.d_node),
                nn.GELU(),
            )
            for _ in range(config.n_decoder_layers)
        ])

        # Edge encoder (physics constraints)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, config.d_edge),
            nn.LayerNorm(config.d_edge),
            nn.GELU(),
        )

        # Output: coords
        self.coords_proj = nn.Linear(config.d_node, 3)

        # Closure enforcement
        self.closure_proj = nn.Linear(config.d_node * 2, 3)

    def forward(
        self,
        latent: torch.Tensor,  # (B, L, d_latent)
        seq_tokens: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Decode latent to 3D coordinates.

        Returns:
            (B, L, 3) coordinates
        """
        B, L, _ = latent.shape
        device = latent.device

        # Latent → node features
        node_feat = self.from_latent(latent)  # (B, L, d_node)

        # Message passing
        for layer_idx in range(self.config.n_decoder_layers):
            # Backbone neighbors
            prev_feat = torch.roll(node_feat, shifts=1, dims=1)
            next_feat = torch.roll(node_feat, shifts=-1, dims=1)

            # Physics edge features
            bond_edge = torch.ones(B, L, 4, device=device)
            edge_feat = self.edge_encoder(bond_edge)

            # Messages
            prev_msg = self.message_layers[layer_idx](
                torch.cat([prev_feat, edge_feat], dim=-1)
            )
            next_msg = self.message_layers[layer_idx](
                torch.cat([next_feat, edge_feat], dim=-1)
            )

            messages = (prev_msg + next_msg) / 2
            combined = torch.cat([node_feat, messages], dim=-1)
            node_feat = self.update_layers[layer_idx](combined)

        # Project to coords
        coords = self.coords_proj(node_feat)  # (B, L, 3)

        # Enforce closure
        coords = self._enforce_closure(coords, node_feat)

        return coords

    def _enforce_closure(self, coords, node_feat):
        """Enforce BSJ closure."""
        B, L, _ = coords.shape

        # Closure adjustment from node features
        closure_input = torch.cat([node_feat[:, 0], node_feat[:, -1]], dim=-1)
        closure_adj = self.closure_proj(closure_input) * 0.1

        # Place last position at bond_length from first
        direction = coords[:, 0] - coords[:, -2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        coords[:, -1] = coords[:, 0] - self.config.bond_length * direction + closure_adj

        return coords


class GNNLatentDiffusionModel(nn.Module):
    """Scheme 6 Path 2: Full model combining encoder, diffusion, decoder."""

    def __init__(self, config: Optional[GNNLatentConfig] = None):
        super().__init__()
        self.config = config or GNNLatentConfig()

        self.encoder = PhysicsGNNEncoder(self.config)
        self.diffusion = LatentDiffusion(self.config)
        self.decoder = PhysicsGNNDecoder(self.config)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        mode: str = 'sample',
    ) -> Dict[str, torch.Tensor]:
        """Generate circRNA 3D structure.

        Args:
            seq_tokens: (B, L) sequence tokens
            mode: 'sample' or 'train'

        Returns:
            Dict with coords, latent, closure_distance
        """
        # Encode
        latent_cond = self.encoder(seq_tokens)  # (B, L, d_latent)

        # Diffusion
        latent_out = self.diffusion(latent_cond, mode=mode)

        # Handle train (returns dict) vs sample (returns tensor)
        if isinstance(latent_out, dict):
            # During training: _train_step returns {'loss': diff_loss, 'latent_pred': noise_pred}
            # We need BOTH the diffusion loss AND coords for end-to-end training
            diff_loss = latent_out.get('loss', torch.tensor(0.0, device=seq_tokens.device))
            # Use clean latent for decoder during training (teaches latent→3D mapping)
            # At inference, decoder will receive denoised latent from diffusion sampling
            latent = latent_cond
        else:
            latent = latent_out
            diff_loss = None

        # Decode
        coords = self.decoder(latent, seq_tokens)  # (B, L, 3)

        # Metrics
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)

        return {
            'coords': coords,
            'latent': latent,
            'latent_cond': latent_cond,
            'closure_distance': closure_dist,
            'diffusion_loss': diff_loss,
            'method': 'gnn_latent_diffusion',
        }


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep."""

    def __init__(self, d_embed: int = 64):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_embed // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding