"""
diffusion_gnn_hybrid.py — Diffusion + GNN hybrid for circRNA 3D structure.

Scheme 6: Experimental architecture combining diffusion and GNN.

Architecture:
    GNN Encoder: sequence + physics constraints → latent conditioning
    Diffusion: latent noise → latent structure
    GNN Decoder: latent structure → 3D coords (physics-valid)

This hybrid aims to:
1. Use GNN for physics-aware encoding (bond, pair, clash as edge features)
2. Use diffusion for flexible generation
3. Use GNN decoder for physics validation
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGNN(nn.Module):
    """GNN with physics constraints as edge features.

    For circRNA, we use a sparse graph:
    - Nodes: nucleotide positions
    - Edges: backbone bonds (adjacent), pair constraints, local neighbors

    Edge features encode physics:
    - Bond length target
    - Pair distance target
    - Electrostatic repulsion
    - Stacking energy
    """

    def __init__(
        self,
        d_node: int = 64,
        d_edge: int = 32,
        n_layers: int = 4,
        bond_length: float = 5.9,
        pair_distance: float = 10.6,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.bond_length = bond_length
        self.pair_distance = pair_distance

        # Node encoder (from sequence)
        self.node_embed = nn.Embedding(5, d_node)

        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(d_node + d_edge, d_node) for _ in range(n_layers)
        ])

        # Node update layers
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_node * 2, d_node),
                nn.LayerNorm(d_node),
                nn.GELU(),
            ) for _ in range(n_layers)
        ])

        # Edge feature encoder (physics-based)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, d_edge),  # bond, pair, elec, stack
            nn.LayerNorm(d_edge),
            nn.GELU(),
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GNN forward pass.

        Args:
            seq_tokens: (B, L) sequence tokens
            coords: (B, L, 3) coordinates (for computing actual distances)
            pair_probs: (B, L, L) pair probabilities

        Returns:
            (B, L, d_node) node embeddings with physics awareness
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Initialize node features from sequence
        node_feat = self.node_embed(seq_tokens)  # (B, L, d_node)

        # Build sparse edge list
        # Backbone edges: (i, i+1) for all i
        # Pair edges: (i, j) where pair_probs > threshold

        # Message passing
        for layer_idx in range(len(self.message_layers)):
            # Aggregate messages from neighbors
            messages = self._aggregate_messages(
                node_feat, coords, pair_probs, layer_idx, device
            )

            # Update node features
            combined = torch.cat([node_feat, messages], dim=-1)
            node_feat = self.update_layers[layer_idx](combined)

        return node_feat

    def _aggregate_messages(
        self,
        node_feat: torch.Tensor,
        coords: Optional[torch.Tensor],
        pair_probs: Optional[torch.Tensor],
        layer_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Aggregate messages from neighboring nodes."""
        B, L, d_node = node_feat.shape

        # Backbone messages (always exist)
        # For each position, messages from i-1 and i+1
        prev_feat = torch.roll(node_feat, shifts=1, dims=1)  # (B, L, d_node)
        next_feat = torch.roll(node_feat, shifts=-1, dims=1)

        # Edge features for backbone
        bond_features = torch.ones(B, L, 4, device=device) * 0.5  # bond=0.5
        bond_features[:, :, 0] = 1.0  # bond edge type

        if coords is not None:
            # Actual bond distances
            bond_dist = torch.norm(
                coords - torch.roll(coords, shifts=1, dims=1), dim=-1
            )  # (B, L)
            bond_features[:, :, 1] = (bond_dist - self.bond_length) / self.bond_length

        bond_edge_feat = self.edge_encoder(bond_features)  # (B, L, d_edge)

        # Messages from backbone neighbors
        prev_msg = self.message_layers[layer_idx](
            torch.cat([prev_feat, bond_edge_feat], dim=-1)
        )
        next_msg = self.message_layers[layer_idx](
            torch.cat([next_feat, bond_edge_feat], dim=-1)
        )

        messages = (prev_msg + next_msg) / 2

        # Optional: pair constraint messages (if pair_probs provided)
        if pair_probs is not None and layer_idx > 0:
            # Top pair edges for each position
            pair_threshold = 0.3
            for i in range(L):
                for j in range(L):
                    if pair_probs[0, i, j] > pair_threshold and abs(i - j) > 3:
                        # Compute pair edge feature
                        pair_feat = torch.zeros(4, device=device)
                        pair_feat[0] = 0.0  # not bond
                        pair_feat[1] = 1.0  # pair edge type

                        if coords is not None:
                            pair_dist = torch.norm(
                                coords[:, i] - coords[:, j], dim=-1
                            )
                            pair_feat[2] = (pair_dist - self.pair_distance) / self.pair_distance

                        pair_edge = self.edge_encoder(pair_feat.unsqueeze(0).unsqueeze(0))

                        # Add message from paired position
                        pair_msg = self.message_layers[layer_idx](
                            torch.cat([node_feat[:, j], pair_edge.squeeze(0)], dim=-1)
                        )
                        weight = pair_probs[0, i, j]
                        messages[:, i] += weight * pair_msg.squeeze(0)

        return messages


class DiffusionGNNHybrid(nn.Module):
    """Scheme 6: Diffusion + GNN hybrid for circRNA 3D structure.

    Pipeline:
        1. GNN Encoder: physics-aware encoding
        2. Latent Diffusion: generate latent structure
        3. GNN Decoder: decode to physics-valid 3D coords
    """

    def __init__(
        self,
        d_node: int = 64,
        d_latent: int = 128,
        n_diffusion_steps: int = 50,
        n_gnn_layers: int = 4,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_latent = d_latent
        self.n_steps = n_diffusion_steps
        self.bond_length = bond_length

        # GNN Encoder (physics-aware)
        self.gnn_encoder = PhysicsGNN(
            d_node=d_node,
            n_layers=n_gnn_layers,
            bond_length=bond_length,
        )

        # Project to latent space
        self.to_latent = nn.Linear(d_node, d_latent)

        # Latent diffusion
        self.latent_diffusion = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

        # GNN Decoder
        self.gnn_decoder = PhysicsGNN(
            d_node=d_node,
            n_layers=n_gnn_layers,
            bond_length=bond_length,
        )

        # Project to coordinates
        self.to_coords = nn.Linear(d_node, 3)

        # Closure enforcement
        self.closure_layer = nn.Linear(d_node * 2, 3)

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate circRNA 3D structure.

        Args:
            seq_tokens: (B, L) sequence tokens
            pair_probs: (B, L, L) pair probabilities from CircPairformer

        Returns:
            Dict with coords, latent, closure_distance
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 1. GNN Encoder: physics-aware encoding
        node_feat = self.gnn_encoder(seq_tokens, coords=None, pair_probs=pair_probs)

        # 2. Project to latent space
        latent = self.to_latent(node_feat)  # (B, L, d_latent)

        # 3. Latent diffusion sampling
        latent = self._latent_diffusion_sample(latent, B, L, device)

        # 4. GNN Decoder: decode with physics validation
        # Initialize coords from latent
        coords_init = self.to_coords(self.gnn_decoder(
            seq_tokens,
            coords=self.to_coords(node_feat),  # Initial estimate
            pair_probs=pair_probs,
        ))

        # 5. Refine with decoder
        node_feat_decoded = self.gnn_decoder(
            seq_tokens,
            coords=coords_init,
            pair_probs=pair_probs,
        )
        coords = self.to_coords(node_feat_decoded)

        # 6. Enforce circular closure
        coords = self._enforce_closure(coords, node_feat_decoded)

        # Compute closure distance
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)

        return {
            'coords': coords,
            'latent': latent,
            'closure_distance': closure_dist,
            'method': 'diffusion_gnn_hybrid',
        }

    def _latent_diffusion_sample(
        self,
        latent_cond: torch.Tensor,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample latent through diffusion."""
        # Start from noise
        latent = torch.randn(B, L, self.d_latent, device=device)

        # Denoise
        for t in reversed(range(self.n_steps)):
            # Simple denoising step
            alpha_bar = self.alpha_bars[t]
            latent = latent * alpha_bar.sqrt() + latent_cond * (1 - alpha_bar).sqrt()

            latent = self.latent_diffusion(latent)

        return latent

    def _enforce_closure(
        self,
        coords: torch.Tensor,
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce BSJ closure using node features."""
        B, L, _ = coords.shape

        # Use first and last node features to predict closure adjustment
        closure_input = torch.cat([node_feat[:, 0], node_feat[:, -1]], dim=-1)
        closure_adj = self.closure_layer(closure_input)  # (B, 3)

        # Adjust last position to be bond_length from first
        direction = coords[:, 0] - coords[:, -2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        coords[:, -1] = coords[:, 0] - self.bond_length * direction + closure_adj * 0.1

        return coords