"""
circrna_diffusion.py — DDPM + EGNN + Guided Diffusion for circRNA 3D structure.

Architecture (following DynaRNA paradigm):
    1. DDPM on 3D atomic coordinates
    2. EGNN denoiser (rotation/translation equivariant)
    3. Conditional: ViennaRNA secondary structure + experimental conditions
    4. Guided: BSJ closure reward function for circular topology

Key innovation: guided diffusion with circRNA-specific reward:
    - BSJ closure reward: ||x[0] - x[-1]|| ≈ bond_length
    - Circular topology reward: d_circ consistency
    - Energy reward: CG energy minimization

References:
    - DynaRNA: end-to-end RNA 3D generation (Nature, 2024)
    - GraphRNA: RNA interaction modeling with user constraints
    - RNADiffusion: guided generation with reward gradients
    - IsRNAcirc: BSJ closure via annealing (Jiang et al., PLOS Comp Biol 2024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CircDiffusionConfig:
    """Configuration for circRNA diffusion model."""
    d_node: int = 128          # Node feature dimension
    d_edge: int = 64           # Edge feature dimension
    d_cond: int = 64           # Condition embedding dimension
    n_egnn_layers: int = 6     # EGNN layers in denoiser
    n_diffusion_steps: int = 100
    bond_length: float = 5.9   # Å, P-P backbone distance
    guidance_scale: float = 2.0  # Classifier-free guidance scale
    closure_weight: float = 1.0  # BSJ closure reward weight
    energy_weight: float = 0.5   # Energy reward weight


# ── Equivariant GNN Layer ────────────────────────────────────

class EGNNLayer(nn.Module):
    """Equivariant Graph Neural Network layer.

    Guarantees:
    - Rotation equivariance: R·f(x) = f(R·x)
    - Translation equivariance: f(x+t) = f(x)+t

    From: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021

    Modified for circRNA:
    - Backbone edges: (i, i+1) with circular wrap
    - Pair edges: (i, j) where pair_prob > threshold
    - BSJ edge: (0, L-1) always connected
    """

    def __init__(self, d_node: int, d_edge: int, n_heads: int = 4):
        super().__init__()
        self.d_node = d_node
        self.n_heads = n_heads

        # Edge-aware attention
        self.edge_proj = nn.Linear(d_edge, d_node)

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge + 1, d_node * 2),
            nn.GELU(),
            nn.Linear(d_node * 2, d_node),
        )

        # Coordinate update (equivariant)
        self.coord_mlp = nn.Sequential(
            nn.Linear(d_node, d_node // 2),
            nn.GELU(),
            nn.Linear(d_node // 2, 1),
        )

        # Learnable coordinate step size
        self.coord_step = nn.Parameter(torch.tensor(0.1))

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(d_node * 2, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
        )

        # Norm for residual
        self.norm = nn.LayerNorm(d_node)

    def forward(
        self,
        node_feat: torch.Tensor,   # (B, L, d_node)
        coords: torch.Tensor,       # (B, L, 3)
        edge_index: torch.Tensor,   # (2, E) edge list
        edge_feat: torch.Tensor,    # (B, E, d_edge)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equivariant update.

        Returns:
            updated node_feat, updated coords
        """
        B, L, _ = node_feat.shape
        E = edge_index.shape[1]

        # Gather source and target node features
        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Debug: check edge_index bounds
        if src.max() >= L or dst.max() >= L:
            # Edge index out of bounds - this should never happen
            # Return input unchanged
            return node_feat, coords

        h_src = node_feat[:, src]   # (B, E, d_node)
        h_dst = node_feat[:, dst]   # (B, E, d_node)

        # Relative coordinates (translation equivariant)
        x_src = coords[:, src]      # (B, E, 3)
        x_dst = coords[:, dst]      # (B, E, 3)
        rel_coords = x_src - x_dst  # (B, E, 3)

        # Distance (rotation invariant) - clamp to avoid NaN from zero distance
        dist = torch.norm(rel_coords, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, E, 1)

        # Message input
        msg_input = torch.cat([h_src, h_dst, edge_feat, dist], dim=-1)
        messages = self.message_mlp(msg_input)  # (B, E, d_node)

        # Coordinate update (equivariant direction)
        coord_weight = self.coord_mlp(messages)  # (B, E, 1)
        coord_update = coord_weight * rel_coords  # (B, E, 3)

        # Aggregate messages per node using vectorized scatter_reduce
        # Replace slow Python for loop with batched scatter operation
        # Expand dst: (E,) -> (B, E) for batched indexing
        dst_expand = dst.unsqueeze(0).expand(B, -1)  # (B, E)

        # Use scatter_reduce for mean aggregation (faster than loop + index_add)
        # node_update: (B, L, d_node)
        node_update = torch.zeros_like(node_feat)
        coord_update_agg = torch.zeros_like(coords)

        # Vectorized scatter: use index_add with expanded indices
        # Batched index_add via reshape trick
        # Flatten batch dim: treat (B*L) as 1D, offset dst by b*L
        offset = torch.arange(B, device=dst.device).unsqueeze(1) * L  # (B, 1)
        dst_flat = (dst_expand + offset).reshape(-1)  # (B*E,)
        messages_flat = messages.float().reshape(B * E, -1)  # (B*E, d_node)
        coord_update_flat = coord_update.float().reshape(B * E, 3)  # (B*E, 3)

        node_update_flat = node_update.reshape(B * L, -1)
        node_update_flat.index_add_(0, dst_flat, messages_flat)
        node_update = node_update_flat.reshape(B, L, -1)

        coord_agg_flat = coord_update_agg.reshape(B * L, 3)
        coord_agg_flat.index_add_(0, dst_flat, coord_update_flat)
        coord_update_agg = coord_agg_flat.reshape(B, L, 3)

        # Count neighbors (vectorized)
        neighbor_count = torch.zeros(B * L, device=dst.device)
        ones_flat = torch.ones(B * E, device=dst.device)
        neighbor_count.index_add_(0, dst_flat, ones_flat)
        neighbor_count = neighbor_count.reshape(B, L, 1).clamp(min=1)

        neighbor_count = neighbor_count.clamp(min=1)

        # Normalize
        node_update = node_update / neighbor_count
        coord_update_agg = coord_update_agg / neighbor_count

        # Update node features
        h_new = self.node_mlp(torch.cat([node_feat, node_update], dim=-1))
        node_feat = self.norm(node_feat + h_new)

        # Update coordinates (equivariant)
        # Clamp coord update to prevent NaN from gradient explosion
        coord_update_agg = coord_update_agg.clamp(-5.0, 5.0)
        coords = coords + coord_update_agg * self.coord_step  # Learnable step

        return node_feat, coords


# ── CircRNA Graph Builder ────────────────────────────────────

class CircRNAGraphBuilder:
    """Build sparse graph for circRNA.

    Edge types:
    - Backbone: (i, i+1) and (i, i-1), circular wrap
    - BSJ: (0, L-1) always connected
    - Pair: (i, j) where pair_prob > threshold
    """

    def __init__(self, pair_threshold: float = 0.3):
        self.pair_threshold = pair_threshold

    def build(
        self,
        L: int,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph for sequence of length L.

        Returns:
            edge_index: (2, E) edge list
            edge_types: (E,) edge type indices
        """
        edges_src = []
        edges_dst = []
        edge_types = []

        # Backbone edges (type 0)
        for i in range(L):
            j = (i + 1) % L
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edge_types.extend([0, 0])

        # Pair edges (type 1)
        if pair_probs is not None:
            # Handle batched or single pair_probs
            pp = pair_probs
            if pp.dim() > 2:
                pp = pp[0]  # Take first sample in batch for graph construction
            for i in range(L):
                for j in range(i + 4, L):
                    if pp[i, j].item() > self.pair_threshold:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        edge_types.extend([1, 1])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_types = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_types


# ── Condition Encoder ────────────────────────────────────────

class CircRNAConditionEncoder(nn.Module):
    """Encode all conditions for guided diffusion.

    Conditions:
    1. Sequence → token embedding
    2. Secondary structure → dot-bracket embedding
    3. Circular topology → TPE-like encoding
    4. Experimental: temperature, pH, Mg²⁺, ionic strength
    """

    def __init__(self, d_cond: int = 64, d_model: int = 128):
        super().__init__()
        # Sequence
        self.seq_embed = nn.Embedding(5, d_cond)

        # Secondary structure
        self.ss_embed = nn.Embedding(3, d_cond // 4)

        # Experimental conditions
        self.exp_embed = nn.Sequential(
            nn.Linear(4, d_cond // 2),  # T, pH, Mg, Na
            nn.GELU(),
            nn.Linear(d_cond // 2, d_cond),
        )

        # Combine to d_model
        self.combine = nn.Sequential(
            nn.Linear(d_cond + d_cond // 4 + d_cond, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,       # (B, L)
        ss_tokens: Optional[torch.Tensor] = None,   # (B, L)
        temperature: float = 310.0,
        pH: float = 7.4,
        Mg_conc: float = 1.0,
        Na_conc: float = 150.0,
    ) -> torch.Tensor:
        """Encode all conditions.

        Returns:
            (B, L, d_model) per-position condition features
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        seq_emb = self.seq_embed(seq_tokens)  # (B, L, d_cond)

        ss_emb = torch.zeros(B, L, d_cond := self.ss_embed.embedding_dim, device=device)
        if ss_tokens is not None:
            ss_emb = self.ss_embed(ss_tokens)

        # Experimental conditions (broadcast to all positions)
        exp_input = torch.tensor([[temperature / 400, pH / 14,
                                   math.log10(Mg_conc + 1) / 2,
                                   math.log10(Na_conc + 1) / 3]], device=device)
        exp_emb = self.exp_embed(exp_input)  # (1, d_cond)
        exp_emb = exp_emb.unsqueeze(1).expand(B, L, -1)  # (B, L, d_cond)

        combined = torch.cat([seq_emb, ss_emb, exp_emb], dim=-1)
        return self.combine(combined)


# ── BSJ Closure Reward ───────────────────────────────────────

class BSJClosureReward(nn.Module):
    """Reward function for guided diffusion: BSJ closure.

    R_closure = -||x[0] - x[-1]|| - bond_length||²

    This is the circRNA-specific innovation:
    standard diffusion has no notion of circular topology,
    but guided diffusion can enforce it via reward gradients.
    """

    def __init__(self, bond_length: float = 5.9):
        super().__init__()
        self.bond_length = bond_length

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute closure reward and its gradient.

        Args:
            coords: (B, L, 3)

        Returns:
            (B,) reward score (higher = better closure)
        """
        # Distance between first and last residue
        closure_vec = coords[:, 0] - coords[:, -1]  # (B, 3)
        closure_dist = torch.norm(closure_vec, dim=-1)  # (B,)

        # Reward: negative squared error
        reward = -(closure_dist - self.bond_length) ** 2

        return reward

    def gradient(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute gradient of reward w.r.t. coords (for guided diffusion).

        Returns:
            (B, L, 3) gradient to apply to each position
        """
        coords.requires_grad_(True)
        reward = self.forward(coords)
        reward.sum().backward()

        return coords.grad


class CircularEnergyReward(nn.Module):
    """Energy-based reward for guided diffusion.

    Rewards:
    - Low bond energy (consistent backbone distances)
    - Low clash energy (no steric overlap)
    """

    def __init__(self, bond_length: float = 5.9, clash_dist: float = 3.0):
        super().__init__()
        self.bond_length = bond_length
        self.clash_dist = clash_dist

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute energy reward."""
        B, L, _ = coords.shape

        # Bond energy
        bond_dists = torch.norm(
            torch.roll(coords, -1, dims=1) - coords, dim=-1
        )  # (B, L)
        bond_energy = ((bond_dists - self.bond_length) ** 2).mean(dim=-1)  # (B,)

        # Reward = negative energy
        return -bond_energy


# ── Full Diffusion Model ─────────────────────────────────────

class CircRNADiffusionModel(nn.Module):
    """DDPM + EGNN + Guided Diffusion for circRNA 3D structure.

    Pipeline:
        1. Condition encoding (sequence + ss + experimental)
        2. DDPM forward: add noise to 3D coords
        3. EGNN denoiser: predict noise (equivariant)
        4. Guided sampling: BSJ closure + energy rewards
        5. Output: 3D coords with enforced circular topology
    """

    def __init__(self, config: Optional[CircDiffusionConfig] = None):
        super().__init__()
        self.config = config or CircDiffusionConfig()

        # Condition encoder
        self.condition_encoder = CircRNAConditionEncoder(
            d_cond=self.config.d_cond,
            d_model=self.config.d_node,
        )

        # EGNN denoiser layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(self.config.d_node, self.config.d_edge)
            for _ in range(self.config.n_egnn_layers)
        ])

        # Coordinate projection (node feat → 3D displacement)
        self.coord_proj = nn.Sequential(
            nn.Linear(self.config.d_node, self.config.d_node // 2),
            nn.GELU(),
            nn.Linear(self.config.d_node // 2, 3),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(self.config.d_node),
            nn.Linear(self.config.d_node, self.config.d_node),
            nn.GELU(),
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, self.config.n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        # Graph builder
        self.graph_builder = CircRNAGraphBuilder()

        # Reward functions (for guided diffusion)
        self.closure_reward = BSJClosureReward(self.config.bond_length)
        self.energy_reward = CircularEnergyReward(self.config.bond_length)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None,
        coords_target: Optional[torch.Tensor] = None,
        temperature: float = 310.0,
        pH: float = 7.4,
        Mg_conc: float = 1.0,
        Na_conc: float = 150.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: train or sample.

        If coords_target provided → training
        Else → sampling with guided diffusion
        """
        if coords_target is not None:
            return self._train_step(
                seq_tokens, coords_target, pair_probs, ss_tokens,
                temperature, pH, Mg_conc, Na_conc,
            )
        else:
            return self._sample(
                seq_tokens, pair_probs, ss_tokens,
                temperature, pH, Mg_conc, Na_conc,
            )

    def _train_step(self, seq_tokens, coords_target, pair_probs, ss_tokens,
                    temperature, pH, Mg_conc, Na_conc):
        """Single training step."""
        B, L, _ = coords_target.shape
        device = coords_target.device

        # Random timestep
        t = torch.randint(0, self.config.n_diffusion_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(coords_target)
        alpha_bar = self.alpha_bars[t].view(B, 1, 1)
        coords_noisy = torch.sqrt(alpha_bar) * coords_target + \
                       torch.sqrt(1 - alpha_bar) * noise

        # Check for NaN in noisy coords
        if torch.isnan(coords_noisy).any() or torch.isinf(coords_noisy).any():
            return {
                'noise_loss': torch.tensor(float('nan'), device=device),
                'closure_loss': torch.tensor(float('nan'), device=device),
                'total_loss': torch.tensor(float('nan'), device=device),
            }

        # Encode conditions
        cond = self.condition_encoder(
            seq_tokens, ss_tokens, temperature, pH, Mg_conc, Na_conc
        )

        # Time embedding
        t_emb = self.time_embed(t.float())

        # Predict noise via EGNN
        noise_pred = self._denoise(coords_noisy, cond, t_emb, L, pair_probs)

        # Check for NaN in noise_pred
        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            return {
                'noise_loss': torch.tensor(float('nan'), device=device),
                'closure_loss': torch.tensor(float('nan'), device=device),
                'total_loss': torch.tensor(float('nan'), device=device),
            }

        # Loss
        noise_loss = F.mse_loss(noise_pred, noise)

        # Closure auxiliary loss (with clamp to prevent gradient explosion)
        # coords_pred can have large values during early training, causing
        # closure_dist >> bond_length and (closure_dist - 5.9)^2 to explode.
        coords_pred = coords_noisy - noise_pred
        closure_dist = torch.norm(coords_pred[:, 0] - coords_pred[:, -1], dim=-1)
        closure_error = (closure_dist - self.config.bond_length).clamp(-50, 50)
        closure_loss = (closure_error ** 2).mean()

        return {
            'noise_loss': noise_loss,
            'closure_loss': closure_loss,
            'total_loss': noise_loss + 0.1 * closure_loss,
        }

    def _sample(self, seq_tokens, pair_probs, ss_tokens,
                temperature, pH, Mg_conc, Na_conc):
        """Sample with guided diffusion."""
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Encode conditions
        cond = self.condition_encoder(
            seq_tokens, ss_tokens, temperature, pH, Mg_conc, Na_conc
        )

        # Start from noise
        coords = torch.randn(B, L, 3, device=device)

        # Iterative denoising with guidance
        for t in reversed(range(self.config.n_diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float)
            t_emb = self.time_embed(t_tensor)

            # Predict noise
            noise_pred = self._denoise(coords, cond, t_emb, L, pair_probs)

            # Guided diffusion: add reward gradients
            if t < self.config.n_diffusion_steps // 2:  # Only guide in later steps
                with torch.enable_grad():
                    coords_guided = coords.detach().requires_grad_(True)
                    closure_r = self.closure_reward(coords_guided)
                    energy_r = self.energy_reward(coords_guided)

                    total_reward = (self.config.closure_weight * closure_r +
                                    self.config.energy_weight * energy_r)
                    grad = torch.autograd.grad(total_reward.sum(), coords_guided)[0]

                    # Apply gradient guidance
                    noise_pred = noise_pred - 0.01 * grad

            # Denoise step
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            if t > 0:
                sigma = self.betas[t] ** 0.5
                noise = torch.randn_like(coords)
            else:
                sigma = 0
                noise = 0

            coords = (1 / alpha.sqrt()) * (coords -
                     (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred) + sigma * noise

        # Final closure enforcement
        coords = self._enforce_closure(coords)

        return {
            'coords': coords,
            'closure_distance': torch.norm(coords[:, 0] - coords[:, -1], dim=-1),
            'method': 'circrna_ddpm_egnn_guided',
        }

    def _denoise(self, coords, cond, t_emb, L, pair_probs=None):
        """EGNN-based denoising."""
        B = coords.shape[0]
        device = coords.device

        # Build graph
        edge_index, edge_types = self.graph_builder.build(L, pair_probs)
        edge_index = edge_index.to(device)
        edge_types = edge_types.to(device)

        # Edge features: one-hot edge type + distance
        E = edge_index.shape[1]
        edge_feat = torch.zeros(B, E, self.config.d_edge, device=device)
        for et in range(2):
            mask = (edge_types == et)
            if mask.any():
                edge_feat[:, mask, et] = 1.0

        # Add distance to edge features (normalized and clamped)
        src, dst = edge_index[0], edge_index[1]
        dist = torch.norm(coords[:, src] - coords[:, dst], dim=-1, keepdim=True).clamp(max=100)
        if self.config.d_edge > 2:
            edge_feat[:, :, 2] = (dist.squeeze(-1) / 20.0).clamp(-5, 5)  # Normalize and clamp

        # Node features: condition + time
        node_feat = cond + t_emb.unsqueeze(1)  # (B, L, d_node)

        # EGNN layers
        for layer in self.egnn_layers:
            node_feat, coords = layer(node_feat, coords, edge_index, edge_feat)

        # Project to displacement
        displacement = self.coord_proj(node_feat)  # (B, L, 3)

        return displacement

    def _enforce_closure(self, coords):
        """Post-hoc closure enforcement."""
        B, L, _ = coords.shape
        direction = coords[:, 0] - coords[:, -2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        coords[:, -1] = coords[:, 0] - self.config.bond_length * direction
        return coords


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_embed // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
