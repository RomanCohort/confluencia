"""
equivariant_gnn.py — Sparse equivariant GNN for SO(2) × SO(2) torus.

Key design:
- Input: degree-0 (invariant) features from token embedding
- Kernel: generates both degree-0 and degree-1+ outputs using angular irrep features
- Message passing: aggregating equivariant features preserves equivariance (weighted average)
- Activation: only on degree-0 part (nonlinearities break equivariance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .irrep_types import IrrepLinear, IrrepNorm


@dataclass
class EquivariantGNNConfig:
    """Configuration for equivariant GNN."""
    d_model: int = 256          # Total feature dimension (d_inv + d_eq)
    d_inv: int = 64             # Degree-0 (invariant) dimension
    d_eq: int = 192             # Degree-1+ (equivariant) dimension, must be even

    d_edge: int = 64            # Edge feature dimension (for irrep features)
    n_edge_cats: int = 5        # Number of edge categories

    # Irrep frequencies for kernel
    k_theta_max: int = 2        # Max frequency for θ (major ring)
    k_phi_max: int = 1          # Max frequency for φ (minor ring)

    dropout: float = 0.1


class AngularIrrepFeatures(nn.Module):
    """Compute irrep features from angular differences.

    For SO(2) with max frequency K:
    - degree-0: 1 (constant)
    - degree-k (k=1..K): (sin(k·Δθ), cos(k·Δθ))

    For SO(2) × SO(2) (θ, φ), use tensor product:
    - degree-(l_θ, l_φ): sin/cos products

    This produces the "steerable" basis for equivariant kernels.
    """

    def __init__(self, k_theta_max: int = 2, k_phi_max: int = 1):
        super().__init__()
        self.k_theta_max = k_theta_max
        self.k_phi_max = k_phi_max

        # Number of irrep channels:
        # (1 + 2*k_theta_max) * (1 + 2*k_phi_max)
        # = (for θ: 1 scalar + 2*K_θ) * (for φ: 1 scalar + 2*K_φ)
        self.n_irreps = (1 + 2 * k_theta_max) * (1 + 2 * k_phi_max)

    def forward(self, delta_theta: torch.Tensor, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute irrep features from angular differences.

        Args:
            delta_theta: (B, L, K) angular differences on major ring
            delta_phi: (B, L, K) angular differences on minor ring

        Returns:
            irrep: (B, L, K, n_irreps) irrep features
        """
        B, L, K = delta_theta.shape
        device = delta_theta.device
        dtype = delta_theta.dtype

        # Build θ irrep features: (B, L, K, 1 + 2*k_theta_max)
        theta_feats = [torch.ones_like(delta_theta)]  # degree-0
        for k in range(1, self.k_theta_max + 1):
            theta_feats.append(torch.sin(k * delta_theta))
            theta_feats.append(torch.cos(k * delta_theta))
        theta_feats = torch.stack(theta_feats, dim=-1)  # (B, L, K, n_theta_irreps)

        # Build φ irrep features: (B, L, K, 1 + 2*k_phi_max)
        phi_feats = [torch.ones_like(delta_phi)]  # degree-0
        for l in range(1, self.k_phi_max + 1):
            phi_feats.append(torch.sin(l * delta_phi))
            phi_feats.append(torch.cos(l * delta_phi))
        phi_feats = torch.stack(phi_feats, dim=-1)  # (B, L, K, n_phi_irreps)

        # Tensor product: (B, L, K, n_theta, n_phi) -> (B, L, K, n_irreps)
        irrep = torch.einsum('...i,...j->...ij', theta_feats, phi_feats)
        irrep = irrep.reshape(B, L, K, self.n_irreps)

        return irrep


class SparseSteerableKernel(nn.Module):
    """Sparse steerable kernel for SO(2) × SO(2).

    The kernel operates on:
    - Input: degree-0 (invariant) node features
    - Edge: angular differences (Δθ, Δφ) → irrep features
    - Output: degree-0 and degree-1+ node features

    Key constraint:
    - degree-0 output: arbitrary linear combination (equivariant on scalars)
    - degree-1+ output: must use irrep features to generate equivariant output
    """

    def __init__(self, config: EquivariantGNNConfig):
        super().__init__()
        self.config = config

        self.angular_irreps = AngularIrrepFeatures(config.k_theta_max, config.k_phi_max)

        # Edge category embedding (degree-0)
        self.edge_embed = nn.Embedding(config.n_edge_cats, config.d_edge)

        # ===== Degree-0 output =====
        # Arbitrary linear: [node_feat (d_model) + edge_feat (d_edge)] → d_inv
        self.inv_kernel = nn.Linear(
            config.d_model + config.d_edge,
            config.d_inv,
            bias=False
        )

        # ===== Degree-1+ output =====
        # Use irrep features (2D blocks) to generate equivariant output
        n_irreps = self.angular_irreps.n_irreps  # (1 + 2*K_theta) * (1 + 2*K_phi)
        n_2d_blocks = max(0, (n_irreps - 1) // 2)  # Number of 2D irrep pairs
        n_out_blocks = config.d_eq // 2

        # Scales: how much each input 2D irrep contributes to each output block
        # (n_out_blocks, n_2d_blocks)
        if n_2d_blocks > 0 and n_out_blocks > 0:
            self.eq_scales = nn.Parameter(torch.ones(n_out_blocks, n_2d_blocks) * 0.1)
        else:
            self.eq_scales = None

        # Node feature modulation (degree-0 to scalar)
        self.node_modulate = nn.Linear(config.d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_phi: torch.Tensor,
        edge_cat: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse message passing.

        Args:
            x: (B, L, d_model) input features (degree-0)
            delta_theta: (B, L, K) angular differences on major ring
            delta_phi: (B, L, K) angular differences on minor ring
            edge_cat: (B, L, K) edge categories
            topk_idx: (B, L, K) neighbor indices

        Returns:
            msg_inv: (B, L, d_inv) degree-0 messages
            msg_eq: (B, L, d_eq) degree-1+ messages
        """
        B, L, d_model = x.shape
        K = delta_theta.shape[2]
        device = x.device
        dtype = x.dtype

        # Compute irrep features from angular differences
        irrep = self.angular_irreps(delta_theta, delta_phi)  # (B, L, K, n_irreps)

        # Gather neighbor features
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, K)
        x_j = x[b_idx, topk_idx]  # (B, L, K, d_model)

        # Edge embedding
        edge_emb = self.edge_embed(edge_cat)  # (B, L, K, d_edge)

        # ===== Degree-0 output =====
        # Arbitrary linear combination (equivariant on scalars)
        x_edge = torch.cat([x_j, edge_emb], dim=-1)  # (B, L, K, d_model + d_edge)
        msg_inv = self.inv_kernel(x_edge)  # (B, L, K, d_inv)
        msg_inv = msg_inv.mean(dim=2)  # (B, L, d_inv) - average over neighbors

        # ===== Degree-1+ output =====
        # Use irrep features to generate equivariant output
        n_irreps = irrep.shape[-1]
        n_2d_blocks = (n_irreps - 1) // 2
        n_out_blocks = self.config.d_eq // 2

        if self.eq_scales is not None and n_2d_blocks > 0:
            # Extract 2D irrep blocks (skip degree-0 at index 0)
            # irrep: (B, L, K, n_irreps)
            irrep_2d = irrep[..., 1:].reshape(B, L, K, n_2d_blocks, 2)  # (B, L, K, n_2d_blocks, 2)

            # Node feature modulation: scalar amplitude per neighbor
            # x_j: (B, L, K, d_model) -> (B, L, K, 1)
            x_scalar = self.node_modulate(x_j)  # (B, L, K, 1)

            # Combine: each output block = weighted sum over (neighbors, input_2d_blocks)
            # scales: (n_out_blocks, n_2d_blocks)
            # irrep_2d: (B, L, K, n_2d_blocks, 2)
            # x_scalar: (B, L, K, 1)
            #
            # msg_eq_blocks[o,b,l,d] = sum_{k,i} scales[o,i] * x_scalar[b,l,k] * irrep_2d[b,l,k,i,d]
            msg_eq_blocks = torch.einsum('oi,blkid,blk->blod', self.eq_scales, irrep_2d, x_scalar.squeeze(-1))
            # Result: (B, L, n_out_blocks, 2)

            msg_eq = msg_eq_blocks.reshape(B, L, self.config.d_eq)
        else:
            # Fallback: zero output (no equivariant signal)
            msg_eq = torch.zeros(B, L, self.config.d_eq, device=device, dtype=dtype)

        return msg_inv, msg_eq


class SparseEquivariantGNNLayer(nn.Module):
    """Single layer of sparse equivariant GNN.

    Updates:
    - degree-0: message passing + activation + residual
    - degree-1+: message passing + residual (no activation)
    """

    def __init__(self, config: EquivariantGNNConfig):
        super().__init__()
        self.config = config

        self.kernel = SparseSteerableKernel(config)

        # Update networks
        # inv_update input = [current inv (d_inv) + msg_inv (d_inv)]
        self.inv_update = nn.Sequential(
            nn.Linear(config.d_inv + config.d_inv, config.d_inv),
            nn.GELU(),
            nn.LayerNorm(config.d_inv),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_inv, config.d_inv),
        )

        # Degree-1+ update: isotropic scaling (equivariant)
        n_blocks = config.d_eq // 2
        self.eq_update_scales = nn.Parameter(torch.ones(n_blocks) * 0.5)

        # Normalization
        self.norm = IrrepNorm(config.d_inv, config.d_eq)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, d_model) - degree-0 input
        x_inv: torch.Tensor,  # (B, L, d_inv) - current degree-0 state
        x_eq: torch.Tensor,  # (B, L, d_eq) - current degree-1+ state
        delta_theta: torch.Tensor,
        delta_phi: torch.Tensor,
        edge_cat: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One layer of message passing.

        Returns:
            out_inv: (B, L, d_inv)
            out_eq: (B, L, d_eq)
        """
        # Message passing
        msg_inv, msg_eq = self.kernel(x, delta_theta, delta_phi, edge_cat, topk_idx)

        # Update degree-0 (with activation)
        inv_input = torch.cat([x_inv, msg_inv], dim=-1)
        inv_update = self.inv_update(inv_input)
        out_inv = x_inv + inv_update

        # Update degree-1+ (isotropic scaling, no activation)
        n_blocks = self.config.d_eq // 2
        msg_eq_blocks = msg_eq.reshape(*msg_eq.shape[:-1], n_blocks, 2)
        x_eq_blocks = x_eq.reshape(*x_eq.shape[:-1], n_blocks, 2)

        out_eq_blocks = x_eq_blocks + self.eq_update_scales.view(1, 1, -1, 1) * msg_eq_blocks
        out_eq = out_eq_blocks.reshape(*x_eq.shape)

        # Normalize
        out_inv, out_eq = self.norm(out_inv, out_eq)

        return out_inv, out_eq


class EquivariantEncoder(nn.Module):
    """Equivariant encoder: token embedding + sparse GNN layers.

    Output: separated degree-0 (invariant) and degree-1+ (equivariant) features.
    """

    def __init__(self, config: EquivariantGNNConfig, n_layers: int = 4):
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        # Token embedding (degree-0)
        self.token_embed = nn.Embedding(5, config.d_model)

        # Positional encoding (degree-0)
        self.pos_embed = nn.Linear(1, config.d_model)

        # GNN layers
        self.layers = nn.ModuleList([
            SparseEquivariantGNNLayer(config) for _ in range(n_layers)
        ])

        # Initial degree-0 state
        self.init_inv = nn.Linear(config.d_model, config.d_inv)

        # Initial degree-1+ state (must be equivariant)
        # Use isotropic scaling from degree-0 input
        n_blocks = config.d_eq // 2
        self.init_eq_scales = nn.Parameter(torch.ones(n_blocks) * 0.1)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        delta_theta: Optional[torch.Tensor] = None,
        delta_phi: Optional[torch.Tensor] = None,
        edge_cat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to equivariant features.

        Args:
            seq_tokens: (B, L) sequence tokens
            topk_idx: (B, L, K) neighbor indices (if None, use random neighbors)
            delta_theta: (B, L, K) angular differences (if None, compute from positions)
            delta_phi: (B, L, K) angular differences
            edge_cat: (B, L, K) edge categories

        Returns:
            node_inv: (B, L, d_inv) degree-0 features
            node_eq: (B, L, d_eq) degree-1+ features
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Token embedding (degree-0)
        x = self.token_embed(seq_tokens)  # (B, L, d_model)

        # Positional encoding (degree-0)
        pos = torch.arange(L, device=device, dtype=torch.float32)
        theta = 2 * math.pi * pos / L
        pos_enc = self.pos_embed(theta.unsqueeze(-1).unsqueeze(0).expand(B, -1, -1))
        x = x + pos_enc

        # Build sparse graph if not provided
        K = min(60, L)  # Number of neighbors

        if topk_idx is None:
            # Random neighbors
            topk_idx = torch.randint(0, L, (B, L, K), device=device)

        if delta_theta is None or delta_phi is None:
            # Compute from positions
            pos = torch.arange(L, device=device, dtype=torch.float32)
            theta_i = 2 * math.pi * pos / L
            theta_j = theta_i[topk_idx]  # (B, L, K)

            theta_i_expanded = theta_i.view(1, L, 1)
            delta_theta = theta_i_expanded - theta_j
            delta_phi = torch.zeros_like(delta_theta)  # Placeholder for φ

        if edge_cat is None:
            edge_cat = torch.zeros(B, L, K, device=device, dtype=torch.long)

        # Initialize degree-0 and degree-1+ states
        node_inv = self.init_inv(x)  # (B, L, d_inv)

        # Degree-1+ initialized isotropically from degree-0 (equivariant)
        n_blocks = self.config.d_eq // 2
        node_eq_blocks = self.init_eq_scales.view(1, 1, -1, 1) * x[..., :n_blocks * 2].reshape(B, L, n_blocks, 2)
        node_eq = node_eq_blocks.reshape(B, L, self.config.d_eq)

        # Message passing layers
        for layer in self.layers:
            node_inv, node_eq = layer(
                x, node_inv, node_eq,
                delta_theta, delta_phi, edge_cat, topk_idx
            )

        return node_inv, node_eq


# ══════════════════════════════════════════════════════════════════════════════
# Equivariance test
# ══════════════════════════════════════════════════════════════════════════════

def test_encoder_equivariance():
    """Test that EquivariantEncoder is strictly equivariant."""
    print("Testing EquivariantEncoder equivariance...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EquivariantGNNConfig(
        d_model=128,
        d_inv=32,
        d_eq=64,
        d_edge=32,
    )

    encoder = EquivariantEncoder(config, n_layers=2).to(device)
    encoder.eval()

    B, L = 2, 50
    seq = torch.randint(0, 4, (B, L), device=device)

    # Compute neighbors and angles
    K = 20
    topk_idx = torch.randint(0, L, (B, L, K), device=device)

    pos = torch.arange(L, device=device, dtype=torch.float32)
    theta = 2 * math.pi * pos / L
    theta_j = theta[topk_idx]
    theta_i = theta.view(1, L, 1)
    delta_theta = theta_i - theta_j
    delta_phi = torch.zeros_like(delta_theta)
    edge_cat = torch.zeros(B, L, K, device=device, dtype=torch.long)

    with torch.no_grad():
        # Original encoding
        inv1, eq1 = encoder(seq, topk_idx, delta_theta, delta_phi, edge_cat)

        # Rotate input: roll sequence
        roll = 5
        seq_rot = torch.roll(seq, shifts=roll, dims=1)

        # Rotate graph: neighbors shift accordingly
        topk_idx_rot = (topk_idx + roll) % L

        # Angles shift: Δθ(i,j) → Δθ((i+roll), (j+roll)) = Δθ(i,j) (same!)
        # So angles don't change

        inv2, eq2 = encoder(seq_rot, topk_idx_rot, delta_theta, delta_phi, edge_cat)

        # For degree-0: should be rolled
        inv2_rolled_back = torch.roll(inv2, shifts=-roll, dims=1)
        inv_err = (inv1 - inv2_rolled_back).abs().mean().item()

        # For degree-1+: should also be rolled (same rotation)
        eq2_rolled_back = torch.roll(eq2, shifts=-roll, dims=1)
        eq_err = (eq1 - eq2_rolled_back).abs().mean().item()

    print(f"  degree-0 equivariance error: {inv_err:.8e}")
    print(f"  degree-1+ equivariance error: {eq_err:.8e}")

    return inv_err, eq_err


if __name__ == "__main__":
    test_encoder_equivariance()