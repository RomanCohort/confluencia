"""steerable_kernel.py — SO(2)×SO(2) steerable message passing (Phase 2).

Extracted from scheme10_circ_equivariant_gnn.py so the immune fingerprint
heads (PKR, m6A) can use exact ring-equivariant message passing without
depending on the full S10 trunk (token embedding, coord predictor, closure
reward, etc.). This module is a *drop-in copy* of the two core classes:

    - SO2SteerableKernel  : the per-edge steerable convolution kernel
    - CircEquivariantGNNLayer : one layer of equivariant message passing
                                + ring-invariant readout

The ONLY change vs the S10 originals: CircEquivariantGNNLayer no longer takes
a Scheme10Config dataclass — it takes explicit scalar args, so it can be
constructed from the immune head's own dimensions without spinning up a full
S10 config object. SO2SteerableKernel was already config-free.

Equivariance guarantee (unchanged from S10):
    Because the irrep basis φ^{(k,l)}(Δθ, Δφ) = [sin(kΔθ), cos(kΔθ)] ⊗
    [sin(lΔφ), cos(lΔφ)] is built from angle *differences*, rotating every
node's (θ, φ) by the same (δθ, δφ) leaves all Δθ, Δφ unchanged, so the
messages are *exactly* SO(2)×SO(2)-equivariant. No approximation, no CG
coefficient bookkeeping — the basis choice does the work.

See scheme10_circ_equivariant_gnn.py for the original design docstring.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SO2SteerableKernel(nn.Module):
    """Steerable convolution kernel over SO(2)×SO(2).

    For each pair (i, j) with angular difference (Δθ, Δφ) and edge category c,
    the kernel computes a rotation-equivariant message:

        m[i,j] = Σ_{k,l}  φ_c^{(k,l)}(Δθ, Δφ) ⊗ W_c^{(k,l)} x_j

    where φ^{(k,l)}(Δθ, Δφ) = [sin(kΔθ), cos(kΔθ)] ⊗ [sin(lΔφ), cos(lΔφ)]
    is the (k,l) irrep of SO(2)×SO(2), and W_c^{(k,l)} is a learnable linear
    map per edge category. Because φ is built from sin/cos of the angle
    differences, rotating both endpoints by the same amount leaves Δθ, Δφ
    unchanged, so the message is *exactly* equivariant.

    Args:
        d_model: node feature dim
        d_edge: per-irrep edge channel dim
        n_edge_cats: number of edge categories
        k_theta, k_phi: highest irrep orders (inclusive; irrep orders run 0..k).
    """

    def __init__(
        self,
        d_model: int,
        d_edge: int,
        n_edge_cats: int,
        k_theta: int = 2,
        k_phi: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_edge = d_edge
        self.n_edge_cats = n_edge_cats
        self.k_theta = k_theta
        self.k_phi = k_phi

        # irrep degree 0 is the constant [1] term (invariant channel).
        # degrees 1..k contribute [sin(kΔ), cos(kΔ)] (2 channels each).
        # total irrep channels per ring = 1 + 2*k.
        self.n_theta_irreps = 1 + 2 * k_theta
        self.n_phi_irreps = 1 + 2 * k_phi
        # Full (kθ, kφ) irrep tensor: n_theta_irreps * n_phi_irreps channels.
        n_irrep_channels = self.n_theta_irreps * self.n_phi_irreps

        # Per-category, per-irrep linear maps: (i, j) -> message.
        # Implemented as one grouped Linear over n_irrep_channels*d_edge.
        # Shape: (n_edge_cats, n_irrep_channels * d_edge, d_model)
        self.kernel = nn.Parameter(
            torch.empty(n_edge_cats, n_irrep_channels * d_edge, d_model)
        )
        nn.init.xavier_uniform_(self.kernel)

        # Lift node features to per-edge irrep channels (shared across categories).
        self.node_lift = nn.Linear(d_model, n_irrep_channels * d_edge, bias=False)

        # Edge-category embedding (learned), mixed into the irrep channels.
        self.cat_embed = nn.Embedding(n_edge_cats, d_edge)

    def _irrep_features(
        self,
        delta_theta: torch.Tensor,  # (B, L, L)
        delta_phi: torch.Tensor,    # (B, L, L)
    ) -> torch.Tensor:
        """Build the (B, L, L, n_irrep_channels) irrep feature tensor.

        Order: outer product of θ-irreps and φ-irreps.
        θ-irreps: [1, sin(Δθ), cos(Δθ), sin(2Δθ), cos(2Δθ), ...]
        φ-irreps: [1, sin(Δφ), cos(Δφ), ...]
        """
        B, L, _ = delta_theta.shape
        theta_feats = [torch.ones_like(delta_theta)]  # degree 0
        for k in range(1, self.k_theta + 1):
            theta_feats.append(torch.sin(k * delta_theta))
            theta_feats.append(torch.cos(k * delta_theta))
        # (B, L, L, n_theta_irreps)
        theta_feats = torch.stack(theta_feats, dim=-1)

        phi_feats = [torch.ones_like(delta_phi)]
        for l in range(1, self.k_phi + 1):
            phi_feats.append(torch.sin(l * delta_phi))
            phi_feats.append(torch.cos(l * delta_phi))
        phi_feats = torch.stack(phi_feats, dim=-1)

        # Outer product: (B, L, L, n_theta, n_phi) -> (B, L, L, n_theta*n_phi)
        irrep = (theta_feats.unsqueeze(-1) * phi_feats.unsqueeze(-2))
        return irrep.reshape(B, L, L, -1)

    def forward(
        self,
        x: torch.Tensor,           # (B, L, d_model) node features
        delta_theta: torch.Tensor,  # (B, L, L)
        delta_phi: torch.Tensor,    # (B, L, L)
        edge_cat: torch.Tensor,     # (B, L, L) long
    ) -> torch.Tensor:
        """Return steerable messages (B, L, L, d_model).

        Memory-efficient implementation: instead of gathering per-position
        kernel rows (which materializes a (B,L,L,K,M) tensor ~7GB on CPU
        for typical configs), we first contract each of the C=5 kernels
        against flat to get (B,L,L,C,M) partial products, then gather by
        edge_cat. Memory cost: C*M per position instead of K*M.
        """
        B, L, _ = x.shape
        C = self.n_edge_cats

        irrep = self._irrep_features(delta_theta, delta_phi)  # (B,L,L,K_irrep)
        K_irrep = irrep.shape[-1]

        # Lift node j into per-edge channels, broadcast over axis i.
        lifted = self.node_lift(x)  # (B, L, K_irrep*d_edge)
        lifted = lifted.reshape(B, L, K_irrep, self.d_edge)
        # Broadcast to (B, L, L, K_irrep, d_edge): node j along axis 1.
        lifted_ij = lifted.unsqueeze(2).expand(B, L, L, K_irrep, self.d_edge)

        # Modulate by irrep features (Δθ, Δφ)-dependent.
        modulated = lifted_ij * irrep.unsqueeze(-1)  # (B,L,L,K_irrep,d_edge)

        # Mix in edge-category embedding (category-dependent gating).
        cat_emb = self.cat_embed(edge_cat)  # (B,L,L,d_edge)
        modulated = modulated * (1.0 + cat_emb.unsqueeze(-2))  # (B,L,L,K_irrep,d_edge)

        # Flatten irrep*edge dimension -> flat (B,L,L,K) where K = K_irrep*d_edge.
        K = K_irrep * self.d_edge
        flat = modulated.reshape(B, L, L, K)  # (B,L,L,K)

        # Memory-efficient: contract per-category kernels one at a time,
        # accumulate into a (B,L,L,C,d_model) buffer.
        partial = torch.einsum('blik,ckm->blicm', flat, self.kernel)  # (B,L,L,C,d_model)

        # Gather by edge_cat: msg[bli,m] = partial[bli, cat[bli], m]
        msg = partial[
            torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, L),
            torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, L),
            torch.arange(L, device=x.device).view(1, 1, L).expand(B, L, L),
            edge_cat,
        ]  # (B, L, L, d_model)

        return msg


class CircEquivariantGNNLayer(nn.Module):
    """One layer of SO(2)×SO(2) equivariant message passing + ring-invariant readout.

    Forward:
        messages = SO2SteerableKernel(x, Δθ, Δφ, edge_cat)   # (B,L,L,d_model)
        agg      = mean over j of messages                     # (B,L,d_model)
        x'       = x + MLP_update(concat[x, agg])              # residual

    The aggregation (mean over the ring) is itself ring-equivariant: shifting
    every node's θ by a constant does not change the set {messages[i,j]}_j,
    so the aggregated feature transforms covariantly with x.

    Args (config-free, unlike the S10 original which took Scheme10Config):
        d_model: node feature dim
        d_edge: per-irrep edge channel dim
        n_edge_cats: number of edge categories
        k_theta, k_phi: highest irrep orders
        dropout: dropout rate in the update MLP
    """

    def __init__(
        self,
        d_model: int,
        d_edge: int = 32,
        n_edge_cats: int = 5,
        k_theta: int = 2,
        k_phi: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel = SO2SteerableKernel(
            d_model=d_model,
            d_edge=d_edge,
            n_edge_cats=n_edge_cats,
            k_theta=k_theta,
            k_phi=k_phi,
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,            # (B, L, d_model)
        delta_theta: torch.Tensor,   # (B, L, L)
        delta_phi: torch.Tensor,     # (B, L, L)
        edge_cat: torch.Tensor,     # (B, L, L) long
        lengths: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        msg = self.kernel(x, delta_theta, delta_phi, edge_cat)  # (B,L,L,d_model)

        # Mask padding keys so they do not contribute to the mean.
        B, L, _ = x.shape
        device = x.device
        pos = torch.arange(L, device=device).unsqueeze(0)
        valid = (pos < lengths.unsqueeze(1)).float()  # (B, L)
        mask_ij = valid.unsqueeze(2) * valid.unsqueeze(1)  # (B, L, L)
        denom = mask_ij.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, L, 1)
        agg = (msg * mask_ij.unsqueeze(-1)).sum(dim=2) / denom  # (B, L, d_model)

        x_new = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + x_new)
