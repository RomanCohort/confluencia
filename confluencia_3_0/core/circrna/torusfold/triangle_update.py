"""
triangle_update.py — Torus-adapted Triangle Update modules.

Based on AlphaFold2's core innovation: Triangle Multiplicative Update
and Triangle Attention, adapted for circRNA's torus topology.

AlphaFold2's pair representation uses triangle operations because
of the triangle inequality in 3D Euclidean geometry:
  d(i,j) ≤ d(i,k) + d(k,j)

For circRNA on a torus, the analogous constraint is:
  d_circ(i,j) ≤ min(d_circ(i,k) + d_circ(k,j), L)  (on S¹)

This module implements:
1. TriangleMultiplicativeUpdate (outgoing/incoming)
   - CircRNA adaptation: indices wrap around mod L
2. TriangleAttention (starting/ending node)
   - CircRNA adaptation: circular relative position bias
3. CircPairformer — full pair update block (AF3-style)
   - Simplified from AF3's Pairformer (no MSA stack needed
     since circRNA doesn't use MSA like proteins)

Key difference from AF2/AF3:
- No MSA representation (circRNA lacks evolutionary depth)
- Circular topology replaces linear topology
- Pair representation encodes base-pairing + IRS relationships
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpe import CircularRelativeBias


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Triangle Multiplicative Update for circRNA pair representation.

    Adapted from AlphaFold2's TriangleMultiplicativeUpdate (outgoing/incoming).

    For three positions i, j, k on a circular RNA:
    - Outgoing: z[i,j] += Σ_k (a[i,k] ⊙ b[j,k])  — updates from paths i→k→j
    - Incoming: z[i,j] += Σ_k (a[k,i] ⊙ b[k,j])  — updates from paths k→i, k→j

    In circRNA, these "paths" can cross the back-splice junction.
    The summation over k uses circular indexing where positions
    wrap around mod L.

    Args:
        c_z: Pair representation dimension (default: 128, matching AF2)
        c_hidden: Hidden dimension for multiplicative update
        direction: "outgoing" or "incoming"
    """

    def __init__(
        self,
        c_z: int = 128,
        c_hidden: Optional[int] = None,
        direction: str = "outgoing",
    ):
        super().__init__()
        assert direction in ("outgoing", "incoming")

        self.c_z = c_z
        self.c_hidden = c_hidden or c_z
        self.direction = direction

        # Layer norm on input
        self.layer_norm_input = nn.LayerNorm(c_z)

        # Project to left and right factors
        self.linear_a = nn.Linear(c_z, self.c_hidden)
        self.linear_b = nn.Linear(c_z, self.c_hidden)

        # Gate on the multiplicative product
        self.linear_g = nn.Linear(c_z, self.c_hidden)
        # Output projection
        self.linear_z = nn.Linear(self.c_hidden, c_z)
        # Output gate
        self.linear_g_out = nn.Linear(c_z, c_z)

        # Initialize gates to output zero initially (following AF2)
        nn.init.zeros_(self.linear_g.weight)
        nn.init.ones_(self.linear_g.bias)
        nn.init.zeros_(self.linear_g_out.weight)
        nn.init.ones_(self.linear_g_out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, L, L, c_z) pair representation

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        z_input = z

        # Layer norm
        z = self.layer_norm_input(z)

        # Compute left and right factors
        a = self.linear_a(z)  # (B, L, L, c_hidden)
        b = self.linear_b(z)  # (B, L, L, c_hidden)

        # Compute gate for the product
        g = torch.sigmoid(self.linear_g(z))  # (B, L, L, c_hidden)

        if self.direction == "outgoing":
            # z[i,j] += Σ_k (a[i,k] * b[j,k])
            # In circRNA: k wraps around the circle
            # This is equivalent to: a * b^T where we transpose the k axis
            # a: (B, L, L, c_hidden) → treat as (B, L, L) with features c_hidden
            # b: (B, L, L, c_hidden) → transpose first two dims for "j,k"
            # Result: for each i,j, sum over k of a[i,k] * b[j,k]

            # Efficient implementation via matrix multiplication
            # a_transposed: (B, c_hidden, L, L) for batched matmul
            a_t = a.permute(0, 3, 1, 2)  # (B, c_hidden, L, L) = a[i,k]
            b_t = b.permute(0, 3, 2, 1)  # (B, c_hidden, L, L) = b[j,k] → need b[k,j]

            # For outgoing: we need sum_k a[i,k] * b[j,k]
            # This is: a[i,:] dot b[j,:] = matmul(a, b^T) per channel
            # = a_t @ b_t.permute(0,1,3,2) → but b_t is already permuted
            # Let me be precise:
            # a[i,k] → a_t[:,:,i,k]
            # b[j,k] → b[:,:,j,k] → b.permute(0,3,j,k)
            # Sum over k: (a[i,k] * b[j,k]) → torch.einsum('bckik,bcjck->bcij', ...)

            # Simpler: reshape for einsum
            # outgoing product: sum_k a_ik * b_jk
            product = torch.einsum('bikc,bjkc->bijc', a, b)  # (B, L, L, c_hidden)

        else:  # incoming
            # z[i,j] += Σ_k (a[k,i] * b[k,j])
            # For circRNA: this captures paths k→i and k→j
            product = torch.einsum('bkic,bkjc->bijc', a, b)  # (B, L, L, c_hidden)

        # Apply gate
        product = product * g

        # Project to output dimension
        z_update = self.linear_z(product)  # (B, L, L, c_z)

        # Output gate
        g_out = torch.sigmoid(self.linear_g_out(z_input))  # (B, L, L, c_z)

        # Combine
        z_out = z_input + g_out * z_update

        return z_out


class TriangleAttention(nn.Module):
    """
    Triangle Attention for circRNA pair representation.

    Adapted from AlphaFold2's TriangleAttention (starting/ending node).

    For a pair matrix z[i,j]:
    - Starting node: for each row i, attend over columns j using pair[i,:] as bias
    - Ending node: for each column j, attend over rows i using pair[:,j] as bias

    In circRNA, the attention includes circular relative position bias,
    ensuring that positions near the back-splice junction are treated
    correctly (d_circ instead of d_linear).

    Args:
        c_z: Pair representation dimension
        n_heads: Number of attention heads
        direction: "starting" (row-wise) or "ending" (column-wise)
        max_circ_dist: Max circular distance for bias
    """

    def __init__(
        self,
        c_z: int = 128,
        n_heads: int = 4,
        direction: str = "starting",
        max_circ_dist: int = 128,
    ):
        super().__init__()
        assert direction in ("starting", "ending")

        self.c_z = c_z
        self.n_heads = n_heads
        self.head_dim = c_z // n_heads
        self.direction = direction

        # Layer norm
        self.layer_norm = nn.LayerNorm(c_z)

        # Attention projections
        self.q_proj = nn.Linear(c_z, c_z)
        self.k_proj = nn.Linear(c_z, c_z)
        self.v_proj = nn.Linear(c_z, c_z)
        self.out_proj = nn.Linear(c_z, c_z)

        # Circular relative position bias
        self.circ_bias = CircularRelativeBias(n_heads, max_dist=max_circ_dist)

        # Gate
        self.linear_g = nn.Linear(c_z, c_z)
        nn.init.zeros_(self.linear_g.weight)
        nn.init.ones_(self.linear_g.bias)

    def forward(self, z: torch.Tensor, physics_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: (B, L, L, c_z) pair representation
            physics_bias: (B, n_heads, L, L) optional physics attention bias

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        B, L, _, _ = z.shape

        z_input = z
        z_norm = self.layer_norm(z)

        scale = self.head_dim ** -0.5

        if self.direction == "starting":
            # Row-wise attention: for each i, attend over all j
            # Query: z[i,j], Key: z[i,k], Value: z[i,k]
            # Bias: circular distance between j and k

            # Permute to (B, L, L, c_z) → treat as (B*L, L, c_z)
            z_flat = z_norm.reshape(B * L, L, self.c_z)

            q = self.q_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*L, H, L, L)

            # Add circular bias: (1, H, L, L) → (B*L, H, L, L)
            # Each of the B*L "rows" gets the same bias
            circ_bias = self.circ_bias(L)  # (1, H, L, L)
            attn = attn + circ_bias  # broadcasting: (B*L, H, L, L) + (1, H, L, L)

            # Add physics bias if provided
            if physics_bias is not None:
                # physics_bias: (B, H, L, L) → (B*L, H, L, L)
                phys = physics_bias.unsqueeze(1).expand(-1, L, -1, -1, -1)
                phys = phys.reshape(B * L, self.n_heads, L, L)
                attn = attn + phys

            attn = F.softmax(attn, dim=-1)

            out = torch.matmul(attn, v)  # (B*L, H, L, head_dim)
            out = out.transpose(1, 2).reshape(B * L, L, self.c_z)
            out = self.out_proj(out)

            z_update = out.reshape(B, L, L, self.c_z)

        else:  # ending — column-wise attention
            # For each j, attend over all i
            # Permute: (B, L, L, c_z) → transpose first two dims → (B, L, L, c_z)
            z_perm = z_norm.permute(0, 2, 1, 3).contiguous()  # swap i and j
            z_flat = z_perm.reshape(B * L, L, self.c_z)

            q = self.q_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(z_flat).view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            circ_bias = self.circ_bias(L)
            attn = attn + circ_bias.repeat(B * L, 1, 1, 1)

            attn = F.softmax(attn, dim=-1)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B * L, L, self.c_z)
            out = self.out_proj(out)

            z_update = out.reshape(B, L, L, self.c_z).permute(0, 2, 1, 3).contiguous()

        # Gate
        g = torch.sigmoid(self.linear_g(z_input))
        z_out = z_input + g * z_update

        return z_out


class PairTransition(nn.Module):
    """
    Pair transition block (following AF2/AF3).

    Simple MLP transition for pair representation:
    LN → Linear(c_z, c_hidden) → ReLU → Linear(c_hidden, c_z) → Gate
    """

    def __init__(self, c_z: int = 128, c_hidden: Optional[int] = None):
        super().__init__()
        c_hidden = c_hidden or c_z * 4

        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_1 = nn.Linear(c_z, c_hidden)
        self.linear_2 = nn.Linear(c_hidden, c_z)

        # Gate
        self.linear_g = nn.Linear(c_z, c_z)
        nn.init.zeros_(self.linear_g.weight)
        nn.init.ones_(self.linear_g.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_input = z
        z = self.layer_norm(z)

        g = torch.sigmoid(self.linear_g(z_input))
        z = F.relu(self.linear_1(z))
        z = self.linear_2(z)

        return z_input + g * z


class CircPairformerBlock(nn.Module):
    """
    CircRNA Pairformer Block — adapted from AlphaFold3's Pairformer.

    AF3's Pairformer simplifies AF2's Evoformer by removing the MSA stack
    (since we don't have evolutionary data for circRNA like proteins have MSA).

    The block updates the pair representation through:
    1. Triangle Multiplicative Update (outgoing)
    2. Triangle Multiplicative Update (incoming)
    3. Triangle Attention (starting node)
    4. Triangle Attention (ending node)
    5. Pair Transition

    All with circular topology adaptations.

    Scheme 5 extension: Physics constraint embedding as attention bias.
    - Electrostatic repulsion (phosphate backbone -1 charge)
    - Base stacking energy (π-π interaction along helix)
    - Van der Waals clash penalty
    - BSJ closure distance penalty

    Args:
        c_z: Pair representation dimension (default: 128)
        c_hidden_tri: Hidden dim for triangle updates
        n_heads_tri: Heads for triangle attention
        max_circ_dist: Max circular distance for bias
        use_physics_bias: Enable physics constraint embedding (Scheme 5)
    """

    def __init__(
        self,
        c_z: int = 128,
        c_hidden_tri: Optional[int] = None,
        n_heads_tri: int = 4,
        max_circ_dist: int = 128,
        use_physics_bias: bool = True,
    ):
        super().__init__()
        c_hidden_tri = c_hidden_tri or c_z

        # Triangle updates
        self.tri_mul_out = TriangleMultiplicativeUpdate(
            c_z, c_hidden_tri, direction="outgoing")
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            c_z, c_hidden_tri, direction="incoming")

        # Triangle attention
        self.tri_att_start = TriangleAttention(
            c_z, n_heads_tri, direction="starting", max_circ_dist=max_circ_dist)
        self.tri_att_end = TriangleAttention(
            c_z, n_heads_tri, direction="ending", max_circ_dist=max_circ_dist)

        # Transition
        self.pair_transition = PairTransition(c_z)

        # Physics constraint embedding (Scheme 5)
        self.use_physics_bias = use_physics_bias
        if use_physics_bias:
            self.physics_bias = PhysicsConstraintBias(
                n_heads=n_heads_tri,
                max_circ_dist=max_circ_dist,
            )

    def forward(
        self,
        z: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, L, L, c_z) pair representation
            coords: (B, L, 3) optional coordinates for physics bias
            sequence: (B, L) optional sequence for physics bias

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        # Compute physics bias if enabled and coords provided
        physics_attn_bias = None
        if self.use_physics_bias and coords is not None:
            physics_attn_bias = self.physics_bias(coords, sequence)

        z = self.tri_mul_out(z)
        z = self.tri_mul_in(z)
        z = self.tri_att_start(z, physics_attn_bias)
        z = self.tri_att_end(z, physics_attn_bias)
        z = self.pair_transition(z)

        return z


class PhysicsConstraintBias(nn.Module):
    """
    Physics constraint embedding as attention bias (Scheme 5).

    Computes physics-based attention bias from coordinates:
    - Electrostatic: q²/d for phosphate repulsion
    - Stacking: penalty for non-optimal stacking distance
    - Clash: penalty for steric overlap
    - Closure: reward for positions near BSJ

    This enables the model to learn physics-compliant predictions
    during training, reducing post-processing dependency.
    """

    def __init__(
        self,
        n_heads: int = 4,
        max_circ_dist: int = 128,
        bond_length: float = 5.9,
        stack_distance: float = 3.4,
        clash_distance: float = 3.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.bond_length = bond_length
        self.stack_distance = stack_distance
        self.clash_distance = clash_distance

        # Physics bias weights (learnable)
        self.w_elec = nn.Parameter(torch.tensor(0.05))
        self.w_stack = nn.Parameter(torch.tensor(0.3))
        self.w_clash = nn.Parameter(torch.tensor(10.0))
        self.w_closure = nn.Parameter(torch.tensor(1.0))

        # Project physics features to attention bias
        self.physics_to_bias = nn.Linear(4, n_heads)

    def forward(
        self,
        coords: torch.Tensor,       # (B, L, 3)
        sequence: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """Compute physics-based attention bias.

        Returns:
            (B, n_heads, L, L) attention bias tensor
        """
        B, L, _ = coords.shape
        device = coords.device

        # Distance matrix
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
        distances = torch.norm(diff, dim=-1)  # (B, L, L)

        # Avoid zero distances (self-pairs)
        distances = distances + 1e-8

        # 1. Electrostatic repulsion (phosphate backbone)
        # Simplified: q²/d for non-adjacent pairs
        elec_bias = torch.zeros(B, L, L, device=device)
        # Only apply to non-adjacent (|i-j| > 1) and within cutoff
        for i in range(L):
            for j in range(L):
                if abs(i - j) > 1 and distances[0, i, j] < 20.0:
                    elec_bias[:, i, j] = self.w_elec / distances[:, i, j]

        # 2. Base stacking (prefer ~3.4Å vertical separation)
        # Adjacent bases should have consistent z-offset
        stack_bias = torch.zeros(B, L, L, device=device)
        for i in range(L):
            j = (i + 1) % L
            dz = torch.abs(coords[:, i, 2] - coords[:, j, 2])
            stack_bias[:, i, j] = self.w_stack * (dz - self.stack_distance) ** 2
            stack_bias[:, j, i] = stack_bias[:, i, j]

        # 3. Clash penalty (too close)
        clash_bias = torch.zeros(B, L, L, device=device)
        clash_mask = (distances < self.clash_distance) & \
                     (torch.abs(torch.arange(L, device=device).unsqueeze(0) - \
                      torch.arange(L, device=device).unsqueeze(1)) > 1)
        clash_bias = self.w_clash * torch.where(
            clash_mask,
            (self.clash_distance - distances) ** 2,
            torch.zeros_like(distances)
        )

        # 4. Closure reward (BSJ region should be close)
        closure_bias = torch.zeros(B, L, L, device=device)
        # Reward connections near BSJ (positions 0 and L-1)
        bsj_dist = distances[:, 0, L-1]
        closure_bias[:, 0, L-1] = -self.w_closure * (bsj_dist - self.bond_length) ** 2
        closure_bias[:, L-1, 0] = closure_bias[:, 0, L-1]

        # Stack physics features
        physics_features = torch.stack([
            elec_bias, stack_bias, clash_bias, closure_bias
        ], dim=-1)  # (B, L, L, 4)

        # Project to attention bias
        attn_bias = self.physics_to_bias(physics_features)  # (B, L, L, n_heads)
        attn_bias = attn_bias.permute(0, 3, 1, 2)  # (B, n_heads, L, L)

        return attn_bias


class CircPairformerStack(nn.Module):
    """
    Stack of CircPairformerBlocks — the circRNA equivalent of
    AlphaFold3's Pairformer stack.

    Unlike AF3 which has 48 Evoformer/Pairformer layers, we use
    fewer (4-8) because:
    1. circRNA sequences are shorter (200-500 nt vs 1000+ residues)
    2. We don't have MSA, so pair updates are simpler
    3. Training data is smaller

    Args:
        n_blocks: Number of Pairformer blocks (default: 4)
        c_z: Pair representation dimension
        c_hidden_tri: Hidden dim for triangle updates
        n_heads_tri: Heads for triangle attention
        max_circ_dist: Max circular distance for bias
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_z: int = 128,
        c_hidden_tri: Optional[int] = None,
        n_heads_tri: int = 4,
        max_circ_dist: int = 128,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CircPairformerBlock(c_z, c_hidden_tri, n_heads_tri, max_circ_dist)
            for _ in range(n_blocks)
        ])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, L, L, c_z) pair representation

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        for block in self.blocks:
            z = block(z)

        return self.layer_norm(z)
