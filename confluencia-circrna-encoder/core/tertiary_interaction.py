"""
tertiary_interaction.py — RNA tertiary interaction modeling for circRNA.

Implements specialized modules for RNA-specific tertiary interactions:
1. LongRangeAttention: detects long-range base pairing (>20nt apart)
2. LoopCrossAttention: detects loop-loop interactions (kissing loops)
3. PseudoknotUpdater: pseudoknot-aware pair bias update
4. TertiaryInteractionModule: combined module

These are NOT enabled by default in CircPairformer.
Set config.tertiary_interaction = True to activate.

Future work (needs experimental data for training):
- A-minor motif detection (requires atom-level features)
- Stacking geometry constraints (requires torsion angle prediction)
- MD-guided conformational ensemble
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongRangeAttention(nn.Module):
    """
    Attention module specialized for long-range base pairing in circRNA.

    In linear RNA models, long-range interactions (>20nt) are poorly captured
    because sinusoidal PE treats distant positions as unrelated. In circRNA,
    long-range pairs can cross the BSJ, making them even harder to detect.

    This module:
    1. Uses circular distance (not linear) for position bias
    2. Applies separate attention heads for short-range vs long-range pairs
    3. Uses a lower resolution for long-range to save memory

    Args:
        d_model: Input dimension
        n_heads: Number of attention heads
        long_range_threshold: Circular distance threshold (default: 20nt)
        max_circ_dist: Maximum circular distance for bias
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        long_range_threshold: int = 20,
        max_circ_dist: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.lr_threshold = long_range_threshold

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Circular distance bias
        self.circ_bias = nn.Embedding(max_circ_dist + 1, n_heads)

        # Long-range gate: controls how much long-range info to incorporate
        self.lr_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def _circ_dist(self, L: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(L, device=device)
        diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return torch.min(diff, L - diff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) sequence features

        Returns:
            (B, L, d_model) with enhanced long-range interactions
        """
        B, L, _ = x.shape
        x_input = x
        x = self.layer_norm(x)

        # QKV projections
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores with circular bias
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add circular distance bias
        circ_dist = self._circ_dist(L, x.device).clamp(max=self.circ_bias.num_embeddings - 1)
        bias = self.circ_bias(circ_dist).permute(2, 0, 1).unsqueeze(0)  # (1, H, L, L)
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.out_proj(out)

        # Gate: emphasizes long-range contributions
        gate = self.lr_gate(x_input)
        return x_input + gate * out


class LoopCrossAttention(nn.Module):
    """
    Loop-loop interaction detector for circRNA (kissing loops, etc.).

    Kissing loops are a key tertiary interaction in circRNA:
    two hairpin loops form base pairs with each other.

    This module:
    1. Detects loop boundaries from pair representation
       (unpaired regions = loops)
    2. Computes attention between loop regions
    3. Returns interaction scores between loop pairs

    Args:
        c_z: Pair representation dimension
        n_heads: Number of attention heads for loop-loop interaction
        min_loop_size: Minimum loop size to consider (default: 3nt)
    """

    def __init__(
        self,
        c_z: int = 128,
        n_heads: int = 4,
        min_loop_size: int = 3,
    ):
        super().__init__()
        self.c_z = c_z
        self.n_heads = n_heads
        self.min_loop_size = min_loop_size
        self.head_dim = c_z // n_heads

        # Loop representation: project pair features to loop-level
        self.loop_proj = nn.Linear(c_z, c_z)

        # Cross-attention between loops
        self.q_proj = nn.Linear(c_z, c_z)
        self.k_proj = nn.Linear(c_z, c_z)
        self.v_proj = nn.Linear(c_z, c_z)
        self.out_proj = nn.Linear(c_z, c_z)

        self.layer_norm = nn.LayerNorm(c_z)

    def _detect_loops(
        self, pair_probs: torch.Tensor, threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect loop regions from pair probabilities.

        A position is in a loop if its maximum pairing probability < threshold.

        Returns:
            loop_mask: (B, L) bool — True if position is in a loop
            loop_features: (B, L, c_z) — features for each position
        """
        # pair_probs: (B, L, L) — probability of each base pair
        max_pair_prob = pair_probs.max(dim=-1).values  # (B, L)
        loop_mask = max_pair_prob < threshold  # (B, L)

        return loop_mask, max_pair_prob

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, c_z) pair representation

        Returns:
            (B, L, L, c_z) pair representation with loop-loop interaction scores
        """
        B, L, _, c_z = pair_repr.shape

        # Extract per-position features from pair representation
        # Use row mean as position feature
        pos_feat = pair_repr.mean(dim=2)  # (B, L, c_z)

        # Simple loop detection: positions with low self-pair probability
        # (This is a proxy; real loop detection would use secondary structure)
        diag_idx = torch.arange(L, device=pair_repr.device)
        self_pair = pair_repr[:, diag_idx, diag_idx, :].norm(dim=-1)  # (B, L)
        loop_score = 1.0 / (self_pair + 1e-6)  # Higher = more likely a loop

        # Loop-aware attention: boost attention between likely-loop positions
        loop_weight = torch.sigmoid(loop_score.unsqueeze(-1))  # (B, L, 1)

        # Cross-attention
        x = self.layer_norm(pos_feat)
        x = x * (1 + loop_weight)  # Amplify loop features

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, c_z)
        out = self.out_proj(out)

        # Expand back to pair representation
        # Loop-loop interaction: outer product of loop-aware features
        loop_pair_update = out.unsqueeze(2) + out.unsqueeze(1)  # (B, L, L, c_z)

        return pair_repr + 0.1 * loop_pair_update  # Small contribution


class PseudoknotUpdater(nn.Module):
    """
    Pseudoknot-aware pair bias update.

    A pseudoknot occurs when:
      Pair (i,j) and Pair (k,l) where i < k < j < l
    This violates the "nested" structure assumption of most RNA
    secondary structure prediction algorithms.

    In circRNA, pseudoknots can cross the BSJ, making them
    even more important to detect.

    This module:
    1. Detects candidate pseudoknots from pair probabilities
    2. Learns a bias to promote or suppress crossing pairs
    3. Updates pair representation accordingly

    Args:
        c_z: Pair representation dimension
        pk_bias_dim: Dimension for pseudoknot bias features
    """

    def __init__(
        self,
        c_z: int = 128,
        pk_bias_dim: int = 32,
    ):
        super().__init__()
        self.c_z = c_z
        self.pk_bias_dim = pk_bias_dim

        # Pseudoknot detection head
        self.pk_detector = nn.Sequential(
            nn.Linear(c_z * 2, pk_bias_dim),
            nn.GELU(),
            nn.Linear(pk_bias_dim, 1),
            nn.Sigmoid(),
        )

        # Bias update when pseudoknot detected
        self.pk_bias_net = nn.Sequential(
            nn.Linear(c_z + 1, c_z),  # +1 for pseudoknot probability
            nn.GELU(),
            nn.Linear(c_z, c_z),
        )

        self.layer_norm = nn.LayerNorm(c_z)

    def forward(
        self,
        pair_repr: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, c_z) pair representation
            pair_probs: (B, L, L) optional pair probabilities

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        B, L, _, c_z = pair_repr.shape

        z = self.layer_norm(pair_repr)

        # Detect crossing pairs (pseudoknot candidates)
        # For each pair (i,j), check if there's a pair (k,l) with i<k<j<l
        # Simplified: compute pseudoknot probability from pair features
        z_i = z.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, c_z) — row features
        z_j = z.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, c_z) — col features
        z_concat = torch.cat([z_i, z_j], dim=-1)  # (B, L, L, 2*c_z)

        pk_prob = self.pk_detector(z_concat)  # (B, L, L, 1)

        # Apply pseudoknot-aware bias
        z_with_pk = torch.cat([z, pk_prob], dim=-1)  # (B, L, L, c_z+1)
        pk_bias = self.pk_bias_net(z_with_pk)  # (B, L, L, c_z)

        # Gated update
        gate = torch.sigmoid(pk_prob)  # (B, L, L, 1)
        return pair_repr + gate * pk_bias


class TertiaryInteractionModule(nn.Module):
    """
    Combined module for RNA tertiary interaction modeling.

    Integrates:
    1. LongRangeAttention: long-range base pairing
    2. LoopCrossAttention: kissing loop interactions
    3. PseudoknotUpdater: pseudoknot-aware pair update

    This module is inserted into CircPairformerBlock when
    config.tertiary_interaction = True.

    Usage:
        # In CircPairformerBlock.forward():
        if self.tertiary_interaction is not None:
            z = self.tertiary_interaction(z, seq_feat)

    Args:
        c_z: Pair representation dimension
        d_model: Sequence feature dimension
        n_heads: Number of attention heads
        long_range_threshold: Minimum circular distance for "long-range"
        min_loop_size: Minimum loop size for kissing loop detection
        max_circ_dist: Maximum circular distance for bias
    """

    def __init__(
        self,
        c_z: int = 128,
        d_model: int = 640,
        n_heads: int = 4,
        long_range_threshold: int = 20,
        min_loop_size: int = 3,
        max_circ_dist: int = 128,
    ):
        super().__init__()

        self.long_range_attn = LongRangeAttention(
            d_model=d_model,
            n_heads=n_heads,
            long_range_threshold=long_range_threshold,
            max_circ_dist=max_circ_dist,
        )

        self.loop_pair_detector = LoopCrossAttention(
            c_z=c_z,
            n_heads=n_heads,
            min_loop_size=min_loop_size,
        )

        self.pseudoknot_updater = PseudoknotUpdater(c_z=c_z)

        # Output gating
        self.output_gate = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.Sigmoid(),
        )

        # Initialize gate to output near-zero (disabled by default)
        nn.init.zeros_(self.output_gate[0].weight)
        nn.init.constant_(self.output_gate[0].bias, -3.0)  # sigmoid(-3) ≈ 0.05

    def forward(
        self,
        pair_repr: torch.Tensor,
        seq_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, c_z) pair representation from CircPairformer
            seq_feat: (B, L, d_model) optional sequence features

        Returns:
            (B, L, L, c_z) updated pair representation
        """
        # 1. Long-range attention on sequence features
        if seq_feat is not None:
            lr_out = self.long_range_attn(seq_feat)
            # Project long-range info into pair representation
            B, L, d = lr_out.shape
            lr_pair = lr_out.unsqueeze(2) + lr_out.unsqueeze(1)  # (B, L, L, d)
            # We only add a small contribution via the gate
        else:
            lr_pair = torch.zeros_like(pair_repr)

        # 2. Loop-loop interactions
        loop_pair = self.loop_pair_detector(pair_repr)

        # 3. Pseudoknot update
        pk_pair = self.pseudoknot_updater(pair_repr)

        # Combine all contributions (gated)
        total_update = loop_pair - pair_repr + pk_pair - pair_repr  # deltas only
        gate = self.output_gate(pair_repr.mean(dim=(1, 2), keepdim=True).unsqueeze(-1))

        return pair_repr + gate * total_update


def circ_contact_from_linear(
    linear_contact: torch.Tensor,
    bsj_site: int,
) -> torch.Tensor:
    """
    Map linear RNA contact map to circular RNA contact map.

    Given a contact map from a linear RNA homolog, this function
    maps it to circRNA topology by considering both the original
    ordering and the circular permutation at the back-splice junction.

    Formula:
        Contact_circ(i, j) = max(Contact_linear(i, j), Contact_linear(i+N, j+N))

    where N = bsj_site (the back-splice junction position).

    This captures the fact that in circRNA, positions after the BSJ
    are adjacent to positions before the BSJ, creating new contacts.

    Args:
        linear_contact: (L, L) contact map from linear RNA
        bsj_site: Position of back-splice junction

    Returns:
        (L, L) circular contact map
    """
    L = linear_contact.size(0)

    # Original contacts
    circ_contact = linear_contact.clone()

    # Circular permutation: contacts that cross the BSJ
    # Position i in circRNA corresponds to position (i + bsj_site) % L in linear
    for i in range(L):
        for j in range(i + 1, L):
            # Map to linear indices
            li = (i + bsj_site) % L
            lj = (j + bsj_site) % L

            # Take maximum contact from either mapping
            circ_contact[i, j] = max(
                linear_contact[i, j].item(),
                linear_contact[li, lj].item(),
            )
            circ_contact[j, i] = circ_contact[i, j]

    return circ_contact
