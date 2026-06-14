"""
irs_pair.py — IRS (Internal Reverse-complement Sequence) Pair Predictor.

circRNA has unique structural features that linear RNA lacks:

1. **Internal Reverse-complement Sequences (IRS)**: Long-range
   base-pairing between distant regions of the same circle.
   Particularly important: pairs that cross the back-splice
   junction (BSJ), which cannot exist in linear RNA.

2. **BSJ-crossing pairs**: Base pairs (i, j) where i and j are
   on opposite sides of the back-splice junction. These pairs
   effectively "tie" the circle together and are critical for
   circRNA stability and function.

This module predicts the full pair probability matrix P[i, j]
representing the probability that nucleotides i and j are
base-paired in the circRNA structure.

Key design choices:
- Uses circular distance: d_circ(i, j) = min(|i-j|, L-|i-j|)
- Predicts symmetric pair matrix: P[i, j] = P[j, i]
- Respects BSJ symmetry: P[i, j] = P[(i-1) mod L, (j+1) mod L]
  (shifting the sequence by 1 around the circle should preserve
  the pair matrix after re-indexing)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpe import TorusPositionalEncoding2D, CircularRelativeBias


def circular_distance_matrix(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Compute circular distance matrix.

    d_circ(i, j) = min(|i - j|, L - |i - j|)

    Args:
        seq_len: Sequence length
        device: Device

    Returns:
        Tensor of shape (L, L) with circular distances
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    circ_dist = torch.min(diff.abs(), seq_len - diff.abs())
    return circ_dist


def is_bsj_crossing(i: int, j: int, seq_len: int) -> bool:
    """
    Check if base pair (i, j) crosses the back-splice junction.

    A pair crosses BSJ if the shorter arc between i and j
    contains the BSJ (position 0/L boundary).

    Equivalent to: |i - j| > L/2 in linear distance.
    """
    return abs(i - j) > seq_len / 2


class IRSPairModule(nn.Module):
    """
    Predict circRNA base-pairing matrix with BSJ-aware symmetry.

    Architecture:
    1. Project per-position embeddings → pair representation
    2. Apply 2D torus attention across (i, j) position pairs
    3. Predict pair probabilities P[i, j]

    Args:
        d_model: Per-position embedding dimension
        d_pair: Pair representation dimension (default: 64)
        n_heads: Number of attention heads
        n_layers: Number of pair-processing layers
        max_circ_dist: Maximum circular distance to encode
        predict_3d_coords: Whether to predict 3D coords from pairs
    """

    def __init__(
        self,
        d_model: int = 640,
        d_pair: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        max_circ_dist: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_circ_dist = max_circ_dist

        # Initial pair representation: outer product of position embeddings
        self.pair_init_left = nn.Linear(d_model, d_pair)
        self.pair_init_right = nn.Linear(d_model, d_pair)

        # Circular distance bias
        self.circ_bias = CircularRelativeBias(n_heads, max_dist=max_circ_dist)

        # BSJ-aware bias (learnable)
        self.bsj_embedding = nn.Embedding(1, d_pair)
        self.bsj_cross_weight = nn.Parameter(torch.tensor(0.5))

        # 2D Torus PE for pair positions
        self.tpe_2d = TorusPositionalEncoding2D(
            d_model=d_pair * 2,
            n_harmonics=8,
        )

        # Pair processing layers (axial attention to keep O(L*d) not O(L²d²))
        self.pair_layers = nn.ModuleList([
            AxialPairBlock(d_pair, n_heads)
            for _ in range(n_layers)
        ])

        # Output head: pair probability
        self.pair_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, 1),
        )

    def _build_initial_pair_repr(
        self,
        sequence_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build initial pair representation via outer product.

        Args:
            sequence_repr: (B, L, d_model) per-position embeddings

        Returns:
            (B, L, L, d_pair) initial pair tensor
        """
        left = self.pair_init_left(sequence_repr)  # (B, L, d_pair)
        right = self.pair_init_right(sequence_repr)  # (B, L, d_pair)

        # Outer product: pair[i, j] = left[i] + right[j]
        pair = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, d_pair)
        return pair

    def _add_circular_distance_features(
        self,
        pair: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Add circular distance features to pair representation.
        """
        circ_dist = circular_distance_matrix(seq_len, pair.device)  # (L, L)
        circ_dist = circ_dist.clamp(0, self.max_circ_dist)

        # One-hot encode distance, then project
        dist_oh = F.one_hot(circ_dist.long(), num_classes=self.max_circ_dist + 1)  # (L, L, max+1)
        dist_emb = dist_oh.float() @ self.circ_bias.relative_bias.weight  # (L, L, n_heads)
        dist_emb = dist_emb.mean(dim=-1, keepdim=True).unsqueeze(0)  # (1, L, L, 1)

        return pair + dist_emb

    def _enforce_bsj_symmetry(self, pair_matrix: torch.Tensor) -> torch.Tensor:
        """
        Enforce back-splice junction symmetry.

        For a circRNA, the pair matrix should be invariant under
        a cyclic shift of both i and j indices:
        P[i, j] = P[(i+1) mod L, (j+1) mod L]

        This is approximated by averaging P with its rolled version.
        """
        # Roll both dimensions
        rolled = torch.roll(pair_matrix, shifts=(1, 1), dims=(1, 2))
        # Symmetrize
        return 0.5 * (pair_matrix + rolled)

    def forward(
        self,
        sequence_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict circRNA base-pairing matrix.

        Args:
            sequence_repr: (B, L, d_model) from CircEquivariantBackbone

        Returns:
            Dict with:
                - pair_logits: (B, L, L) raw pair logits
                - pair_probs: (B, L, L) pair probabilities
                - pair_repr: (B, L, L, d_pair) pair representations
                - bsj_pair_mask: (L, L) mask for BSJ-crossing pairs
        """
        batch_size, seq_len, _ = sequence_repr.shape

        # 1. Initial pair representation
        pair = self._build_initial_pair_repr(sequence_repr)  # (B, L, L, d_pair)

        # 2. Add circular distance features
        pair = self._add_circular_distance_features(pair, seq_len)

        # 3. Process with axial pair blocks
        for layer in self.pair_layers:
            pair = layer(pair)

        # 4. Enforce BSJ symmetry on pair representation
        pair = self._enforce_bsj_symmetry(pair)

        # 5. Predict pair probabilities
        pair_logits = self.pair_head(pair).squeeze(-1)  # (B, L, L)

        # Enforce symmetry: P[i, j] = P[j, i]
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(-1, -2))

        # Enforce BSJ symmetry
        pair_logits = self._enforce_bsj_symmetry(pair_logits)

        pair_probs = torch.sigmoid(pair_logits)

        # Build BSJ pair mask
        positions = torch.arange(seq_len, device=pair.device)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        bsj_mask = (diff.abs() >= seq_len / 2).float()  # (L, L) — >= because circ_dist max == L/2 for even L

        return {
            "pair_logits": pair_logits,
            "pair_probs": pair_probs,
            "pair_repr": pair,
            "bsj_pair_mask": bsj_mask,
        }


class AxialPairBlock(nn.Module):
    """
    Axial attention block for pair representation.

    Standard O(L²) attention on pair tensors is too expensive.
    We use axial attention: first attend along rows (varying j),
    then along columns (varying i). This reduces complexity to O(L²d).
    """

    def __init__(self, d_pair: int, n_heads: int):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_pair)
        self.norm2 = nn.LayerNorm(d_pair)
        self.norm3 = nn.LayerNorm(d_pair)
        self.norm4 = nn.LayerNorm(d_pair)

        self.ff = nn.Sequential(
            nn.Linear(d_pair, d_pair * 4),
            nn.GELU(),
            nn.Linear(d_pair * 4, d_pair),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair: (B, L, L, d_pair)
        Returns:
            (B, L, L, d_pair)
        """
        B, L, _, d = pair.shape

        # Row attention: attend along j dimension (for each i, attend over all j)
        residual = pair
        pair = self.norm1(pair)
        pair_flat = pair.view(B * L, L, d)  # (B*L, L, d) — for each i, L j's
        pair_flat, _ = self.row_attn(pair_flat, pair_flat, pair_flat)
        pair = pair_flat.view(B, L, L, d)
        pair = residual + pair

        # Column attention: attend along i dimension
        residual = pair
        pair = self.norm2(pair)
        pair_flat = pair.permute(0, 2, 1, 3).contiguous().view(B * L, L, d)  # (B*L, L, d)
        pair_flat, _ = self.col_attn(pair_flat, pair_flat, pair_flat)
        pair = pair_flat.view(B, L, L, d).permute(0, 2, 1, 3).contiguous()
        pair = residual + pair

        # Feed-forward
        residual = pair
        pair = self.norm3(pair)
        pair = residual + self.ff(pair)
        pair = self.norm4(pair)

        return pair


class BSJPairAnalyzer(nn.Module):
    """
    Specialized analyzer for BSJ-crossing pairs.

    These pairs are unique to circRNA and cannot be predicted by
    linear RNA models. They are critical for circRNA stability
    and function.
    """

    def __init__(self, d_pair: int = 64):
        super().__init__()
        self.bsj_mlp = nn.Sequential(
            nn.Linear(d_pair * 2, d_pair),
            nn.GELU(),
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, 1),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        bsj_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze BSJ-crossing pairs.

        Args:
            pair_repr: (B, L, L, d_pair) pair representations
            bsj_mask: (L, L) binary mask for BSJ-crossing pairs

        Returns:
            Dict with:
                - bsj_pair_count: Expected number of BSJ-crossing pairs
                - bsj_stability_score: Predicted circRNA stability from BSJ pairs
                - bsj_pair_features: (B, L, L) features for BSJ pairs
        """
        B, L, _, d = pair_repr.shape

        # Symmetrize pair features around BSJ
        # P[i, j] for BSJ pair should equal P[i+1, j+1] mod L
        pair_sym = 0.5 * (pair_repr + torch.roll(pair_repr, shifts=(1, 1), dims=(1, 2)))

        # Concatenate forward and reverse BSJ pair features
        # (i paired with j) and (i+1 paired with j+1) should be similar
        bsj_features = self.bsj_mlp(
            torch.cat([pair_sym, torch.roll(pair_sym, shifts=(-1, -1), dims=(1, 2))], dim=-1)
        ).squeeze(-1)  # (B, L, L)

        # Apply BSJ mask
        bsj_pair_features = bsj_features * bsj_mask.unsqueeze(0)

        # Predicted BSJ pair count
        bsj_pair_count = bsj_pair_features.sum(dim=(1, 2)) / 2  # (B,)

        # Stability score: more BSJ pairs → more stable circular structure
        bsj_stability = torch.tanh(bsj_pair_count / 10.0)  # normalized

        return {
            "bsj_pair_count": bsj_pair_count,
            "bsj_stability_score": bsj_stability,
            "bsj_pair_features": bsj_pair_features,
        }
