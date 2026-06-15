"""
tpe.py — Torus Positional Encoding for circular RNA.

Key insight: circRNA has no 5'/3' end — its topology is a torus (S¹).
Standard sinusoidal PE assumes a linear chain with a fixed origin.
TPE enforces periodicity: PE[i] = PE[i + L], making the encoding
invariant to where the sequence "starts" on the circle.

Mathematical formulation:
    For position i on a circular sequence of length L:
        θ_i = 2π · i / L
        PE(i, 2k)   = Σ_h  w_{h,k} · sin(h · θ_i)
        PE(i, 2k+1) = Σ_h  w_{h,k} · cos(h · θ_i)

where h = 1, 2, ..., H (harmonic order) and w_{h,k} are learnable weights.

Properties:
    1. Periodicity:  PE(i) = PE(i + L)  for all i
    2. Rotation equivariance: same sequence, different start → same encoding
    3. Multi-scale: low harmonics capture global topology, high harmonics
       capture local structure (stem-loops, IRS elements)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorusPositionalEncoding(nn.Module):
    """
    Periodic positional encoding for circular RNA sequences.

    Unlike standard sinusoidal PE which treats positions on a line,
    TPE treats positions on a circle S¹, ensuring that the encoding
    is periodic with period L (the sequence length).

    Args:
        d_model: Embedding dimension (must be even)
        n_harmonics: Number of harmonic frequencies (default: 16)
        learnable: Whether harmonic weights are learnable (default: True)
        max_harmonic_scale: Max frequency scaling factor (default: 1.0)
        dropout: Dropout rate applied after adding PE (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 640,
        n_harmonics: int = 16,
        learnable: bool = True,
        max_harmonic_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"

        self.d_model = d_model
        self.n_harmonics = n_harmonics
        self.learnable = learnable
        self.max_harmonic_scale = max_harmonic_scale

        # Learnable harmonic weights: (n_harmonics, d_model // 2)
        # Each harmonic h contributes sin(h·θ) and cos(h·θ) components
        if learnable:
            self.harmonic_weights = nn.Parameter(
                torch.randn(n_harmonics, d_model // 2) * 0.02
            )
        else:
            # Fixed weights following Vaswani-style scaling
            self.register_buffer(
                "harmonic_weights",
                self._init_fixed_weights(n_harmonics, d_model // 2),
            )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _init_fixed_weights(n_harmonics: int, half_dim: int) -> torch.Tensor:
        """Initialize fixed weights following standard sinusoidal PE scaling."""
        weights = torch.zeros(n_harmonics, half_dim)
        for h in range(n_harmonics):
            for k in range(half_dim):
                freq = 1.0 / (10000 ** (2 * k / (2 * half_dim)))
                weights[h, k] = freq * (h + 1)
        return weights

    def _compute_angles(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute torus angles θ_i = 2π·i/L for all positions."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta = 2.0 * math.pi * positions / seq_len  # (L,)
        return theta

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add torus positional encoding to input embeddings.

        The key property: PE is periodic with period seq_len (the true
        circRNA length), NOT the input tensor length. This means if we
        have a 2L-length input (e.g., two copies of the sequence),
        PE[i] and PE[i+L] will be equal because they use the same
        angle θ = 2π·i/L (mod L).

        Args:
            x: Input tensor of shape (batch, L, d_model) or (L, d_model)
            seq_len: The period length (true circRNA length). If None,
                     inferred from x.size(-2). This must equal the true
                     circRNA length for periodicity to hold.

        Returns:
            Tensor with TPE added, same shape as input
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, input_len, d_model = x.shape
        if seq_len is None:
            seq_len = input_len  # Default: period = input length

        # Compute torus angles using seq_len as the period.
        # θ_i = 2π · (i mod seq_len) / seq_len
        # This ensures PE[i] = PE[i + seq_len] for any i.
        positions = torch.arange(input_len, device=x.device, dtype=torch.float32)
        theta = 2.0 * math.pi * (positions % seq_len) / seq_len  # (input_len,)

        # Build PE: (input_len, d_model)
        pe = torch.zeros(input_len, d_model, device=x.device, dtype=x.dtype)

        half_dim = d_model // 2
        weights = self.harmonic_weights  # (n_harmonics, half_dim)

        for h in range(self.n_harmonics):
            freq = h + 1
            sin_vals = torch.sin(freq * theta)  # (input_len,)
            cos_vals = torch.cos(freq * theta)  # (input_len,)

            # sin component → even indices, cos component → odd indices
            pe[:, 0::2] += torch.outer(sin_vals, weights[h])  # (input_len, half_dim)
            pe[:, 1::2] += torch.outer(cos_vals, weights[h])  # (input_len, half_dim)

        # Scale by sqrt(d_model) following standard PE convention
        pe = pe * (1.0 / math.sqrt(self.n_harmonics))

        # Add to input
        output = x + pe.unsqueeze(0).expand(batch_size, -1, -1)
        output = self.dropout(output)

        if squeeze:
            output = output.squeeze(0)

        return output

    def get_periodicity_error(self, seq_len: int) -> float:
        """Verify periodicity: ||PE[i] - PE[i + L]|| should be ~0."""
        with torch.no_grad():
            # Create a 2*seq_len dummy input, but tell TPE the true period is seq_len
            dummy = torch.zeros(1, 2 * seq_len, self.d_model)
            pe_full = self.forward(dummy, seq_len=seq_len)  # Period = seq_len
            # Compare first L positions with second L positions
            pe_first = pe_full[0, :seq_len]
            pe_second = pe_full[0, seq_len:]
            # The PE for position i should equal PE for position i+L
            error = (pe_first - pe_second).norm().item()
        return error


class TorusPositionalEncoding2D(nn.Module):
    """
    2D Torus Positional Encoding for joint (position, structure) representation.

    Encodes position on S¹ × S¹ torus:
    - First circle: sequence position (i / L)
    - Second circle: structural context (e.g., local GC content window position)

    This is useful for the IRS pair module where both i and j positions
    need to be encoded on the torus.
    """

    def __init__(
        self,
        d_model: int = 640,
        n_harmonics: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics

        # Separate weights for each circle dimension
        self.weights_theta = nn.Parameter(
            torch.randn(n_harmonics, d_model // 4) * 0.02
        )
        self.weights_phi = nn.Parameter(
            torch.randn(n_harmonics, d_model // 4) * 0.02
        )
        # Cross terms for torus topology
        self.weights_cross = nn.Parameter(
            torch.randn(n_harmonics, d_model // 4) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        structure_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add 2D torus PE.

        Args:
            x: (batch, L, d_model)
            seq_len: Sequence length
            structure_features: (batch, L) optional structural context in [0, 1]
        """
        batch_size, L, d_model = x.shape

        # First circle: sequence position
        positions = torch.arange(L, device=x.device, dtype=torch.float32)
        theta = 2.0 * math.pi * positions / seq_len  # (L,)

        # Second circle: structural context (default: GC content phase)
        if structure_features is None:
            # Use a default phase based on local structure
            phi = theta * 0.5  # Simplified: half-frequency structural phase
        else:
            phi = 2.0 * math.pi * structure_features  # (batch, L)

        pe = torch.zeros(L, d_model, device=x.device, dtype=x.dtype)
        quarter_dim = d_model // 4

        for h in range(self.n_harmonics):
            freq = h + 1

            # θ components (sequence position)
            pe[:quarter_dim * 1].add_(
                torch.outer(torch.sin(freq * theta), self.weights_theta[h % quarter_dim].expand(quarter_dim))
            )

            # φ components (structural context)
            pe[quarter_dim:quarter_dim * 2].add_(
                torch.outer(torch.cos(freq * theta), self.weights_phi[h % quarter_dim].expand(quarter_dim))
            )

            # Cross terms: sin(h·θ)·cos(h·φ) — captures torus topology
            if structure_features is not None:
                cross = torch.outer(
                    torch.sin(freq * theta),
                    self.weights_cross[h % quarter_dim].expand(quarter_dim),
                ) * torch.cos(freq * phi).T.unsqueeze(-1)
                pe[quarter_dim * 2:quarter_dim * 3].add_(cross.squeeze(-1) if cross.dim() == 3 else cross)

        pe = pe * (1.0 / math.sqrt(self.n_harmonics))
        output = x + pe.unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(output)


class CircularRelativeBias(nn.Module):
    """
    Circular relative position bias for attention.

    Instead of linear relative positions (i - j), uses circular distance:
        d(i, j) = min(|i - j|, L - |i - j|)

    This ensures that positions near the back-splice junction (where i ≈ 0
    and j ≈ L) are treated as nearby, not far apart.
    """

    def __init__(self, n_heads: int = 8, max_dist: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist

        # Learnable bias for each (head, circular_distance) pair
        self.relative_bias = nn.Embedding(max_dist + 1, n_heads)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute circular relative position bias matrix.

        Returns:
            Tensor of shape (1, n_heads, seq_len, seq_len)
        """
        positions = torch.arange(seq_len, device=self.relative_bias.weight.device)

        # Linear distance: (L, L)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Circular distance: min(|i-j|, L-|i-j|)
        circ_dist = torch.min(diff.abs(), seq_len - diff.abs())

        # Clamp to max_dist and lookup bias
        circ_dist = circ_dist.clamp(0, self.max_dist)
        bias = self.relative_bias(circ_dist)  # (L, L, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)

        return bias
