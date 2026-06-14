"""
equivariant_backbone.py — Rotation-equivariant backbone for circular RNA.

circRNA has no canonical start position — the same molecule can be
sequenced starting from any nucleotide. This module ensures the
model's representation is equivariant to this circular rotation.

Two strategies:
1. TPE Injection: Replace standard PE with Torus PE in the backbone
2. Rotation Augmentation: Average embeddings over multiple rotation
   starting points (more expensive but guarantees invariance)

The module wraps RNA-FM (or any ESM-based backbone) and injects
torus-aware position information.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpe import TorusPositionalEncoding, CircularRelativeBias


class TorusTransformerLayer(nn.Module):
    """
    Single transformer layer with torus positional encoding and
    circular relative position bias.

    Replaces the standard attention mechanism with:
    - Circular relative position bias (instead of linear)
    - TPE-augmented queries/keys
    """

    def __init__(
        self,
        d_model: int = 640,
        n_heads: int = 10,
        d_ff: int = 2560,
        dropout: float = 0.1,
        max_circ_dist: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention with circular relative bias
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.circ_bias = CircularRelativeBias(n_heads, max_dist=max_circ_dist)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, L, d_model) — already TPE-augmented
            seq_len: Explicit sequence length for circular bias
        """
        if seq_len is None:
            seq_len = x.size(1)

        residual = x
        x = self.norm1(x)

        batch_size, L, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with circular bias
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # Add circular relative position bias
        circ_bias = self.circ_bias(seq_len)  # (1, H, L, L)
        attn = attn + circ_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, L, self.d_model)
        out = self.out_proj(out)

        x = residual + out

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x


class CircEquivariantBackbone(nn.Module):
    """
    Rotation-equivariant backbone for circular RNA.

    Architecture:
    1. RNA-FM backbone → frozen embeddings (640-dim)
    2. TPE injection → add torus positional encoding
    3. TorusTransformer layers → refine with circular attention
    4. Optional: rotation augmentation for guaranteed invariance

    Args:
        d_model: Model dimension (default: 640, matching RNA-FM)
        n_torus_layers: Number of torus transformer layers (default: 4)
        n_heads: Attention heads per layer (default: 10)
        d_ff: Feed-forward dimension (default: 2560)
        n_harmonics: TPE harmonic count (default: 16)
        n_rot_augments: Rotation augmentation count (0 = disabled)
        dropout: Dropout rate (default: 0.1)
        max_circ_dist: Max circular distance for relative bias (default: 128)
    """

    def __init__(
        self,
        d_model: int = 640,
        n_torus_layers: int = 4,
        n_heads: int = 10,
        d_ff: int = 2560,
        n_harmonics: int = 16,
        n_rot_augments: int = 0,
        dropout: float = 0.1,
        max_circ_dist: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rot_augments = n_rot_augments

        # TPE layer
        self.tpe = TorusPositionalEncoding(
            d_model=d_model,
            n_harmonics=n_harmonics,
            dropout=dropout,
        )

        # Torus transformer layers
        self.torus_layers = nn.ModuleList([
            TorusTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_circ_dist=max_circ_dist,
            )
            for _ in range(n_torus_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Backbone placeholder (loaded from RNA-FM)
        self.backbone = None
        self.alphabet = None
        self.backbone_loaded = False

    def load_backbone(self, model_path: Optional[str] = None, device: str = "cpu"):
        """Load RNA-FM / ESM2 backbone (frozen)."""
        if self.backbone_loaded:
            return

        try:
            import esm
        except ImportError:
            raise ImportError("RNA-FM not installed. Run: pip install fair-esm")

        if model_path:
            from pathlib import Path
            if Path(model_path).exists():
                self.backbone, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
            else:
                self.backbone, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            self.backbone, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone = self.backbone.to(device)
        self.backbone_loaded = True

    def _encode_with_backbone(
        self,
        sequences: List[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Get frozen backbone embeddings for a batch of sequences."""
        if not self.backbone_loaded:
            self.load_backbone(device=device)

        # U → T for ESM compatibility
        seqs_t = [s.replace("U", "T").replace("u", "t") for s in sequences]

        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([
            (f"seq_{i}", s) for i, s in enumerate(seqs_t)
        ])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.backbone(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False,
            )
            embeddings = results["representations"][33]

            # Mean pooling over non-padding positions
            mask = (batch_tokens != self.alphabet.padding_idx).float().unsqueeze(-1)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled  # (batch, d_model)

    @staticmethod
    def _rotate_sequence(seq: str, offset: int) -> str:
        """Rotate a circular sequence by offset positions."""
        offset = offset % len(seq)
        if offset == 0:
            return seq
        return seq[offset:] + seq[:offset]

    def forward(
        self,
        sequences: List[str],
        seq_lengths: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with rotation equivariance.

        Args:
            sequences: List of circRNA sequences (ACGU format)
            seq_lengths: Explicit sequence lengths (inferred if None)
            device: Device

        Returns:
            Dict with:
                - embedding: (batch, d_model) — rotation-invariant embedding
                - sequence_repr: (batch, L, d_model) — per-position representations
                - rotation_augmented: bool — whether rotation aug was used
        """
        batch_size = len(sequences)

        if self.n_rot_augments > 0:
            # Strategy 2: Rotation augmentation
            all_embs = []

            for k in range(self.n_rot_augments):
                # Rotate each sequence
                rotated = [
                    self._rotate_sequence(s, k * len(s) // self.n_rot_augments)
                    for s in sequences
                ]

                # Get per-position embeddings from backbone
                if not self.backbone_loaded:
                    self.load_backbone(device=device)

                seqs_t = [s.replace("U", "T").replace("u", "t") for s in rotated]
                batch_converter = self.alphabet.get_batch_converter()
                _, _, batch_tokens = batch_converter([
                    (f"seq_{i}", s) for i, s in enumerate(seqs_t)
                ])
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = self.backbone(
                        batch_tokens,
                        repr_layers=[33],
                        return_contacts=False,
                    )
                    token_emb = results["representations"][33]  # (B, L+2, d)

                    # Remove BOS/EOS tokens
                    token_emb = token_emb[:, 1:-1, :]  # (B, L, d)

                # Add TPE
                L = token_emb.size(1)
                token_emb = self.tpe(token_emb, seq_len=L)

                # Apply torus transformer layers
                for layer in self.torus_layers:
                    token_emb = layer(token_emb, seq_len=L)

                # Un-rotate: shift back to original positions
                for i, seq in enumerate(sequences):
                    offset = k * len(seq) // self.n_rot_augments
                    if offset > 0:
                        token_emb[i] = torch.roll(token_emb[i], shifts=-offset, dims=0)

                all_embs.append(token_emb)

            # Average over rotation augmentations
            sequence_repr = torch.stack(all_embs).mean(dim=0)  # (B, L, d)

        else:
            # Strategy 1: TPE injection only (no rotation aug)
            if not self.backbone_loaded:
                self.load_backbone(device=device)

            seqs_t = [s.replace("U", "T").replace("u", "t") for s in sequences]
            batch_converter = self.alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([
                (f"seq_{i}", s) for i, s in enumerate(seqs_t)
            ])
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = self.backbone(
                    batch_tokens,
                    repr_layers=[33],
                    return_contacts=False,
                )
                token_emb = results["representations"][33]
                token_emb = token_emb[:, 1:-1, :]  # Remove BOS/EOS

            L = token_emb.size(1)
            token_emb = self.tpe(token_emb, seq_len=L)

            for layer in self.torus_layers:
                token_emb = layer(token_emb, seq_len=L)

            sequence_repr = token_emb

        # Final norm
        sequence_repr = self.final_norm(sequence_repr)

        # Global embedding: mean pooling
        global_emb = sequence_repr.mean(dim=1)  # (B, d_model)

        return {
            "embedding": global_emb,
            "sequence_repr": sequence_repr,
            "rotation_augmented": self.n_rot_augments > 0,
        }

    def get_rotation_invariance_error(
        self,
        sequence: str,
        device: str = "cpu",
    ) -> float:
        """
        Measure rotation invariance: same sequence with different
        starting points should give similar embeddings.

        Returns:
            Max cosine distance between rotation pairs
        """
        L = len(sequence)
        n_test = min(4, L)

        embs = []
        for k in range(n_test):
            offset = k * L // n_test
            rotated = self._rotate_sequence(sequence, offset)
            result = self.forward([rotated], device=device)
            embs.append(result["embedding"])

        # Compute pairwise cosine distances
        max_dist = 0.0
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                cos_sim = F.cosine_similarity(embs[i], embs[j], dim=-1).item()
                dist = 1.0 - cos_sim
                max_dist = max(max_dist, dist)

        return max_dist