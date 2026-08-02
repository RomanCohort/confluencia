"""
mixed_attention.py — SlidingWindow + GlobalAnchor mixed attention (O(L)).

Shared by both:
  - scheme10_equivariant.py (used inside the model)
  - coord_diffusion.py (the denoiser's cross-attention)

This file is intentionally decoupled from scheme10_equivariant.py to avoid
circular imports.

v4.1: DynamicGlobalAnchorAttention — AF3-style dynamic anchor selection.
  - Learns per-position anchor importance from key features (learned MLP)
  - Per-sample top-K selection + BSJ flank priority
  - Drops uniformly-sampled anchors when structural hotspots exist
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _circular_neighbor_mask(L: int, W: int, device: torch.device) -> torch.Tensor:
    """(L, L) bool mask: token i attends to j iff d_ring(i, j) <= W//2.

    d_ring(i, j) = min(|i-j|, L-|i-j|).  No data copy — pure index math.
    """
    i = torch.arange(L, device=device)
    j = torch.arange(L, device=device)
    diff = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
    circ_dist = torch.minimum(diff, L - diff)
    return circ_dist <= (W // 2)


def _global_anchor_indices(L: int, A: int, bsj_flank: int,
                           device: torch.device) -> torch.Tensor:
    """A anchor positions: BSJ flanks + evenly spaced across interior.

    All indices clamped to [0, L-1].  For short sequences A is capped at L.
    """
    A = min(A, L)
    anchors: set = set()
    half = max(1, bsj_flank)
    for _i in range(min(half, L)):
        anchors.add(_i)
        anchors.add(L - half + _i)
    inner = A - len(anchors)
    if inner > 0 and L > 0:
        for _i in range(inner):
            anchors.add(min(L - 1, (_i + 1) * L // max(inner, 1)))
    return torch.tensor(sorted(anchors)[:A], device=device, dtype=torch.long)


class SlidingWindowAttention(nn.Module):
    """Sliding-window cross-attention: O(L·W)."""

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.window = window
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        device = query.device
        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, L, self.n_heads, self.head_dim)

        scores = torch.einsum("blhd,bkhd->bhkl", q, k) / math.sqrt(self.head_dim)
        mask = _circular_neighbor_mask(L, self.window, device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhkl,bkhd->blhd", attn, v)
        out = out.reshape(B, L, D)
        return self.out_proj(out)


class GlobalAnchorAttention(nn.Module):
    """Global anchor cross-attention with UNIFORM anchor placement: O(L·A)."""

    def __init__(self, d_model: int, n_heads: int = 4, n_anchors: int = 128,
                 bsj_flank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_anchors = n_anchors
        self.bsj_flank = bsj_flank
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        device = query.device
        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim)
        aidx = _global_anchor_indices(L, self.n_anchors, self.bsj_flank, device)
        A = aidx.shape[0]
        ka = self.k_proj(key[:, aidx, :]).view(B, A, self.n_heads, self.head_dim)
        va = self.v_proj(value[:, aidx, :]).view(B, A, self.n_heads, self.head_dim)
        scores = torch.einsum("blhd,bahd->bhal", q, ka) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhal,bahd->blhd", attn, va)
        out = out.reshape(B, L, D)
        return self.out_proj(out)


class DynamicGlobalAnchorAttention(nn.Module):
    """Global anchor cross-attention with DYNAMIC per-sample anchor selection: O(L·A).

    Instead of uniformly-spaced anchors, learns per-position importance scores
    from key features via a small MLP, then selects top-K anchors per sample.
    BSJ flanks are always included regardless of score.

    Inspired by AlphaFold 3's dynamic anchor idea — anchors concentrate in
    structural hotspot regions (e.g., long-range pseudoknots) rather than
    being wasted on low-complexity stretches.

    Supervision (auxiliary loss):
      Top-K selection is discrete and non-differentiable, so the scorer is
      supervised DIRECTLY by an external hotspot signal (pair_probs) via
      `anchor_aux_loss(pair_probs)`. This gives a strong, immediate gradient
      to the scorer — no Gumbel-Softmax / straight-through needed.

      The caller computes:
          hotspot = pair_probs.sum(dim=-1)   # per-position pairing strength
          aux = F.mse_loss(scorer(key).squeeze(-1), hotspot)
      and adds `aux * weight` to the total loss.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_anchors: int = 128,
                 bsj_flank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_anchors = n_anchors
        self.bsj_flank = bsj_flank
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

        # Learned anchor importance scorer: key → scalar score per position.
        # Supervised by anchor_aux_loss() against pair_probs hotspot signal.
        self.anchor_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # [v4 LAMA] Learnable 3-way blend weights for the locality-aware hotspot
        # target: [pair_hotspot, neighbor_density, local_connectivity]. Init
        # uniform (1/3 each) so the model discovers the best blend.
        self.lama_weights = nn.Parameter(torch.zeros(3))

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, L, D = query.shape
        device = query.device
        q = self.q_proj(query).view(B, L, self.n_heads, self.head_dim)

        # Per-position anchor importance from key features
        anchor_scores = self.anchor_scorer(key).squeeze(-1)  # [B, L]

        # Hard top-K anchor selection (no grad through indices)
        hard_idx = self._select_dynamic_anchors(anchor_scores, L, device)  # [B, A]
        A = hard_idx.shape[1]
        D_idx = hard_idx.unsqueeze(-1).expand(-1, -1, D)
        ka = self.k_proj(key.gather(1, D_idx)).view(B, A, self.n_heads, self.head_dim)
        va = self.v_proj(value.gather(1, D_idx)).view(B, A, self.n_heads, self.head_dim)

        scores = torch.einsum("blhd,bahd->bhal", q, ka) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhal,bahd->blhd", attn, va)
        out = out.reshape(B, L, D)
        return self.out_proj(out)

    def anchor_aux_loss(self, key: torch.Tensor,
                        pair_probs: torch.Tensor) -> torch.Tensor:
        """LAMA-style auxiliary loss: supervise the scorer with a 3-way fused
        locality-aware hotspot signal (inspired by Mac-Diff's LAMA-attention).

        The target per position is a weighted fusion of:
          (1) pair_hotspot     — row-sum of pair_probs (long-range pairing strength)
          (2) neighbor_density — 1D Gaussian-weighted sum of nearby hotspots
                                 (local structural complexity: stems peak, linkers low)
          (3) local_connectivity — hotspot × neighbor_density (true structural hubs,
                                 e.g. pseudoknot crossing points)

        This replaces the single-hotspot supervision. The scorer still only sees
        `key` features at inference (no pair_probs in forward) — the 3-way signal
        is the richer training target that the scorer learns to predict. Weights
        α/β/γ are learnable (softmax-normalized) so the model discovers the blend.

        Args:
            key:        [B, L, D]
            pair_probs: [B, L, L]
        Returns: scalar MSE loss.
        """
        scores = self.anchor_scorer(key).squeeze(-1)  # [B, L]
        B, L, _ = pair_probs.shape
        device = pair_probs.device

        # (1) Pair hotspot: long-range pairing strength per position
        hotspot = pair_probs.sum(dim=-1)  # [B, L]

        # (2) Neighbor density: 1D Gaussian-weighted local hotspot sum.
        # G[i,j] = exp(-(i-j)^2 / 2h^2), h = max(3, L*0.02) — narrow bandwidth so
        # only ~1 stem-width neighbors count (Mac-Diff uses h=2 for proteins).
        h = max(3.0, L * 0.02)
        idx = torch.arange(L, device=device, dtype=torch.float32)
        diff = (idx.unsqueeze(0) - idx.unsqueeze(1))  # [L, L]
        gauss = torch.exp(-(diff ** 2) / (2.0 * h * h))  # [L, L]
        gauss = gauss / gauss.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # row-norm
        neighbor_density = torch.bmm(hotspot.unsqueeze(1), gauss.unsqueeze(0).expand(B, -1, -1)).squeeze(1)  # [B, L]

        # (3) Local connectivity: hub score = own pairing × local density
        # (positions that both pair strongly AND sit in a dense region = pseudoknots)
        local_connectivity = hotspot * neighbor_density

        # Normalize each channel per-sample to [0, 1] (stable across lengths)
        def norm01(t):
            m = t.amax(dim=-1, keepdim=True).clamp(min=1e-8)
            return t / m
        h_n, d_n, c_n = norm01(hotspot), norm01(neighbor_density), norm01(local_connectivity)

        # Learnable 3-way blend (softmax so weights sum to 1)
        w = torch.softmax(self.lama_weights, dim=0)  # [3]
        target = w[0] * h_n + w[1] * d_n + w[2] * c_n  # [B, L]

        scores_norm = torch.sigmoid(scores)
        return F.mse_loss(scores_norm, target)

    @torch.no_grad()
    def _select_dynamic_anchors(self, scores: torch.Tensor, L: int,
                                device: torch.device) -> torch.Tensor:
        """Per-sample top-K anchor selection with BSJ flank priority.

        Returns: [B, A] long tensor of anchor indices.
        """
        B = scores.shape[0]
        A = min(self.n_anchors, L)
        half = max(1, self.bsj_flank)

        # BSJ flanks: always included
        bsj_set: list[int] = []
        for i in range(min(half, L)):
            bsj_set.append(i)
            bsj_set.append(L - half + i)
        bsj_set = sorted(set(bsj_set))
        bsj_tensor = torch.tensor(bsj_set, device=device, dtype=torch.long)
        n_bsj = len(bsj_set)

        remaining = A - n_bsj
        if remaining <= 0:
            return bsj_tensor.unsqueeze(0).expand(B, -1).contiguous()

        # Mask out BSJ positions from dynamic selection
        mask = torch.ones(L, dtype=torch.bool, device=device)
        for i in bsj_set:
            mask[i] = False

        masked_scores = scores.masked_fill(~mask, float('-inf'))  # [B, L]
        topk_idx = masked_scores.topk(remaining, dim=-1).indices  # [B, remaining]

        anchors = torch.cat([
            bsj_tensor.unsqueeze(0).expand(B, -1),
            topk_idx,
        ], dim=-1)
        return anchors


class MixedHybridAttention(nn.Module):
    """Sliding-window + global-anchor mixed attention.

    Default 4-layer (local ↔ global interleaved):
      L0: SlidingWindow(W=256)  L1: GlobalAnchor(A=128)  [or Dynamic if enabled]
      L2: SlidingWindow(W=256)  L3: GlobalAnchor(A=128)  [or Dynamic if enabled]
    Each layer has residual + LayerNorm.

    use_dynamic_anchors=True: odd layers use DynamicGlobalAnchorAttention
      (per-sample top-K anchor selection from learned importance scores).
    """

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4,
                 window: int = 256, n_anchors: int = 128, bsj_flank: int = 32,
                 dropout: float = 0.1, use_dynamic_anchors: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_dynamic = use_dynamic_anchors
        for i in range(n_layers):
            if i % 2 == 0:
                self.layers.append(SlidingWindowAttention(
                    d_model, n_heads, window=window, dropout=dropout))
            else:
                if use_dynamic_anchors:
                    self.layers.append(DynamicGlobalAnchorAttention(
                        d_model, n_heads, n_anchors=n_anchors,
                        bsj_flank=bsj_flank, dropout=dropout))
                else:
                    self.layers.append(GlobalAnchorAttention(
                        d_model, n_heads, n_anchors=n_anchors,
                        bsj_flank=bsj_flank, dropout=dropout))
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = query
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(query, key, value)
            x = norm(x)
        return x
