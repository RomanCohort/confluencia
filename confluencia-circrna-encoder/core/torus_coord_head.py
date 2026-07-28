"""
torus_coord_head.py — TorusFold v2 Scheme 9: Torus-parameterized structure head.

Architectural upgrade (see project-torusfold-arch-v2.md, Change 1):

    Replace SE(3) free-particle representation with explicit torus coordinates
        (θ, φ, r) ∈ S¹ × S¹ × R⁺
    so that the closure constraint becomes a STRUCTURAL HARD GUARANTEE
    rather than a soft loss term.

    θ_i ∈ [0, 2π)   — major ring angle (circRNA closed loop around main axis)
    φ_i ∈ [0, 2π)   — cross-section angle (double helix around cross-section center)
    r_i ∈ R⁺        — cross-section radius

    Closure: θ_{N+1} = θ_1 (mod 2π) — guaranteed by construction, not by loss.

This is registered as structure_mode="torus" in TorusFoldConfig, making it
Scheme 9 alongside the existing S1–S7 (simple/diffusion/physics_b/physics_ba).

Why a separate head (not a full backbone replacement):
    The existing ESM2 + TPE + CircPairformer trunk (S6 converged at val=0.435,
    per project_torusfold) is preserved. TorusCoordHead sits on top of the
    refined pair representation as a new structure_mode option, allowing A/B
    comparison without destroying the S6 checkpoint.

Equivalence:
    SE(3) → SE(3) ⋊ U(1)₂  (torus rotation + spatial rigid body)
    The kabsch cap loss is removed; r-norm constraint prevents cross-section
    collapse instead.

References:
    - project-torusfold-arch-v2.md (Change 1: TorusBackbone)
    - feedback_torusfold_training.md (normalized-space training rationale)
    - torusfold-immune-fingerprints.md (downstream 5 fingerprints)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorusCoordHead(nn.Module):
    """
    Scheme 9 structure head: torus-parameterized 3D coordinates.

    Given the refined pair representation from CircPairformer, predict
    per-residue torus coordinates (θ, φ, r) and convert to Cartesian (x, y, z).

    Closure is structurally guaranteed: the last residue reuses the first
    residue's θ (mod 2π), so coords[:, 0] ≡ coords[:, -1] by construction.

    Args:
        d_pair: Pair representation dim (c_z, default 128)
        d_hidden: Hidden dim of the torus projection MLP
        n_rbf: RBF bins for an auxiliary distance prediction (for compatibility
            with SimpleStructureHead's dist_pred output)
        r_scale: Soft scale on the cross-section radius. r = r_scale * softplus(...)
            Keep < 1.0 to encourage compact cross-sections.
        bond_length: Adjacent-phosphate bond length (Å), used for the bond loss
            reported in forward (not used internally — caller may consume it).
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 256,
        n_rbf: int = 16,
        r_scale: float = 0.5,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.d_pair = d_pair
        self.n_rbf = n_rbf
        self.r_scale = r_scale
        self.bond_length = bond_length

        # Per-position feature: aggregate pair_repr over j → (B, L, d_pair)
        # Then project to (θ, φ, r) torus coordinates.
        self.torus_proj = nn.Sequential(
            nn.Linear(d_pair, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 3),  # (θ_pre, φ_pre, r_pre)
        )

        # Auxiliary distance head (kept for output contract compatibility
        # with SimpleStructureHead: dist_pred in (B, L, L)).
        self.dist_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, n_rbf),
        )
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.5, 30.0, n_rbf),
        )

        # Confidence head (per-residue, like SimpleStructureHead's confidence).
        self.confidence = nn.Sequential(
            nn.Linear(d_pair, d_pair // 2),
            nn.GELU(),
            nn.Linear(d_pair // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pair_repr: torch.Tensor,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pair_repr: (B, L, L, d_pair) — refined by CircPairformer
            pair_probs: optional (B, L, L) base-pair probabilities, used to
                modulate φ (cross-section angle) so that paired residues sit
                on opposite sides of the cross-section. None → no modulation.

        Returns:
            Dict matching SimpleStructureHead's contract:
                coords: (B, L, 3) Cartesian coordinates
                dist_pred: (B, L, L) predicted pairwise distances (auxiliary)
                confidence: (B, L) per-residue confidence in [0, 100]
                closure_dist: (B,) — should be ≈ 0 by construction
                torus_coords: (B, L, 3) raw (θ, φ, r) for downstream
                              immune-fingerprint heads
                closure_loss: (B,) hard-closure residual (should be ~0)
        """
        B, L, _, d = pair_repr.shape
        device = pair_repr.device

        # 1. Aggregate pair_repr over j → per-residue features.
        #    Mean over axis 2 (the j axis) gives a per-i summary.
        per_res = pair_repr.mean(dim=2)  # (B, L, d_pair)

        # 2. Predict raw torus coordinates.
        raw = self.torus_proj(per_res)  # (B, L, 3)
        theta_pre = raw[..., 0]
        phi_pre = raw[..., 1]
        r_pre = raw[..., 2]

        # 3. Map to constrained ranges.
        #    θ ∈ [-π, π] (S¹), φ ∈ [-π, π] (S¹), r ∈ R⁺.
        theta = torch.tanh(theta_pre) * math.pi
        phi = torch.tanh(phi_pre) * math.pi
        r = self.r_scale * F.softplus(r_pre)  # R⁺, bounded by r_scale * log(2)

        # 4. Optional pair-probability modulation on φ:
        #    if residue i is paired with j, place them on opposite sides of
        #    the cross-section (φ_j ≈ φ_i + π). Soft modulation, not hard.
        if pair_probs is not None:
            # Find strongest partner for each i
            partner = pair_probs.argmax(dim=-1)  # (B, L)
            partner_phi = torch.gather(phi, 1, partner)  # (B, L)
            # Soft pull towards partner_phi + π
            target_phi = partner_phi + math.pi
            # Wrap target to [-π, π]
            target_phi = torch.atan2(
                torch.sin(target_phi), torch.cos(target_phi)
            )
            phi = 0.7 * phi + 0.3 * target_phi

        # 5. Hard closure: force the last residue's (θ, φ, r) to equal the
        #    first's, so coords[:, 0] ≡ coords[:, -1] by construction.
        #    This is the structural guarantee — closure_dist will be ~0
        #    regardless of training dynamics, replacing the kabsch cap loss.
        theta = torch.cat([theta[..., :-1], theta[..., 0:1]], dim=-1)
        phi = torch.cat([phi[..., :-1], phi[..., 0:1]], dim=-1)
        r = torch.cat([r[..., :-1], r[..., 0:1]], dim=-1)

        # 6. Convert (θ, φ, r) → (x, y, z).
        #    Major ring in the xy-plane; cross-section rotation φ tilts the
        #    residue out of plane. This is a coarse-grained model — one
        #    point per nucleotide (C1' or P proxy).
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        cos_p = torch.cos(phi)
        sin_p = torch.sin(phi)

        # Major-ring radius = R + r * cos(φ); out-of-plane = r * sin(φ).
        # We set R = bond_length * L / (2π) so that the major ring closes
        # naturally at the given bond length.
        R = self.bond_length * L / (2.0 * math.pi)
        major_R = R + r * cos_p
        x = major_R * cos_t
        y = major_R * sin_t
        z = r * sin_p
        coords = torch.stack([x, y, z], dim=-1)  # (B, L, 3)

        # 7. Closure distance (should be ~0 by construction).
        closure_dist = (coords[:, 0] - coords[:, -1]).norm(dim=-1)  # (B,)

        # 8. Auxiliary distance prediction (for contract compatibility).
        dist_logits = self.dist_head(pair_repr)  # (B, L, L, n_rbf)
        dist_probs = F.softmax(dist_logits, dim=-1)
        dist_pred = (dist_probs * self.rbf_centers).sum(dim=-1)  # (B, L, L)
        dist_pred = 0.5 * (dist_pred + dist_pred.transpose(-1, -2))

        # 9. Per-residue confidence (scaled to [0, 100]).
        confidence = self.confidence(per_res).squeeze(-1) * 100.0  # (B, L)

        # 10. Hard-closure residual (for diagnostics — should be ~0).
        closure_residual = closure_dist.detach()

        return {
            "coords": coords,
            "dist_pred": dist_pred,
            "confidence": confidence,
            "closure_dist": closure_dist,
            "torus_coords": torch.stack(
                [theta, phi, r], dim=-1
            ),  # (B, L, 3) raw torus
            "closure_loss": closure_residual,  # diagnostic, ~0 by construction
            "structure_method": "torus",
        }


class RIGIWalkAttention(nn.Module):
    """
    Scheme 9 RIG-I CTD walk-attention head (Change 4 in
    project-torusfold-arch-v2.md).

    Models the ATPase-driven 5'→3' translocation of RIG-I's C-terminal
    domain along the RNA backbone. RIG-I nominally recognizes 5'-ppp
    ends, but recent work shows it can also scan internal dsRNA regions
    via ATPase stepwise translocation.

    For circRNA (no free 5' end), this pathway is naturally attenuated —
    so this head serves as a NEGATIVE CONTROL: its output should be
    significantly lower on circular RNAs than on linear RNAs with the
    same internal dsRNA content. This makes it a useful regularizer and
    a sanity-check head during training.

    Implementation:
        sliding-window self-attention with stride = window // 2
        Each window is processed by a small multi-head attention block
        that predicts a per-position "internal-dsRNA accessibility" score.
        The scores are then pooled to a single per-molecule scalar.

    Args:
        d_model: Trunk per-position feature dim
        d_hidden: Hidden dim of the per-window attention block
        window: Sliding window size (ATPase step length, ~15 nt)
        n_heads: Attention heads inside each window
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 128,
        window: int = 15,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.stride = max(1, window // 2)

        # Per-window self-attention block (simulates RIG-I CTD scanning
        # one dsRNA segment at a time).
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_model,
            vdim=d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Score head: per-position internal-dsRNA accessibility logit.
        self.score = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 1),
        )

        # Global pool to a single per-molecule RIG-I activation score.
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        sequence_repr: torch.Tensor,  # (B, L, d_model)
        mask: Optional[torch.Tensor] = None,  # (B, L) 1=valid, 0=pad
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence_repr: (B, L, d_model) per-position trunk features.
            mask: (B, L) valid-position mask (1 for nucleotide, 0 for pad).

        Returns:
            Dict with:
                rigi_per_pos: (B, L) per-position internal-dsRNA accessibility
                rigi_score: (B,) molecule-level RIG-I activation score
        """
        B, L, d = sequence_repr.shape
        device = sequence_repr.device

        # Sliding-window attention. For each window we run self-attention
        # and write the refined representation back to the center position.
        # We accumulate contributions in a sum buffer and normalize by count.
        accum = torch.zeros_like(sequence_repr)
        count = torch.zeros(B, L, 1, device=device)

        # Build key padding mask per window (True = padding to ignore in MHA).
        for start in range(0, max(L - self.window + 1, 1), self.stride):
            end = min(start + self.window, L)
            window_len = end - start
            window = sequence_repr[:, start:end, :]  # (B, w, d)

            # Key padding mask for MHA: True = ignore
            if mask is not None:
                key_padding_mask = (mask[:, start:end] == 0)  # (B, w)
                # If an entire window is padding, MHA would NaN; skip it.
                if key_padding_mask.all(dim=-1).any():
                    # Replace all-pad windows with a no-op by setting
                    # key_padding_mask to all-False (attend to self only).
                    all_pad_rows = key_padding_mask.all(dim=-1)  # (B,)
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_pad_rows] = False
            else:
                key_padding_mask = None

            refined, _ = self.attn(
                window, window, window,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            refined = self.norm(window + refined)

            # Scatter-add refined representations back to their positions.
            accum[:, start:end, :] += refined
            count[:, start:end, :] += 1.0

        # Avoid division by zero for positions not covered by any window
        # (can happen at the very end when L < window or stride alignment).
        count = count.clamp(min=1.0)
        walked = accum / count  # (B, L, d)

        # Per-position score → accessibility logit
        per_pos_logit = self.score(walked).squeeze(-1)  # (B, L)
        per_pos = torch.sigmoid(per_pos_logit)

        # Molecule-level score: masked mean of per-position scores,
        # then a final scalar projection. This is the RIG-I activation
        # score — expected to be LOW for circRNA (negative control).
        if mask is not None:
            denom = mask.sum(dim=-1).clamp(min=1.0)
            pooled_feat = (walked * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        else:
            pooled_feat = walked.mean(dim=1)
        rigi_score = torch.sigmoid(self.pool(pooled_feat).squeeze(-1))  # (B,)

        return {
            "rigi_per_pos": per_pos,
            "rigi_score": rigi_score,
        }


class ImmuneFingerprintHeads(nn.Module):
    """
    Scheme 9 multi-task immune-fingerprint heads (Change 2 in
    project-torusfold-arch-v2.md).

    Sits on top of the shared trunk (sequence_repr + pair_repr + torus_coords)
    and predicts 5 structural immune-activity fingerprints in an end-to-end
    differentiable fashion. Each head is independently toggleable so missing
    data does not block training of the others.

    The 5 fingerprints (see torusfold-immune-fingerprints.md):
        pkr:     long-stem ratio + SASA  → PKR activation
        nlrp3:   persistence length       → NLRP3 scaffold
        drach:   DRACH × in_loop × SASA   → m6A shielding
        tlr7:    GU-rich single-loop      → TLR7 activation (auxiliary;
                                           full TLR7 modeling is post-hoc)
        sponge:  miRNA duplex distance    → sponge potency

    Forward returns a dict of tensors; missing labels are simply not used
    in the loss (caller computes per-head losses only for present targets).

    --- Change 5: m6A exposure-proxy routing (orbit C) ---

    `enable_fingerprint_2d` is a NARROW switch, not a whole-head 2D routing.
    It only swaps the *solvent-exposure* term inside the DRACH head:

        m6a_write_prob = is_drach * in_loop * exposure_proxy
        exposure_proxy = sigmoid(torus_coords[..., 2])      if 2D=False  [3D]
        exposure_proxy = 1 - pair_probs.mean(dim=2)          if 2D=True   [2D]

    `is_drach` and `in_loop` are ALWAYS predicted from the shared trunk feat
    (which contains torus_coords). The question this switch isolates is:
    "does the solvent-exposure sub-quantity need 3D, or is a 2D single-
    strandedness proxy sufficient?" — NOT "does the whole m6A head need 3D".
    See TorusFoldConfig.enable_fingerprint_2d for the full rationale
    (circularity of self-generated SASA labels on circRNA).
    """

    def __init__(
        self,
        d_model: int,
        c_z: int,
        d_torus: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        enable_pkr: bool = True,
        enable_nlrp3: bool = True,
        enable_drach: bool = True,
        enable_tlr7: bool = True,
        enable_sponge: bool = True,
        enable_rigi: bool = False,  # Change 4: RIG-I walk-attention (negative control)
        rigi_window: int = 15,
        rigi_d_hidden: int = 128,
        enable_fingerprint_2d: bool = False,  # Change 5: 2D-only mode for m6A (orbit C)
    ):
        super().__init__()
        self.enable_pkr = enable_pkr
        self.enable_nlrp3 = enable_nlrp3
        self.enable_drach = enable_drach
        self.enable_tlr7 = enable_tlr7
        self.enable_sponge = enable_sponge
        self.enable_rigi = enable_rigi
        self.enable_fingerprint_2d = enable_fingerprint_2d

        in_dim = d_model + c_z + d_torus

        def _mlp(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        if enable_pkr:
            # 2 outputs: stem_length_bin (logits over bins), sasa (regression)
            self.pkr_head = _mlp(2)
        if enable_nlrp3:
            # 1 output: persistence length (regression, nm)
            self.nlrp3_head = _mlp(1)
        if enable_drach:
            # 2 outputs: is_DRACH logit, in_loop logit
            self.drach_head = _mlp(2)
        if enable_tlr7:
            # 1 output: GU-rich loop density (regression)
            self.tlr7_head = _mlp(1)
        if enable_sponge:
            # 1 output: duplex-compat score (regression)
            self.sponge_head = _mlp(1)
        if enable_rigi:
            self.rigi_head = RIGIWalkAttention(
                d_model=d_model,
                d_hidden=rigi_d_hidden,
                window=rigi_window,
                n_heads=4,
                dropout=dropout,
            )

    def forward(
        self,
        sequence_repr: torch.Tensor,  # (B, L, d_model)
        pair_repr: torch.Tensor,       # (B, L, L, c_z)
        torus_coords: torch.Tensor,    # (B, L, 3) raw (θ, φ, r)
        pair_probs: Optional[torch.Tensor] = None,  # (B, L, L) optional; used for 2D m6A exposure proxy
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = sequence_repr.shape
        per_res_pair = pair_repr.mean(dim=2)  # (B, L, c_z)
        feat = torch.cat(
            [sequence_repr, per_res_pair, torus_coords], dim=-1
        )  # (B, L, in_dim)

        out: Dict[str, torch.Tensor] = {}

        if self.enable_pkr:
            pkr = self.pkr_head(feat)  # (B, L, 2)
            out["pkr_stem_logit"] = pkr[..., 0]
            out["pkr_sasa"] = torch.sigmoid(pkr[..., 1])

        if self.enable_nlrp3:
            # Aggregate globally: persistence length is a whole-molecule scalar.
            global_feat = feat.mean(dim=1)  # (B, in_dim)
            out["nlrp3_persistence_length"] = self.nlrp3_head(global_feat).squeeze(-1)

        if self.enable_drach:
            drach = self.drach_head(feat)  # (B, L, 2)
            is_drach = torch.sigmoid(drach[..., 0])
            in_loop = torch.sigmoid(drach[..., 1])
            # m6A write probability = DRACH ∧ in_loop ∧ exposure.
            # Scheme 9 (3D): sasa_proxy = torus_coords[..., 2] (geometric radius).
            # Orbit C (2D): exposure_proxy = 1 - pair_probs (single-strandedness).
            if self.enable_fingerprint_2d and pair_probs is not None:
                # 2D proxy: paired regions are buried, unpaired are exposed.
                # per-residue pairing = mean over j of pair_probs (B,L) (0=unpaired).
                pair_per_res = pair_probs.mean(dim=2)  # (B, L)
                exposure_proxy = 1.0 - pair_per_res  # 0 (paired) -> 1 (exposed)
            else:
                # 3D proxy: torus radius r is a crude SASA proxy.
                # Also the fallback when 2D mode is on but pair_probs is missing
                # (e.g. caller using an older forward() signature).
                exposure_proxy = torch.sigmoid(torus_coords[..., 2])
            out["drach_is_drach"] = is_drach
            out["drach_in_loop"] = in_loop
            out["m6a_write_prob"] = is_drach * in_loop * exposure_proxy

        if self.enable_tlr7:
            out["tlr7_gu_density"] = torch.sigmoid(
                self.tlr7_head(feat).squeeze(-1)
            )

        if self.enable_sponge:
            global_feat = feat.mean(dim=1)
            out["sponge_score"] = torch.sigmoid(
                self.sponge_head(global_feat).squeeze(-1)
            )

        if self.enable_rigi:
            # RIG-I walk-attention uses the raw sequence_repr (d_model dim),
            # not the concatenated feat, because MultiheadAttention expects
            # a fixed embed_dim. Negative-control head: circRNA (no 5'-ppp)
            # should produce a lower score than linear RNA with the same
            # internal dsRNA content.
            rigi_out = self.rigi_head(sequence_repr, mask=None)
            out["rigi_per_pos"] = rigi_out["rigi_per_pos"]
            out["rigi_score"] = rigi_out["rigi_score"]

        return out


# ----------------------------------------------------------------------
# Differentiable secondary-structure layer (P2 / 建议 2)
# ----------------------------------------------------------------------

# Base-pair compatibility matrix over the (A, U, G, C) alphabet.
# Watson-Crick pairs (A-U, G-C) score +1.0; G-U wobble +0.5; else 0.
# This is the ONLY thermodynamic prior baked in — everything else is
# learned via the per-pair MLP on top.
_BP_COMPAT = torch.tensor([
    # A    U    G    C
    [0.0, 1.0, 0.0, 0.0],  # A
    [1.0, 0.0, 0.5, 0.0],  # U
    [0.0, 0.5, 0.0, 1.0],  # G
    [0.0, 0.0, 1.0, 0.0],  # C
], dtype=torch.float32)


class DifferentiableSecondaryStructure(nn.Module):
    """
    Differentiable RNA secondary-structure folding layer.

    Problem it solves (scientific honesty — see evolution.py
    `_saliency_hotspots`): the main TorusFold trunk uses a FROZEN ESM2
    backbone over DISCRETE tokens, so there is no differentiable path
    from token identity to fitness. This module provides a parallel,
    fully-differentiable path that consumes continuous token
    probabilities (a relaxed one-hot) and produces a soft pairing matrix
    plus structural features, yielding a TRUE ∂fitness/∂sequence_token.

    Architecture (MXfold2 / NuSpeak-inspired, lightweight):
      1. token_logits (B, L, 4) → soft one-hot via Gumbel-Softmax
         (hard=training-forward uses argmax, backward uses soft gradient —
          the straight-through estimator, Bengio et al. 2013).
      2. Pairing score S[i,j] = soft_tok[i] · M · soft_tok[j]^T
         (M = learned-perturbed Watson-Crick compatibility), then refined
         by a small MLP on (tok_emb_i, tok_emb_j, circular-distance feat).
      3. Soft pairing matrix P = sigmoid(S), masked to forbid i==j and
         |i-j| < min_loop (steric / hairpin loop constraint).
      4. Aggregated structural features: stem_density, mean_pair_prob,
         gc_content — all differentiable, consumable by immune heads.

    This is an AUXILIARY differentiable path. It does NOT replace the
    ESM2 trunk; it runs in parallel and feeds an extra structural
    feature vector that admits true token-level gradients. The main
    ESM2-based heads remain unchanged for prediction quality.

    References:
      - Sato et al., MXfold2 (BMC Bioinformatics 2021) — differentiable
        DP folding; we use the soft-pairing relaxation, not the full DP.
      - NuSpeak / Diekman et al. — sequence-design via differentiable
        structure objectives.
      - Bengio et al. 2013 — straight-through estimator.
    """

    NUC_ORDER = ('A', 'U', 'G', 'C')

    def __init__(
        self,
        d_model: int = 128,
        min_loop: int = 3,
        tau: float = 1.0,
        hard_ste: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.min_loop = min_loop
        self.tau = tau
        self.hard_ste = hard_ste

        # Learnable perturbation on top of the fixed WC compatibility.
        self.register_buffer('bp_compat', _BP_COMPAT.clone())
        self.bp_perturb = nn.Parameter(torch.zeros(4, 4))

        # Per-position token embedding (lifts 4-dim one-hot to d_model).
        self.tok_embed = nn.Linear(4, d_model)

        # Circular distance embedding (ring-aware — circRNA has no ends).
        self.max_circ_dist = 256
        self.dist_embed = nn.Embedding(self.max_circ_dist + 1, 16)

        # Pair refinement MLP: (tok_i, tok_j, dist_feat) → scalar logit.
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 16, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1),
        )

    @staticmethod
    def _circular_distance_matrix(L: int, device) -> torch.Tensor:
        """Min distance along the ring between positions i, j. (L, L)."""
        idx = torch.arange(L, device=device)
        d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return torch.min(d, L - d)

    def forward(
        self,
        token_logits: torch.Tensor,   # (B, L, 4) — pre-softmax logits
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_logits: (B, L, 4) raw logits over (A,U,G,C). In eval
                mode these can come from argmax (hard sequence); in train
                / design mode they are the design parameters.

        Returns dict with:
            soft_tokens     (B, L, 4)   — relaxed one-hot (differentiable)
            pair_prob       (B, L, L)   — soft pairing probability matrix
            stem_density    (B,)        — mean off-diagonal pair prob
            mean_pair_prob  (B,)        — mean pair prob (inc. diag=0)
            gc_content      (B,)        — soft GC fraction (differentiable)
            token_probs     (B, L, 4)   — softmax probs (for diagnostics)
        """
        B, L, _ = token_logits.shape
        device = token_logits.device

        # 1. Straight-through Gumbel-Softmax: forward = hard one-hot,
        #    backward = soft (reparameterized) gradient.
        if self.training and self.hard_ste:
            soft = F.gumbel_softmax(token_logits, tau=self.tau, hard=True)
            # `hard=True` already applies STE inside F.gumbel_softmax.
        else:
            soft = F.gumbel_softmax(token_logits, tau=self.tau, hard=False)
        token_probs = F.softmax(token_logits, dim=-1)
        soft_tokens = soft  # (B, L, 4)

        # 2. Compatibility matrix (fixed prior + learned perturbation).
        M = F.softplus(self.bp_compat + self.bp_perturb)  # (4, 4), >=0

        # 3. Pairing score from compatibility: for each (i, j),
        #    score = soft_tok[i] · M · soft_tok[j]^T  → (B, L, L)
        #    Einsum: b i k, k l, b j l → b i j
        pair_score = torch.einsum(
            'bik,kl,bjl->bij', soft_tokens, M, soft_tokens
        )  # (B, L, L)

        # 4. Refine with token embeddings + circular distance feature.
        tok_emb = self.tok_embed(soft_tokens)  # (B, L, d_model)
        circ_d = self._circular_distance_matrix(L, device)
        circ_d_clamped = circ_d.clamp(0, self.max_circ_dist).long()
        dist_feat = self.dist_embed(circ_d_clamped)  # (L, L, 16)

        left = tok_emb.unsqueeze(2).expand(-1, -1, L, -1)   # (B,L,L,d)
        right = tok_emb.unsqueeze(1).expand(-1, L, -1, -1)  # (B,L,L,d)
        pair_in = torch.cat([left, right, dist_feat.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1)
        pair_score = pair_score + self.pair_mlp(pair_in).squeeze(-1)

        # 5. Mask: forbid self-pairing and too-short loops (hairpin constraint).
        eye = torch.eye(L, device=device, dtype=torch.bool)
        too_close = circ_d < self.min_loop
        forbidden = eye | too_close
        pair_score = pair_score.masked_fill(forbidden.unsqueeze(0), -1e9)
        pair_prob = torch.sigmoid(pair_score)  # (B, L, L)

        # Symmetrize (pairing is undirected).
        pair_prob = 0.5 * (pair_prob + pair_prob.transpose(1, 2))

        # 6. Aggregated differentiable structural features.
        off_diag_mask = (~eye).float().unsqueeze(0)
        stem_density = (pair_prob * off_diag_mask).sum(dim=(1, 2)) / (L * (L - 1))
        mean_pair_prob = pair_prob.mean(dim=(1, 2))
        # GC content: soft sum of G+C probabilities (indices 2, 3).
        gc_content = soft_tokens[..., 2].sum(dim=1) + soft_tokens[..., 3].sum(dim=1)
        gc_content = gc_content / L

        return {
            'soft_tokens': soft_tokens,
            'pair_prob': pair_prob,
            'stem_density': stem_density,
            'mean_pair_prob': mean_pair_prob,
            'gc_content': gc_content,
            'token_probs': token_probs,
        }

