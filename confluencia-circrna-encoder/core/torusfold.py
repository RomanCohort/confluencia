"""
torusfold.py — TorusFold v2: AlphaFold3-inspired architecture for circRNA.

Architecture (v2, AF3-inspired):

    Input: circRNA sequence (A,C,G,U) + gene expression
              │
    ┌─────────▼──────────┐
    │  TPE Layer          │  Torus Positional Encoding
    │  sin(2π·i/L)+harmonics│
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  CircEquivariant    │  Rotation-equivariant backbone
    │  Backbone           │  ESM2 (frozen) + TPE + Torus Transformer
    └─────────┬──────────┘
              │ sequence_repr (B, L, d_model)
              │
    ┌─────────▼──────────┐
    │  Pair Initialization│  z[i,j] = left(seq[i]) + right(seq[j])
    │  + Circular Distance│  + d_circ(i,j) features
    └─────────┬──────────┘
              │ pair_repr (B, L, L, c_z)
              │
    ┌─────────▼──────────┐
    │  CircPairformer     │  AF3-style Pairformer (x4 blocks)
    │  Stack              │  ├─ TriangleMulUpdate (outgoing)
    │                     │  ├─ TriangleMulUpdate (incoming)
    │                     │  ├─ TriangleAttention (starting)
    │                     │  ├─ TriangleAttention (ending)
    │                     │  └─ PairTransition
    └─────────┬──────────┘
              │ refined pair_repr
              │
    ┌─────────▼──────────┐
    │  BSJ Analyzer       │  Back-splice junction pair analysis
    │  + IRS Pair Head    │  Pair probability P[i,j] + BSJ mask
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Structure Module   │  AF3-style diffusion OR simple head
    │  (Diffusion/MDS)   │  with circular closure constraint
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Multi-task Heads   │  Composite(8) + Report(4) + Response(3)
    │                     │  + Translation + Stability + Immune pathway
    └─────────────────────┘

Key improvements over v1:
1. CircPairformer replaces simple axial attention (AF3-inspired triangle updates)
2. Diffusion structure module replaces simple refinement (AF3-inspired)
3. Larger pair representation (c_z=128, matching AF2)
4. BSJ-aware triangle operations (circular topology)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpe import TorusPositionalEncoding, CircularRelativeBias
from .equivariant_backbone import CircEquivariantBackbone
from .triangle_update import CircPairformerStack
from .diffusion_structure import CircDiffusionStructure, SimpleStructureHead
from .irs_pair import BSJPairAnalyzer, circular_distance_matrix
from .physics_structure_head import PhysicsStructureHead
from .torus_coord_head import TorusCoordHead, ImmuneFingerprintHeads


def coords_to_distance_histogram(
    coords: torch.Tensor,
    n_bins: int = 32,
    d_max: float = 30.0,
) -> torch.Tensor:
    """
    Lightweight differentiable distance-distribution proxy for Cryo-EM
    density fitting (Change 3 in project-torusfold-arch-v2.md).

    Full voxel-grid KL against an MRC density map is O(B·L·N) and memory-
    explosive. In practice, the most informative 1D summary of a Cryo-EM
    density map for a coarse-grained RNA is its pairwise-distance histogram
    (the radial distribution function), which can be precomputed once from
    the experimental map and then compared to the histogram of the predicted
    coordinates. This is differentiable, cheap, and captures the global
    shape signature that density fitting cares about.

    Args:
        coords: (B, L, 3) predicted Cartesian coordinates.
        n_bins: Number of distance bins.
        d_max: Maximum distance to bin (Å).

    Returns:
        (B, n_bins) normalized distance histogram (sums to 1 per molecule).
    """
    B, L, _ = coords.shape
    # Pairwise distances (upper triangle to avoid double-counting + diagonal).
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
    dist = diff.norm(dim=-1)  # (B, L, L)
    # Take upper triangle (i < j)
    iu = torch.triu_indices(L, L, offset=1, device=coords.device)
    pair_dists = dist[:, iu[0], iu[1]]  # (B, n_pairs)

    # Soft binning via differentiable histogram (Gaussian kernels on bins).
    bin_centers = torch.linspace(
        d_max / n_bins / 2, d_max - d_max / n_bins / 2, n_bins,
        device=coords.device,
    )
    sigma = d_max / n_bins  # bin width as kernel std
    # (B, n_pairs, 1) - (1, 1, n_bins) → (B, n_pairs, n_bins)
    contrib = torch.exp(
        -((pair_dists.unsqueeze(-1) - bin_centers) ** 2) / (2 * sigma ** 2)
    )
    hist = contrib.sum(dim=1)  # (B, n_bins)
    hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return hist


def density_fitting_loss(
    pred_coords: torch.Tensor,
    em_distance_histogram: torch.Tensor,
    n_bins: int = 32,
    d_max: float = 30.0,
) -> torch.Tensor:
    """
    Cryo-EM density-fitting loss (Change 3, project-torusfold-arch-v2.md).

    KL divergence between the predicted-structure distance histogram and
    the experimental distance histogram derived from a Cryo-EM density map.

    Args:
        pred_coords: (B, L, 3) predicted Cartesian coordinates.
        em_distance_histogram: (B, n_bins) or (n_bins,) experimental
            pairwise-distance histogram (precomputed from MRC map).
        n_bins: Histogram bins (must match em_distance_histogram).
        d_max: Max distance (Å).

    Returns:
        Scalar loss (mean KL over batch).
    """
    pred_hist = coords_to_distance_histogram(
        pred_coords, n_bins=n_bins, d_max=d_max
    )  # (B, n_bins)

    # Match shapes: broadcast em_distance_histogram to (B, n_bins)
    if em_distance_histogram.dim() == 1:
        em_hist = em_distance_histogram.unsqueeze(0).expand_as(pred_hist)
    else:
        em_hist = em_distance_histogram

    # Add small epsilon to avoid log(0) and division by zero.
    eps = 1e-8
    pred_hist = pred_hist.clamp(min=eps)
    em_hist = em_hist.clamp(min=eps)
    em_hist = em_hist / em_hist.sum(dim=-1, keepdim=True)

    # KL(em || pred) — we want pred to match em, so em is the "true" distribution.
    kl = (em_hist * (em_hist.log() - pred_hist.log())).sum(dim=-1)  # (B,)
    return kl.mean()


@dataclass
class TorusFoldConfig:
    """Configuration for TorusFold v2 model."""

    # Backbone
    d_model: int = 640
    n_torus_layers: int = 4
    n_heads: int = 10
    d_ff: int = 2560
    n_harmonics: int = 16
    n_rot_augments: int = 0
    max_circ_dist: int = 128

    # Pairformer (AF3-style)
    c_z: int = 128             # Pair representation dim (AF2 uses 128)
    c_hidden_tri: int = 128    # Hidden dim for triangle updates
    n_pairformer_blocks: int = 4  # Number of Pairformer blocks
    n_heads_tri: int = 4       # Heads for triangle attention

    # Structure module
    structure_mode: str = "simple"   # "simple", "diffusion", "physics_b", "physics_ba", "torus"
    d_cond: int = 256          # Diffusion conditioning dim
    d_coord: int = 64          # Coordinate feature dim
    n_diffusion_steps: int = 100  # Diffusion inference steps
    n_denoiser_layers: int = 4
    n_rbf: int = 16
    bond_length: float = 5.9  # Å, P-P backbone distance (adjacent phosphates)
    # Note: C1'-C1' adjacent distance is ~3.4 Å, but physics module uses P-P for coarse-grained model

    # Scheme 9 (TorusCoordHead) parameters
    torus_d_hidden: int = 256       # Hidden dim of the torus projection MLP
    torus_r_scale: float = 0.5      # Soft scale on cross-section radius r
    # When structure_mode == "torus", closure is a structural hard guarantee
    # (kabsch cap loss is bypassed; r-norm constraint prevents collapse).

    # Scheme 9 immune-fingerprint heads (Change 2 in project-torusfold-arch-v2.md)
    enable_immune_fingerprints: bool = True
    enable_pkr_head: bool = True       # long-stem ratio + SASA → PKR activation
    enable_nlrp3_head: bool = True     # persistence length → NLRP3 scaffold
    enable_drach_head: bool = True     # DRACH × in_loop × SASA → m6A shielding
    enable_tlr7_head: bool = True      # GU-rich single-loop → TLR7 activation
    enable_sponge_head: bool = True    # miRNA duplex distance → sponge potency
    # Change 4: RIG-I walk-attention (negative control; off by default since
    # it adds sliding-window cost and is mainly a regularizer/sanity check).
    enable_rigi_head: bool = False
    rigi_window: int = 15              # ATPase step length (~15 nt)

    # Change 5: m6A solvent-exposure proxy routing (orbit C).
    #
    # SEMANTICS (read before changing): this flag does NOT route the whole m6A
    # head to 2D. It only swaps the solvent-exposure term in the m6A write
    # probability:
    #     m6a_write_prob = is_drach * in_loop * exposure_proxy
    #   - False (default): exposure_proxy = sigmoid(torus_coords[..., 2])   [3D]
    #   - True          : exposure_proxy = 1 - mean(pair_probs)            [2D]
    #
    # `is_drach` and `in_loop` are always predicted from the shared trunk feat
    # (sequence_repr + pair summary + torus_coords). This is intentional: their
    # physical roots are 1D (DRACH motif) and 2D (loop region) respectively, but
    # they go through the shared representation rather than a forced-2D path.
    # The 2D/3D question we are isolating here is narrowly: "does the *solvent
    # exposure* sub-quantity need 3D, or is a 2D single-strandedness proxy
    # enough?" — NOT "does the whole m6A head need 3D".
    #
    # NLRP3 keeps its 3D path regardless of this flag (persistence length
    # genuinely requires 3D geometry). PKR/TLR7/sponge likewise unchanged.
    #
    # Rationale for isolating exposure: no per-residue SASA labels exist for
    # circRNA; a self-generated 3D SASA label would be circular (label and
    # input share the same forward). The 2D single-strandedness proxy is
    # directly supervised by pair_probs, sidestepping that circularity.
    enable_fingerprint_2d: bool = False

    # Physics-based structure parameters (only for physics_b / physics_ba)
    pair_distance: float = 10.6       # Å, WC C1'-C1' distance
    n_solver_samples: int = 20        # Number of solver conformations
    n_minimize_steps: int = 500       # OpenMM minimization steps
    n_md_steps: int = 5000            # OpenMM MD steps
    use_dl_bias: bool = True          # DL bias in CG MD
    closure_tolerance: float = 0.5    # Å tolerance for closure

    # Multi-task heads
    composite_keys: List[str] = field(default_factory=lambda: [
        "immunotherapy_score", "tumor_killing_index",
        "overall_immunogenicity", "immune_cycle_score",
        "tme_score", "therapeutic_window", "tide_score", "ips",
    ])
    report_keys: List[str] = field(default_factory=lambda: [
        "rig_i_score", "tlr_score", "pkr_score", "trained_model_risk",
    ])
    response_classes: List[str] = field(default_factory=lambda: [
        "likely_non_responder", "intermediate", "likely_responder",
    ])

    # TorusFold-specific heads
    translation_efficiency: bool = True
    circ_stability: bool = True
    immune_pathway: bool = True
    bsj_confidence: bool = True

    # Gene expression
    gene_cols: List[str] = field(default_factory=lambda: [
        "TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"
    ])
    gene_dim: int = 6

    # Training
    hidden_dim: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8

    # Loss weights
    w_closure: float = 1.0
    w_bond: float = 0.5
    w_clash: float = 0.1
    w_pair: float = 1.0
    w_composite: float = 1.0
    w_report: float = 1.0
    w_response: float = 1.0

    # Scheme 9 immune-fingerprint loss weights (each head independently weighted)
    w_pkr_stem: float = 0.5
    w_pkr_sasa: float = 0.5
    w_nlrp3_pl: float = 0.5
    w_drach_is: float = 0.3
    w_drach_loop: float = 0.3
    w_m6a_write: float = 0.5
    w_tlr7: float = 0.3
    w_sponge: float = 0.5
    # Change 4: RIG-I walk-attention loss (negative control)
    w_rigi: float = 0.2
    # Change 3: Cryo-EM density-fitting loss (only active when targets
    # contain em_distance_histogram). Low weight — it's an auxiliary anchor.
    w_density_fitting: float = 0.2
    density_n_bins: int = 32
    density_d_max: float = 30.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class CompositeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReportHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResponseHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairInitialization(nn.Module):
    """
    Initialize pair representation from per-position embeddings.

    Following AF2/AF3: z[i,j] = Linear_left(seq[i]) + Linear_right(seq[j])
                       + circular_distance_features(i,j)
    """

    def __init__(self, d_model: int, c_z: int, max_circ_dist: int = 128):
        super().__init__()
        self.left_proj = nn.Linear(d_model, c_z)
        self.right_proj = nn.Linear(d_model, c_z)

        # Circular distance embedding
        self.dist_embedding = nn.Embedding(max_circ_dist + 1, c_z)

        # Relative position bias (circular)
        self.circ_bias = CircularRelativeBias(
            n_heads=1, max_dist=max_circ_dist
        )

    def forward(self, sequence_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_repr: (B, L, d_model)

        Returns:
            (B, L, L, c_z) initial pair representation
        """
        B, L, _ = sequence_repr.shape

        left = self.left_proj(sequence_repr)    # (B, L, c_z)
        right = self.right_proj(sequence_repr)  # (B, L, c_z)

        # Outer sum: z[i,j] = left[i] + right[j]
        pair = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, c_z)

        # Add circular distance features
        circ_dist = circular_distance_matrix(L, sequence_repr.device)  # (L, L)
        circ_dist_clamped = circ_dist.clamp(0, self.dist_embedding.num_embeddings - 1).long()
        dist_feat = self.dist_embedding(circ_dist_clamped)  # (L, L, c_z)
        pair = pair + dist_feat.unsqueeze(0)  # broadcast over batch

        return pair


class PairPredictionHead(nn.Module):
    """Predict base-pair probability from refined pair representation."""

    def __init__(self, c_z: int = 128):
        super().__init__()
        self.pair_head = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.GELU(),
            nn.LayerNorm(c_z),
            nn.Linear(c_z, 1),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, c_z)

        Returns:
            pair_probs: (B, L, L) symmetric pair probabilities
        """
        logits = self.pair_head(pair_repr).squeeze(-1)  # (B, L, L)

        # Enforce symmetry: P[i,j] = P[j,i]
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        # Soft BSJ rotational consistency: encourage P[i,j] ≈ P[(i+1)%L, (j+1)%L]
        # Not hard-enforced because real circRNA structures depend on sequence context
        rolled = torch.roll(logits, shifts=(1, 1), dims=(1, 2))
        logits = 0.9 * logits + 0.1 * rolled

        probs = torch.sigmoid(logits)
        return probs


class TorusFold(nn.Module):
    """
    TorusFold v2: AlphaFold3-inspired architecture for circRNA.

    Integrates:
    - TPE (torus positional encoding)
    - CircEquivariantBackbone (ESM2 + TPE)
    - CircPairformerStack (AF3-style triangle updates)
    - Diffusion/Simple structure head
    - Multi-task heads
    """

    def __init__(self, config: Optional[TorusFoldConfig] = None):
        super().__init__()
        self.config = config or TorusFoldConfig()
        c = self.config

        # 1. Rotation-equivariant backbone
        self.backbone = CircEquivariantBackbone(
            d_model=c.d_model,
            n_torus_layers=c.n_torus_layers,
            n_heads=c.n_heads,
            d_ff=c.d_ff,
            n_harmonics=c.n_harmonics,
            n_rot_augments=c.n_rot_augments,
            dropout=c.dropout,
            max_circ_dist=c.max_circ_dist,
        )

        # 2. Pair initialization (AF3-style)
        self.pair_init = PairInitialization(
            d_model=c.d_model, c_z=c.c_z, max_circ_dist=c.max_circ_dist,
        )

        # 3. CircPairformer Stack (AF3-style, replaces IRS axial attention)
        self.pairformer = CircPairformerStack(
            n_blocks=c.n_pairformer_blocks,
            c_z=c.c_z,
            c_hidden_tri=c.c_hidden_tri,
            n_heads_tri=c.n_heads_tri,
            max_circ_dist=c.max_circ_dist,
        )

        # 4. Pair prediction head (base-pairing)
        self.pair_head = PairPredictionHead(c_z=c.c_z)

        # 5. BSJ analyzer
        self.bsj_analyzer = BSJPairAnalyzer(d_pair=c.c_z)

        # 6. Structure head (diffusion, simple, physics-based, or torus)
        if c.structure_mode == "diffusion":
            self.structure_head = CircDiffusionStructure(
                d_pair=c.c_z,
                d_cond=c.d_cond,
                d_coord=c.d_coord,
                n_layers=c.n_denoiser_layers,
                n_steps=c.n_diffusion_steps,
                bond_length=c.bond_length,
            )
        elif c.structure_mode in ("physics_b", "physics_ba"):
            # Physics-based structure head (Plan B + optional Plan A)
            self.structure_head = PhysicsStructureHead(
                c_z=c.c_z,
                structure_mode=c.structure_mode,
                bond_length=c.bond_length,
                pair_distance=c.pair_distance,
                n_solver_samples=c.n_solver_samples,
                n_minimize_steps=c.n_minimize_steps,
                n_md_steps=c.n_md_steps,
                use_dl_bias=c.use_dl_bias,
                closure_tolerance=c.closure_tolerance,
            )
        elif c.structure_mode == "torus":
            # Scheme 9: torus-parameterized structure head.
            # Closure is a structural hard guarantee (kabsch cap bypassed).
            self.structure_head = TorusCoordHead(
                d_pair=c.c_z,
                d_hidden=c.torus_d_hidden,
                n_rbf=c.n_rbf,
                r_scale=c.torus_r_scale,
                bond_length=c.bond_length,
            )
        else:
            self.structure_head = SimpleStructureHead(
                d_pair=c.c_z,
                d_coord=c.d_coord,
                n_rbf=c.n_rbf,
            )

        # 6b. Immune-fingerprint heads (Scheme 9, Change 2)
        #     Toggleable per-head; missing labels simply don't contribute to loss.
        if c.enable_immune_fingerprints:
            self.immune_heads = ImmuneFingerprintHeads(
                d_model=c.d_model,
                c_z=c.c_z,
                d_torus=3,
                hidden_dim=c.hidden_dim,
                dropout=c.dropout,
                enable_pkr=c.enable_pkr_head,
                enable_nlrp3=c.enable_nlrp3_head,
                enable_drach=c.enable_drach_head,
                enable_tlr7=c.enable_tlr7_head,
                enable_sponge=c.enable_sponge_head,
                enable_rigi=c.enable_rigi_head,
                rigi_window=c.rigi_window,
                enable_fingerprint_2d=c.enable_fingerprint_2d,  # Change 5: orbit C routing
            )
        else:
            self.immune_heads = None

        # 7. Multi-task heads
        input_dim = c.d_model + c.gene_dim + c.c_z + 1

        self.composite_head = CompositeHead(
            input_dim=input_dim, hidden_dim=c.hidden_dim,
            n_outputs=len(c.composite_keys), dropout=c.dropout,
        )
        self.report_head = ReportHead(
            input_dim=input_dim, hidden_dim=c.hidden_dim,
            n_outputs=len(c.report_keys), dropout=c.dropout,
        )
        self.response_head = ResponseHead(
            input_dim=input_dim, hidden_dim=c.hidden_dim,
            n_classes=len(c.response_classes), dropout=c.dropout,
        )

        # 8. TorusFold-specific heads
        if c.translation_efficiency:
            self.translation_head = nn.Sequential(
                nn.Linear(input_dim, c.hidden_dim // 2), nn.GELU(),
                nn.Dropout(c.dropout), nn.Linear(c.hidden_dim // 2, 1), nn.Sigmoid(),
            )
        if c.circ_stability:
            self.stability_head = nn.Sequential(
                nn.Linear(input_dim, c.hidden_dim // 2), nn.GELU(),
                nn.Dropout(c.dropout), nn.Linear(c.hidden_dim // 2, 1), nn.Sigmoid(),
            )
        if c.immune_pathway:
            self.immune_pathway_head = nn.Sequential(
                nn.Linear(input_dim, c.hidden_dim // 2), nn.GELU(),
                nn.Dropout(c.dropout), nn.Linear(c.hidden_dim // 2, 3),
            )
        if c.bsj_confidence:
            self.bsj_confidence_head = nn.Sequential(
                nn.Linear(c.c_z + 1, c.c_z), nn.GELU(),
                nn.Linear(c.c_z, 1), nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        for module in [self.composite_head, self.report_head, self.response_head]:
            for layer in module.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        sequences: List[str],
        gene_expr: torch.Tensor,
        device: str = "cpu",
        predict_structure: bool = True,
    ) -> Dict[str, torch.Tensor]:
        c = self.config

        # 1. Backbone → sequence representation
        backbone_out = self.backbone(sequences, device=device)
        global_emb = backbone_out["embedding"]          # (B, d_model)
        sequence_repr = backbone_out["sequence_repr"]    # (B, L, d_model)

        # 2. Initialize pair representation
        pair_repr = self.pair_init(sequence_repr)  # (B, L, L, c_z)

        # 3. CircPairformer: refine pair representation
        pair_repr = self.pairformer(pair_repr)  # (B, L, L, c_z)

        # 4. Pair prediction
        pair_probs = self.pair_head(pair_repr)  # (B, L, L)

        # 5. BSJ analysis
        L = pair_repr.size(1)
        positions = torch.arange(L, device=pair_repr.device)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        bsj_mask = (diff.abs() >= L / 2).float()  # (L, L) — >= because circ_dist max == L/2 for even L

        bsj_out = self.bsj_analyzer(pair_repr, bsj_mask)
        bsj_stability = bsj_out["bsj_stability_score"]  # (B,)

        # 6. Structure prediction
        structure_out = None
        if predict_structure:
            if isinstance(self.structure_head, PhysicsStructureHead):
                structure_out = self.structure_head(
                    pair_repr, pair_probs=pair_probs, sequences=sequences
                )
            elif isinstance(self.structure_head, TorusCoordHead):
                # Scheme 9: pass pair_probs for φ modulation.
                # Closure is structurally guaranteed; closure_loss is diagnostic.
                structure_out = self.structure_head(
                    pair_repr, pair_probs=pair_probs
                )
            else:
                structure_out = self.structure_head(pair_repr)
            # Normalize output keys
            if "closure_dist" in structure_out and "closure_distance" not in structure_out:
                structure_out["closure_distance"] = structure_out["closure_dist"]

        # 6b. Immune-fingerprint heads (Scheme 9, Change 2)
        immune_out = None
        if self.immune_heads is not None and structure_out is not None:
            torus_coords = structure_out.get(
                "torus_coords",
                # Fallback: synthesize a (B, L, 3) zero torus if structure head
                # did not produce one (e.g. non-torus structure_mode).
                torch.zeros(
                    sequence_repr.size(0),
                    sequence_repr.size(1),
                    3,
                    device=sequence_repr.device,
                ),
            )
            immune_out = self.immune_heads(
                sequence_repr=sequence_repr,
                pair_repr=pair_repr,
                torus_coords=torus_coords,
                pair_probs=pair_probs,  # Change 5: 2D m6A proxy needs pair_probs
            )

        # 7. Multi-task heads
        gene_expr = gene_expr.to(device)
        # Pool pair_repr (B, L, L, c_z) over both residue axes → (B, c_z).
        # NOTE: mean(dim=(2,3)) would leave the i axis intact → (B, L); the
        # intended whole-molecule pair summary is (B, c_z), so pool dims 1,2.
        struct_feat = pair_repr.mean(dim=(1, 2))  # (B, c_z)

        multi_input = torch.cat([
            global_emb, gene_expr, struct_feat,
            bsj_stability.unsqueeze(-1),
        ], dim=-1)

        composite = self.composite_head(multi_input)
        report = self.report_head(multi_input)
        response_logits = self.response_head(multi_input)
        response_probs = F.softmax(response_logits, dim=-1)

        # 8. Assemble results
        result = {
            "embedding": global_emb,
            "sequence_repr": sequence_repr,
            "pair_repr": pair_repr,
            "pair_probs": pair_probs,
            "bsj_pair_mask": bsj_mask,
            "bsj_pair_count": bsj_out["bsj_pair_count"],
            "bsj_stability_score": bsj_stability,
            "composite": composite,
            "report": report,
            "response_logits": response_logits,
            "response_probs": response_probs,
        }

        if structure_out is not None:
            result["coords"] = structure_out["coords"]
            result["confidence"] = structure_out["confidence"]
            result["closure_distance"] = structure_out.get(
                "closure_distance", structure_out.get("closure_dist", torch.tensor(0.0)))
            if "dist_pred" in structure_out:
                result["dist_pred"] = structure_out["dist_pred"]
            if "closure_loss" in structure_out:
                result["closure_loss"] = structure_out["closure_loss"]
            if "torus_coords" in structure_out:
                result["torus_coords"] = structure_out["torus_coords"]

        # Scheme 9 immune-fingerprint outputs
        if immune_out is not None:
            result["immune_fingerprints"] = immune_out

        if c.translation_efficiency:
            result["translation_efficiency"] = self.translation_head(multi_input)
        if c.circ_stability:
            result["circ_stability"] = self.stability_head(multi_input)
        if c.immune_pathway:
            pathway_logits = self.immune_pathway_head(multi_input)
            result["immune_pathway_probs"] = F.softmax(pathway_logits, dim=-1)
            result["immune_pathway_names"] = ["RIG-I", "TLR", "PKR"]
        if c.bsj_confidence:
            bsj_conf_input = torch.cat([struct_feat, bsj_stability.unsqueeze(-1)], dim=-1)
            result["bsj_confidence"] = self.bsj_confidence_head(bsj_conf_input)

        return result

    def predict_single(
        self,
        sequence: str,
        gene_expr: Dict[str, float],
        device: str = "cpu",
    ) -> Dict[str, float]:
        c = self.config
        gene_values = [gene_expr.get(g, 0.5) for g in c.gene_cols]
        gene_tensor = torch.tensor([gene_values], dtype=torch.float32)

        outputs = self.forward([sequence], gene_tensor, device=device)

        result = {}
        for i, k in enumerate(c.composite_keys):
            result[k] = outputs["composite"][0, i].item()
        for i, k in enumerate(c.report_keys):
            result[k] = outputs["report"][0, i].item()
        for i, cls in enumerate(c.response_classes):
            result[f"prob_{cls}"] = outputs["response_probs"][0, i].item()

        result["predicted_response"] = c.response_classes[
            outputs["response_probs"][0].argmax().item()
        ]
        result["bsj_pair_count"] = outputs["bsj_pair_count"][0].item()
        result["bsj_stability"] = outputs["bsj_stability_score"][0].item()

        if c.translation_efficiency and "translation_efficiency" in outputs:
            result["translation_efficiency"] = outputs["translation_efficiency"][0, 0].item()
        if c.circ_stability and "circ_stability" in outputs:
            result["circ_stability"] = outputs["circ_stability"][0, 0].item()
        if c.immune_pathway and "immune_pathway_probs" in outputs:
            for i, name in enumerate(outputs["immune_pathway_names"]):
                result[f"immune_pathway_{name}"] = outputs["immune_pathway_probs"][0, i].item()
        if c.bsj_confidence and "bsj_confidence" in outputs:
            result["bsj_confidence"] = outputs["bsj_confidence"][0, 0].item()
        if "coords" in outputs:
            result["closure_distance"] = outputs["closure_distance"][0].item()
            result["mean_confidence"] = outputs["confidence"][0].mean().item()

        return result

    def predict_sequence(
        self,
        sequence: str,
        gene_expr: Optional[Dict[str, float]] = None,
        flatten_fingerprints: bool = True,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Convenience wrapper for single-sequence prediction with optional
        immune_fingerprints flattening.

        Unlike predict_single() (which omits immune_fingerprints), this method
        also flattens scalar immune-fingerprint features (sponge_score,
        nlrp3_persistence_length, etc.) into top-level keys so downstream
        consumers (e.g., the Simulacrum therapy bridge) can read them directly.

        Args:
            sequence: circRNA RNA sequence (e.g., "GUCCCCCCUCC")
            gene_expr: Gene expression dict. If None, uses default tumor genes.
            flatten_fingerprints: If True, unpack scalar immune-fingerprint
                                  features into top-level keys (per-residue
                                  vectors are mean-pooled with a '_mean' suffix).
                                  If False, fingerprints stay nested under
                                  'immune_fingerprints'.
            device: 'cpu' or 'cuda'.

        Returns:
            Flat dict with report/composite/response scalars, plus (if
            flatten_fingerprints=True) scalar immune features:
            sponge_score, nlrp3_persistence_length, pkr_stem_logit_mean,
            pkr_sasa_mean, drach_is_drach_mean, drach_in_loop_mean,
            m6a_write_prob_mean, tlr7_gu_density_mean.
        """
        if gene_expr is None:
            gene_expr = {
                "TROP2": 7.2, "NECTIN4": 5.1, "LIV-1": 3.5,
                "B7-H4": 6.0, "MKI67": 8.0, "MYC": 4.5,
            }

        c = self.config
        gene_values = [gene_expr.get(g, 0.5) for g in c.gene_cols]
        gene_tensor = torch.tensor([gene_values], dtype=torch.float32, device=device)

        outputs = self.forward([sequence], gene_tensor, device=device, predict_structure=True)

        result = {}
        for i, k in enumerate(c.composite_keys):
            result[k] = outputs["composite"][0, i].item()
        for i, k in enumerate(c.report_keys):
            result[k] = outputs["report"][0, i].item()
        for i, cls in enumerate(c.response_classes):
            result[f"prob_{cls}"] = outputs["response_probs"][0, i].item()
        result["predicted_response"] = c.response_classes[
            outputs["response_probs"][0].argmax().item()
        ]
        result["bsj_pair_count"] = outputs["bsj_pair_count"][0].item()
        result["bsj_stability"] = outputs["bsj_stability_score"][0].item()

        if c.translation_efficiency and "translation_efficiency" in outputs:
            result["translation_efficiency"] = outputs["translation_efficiency"][0, 0].item()
        if c.circ_stability and "circ_stability" in outputs:
            result["circ_stability"] = outputs["circ_stability"][0, 0].item()
        if c.immune_pathway and "immune_pathway_probs" in outputs:
            for i, name in enumerate(outputs["immune_pathway_names"]):
                result[f"immune_pathway_{name}"] = outputs["immune_pathway_probs"][0, i].item()
        if c.bsj_confidence and "bsj_confidence" in outputs:
            result["bsj_confidence"] = outputs["bsj_confidence"][0, 0].item()
        if "coords" in outputs:
            result["closure_distance"] = outputs["closure_distance"][0].item()
            result["mean_confidence"] = outputs["confidence"][0].mean().item()

        if flatten_fingerprints and c.enable_immune_fingerprints and self.immune_heads is not None:
            fp = outputs.get("immune_fingerprints", {})
            scalar_keys = [
                "pkr_stem_logit", "pkr_sasa", "nlrp3_persistence_length",
                "drach_is_drach", "drach_in_loop", "m6a_write_prob",
                "tlr7_gu_density", "sponge_score",
            ]
            for key in scalar_keys:
                if key not in fp:
                    continue
                tensor = fp[key]
                if tensor.dim() <= 1 or tensor.numel() == 1:
                    result[key] = tensor.squeeze().item() if tensor.numel() == 1 else tensor[0].item()
                elif tensor.dim() == 2 and tensor.size(0) == 1:
                    result[key + "_mean"] = tensor[0].mean().item()

        return result

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        c = self.config
        losses = {}

        if "composite_target" in targets:
            losses["composite_loss"] = F.mse_loss(
                predictions["composite"], targets["composite_target"])

        if "report_target" in targets:
            losses["report_loss"] = F.mse_loss(
                predictions["report"], targets["report_target"])

        if "response_target" in targets:
            losses["response_loss"] = F.cross_entropy(
                predictions["response_logits"], targets["response_target"])

        if "pair_target" in targets:
            losses["pair_loss"] = F.binary_cross_entropy(
                predictions["pair_probs"], targets["pair_target"])

        if "closure_loss" in predictions:
            losses["closure_loss"] = predictions["closure_loss"]

        # Change 3: Cryo-EM density-fitting loss.
        # Only active when targets contain an experimental distance histogram
        # (precomputed from an MRC density map). Missing → skip.
        if "em_distance_histogram" in targets and "coords" in predictions:
            losses["density_fitting_loss"] = density_fitting_loss(
                predictions["coords"],
                targets["em_distance_histogram"],
                n_bins=c.density_n_bins,
                d_max=c.density_d_max,
            )

        # Scheme 9 immune-fingerprint losses (Change 2, project-torusfold-arch-v2.md)
        if "immune_fingerprints" in predictions and self.immune_heads is not None:
            if "pkr_stem_target" in targets:
                # MSE on stem length logits (binned regression)
                losses["pkr_stem_loss"] = F.mse_loss(
                    predictions["immune_fingerprints"]["pkr_stem_logit"],
                    targets["pkr_stem_target"]
                )
            if "pkr_sasa_target" in targets:
                # L1 on sasa (regression)
                losses["pkr_sasa_loss"] = F.l1_loss(
                    predictions["immune_fingerprints"]["pkr_sasa"],
                    targets["pkr_sasa_target"]
                )
            if "nlrp3_persistence_length_target" in targets:
                losses["nlrp3_pl_loss"] = F.l1_loss(
                    predictions["immune_fingerprints"]["nlrp3_persistence_length"],
                    targets["nlrp3_persistence_length_target"]
                )
            if "drach_is_drach_target" in targets:
                losses["drach_is_loss"] = F.binary_cross_entropy(
                    predictions["immune_fingerprints"]["drach_is_drach"],
                    targets["drach_is_drach_target"]
                )
            if "drach_in_loop_target" in targets:
                losses["drach_loop_loss"] = F.binary_cross_entropy(
                    predictions["immune_fingerprints"]["drach_in_loop"],
                    targets["drach_in_loop_target"]
                )
            if "m6a_write_prob_target" in targets:
                # Contrastive: m6A-positive samples → prob ~1, negatives → prob ~0
                losses["m6a_write_loss"] = F.binary_cross_entropy(
                    predictions["immune_fingerprints"]["m6a_write_prob"],
                    targets["m6a_write_prob_target"]
                )
            if "tlr7_gu_density_target" in targets:
                losses["tlr7_loss"] = F.mse_loss(
                    predictions["immune_fingerprints"]["tlr7_gu_density"],
                    targets["tlr7_gu_density_target"]
                )
            if "sponge_score_target" in targets:
                losses["sponge_loss"] = F.mse_loss(
                    predictions["immune_fingerprints"]["sponge_score"],
                    targets["sponge_score_target"]
                )
            if "rigi_score_target" in targets:
                # Negative-control loss: linear RNA → target ~1, circRNA → target ~0.
                losses["rigi_loss"] = F.binary_cross_entropy(
                    predictions["immune_fingerprints"]["rigi_score"],
                    targets["rigi_score_target"]
                )

        # Total
        if losses:
            total = torch.tensor(0.0, device=next(iter(losses.values())).device)
            for k, v in losses.items():
                if k == "total_loss":
                    continue
                w_key = f"w_{k.replace('_loss', '')}"
                weight = getattr(c, w_key, 1.0)
                total = total + weight * v
            losses["total_loss"] = total

        return losses

    def save(self, path: str):
        state = {
            "config": self.config.to_dict(),
            "backbone": self.backbone.state_dict(),
            "pair_init": self.pair_init.state_dict(),
            "pairformer": self.pairformer.state_dict(),
            "pair_head": self.pair_head.state_dict(),
            "bsj_analyzer": self.bsj_analyzer.state_dict(),
            "structure_head": self.structure_head.state_dict(),
            "composite_head": self.composite_head.state_dict(),
            "report_head": self.report_head.state_dict(),
            "response_head": self.response_head.state_dict(),
        }
        for attr in ["translation_head", "stability_head",
                      "immune_pathway_head", "bsj_confidence_head"]:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr).state_dict()
        if self.immune_heads is not None:
            state["immune_heads"] = self.immune_heads.state_dict()
        torch.save(state, path)

    def load(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        if "config" in state:
            # 过滤历史遗留字段（如 esm_model）—— dataclass 严格不接受未知
            # kwargs，旧权重存了已删字段会直接 TypeError。只传当前 config 认识的。
            import dataclasses as _dc
            _known = {f.name for f in _dc.fields(TorusFoldConfig)}
            _cfg_kwargs = {k: v for k, v in state["config"].items() if k in _known}
            _dropped = set(state["config"].keys()) - _known
            if _dropped:
                print(f"[TorusFold.load] 丢弃历史遗留 config 字段: {sorted(_dropped)}")
            self.config = TorusFoldConfig(**_cfg_kwargs)

        # 兼容两种权重格式：
        #   1) 平铺：state = {'backbone':..., 'pairformer':..., ...}（v2 当前 save）
        #   2) 嵌套：state = {'model_state_dict': {'backbone':..., ...}, 'epoch':..., ...}
        #      （训练 checkpoint 常见格式）
        weights = state.get("model_state_dict", state)

        for key in ["backbone", "pair_init", "pairformer", "pair_head",
                     "bsj_analyzer", "structure_head", "composite_head",
                     "report_head", "response_head"]:
            if key in weights:
                getattr(self, key).load_state_dict(weights[key])

        for attr in ["translation_head", "stability_head",
                      "immune_pathway_head", "bsj_confidence_head"]:
            if attr in weights and hasattr(self, attr):
                getattr(self, attr).load_state_dict(weights[attr])
        if "immune_heads" in weights and self.immune_heads is not None:
            self.immune_heads.load_state_dict(weights["immune_heads"])
