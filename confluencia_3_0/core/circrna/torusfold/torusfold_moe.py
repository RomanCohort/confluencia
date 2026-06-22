"""
torusfold_moe.py — TorusFold Mixture-of-Experts (MOE) Integration

Inspired by DrugMoE (NeurIPS 2024) and SeqTopK routing strategy.

Architecture:
    Input circRNA sequence
        ↓
    Feature Encoder (sequence + structural features)
        ↓
    Gating Network → SeqTopK routing (sequence-level expert selection)
        ↓
    Expert Pool: [Scheme1..7] (pretrained, frozen or fine-tuned)
        ↓
    Weighted Fusion → Final 3D coordinates

Key design choices:
    1. SeqTopK routing: entire sequence selects top-K experts (not per-token)
       - Short sequences → lightweight experts (Scheme 1/2/5)
       - Long sequences → efficient experts (Scheme 7 Mamba)
       - Complex structure → diffusion experts (Scheme 4/6)
    2. Load balancing: auxiliary loss prevents expert collapse
    3. Confidence weighting: each expert outputs confidence, used in fusion
    4. BSJ-aware routing: gate explicitly considers closure difficulty
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class TorusFoldMOEConfig:
    """Configuration for TorusFold MOE model."""
    # Expert pool
    n_experts: int = 7                  # Schemes 1-7
    top_k: int = 2                      # SeqTopK: select top-K experts per sequence
    expert_hidden_dim: int = 128        # Hidden dim for expert models

    # Gating network
    d_gate: int = 64                    # Gating network hidden dim
    gate_features: List[str] = field(default_factory=lambda: [
        'seq_length',       # Sequence length (normalized)
        'gc_content',       # GC content ratio
        'n_pairs',          # Number of base pairs
        'pair_density',     # n_pairs / L
        'has_bsj',          # Whether BSJ constraint exists
        'max_stem_len',     # Longest stem in secondary structure
        'loop_count',       # Number of loops
    ])

    # Fusion
    fusion_mode: str = 'weighted_avg'   # 'weighted_avg', 'stacked_refine', 'confidence'
    use_confidence: bool = True         # Use expert confidence in fusion

    # Training
    gate_lr: float = 5e-4               # Learning rate for gating network
    expert_lr: float = 1e-5             # Fine-tuning LR for experts (if enabled)
    load_balance_weight: float = 0.01   # Auxiliary load balancing loss weight
    freeze_experts: bool = True         # Freeze pretrained experts during MOE training

    # BSJ closure
    bond_length: float = 5.9            # P-P backbone distance in Angstroms


# ═══════════════════════════════════════════════════════════════
# Sequence Feature Extractor
# ═══════════════════════════════════════════════════════════════

class SequenceFeatureExtractor(nn.Module):
    """Extract routing features from circRNA sequence.

    Combines handcrafted features (length, GC, pairs) with
    learned features from a small encoder.
    """

    def __init__(self, d_out: int = 64):
        super().__init__()
        self.d_out = d_out

        # Learnable sequence encoder (lightweight)
        self.seq_embed = nn.Embedding(5, 32)  # A/U/G/C/unk
        self.pos_embed = nn.Embedding(2048, 32)  # circular position

        # Conv for local patterns
        self.conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Combine handcrafted + learned features
        # Handcrafted: 7 features (see config.gate_features)
        self.fc = nn.Sequential(
            nn.Linear(64 + 7, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(
        self,
        seq_ids: torch.Tensor,       # (B, L)
        lengths: List[int],
        pair_probs: Optional[torch.Tensor] = None,  # (B, L, L)
    ) -> torch.Tensor:
        """Extract routing features.

        Returns:
            (B, d_out) sequence-level features for gating
        """
        B, L = seq_ids.shape
        device = seq_ids.device

        # Learned features from sequence
        seq_emb = self.seq_embed(seq_ids)  # (B, L, 32)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos.clamp(max=2047))  # (B, L, 32)
        combined = torch.cat([seq_emb, pos_emb], dim=-1)  # (B, L, 64)
        combined = combined.transpose(1, 2)  # (B, 64, L)
        learned_feat = self.conv(combined).squeeze(-1)  # (B, 64)

        # Handcrafted features
        handcrafted = self._extract_handcrafted(seq_ids, lengths, pair_probs, device)
        # (B, 7)

        # Combine
        feat = torch.cat([learned_feat, handcrafted], dim=-1)  # (B, 64+7)
        return self.fc(feat)  # (B, d_out)

    def _extract_handcrafted(
        self,
        seq_ids: torch.Tensor,
        lengths: List[int],
        pair_probs: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Extract 7 handcrafted routing features."""
        B = seq_ids.shape[0]
        features = torch.zeros(B, 7, device=device)

        for b in range(B):
            L = lengths[b]
            seq = seq_ids[b, :L]

            # 1. Normalized sequence length
            features[b, 0] = L / 1000.0

            # 2. GC content
            gc = ((seq == 2) | (seq == 3)).float().sum()
            features[b, 1] = gc / max(L, 1)

            # 3. Number of base pairs
            if pair_probs is not None:
                pp = pair_probs[b, :L, :L]
                n_pairs = (pp > 0.5).float().sum().item() / 2
                features[b, 2] = n_pairs / max(L, 1) * 10  # normalized

                # 5. Pair density
                features[b, 4] = n_pairs / max(L * (L - 1) / 2, 1)

                # 6. Max stem length (heuristic from pair_probs)
                # Simplified: count longest diagonal run of pairs
                features[b, 5] = self._estimate_max_stem(pp, L) / max(L, 1)
            else:
                features[b, 2] = 0.1  # default
                features[b, 4] = 0.05
                features[b, 5] = 0.1

            # 4. Has BSJ (always true for circRNA)
            features[b, 3] = 1.0

            # 7. Loop count (heuristic)
            features[b, 6] = features[b, 2] * 2  # rough estimate

        return features

    @staticmethod
    def _estimate_max_stem(pair_probs: torch.Tensor, L: int) -> float:
        """Estimate maximum stem length from pair probability matrix."""
        max_stem = 1.0
        for i in range(min(L, 100)):  # Sample first 100 positions
            for j in range(i + 4, min(i + 50, L)):
                if pair_probs[i, j] > 0.5:
                    stem_len = 1
                    for k in range(1, min(20, L - j, i + 1)):
                        if j + k < L and i - k >= 0 and pair_probs[i - k, j + k] > 0.5:
                            stem_len += 1
                        else:
                            break
                    max_stem = max(max_stem, stem_len)
        return max_stem


# ═══════════════════════════════════════════════════════════════
# SeqTopK Gating Network
# ═══════════════════════════════════════════════════════════════

class SeqTopKGating(nn.Module):
    """Sequence-level Top-K gating for expert routing.

    Key difference from standard TopK:
    - Standard: each token independently selects K experts
    - SeqTopK: entire sequence jointly selects K experts

    This is crucial for circRNA: the whole molecule's topology
    determines which modeling approach works best, not individual
    nucleotides.

    Reference: SeqTopK routing (2024) for efficient MoE inference.
    """

    def __init__(self, d_input: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Gating MLP: sequence features → expert scores
        self.gate = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Linear(d_input, n_experts),
        )

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, seq_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gating weights.

        Args:
            seq_features: (B, d_input) sequence-level features

        Returns:
            gate_weights: (B, top_k) normalized weights for selected experts
            expert_indices: (B, top_k) indices of selected experts
            gate_logits: (B, n_experts) raw logits (for load balancing loss)
        """
        # Raw expert scores
        logits = self.gate(seq_features) / self.temperature.clamp(min=0.1)  # (B, n_experts)

        # SeqTopK: select top-K experts for the ENTIRE sequence
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # (B, top_k)

        # Normalized weights via softmax over selected experts
        gate_weights = F.softmax(top_k_logits, dim=-1)  # (B, top_k)

        return gate_weights, top_k_indices, logits

    def load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Auxiliary load balancing loss.

        Encourages uniform expert utilization across the batch.
        From Switch Transformer (Fedus et al., 2022):
            L_bal = n_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i
        """
        B = gate_logits.shape[0]

        # Routing probabilities
        probs = F.softmax(gate_logits, dim=-1)  # (B, n_experts)

        # Fraction of sequences routed to each expert (hard assignment)
        _, top_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        mask = torch.zeros_like(gate_logits)
        mask.scatter_(1, top_indices, 1.0)
        fraction = mask.mean(dim=0)  # (n_experts,)

        # Mean routing probability per expert
        mean_prob = probs.mean(dim=0)  # (n_experts,)

        # Load balance loss
        balance_loss = self.n_experts * (fraction * mean_prob).sum()

        return balance_loss


# ═══════════════════════════════════════════════════════════════
# Expert Wrapper
# ═══════════════════════════════════════════════════════════════

class ExpertWrapper(nn.Module):
    """Wraps a pretrained scheme model as an expert in the MOE.

    Handles:
    - Loading pretrained weights
    - Forward pass with optional confidence estimation
    - Freezing/unfreezing for MOE fine-tuning
    """

    def __init__(self, scheme_id: int, model: nn.Module, bond_length: float = 5.9):
        super().__init__()
        self.scheme_id = scheme_id
        self.model = model
        self.bond_length = bond_length

        # Confidence head: predicts how confident this expert is
        # Input: L, gc_content, closure_error → scalar confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seq_ids: torch.Tensor,       # (B, L)
        pair_probs: Optional[torch.Tensor] = None,
        coords_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run expert prediction.

        Returns:
            Dict with 'coords', 'confidence', 'closure_error'
        """
        # Dispatch to appropriate scheme
        if self.scheme_id == 1:
            out = self.model(seq_ids)
            coords = out['coords']
        elif self.scheme_id == 2:
            # Physics solver: run constraint solver
            coords = self._physics_solve(seq_ids, pair_probs)
        elif self.scheme_id == 3:
            # Dual-engine: needs initial coords
            if coords_init is None:
                coords_init = self._helical_init(seq_ids)
            coords = self.model(seq_ids, coords_init)
        elif self.scheme_id == 4:
            out = self.model(seq_tokens=seq_ids, pair_probs=pair_probs)
            coords = out['coords']
        elif self.scheme_id == 5:
            out = self.model(seq_ids)
            coords = out['coords']
        elif self.scheme_id == 6:
            out = self.model(seq_ids, mode='sample')
            coords = out['coords']
        elif self.scheme_id == 7:
            out = self.model(seq_tokens=seq_ids, pair_probs=pair_probs)
            coords = out.get('coords', coords_init if coords_init is not None else torch.zeros_like(seq_ids).unsqueeze(-1).expand(-1, -1, 3))
        else:
            raise ValueError(f"Unknown scheme: {self.scheme_id}")

        # Compute closure error for confidence estimation
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)  # (B,)
        closure_error = (closure_dist - self.bond_length).abs()  # (B,)

        # Confidence from closure error + sequence features
        B, L = seq_ids.shape
        gc = ((seq_ids == 2) | (seq_ids == 3)).float().mean(dim=-1, keepdim=True)  # (B, 1)
        conf_input = torch.stack([
            torch.ones(B, device=seq_ids.device) * L / 1000.0,
            gc.squeeze(-1),
            closure_error / 10.0,  # normalize
            closure_error.clamp(max=20) / 20.0,
        ], dim=-1)  # (B, 4)
        confidence = self.confidence_head(conf_input).squeeze(-1)  # (B,)

        return {
            'coords': coords,
            'confidence': confidence,
            'closure_error': closure_error,
        }

    def _helical_init(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """Generate helical initial coordinates for Scheme 3."""
        B, L = seq_ids.shape
        device = seq_ids.device
        coords = torch.zeros(B, L, 3, device=device)
        bond_length = self.bond_length
        for i in range(L):
            angle = 2 * np.pi * i / L
            radius = bond_length * L / (2 * np.pi) * 0.5
            coords[:, i, 0] = radius * np.cos(angle)
            coords[:, i, 1] = radius * np.sin(angle)
            coords[:, i, 2] = 2.8 * i - L * 2.8 / 2
        return coords

    def _physics_solve(self, seq_ids, pair_probs):
        """Fallback physics solver for Scheme 2."""
        B, L = seq_ids.shape
        device = seq_ids.device
        coords = self._helical_init(seq_ids)
        return coords


# ═══════════════════════════════════════════════════════════════
# Expert Fusion
# ═══════════════════════════════════════════════════════════════

class ExpertFusion(nn.Module):
    """Fuse predictions from multiple experts.

    Modes:
    1. 'weighted_avg': gate-weighted average of expert predictions
    2. 'confidence': weight by expert confidence scores
    3. 'stacked_refine': stack predictions, refine with small network
    """

    def __init__(self, n_experts: int, top_k: int, d_hidden: int = 128,
                 fusion_mode: str = 'weighted_avg', bond_length: float = 5.9):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.bond_length = bond_length

        if fusion_mode == 'stacked_refine':
            # Stack top-K predictions → small refinement network
            self.refiner = nn.Sequential(
                nn.Linear(top_k * 3, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, 3),
            )

        # BSJ closure correction (always applied)
        self.closure_corrector = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),
        )

    def forward(
        self,
        expert_coords: List[torch.Tensor],  # List of (B, L, 3) from each expert
        gate_weights: torch.Tensor,           # (B, top_k)
        expert_indices: torch.Tensor,         # (B, top_k)
        confidences: Optional[List[torch.Tensor]] = None,  # List of (B,)
    ) -> torch.Tensor:
        """Fuse expert predictions.

        Returns:
            (B, L, 3) fused coordinates
        """
        B, top_k = gate_weights.shape

        if self.fusion_mode == 'weighted_avg':
            coords = self._weighted_avg(expert_coords, gate_weights, expert_indices)
        elif self.fusion_mode == 'confidence':
            coords = self._confidence_weighted(expert_coords, gate_weights,
                                                expert_indices, confidences)
        elif self.fusion_mode == 'stacked_refine':
            coords = self._stacked_refine(expert_coords, gate_weights, expert_indices)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # Post-hoc BSJ closure enforcement
        coords = self._enforce_closure(coords)

        return coords

    def _weighted_avg(self, expert_coords, gate_weights, expert_indices):
        """Gate-weighted average of selected expert predictions."""
        B = gate_weights.shape[0]
        L = expert_coords[0].shape[1]
        device = expert_coords[0].device

        fused = torch.zeros(B, L, 3, device=device)

        for k in range(self.top_k):
            for b in range(B):
                expert_idx = expert_indices[b, k].item()
                weight = gate_weights[b, k]
                fused[b] += weight * expert_coords[expert_idx][b]

        return fused

    def _confidence_weighted(self, expert_coords, gate_weights, expert_indices, confidences):
        """Weight by both gate weights and expert confidence."""
        B = gate_weights.shape[0]
        L = expert_coords[0].shape[1]
        device = expert_coords[0].device

        fused = torch.zeros(B, L, 3, device=device)

        for k in range(self.top_k):
            for b in range(B):
                expert_idx = expert_indices[b, k].item()
                gate_w = gate_weights[b, k]
                conf_w = confidences[expert_idx][b] if confidences else 1.0
                combined_w = gate_w * conf_w
                fused[b] += combined_w * expert_coords[expert_idx][b]

        # Renormalize
        total_w = torch.zeros(B, 1, 1, device=device)
        for k in range(self.top_k):
            for b in range(B):
                expert_idx = expert_indices[b, k].item()
                gate_w = gate_weights[b, k]
                conf_w = confidences[expert_idx][b] if confidences else 1.0
                total_w[b] += gate_w * conf_w
        total_w = total_w.clamp(min=1e-8)
        fused = fused / total_w

        return fused

    def _stacked_refine(self, expert_coords, gate_weights, expert_indices):
        """Stack top-K predictions and refine with small network."""
        B = gate_weights.shape[0]
        L = expert_coords[0].shape[1]
        device = expert_coords[0].device

        # Stack selected expert predictions
        stacked = torch.zeros(B, L, self.top_k * 3, device=device)
        for k in range(self.top_k):
            for b in range(B):
                expert_idx = expert_indices[b, k].item()
                stacked[b, :, k*3:(k+1)*3] = expert_coords[expert_idx][b]

        # Refine
        refined = self.refiner(stacked)  # (B, L, 3)

        # Residual: weighted avg as base + refinement
        base = self._weighted_avg(expert_coords, gate_weights, expert_indices)
        coords = base + 0.1 * refined

        return coords

    def _enforce_closure(self, coords: torch.Tensor) -> torch.Tensor:
        """Enforce BSJ closure on fused coordinates."""
        B, L, _ = coords.shape

        # Move last atom to bond_length distance from first
        direction = coords[:, 0] - coords[:, -2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        coords[:, -1] = coords[:, 0] - self.bond_length * direction

        return coords


# ═══════════════════════════════════════════════════════════════
# TorusFold MOE Model
# ═══════════════════════════════════════════════════════════════

class TorusFoldMOE(nn.Module):
    """TorusFold Mixture-of-Experts: dynamic routing across all schemes.

    Training phases:
        Phase 1: Pre-train each expert independently (already done)
        Phase 2: Freeze experts, train gating network + fusion
        Phase 3 (optional): Fine-tune everything end-to-end

    Inference:
        Input sequence → Gating selects top-K experts
                      → Each expert predicts 3D coords
                      → Fusion produces final coordinates
    """

    # Expert scheme descriptions for reference
    EXPERT_INFO = {
        1: {'name': 'EGNN+Physics', 'complexity': 'medium', 'max_len': 1000,
            'strength': 'equivariant, good for local geometry'},
        2: {'name': 'Batch+Physics', 'complexity': 'low', 'max_len': None,
            'strength': 'no training, always available as fallback'},
        3: {'name': 'Dual-Engine', 'complexity': 'medium', 'max_len': 1000,
            'strength': 'Transformer+physics, good refinement'},
        4: {'name': 'DDPM+EGNN', 'complexity': 'high', 'max_len': 1000,
            'strength': 'diffusion quality, complex folds'},
        5: {'name': 'CircPairformer', 'complexity': 'medium', 'max_len': 1000,
            'strength': 'physics-biased attention, pair-aware'},
        6: {'name': 'GNN+LatentDiff', 'complexity': 'high', 'max_len': 800,
            'strength': 'latent diffusion, diverse conformations'},
        7: {'name': 'Mamba+Diffusion', 'complexity': 'high', 'max_len': None,
            'strength': 'O(L) Mamba, long sequences, efficient'},
    }

    def __init__(self, config: Optional[TorusFoldMOEConfig] = None):
        super().__init__()
        self.config = config or TorusFoldMOEConfig()

        # Feature extractor
        self.feature_extractor = SequenceFeatureExtractor(
            d_out=self.config.d_gate
        )

        # Gating network
        self.gating = SeqTopKGating(
            d_input=self.config.d_gate,
            n_experts=self.config.n_experts,
            top_k=self.config.top_k,
        )

        # Expert wrappers (initialized empty, loaded from pretrained)
        self.experts = nn.ModuleDict()

        # Fusion module
        self.fusion = ExpertFusion(
            n_experts=self.config.n_experts,
            top_k=self.config.top_k,
            d_hidden=self.config.expert_hidden_dim,
            fusion_mode=self.config.fusion_mode,
            bond_length=self.config.bond_length,
        )

    def register_expert(self, scheme_id: int, model: nn.Module):
        """Register a pretrained scheme model as an expert.

        Args:
            scheme_id: 1-7
            model: pretrained model for this scheme
        """
        wrapper = ExpertWrapper(
            scheme_id=scheme_id,
            model=model,
            bond_length=self.config.bond_length,
        )

        if self.config.freeze_experts:
            for param in wrapper.model.parameters():
                param.requires_grad = False

        self.experts[str(scheme_id)] = wrapper
        print(f"  Registered expert {scheme_id}: {self.EXPERT_INFO[scheme_id]['name']}")

    def load_expert_from_file(self, scheme_id: int, model: nn.Module, checkpoint_path: str):
        """Load a pretrained expert from checkpoint file."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        self.register_expert(scheme_id, model)

    def forward(
        self,
        seq_ids: torch.Tensor,       # (B, L)
        lengths: Optional[List[int]] = None,
        pair_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """MOE forward: route, predict, fuse.

        Returns:
            Dict with 'coords', 'gate_weights', 'expert_indices',
            'expert_coords', 'confidences', 'load_balance_loss'
        """
        B, L = seq_ids.shape
        device = seq_ids.device

        if lengths is None:
            lengths = [L] * B

        # Step 1: Extract sequence features for routing
        seq_features = self.feature_extractor(seq_ids, lengths, pair_probs)  # (B, d_gate)

        # Step 2: Gating — select top-K experts
        gate_weights, expert_indices, gate_logits = self.gating(seq_features)
        # gate_weights: (B, top_k), expert_indices: (B, top_k)

        # Step 3: Run selected experts
        expert_coords = {}  # scheme_id → (B, L, 3)
        confidences = {}    # scheme_id → (B,)

        # Get unique experts needed (avoid redundant computation)
        unique_experts = expert_indices.unique().tolist()

        for eid in unique_experts:
            key = str(eid)
            if key not in self.experts:
                # Expert not loaded, use helical fallback
                coords = self._helical_fallback(seq_ids)
                expert_coords[eid] = coords
                confidences[eid] = torch.zeros(B, device=device)
                continue

            out = self.experts[key](
                seq_ids=seq_ids,
                pair_probs=pair_probs,
            )
            expert_coords[eid] = out['coords']
            confidences[eid] = out['confidence']

        # Step 4: Fuse expert predictions
        coords_list = [expert_coords.get(i, self._helical_fallback(seq_ids))
                       for i in range(self.config.n_experts)]
        conf_list = [confidences.get(i, torch.zeros(B, device=device))
                     for i in range(self.config.n_experts)]

        final_coords = self.fusion(
            expert_coords=coords_list,
            gate_weights=gate_weights,
            expert_indices=expert_indices,
            confidences=conf_list if self.config.use_confidence else None,
        )

        # Step 5: Load balancing loss
        load_balance_loss = self.gating.load_balance_loss(gate_logits)

        return {
            'coords': final_coords,
            'gate_weights': gate_weights,
            'expert_indices': expert_indices,
            'expert_coords': expert_coords,
            'confidences': confidences,
            'load_balance_loss': load_balance_loss,
        }

    def _helical_fallback(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """Generate helical coordinates as fallback for missing experts."""
        B, L = seq_ids.shape
        device = seq_ids.device
        coords = torch.zeros(B, L, 3, device=device)
        bond_length = self.config.bond_length
        for i in range(L):
            angle = 2 * np.pi * i / L
            radius = bond_length * L / (2 * np.pi) * 0.5
            coords[:, i, 0] = radius * np.cos(angle)
            coords[:, i, 1] = radius * np.sin(angle)
            coords[:, i, 2] = 2.8 * i - L * 2.8 / 2
        return coords


# ═══════════════════════════════════════════════════════════════
# MOE Training
# ═══════════════════════════════════════════════════════════════

def train_torusfold_moe(
    train_loader,
    val_loader,
    args,
    device,
    pretrained_paths: Optional[Dict[int, str]] = None,
):
    """Train TorusFold MOE model.

    Phase 2 training: freeze experts, train gating + fusion.

    Args:
        pretrained_paths: {scheme_id: path_to_checkpoint}
    """
    from torch.utils.data import DataLoader

    config = TorusFoldMOEConfig(
        n_experts=7,
        top_k=2,
        expert_hidden_dim=args.d_hidden,
        freeze_experts=True,
        load_balance_weight=0.01,
    )

    model = TorusFoldMOE(config).to(device)

    # Load pretrained experts
    if pretrained_paths:
        for scheme_id, ckpt_path in pretrained_paths.items():
            try:
                expert_model = _build_scheme_model(scheme_id, args)
                state_dict = torch.load(ckpt_path, map_location=device)
                expert_model.load_state_dict(state_dict, strict=False)
                model.register_expert(scheme_id, expert_model)
            except Exception as e:
                print(f"  Warning: failed to load expert {scheme_id}: {e}")
    else:
        print("  No pretrained paths provided, experts will use helical fallback")

    # Optimizer: only gating + fusion + confidence heads
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"  Trainable: {name} ({param.numel()} params)")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.gate_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_metrics = {'coord': 0, 'closure': 0, 'balance': 0}
        route_stats = {i: 0 for i in range(1, 8)}  # Track expert usage

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']
            pair_probs = batch.get('pair_probs', None)
            if pair_probs is not None:
                pair_probs = pair_probs.to(device)

            out = model(seq_ids, lengths, pair_probs)
            pred = out['coords']

            # Coordinate loss (per-residue MSE, only valid positions)
            B = len(lengths)
            coord_loss = 0
            for b in range(B):
                valid_L = lengths[b]
                # Center both
                p = pred[b, :valid_L]
                t = target[b, :valid_L]
                p_c = p - p.mean(dim=0)
                t_c = t - t.mean(dim=0)
                coord_loss += torch.mean(torch.sum((p_c - t_c) ** 2, dim=1))
            coord_loss /= B

            # Closure loss
            closure_dist = torch.norm(pred[:, 0] - pred[:, -1], dim=-1)
            closure_loss = ((closure_dist - config.bond_length) ** 2).mean()

            # Load balancing loss
            balance_loss = out['load_balance_loss']

            # Total loss
            loss = coord_loss + 1.0 * closure_loss + config.load_balance_weight * balance_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_metrics['coord'] += coord_loss.item()
            train_metrics['closure'] += closure_loss.item()
            train_metrics['balance'] += balance_loss.item()

            # Track routing statistics
            indices = out['expert_indices']  # (B, top_k)
            for b in range(B):
                for k in range(config.top_k):
                    eid = indices[b, k].item() + 1  # 1-indexed
                    route_stats[eid] = route_stats.get(eid, 0) + 1

        n_batches = len(train_loader)
        avg_train = train_loss / n_batches
        for k in train_metrics:
            train_metrics[k] /= n_batches

        scheduler.step()

        # Validation
        model.eval()
        val_rmsd = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']
                pair_probs = batch.get('pair_probs', None)

                out = model(seq_ids, lengths, pair_probs)
                pred = out['coords']

                B = len(lengths)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred[b, :valid_L]
                    t = target[b, :valid_L]
                    if not (torch.isnan(p).any() or torch.isinf(p).any()):
                        # Kabsch RMSD
                        p_c = p - p.mean(dim=0)
                        t_c = t - t.mean(dim=0)
                        H = t_c.T @ p_c
                        try:
                            U, S, Vt = torch.linalg.svd(H)
                            d = torch.sign(torch.det(Vt.T @ U.T))
                            D = torch.diag(torch.tensor([1, 1, d], device=device, dtype=torch.float32))
                            R = Vt.T @ D @ U.T
                            p_aligned = (R @ p_c.T).T
                            rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
                        except Exception:
                            rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                        val_rmsd += rmsd.item()
                        n_val += 1

        avg_val = val_rmsd / max(n_val, 1)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/torusfold_moe_best.pt")
        else:
            patience_counter += 1

        # Routing statistics
        total_routes = sum(route_stats.values())
        route_str = ' '.join(
            f"S{k}:{v/total_routes:.1%}" for k, v in sorted(route_stats.items()) if v > 0
        )

        print(f"  Epoch {epoch+1}/{args.epochs} "
              f"train={avg_train:.4f} (coord={train_metrics['coord']:.3f}, "
              f"closure={train_metrics['closure']:.3f}, bal={train_metrics['balance']:.4f}) "
              f"val={avg_val:.2f}Å pat={patience_counter}/10 "
              f"route=[{route_str}]")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  MOE training complete: best_val={best_val:.4f}Å")
    return best_val


def _build_scheme_model(scheme_id: int, args):
    """Build an empty scheme model for loading pretrained weights."""
    if scheme_id == 1:
        from confluencia_3_0.core.circrna.torusfold.train_torusfold_3d import CircRNA3DModel
        return CircRNA3DModel(d_hidden=args.d_hidden, n_layers=args.n_layers)
    elif scheme_id == 5:
        # Scheme 5 is defined inline in train_all_schemes.py
        # We need to recreate it here
        class Scheme5Model(nn.Module):
            def __init__(self, d_model=128, n_heads=4, n_blocks=4):
                super().__init__()
                self.embed = nn.Embedding(5, d_model)
                self.circ_pos = nn.Embedding(512, d_model)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads,
                        dim_feedforward=d_model * 2,
                        dropout=0.1, batch_first=True,
                    )
                    for _ in range(n_blocks)
                ])
                self.coord_head = nn.Linear(d_model, 3)
                self.bond_length = 5.9

            def forward(self, seq_ids, coords_init=None):
                B, L = seq_ids.shape
                device = seq_ids.device
                pos = torch.arange(L, device=device) % 512
                h = self.embed(seq_ids) + self.circ_pos(pos)
                for block in self.blocks:
                    h = block(h)
                coords = self.coord_head(h)
                return {'coords': coords}

        return Scheme5Model(d_model=args.d_hidden, n_blocks=args.n_layers)
    elif scheme_id == 7:
        from confluencia_3_0.core.circrna.torusfold.circrna_mamba_diffusion import (
            CircMambaDiffusionModel, CircMambaConfig
        )
        config = CircMambaConfig(d_model=args.d_hidden)
        return CircMambaDiffusionModel(config)
    else:
        raise ValueError(f"Cannot auto-build scheme {scheme_id}, provide model directly")
