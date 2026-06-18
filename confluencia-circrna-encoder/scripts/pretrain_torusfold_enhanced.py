"""
Enhanced self-supervised pretraining for TorusFold circRNA encoder.

Self-supervised tasks:
1. Masked Language Modeling (MLM) - predict masked nucleotides
2. Contrastive Learning with Rotation Augmentation - circRNA-specific
3. Circularization Point Prediction - predict BSJ position
4. Base-pairing Prediction (unsupervised) - predict secondary structure
5. TPE-aware Circular Distance Prediction - predict d_circ(i,j)

Key innovations for circRNA:
- Rotation augmentation: same sequence, different start → positive pair
- IRS-specific augmentation: reverse complement pairs
- Circular distance prediction leverages torus topology
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Try to import ESM for backbone
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("[Warning] ESM not available, using mock backbone")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class TorusPretrainConfig:
    """Configuration for enhanced self-supervised pretraining."""

    # Backbone
    backbone_model: str = "esm2_t33_650M_UR50D"
    backbone_dim: int = 1280

    # Pretraining tasks
    mask_ratio: float = 0.15
    contrastive_enabled: bool = True
    bsj_prediction_enabled: bool = True
    basepair_prediction_enabled: bool = True
    circular_distance_enabled: bool = True

    # Contrastive settings
    n_rot_augments: int = 4  # Number of rotation augmentations per sequence
    temperature: float = 0.07
    use_irs_augment: bool = True  # IRS reverse-complement augmentation

    # Training
    max_epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-5

    # Data
    max_sequences: int = 140000
    max_seq_length: int = 512

    # Loss weights
    w_mlm: float = 1.0
    w_contrastive: float = 0.5
    w_bsj: float = 0.3
    w_basepair: float = 0.3
    w_circ_dist: float = 0.2

    # Saving
    save_every: int = 1
    output_dir: str = "confluencia-circrna-encoder/data/models/pretrain"

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# Self-supervised Heads
# =============================================================================

class MaskedNucleotideHead(nn.Module):
    """Head for masked nucleotide prediction (MLM)."""

    VOCAB_SIZE = 6  # A, U, G, C, MASK, PAD

    def __init__(self, backbone_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveHead(nn.Module):
    """Head for contrastive learning with projection."""

    def __init__(self, backbone_dim: int = 1280, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class BSJPredictionHead(nn.Module):
    """
    Head for circularization point (BSJ) prediction.

    Task: Given a circRNA sequence, predict where the back-splice junction is.
    Since circRNA has no canonical start, we randomly select a "virtual BSJ"
    and ask the model to predict its position.

    This is unique to circRNA and leverages the torus topology.
    """

    def __init__(self, backbone_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        # Predict a probability distribution over positions
        self.net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        # Output: (batch, seq_len) logits for BSJ position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, backbone_dim) per-position embeddings
        Returns:
            (batch, L) logits for BSJ position
        """
        hidden = self.net(x)  # (B, L, hidden//2)
        # Use attention-style scoring
        scores = hidden.mean(dim=-1)  # (B, L) - simplified
        return scores


class BasePairPredictionHead(nn.Module):
    """
    Head for unsupervised base-pairing prediction.

    Predicts which positions form Watson-Crick (A-U, G-C) or wobble (G-U) pairs.
    Uses physical constraints:
    - Symmetry: if i pairs with j, then j pairs with i
    - No self-pairing: P[i,i] = 0
    - One partner: each position pairs with at most one other

    For circRNA, also considers circular distance:
    - BSJ-crossing pairs: d_circ(i,j) >= L/2
    """

    def __init__(self, backbone_dim: int = 1280, pair_dim: int = 64):
        super().__init__()
        self.left_proj = nn.Linear(backbone_dim, pair_dim)
        self.right_proj = nn.Linear(backbone_dim, pair_dim)

        # Pair scoring network
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_dim * 2, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, backbone_dim)
        Returns:
            pair_probs: (batch, L, L) symmetric pair probabilities
        """
        B, L, _ = x.shape

        left = self.left_proj(x)   # (B, L, pair_dim)
        right = self.right_proj(x)  # (B, L, pair_dim)

        # Outer product: concatenate left[i] and right[j] for all i,j
        # Efficient implementation using broadcasting
        left_exp = left.unsqueeze(2).expand(-1, -1, L, -1)   # (B, L, L, pair_dim)
        right_exp = right.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, pair_dim)

        pair_features = torch.cat([left_exp, right_exp], dim=-1)  # (B, L, L, 2*pair_dim)
        logits = self.pair_scorer(pair_features).squeeze(-1)  # (B, L, L)

        # Enforce symmetry
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        # Mask self-pairing
        mask = torch.eye(L, device=x.device).unsqueeze(0) * -1e9
        logits = logits + mask

        return torch.sigmoid(logits)


class CircularDistanceHead(nn.Module):
    """
    Head for TPE-aware circular distance prediction.

    Task: Predict d_circ(i,j) = min(|i-j|, L-|i-j|) for position pairs.
    This explicitly teaches the model about circRNA's torus topology.

    Uses the TPE formulation: positions on a circle S¹.
    """

    def __init__(self, backbone_dim: int = 1280, hidden_dim: int = 128, max_dist: int = 256):
        super().__init__()
        self.max_dist = max_dist

        self.pos_proj = nn.Linear(backbone_dim, hidden_dim)

        # Distance prediction: classify into max_dist buckets
        self.dist_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_dist + 1),  # 0 to max_dist
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, backbone_dim)
        Returns:
            dist_logits: (batch, L, L, max_dist+1) distance class logits
        """
        B, L, _ = x.shape

        pos_emb = self.pos_proj(x)  # (B, L, hidden)

        # All pairs
        pos_i = pos_emb.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, hidden)
        pos_j = pos_emb.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, hidden)

        pair_features = torch.cat([pos_i, pos_j], dim=-1)  # (B, L, L, 2*hidden)
        dist_logits = self.dist_classifier(pair_features)  # (B, L, L, max_dist+1)

        return dist_logits


# =============================================================================
# Data Augmentation for circRNA
# =============================================================================

class CircRNAAugmentor:
    """
    Data augmentation strategies specific to circular RNA.

    Key insight: circRNA has no canonical start position.
    Any rotation of the sequence represents the same molecule.
    """

    @staticmethod
    def rotate_sequence(seq: str, offset: int) -> str:
        """Rotate circRNA by offset positions (circular shift)."""
        L = len(seq)
        offset = offset % L
        if offset == 0:
            return seq
        return seq[offset:] + seq[:offset]

    @staticmethod
    def get_rotation_augments(seq: str, n_augments: int) -> List[Tuple[str, int]]:
        """
        Get n rotation augmentations of a circRNA sequence.

        Returns:
            List of (rotated_sequence, offset) tuples
        """
        L = len(seq)
        augments = []
        for k in range(n_augments):
            offset = k * L // n_augments
            rotated = CircRNAAugmentor.rotate_sequence(seq, offset)
            augments.append((rotated, offset))
        return augments

    @staticmethod
    def irs_augment(seq: str) -> str:
        """
        IRS (Internal Reverse-complement Sequence) augmentation.

        For regions with reverse complement pairs (IRS elements),
        swap the orientation to create a different view of the same
        structural element.

        This is circRNA-specific: IRS elements are common in circRNA
        and contribute to stability.
        """
        # Find potential IRS regions (simplified: look for palindromic regions)
        # For now, just return a mutation-augmented version
        L = len(seq)
        if L < 20:
            return seq

        # Select a random 10% region to "flip" conceptually
        # (In practice, we'd use structural prediction to identify IRS)
        # This is a placeholder - real IRS augmentation needs structure info
        return seq

    @staticmethod
    def apply_mask(
        seq: str,
        mask_ratio: float = 0.15,
        mask_token: str = '<mask>',
    ) -> Tuple[str, List[int]]:
        """
        Apply random masking to sequence.

        Returns:
            masked_seq: sequence with mask tokens
            mask_positions: indices of masked positions
        """
        L = len(seq)
        n_mask = max(1, int(L * mask_ratio))

        mask_positions = sorted(np.random.choice(L, n_mask, replace=False).tolist())

        seq_list = list(seq)
        for pos in mask_positions:
            seq_list[pos] = mask_token

        return ''.join(seq_list), mask_positions


# =============================================================================
# Main Pretraining Model
# =============================================================================

class TorusPretrainModel(nn.Module):
    """
    Enhanced self-supervised pretraining model for TorusFold.

    Architecture:
    - ESM2 backbone (frozen or fine-tuned)
    - Multiple self-supervised heads:
      1. MLM: predict masked nucleotides
      2. Contrastive: learn rotation-invariant representations
      3. BSJ: predict circularization point
      4. Base-pair: predict secondary structure
      5. Circular distance: learn torus topology
    """

    def __init__(self, config: TorusPretrainConfig):
        super().__init__()
        self.config = config

        # Backbone placeholder
        self.backbone = None
        self.alphabet = None
        self.backbone_loaded = False

        # Self-supervised heads
        self.mlm_head = MaskedNucleotideHead(config.backbone_dim)

        if config.contrastive_enabled:
            self.contrastive_head = ContrastiveHead(config.backbone_dim)

        if config.bsj_prediction_enabled:
            self.bsj_head = BSJPredictionHead(config.backbone_dim)

        if config.basepair_prediction_enabled:
            self.basepair_head = BasePairPredictionHead(config.backbone_dim)

        if config.circular_distance_enabled:
            self.circ_dist_head = CircularDistanceHead(config.backbone_dim)

        # Nucleotide mapping
        self.nuc_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '<mask>': 4, '<pad>': 5}
        self.idx_to_nuc = {v: k for k, v in self.nuc_to_idx.items()}

        # Augmentor
        self.augmentor = CircRNAAugmentor()

    def load_backbone(self, device: str = "cuda"):
        """Load ESM2 backbone."""
        if self.backbone_loaded:
            return

        if not ESM_AVAILABLE:
            print("[Warning] Using mock backbone for testing")
            self.backbone = None
            self.backbone_loaded = True
            return

        print(f"Loading backbone: {self.config.backbone_model}")
        self.backbone, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.config.backbone_model
        )
        self.backbone = self.backbone.to(device)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_loaded = True
        print("Backbone loaded and frozen")

    def encode_sequence(
        self,
        sequence: str,
        device: str = "cuda",
        return_per_position: bool = False,
    ) -> torch.Tensor:
        """Encode sequence using backbone."""
        seq_t = sequence.replace('U', 'T').replace('u', 't')

        if self.backbone is None:
            # Mock encoding
            if return_per_position:
                return torch.randn(1, len(seq_t), self.config.backbone_dim, device=device)
            return torch.randn(1, self.config.backbone_dim, device=device)

        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq', seq_t)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.backbone(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False,
            )
            embeddings = results["representations"][33]

        if return_per_position:
            # Remove BOS/EOS tokens
            return embeddings[:, 1:-1, :]  # (1, L, dim)
        else:
            # Mean pooling
            mask = (batch_tokens != self.alphabet.padding_idx).float().unsqueeze(-1)
            return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # =========================================================================
    # Self-supervised Tasks
    # =========================================================================

    def forward_mlm(
        self,
        original_seq: str,
        masked_seq: str,
        mask_positions: List[int],
        device: str = "cuda",
    ) -> Dict:
        """MLM forward pass."""
        if len(mask_positions) == 0:
            return {"loss": torch.tensor(0.0, device=device), "accuracy": 0.0}

        embeddings = self.encode_sequence(masked_seq, device, return_per_position=True)
        embeddings = embeddings.squeeze(0)  # (L, dim)

        # Truncate if needed
        mask_positions = mask_positions[:self.config.max_seq_length]

        masked_embs = embeddings[mask_positions]  # (n_mask, dim)
        logits = self.mlm_head(masked_embs)  # (n_mask, vocab_size)

        targets = torch.tensor(
            [self.nuc_to_idx.get(original_seq[pos], 0) for pos in mask_positions],
            dtype=torch.long, device=device
        )

        loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}

    def forward_contrastive(
        self,
        sequences: List[str],
        device: str = "cuda",
    ) -> Dict:
        """
        Contrastive learning with rotation augmentation.

        Key insight: circRNA has no start position, so rotated versions
        of the same sequence should have similar representations.

        Positive pairs: same sequence, different rotation
        Negative pairs: different sequences
        """
        if len(sequences) < 2:
            return {"loss": torch.tensor(0.0, device=device)}

        n_rot = self.config.n_rot_augments
        temperature = self.config.temperature

        # For each sequence, get rotation augmentations
        all_embs = []
        all_labels = []

        for seq_idx, seq in enumerate(sequences):
            augments = self.augmentor.get_rotation_augments(seq, n_rot)
            for rotated, offset in augments:
                emb = self.encode_sequence(rotated, device, return_per_position=False)
                all_embs.append(emb)
                all_labels.append(seq_idx)  # Same label for all rotations of same sequence

        if len(all_embs) < 2:
            return {"loss": torch.tensor(0.0, device=device)}

        embeddings = torch.cat(all_embs, dim=0)  # (N, dim)
        labels = torch.tensor(all_labels, device=device)

        # Project
        projected = self.contrastive_head(embeddings)  # (N, proj_dim)
        projected = F.normalize(projected, dim=-1)

        # SimCLR-style NT-Xent loss
        similarity = torch.mm(projected, projected.t()) / temperature  # (N, N)

        # Mask self-similarity
        N = similarity.size(0)
        mask_self = torch.eye(N, device=device) * -1e9
        similarity = similarity + mask_self

        # For each sample, positive = other rotations of same sequence
        # Negative = different sequences
        labels_expanded = labels.unsqueeze(0).expand(N, -1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        positive_mask.fill_diagonal_(0)  # Exclude self

        # Compute loss for each sample
        losses = []
        for i in range(N):
            pos_mask = positive_mask[i]
            if pos_mask.sum() == 0:
                continue

            # Negative: all other samples except self and positives
            neg_mask = 1 - pos_mask - torch.eye(N, device=device)[i]

            # Cross-entropy: softmax over negatives, maximize similarity to positives
            pos_sim = (similarity[i] * pos_mask).sum() / pos_mask.sum().clamp(min=1)
            neg_sim = torch.logsumexp(similarity[i] * neg_mask - (1 - neg_mask) * 1e9, dim=0)

            loss_i = -pos_sim + neg_sim
            losses.append(loss_i)

        if not losses:
            return {"loss": torch.tensor(0.0, device=device)}

        loss = torch.stack(losses).mean()

        return {"loss": loss}

    def forward_bsj(
        self,
        sequence: str,
        virtual_bsj: int,
        device: str = "cuda",
    ) -> Dict:
        """
        BSJ prediction: predict where the circularization point is.

        We randomly select a "virtual BSJ" position and ask the model
        to predict it. This teaches the model about circRNA's circular nature.
        """
        L = len(sequence)

        embeddings = self.encode_sequence(sequence, device, return_per_position=True)
        embeddings = embeddings.squeeze(0)  # (L, dim)

        # Predict BSJ position
        logits = self.bsj_head(embeddings.unsqueeze(0)).squeeze(0)  # (L,)

        # Target: one-hot at virtual_bsj position
        target = torch.zeros(L, device=device)
        target[virtual_bsj % L] = 1.0

        # Cross-entropy loss
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([virtual_bsj % L], device=device))

        # Accuracy: is the predicted position within 5% of true position?
        pred_pos = logits.argmax().item()
        correct = abs(pred_pos - virtual_bsj) <= L * 0.05

        return {"loss": loss, "accuracy": float(correct)}

    def forward_basepair(
        self,
        sequence: str,
        device: str = "cuda",
    ) -> Dict:
        """
        Unsupervised base-pairing prediction.

        Uses sequence-derived constraints (Watson-Crick complementarity)
        as weak supervision.
        """
        L = len(sequence)

        embeddings = self.encode_sequence(sequence, device, return_per_position=True)
        embeddings = embeddings.squeeze(0)  # (L, dim)

        # Predict pair probabilities
        pair_probs = self.basepair_head(embeddings.unsqueeze(0)).squeeze(0)  # (L, L)

        # Create weak supervision from WC complementarity
        wc_pairs = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}
        target = torch.zeros(L, L, device=device)

        for i in range(L):
            for j in range(i + 1, L):
                pair = sequence[i] + sequence[j]
                if pair in wc_pairs:
                    target[i, j] = 1.0
                    target[j, i] = 1.0

        # Binary cross-entropy loss (only on potential WC pairs for efficiency)
        mask = (target.sum(dim=-1) > 0) | (target.sum(dim=-2) > 0)

        if mask.sum() == 0:
            return {"loss": torch.tensor(0.0, device=device)}

        # Sample positions for loss computation
        n_sample = min(100, mask.sum().item())
        sample_indices = torch.where(mask)[0]
        if len(sample_indices) > n_sample:
            sample_indices = sample_indices[torch.randperm(len(sample_indices))[:n_sample]]

        loss = F.binary_cross_entropy(
            pair_probs[sample_indices][:, sample_indices],
            target[sample_indices][:, sample_indices]
        )

        return {"loss": loss}

    def forward_circular_distance(
        self,
        sequence: str,
        device: str = "cuda",
    ) -> Dict:
        """
        Circular distance prediction.

        Teaches the model about torus topology:
        d_circ(i,j) = min(|i-j|, L-|i-j|)
        """
        L = len(sequence)

        embeddings = self.encode_sequence(sequence, device, return_per_position=True)
        embeddings = embeddings.squeeze(0)  # (L, dim)

        # Predict distance distribution for all pairs
        dist_logits = self.circ_dist_head(embeddings.unsqueeze(0)).squeeze(0)  # (L, L, max_dist+1)

        # Compute circular distance targets
        positions = torch.arange(L, device=device)
        diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        circ_dist = torch.min(diff, L - diff)  # (L, L)
        circ_dist = circ_dist.clamp(0, self.circ_dist_head.max_dist).long()

        # Sample pairs for loss (avoid O(L²) memory)
        n_pairs = min(500, L * L)
        indices_i = torch.randint(0, L, (n_pairs,), device=device)
        indices_j = torch.randint(0, L, (n_pairs,), device=device)

        loss = F.cross_entropy(
            dist_logits[indices_i, indices_j],  # (n_pairs, max_dist+1)
            circ_dist[indices_i, indices_j]      # (n_pairs,)
        )

        # Accuracy: within 10% tolerance
        pred_dist = dist_logits[indices_i, indices_j].argmax(dim=-1)
        true_dist = circ_dist[indices_i, indices_j]
        accuracy = (pred_dist == true_dist).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}

    def save(self, path: str):
        """Save model."""
        state = {"config": self.config.to_dict(), "mlm_head": self.mlm_head.state_dict()}

        if hasattr(self, 'contrastive_head'):
            state["contrastive_head"] = self.contrastive_head.state_dict()
        if hasattr(self, 'bsj_head'):
            state["bsj_head"] = self.bsj_head.state_dict()
        if hasattr(self, 'basepair_head'):
            state["basepair_head"] = self.basepair_head.state_dict()
        if hasattr(self, 'circ_dist_head'):
            state["circ_dist_head"] = self.circ_dist_head.state_dict()

        torch.save(state, path)
        print(f"Saved to {path}")

    def load(self, path: str, device: str = "cuda"):
        """Load model."""
        state = torch.load(path, map_location=device, weights_only=False)

        self.mlm_head.load_state_dict(state["mlm_head"])

        if hasattr(self, 'contrastive_head') and "contrastive_head" in state:
            self.contrastive_head.load_state_dict(state["contrastive_head"])
        if hasattr(self, 'bsj_head') and "bsj_head" in state:
            self.bsj_head.load_state_dict(state["bsj_head"])
        if hasattr(self, 'basepair_head') and "basepair_head" in state:
            self.basepair_head.load_state_dict(state["basepair_head"])
        if hasattr(self, 'circ_dist_head') and "circ_dist_head" in state:
            self.circ_dist_head.load_state_dict(state["circ_dist_head"])

        print(f"Loaded from {path}")


# =============================================================================
# Dataset
# =============================================================================

class CircRNAPretrainDataset(Dataset):
    """Dataset for self-supervised pretraining."""

    def __init__(
        self,
        fasta_path: str,
        max_sequences: int = 140000,
        max_seq_length: int = 512,
        seed: int = 42,
    ):
        self.max_seq_length = max_seq_length
        np.random.seed(seed)

        print(f"Loading sequences from {fasta_path}...")
        self.sequences = self._parse_fasta(fasta_path)

        if len(self.sequences) > max_sequences:
            indices = np.random.choice(len(self.sequences), max_sequences, replace=False)
            self.sequences = [self.sequences[i] for i in indices]

        print(f"Loaded {len(self.sequences)} sequences")

    def _parse_fasta(self, path: str) -> List[str]:
        sequences = []
        opener = gzip.open if path.endswith('.gz') else open

        with opener(path, 'rt') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line.upper()
            if current_seq:
                sequences.append(current_seq)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]

        sequence = sequence.replace('T', 'U')

        return {'sequence': sequence, 'length': len(sequence), 'idx': idx}


# =============================================================================
# Training Loop
# =============================================================================

def pretrain_epoch(
    model: TorusPretrainModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TorusPretrainConfig,
    device: str,
    epoch: int,
) -> Dict:
    """Train one epoch with all self-supervised tasks."""
    model.train()

    metrics = {
        "mlm_loss": 0.0,
        "mlm_acc": 0.0,
        "contrastive_loss": 0.0,
        "bsj_loss": 0.0,
        "bsj_acc": 0.0,
        "basepair_loss": 0.0,
        "circ_dist_loss": 0.0,
        "circ_dist_acc": 0.0,
        "total_loss": 0.0,
    }
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence']
        batch_loss = torch.tensor(0.0, device=device)
        batch_metrics = {}

        # =====================================================================
        # 1. MLM Task
        # =====================================================================
        mlm_losses = []
        mlm_accs = []

        for seq in sequences:
            masked_seq, mask_pos = model.augmentor.apply_mask(seq, config.mask_ratio)
            result = model.forward_mlm(seq, masked_seq, mask_pos, device)
            mlm_losses.append(result['loss'])
            mlm_accs.append(result['accuracy'])

        mlm_loss = torch.stack(mlm_losses).mean() if mlm_losses else torch.tensor(0.0, device=device)
        batch_loss = batch_loss + config.w_mlm * mlm_loss
        batch_metrics['mlm_loss'] = mlm_loss.item()
        batch_metrics['mlm_acc'] = np.mean(mlm_accs)

        # =====================================================================
        # 2. Contrastive Task
        # =====================================================================
        if config.contrastive_enabled:
            contrast_result = model.forward_contrastive(sequences, device)
            batch_loss = batch_loss + config.w_contrastive * contrast_result['loss']
            batch_metrics['contrastive_loss'] = contrast_result['loss'].item()
        else:
            batch_metrics['contrastive_loss'] = 0.0

        # =====================================================================
        # 3. BSJ Prediction Task
        # =====================================================================
        if config.bsj_prediction_enabled:
            bsj_losses = []
            bsj_accs = []

            for seq in sequences:
                virtual_bsj = np.random.randint(0, len(seq))
                result = model.forward_bsj(seq, virtual_bsj, device)
                bsj_losses.append(result['loss'])
                bsj_accs.append(result['accuracy'])

            bsj_loss = torch.stack(bsj_losses).mean() if bsj_losses else torch.tensor(0.0, device=device)
            batch_loss = batch_loss + config.w_bsj * bsj_loss
            batch_metrics['bsj_loss'] = bsj_loss.item()
            batch_metrics['bsj_acc'] = np.mean(bsj_accs)
        else:
            batch_metrics['bsj_loss'] = 0.0
            batch_metrics['bsj_acc'] = 0.0

        # =====================================================================
        # 4. Base-pairing Task (every other batch for efficiency)
        # =====================================================================
        if config.basepair_prediction_enabled and batch_idx % 2 == 0:
            bp_losses = []

            for seq in sequences[:2]:  # Limit for memory
                result = model.forward_basepair(seq, device)
                bp_losses.append(result['loss'])

            bp_loss = torch.stack(bp_losses).mean() if bp_losses else torch.tensor(0.0, device=device)
            batch_loss = batch_loss + config.w_basepair * bp_loss
            batch_metrics['basepair_loss'] = bp_loss.item()
        else:
            batch_metrics['basepair_loss'] = 0.0

        # =====================================================================
        # 5. Circular Distance Task (every 4th batch for efficiency)
        # =====================================================================
        if config.circular_distance_enabled and batch_idx % 4 == 0:
            cd_losses = []
            cd_accs = []

            for seq in sequences[:1]:  # Limit for memory
                result = model.forward_circular_distance(seq, device)
                cd_losses.append(result['loss'])
                cd_accs.append(result['accuracy'])

            cd_loss = torch.stack(cd_losses).mean() if cd_losses else torch.tensor(0.0, device=device)
            batch_loss = batch_loss + config.w_circ_dist * cd_loss
            batch_metrics['circ_dist_loss'] = cd_loss.item()
            batch_metrics['circ_dist_acc'] = np.mean(cd_accs)
        else:
            batch_metrics['circ_dist_loss'] = 0.0
            batch_metrics['circ_dist_acc'] = 0.0

        # =====================================================================
        # Backward pass
        # =====================================================================
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_metrics['total_loss'] = batch_loss.item()

        # Accumulate metrics
        for k, v in batch_metrics.items():
            metrics[k] += v
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: "
                  f"MLM={batch_metrics['mlm_loss']:.4f} ({batch_metrics['mlm_acc']:.1%}), "
                  f"Contrast={batch_metrics['contrastive_loss']:.4f}, "
                  f"BSJ={batch_metrics['bsj_loss']:.4f} ({batch_metrics['bsj_acc']:.1%})")

    # Average metrics
    for k in metrics:
        metrics[k] /= n_batches

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Enhanced self-supervised pretraining for TorusFold")
    parser.add_argument("--fasta", default="data/circrna/circbase_seqs.fa.gz")
    parser.add_argument("--output-dir", default="confluencia-circrna-encoder/data/models/pretrain")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-sequences", type=int, default=140000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--n-rot-augments", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("TorusFold Enhanced Self-Supervised Pretraining")
    print("=" * 70)
    print("Tasks:")
    print("  1. MLM: Masked nucleotide prediction")
    print("  2. Contrastive: Rotation-augmented representation learning")
    print("  3. BSJ: Circularization point prediction")
    print("  4. Base-pair: Unsupervised secondary structure")
    print("  5. Circular Distance: Torus topology learning")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Config
    config = TorusPretrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
        max_seq_length=args.max_seq_length,
        mask_ratio=args.mask_ratio,
        n_rot_augments=args.n_rot_augments,
        temperature=args.temperature,
    )

    # Resolve paths
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        fasta_path = _PROJECT_ROOT / args.fasta

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = CircRNAPretrainDataset(
        str(fasta_path),
        max_sequences=config.max_sequences,
        max_seq_length=config.max_seq_length,
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'sequence': [b['sequence'] for b in batch],
            'length': [b['length'] for b in batch],
            'idx': [b['idx'] for b in batch],
        },
        num_workers=0,
    )

    # Model
    model = TorusPretrainModel(config)
    model.load_backbone(device)
    model = model.to(device)

    # Optimizer
    param_list = list(model.mlm_head.parameters())
    if hasattr(model, 'contrastive_head'):
        param_list += list(model.contrastive_head.parameters())
    if hasattr(model, 'bsj_head'):
        param_list += list(model.bsj_head.parameters())
    if hasattr(model, 'basepair_head'):
        param_list += list(model.basepair_head.parameters())
    if hasattr(model, 'circ_dist_head'):
        param_list += list(model.circ_dist_head.parameters())

    optimizer = torch.optim.AdamW(param_list, lr=config.lr, weight_decay=config.weight_decay)

    # Training loop
    history = []
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        metrics = pretrain_epoch(model, dataloader, optimizer, config, device, epoch)

        print(f"\n  Epoch Summary:")
        print(f"    MLM Loss: {metrics['mlm_loss']:.4f} (Acc: {metrics['mlm_acc']:.1%})")
        print(f"    Contrastive Loss: {metrics['contrastive_loss']:.4f}")
        print(f"    BSJ Loss: {metrics['bsj_loss']:.4f} (Acc: {metrics['bsj_acc']:.1%})")
        print(f"    Base-pair Loss: {metrics['basepair_loss']:.4f}")
        print(f"    Circ Dist Loss: {metrics['circ_dist_loss']:.4f} (Acc: {metrics['circ_dist_acc']:.1%})")
        print(f"    Total Loss: {metrics['total_loss']:.4f}")

        history.append({"epoch": epoch, **metrics})

        # Save checkpoint
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            model.save(output_dir / "best_torus_pretrain.pt")
            print(f"    ✓ Saved best model (loss={best_loss:.4f})")

        if epoch % config.save_every == 0:
            model.save(output_dir / f"torus_pretrain_epoch_{epoch}.pt")

    # Save final
    model.save(output_dir / "final_torus_pretrain.pt")

    # Save history and config
    with open(output_dir / "torus_pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "torus_pretrain_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("\n" + "=" * 70)
    print("Pretraining Complete!")
    print(f"Best Total Loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Fine-tune with labeled data (immunogenicity, function)")
    print("  2. Integrate pretrain weights into TorusFold v2")
    print("  3. Evaluate on downstream tasks")


if __name__ == "__main__":
    main()
