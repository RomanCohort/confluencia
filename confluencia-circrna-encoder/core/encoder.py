"""
CircRNA Sequence Encoder — RNA-FM backbone with distillation heads.

Architecture (mirrors drug module's torch_predictor but for sequences):
- RNA-FM (640M) backbone → 640-dim embedding
- Multi-task heads:
  - composite scores (sigmoid, 8 outputs)
  - report scores (sigmoid, 4 outputs)
  - response class (softmax, 3 classes)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# RNA-FM imports
try:
    import esm
    RNAFM_AVAILABLE = True
except ImportError:
    RNAFM_AVAILABLE = False
    print("[Warning] RNA-FM not available. Install with: pip install fair-esm")


@dataclass
class CircRNAEncoderConfig:
    """Configuration for CircRNA Encoder (mirrors drug config structure)."""

    # Backbone
    backbone_model: str = "RNA-FM"
    backbone_dim: int = 640
    freeze_backbone: bool = True

    # Multi-task outputs (like drug module)
    composite_keys: List[str] = field(default_factory=lambda: [
        "immunotherapy_score",
        "tumor_killing_index",
        "overall_immunogenicity",
        "immune_cycle_score",
        "tme_score",
        "therapeutic_window",
        "tide_score",
        "ips",
    ])

    report_keys: List[str] = field(default_factory=lambda: [
        "rig_i_score",
        "tlr_score",
        "pkr_score",
        "trained_model_risk",
    ])

    response_classes: List[str] = field(default_factory=lambda: [
        "likely_non_responder",
        "intermediate",
        "likely_responder",
    ])

    # Gene expression input (6 genes)
    gene_cols: List[str] = field(default_factory=lambda: [
        "TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"
    ])
    gene_dim: int = 6

    # Head architecture
    hidden_dim: int = 256
    dropout: float = 0.2

    # Training (mirrors drug module)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8

    def to_dict(self) -> Dict:
        return {
            "backbone_model": self.backbone_model,
            "backbone_dim": self.backbone_dim,
            "freeze_backbone": self.freeze_backbone,
            "composite_keys": self.composite_keys,
            "report_keys": self.report_keys,
            "response_classes": self.response_classes,
            "gene_cols": self.gene_cols,
            "gene_dim": self.gene_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
        }


class CompositeHead(nn.Module):
    """Head for 8 composite scores (sigmoid)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReportHead(nn.Module):
    """Head for 4 report scores (sigmoid)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResponseHead(nn.Module):
    """Head for response classification (softmax)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CircRNAEncoder(nn.Module):
    """
    CircRNA Sequence Encoder with RNA-FM backbone.

    Structure mirrors drug module's torch_predictor:
    - Backbone: RNA-FM (frozen) → 640-dim
    - Input concat: sequence + gene_expr → 646-dim
    - Multi-task heads: composite(8), report(4), response(3)
    """

    def __init__(self, config: Optional[CircRNAEncoderConfig] = None):
        super().__init__()
        self.config = config or CircRNAEncoderConfig()

        # Backbone (loaded separately)
        self.backbone = None
        self.alphabet = None
        self.backbone_loaded = False

        # Input dimension: sequence + gene
        input_dim = self.config.backbone_dim + self.config.gene_dim

        # Multi-task heads (like drug module)
        self.composite_head = CompositeHead(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            n_outputs=len(self.config.composite_keys),
            dropout=self.config.dropout,
        )

        self.report_head = ReportHead(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            n_outputs=len(self.config.report_keys),
            dropout=self.config.dropout,
        )

        self.response_head = ResponseHead(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            n_classes=len(self.config.response_classes),
            dropout=self.config.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        for module in [self.composite_head, self.report_head, self.response_head]:
            for layer in module.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def load_backbone(self, model_path: Optional[str] = None, device: str = "cpu"):
        """Load RNA-FM backbone."""
        if self.backbone_loaded:
            return

        if not RNAFM_AVAILABLE:
            raise ImportError("RNA-FM not installed. Run: pip install fair-esm")

        # Load ESM2/RNA-FM model
        if model_path and Path(model_path).exists():
            self.backbone, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        else:
            # Use ESM2 650M (similar size to RNA-FM)
            self.backbone, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone = self.backbone.to(device)
        self.backbone_loaded = True

    def encode_sequences(self, sequences: List[str], device: str = "cpu") -> torch.Tensor:
        """
        Encode circRNA sequences to embeddings.

        Args:
            sequences: RNA sequences (with U)
            device: Device

        Returns:
            Tensor (batch, 640)
        """
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
                repr_layers=[33],  # Use layer 33 for 650M model
                return_contacts=False,
            )
            embeddings = results["representations"][33]

            # Mean pooling
            mask = (batch_tokens != self.alphabet.padding_idx).float().unsqueeze(-1)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled

    def forward(
        self,
        sequences: List[str],
        gene_expr: torch.Tensor,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (mirrors drug module structure).

        Returns:
            Dict with composite, report, response_logits, response_probs
        """
        # Encode sequences
        seq_emb = self.encode_sequences(sequences, device=device)

        # Concat gene expression
        gene_expr = gene_expr.to(device)
        x = torch.cat([seq_emb, gene_expr], dim=-1)

        # Multi-task heads
        composite = self.composite_head(x)
        report = self.report_head(x)
        response_logits = self.response_head(x)
        response_probs = F.softmax(response_logits, dim=-1)

        return {
            "composite": composite,
            "report": report,
            "response_logits": response_logits,
            "response_probs": response_probs,
            "embedding": seq_emb,
        }

    def predict_single(
        self,
        sequence: str,
        gene_expr: Dict[str, float],
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Single sequence prediction."""
        gene_values = [gene_expr.get(g, 0.5) for g in self.config.gene_cols]
        gene_tensor = torch.tensor([gene_values], dtype=torch.float32)

        outputs = self.forward([sequence], gene_tensor, device=device)

        result = {}
        for i, k in enumerate(self.config.composite_keys):
            result[k] = outputs["composite"][0, i].item()

        for i, k in enumerate(self.config.report_keys):
            result[k] = outputs["report"][0, i].item()

        for i, c in enumerate(self.config.response_classes):
            result[f"prob_{c}"] = outputs["response_probs"][0, i].item()

        result["predicted_response"] = self.config.response_classes[
            outputs["response_probs"][0].argmax().item()
        ]

        return result

    def save(self, path: str):
        """Save model weights."""
        state = {
            "config": self.config.to_dict(),
            "composite_head": self.composite_head.state_dict(),
            "report_head": self.report_head.state_dict(),
            "response_head": self.response_head.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str, device: str = "cpu"):
        """Load model weights."""
        state = torch.load(path, map_location=device)

        if "config" in state:
            self.config = CircRNAEncoderConfig(**state["config"])

        self.composite_head.load_state_dict(state["composite_head"])
        self.report_head.load_state_dict(state["report_head"])
        self.response_head.load_state_dict(state["response_head"])