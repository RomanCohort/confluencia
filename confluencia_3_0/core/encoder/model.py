"""
model.py — CircRNA sequence encoder model.

Dual-branch architecture:
  Branch 1: RNA-FM (frozen) → mean-pool → embed_dim
  Branch 2: Gene expression MLP → gene_proj_dim
  Fusion: concat → shared trunk MLP → multi-task heads

Output: 13-key dict compatible with JointScoringEngine._score_circrna().
"""

from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from confluencia_circrna.encoder.config import (
    CircRNAEncoderConfig,
    RNA_FM_MODELS,
    RESPONSE_CLASSES,
    COMPOSITE_KEYS,
    REPORT_KEYS,
)
from confluencia_circrna.encoder.tokenizer import (
    sanitize_rna_sequence,
    sliding_window_encode,
    encode_gene_expression,
)


# ---------------------------------------------------------------------------
# RNA-FM backbone loader (mirrors ESM2Encoder pattern)
# ---------------------------------------------------------------------------

_loaded_backbone = None  # singleton cache


def _load_rna_fm(config: CircRNAEncoderConfig):
    """Load RNA-FM backbone model + tokenizer.

    Tries in order:
    1. HuggingFace local cache
    2. ModelScope mirror
    3. HuggingFace online download

    Returns (model, tokenizer, device).
    """
    global _loaded_backbone
    if _loaded_backbone is not None:
        return _loaded_backbone

    from transformers import AutoModel, AutoTokenizer

    model_info = RNA_FM_MODELS[config.rna_fm_model]
    model_name = model_info["name"]
    modelscope_name = model_info.get("modelscope")

    print(f"[RNA-FM] Loading: {model_name} (embed_dim={model_info['embed_dim']})")

    model = None
    tokenizer = None

    # 1. Try HuggingFace local cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = model_name.replace("/", "--")
    cache_dir = hf_cache / f"models--{model_cache_name}"
    if cache_dir.exists():
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                if (snap / "config.json").exists():
                    try:
                        print(f"[RNA-FM] Using HF cache: {snap}")
                        tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
                        model = AutoModel.from_pretrained(str(snap), local_files_only=True)
                        break
                    except Exception:
                        continue

    # 2. Try ModelScope mirror
    if model is None and modelscope_name:
        try:
            from modelscope import snapshot_download
            print(f"[RNA-FM] Trying ModelScope: {modelscope_name}")
            model_dir = snapshot_download(modelscope_name)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir)
        except ImportError:
            print("[RNA-FM] modelscope not installed, skipping mirror")
        except Exception as e:
            print(f"[RNA-FM] ModelScope failed: {e}")

    # 3. HuggingFace online download
    if model is None:
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if hf_endpoint:
            print(f"[RNA-FM] Using HF mirror: {hf_endpoint}")
        print(f"[RNA-FM] Downloading from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _loaded_backbone = (model, tokenizer, device)
    return _loaded_backbone


# ---------------------------------------------------------------------------
# CircRNASequenceEncoder model
# ---------------------------------------------------------------------------

class CircRNASequenceEncoder(nn.Module):
    """circRNA sequence + gene expression → 13-key scoring dict.

    Architecture
    ------------
    RNA-FM (frozen) → mean-pool → embed_dim
    Gene expression → Linear(gene_cols, gene_proj_dim) → GELU
    concat(embed_dim, gene_proj_dim) → MLP trunk
    → 8 sigmoid heads (composite scores, [0,1])
    → 4 sigmoid heads (report scores, [0,1])
    → 1 softmax head (predicted_response, 3 classes)

    The output dict is directly compatible with
    JointScoringEngine._score_circrna().
    """

    def __init__(self, config: CircRNAEncoderConfig):
        super().__init__()
        self.config = config

        # --- Branch 1: RNA-FM backbone (loaded lazily) ---
        self._backbone_loaded = False
        self.backbone = None
        self.tokenizer = None
        self.backbone_device = None

        # Placeholder for backbone embed dim (set after loading)
        self._embed_dim = config.embed_dim

        # --- Branch 2: Gene expression projection ---
        self.gene_proj = nn.Sequential(
            nn.Linear(len(config.gene_cols), config.gene_proj_dim),
            nn.GELU(),
        )

        # --- Shared MLP trunk ---
        in_dim = self._embed_dim + config.gene_proj_dim
        layers = []
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        trunk_out = config.hidden_dims[-1]

        # --- Multi-task heads ---
        # 8 composite score heads (sigmoid, [0,1])
        self.composite_heads = nn.ModuleList([
            nn.Linear(trunk_out, 1) for _ in config.composite_keys
        ])

        # 4 report sigmoid heads (rig_i, tlr, pkr, trained_model_risk)
        self.report_sigmoid_heads = nn.ModuleList([
            nn.Linear(trunk_out, 1) for _ in range(config.n_report_sigmoid)
        ])

        # 1 response classification head (softmax, 3 classes)
        self.response_head = nn.Linear(trunk_out, config.n_response_classes)

    def _ensure_backbone(self):
        """Lazily load RNA-FM backbone."""
        if self._backbone_loaded:
            return

        backbone, tokenizer, device = _load_rna_fm(self.config)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.backbone_device = device

        if self.config.freeze_pretrained:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self._backbone_loaded = True

    def _encode_sequence(self, sequences: List[str]) -> torch.Tensor:
        """Encode RNA sequences using RNA-FM backbone.

        Parameters
        ----------
        sequences : list[str]
            RNA nucleotide sequences (batch).

        Returns
        -------
        torch.Tensor
            (batch, embed_dim) pooled embeddings.
        """
        self._ensure_backbone()

        # Tokenize
        sanitized = [sanitize_rna_sequence(s) for s in sequences]
        # Handle empty sequences
        sanitized = [s if s else "AUGC" for s in sanitized]

        encoded = self.tokenizer(
            sanitized,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.backbone_device)
        attention_mask = encoded["attention_mask"].to(self.backbone_device)

        # Forward pass
        ctx = torch.no_grad() if self.config.freeze_pretrained else nullcontext()
        with ctx:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, L, embed_dim)

        # Mean pooling (exclude padding)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        seq_lengths = mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = (last_hidden * mask_expanded).sum(dim=1) / seq_lengths  # (B, embed_dim)

        return pooled

    def _encode_sequence_long(self, sequence: str) -> torch.Tensor:
        """Encode a long RNA sequence using sliding window + mean pool.

        For sequences exceeding max_seq_len, splits into overlapping
        windows, encodes each, and mean-pools the embeddings.
        """
        windows = sliding_window_encode(
            sequence,
            window_size=self.config.sliding_window,
            stride=self.config.sliding_stride,
        )
        if len(windows) == 1:
            return self._encode_sequence([windows[0]])

        # Encode all windows
        embeddings = []
        for w in windows:
            emb = self._encode_sequence([w])  # (1, embed_dim)
            embeddings.append(emb)

        # Mean pool across windows
        return torch.stack(embeddings, dim=0).mean(dim=0)  # (1, embed_dim)

    def forward(
        self,
        sequences: List[str],
        gene_expr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: sequences + gene expression → 13-key dict.

        Parameters
        ----------
        sequences : list[str]
            RNA nucleotide sequences (batch_size).
        gene_expr : torch.Tensor
            Gene expression tensor (batch_size, n_genes).

        Returns
        -------
        dict[str, torch.Tensor]
            13-key scoring dict. Composite keys are sigmoid-bounded [0,1].
            'ips' is rescaled to [0,10]. 'predicted_response' is class index.
        """
        # Branch 1: sequence encoding
        seq_emb = self._encode_sequence(sequences)  # (B, embed_dim)

        # Branch 2: gene expression
        gene_emb = self.gene_proj(gene_expr.to(seq_emb.device))  # (B, gene_proj_dim)

        # Fusion
        x = torch.cat([seq_emb, gene_emb], dim=-1)  # (B, embed_dim + gene_proj_dim)
        x = self.trunk(x)  # (B, hidden_dims[-1])

        # --- Multi-task outputs ---
        result = {}

        # Composite scores (8 sigmoid heads)
        for key, head in zip(self.config.composite_keys, self.composite_heads):
            result[key] = torch.sigmoid(head(x)).squeeze(-1)  # (B,)

        # Rescale ips from [0,1] to [0,10]
        result["ips"] = result["ips"] * 10.0

        # Report scores (4 sigmoid heads)
        report_sigmoid_keys = [k for k in self.config.report_keys if k != "predicted_response"]
        for key, head in zip(report_sigmoid_keys, self.report_sigmoid_heads):
            result[key] = torch.sigmoid(head(x)).squeeze(-1)  # (B,)

        # Predicted response (softmax → class index)
        response_logits = self.response_head(x)  # (B, 3)
        result["predicted_response_logits"] = response_logits
        result["predicted_response_idx"] = response_logits.argmax(dim=-1)  # (B,)

        return result

    @torch.no_grad()
    def predict_dict(
        self,
        sequence: str,
        gene_expr: Dict[str, float],
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        """Single sample prediction → 13-key scoring dict.

        Parameters
        ----------
        sequence : str
            circRNA nucleotide sequence.
        gene_expr : dict
            {gene_name: expression_value} mapping.
        device : str, optional
            Override device (default: auto-detect).

        Returns
        -------
        dict[str, float]
            13-key dict compatible with _score_circrna().
            'predicted_response' is a string (likely_responder / intermediate / likely_non_responder).
        """
        self.eval()
        self._ensure_backbone()

        # Gene expression tensor
        gene_arr = encode_gene_expression(gene_expr, self.config.gene_cols)
        gene_tensor = torch.from_numpy(gene_arr).unsqueeze(0)  # (1, n_genes)

        # Forward
        output = self.forward([sequence], gene_tensor)

        # Convert to plain dict
        result = {}
        for key in self.config.composite_keys:
            result[key] = float(output[key].squeeze())

        # Convert ips back to [0,10] range (already rescaled in forward)
        # result["ips"] is already in [0,10]

        for key in self.config.report_keys:
            if key == "predicted_response":
                idx = int(output["predicted_response_idx"].squeeze())
                result[key] = RESPONSE_CLASSES[idx]
            else:
                result[key] = float(output[key].squeeze())

        return result

    def unfreeze_backbone(self, n_layers: Optional[int] = None):
        """Unfreeze RNA-FM backbone for fine-tuning.

        Parameters
        ----------
        n_layers : int, optional
            Number of layers to unfreeze from the end.
            If None, unfreeze all layers.
        """
        self._ensure_backbone()
        self.config.freeze_pretrained = False

        if n_layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # Freeze all first, then unfreeze last n_layers
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Find encoder layers and unfreeze last n
            encoder = getattr(self.backbone, "encoder", None)
            if encoder is not None:
                layers = getattr(encoder, "layer", [])
                for layer in layers[-n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def get_trainable_param_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-3):
        """Get parameter groups with different learning rates.

        Returns two groups: backbone params (lower LR) and head params (higher LR).
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})

        return groups
