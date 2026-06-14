"""
config.py — Configuration for CircRNA sequence encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# 8 keys used in _score_circrna composite calculation
COMPOSITE_KEYS = [
    "immunotherapy_score", "tumor_killing_index", "overall_immunogenicity",
    "immune_cycle_score", "tme_score", "therapeutic_window",
    "tide_score", "ips",
]

# 5 keys used for reporting only
REPORT_KEYS = [
    "rig_i_score", "tlr_score", "pkr_score",
    "trained_model_risk", "predicted_response",
]

# All 13 keys = COMPOSITE_KEYS + REPORT_KEYS
ALL_KEYS = COMPOSITE_KEYS + REPORT_KEYS

# Gene expression columns the encoder accepts
DEFAULT_GENE_COLS = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"]

# Response class mapping
RESPONSE_CLASSES = ["likely_responder", "intermediate", "likely_non_responder"]

# RNA-FM model variants
RNA_FM_MODELS = {
    "RNA-FM": {
        "name": "facebook/esm2_t30_150M_UR50D",  # RNA-FM uses ESM-2 architecture
        "modelscope": "AI-ModelScope/esm2_t30_150M_UR50D",
        "embed_dim": 640,
        "layers": 30,
    },
    "RNA-FM-small": {
        "name": "facebook/esm2_t12_35M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t12_35M_UR50D",
        "embed_dim": 480,
        "layers": 12,
    },
    "RNA-FM-tiny": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t6_8M_UR50D",
        "embed_dim": 320,
        "layers": 6,
    },
}


@dataclass
class CircRNAEncoderConfig:
    """Configuration for CircRNASequenceEncoder.

    Attributes
    ----------
    rna_fm_model : str
        RNA-FM model variant key (RNA-FM, RNA-FM-small, RNA-FM-tiny).
    freeze_pretrained : bool
        Whether to freeze the pretrained RNA-FM backbone.
    max_seq_len : int
        Maximum sequence length for tokenization (longer sequences use
        sliding window + mean pooling).
    gene_cols : list[str]
        Gene expression column names accepted by the encoder.
    hidden_dims : list[int]
        MLP trunk hidden layer dimensions.
    dropout : float
        Dropout rate for MLP trunk.
    composite_keys : list[str]
        Sub-score keys used in _score_circrna composite calculation.
    report_keys : list[str]
        Sub-score keys used for reporting only.
    """

    # RNA-FM encoder
    rna_fm_model: str = "RNA-FM"
    freeze_pretrained: bool = True
    max_seq_len: int = 1024
    sliding_window: int = 512
    sliding_stride: int = 256

    # Gene expression input
    gene_cols: List[str] = field(default_factory=lambda: list(DEFAULT_GENE_COLS))
    gene_proj_dim: int = 32

    # MLP head
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2

    # Multi-task outputs
    composite_keys: List[str] = field(default_factory=lambda: list(COMPOSITE_KEYS))
    report_keys: List[str] = field(default_factory=lambda: list(REPORT_KEYS))

    # Training
    distill_weight: float = 1.0
    report_weight: float = 0.3
    survival_weight: float = 0.5

    @property
    def embed_dim(self) -> int:
        """RNA-FM embedding dimension."""
        return RNA_FM_MODELS[self.rna_fm_model]["embed_dim"]

    @property
    def n_response_classes(self) -> int:
        """Number of predicted_response classes."""
        return len(RESPONSE_CLASSES)

    @property
    def n_composite(self) -> int:
        """Number of composite sub-scores."""
        return len(self.composite_keys)

    @property
    def n_report_sigmoid(self) -> int:
        """Number of sigmoid report scores (excludes predicted_response)."""
        return len(self.report_keys) - 1  # predicted_response is softmax
