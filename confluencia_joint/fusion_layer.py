"""
Multi-modal fusion strategies for joint Drug-Epitope-PK evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal

import numpy as np


class FusionStrategy(Enum):
    """Available multi-modal fusion strategies."""

    # Weighted concatenation — no training needed, default choice
    WEIGHTED_CONCAT = "weighted_concat"

    # Bilinear cross-attention — reserved for future labeled data
    BILINEAR_CROSS = "bilinear_cross"

    # Attention gating — reserved for future labeled data
    ATTENTION_GATING = "attention_gating"


@dataclass
class FusionWeights:
    """Per-modality weights used by weighted-concat fusion."""

    clinical: float = 0.30       # Clinical/drug efficacy score
    binding: float = 0.20        # MHC-epitope binding score
    kinetics: float = 0.15       # PK/kinetics score
    gene_signature: float = 0.15 # Five-target gene signature score
    circrna: float = 0.20        # circRNA multi-omics score

    def __post_init__(self):
        total = (self.clinical + self.binding + self.kinetics +
                 self.gene_signature + self.circrna)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"FusionWeights must sum to 1.0, got {total}"
            )


@dataclass
class JointFusionLayer:
    """Multi-modal fusion layer.

    Combines Drug clinical outputs, Epitope binding outputs, and PK kinetics
    outputs into a unified representation.

    Parameters
    ----------
    strategy : FusionStrategy
        Which fusion strategy to use. Default: WEIGHTED_CONCAT.
    weights : FusionWeights
        Per-modality weights (only used by WEIGHTED_CONCAT).
        Defaults to 0.40 / 0.35 / 0.25.

    Usage
    -----
    >>> layer = JointFusionLayer()  # default weighted concat
    >>> fused = layer.fuse(clinical_vec, binding_vec, kinetics_vec)
    """

    strategy: FusionStrategy = FusionStrategy.WEIGHTED_CONCAT
    weights: FusionWeights = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = FusionWeights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        gene_signature: Dict[str, float] | np.ndarray | None = None,
        circrna: Dict[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Fuse five modality outputs into a single feature vector.

        Parameters
        ----------
        clinical : dict or ndarray
            Drug clinical outputs.
        binding : dict or ndarray
            MHC-epitope binding outputs.
        kinetics : dict or ndarray
            PK/kinetics outputs.
        gene_signature : dict or ndarray or None
            Five-target gene signature outputs.
        circrna : dict or ndarray or None
            circRNA multi-omics outputs.

        Returns
        -------
        fused : ndarray
            Fused feature vector. Shape depends on strategy.
        """
        if self.strategy == FusionStrategy.WEIGHTED_CONCAT:
            return self._weighted_concat(clinical, binding, kinetics, gene_signature, circrna)
        elif self.strategy == FusionStrategy.BILINEAR_CROSS:
            return self._bilinear_cross(clinical, binding, kinetics, gene_signature, circrna)
        elif self.strategy == FusionStrategy.ATTENTION_GATING:
            return self._attention_gating(clinical, binding, kinetics, gene_signature, circrna)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def fuse_with_weights(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        gene_signature: Dict[str, float] | np.ndarray | None = None,
        circrna: Dict[str, float] | np.ndarray | None = None,
        weights: FusionWeights = None,
    ) -> np.ndarray:
        """Fuse with custom weights (overrides self.weights for this call)."""
        original = self.weights
        self.weights = weights or self.weights
        result = self.fuse(clinical, binding, kinetics, gene_signature, circrna)
        self.weights = original
        return result

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _weighted_concat(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        gene_signature: Dict[str, float] | np.ndarray | None = None,
        circrna: Dict[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Weighted concatenation — no trainable parameters.

        Each modality is scaled by its weight, then concatenated.
        NaN values are replaced with 0.
        """
        w = self.weights
        c = self._to_array(clinical) * w.clinical
        b = self._to_array(binding) * w.binding
        k = self._to_array(kinetics) * w.kinetics
        parts = [c, b, k]
        if gene_signature is not None:
            g = self._to_array(gene_signature) * w.gene_signature
            parts.append(g)
        if circrna is not None:
            r = self._to_array(circrna) * w.circrna
            parts.append(r)
        return np.concatenate(parts)

    def _bilinear_cross(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        gene_signature: Dict[str, float] | np.ndarray | None = None,
        circrna: Dict[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Bilinear cross-interaction — reserved for future labeled data.

        Requires training on labeled (composite_score, recommendation) pairs.
        Currently returns weighted concat as placeholder.
        """
        return self._weighted_concat(clinical, binding, kinetics, gene_signature, circrna)

    def _attention_gating(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        gene_signature: Dict[str, float] | np.ndarray | None = None,
        circrna: Dict[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Soft attention gating — reserved for future labeled data.

        Requires training on (input, composite_score) pairs.
        Currently returns weighted concat as placeholder.
        """
        return self._weighted_concat(clinical, binding, kinetics, gene_signature, circrna)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(x: Dict[str, float] | np.ndarray) -> np.ndarray:
        """Convert dict or ndarray to ndarray with NaN→0 and non-numeric filtering."""
        if isinstance(x, dict):
            vals = []
            for v in x.values():
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue  # skip non-numeric fields like predicted_response
            arr = np.array(vals, dtype=np.float64)
        elif isinstance(x, np.ndarray):
            arr = x.astype(np.float64)
        else:
            arr = np.asarray(x, dtype=np.float64)
        arr = np.where(np.isnan(arr), 0.0, arr)
        return arr