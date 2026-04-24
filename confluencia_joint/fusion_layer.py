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

    clinical: float = 0.40   # Clinical/drug efficacy score
    binding: float = 0.35    # MHC-epitope binding score
    kinetics: float = 0.25   # PK/kinetics score

    def __post_init__(self):
        total = self.clinical + self.binding + self.kinetics
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
    ) -> np.ndarray:
        """Fuse three modality outputs into a single feature vector.

        Parameters
        ----------
        clinical : dict or ndarray
            Drug clinical outputs (efficacy, binding, immune activation, safety).
        binding : dict or ndarray
            MHC-epitope binding outputs (efficacy, uncertainty).
        kinetics : dict or ndarray
            PK/kinetics outputs (Cmax, tmax, half_life, AUC, therapeutic index).

        Returns
        -------
        fused : ndarray
            Fused feature vector. Shape depends on strategy.
        """
        if self.strategy == FusionStrategy.WEIGHTED_CONCAT:
            return self._weighted_concat(clinical, binding, kinetics)
        elif self.strategy == FusionStrategy.BILINEAR_CROSS:
            return self._bilinear_cross(clinical, binding, kinetics)
        elif self.strategy == FusionStrategy.ATTENTION_GATING:
            return self._attention_gating(clinical, binding, kinetics)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def fuse_with_weights(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
        weights: FusionWeights,
    ) -> np.ndarray:
        """Fuse with custom weights (overrides self.weights for this call)."""
        original = self.weights
        self.weights = weights
        result = self.fuse(clinical, binding, kinetics)
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
    ) -> np.ndarray:
        """Weighted concatenation — no trainable parameters.

        Each modality is scaled by its weight, then concatenated.
        NaN values are replaced with 0.
        """
        w = self.weights
        c = self._to_array(clinical) * w.clinical
        b = self._to_array(binding) * w.binding
        k = self._to_array(kinetics) * w.kinetics
        return np.concatenate([c, b, k])

    def _bilinear_cross(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
    ) -> np.ndarray:
        """Bilinear cross-interaction — reserved for future labeled data.

        Projects the outer product of clinical×binding through a learned
        matrix W, then concatenates with kinetics.
        This strategy requires training on labeled composite outcomes.
        Currently returns weighted concat as fallback.
        """
        # TODO(when labeled composite data available):
        #   W = np.random.randn(len(binding) * len(clinical), len(kinetics) * 2)
        #   x_cb = np.outer(clinical, binding).flatten()
        #   return np.concatenate([x_cb @ W, kinetics])
        return self._weighted_concat(clinical, binding, kinetics)

    def _attention_gating(
        self,
        clinical: Dict[str, float] | np.ndarray,
        binding: Dict[str, float] | np.ndarray,
        kinetics: Dict[str, float] | np.ndarray,
    ) -> np.ndarray:
        """Soft attention gating — reserved for future labeled data.

        Uses a learned attention vector to produce a weighted sum of the
        three modality representations.
        Currently returns weighted concat as fallback.
        """
        # TODO(when labeled composite data available):
        #   logits = [w_c * clinical.mean(), w_b * binding.mean(), w_k * kinetics.mean()]
        #   alpha = softmax(logits)
        #   return alpha[0]*clinical + alpha[1]*binding + alpha[2]*kinetics
        return self._weighted_concat(clinical, binding, kinetics)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(x: Dict[str, float] | np.ndarray) -> np.ndarray:
        """Convert dict or ndarray to ndarray with NaN→0."""
        if isinstance(x, dict):
            arr = np.array(list(x.values()), dtype=np.float64)
        elif isinstance(x, np.ndarray):
            arr = x.astype(np.float64)
        else:
            arr = np.asarray(x, dtype=np.float64)
        arr = np.where(np.isnan(arr), 0.0, arr)
        return arr