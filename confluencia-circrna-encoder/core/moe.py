"""
moe.py — Mixture of Experts model for circRNA prediction.

Adapted from drug 2.0's moe.py for circRNA context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class MOEConfig:
    """Configuration for Mixture of Experts."""

    # Expert models
    n_experts: int = 5

    # Gating network
    gating_hidden_dim: int = 32

    # Training
    learning_rate: float = 0.01
    n_epochs: int = 100
    batch_size: int = 32


class ExpertModel:
    """Individual expert model for circRNA prediction."""

    def __init__(self, expert_type: str):
        self.expert_type = expert_type
        self.weights = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit expert model."""
        # Simple linear model for each expert
        n_features = X.shape[1]

        # Ridge regression
        self.weights = np.linalg.lstsq(
            X.T @ X + 0.1 * np.eye(n_features),
            X.T @ y,
            rcond=None
        )[0]

        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using expert."""
        if not self.fitted:
            return np.zeros(X.shape[0])

        return X @ self.weights


class GatingNetwork:
    """Gating network for expert selection."""

    def __init__(self, n_experts: int, input_dim: int, hidden_dim: int = 32):
        self.n_experts = n_experts

        # Simple weights for gating
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_experts) * 0.1
        self.b2 = np.zeros(n_experts)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass to get expert weights."""
        # Hidden layer
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU

        # Output
        logits = h @ self.W2 + self.b2

        # Softmax
        weights = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return weights


class CircRNAMOE:
    """
    Mixture of Experts for circRNA prediction.

    Experts:
    - Expert 1: GC-based immunogenicity
    - Expert 2: Entropy-based stability
    - Expert 3: Motif-based TLR activation
    - Expert 4: Length-based persistence
    - Expert 5: Combined features
    """

    EXPERT_TYPES = [
        'gc_expert',
        'entropy_expert',
        'motif_expert',
        'length_expert',
        'combined_expert',
    ]

    def __init__(self, config: Optional[MOEConfig] = None):
        self.config = config or MOEConfig()

        # Initialize experts
        self.experts = [ExpertModel(t) for t in self.EXPERT_TYPES[:self.config.n_experts]]

        # Initialize gating
        self.gating = None
        self.fitted = False

    def extract_features(self, sequences: List[str]) -> np.ndarray:
        """Extract features for MOE."""
        features = []

        for seq in sequences:
            seq = seq.upper().replace('T', 'U')
            length = len(seq)

            # GC content
            gc = sum(1 for c in seq if c in 'GC') / max(length, 1)

            # AU content
            au = sum(1 for c in seq if c in 'AU') / max(length, 1)

            # Entropy
            counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
            for c in seq:
                if c in counts:
                    counts[c] += 1

            probs = [counts[n] / max(length, 1) for n in 'AUGC']
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

            # GU motifs
            gu = sum(1 for i in range(len(seq)-1) if seq[i:i+2] in ['GU', 'UG']) / max(length-1, 1)

            # Length normalized
            length_norm = min(length / 500, 1.0)

            # Repeat content
            max_repeat = 0
            for nuc in 'AUGC':
                count = 0
                max_c = 0
                for c in seq:
                    if c == nuc:
                        count += 1
                        max_c = max(max_c, count)
                    else:
                        count = 0
                max_repeat = max(max_repeat, max_c)
            repeat_ratio = max_repeat / max(length, 1)

            features.append([gc, au, entropy, gu, length_norm, repeat_ratio])

        return np.array(features)

    def fit(self, sequences: List[str], labels: np.ndarray):
        """
        Fit MOE model.

        Args:
            sequences: List of circRNA sequences
            labels: Target labels
        """
        X = self.extract_features(sequences)

        # Initialize gating
        self.gating = GatingNetwork(
            self.config.n_experts,
            X.shape[1],
            self.config.gating_hidden_dim
        )

        # Fit each expert
        for expert in self.experts:
            expert.fit(X, labels)

        # Simple gating training
        for epoch in range(self.config.n_epochs):
            # Get expert predictions
            expert_preds = np.array([e.predict(X) for e in self.experts])

            # Get gating weights
            gating_weights = self.gating.forward(X)

            # Combined prediction
            combined_pred = np.sum(expert_preds * gating_weights.T, axis=0)

            # Update gating (gradient descent)
            error = labels - combined_pred
            # Simple gradient update
            self.gating.W2 += 0.01 * error.mean() * np.sign(self.gating.W2)

        self.fitted = True

    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Predict using MOE.

        Args:
            sequences: List of circRNA sequences

        Returns:
            Predictions
        """
        if not self.fitted:
            # Use simple features if not fitted
            return self._simple_predict(sequences)

        X = self.extract_features(sequences)

        # Get expert predictions
        expert_preds = np.array([e.predict(X) for e in self.experts])

        # Get gating weights
        gating_weights = self.gating.forward(X)

        # Combined prediction
        combined_pred = np.sum(expert_preds * gating_weights.T, axis=0)

        return np.clip(combined_pred, 0, 1)

    def _simple_predict(self, sequences: List[str]) -> np.ndarray:
        """Simple prediction without training."""
        from .innate_immune import quick_predict

        predictions = []

        for seq in sequences:
            immune = quick_predict(seq)
            predictions.append(immune['overall_score'])

        return np.array(predictions)

    def predict_with_weights(self, sequences: List[str]) -> Dict:
        """Predict with expert weight breakdown."""
        X = self.extract_features(sequences)

        if not self.fitted:
            return {
                'predictions': self._simple_predict(sequences),
                'expert_weights': None,
            }

        # Get expert predictions
        expert_preds = np.array([e.predict(X) for e in self.experts])

        # Get gating weights
        gating_weights = self.gating.forward(X)

        # Combined
        combined = np.sum(expert_preds * gating_weights.T, axis=0)

        return {
            'predictions': combined,
            'expert_weights': gating_weights,
            'expert_predictions': expert_preds,
            'expert_types': [e.expert_type for e in self.experts],
        }


def train_moe_model(sequences: List[str], labels: np.ndarray) -> CircRNAMOE:
    """Train MOE model."""
    model = CircRNAMOE()
    model.fit(sequences, labels)
    return model


def quick_moe_predict(sequences: List[str]) -> np.ndarray:
    """Quick MOE prediction."""
    model = CircRNAMOE()
    return model.predict(sequences)