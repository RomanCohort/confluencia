from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

# Import canonical AA constants from shared
from confluencia_shared.features import AA_ORDER, AA_TO_IDX


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def _scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor (seq_len, d_k)
        k: Key tensor (seq_len, d_k)
        v: Value tensor (seq_len, d_v)
        mask: Optional attention mask (seq_len, seq_len)

    Returns:
        (attended_output, attention_weights)
    """
    d_k = float(q.shape[-1])
    # Scaled dot-product
    scores = (q @ k.T) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Attention weights
    attn_weights = _softmax(scores, axis=-1)

    # Attend to values
    output = attn_weights @ v

    return output, attn_weights


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.shape[0] == 0:
        return x.copy()
    window = max(1, int(window))
    if window == 1:
        return x.copy()

    out = np.zeros_like(x, dtype=np.float32)
    csum = np.cumsum(np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), x]), axis=0)
    for i in range(x.shape[0]):
        lo = max(0, i - window + 1)
        hi = i + 1
        out[i] = (csum[hi] - csum[lo]) / float(hi - lo)
    return out


@dataclass(frozen=True)
class Mamba3Config:
    """Configuration for Mamba3LiteEncoder.

    Attributes:
        d_model: Dimension of the hidden state.
        local_window: Window size for local pooling (residue-level).
        meso_window: Window size for meso pooling (secondary structure-level).
        global_window: Window size for global pooling (domain-level).
        seed: Random seed for weight initialization.
        decay_fast: Base decay rate for fast time constant (residue-level).
        decay_mid: Base decay rate for mid time constant (secondary structure-level).
        decay_slow: Base decay rate for slow time constant (domain-level).
        gate_scale_fast: Scale factor for fast gate modulation.
        gate_scale_mid: Scale factor for mid gate modulation.
        gate_scale_slow: Scale factor for slow gate modulation.
    """
    d_model: int = 24      # Latent dimension; balances expressiveness (64+) vs small-sample overfitting risk.
                          # 24 keeps parameter count ~2k for the encoder, suitable for N<300 training sets.
    local_window: int = 3   # ~1 turn of alpha-helix (3.6 residues/turn); captures local backbone interactions.
    meso_window: int = 11   # ~1 secondary structure element (alpha-helix ~10-12 residues); bridges local→domain scale.
    global_window: int = 33 # ~1 small protein domain; captures long-range sequence context.
    seed: int = 42
    # Decay rates (retention per recurrence step, 0→forget immediately, 1→perfect memory).
    # These span ~1.5 to ~30 residue half-lives: fast decays quickly (residue-level),
    # slow retains across the full sequence (domain-level).
    # Half-life in residues ≈ -1/ln(decay): fast≈2.5, mid≈9.5, slow≈32.
    decay_fast: float = 0.72
    decay_mid: float = 0.90
    decay_slow: float = 0.97
    # Gate modulation scales: how much the learned sigmoid gate can perturb the base decay.
    # Kept small so the recurrence stays stable; fast has the largest range because
    # residue-level features benefit most from position-dependent adaptation.
    gate_scale_fast: float = 0.20  # effective range: 0.52–0.92
    gate_scale_mid: float = 0.08   # effective range: 0.82–0.98
    gate_scale_slow: float = 0.02  # effective range: 0.95–0.99


class Mamba3LiteEncoder:
    """A lightweight, Mamba-inspired encoder for long peptide sequences.

    This implementation uses selective state-space recurrences with three time constants
    and neighborhood pooling to keep compute linear in sequence length.

    The encoder is deterministic and reproducible:
    - Weights are initialized from a fixed seed
    - State can be saved/loaded via get_state()/set_state()
    - Supports pickle serialization via __getstate__/__setstate__

    Example:
        encoder = Mamba3LiteEncoder(Mamba3Config(seed=42))
        features = encoder.encode("SIINFEKL")

        # Save state for reproducibility
        state = encoder.get_state()

        # Restore in another process
        encoder2 = Mamba3LiteEncoder()
        encoder2.set_state(state)
    """

    def __init__(self, config: Mamba3Config | None = None) -> None:
        self.config = config or Mamba3Config()
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights from config seed. Called once on construction."""
        rng = np.random.default_rng(self.config.seed)

        n_token = len(AA_ORDER) + 1
        d = int(self.config.d_model)

        self.embedding = rng.normal(0.0, 0.2, size=(n_token, d)).astype(np.float32)
        # Init std=0.2 keeps initial token embeddings small relative to decay dynamics,
        # preventing gradient explosion in early training (He-init would use ~0.2 for d=24).
        self.gate_w = rng.normal(0.0, 0.15, size=(d, 3)).astype(np.float32)
        self.gate_b = rng.normal(0.0, 0.05, size=(3,)).astype(np.float32)
        # Gate weights use smaller init (0.15, 0.05) so initial gates are near-sigmoid(0)=0.5,
        # meaning decay starts close to the configured base rates before learning adapts them.

        # Self-attention projection weights (d_model -> d_attn)
        # d_attn is a reduced dimension to keep parameter count manageable for small samples
        d_attn = max(8, d // 2)  # Half the model dimension for QKV projections
        self.query_w = rng.normal(0, 0.1, size=(d, d_attn)).astype(np.float32)
        self.key_w = rng.normal(0, 0.1, size=(d, d_attn)).astype(np.float32)
        self.value_w = rng.normal(0, 0.1, size=(d, d_attn)).astype(np.float32)
        self.attn_out_w = rng.normal(0, 0.1, size=(d_attn, d)).astype(np.float32)
        self.attn_out_b = np.zeros((d,), dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get encoder state for serialization.

        Returns:
            Dict containing config and weight arrays.
        """
        return {
            "config": {
                "d_model": self.config.d_model,
                "local_window": self.config.local_window,
                "meso_window": self.config.meso_window,
                "global_window": self.config.global_window,
                "seed": self.config.seed,
                "decay_fast": self.config.decay_fast,
                "decay_mid": self.config.decay_mid,
                "decay_slow": self.config.decay_slow,
                "gate_scale_fast": self.config.gate_scale_fast,
                "gate_scale_mid": self.config.gate_scale_mid,
                "gate_scale_slow": self.config.gate_scale_slow,
            },
            "embedding": self.embedding.copy(),
            "gate_w": self.gate_w.copy(),
            "gate_b": self.gate_b.copy(),
            "query_w": self.query_w.copy(),
            "key_w": self.key_w.copy(),
            "value_w": self.value_w.copy(),
            "attn_out_w": self.attn_out_w.copy(),
            "attn_out_b": self.attn_out_b.copy(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore encoder state from serialization.

        Args:
            state: Dict from get_state().
        """
        # Reconstruct config
        cfg = state["config"]
        self.config = Mamba3Config(**cfg)
        # Restore weights
        self.embedding = state["embedding"].copy()
        self.gate_w = state["gate_w"].copy()
        self.gate_b = state["gate_b"].copy()
        self.query_w = state.get("query_w", np.zeros((self.config.d_model, 8), dtype=np.float32))
        self.key_w = state.get("key_w", np.zeros((self.config.d_model, 8), dtype=np.float32))
        self.value_w = state.get("value_w", np.zeros((self.config.d_model, 8), dtype=np.float32))
        self.attn_out_w = state.get("attn_out_w", np.zeros((8, self.config.d_model), dtype=np.float32))
        self.attn_out_b = state.get("attn_out_b", np.zeros((self.config.d_model,), dtype=np.float32))

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return self.get_state()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.set_state(state)

    def _tokenize(self, seq: str) -> np.ndarray:
        seq = str(seq or "").strip().upper().replace(" ", "")
        ids = [AA_TO_IDX.get(ch, len(AA_ORDER)) for ch in seq]
        return np.asarray(ids, dtype=np.int64)

    def encode(self, seq: str) -> Dict[str, np.ndarray]:
        ids = self._tokenize(seq)
        d = int(self.config.d_model)
        if ids.size == 0:
            z = np.zeros((d,), dtype=np.float32)
            return {
                "summary": np.concatenate([z, z, z, z], axis=0),
                "local_pool": z,
                "meso_pool": z,
                "global_pool": z,
                "token_hidden": np.zeros((0, d), dtype=np.float32),
                "attn_weights": np.zeros((0, 0), dtype=np.float32),
            }

        x = self.embedding[ids]
        gates = _sigmoid(x @ self.gate_w + self.gate_b)

        s_fast = np.zeros((d,), dtype=np.float32)
        s_mid = np.zeros((d,), dtype=np.float32)
        s_slow = np.zeros((d,), dtype=np.float32)
        hidden = np.zeros_like(x, dtype=np.float32)

        # Use configurable decay constants
        decay_fast = self.config.decay_fast
        decay_mid = self.config.decay_mid
        decay_slow = self.config.decay_slow
        gs_fast = self.config.gate_scale_fast
        gs_mid = self.config.gate_scale_mid
        gs_slow = self.config.gate_scale_slow

        for i in range(x.shape[0]):
            xi = x[i]
            g = gates[i]
            a_fast = decay_fast + gs_fast * float(g[0])
            a_mid = decay_mid + gs_mid * float(g[1])
            a_slow = decay_slow + gs_slow * float(g[2])

            s_fast = a_fast * s_fast + (1.0 - a_fast) * xi
            s_mid = a_mid * s_mid + (1.0 - a_mid) * xi
            s_slow = a_slow * s_slow + (1.0 - a_slow) * xi
            hidden[i] = 0.5 * s_fast + 0.3 * s_mid + 0.2 * s_slow

        # Apply self-attention on hidden states
        # Project to Q, K, V
        q = hidden @ self.query_w
        k = hidden @ self.key_w
        v = hidden @ self.value_w

        # Compute attention with causal mask
        seq_len = x.shape[0]
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        attn_output, attn_weights = _scaled_dot_product_attention(q, k, v, mask=~causal_mask)

        # Project back and add residual connection
        attn_output = attn_output @ self.attn_out_w + self.attn_out_b
        hidden = hidden + 0.1 * attn_output  # Small residual connection (10% weight)

        local_hidden = _rolling_mean(hidden, self.config.local_window)
        meso_hidden = _rolling_mean(hidden, self.config.meso_window)
        global_hidden = _rolling_mean(hidden, self.config.global_window)

        local_pool = local_hidden.mean(axis=0).astype(np.float32)
        meso_pool = meso_hidden.mean(axis=0).astype(np.float32)
        global_pool = global_hidden.mean(axis=0).astype(np.float32)

        summary = np.concatenate(
            [
                hidden.mean(axis=0),
                hidden.max(axis=0),
                hidden[-1],
                0.5 * local_pool + 0.3 * meso_pool + 0.2 * global_pool,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "summary": summary,
            "local_pool": local_pool,
            "meso_pool": meso_pool,
            "global_pool": global_pool,
            "token_hidden": hidden,
            "attn_weights": attn_weights.astype(np.float32),
        }

    def feature_names(self) -> List[str]:
        d = int(self.config.d_model)
        names: List[str] = []
        names += [f"mamba_summary_mean_{i}" for i in range(d)]
        names += [f"mamba_summary_max_{i}" for i in range(d)]
        names += [f"mamba_summary_last_{i}" for i in range(d)]
        names += [f"mamba_summary_mix_{i}" for i in range(d)]
        names += [f"nb_local_{i}" for i in range(d)]
        names += [f"nb_meso_{i}" for i in range(d)]
        names += [f"nb_global_{i}" for i in range(d)]
        return names
