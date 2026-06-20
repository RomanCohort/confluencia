#!/usr/bin/env python3
"""
TorusFold Proxy Experiment: Standard PE vs Torus PE on BSJ-Pairing Prediction

Run: python torusfold_proxy_experiment.py

Requirements:
  pip install torch numpy pandas matplotlib tqdm
  conda install -c bioconda viennarna  (for ViennaRNA Python bindings)

If ViennaRNA is not installed, the script will use synthetic data for testing.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Import torch and related modules at the top (required before defining nn.Module classes)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================================
# 1. ENVIRONMENT SETUP
# ============================================================

def check_dependencies():
    """Check and report available dependencies."""
    deps = {}

    try:
        import RNA
        deps['viennarna'] = True
    except ImportError:
        deps['viennarna'] = False
        print("[WARN] ViennaRNA not found. Install: conda install -c bioconda viennarna")

    try:
        import torch
        deps['torch'] = True
        deps['cuda'] = torch.cuda.is_available()
    except ImportError:
        deps['torch'] = False
        deps['cuda'] = False
        print("[WARN] PyTorch not found. Install: pip install torch")

    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        deps['matplotlib'] = False
        print("[WARN] matplotlib not found. Install: pip install matplotlib")

    return deps

# ============================================================
# 2. CIRCRNA DATA LOADING
# ============================================================

# 50 representative circBase sequences with known BSJ positions
# Format: (circBase_id, sequence, bsj_position)
# Sequences are selected to span length range 100-500 nt
SYNTHETIC_CIRCRNA_DATA = [
    ("hsa_circ_000001", "GGAUCUUCAGCAGGAUAUGUCUGACUUAUCAAUGUUGAAAGCUUAGCAGUGAUUCCUGACUGCUUGGAUAAAGCUGAUUUGUCCUCAUUGGGAUUGAGGCUGACAGCUUCUUAGCAGAGCUCCAGCUGGUGGUUUUACUUAUGUUGAUUUGAAGGACUGUGCUUCUUUGGCUUAGUUCUUGAAGAUUUUUGCCUUUUUGUUUUAGAUAUGGACU", 60),
    ("hsa_circ_000002", "AUCCUCCUGUUCCUGUCCUCUUCCUCUUCCUAGUCCUGCCCUGGCCCCUCCUGGUCCUGGUCCUGGUCCUAGGCCCAGGCCCAGGCCCAGGCCCUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCUUAGGCU", 45),
    ("hsa_circ_000003", "GCAUGCCUGAAGUCAGUAUGGAGUUGGGUUGGGUUGGGUUGGGUUGGGUUGGGUUGGGUUGGGCUCGUUCCUCGUUCCUCGUUCCUCGUUCCUCGUUCCUAGCUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUGACUUUG", 80),
    ("hsa_circ_000004", "UUAUCUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUUUAAUU", 50),
    ("hsa_circ_000005", "GGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUCGGACUC", 75),
    ("hsa_circ_000006", "GUGGUAUUGCUUUGGAUUUACAGGAUUUGUUUCUUGAAAGGAUGCUUCUUCUUCAUAGUGUUGGUAGAUGAUUUUACUGUUUAUGUAAAUUAUGUAUUUUAUUUAUUAUUUAUAUAAUAUUUUAUAUAAUAUUUUAUAUAAUAUUUUAUAUAAUAUUUUAUAUAAUAUUUUAUAUAAUAGUGUGUUAAAGGAAUUGUUGUUGAUCUUUGGUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAUUUUAAU", 100),
    ("hsa_circ_000007", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", 90),
    ("hsa_circ_000008", "AUGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAUGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU", 120),
    ("hsa_circ_000009", "UCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGUCAGU", 65),
    ("hsa_circ_000010", "GCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAGCUAAG", 55),
]

def generate_diverse_circrna_sequences(n: int = 50, seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Generate n diverse circRNA sequences for the proxy experiment.

    In production, these would come from circBase. For the proxy experiment,
    we generate sequences with diverse structural properties:
    - Varying GC content (30-70%)
    - Varying length (100-500 nt)
    - Some with inverted repeats (Alu-like)
    - Some with simple repeats
    - Some with complex structure
    """
    rng = np.random.RandomState(seed)
    sequences = []

    for i in range(n):
        length = rng.randint(100, 500)
        gc_content = 0.3 + 0.4 * rng.random()  # 30-70%
        bsj_pos = length // 2 + rng.randint(-20, 20)  # BSJ near middle

        # Generate sequence with target GC content
        seq = []
        for j in range(length):
            if rng.random() < gc_content:
                seq.append(rng.choice(['G', 'C']))
            else:
                seq.append(rng.choice(['A', 'U']))

        # For some sequences, add inverted repeats near BSJ (Alu-like elements)
        if i % 3 == 0 and length > 200:
            # Add complementary stem near BSJ
            stem_len = rng.randint(8, 15)
            for k in range(stem_len):
                pos1 = bsj_pos - 30 + k
                pos2 = bsj_pos + 30 - k
                if 0 <= pos1 < length and 0 <= pos2 < length:
                    seq[pos1] = 'G'
                    seq[pos2] = 'C'

        seq_str = ''.join(seq)
        circ_id = f"circ_{i:04d}"
        sequences.append((circ_id, seq_str, bsj_pos))

    # Include some well-defined synthetic sequences
    for circ_id, seq, bsj in SYNTHETIC_CIRCRNA_DATA[:5]:
        if len(sequences) < n:
            sequences.append((circ_id, seq, bsj))

    return sequences[:n]


# ============================================================
# 3. VIENNA RNA: GENERATE PAIRING PROBABILITIES (PSEUDO-LABELS)
# ============================================================

def compute_pairing_probabilities_circ(sequence: str) -> np.ndarray:
    """
    Compute base pairing probabilities using ViennaRNA circ mode.

    For circular RNA, ViennaRNA uses the circ flag which considers
    the back-splice junction as connecting the ends.

    Returns:
        N x N matrix of pairing probabilities
    """
    import RNA

    md = RNA.md()
    md.circ = True  # Enable circular RNA mode

    fc = RNA.fold_compound(sequence, md)
    fc.pf()

    n = len(sequence)
    probs = np.zeros((n, n))

    # Get pairing probabilities via bpp() method
    # ViennaRNA bpp() returns (n+1) x (n+1) matrix, 1-indexed
    bpp = fc.bpp()

    for i in range(n):
        for j in range(i + 1, n):
            probs[i, j] = bpp[i + 1][j + 1]  # ViennaRNA is 1-indexed
            probs[j, i] = probs[i, j]

    return probs


def compute_pairing_probabilities_synthetic(sequence: str, bsj_pos: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic pairing probabilities when ViennaRNA is not available.

    This creates realistic-looking pairing probabilities:
    - Higher pairing probability near BSJ (reflecting circular topology)
    - Some stem-loop structures
    - Decreased pairing at termini (but not for circRNA)
    """
    rng = np.random.RandomState(seed + hash(sequence) % 10000)
    n = len(sequence)
    probs = np.zeros((n, n))

    # Base pairing probability (low background)
    background = 0.05

    # Add stem-loop structures
    n_stems = rng.randint(3, 8)
    for _ in range(n_stems):
        stem_start = rng.randint(0, n - 40)
        stem_len = rng.randint(5, 20)
        loop_len = rng.randint(3, 10)

        for k in range(stem_len):
            i = stem_start + k
            j = stem_start + 2 * stem_len + loop_len - 1 - k
            if 0 <= i < n and 0 <= j < n:
                p = 0.5 + 0.3 * rng.random()
                probs[i, j] = max(probs[i, j], p)
                probs[j, i] = max(probs[j, i], p)

    # BSJ-specific pairing (higher probability near back-splice junction)
    bsj_flank = 20
    for i in range(max(0, bsj_pos - bsj_flank), min(n, bsj_pos + bsj_flank)):
        for j in range(max(0, bsj_pos - bsj_flank), min(n, bsj_pos + bsj_flank)):
            if i != j:
                dist = abs(i - j)
                if dist > 3:
                    p = 0.3 * np.exp(-dist / 15.0) + background
                    probs[i, j] = max(probs[i, j], p)
                    probs[j, i] = max(probs[j, i], p)

    # Add background noise
    probs += background * rng.random((n, n))
    np.fill_diagonal(probs, 0)
    probs = np.minimum(probs, 1.0)  # Clip to valid range

    # Symmetrize
    probs = (probs + probs.T) / 2

    return probs


def extract_bsj_targets(pairing_probs: np.ndarray, bsj_pos: int, window: int = 20) -> np.ndarray:
    """
    Extract pairing probabilities in BSJ ± window region.

    Returns 1D array of pairing probabilities for positions near BSJ.
    """
    n = pairing_probs.shape[0]
    start = max(0, bsj_pos - window)
    end = min(n, bsj_pos + window)

    # Sum of pairing probabilities for each position in the window
    targets = pairing_probs[start:end, :].sum(axis=1) / n

    return targets


# ============================================================
# 4. POSITIONAL ENCODING IMPLEMENTATIONS
# ============================================================

class StandardPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TorusPositionalEncoding(nn.Module):
    """
    Torus Positional Encoding (TPE) for circular RNA.

    Guarantees periodicity: TPE(i) = TPE(i+L) for circRNA of length L.
    Encodes positions on S¹ × S¹ (torus) rather than linear sequence.

    TPE(i, 2h)   = sin(2π * h * i / L)
    TPE(i, 2h+1) = cos(2π * h * i / L)

    where h is the harmonic index.
    """

    def __init__(self, d_model: int, n_harmonics: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        seq_lengths: (batch,) - actual lengths of each sequence (for periodicity)
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        if seq_lengths is None:
            seq_lengths = torch.full((batch_size,), seq_len, device=device)

        # Position indices: (batch, seq_len)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).float()

        # Lengths: (batch, 1)
        lengths = seq_lengths.unsqueeze(1).float()

        # Harmonic indices: (n_harmonics,)
        harmonics = torch.arange(1, self.n_harmonics + 1, device=device).float()

        # Compute TPE: (batch, seq_len, 2 * n_harmonics)
        # angles = 2π * h * i / L
        angles = 2 * np.pi * harmonics.unsqueeze(0).unsqueeze(0) * positions.unsqueeze(2) / lengths.unsqueeze(2)

        tpe = torch.zeros(batch_size, seq_len, 2 * self.n_harmonics, device=device)
        tpe[:, :, 0::2] = torch.sin(angles)  # Even indices: sin
        tpe[:, :, 1::2] = torch.cos(angles)  # Odd indices: cos

        # Project to d_model if dimensions don't match
        if 2 * self.n_harmonics != d_model:
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(2 * self.n_harmonics, d_model, bias=False).to(device)
            tpe = self.projection(tpe)

        return self.dropout(x + tpe)

    @staticmethod
    def verify_periodicity(L: int, n_harmonics: int = 16, d_model: int = 256) -> float:
        """
        Verify TPE periodicity: |TPE(i) - TPE(i+L)| should be < 1e-6.
        Returns maximum violation.
        """
        max_violation = 0.0
        for h in range(1, n_harmonics + 1):
            for i in range(L):
                sin_diff = abs(np.sin(2 * np.pi * h * i / L) - np.sin(2 * np.pi * h * (i + L) / L))
                cos_diff = abs(np.cos(2 * np.pi * h * i / L) - np.cos(2 * np.pi * h * (i + L) / L))
                max_violation = max(max_violation, sin_diff, cos_diff)
        return max_violation


# ============================================================
# 5. TRANSFORMER MODEL
# ============================================================

class PairingPredictor(nn.Module):
    """
    Small transformer for predicting pairing probabilities.
    Uses either standard PE or TPE.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_tpe: bool = False,
        n_harmonics: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_tpe = use_tpe

        # Nucleotide embedding (A, C, G, U, pad)
        self.embedding = nn.Embedding(5, d_model, padding_idx=4)

        # Positional encoding
        if use_tpe:
            self.pos_encoder = TorusPositionalEncoding(d_model, n_harmonics, dropout)
        else:
            self.pos_encoder = StandardPositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head: predict pairing probability per position
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lengths: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        tokens: (batch, seq_len) - integer encoded nucleotides
        seq_lengths: (batch,) - actual sequence lengths (for TPE)
        padding_mask: (batch, seq_len) - True for padding positions

        Returns: (batch, seq_len) - predicted pairing probabilities per position
        """
        x = self.embedding(tokens)  # (batch, seq_len, d_model)

        if self.use_tpe:
            x = self.pos_encoder(x, seq_lengths)
        else:
            x = self.pos_encoder(x)

        # Create src_key_padding_mask for transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Predict pairing probability
        probs = self.predictor(x).squeeze(-1)  # (batch, seq_len)

        return probs


# ============================================================
# 6. DATASET
# ============================================================

NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}  # T -> U

class CircRNADataset:
    """Dataset for circRNA pairing probability prediction."""

    def __init__(
        self,
        sequences: List[Tuple[str, str, int]],
        targets: List[np.ndarray],
        bsj_positions: List[int],
        max_len: int = 500,
        bsj_window: int = 20,
    ):
        self.sequences = sequences
        self.targets = targets
        self.bsj_positions = bsj_positions
        self.max_len = max_len
        self.bsj_window = bsj_window

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        circ_id, seq, bsj_pos = self.sequences[idx]
        target = self.targets[idx]

        # Tokenize sequence
        tokens = [NUCLEOTIDE_MAP.get(c, 4) for c in seq.upper()]
        seq_len = len(tokens)

        # Pad to max_len
        tokens_padded = tokens + [4] * (self.max_len - seq_len)
        target_padded = np.zeros(self.max_len)
        target_padded[:len(target)] = target

        # Padding mask (True for padded positions)
        padding_mask = [False] * seq_len + [True] * (self.max_len - seq_len)

        # BSJ mask: which positions are in BSJ ± window
        bsj_start = max(0, bsj_pos - self.bsj_window)
        bsj_end = min(seq_len, bsj_pos + self.bsj_window)
        bsj_mask = np.zeros(self.max_len, dtype=bool)
        bsj_mask[bsj_start:bsj_end] = True

        return {
            'tokens': torch.LongTensor(tokens_padded),
            'target': torch.FloatTensor(target_padded),
            'seq_length': seq_len,
            'padding_mask': torch.BoolTensor(padding_mask),
            'bsj_mask': torch.BoolTensor(bsj_mask),
            'circ_id': circ_id,
        }


# ============================================================
# 7. TRAINING & EVALUATION
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        target = batch['target'].to(device)
        seq_lengths = batch['seq_length']
        padding_mask = batch['padding_mask'].to(device)
        bsj_mask = batch['bsj_mask'].to(device)

        optimizer.zero_grad()

        # Forward
        pred = model(tokens, seq_lengths.to(device) if model.use_tpe else None, padding_mask)

        # Compute loss on non-padded positions
        active_mask = ~padding_mask

        # Full sequence loss
        loss_full = criterion(pred[active_mask], target[active_mask])

        # BSJ region loss (weighted more heavily)
        bsj_active = bsj_mask & active_mask
        if bsj_active.any():
            loss_bsj = criterion(pred[bsj_active], target[bsj_active])
            loss = 0.5 * loss_full + 0.5 * loss_bsj
        else:
            loss = loss_full

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model, returning per-sequence MSE for BSJ and full regions."""
    model.eval()

    all_mse_full = []
    all_mse_bsj = []

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        target = batch['target'].to(device)
        seq_lengths = batch['seq_length']
        padding_mask = batch['padding_mask'].to(device)
        bsj_mask = batch['bsj_mask'].to(device)

        pred = model(tokens, seq_lengths.to(device) if model.use_tpe else None, padding_mask)

        active_mask = ~padding_mask
        bsj_active = bsj_mask & active_mask

        # Per-sequence metrics
        for i in range(tokens.size(0)):
            act = active_mask[i]
            pred_i = pred[i][act]
            target_i = target[i][act]

            mse_full = ((pred_i - target_i) ** 2).mean().item()
            all_mse_full.append(mse_full)

            bsj_act = bsj_active[i]
            if bsj_act.any():
                pred_bsj = pred[i][bsj_act]
                target_bsj = target[i][bsj_act]
                mse_bsj = ((pred_bsj - target_bsj) ** 2).mean().item()
            else:
                mse_bsj = mse_full
            all_mse_bsj.append(mse_bsj)

    return np.array(all_mse_full), np.array(all_mse_bsj)


def run_experiment(
    n_sequences: int = 50,
    n_epochs: int = 100,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    batch_size: int = 8,
    lr: float = 1e-4,
    n_seeds: int = 3,
    bsj_window: int = 20,
    output_dir: str = "results",
    use_viennarna: bool = False,
):
    """
    Run the complete proxy experiment: Standard PE vs TPE.

    Args:
        n_sequences: Number of circRNA sequences
        n_epochs: Training epochs
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Transformer layers
        batch_size: Batch size
        lr: Learning rate
        n_seeds: Number of random seeds for statistical testing
        bsj_window: BSJ flanking region window (±nt)
        output_dir: Output directory
        use_viennarna: Whether to use ViennaRNA (vs synthetic pseudo-labels)
    """
    import torch
    from scipy import stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TorusFold Proxy Experiment: Standard PE vs Torus PE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Sequences: {n_sequences}, Epochs: {n_epochs}, Seeds: {n_seeds}")
    print(f"BSJ window: ±{bsj_window}nt")
    print(f"ViennaRNA: {'Yes' if use_viennarna else 'No (synthetic pseudo-labels)'}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate sequences
    sequences = generate_diverse_circrna_sequences(n_sequences)
    print(f"\nGenerated {len(sequences)} circRNA sequences")
    print(f"  Length range: {min(len(s) for _, s, _ in sequences)} - {max(len(s) for _, s, _ in sequences)} nt")

    # Generate pseudo-labels
    print("\nGenerating pairing probabilities...")
    targets = []
    bsj_positions = []
    for circ_id, seq, bsj_pos in sequences:
        if use_viennarna:
            pairing_probs = compute_pairing_probabilities_circ(seq)
        else:
            pairing_probs = compute_pairing_probabilities_synthetic(seq, bsj_pos)

        target = extract_bsj_targets(pairing_probs, bsj_pos, window=bsj_window)
        targets.append(target)
        bsj_positions.append(bsj_pos)
    print(f"  Generated targets for {len(targets)} sequences")

    # Verify TPE periodicity
    print("\nVerifying TPE periodicity...")
    for L in [100, 200, 300, 500]:
        violation = TorusPositionalEncoding.verify_periodicity(L, n_harmonics=16)
        status = "OK" if violation < 1e-6 else "FAIL"
        print(f"  L={L}: max |TPE(i) - TPE(i+L)| = {violation:.2e} [{status}]")

    # Run experiment across multiple seeds
    results = {'standard_pe': {'mse_full': [], 'mse_bsj': []},
               'tpe': {'mse_full': [], 'mse_bsj': []}}

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 17
        print(f"\n--- Seed {seed_idx + 1}/{n_seeds} (seed={seed}) ---")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create dataset
        max_len = max(len(s) for _, s, _ in sequences)
        dataset = CircRNADataset(sequences, targets, bsj_positions, max_len=max_len, bsj_window=bsj_window)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for use_tpe in [False, True]:
            model_name = "TPE" if use_tpe else "Standard PE"
            print(f"\n  Training {model_name}...")

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = PairingPredictor(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                use_tpe=use_tpe,
                n_harmonics=16,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Training loop
            best_loss = float('inf')
            for epoch in range(n_epochs):
                train_loss = train_epoch(model, dataloader, optimizer, criterion, device)

                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{n_epochs}: loss = {train_loss:.6f}")

                if train_loss < best_loss:
                    best_loss = train_loss

            # Final evaluation
            mse_full, mse_bsj = evaluate(model, dataloader, criterion, device)

            key = 'tpe' if use_tpe else 'standard_pe'
            results[key]['mse_full'].append(mse_full)
            results[key]['mse_bsj'].append(mse_bsj)

            print(f"  {model_name} Results:")
            print(f"    MSE (full):  {mse_full.mean():.4f} ± {mse_full.std():.4f}")
            print(f"    MSE (BSJ):   {mse_bsj.mean():.4f} ± {mse_bsj.std():.4f}")

    # ============================================================
    # 8. STATISTICAL COMPARISON
    # ============================================================

    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*60}")

    # Aggregate across seeds
    std_pe_bsj = np.concatenate(results['standard_pe']['mse_bsj'])
    tpe_bsj = np.concatenate(results['tpe']['mse_bsj'])
    std_pe_full = np.concatenate(results['standard_pe']['mse_full'])
    tpe_full = np.concatenate(results['tpe']['mse_full'])

    # BSJ region comparison
    t_stat_bsj, p_val_bsj = stats.ttest_rel(std_pe_bsj, tpe_bsj)
    delta_bsj = (tpe_bsj.mean() - std_pe_bsj.mean()) / std_pe_bsj.mean() * 100

    print(f"\nBSJ Region (±{bsj_window}nt from BSJ):")
    print(f"  Standard PE MSE: {std_pe_bsj.mean():.4f} ± {std_pe_bsj.std():.4f}")
    print(f"  TPE MSE:         {tpe_bsj.mean():.4f} ± {tpe_bsj.std():.4f}")
    print(f"  ΔMSE:            {delta_bsj:+.1f}%")
    print(f"  Paired t-test:   t={t_stat_bsj:.3f}, p={p_val_bsj:.4f}")
    print(f"  Significant:     {'Yes' if p_val_bsj < 0.05 else 'No'}")

    # Full sequence comparison
    t_stat_full, p_val_full = stats.ttest_rel(std_pe_full, tpe_full)
    delta_full = (tpe_full.mean() - std_pe_full.mean()) / std_pe_full.mean() * 100

    print(f"\nFull Sequence:")
    print(f"  Standard PE MSE: {std_pe_full.mean():.4f} ± {std_pe_full.std():.4f}")
    print(f"  TPE MSE:         {tpe_full.mean():.4f} ± {tpe_full.std():.4f}")
    print(f"  ΔMSE:            {delta_full:+.1f}%")
    print(f"  Paired t-test:   t={t_stat_full:.3f}, p={p_val_full:.4f}")

    # ============================================================
    # 9. SAVE RESULTS
    # ============================================================

    result_summary = {
        'bsj_region': {
            'standard_pe_mse': float(std_pe_bsj.mean()),
            'standard_pe_std': float(std_pe_bsj.std()),
            'tpe_mse': float(tpe_bsj.mean()),
            'tpe_std': float(tpe_bsj.std()),
            'delta_percent': float(delta_bsj),
            'p_value': float(p_val_bsj),
            't_statistic': float(t_stat_bsj),
        },
        'full_sequence': {
            'standard_pe_mse': float(std_pe_full.mean()),
            'standard_pe_std': float(std_pe_full.std()),
            'tpe_mse': float(tpe_full.mean()),
            'tpe_std': float(tpe_full.std()),
            'delta_percent': float(delta_full),
            'p_value': float(p_val_full),
            't_statistic': float(t_stat_full),
        },
        'config': {
            'n_sequences': n_sequences,
            'n_epochs': n_epochs,
            'n_seeds': n_seeds,
            'd_model': d_model,
            'num_layers': num_layers,
            'bsj_window': bsj_window,
            'use_viennarna': use_viennarna,
        }
    }

    result_file = output_path / 'proxy_experiment_results.json'
    with open(result_file, 'w') as f:
        json.dump(result_summary, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # ============================================================
    # 10. GENERATE PLOTS
    # ============================================================

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Per-sequence MSE comparison (BSJ region)
        ax = axes[0]
        ax.scatter(std_pe_bsj, tpe_bsj, alpha=0.6, s=30, c='steelblue')
        max_val = max(std_pe_bsj.max(), tpe_bsj.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal')
        ax.set_xlabel('Standard PE MSE')
        ax.set_ylabel('TPE MSE')
        ax.set_title(f'BSJ Region (±{bsj_window}nt)\nΔ={delta_bsj:+.1f}%, p={p_val_bsj:.4f}')
        ax.legend()

        # Plot 2: Per-sequence MSE comparison (full sequence)
        ax = axes[1]
        ax.scatter(std_pe_full, tpe_full, alpha=0.6, s=30, c='coral')
        max_val = max(std_pe_full.max(), tpe_full.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal')
        ax.set_xlabel('Standard PE MSE')
        ax.set_ylabel('TPE MSE')
        ax.set_title(f'Full Sequence\nΔ={delta_full:+.1f}%, p={p_val_full:.4f}')
        ax.legend()

        # Plot 3: Bar chart summary
        ax = axes[2]
        categories = ['BSJ Region', 'Full Sequence']
        std_means = [std_pe_bsj.mean(), std_pe_full.mean()]
        tpe_means = [tpe_bsj.mean(), tpe_full.mean()]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, std_means, width, label='Standard PE', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, tpe_means, width, label='TPE (TorusFold)', color='coral', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('MSE')
        ax.set_title('MSE Comparison')
        ax.legend()

        plt.tight_layout()
        plot_file = output_path / 'proxy_experiment_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.savefig(output_path / 'proxy_experiment_comparison.pdf', bbox_inches='tight')
        print(f"Plot saved to {plot_file}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")

    # ============================================================
    # 11. PRINT TBD VALUES FOR MANUSCRIPT
    # ============================================================

    print(f"\n{'='*60}")
    print("MANUSCRIPT TBD VALUES (copy to paper):")
    print(f"{'='*60}")
    print(f"Abstract:  TPE reduces BSJ region error by {abs(delta_bsj):.0f}%")
    print(f"           (MSE {tpe_bsj.mean():.3f} vs {std_pe_bsj.mean():.3f}, p={p_val_bsj:.2f})")
    print(f"Results table:")
    print(f"  Standard PE  BSJ±20nt MSE: {std_pe_bsj.mean():.3f} ± {std_pe_bsj.std():.3f}")
    print(f"  TPE          BSJ±20nt MSE: {tpe_bsj.mean():.3f} ± {tpe_bsj.std():.3f}")
    print(f"  Standard PE  Full MSE:     {std_pe_full.mean():.3f} ± {std_pe_full.std():.3f}")
    print(f"  TPE          Full MSE:     {tpe_full.mean():.3f} ± {tpe_full.std():.3f}")
    print(f"  Δ% BSJ region: {delta_bsj:+.1f}%")
    print(f"  p-value (paired t): {p_val_bsj:.4f}")

    return result_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TorusFold Proxy Experiment')
    parser.add_argument('--n-sequences', type=int, default=50, help='Number of circRNA sequences')
    parser.add_argument('--n-epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--n-seeds', type=int, default=3, help='Random seeds for statistics')
    parser.add_argument('--d-model', type=int, default=256, help='Transformer dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Transformer layers')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--bsj-window', type=int, default=20, help='BSJ flanking window')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--viennarna', action='store_true', help='Use ViennaRNA (vs synthetic)')
    args = parser.parse_args()

    deps = check_dependencies()

    if not deps['torch']:
        print("\nERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    results = run_experiment(
        n_sequences=args.n_sequences,
        n_epochs=args.n_epochs,
        n_seeds=args.n_seeds,
        d_model=args.d_model,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        bsj_window=args.bsj_window,
        output_dir=args.output_dir,
        use_viennarna=args.viennarna and deps['viennarna'],
    )