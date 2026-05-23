"""
train_pipeline.py — Complete training pipeline for circRNA encoder.

Pipeline:
1. Self-supervised pretraining (circBase 140k sequences, no labels)
2. Fine-tuning (with labels if available)

Usage:
    python train_pipeline.py --mode full
    python train_pipeline.py --mode pretrain
    python train_pipeline.py --mode finetune
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import encoder components
sys.path.insert(0, str(_PROJECT_ROOT / "confluencia-circrna-encoder"))
from core.encoder import CircRNAEncoder, CircRNAEncoderConfig
from core.training import train_model

# Try to import ESM
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("[Warning] ESM not available, install with: pip install fair-esm")


class PretrainConfig:
    """Pretraining configuration."""

    def __init__(
        self,
        fasta_path: str = "data/circrna/circbase_seqs.fa.gz",
        max_sequences: int = 140000,
        max_seq_length: int = 512,
        backbone_model: str = "esm2_t33_650M_UR50D",
        backbone_dim: int = 1280,
        pretrain_epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-4,
        mask_ratio: float = 0.15,
        output_dir: str = "confluencia-circrna-encoder/data/models",
    ):
        self.fasta_path = fasta_path
        self.max_sequences = max_sequences
        self.max_seq_length = max_seq_length
        self.backbone_model = backbone_model
        self.backbone_dim = backbone_dim
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.output_dir = output_dir

    def to_dict(self):
        return {
            "fasta_path": self.fasta_path,
            "max_sequences": self.max_sequences,
            "pretrain_epochs": self.pretrain_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }


class FineTuneConfig:
    """Fine-tuning configuration."""

    def __init__(
        self,
        labeled_data: str = "data/circrna/unified_training_data.csv",
        finetune_epochs: int = 20,
        batch_size: int = 8,
        lr: float = 1e-3,
    ):
        self.labeled_data = labeled_data
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr

    def to_dict(self):
        return {
            "labeled_data": self.labeled_data,
            "finetune_epochs": self.finetune_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }


class CircRNAPretrainDataset(Dataset):
    """Dataset for self-supervised pretraining."""

    def __init__(self, fasta_path: str, max_sequences: int, max_seq_length: int, seed: int = 42):
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
            start = np.random.randint(0, len(sequence) - self.max_seq_length)
            sequence = sequence[start:start + self.max_seq_length]
        sequence = sequence.replace('T', 'U')
        return {'sequence': sequence, 'idx': idx}


class MaskedLMHead(nn.Module):
    """Masked nucleotide prediction head."""

    VOCAB_SIZE = 4  # A, U, G, C

    def __init__(self, backbone_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.VOCAB_SIZE),
        )

    def forward(self, x):
        return self.net(x)


class PretrainModel(nn.Module):
    """Self-supervised pretraining model."""

    def __init__(self, backbone_dim: int = 1280):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.backbone = None
        self.alphabet = None
        self.mlm_head = MaskedLMHead(backbone_dim)
        self.nuc_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    def load_backbone(self, device: str = "cuda"):
        if not ESM_AVAILABLE:
            print("[Warning] No backbone, using mock")
            return

        print("Loading ESM2 backbone...")
        self.backbone, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.backbone = self.backbone.to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone loaded (frozen)")

    def encode(self, sequence: str, device: str = "cuda") -> torch.Tensor:
        seq_t = sequence.replace('U', 'T')
        if self.backbone is None:
            return torch.randn(len(seq_t), self.backbone_dim, device=device)

        batch_converter = self.alphabet.get_batch_converter()
        _, _, tokens = batch_converter([('seq', seq_t)])
        tokens = tokens.to(device)

        with torch.no_grad():
            results = self.backbone(tokens, repr_layers=[33])
            return results["representations"][33].squeeze(0)

    def apply_mask(self, sequence: str, mask_ratio: float = 0.15):
        seq_len = len(sequence)
        n_mask = int(seq_len * mask_ratio)
        mask_positions = np.random.choice(seq_len, n_mask, replace=False).tolist()
        masked_seq = list(sequence)
        for pos in mask_positions:
            masked_seq[pos] = 'M'  # Mask marker
        return ''.join(masked_seq), mask_positions

    def forward(self, sequence: str, mask_ratio: float, device: str = "cuda"):
        masked_seq, mask_positions = self.apply_mask(sequence, mask_ratio)
        if len(mask_positions) == 0:
            return {'loss': torch.tensor(0.0, device=device), 'accuracy': 0.0}

        # Encode masked sequence
        embeddings = self.encode(masked_seq, device)

        # Mask embeddings at masked positions
        mask_emb = torch.zeros(embeddings.shape[1], device=device)
        for pos in mask_positions:
            if pos < embeddings.shape[0]:
                embeddings[pos] = mask_emb

        # Predict
        if len(mask_positions) > embeddings.shape[0]:
            mask_positions = mask_positions[:embeddings.shape[0]]

        logits = self.mlm_head(embeddings[mask_positions])
        targets = torch.tensor(
            [self.nuc_to_idx.get(sequence[pos], 0) for pos in mask_positions],
            dtype=torch.long, device=device
        )

        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(-1) == targets).float().mean().item()

        return {'loss': loss, 'accuracy': accuracy}


def pretrain_epoch(model: PretrainModel, dataloader: DataLoader, optimizer, mask_ratio: float, device: str):
    """Train one pretraining epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence']

        losses = []
        accs = []

        for seq in sequences:
            result = model.forward(seq, mask_ratio, device)
            if result['loss'].item() > 0:
                losses.append(result['loss'])
                accs.append(result['accuracy'])

        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += np.mean(accs)
            n += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: Loss={total_loss/max(n,1):.4f}, Acc={total_acc/max(n,1):.2%}")

    return {'loss': total_loss / max(n, 1), 'accuracy': total_acc / max(n, 1)}


def run_pretrain(config: PretrainConfig, device: str = "cuda"):
    """Run self-supervised pretraining."""
    print("\n" + "=" * 70)
    print("STAGE 1: Self-Supervised Pretraining")
    print("=" * 70)
    print(f"Data: {config.fasta_path}")
    print(f"Epochs: {config.pretrain_epochs}")
    print(f"Mask ratio: {config.mask_ratio}")
    print("=" * 70)

    # Dataset
    dataset = CircRNAPretrainDataset(
        config.fasta_path,
        config.max_sequences,
        config.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: {'sequence': [x['sequence'] for x in b]}
    )

    # Model
    model = PretrainModel(config.backbone_dim)
    model.load_backbone(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.mlm_head.parameters(), lr=config.lr)

    # Training
    history = []
    best_loss = float('inf')
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.pretrain_epochs + 1):
        print(f"\nEpoch {epoch}/{config.pretrain_epochs}")
        metrics = pretrain_epoch(model, dataloader, optimizer, config.mask_ratio, device)

        print(f"  Summary: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.2%}")
        history.append({'epoch': epoch, **metrics})

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(model.mlm_head.state_dict(), output_dir / "pretrain_head.pt")
            print(f"  ✓ Saved best head")

    # Save history
    with open(output_dir / "pretrain_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Pretraining complete! Best loss: {best_loss:.4f}")
    return model


def run_finetune(config: FineTuneConfig, pretrain_model: PretrainModel, device: str = "cuda"):
    """Run fine-tuning with labels."""
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-Tuning (with labels)")
    print("=" * 70)

    # Check if labeled data exists
    labeled_path = Path(config.labeled_data)
    if not labeled_path.exists():
        labeled_path = _PROJECT_ROOT / config.labeled_data

    if not labeled_path.exists():
        print("⚠ No labeled data found, skipping fine-tuning")
        print(f"  Expected: {labeled_path}")
        return None

    print(f"Data: {labeled_path}")

    # Run fine-tuning
    output_dir = Path("confluencia-circrna-encoder/data/models")

    result = train_model(
        train_csv=str(labeled_path),
        output_dir=str(output_dir),
        epochs=config.finetune_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=device,
    )

    print(f"\n✓ Fine-tuning complete!")
    return result


def main():
    parser = argparse.ArgumentParser(description="Complete circRNA training pipeline")
    parser.add_argument("--mode", choices=["full", "pretrain", "finetune"], default="full")
    parser.add_argument("--fasta", default="data/circrna/circbase_seqs.fa.gz")
    parser.add_argument("--labeled-data", default="data/circrna/unified_training_data.csv")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-sequences", type=int, default=140000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("circRNA Encoder Training Pipeline")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Configs
    pretrain_config = PretrainConfig(
        fasta_path=args.fasta,
        max_sequences=args.max_sequences,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    finetune_config = FineTuneConfig(
        labeled_data=args.labeled_data,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size * 2,
        lr=args.lr * 10,
    )

    # Run pipeline
    pretrain_model = None

    if args.mode in ["full", "pretrain"]:
        pretrain_model = run_pretrain(pretrain_config, device)

    if args.mode in ["full", "finetune"]:
        run_finetune(finetune_config, pretrain_model, device)

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print("Models saved to: confluencia-circrna-encoder/data/models/")
    print("  - pretrain_head.pt (pretrained MLM head)")
    print("  - best.pt (fine-tuned encoder)")
    print("  - final.pt (final model)")


if __name__ == "__main__":
    main()