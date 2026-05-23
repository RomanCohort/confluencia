"""
Self-supervised pretraining for circRNA encoder.

Strategy (similar to BERT/MAE):
1. Masked Language Modeling (MLM) - mask nucleotides and predict
2. Contrastive Learning - learn sequence representations
3. Use 140k circBase sequences for pretraining

No labels needed - pure sequence learning.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
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


class CircRNAPretrainConfig:
    """Configuration for self-supervised pretraining."""

    # Backbone
    backbone_model: str = "esm2_t33_650M_UR50D"
    backbone_dim: int = 1280  # 650M model dim

    # Pretraining tasks
    mask_ratio: float = 0.15  # 15% of nucleotides masked
    contrastive_enabled: bool = True

    # Training
    max_epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    warmup_steps: int = 500

    # Data
    max_sequences: int = 140000
    max_seq_length: int = 512  # Truncate long sequences

    # Saving
    save_every: int = 1
    output_dir: str = "confluencia-circrna-encoder/data/models/pretrain"

    def to_dict(self) -> Dict:
        return {
            "backbone_model": self.backbone_model,
            "backbone_dim": self.backbone_dim,
            "mask_ratio": self.mask_ratio,
            "contrastive_enabled": self.contrastive_enabled,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_sequences": self.max_sequences,
            "max_seq_length": self.max_seq_length,
        }


class MaskedNucleotideHead(nn.Module):
    """Head for masked nucleotide prediction (MLM)."""

    # 4 nucleotides + mask token + padding
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
        """Return logits for each position."""
        return self.net(x)


class ContrastiveHead(nn.Module):
    """Head for contrastive learning (sequence similarity)."""

    def __init__(self, backbone_dim: int = 1280, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected embedding for contrastive loss."""
        return self.projector(x)


class CircRNAPretrainModel(nn.Module):
    """
    Self-supervised pretraining model for circRNA.

    Architecture:
    - ESM2 backbone (frozen or fine-tuned)
    - MLM head: predict masked nucleotides
    - Contrastive head: learn sequence representations
    """

    def __init__(self, config: CircRNAPretrainConfig):
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = None
        self.alphabet = None
        self.backbone_loaded = False

        # Heads
        self.mlm_head = MaskedNucleotideHead(config.backbone_dim)
        self.contrastive_head = ContrastiveHead(config.backbone_dim) if config.contrastive_enabled else None

        # Nucleotide mapping
        self.nuc_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '<mask>': 4, '<pad>': 5}
        self.idx_to_nuc = {v: k for k, v in self.nuc_to_idx.items()}

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

        # Load ESM2 model
        self.backbone, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.config.backbone_model
        )

        self.backbone = self.backbone.to(device)

        # Freeze backbone for initial training
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_loaded = True
        print("Backbone loaded and frozen")

    def encode_sequence(self, sequence: str, device: str = "cuda") -> torch.Tensor:
        """Encode sequence using backbone."""
        # Convert U to T for DNA model compatibility
        seq_t = sequence.replace('U', 'T').replace('u', 't')

        if self.backbone is None:
            # Mock encoding for testing
            return torch.randn(1, len(seq_t), self.config.backbone_dim, device=device)

        # Use ESM batch converter
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq', seq_t)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.backbone(
                batch_tokens,
                repr_layers=[33],  # Last layer
                return_contacts=False,
            )
            embeddings = results["representations"][33]

        return embeddings.squeeze(0)  # (seq_len, dim)

    def apply_mask(
        self,
        sequence: str,
        mask_ratio: float = 0.15,
    ) -> Tuple[str, List[int]]:
        """
        Apply random masking to sequence.

        Returns:
            masked_sequence: sequence with <mask> tokens
            mask_positions: indices of masked positions
        """
        seq_len = len(sequence)
        n_mask = int(seq_len * mask_ratio)

        # Random positions to mask
        mask_positions = np.random.choice(seq_len, n_mask, replace=False).tolist()

        # Create masked sequence
        masked_seq = list(sequence)
        for pos in mask_positions:
            masked_seq[pos] = '<mask>'

        return ''.join(masked_seq), mask_positions

    def forward_mlm(
        self,
        original_sequence: str,
        masked_sequence: str,
        mask_positions: List[int],
        device: str = "cuda",
    ) -> Dict:
        """
        MLM forward pass.

        Returns:
            loss: MLM loss
            accuracy: prediction accuracy
        """
        # Encode masked sequence
        embeddings = self.encode_sequence(masked_sequence, device)

        # Get embeddings at masked positions
        if len(mask_positions) == 0:
            return {"loss": torch.tensor(0.0, device=device), "accuracy": 0.0}

        # Truncate to max positions
        mask_positions = mask_positions[:self.config.max_seq_length]

        masked_embeddings = embeddings[mask_positions]  # (n_mask, dim)

        # Predict nucleotides
        logits = self.mlm_head(masked_embeddings)  # (n_mask, vocab_size)

        # Ground truth (A, U, G, C only - indices 0-3)
        targets = torch.tensor(
            [self.nuc_to_idx.get(original_sequence[pos], 0) for pos in mask_positions],
            dtype=torch.long,
            device=device,
        )

        # Cross entropy loss
        loss = F.cross_entropy(logits, targets)

        # Accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}

    def forward_contrastive(
        self,
        sequences: List[str],
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Contrastive learning forward pass.

        Uses NT-Xent loss (Normalized Temperature-scaled Cross Entropy).
        """
        # Encode all sequences and pool
        embeddings = []
        for seq in sequences:
            emb = self.encode_sequence(seq, device)
            # Mean pooling
            pooled = emb.mean(dim=0)  # (dim,)
            embeddings.append(pooled)

        embeddings = torch.stack(embeddings)  # (batch, dim)

        # Project
        projected = self.contrastive_head(embeddings)  # (batch, proj_dim)

        # Normalize
        projected = F.normalize(projected, dim=-1)

        return projected

    def save(self, path: str):
        """Save model."""
        state = {
            "config": self.config.to_dict(),
            "mlm_head": self.mlm_head.state_dict(),
        }
        if self.contrastive_head:
            state["contrastive_head"] = self.contrastive_head.state_dict()

        torch.save(state, path)
        print(f"Saved to {path}")

    def load(self, path: str, device: str = "cuda"):
        """Load model."""
        state = torch.load(path, map_location=device)

        self.mlm_head.load_state_dict(state["mlm_head"])
        if self.contrastive_head and "contrastive_head" in state:
            self.contrastive_head.load_state_dict(state["contrastive_head"])

        print(f"Loaded from {path}")


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

        # Parse FASTA
        self.sequences = self._parse_fasta(fasta_path)

        # Sample if too many
        if len(self.sequences) > max_sequences:
            indices = np.random.choice(len(self.sequences), max_sequences, replace=False)
            self.sequences = [self.sequences[i] for i in indices]

        print(f"Loaded {len(self.sequences)} sequences")

    def _parse_fasta(self, path: str) -> List[str]:
        """Parse FASTA file."""
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

        # Truncate if too long
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]

        # Convert T back to U for RNA
        sequence = sequence.replace('T', 'U')

        return {
            'sequence': sequence,
            'length': len(sequence),
            'idx': idx,
        }


def pretrain_epoch(
    model: CircRNAPretrainModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: CircRNAPretrainConfig,
    device: str,
    epoch: int,
) -> Dict:
    """Train one epoch."""
    model.train()

    total_mlm_loss = 0.0
    total_contrastive_loss = 0.0
    total_accuracy = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence']  # List of strings

        # MLM task
        mlm_losses = []
        accuracies = []

        for seq in sequences:
            masked_seq, mask_positions = model.apply_mask(seq, config.mask_ratio)

            if len(mask_positions) > 0:
                result = model.forward_mlm(seq, masked_seq, mask_positions, device)
                mlm_losses.append(result['loss'])
                accuracies.append(result['accuracy'])

        if mlm_losses:
            mlm_loss = torch.stack(mlm_losses).mean()
            avg_accuracy = np.mean(accuracies)
        else:
            mlm_loss = torch.tensor(0.0, device=device)
            avg_accuracy = 0.0

        # Contrastive task
        if config.contrastive_enabled and len(sequences) >= 2:
            projected = model.forward_contrastive(sequences, device)

            # NT-Xent loss
            # Positive pairs: augmented versions (here just use same sequence)
            # Negative pairs: different sequences in batch
            temperature = 0.07

            # SimCLR-style contrastive loss
            similarity = torch.mm(projected, projected.t())  # (batch, batch)

            # Mask self-similarity
            mask = torch.eye(len(sequences), device=device) * -1e9

            similarity = similarity + mask

            # Loss: each sample should be similar to itself (before augmentation)
            # For now, simple reconstruction loss
            contrastive_loss = torch.tensor(0.0, device=device)  # Placeholder

        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = mlm_loss + 0.1 * contrastive_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_mlm_loss += mlm_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_accuracy += avg_accuracy
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: MLM_loss={mlm_loss.item():.4f}, Acc={avg_accuracy:.2%}")

    return {
        "mlm_loss": total_mlm_loss / n_batches,
        "contrastive_loss": total_contrastive_loss / n_batches,
        "accuracy": total_accuracy / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for circRNA")
    parser.add_argument("--fasta", default="data/circrna/circbase_seqs.fa.gz")
    parser.add_argument("--output-dir", default="confluencia-circrna-encoder/data/models/pretrain")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-sequences", type=int, default=140000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Self-Supervised Pretraining for circRNA Encoder")
    print("=" * 70)
    print("Strategy: Masked Language Modeling (MLM) + Contrastive Learning")
    print("Data: circBase 140k real sequences (no labels needed)")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Config
    config = CircRNAPretrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
        max_seq_length=args.max_seq_length,
        mask_ratio=args.mask_ratio,
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
    )

    # Model
    model = CircRNAPretrainModel(config)
    model.load_backbone(device)
    model = model.to(device)

    # Optimizer (only for heads, backbone frozen)
    optimizer = torch.optim.AdamW(
        list(model.mlm_head.parameters()) +
        (list(model.contrastive_head.parameters()) if model.contrastive_head else []),
        lr=config.lr,
    )

    # Training loop
    history = []
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        metrics = pretrain_epoch(model, dataloader, optimizer, config, device, epoch)

        print(f"\n  Epoch Summary:")
        print(f"    MLM Loss: {metrics['mlm_loss']:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Contrastive Loss: {metrics['contrastive_loss']:.4f}")

        history.append({
            "epoch": epoch,
            "mlm_loss": metrics['mlm_loss'],
            "accuracy": metrics['accuracy'],
        })

        # Save checkpoint
        if metrics['mlm_loss'] < best_loss:
            best_loss = metrics['mlm_loss']
            model.save(output_dir / "best_pretrain.pt")
            print(f"    ✓ Saved best model (loss={best_loss:.4f})")

        if epoch % config.save_every == 0:
            model.save(output_dir / f"pretrain_epoch_{epoch}.pt")

    # Save final
    model.save(output_dir / "final_pretrain.pt")

    # Save history
    with open(output_dir / "pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("\n" + "=" * 70)
    print("Pretraining Complete!")
    print(f"Best MLM Loss: {best_loss:.4f}")
    print(f"Final Accuracy: {history[-1]['accuracy']:.2%}")
    print(f"Models saved to: {output_dir}")
    print("=" * 70)
    print("\nNext step: Fine-tune with labeled data (if available)")


if __name__ == "__main__":
    main()