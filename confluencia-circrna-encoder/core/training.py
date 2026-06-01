"""
Training module for circRNA encoder.

Mirrors drug module's training.py structure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .encoder import CircRNAEncoder, CircRNAEncoderConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CircRNATrainDataset(Dataset):
    """Training dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        gene_cols: List[str] = None,
        composite_targets: List[str] = None,
        report_targets: List[str] = None,
    ):
        self.df = df
        self.sequence_col = sequence_col
        self.gene_cols = gene_cols or ["gene_TROP2", "gene_NECTIN4", "gene_LIV-1", "gene_B7-H4", "gene_MKI67", "gene_MYC"]
        self.composite_targets = composite_targets or [
            "target_immunotherapy_score", "target_tumor_killing_index",
            "target_overall_immunogenicity", "target_immune_cycle_score",
            "target_tme_score", "target_therapeutic_window",
            "target_tide_score", "target_ips",
        ]
        self.report_targets = report_targets or [
            "target_rig_i_score", "target_tlr_score",
            "target_pkr_score", "target_trained_model_risk",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sequence = str(row[self.sequence_col])

        gene_expr = torch.tensor(
            [row.get(c, 0.5) for c in self.gene_cols],
            dtype=torch.float32
        )

        composite = torch.tensor(
            [row.get(c, 0.5) for c in self.composite_targets],
            dtype=torch.float32
        )

        report = torch.tensor(
            [row.get(c, 0.5) for c in self.report_targets],
            dtype=torch.float32
        )

        response_str = row.get("target_predicted_response", "intermediate")
        response_map = {"likely_non_responder": 0, "intermediate": 1, "likely_responder": 2}
        response = torch.tensor(response_map.get(response_str, 1), dtype=torch.long)

        return {
            "sequence": sequence,
            "gene_expr": gene_expr,
            "composite_target": composite,
            "report_target": report,
            "response_target": response,
        }


class MultiTaskLoss(nn.Module):
    """Multi-task distillation loss."""

    def __init__(
        self,
        composite_weight: float = 1.0,
        report_weight: float = 0.5,
        response_weight: float = 0.8,
    ):
        super().__init__()
        self.composite_weight = composite_weight
        self.report_weight = report_weight
        self.response_weight = response_weight
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        composite_loss = self.bce(outputs["composite"], targets["composite_target"])
        report_loss = self.bce(outputs["report"], targets["report_target"])
        response_loss = self.ce(outputs["response_logits"], targets["response_target"])

        total = (
            self.composite_weight * composite_loss +
            self.report_weight * report_loss +
            self.response_weight * response_loss
        )

        return {
            "total": total,
            "composite": composite_loss.item(),
            "report": report_loss.item(),
            "response": response_loss.item(),
        }


def train_epoch(
    model: CircRNAEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: str,
    max_grad_norm: float = 1.0,
) -> Dict:
    """Train one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        sequences = batch["sequence"]
        gene_expr = batch["gene_expr"].to(device)
        composite_target = batch["composite_target"].to(device)
        report_target = batch["report_target"].to(device)
        response_target = batch["response_target"].to(device)

        optimizer.zero_grad()

        outputs = model.forward(sequences, gene_expr, device=device)

        targets = {
            "composite_target": composite_target,
            "report_target": report_target,
            "response_target": response_target,
        }
        loss_dict = loss_fn(outputs, targets)

        loss_dict["total"].backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        total_loss += loss_dict["total"].item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def validate(
    model: CircRNAEncoder,
    dataloader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: str,
) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"]
            gene_expr = batch["gene_expr"].to(device)
            composite_target = batch["composite_target"].to(device)
            report_target = batch["report_target"].to(device)
            response_target = batch["response_target"].to(device)

            outputs = model.forward(sequences, gene_expr, device=device)

            targets = {
                "composite_target": composite_target,
                "report_target": report_target,
                "response_target": response_target,
            }
            loss_dict = loss_fn(outputs, targets)

            total_loss += loss_dict["total"].item()
            n_batches += 1

    return {"loss": total_loss / n_batches}


def save_checkpoint(
    model: CircRNAEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    path: Path,
):
    """
    Save complete checkpoint with metadata.

    Includes:
    - Model heads state dicts
    - Optimizer state
    - Scheduler state
    - Training metadata
    """
    torch.save({
        'epoch': epoch,
        'config': model.config.to_dict(),
        'composite_head': model.composite_head.state_dict(),
        'report_head': model.report_head.state_dict(),
        'response_head': model.response_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
    }, path)


def train_model(
    train_csv: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
    patience: int = 5,
    max_grad_norm: float = 1.0,
) -> Dict:
    """
    Train circRNA encoder model.

    Args:
        train_csv: Training data CSV
        output_dir: Output directory
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        seed: Random seed

    Returns:
        Training history
    """
    print("=" * 60)
    print("Training CircRNA Encoder")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} samples")

    # Split
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Datasets
    train_dataset = CircRNATrainDataset(train_df)
    val_dataset = CircRNATrainDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    config = CircRNAEncoderConfig(lr=lr, batch_size=batch_size)
    model = CircRNAEncoder(config)

    try:
        model.load_backbone(device=device)
        print("Backbone loaded")
    except Exception as e:
        print(f"Warning: backbone not loaded: {e}")

    model = model.to(device)

    # Loss & Optimizer
    loss_fn = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training with early stopping
    best_val_loss = float("inf")
    no_improve_count = 0
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            max_grad_norm=max_grad_norm
        )
        print(f"  Train: {train_metrics['loss']:.4f}")

        val_metrics = validate(model, val_loader, loss_fn, device)
        print(f"  Val: {val_metrics['loss']:.4f}")

        # Scheduler step
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]['lr'],
        })

        # Early stopping logic
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve_count = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, output_path / "best.pt")
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epochs")

            if no_improve_count >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["loss"], output_path / "final.pt")

    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 60)

    return {"history": history, "best_val_loss": best_val_loss}