"""
Train CircRNA Encoder — Command-line training script.

Usage:
    python scripts/train.py \
        --training-data data/training_pairs.csv \
        --output-dir data/models \
        --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from core.training import train_model


def main():
    parser = argparse.ArgumentParser(description="Train CircRNA Encoder")
    parser.add_argument("--training-data", required=True, help="Training CSV path")
    parser.add_argument("--output-dir", default="data/models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve paths
    train_csv = Path(args.training_data)
    if not train_csv.exists():
        train_csv = _PROJECT_ROOT.parent / args.training_data

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir = _PROJECT_ROOT / args.output_dir

    train_model(
        train_csv=str(train_csv),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()