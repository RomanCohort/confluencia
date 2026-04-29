"""Scoring model training wrapper for molecule generation.

This module provides a thin wrapper around the existing PyTorch training
implementation (`train_torch_bundle`) to train a scoring model from a CSV
containing SMILES and a numeric score column. It also exposes a small CLI
for quick local training.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import argparse
import pandas as pd

from .torch_predictor import (
    train_torch_bundle,
    save_torch_bundle,
    load_torch_bundle,
    TorchDrugModelBundle,
)


def train_scoring_model_from_df(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    score_col: str = "score",
    **train_kwargs: Any,
) -> Tuple[TorchDrugModelBundle, Dict[str, Any]]:
    """Train a scoring model from a DataFrame.

    Args:
        df: DataFrame containing SMILES and a numeric score column.
        smiles_col: name of the SMILES column.
        score_col: name of the target score column.
        train_kwargs: forwarded to `train_torch_bundle`.

    Returns:
        (bundle, metrics) as returned by `train_torch_bundle`.
    """
    bundle, metrics = train_torch_bundle(df, smiles_col=smiles_col, target_col=score_col, **train_kwargs)
    return bundle, metrics


def train_scoring_model_cli(argv: None | list = None) -> int:
    parser = argparse.ArgumentParser(description="Train a scoring model from CSV")
    parser.add_argument("csv", help="CSV file with SMILES and score column")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    parser.add_argument("--score-col", default="score", help="Numeric score column name")
    parser.add_argument("--out", default="scoring_model.pt", help="Output path for trained bundle")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=str, default="512,256", help="Comma-separated hidden sizes")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x]

    bundle, metrics = train_scoring_model_from_df(
        df,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        random_state=int(args.seed),
    )

    save_torch_bundle(bundle, args.out)
    print("Saved model to:", args.out)
    print("Metrics:")
    for k, v in metrics.items():
        if k == "history":
            continue
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(train_scoring_model_cli())
