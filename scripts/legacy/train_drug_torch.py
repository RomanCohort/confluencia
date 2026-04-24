from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drug.torch_predictor import save_torch_bundle, train_torch_bundle


def _parse_hidden_sizes(text: str) -> List[int]:
    raw = [s.strip() for s in str(text).split(",") if s.strip()]
    if not raw:
        return [512, 256]
    sizes: List[int] = []
    for s in raw:
        v = int(s)
        if v <= 0:
            raise ValueError("hidden_sizes must be positive")
        sizes.append(v)
    return sizes


def _parse_env_cols(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    cols = [s.strip() for s in text.split(",") if s.strip()]
    return cols or None


def main() -> int:
    p = argparse.ArgumentParser(description="Train torch drug efficacy predictor")
    p.add_argument("--data", required=True, help="Training CSV path")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", default="efficacy")
    p.add_argument("--env-cols", default="", help="Comma-separated env cols")
    p.add_argument("--hidden-sizes", default="512,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--featurizer-version", type=int, default=2)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--out", default="models/drug_torch_model.pt")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    env_cols = _parse_env_cols(args.env_cols)

    bundle, metrics = train_torch_bundle(
        df,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        env_cols=env_cols,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        random_state=args.seed,
        featurizer_version=args.featurizer_version,
        radius=args.radius,
        n_bits=args.n_bits,
        use_cuda=bool(args.use_cuda),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_bundle(bundle, str(out_path))

    print("== Torch Training Done ==")
    print(f"model_out: {out_path}")
    print(f"smiles_col: {bundle.smiles_col}")
    print(f"target_col: {bundle.target_col}")
    print(f"env_cols: {bundle.env_cols}")
    print(f"n_features: {metrics['n_features']}")
    print(f"invalid_smiles: {metrics['invalid_smiles']}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
