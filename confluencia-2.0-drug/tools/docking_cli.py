from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def cmd_train(args: argparse.Namespace) -> int:
    from src.drug.docking_cross_attention import train_docking_bundle, save_docking_bundle

    df = pd.read_csv(args.data)
    bundle, metrics = train_docking_bundle(
        df,
        ligand_col=args.smiles_col,
        protein_col=args.protein_col,
        target_col=args.target,
        lig_max_len=args.lig_max_len,
        prot_max_len=args.prot_max_len,
        min_char_freq=args.min_char_freq,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        random_state=args.seed,
        use_cuda=args.cuda,
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_docking_bundle(bundle, str(out_path))

    print("== Docking training done ==")
    print(f"model_out: {out_path}")
    print(f"ligand_col: {bundle.ligand_col}")
    print(f"protein_col: {bundle.protein_col}")
    print(f"target_col: {bundle.target_col}")
    print(f"MAE:  {metrics['mae']:.6g}")
    print(f"RMSE: {metrics['rmse']:.6g}")
    print(f"R2:   {metrics['r2']:.6g}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from src.drug.docking_cross_attention import load_docking_bundle, predict_docking_one

    bundle = load_docking_bundle(args.model)
    pred = predict_docking_one(bundle, smiles=args.smiles, protein=args.protein, use_cuda=args.cuda)
    print("== Docking prediction ==")
    print(f"smiles: {args.smiles}")
    print(f"protein_len: {len(args.protein)}")
    print(f"pred: {pred:.6g}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from src.drug.docking_cross_attention import load_docking_bundle, predict_docking_batch

    bundle = load_docking_bundle(args.model)
    df = pd.read_csv(args.candidates)

    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles_col '{args.smiles_col}' in candidates CSV")
    if args.protein_col not in df.columns:
        raise ValueError(f"Missing protein_col '{args.protein_col}' in candidates CSV")

    ligands = df[args.smiles_col].astype(str).tolist()
    proteins = df[args.protein_col].astype(str).tolist()
    preds = predict_docking_batch(bundle, ligands, proteins, batch_size=args.batch_size, use_cuda=args.cuda)

    out = df.copy()
    out[args.out_col] = preds

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("== Docking screening done ==")
    print(f"candidates: {args.candidates}")
    print(f"out: {out_path}")
    print(f"out_col: {args.out_col}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docking prediction CLI (cross-attention SMILES x protein)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train docking model")
    p_train.add_argument("--data", required=True, help="training CSV")
    p_train.add_argument("--smiles-col", default="smiles", help="SMILES column")
    p_train.add_argument("--protein-col", default="protein", help="protein/receptor column")
    p_train.add_argument("--target", default="docking_score", help="target column")
    p_train.add_argument("--lig-max-len", type=int, default=128)
    p_train.add_argument("--prot-max-len", type=int, default=512)
    p_train.add_argument("--min-char-freq", type=int, default=1)
    p_train.add_argument("--emb-dim", type=int, default=128)
    p_train.add_argument("--n-heads", type=int, default=4)
    p_train.add_argument("--n-layers", type=int, default=2)
    p_train.add_argument("--ff-dim", type=int, default=256)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cuda", action="store_true", help="use CUDA if available")
    p_train.add_argument("--model-out", default="models/docking_crossattn.pt")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="predict docking score for one pair")
    p_pred.add_argument("--model", required=True, help="model .pt")
    p_pred.add_argument("--smiles", required=True, help="SMILES")
    p_pred.add_argument("--protein", required=True, help="protein/receptor sequence")
    p_pred.add_argument("--cuda", action="store_true", help="use CUDA if available")
    p_pred.set_defaults(func=cmd_predict)

    p_screen = sub.add_parser("screen", help="batch predict docking scores")
    p_screen.add_argument("--model", required=True, help="model .pt")
    p_screen.add_argument("--candidates", required=True, help="candidate CSV")
    p_screen.add_argument("--smiles-col", default="smiles")
    p_screen.add_argument("--protein-col", default="protein")
    p_screen.add_argument("--out", default="docking_predictions.csv")
    p_screen.add_argument("--out-col", default="dock_pred")
    p_screen.add_argument("--batch-size", type=int, default=64)
    p_screen.add_argument("--cuda", action="store_true", help="use CUDA if available")
    p_screen.set_defaults(func=cmd_screen)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
