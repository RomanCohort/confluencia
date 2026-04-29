from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .predictor import EpitopeModelBundle, infer_env_cols, make_xy, train_bundle
from confluencia_shared.metrics import rmse as _rmse


@dataclass
class EpitopeTrainResult:
    model_path: str
    metrics: Dict[str, Any]
    env_cols: List[str]


@dataclass
class EpitopeEvalResult:
    model_path: str
    data_path: str
    metrics: Dict[str, float]


def train_epitope_from_csv(
    *,
    data_path: str,
    model_out: str,
    sequence_col: str = "sequence",
    target_col: str = "efficacy",
    env_cols: Optional[Sequence[str]] = None,
    model_name: str = "hgb",
    test_size: float = 0.2,
    seed: int = 42,
    featurizer_version: int = 2,
    mlp_alpha: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_patience: int = 10,
    sgd_alpha: float = 1e-4,
    sgd_l1_ratio: float = 0.15,
    sgd_early_stopping: bool = True,
    hgb_l2: float = 0.0,
) -> EpitopeTrainResult:
    df = pd.read_csv(data_path)
    resolved_env_cols = infer_env_cols(df, sequence_col=sequence_col, target_col=target_col, env_cols=env_cols)

    bundle, metrics = train_bundle(
        df,
        sequence_col=sequence_col,
        target_col=target_col,
        env_cols=list(resolved_env_cols),
        model_name=model_name,  # type: ignore[arg-type]
        test_size=float(test_size),
        random_state=int(seed),
        featurizer_version=int(featurizer_version),
        mlp_alpha=float(mlp_alpha),
        mlp_early_stopping=bool(mlp_early_stopping),
        mlp_patience=int(mlp_patience),
        sgd_alpha=float(sgd_alpha),
        sgd_l1_ratio=float(sgd_l1_ratio),
        sgd_early_stopping=bool(sgd_early_stopping),
        hgb_l2=float(hgb_l2),
    )

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    return EpitopeTrainResult(
        model_path=str(out_path),
        metrics=metrics,
        env_cols=list(resolved_env_cols),
    )


def evaluate_epitope_from_csv(
    *,
    model_path: str,
    data_path: str,
    sequence_col: Optional[str] = None,
    target_col: Optional[str] = None,
    env_cols: Optional[Sequence[str]] = None,
) -> EpitopeEvalResult:
    bundle: EpitopeModelBundle = joblib.load(model_path)
    df = pd.read_csv(data_path)

    used_sequence_col = sequence_col or bundle.sequence_col
    used_target_col = target_col or bundle.target_col
    used_env_cols = list(env_cols) if env_cols is not None else list(bundle.env_cols)

    x, y, _, _ = make_xy(
        df,
        sequence_col=used_sequence_col,
        target_col=used_target_col,
        env_cols=used_env_cols,
        featurizer=None,
        env_medians=dict(bundle.env_medians),
    )

    y_pred = np.asarray(bundle.model.predict(x)).reshape(-1)

    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": _rmse(y, y_pred),
        "r2": float(r2_score(y, y_pred)),
        "n_samples": float(len(y)),
    }

    return EpitopeEvalResult(
        model_path=str(model_path),
        data_path=str(data_path),
        metrics=metrics,
    )


def _parse_env_cols(text: str) -> Optional[List[str]]:
    if not text:
        return None
    cols = [c.strip() for c in text.split(",") if c.strip()]
    return cols or None


def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dedicated epitope training and evaluation module")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train epitope model from CSV")
    p_train.add_argument("--data", required=True, help="Input training CSV path")
    p_train.add_argument("--model-out", required=True, help="Output model .joblib path")
    p_train.add_argument("--sequence-col", default="sequence")
    p_train.add_argument("--target-col", default="efficacy")
    p_train.add_argument("--env-cols", default="", help="Comma-separated env columns")
    p_train.add_argument("--model", default="hgb", choices=["hgb", "gbr", "rf", "mlp", "sgd"])
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--featurizer-version", type=int, default=1)
    p_train.add_argument("--mlp-alpha", type=float, default=1e-4)
    p_train.add_argument("--mlp-early-stopping", action="store_true")
    p_train.add_argument("--mlp-patience", type=int, default=10)
    p_train.add_argument("--sgd-alpha", type=float, default=1e-4)
    p_train.add_argument("--sgd-l1-ratio", type=float, default=0.15)
    p_train.add_argument("--sgd-early-stopping", action="store_true")
    p_train.add_argument("--hgb-l2", type=float, default=0.0)
    p_train.add_argument("--metrics-out", default="", help="Optional JSON output path")

    p_eval = sub.add_parser("eval", help="Evaluate trained epitope model on CSV")
    p_eval.add_argument("--model", required=True, help="Model .joblib path")
    p_eval.add_argument("--data", required=True, help="Evaluation CSV path")
    p_eval.add_argument("--sequence-col", default="")
    p_eval.add_argument("--target-col", default="")
    p_eval.add_argument("--env-cols", default="", help="Comma-separated env columns")
    p_eval.add_argument("--metrics-out", default="", help="Optional JSON output path")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "train":
        result = train_epitope_from_csv(
            data_path=args.data,
            model_out=args.model_out,
            sequence_col=args.sequence_col,
            target_col=args.target_col,
            env_cols=_parse_env_cols(args.env_cols),
            model_name=args.model,
            test_size=args.test_size,
            seed=args.seed,
            featurizer_version=args.featurizer_version,
            mlp_alpha=float(args.mlp_alpha),
            mlp_early_stopping=bool(args.mlp_early_stopping),
            mlp_patience=int(args.mlp_patience),
            sgd_alpha=float(args.sgd_alpha),
            sgd_l1_ratio=float(args.sgd_l1_ratio),
            sgd_early_stopping=bool(args.sgd_early_stopping),
            hgb_l2=float(args.hgb_l2),
        )
        payload = asdict(result)
        _write_json(args.metrics_out, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval":
        result = evaluate_epitope_from_csv(
            model_path=args.model,
            data_path=args.data,
            sequence_col=(args.sequence_col or None),
            target_col=(args.target_col or None),
            env_cols=_parse_env_cols(args.env_cols),
        )
        payload = asdict(result)
        _write_json(args.metrics_out, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
