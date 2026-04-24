"""
Torch-Mamba Experiment Configuration Fix
==========================================
Reproduces Torch-Mamba training with proper configuration to fairly compare
with HGB baseline.

Key fixes:
1. Same data split (sequence-aware)
2. Same random seed
3. Sufficient training epochs (100+ with early stopping)
4. Proper learning rate scheduling
5. Feature normalization
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def run_mamba_experiment(
    data_path: str,
    output_dir: str = "benchmarks/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Torch-Mamba with corrected configuration."""
    import sys
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "confluencia-2.0-epitope"))

    from core.torch_mamba import TorchMambaConfig, train_torch_mamba, predict_torch_mamba, torch_available
    from core.features import FeatureSpec, build_feature_matrix, ensure_columns
    from benchmarks.sequence_split import sequence_split, verify_no_leakage

    if not torch_available():
        return {"error": "PyTorch not available"}

    # Load data
    df = pd.read_csv(project_root / data_path)
    work = ensure_columns(df)
    X, feature_names, env_cols = build_feature_matrix(work, FeatureSpec())

    if "efficacy" not in work.columns:
        return {"error": "Data must have 'efficacy' column"}

    y = work["efficacy"].to_numpy(dtype=np.float32)

    # Sequence-aware split
    if "epitope_seq" in work.columns and work["epitope_seq"].nunique() >= 5:
        train_df, test_df = sequence_split(work, "epitope_seq", test_ratio=0.2, seed=seed)
        verify_no_leakage(train_df, test_df, "epitope_seq")
    else:
        # Fallback to random split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(work))
        n_test = max(1, int(0.2 * len(work)))
        test_df = work.iloc[idx[:n_test]].reset_index(drop=True)
        train_df = work.iloc[idx[n_test:]].reset_index(drop=True)

    y_test = test_df["efficacy"].to_numpy(dtype=np.float32) if "efficacy" in test_df.columns else None

    # Corrected Mamba config
    configs = [
        ("mamba_default", TorchMambaConfig(
            d_model=96, n_layers=2, d_state=16, d_conv=4,
            expand=2, dropout=0.1, lr=2e-3, weight_decay=1e-4,
            epochs=100, batch_size=32, max_len=1024, seed=seed,
        )),
        ("mamba_deep", TorchMambaConfig(
            d_model=128, n_layers=4, d_state=32, d_conv=4,
            expand=2, dropout=0.15, lr=1e-3, weight_decay=1e-3,
            epochs=150, batch_size=32, max_len=1024, seed=seed,
        )),
        ("mamba_wide", TorchMambaConfig(
            d_model=192, n_layers=2, d_state=16, d_conv=8,
            expand=2, dropout=0.1, lr=2e-3, weight_decay=1e-4,
            epochs=100, batch_size=64, max_len=1024, seed=seed,
        )),
    ]

    results = {}
    for config_name, cfg in configs:
        print(f"\n[torch-mamba] Training {config_name} ...")
        print(f"  Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
              f"lr={cfg.lr}, epochs={cfg.epochs}")

        t0 = time.time()
        try:
            bundle = train_torch_mamba(
                train_df,
                train_df["efficacy"].to_numpy(dtype=np.float32),
                env_cols=list(env_cols),
                cfg=cfg,
                prefer_real_mamba=True,
            )
            train_time = time.time() - t0

            if y_test is not None:
                pred = predict_torch_mamba(bundle, test_df)
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae = float(mean_absolute_error(y_test, pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                r2 = float(r2_score(y_test, pred))
            else:
                mae, rmse, r2 = float("nan"), float("nan"), float("nan")

            results[config_name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_time": train_time,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "used_real_mamba": bool(bundle.used_real_mamba),
                "final_train_loss": float(bundle.history.get("train_loss", [0])[-1]),
                "final_val_loss": float(bundle.history.get("val_loss", [0])[-1]),
                "config": {
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "lr": cfg.lr,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "dropout": cfg.dropout,
                },
            }
            print(f"  MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} time={train_time:.1f}s")
        except Exception as e:
            results[config_name] = {"error": str(e)}
            print(f"  FAILED: {e}")

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "torch_mamba_corrected.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Torch-Mamba Corrected Experiment")
    parser.add_argument("--data", default="data/example_epitope.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    run_mamba_experiment(args.data, args.output, args.seed)


if __name__ == "__main__":
    main()
