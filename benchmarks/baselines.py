"""
Confluencia Baseline Comparison Framework
==========================================
Compare against standard ML baselines and existing tools.

Usage:
    python -m benchmarks.baselines --module epitope --data data/example_epitope.csv
    python -m benchmarks.baselines --module drug --data data/example_drug.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EPITOPE_DIR = _PROJECT_ROOT / "confluencia-2.0-epitope"
_DRUG_DIR = _PROJECT_ROOT / "confluencia-2.0-drug"


def _ensure_path(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)


_EPITOPE_COL_MAP = {
    "sequence": "epitope_seq",
    "concentration": "dose",
    "cell_density": "circ_expr",
    "incubation_hours": "treatment_time",
}


def _normalise_epitope_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw, internal in _EPITOPE_COL_MAP.items():
        if raw in out.columns and internal not in out.columns:
            out[internal] = out[raw]
    return out

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    # Classic linear
    "ridge": lambda seed: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    "lasso": lambda seed: Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.01))]),
    "elasticnet": lambda seed: Pipeline([("scaler", StandardScaler()), ("enet", ElasticNet(alpha=0.01))]),
    # Tree-based
    "rf": lambda seed: RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed, n_jobs=1),
    "hgb": lambda seed: HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=seed),
    "gbr": lambda seed: GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=seed),
    "xgboost": None,  # placeholder, requires xgboost
    "lightgbm": None,  # placeholder, requires lightgbm
    # Neural
    "mlp": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=seed)),
    ]),
    "svr": lambda seed: Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0))]),
    # MOE (Confluencia's custom)
    "moe": None,  # special handling below
}

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute paired t-test statistics."""
    from scipy import stats
    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled < 1e-8:
        return 0.0
    return float((a.mean() - b.mean()) / s_pooled)


def interpret_cohens_d(d: float) -> str:
    """Interpret effect size magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(statistic(sample))
    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Sequence-aware splitting
# ---------------------------------------------------------------------------


def sequence_aware_split(
    df: pd.DataFrame,
    seq_col: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by unique sequences: no sequence appears in both train and test."""
    unique_seqs = df[seq_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)
    n_test = max(1, int(len(unique_seqs) * test_ratio))
    test_seqs = set(unique_seqs[:n_test])
    test_idx = np.array([i for i, s in enumerate(df[seq_col]) if s in test_seqs])
    train_idx = np.array([i for i, s in enumerate(df[seq_col]) if s not in test_seqs])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    module: str = "epitope",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix for baseline comparison."""
    if module == "epitope":
        _ensure_path(str(_EPITOPE_DIR))
        from core.features import FeatureSpec, build_feature_matrix
        work = _normalise_epitope_columns(df)
        if "epitope_seq" not in work.columns:
            work["epitope_seq"] = ""
        X, names, env_cols = build_feature_matrix(work, FeatureSpec())
    else:
        _ensure_path(str(_DRUG_DIR))
        from core.features import build_feature_matrix, MixedFeatureSpec
        spec = MixedFeatureSpec(prefer_rdkit=False)
        X, names, env_cols = build_feature_matrix(df, spec)

    # Target
    if "efficacy" in df.columns:
        y = df["efficacy"].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Data must have 'efficacy' column.")
    return X, y, names


# ---------------------------------------------------------------------------
# MOE Regressor (inlined for portability)
# ---------------------------------------------------------------------------


class MOERegressor:
    """Simple MOE for baseline comparison."""

    def __init__(self, experts: List[str] = None, folds: int = 4, seed: int = 42):
        self.experts = experts or ["ridge", "hgb", "rf"]
        self.folds = folds
        self.seed = seed
        self.models_: Dict[str, Any] = {}
        self.weights_: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MOERegressor":
        from sklearn.metrics import mean_squared_error
        n = len(y)
        kf = KFold(n_splits=min(self.folds, max(2, n // 10)), shuffle=True, random_state=self.seed)
        scores = {}
        for name in self.experts:
            if name not in BASELINE_MODELS or BASELINE_MODELS[name] is None:
                continue
            oof = np.zeros(n, dtype=np.float32)
            for tr, va in kf.split(X):
                model = BASELINE_MODELS[name](self.seed)
                model.fit(X[tr], y[tr])
                oof[va] = model.predict(X[va]).astype(np.float32)
            rmse = np.sqrt(mean_squared_error(y, oof))
            scores[name] = rmse
            final_model = BASELINE_MODELS[name](self.seed)
            final_model.fit(X, y)
            self.models_[name] = final_model

        inv = np.array([1.0 / max(scores.get(k, 1e6), 1e-6) for k in self.experts])
        inv = inv / max(inv.sum(), 1e-8)
        self.weights_ = {k: float(w) for k, w in zip(self.experts, inv) if k in self.models_}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0], dtype=np.float32)
        for name, model in self.models_.items():
            w = self.weights_.get(name, 0.0)
            preds += w * model.predict(X).astype(np.float32)
        return preds


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    model_name: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float
    train_time_mean: float
    train_time_std: float
    n_features: int
    fold_scores: Dict[str, List[float]]


def evaluate_model_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> EvaluationResult:
    """Evaluate a single model with repeated k-fold CV."""
    mae_scores, rmse_scores, r2_scores, time_scores = [], [], [], []

    for rep in range(n_repeats):
        kf = KFold(n_splits=min(n_folds, max(2, len(y) // 5)),
                   shuffle=True, random_state=seed + rep)
        for tr, va in kf.split(X):
            if model_name == "moe":
                model = MOERegressor(seed=seed + rep)
            else:
                model = BASELINE_MODELS[model_name](seed + rep)

            t0 = time.time()
            model.fit(X[tr], y[tr])
            elapsed = time.time() - t0

            pred = model.predict(X[va])
            mae_scores.append(mean_absolute_error(y[va], pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y[va], pred)))
            r2_scores.append(r2_score(y[va], pred))
            time_scores.append(elapsed)

    return EvaluationResult(
        model_name=model_name,
        mae_mean=float(np.mean(mae_scores)),
        mae_std=float(np.std(mae_scores)),
        rmse_mean=float(np.mean(rmse_scores)),
        rmse_std=float(np.std(rmse_scores)),
        r2_mean=float(np.mean(r2_scores)),
        r2_std=float(np.std(r2_scores)),
        train_time_mean=float(np.mean(time_scores)),
        train_time_std=float(np.std(time_scores)),
        n_features=int(X.shape[1]),
        fold_scores={
            "mae": mae_scores,
            "rmse": rmse_scores,
            "r2": r2_scores,
            "time": time_scores,
        },
    )


def compare_to_baseline(
    moe_result: EvaluationResult,
    baseline_result: EvaluationResult,
) -> Dict[str, Any]:
    """Compare MOE to a baseline with statistical tests."""
    comparison = {
        "baseline": baseline_result.model_name,
        "moe_mae": moe_result.mae_mean,
        "baseline_mae": baseline_result.mae_mean,
        "mae_improvement": (baseline_result.mae_mean - moe_result.mae_mean) / baseline_result.mae_mean * 100,
        "moe_r2": moe_result.r2_mean,
        "baseline_r2": baseline_result.r2_mean,
    }

    # Paired t-test on fold MAE scores
    moe_mae = np.array(moe_result.fold_scores["mae"])
    base_mae = np.array(baseline_result.fold_scores["mae"])
    comparison["paired_t"] = paired_t_test(moe_mae, base_mae)
    comparison["cohens_d"] = cohens_d(moe_mae, base_mae)
    comparison["effect_size_interpretation"] = interpret_cohens_d(comparison["cohens_d"])

    # Bootstrap CI on MAE difference
    diff = moe_mae - base_mae
    comparison["mae_diff_ci_95"] = list(bootstrap_confidence_interval(diff, seed=42))

    return comparison


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_comparison(
    module: str,
    data_path: str,
    models: List[str] = None,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    output_dir: str = "benchmarks/results",
) -> str:
    """Run baseline comparison for a module."""
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / data_path)

    print(f"Loading data: {data_path}")
    X, y, names = build_features(df, module)
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")

    models = models or ["ridge", "rf", "hgb", "gbr", "mlp", "moe"]
    results: Dict[str, Any] = {}

    for model_name in models:
        if model_name not in BASELINE_MODELS and model_name != "moe":
            print(f"Skipping unknown model: {model_name}")
            continue
        if BASELINE_MODELS.get(model_name) is None and model_name != "moe":
            print(f"Skipping unavailable model: {model_name}")
            continue

        print(f"Evaluating {model_name:15s} ... ", end="", flush=True)
        result = evaluate_model_cv(model_name, X, y, n_folds, n_repeats, seed)
        results[model_name] = {
            "mae": {"mean": result.mae_mean, "std": result.mae_std},
            "rmse": {"mean": result.rmse_mean, "std": result.rmse_std},
            "r2": {"mean": result.r2_mean, "std": result.r2_std},
            "train_time": {"mean": result.train_time_mean, "std": result.train_time_std},
            "n_features": result.n_features,
        }
        print(f"MAE={result.mae_mean:.4f}+/-{result.mae_std:.4f} R2={result.r2_mean:.4f}+/-{result.r2_std:.4f}")

    # Compare MOE to all baselines
    if "moe" in results:
        moe_res = EvaluationResult(
            model_name="moe",
            mae_mean=results["moe"]["mae"]["mean"],
            mae_std=results["moe"]["mae"]["std"],
            rmse_mean=results["moe"]["rmse"]["mean"],
            rmse_std=results["moe"]["rmse"]["std"],
            r2_mean=results["moe"]["r2"]["mean"],
            r2_std=results["moe"]["r2"]["std"],
            train_time_mean=results["moe"]["train_time"]["mean"],
            train_time_std=results["moe"]["train_time"]["std"],
            n_features=results["moe"]["n_features"],
            fold_scores={},
        )
        comparisons = {}
        for model_name in models:
            if model_name == "moe":
                continue
            if model_name not in results:
                continue
            base_res = EvaluationResult(
                model_name=model_name,
                mae_mean=results[model_name]["mae"]["mean"],
                mae_std=results[model_name]["mae"]["std"],
                rmse_mean=results[model_name]["rmse"]["mean"],
                rmse_std=results[model_name]["rmse"]["std"],
                r2_mean=results[model_name]["r2"]["mean"],
                r2_std=results[model_name]["r2"]["std"],
                train_time_mean=results[model_name]["train_time"]["mean"],
                train_time_std=results[model_name]["train_time"]["std"],
                n_features=results[model_name]["n_features"],
                fold_scores={},
            )
            comparisons[model_name] = {
                "baseline": model_name,
                "moe_mae": results["moe"]["mae"]["mean"],
                "baseline_mae": results[model_name]["mae"]["mean"],
                "mae_improvement_pct": (results[model_name]["mae"]["mean"] - results["moe"]["mae"]["mean"])
                                        / max(results[model_name]["mae"]["mean"], 1e-6) * 100,
                "moe_r2": results["moe"]["r2"]["mean"],
                "baseline_r2": results[model_name]["r2"]["mean"],
            }
        results["_moe_vs_baselines"] = comparisons

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baselines_{module}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Confluencia Baseline Comparison")
    parser.add_argument("--module", choices=["epitope", "drug"], required=True)
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--models", nargs="+", default=["ridge", "rf", "hgb", "gbr", "mlp", "moe"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    run_baseline_comparison(
        args.module, args.data, args.models, args.folds, args.repeats, args.seed, args.output,
    )


if __name__ == "__main__":
    main()
