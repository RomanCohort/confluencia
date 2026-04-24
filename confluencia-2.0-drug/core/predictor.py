from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

from .featurizer import MoleculeFeatures
from confluencia_shared.training import make_metric_suggestions, make_training_suggestions
from confluencia_shared.metrics import rmse as _rmse
from confluencia_shared.models import ModelName
from confluencia_shared.protocols import PredictableRegressor
try:
    from confluencia_shared.optim.differential_evolution import de_optimize
except Exception:
    from common.optim.differential_evolution import de_optimize


@dataclass
class DrugModelBundle:
    model: PredictableRegressor
    env_cols: List[str]
    smiles_col: str
    target_col: str
    env_medians: Dict[str, float]
    feature_names: List[str]
    created_at: str
    version: int = 1
    featurizer_version: int = 2
    radius: int = 2
    n_bits: int = 2048


def build_model(
    model_name: ModelName,
    random_state: int = 42,
    *,
    mlp_alpha: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_patience: int = 10,
    ridge_alpha: float = 1.0,
    hgb_l2: float = 0.0,
) -> PredictableRegressor:
    """Build a regressor with drug-optimized settings.

    Uses shared ModelFactory with drug preset for consistency.
    """
    from confluencia_shared.models import ModelConfig, ModelFactory

    config = ModelConfig.for_drug()
    # Override with custom parameters if provided
    config.mlp_alpha = mlp_alpha
    config.mlp_early_stopping = mlp_early_stopping
    config.mlp_patience = mlp_patience
    config.ridge_alpha = ridge_alpha
    config.hgb_l2 = hgb_l2

    factory = ModelFactory(config)
    return factory.build(model_name, random_state)


def infer_env_cols(df, smiles_col: str, target_col: str, env_cols: Optional[Sequence[str]] = None) -> List[str]:
    if env_cols is not None and len(env_cols) > 0:
        return list(env_cols)

    numeric_cols = [
        c for c in df.columns if c not in (smiles_col, target_col) and is_numeric_dtype(df[c])
    ]
    return numeric_cols


def make_xy(
    df,
    smiles_col: str,
    target_col: str,
    env_cols: List[str],
    featurizer: Optional[MoleculeFeatures] = None,
    env_medians: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[str], np.ndarray]:
    if featurizer is None:
        featurizer = MoleculeFeatures(version=2)

    df = df.copy()
    df[smiles_col] = df[smiles_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    smiles_list = df[smiles_col].astype(str).tolist()
    mol_x, valids = featurizer.transform_many(smiles_list)

    env_df = df[env_cols].copy() if env_cols else df.iloc[:, 0:0].copy()

    if env_medians is None:
        env_medians = {c: float(pd.to_numeric(env_df[c], errors="coerce").median()) for c in env_cols}

    for c in env_cols:
        env_df[c] = pd.to_numeric(env_df[c], errors="coerce")
        env_df[c] = env_df[c].fillna(env_medians[c])

    env_x = env_df.to_numpy(dtype=np.float32) if env_cols else np.zeros((len(df), 0), dtype=np.float32)

    x = np.concatenate([mol_x, env_x], axis=1)
    y = df[target_col].to_numpy(dtype=np.float32)

    feature_names = featurizer.feature_names() + [f"env_{c}" for c in env_cols]
    return x, y, env_medians, feature_names, valids


def train_bundle(
    df,
    smiles_col: str = "smiles",
    target_col: str = "efficacy",
    env_cols: Optional[List[str]] = None,
    model_name: ModelName = "gbr",
    test_size: float = 0.2,
    random_state: int = 42,
    featurizer_version: int = 2,
    radius: int = 2,
    n_bits: int = 2048,
    drop_invalid_smiles: bool = True,
    mlp_alpha: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_patience: int = 10,
    ridge_alpha: float = 1.0,
    hgb_l2: float = 0.0,
) -> Tuple[DrugModelBundle, Dict[str, float]]:
    if df is None or len(df) == 0:
        raise ValueError("训练数据为空，请检查上传的 CSV 是否包含有效样本。")
    logger.info(f"Training drug model: n_samples={len(df)}, model={model_name}")
    df = df.copy()
    df[smiles_col] = df[smiles_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()
    env_cols = infer_env_cols(df, smiles_col=smiles_col, target_col=target_col, env_cols=env_cols)

    featurizer = MoleculeFeatures(version=int(featurizer_version), radius=int(radius), n_bits=int(n_bits))
    x, y, env_medians, feature_names, valids = make_xy(
        df,
        smiles_col=smiles_col,
        target_col=target_col,
        env_cols=list(env_cols),
        featurizer=featurizer,
        env_medians=None,
    )

    invalid_smiles = int((~valids).sum())
    if bool(drop_invalid_smiles) and invalid_smiles > 0:
        keep = valids.astype(bool)
        x = x[keep]
        y = y[keep]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=float(test_size), random_state=int(random_state))

    model = build_model(
        model_name=model_name,
        random_state=int(random_state),
        mlp_alpha=float(mlp_alpha),
        mlp_early_stopping=bool(mlp_early_stopping),
        mlp_patience=int(mlp_patience),
        ridge_alpha=float(ridge_alpha),
        hgb_l2=float(hgb_l2),
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    history: Dict[str, List[float]] = {}
    try:
        if hasattr(model, "named_steps") and "mlp" in model.named_steps:
            mlp = model.named_steps.get("mlp")
            if hasattr(mlp, "loss_curve_"):
                history["train_loss"] = list(getattr(mlp, "loss_curve_", []))
            if hasattr(mlp, "validation_scores_"):
                val_scores = list(getattr(mlp, "validation_scores_", []))
                history["val_loss"] = [float(1.0 - v) for v in val_scores]
    except Exception:
        history = {}

    metrics = {
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": _rmse(y_val, y_pred),
        "r2": float(r2_score(y_val, y_pred)),
        "explained_variance": float(explained_variance_score(y_val, y_pred)),
        "max_error": float(max_error(y_val, y_pred)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(x.shape[1]),
        "invalid_smiles": int(invalid_smiles),
        "dropped_invalid_smiles": int(invalid_smiles if bool(drop_invalid_smiles) else 0),
        "history": history,
        "suggestions": make_training_suggestions(history) if history else make_metric_suggestions({
            "r2": float(r2_score(y_val, y_pred)),
            "rmse": _rmse(y_val, y_pred),
        }),
        "feature_importances": list(model.feature_importances_) if hasattr(model, "feature_importances_") else None,
        "y_val": y_val.tolist(),
        "y_pred": y_pred.tolist(),
    }

    bundle = DrugModelBundle(
        model=model,
        env_cols=list(env_cols),
        smiles_col=str(smiles_col),
        target_col=str(target_col),
        env_medians=env_medians,
        feature_names=feature_names,
        created_at=datetime.now().isoformat(timespec="seconds"),
        featurizer_version=int(featurizer_version),
        radius=int(radius),
        n_bits=int(n_bits),
    )

    logger.info(
        f"Drug training complete: mae={metrics['mae']:.4f}, "
        f"rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}"
    )

    return bundle, metrics


def cross_validate(
    df,
    *,
    smiles_col: str = "smiles",
    target_col: str = "efficacy",
    env_cols: Optional[List[str]] = None,
    model_name: ModelName = "gbr",
    n_splits: int = 5,
    random_state: int = 42,
    featurizer_version: int = 2,
    radius: int = 2,
    n_bits: int = 2048,
    drop_invalid_smiles: bool = True,
) -> Dict[str, object]:
    """KFold cross-validation for drug predictor.

    Returns a dict containing per-fold metrics and mean/std summaries.
    """

    env_cols = infer_env_cols(df, smiles_col=smiles_col, target_col=target_col, env_cols=env_cols)
    featurizer = MoleculeFeatures(version=int(featurizer_version), radius=int(radius), n_bits=int(n_bits))

    x_all, y_all, _, feature_names, valids = make_xy(
        df,
        smiles_col=smiles_col,
        target_col=target_col,
        env_cols=list(env_cols),
        featurizer=featurizer,
        env_medians=None,
    )

    invalid_smiles = int((~valids).sum())
    if bool(drop_invalid_smiles) and invalid_smiles > 0:
        keep = valids.astype(bool)
        x_all = x_all[keep]
        y_all = y_all[keep]

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))

    per_fold: List[Dict[str, float]] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_all), start=1):
        x_tr, x_va = x_all[tr_idx], x_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        model = build_model(model_name=model_name, random_state=int(random_state) + fold)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_va)

        per_fold.append(
            {
                "fold": float(fold),
                "mae": float(mean_absolute_error(y_va, y_pred)),
                "rmse": _rmse(y_va, y_pred),
                "r2": float(r2_score(y_va, y_pred)),
                "n_train": float(len(y_tr)),
                "n_val": float(len(y_va)),
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([d[key] for d in per_fold]))

    def _std(key: str) -> float:
        return float(np.std([d[key] for d in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0

    summary = {
        "mae_mean": _mean("mae"),
        "mae_std": _std("mae"),
        "rmse_mean": _mean("rmse"),
        "rmse_std": _std("rmse"),
        "r2_mean": _mean("r2"),
        "r2_std": _std("r2"),
        "n_samples": int(len(y_all)),
        "n_features": int(x_all.shape[1]),
        "invalid_smiles": int(invalid_smiles),
        "dropped_invalid_smiles": int(invalid_smiles if bool(drop_invalid_smiles) else 0),
    }

    return {
        "params": {
            "model": str(model_name),
            "smiles_col": str(smiles_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "featurizer_version": int(featurizer_version),
            "radius": int(radius),
            "n_bits": int(n_bits),
            "drop_invalid_smiles": bool(drop_invalid_smiles),
        },
        "feature_names": feature_names,
        "per_fold": per_fold,
        "summary": summary,
    }


def predict_one(
    bundle: DrugModelBundle,
    smiles: str,
    env_params: Optional[Dict[str, float]] = None,
) -> float:
    featurizer = MoleculeFeatures(version=int(bundle.featurizer_version), radius=int(bundle.radius), n_bits=int(bundle.n_bits))
    mol_x, ok = featurizer.transform_one(smiles)

    env_params = env_params or {}
    env_vec = []
    for c in bundle.env_cols:
        if c in env_params:
            env_vec.append(float(env_params[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.array(env_vec, dtype=np.float32) if bundle.env_cols else np.zeros((0,), dtype=np.float32)
    x = np.concatenate([mol_x, env_x], axis=0).reshape(1, -1)

    y_pred = bundle.model.predict(x)
    # If SMILES invalid, still return prediction on zeros; callers can validate separately if desired.
    return float(np.asarray(y_pred).reshape(-1)[0])


def suggest_env_by_de_drug(
    bundle: DrugModelBundle,
    smiles: str,
    env_bounds: Sequence[Tuple[float, float]],
    maximize: bool = True,
    de_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    使用差分进化搜索药物模型的数值环境参数以最大化/最小化预测值。

    Returns: (best_env_vector, best_prediction)
    """
    if de_kwargs is None:
        de_kwargs = {"pop_size": max(20, 5 * len(env_bounds)), "max_iter": 100, "F": 0.8, "CR": 0.9}

    featurizer = MoleculeFeatures(version=int(bundle.featurizer_version), radius=int(bundle.radius), n_bits=int(bundle.n_bits))
    mol_x, ok = featurizer.transform_one(smiles)

    def objective(env_vec: np.ndarray) -> float:
        x = np.concatenate([mol_x, env_vec.reshape(-1).astype(np.float32)], axis=0).reshape(1, -1)
        pred = bundle.model.predict(x)
        return float(np.asarray(pred).reshape(-1)[0])

    best_env, best_val = de_optimize(objective, env_bounds, maximize=maximize, **de_kwargs)
    return np.asarray(best_env, dtype=float), float(best_val)
