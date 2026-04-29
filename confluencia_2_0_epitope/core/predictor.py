from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .featurizer import SequenceFeatures
from confluencia_shared.training import make_metric_suggestions, make_training_suggestions
from confluencia_shared.metrics import rmse as _rmse
from confluencia_shared.models import ModelName
from confluencia_shared.protocols import PredictableRegressor
try:
    from confluencia_shared.optim.differential_evolution import de_optimize
except Exception:
    from common.optim.differential_evolution import de_optimize


def _reg_metrics(y_true, y_pred):
    from confluencia_shared.metrics import reg_metrics
    return reg_metrics(y_true, y_pred)


@dataclass
class EpitopeModelBundle:
    model: PredictableRegressor
    env_cols: List[str]
    sequence_col: str
    target_col: str
    env_medians: Dict[str, float]
    feature_names: List[str]
    created_at: str
    version: int = 1
    featurizer_version: int = 1


def build_model(
    model_name: ModelName,
    random_state: int = 42,
    *,
    mlp_alpha: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_patience: int = 10,
    sgd_alpha: float = 1e-4,
    sgd_l1_ratio: float = 0.15,
    sgd_early_stopping: bool = True,
    hgb_l2: float = 0.0,
):
    """Build a regressor with epitope-optimized settings.

    Uses shared ModelFactory with epitope preset for consistency.
    """
    from confluencia_shared.models import ModelConfig, ModelFactory

    config = ModelConfig.for_epitope()
    # Override with custom parameters if provided
    config.mlp_alpha = mlp_alpha
    config.mlp_early_stopping = mlp_early_stopping
    config.mlp_patience = mlp_patience
    config.sgd_alpha = sgd_alpha
    config.sgd_l1_ratio = sgd_l1_ratio
    config.sgd_early_stopping = sgd_early_stopping
    config.hgb_l2 = hgb_l2

    factory = ModelFactory(config)
    return factory.build(model_name, random_state)


def infer_env_cols(df, sequence_col: str, target_col: str, env_cols: Optional[Sequence[str]] = None) -> List[str]:
    if env_cols is not None and len(env_cols) > 0:
        return list(env_cols)

    # auto: all numeric columns excluding target
    numeric_cols = [
        c
        for c in df.columns
        if c not in (sequence_col, target_col) and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def make_xy(
    df,
    sequence_col: str,
    target_col: str,
    env_cols: List[str],
    featurizer: Optional[SequenceFeatures] = None,
    env_medians: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[str]]:
    if featurizer is None:
        featurizer = SequenceFeatures(version=1)

    df = df.copy()
    df[sequence_col] = df[sequence_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    sequences = df[sequence_col].astype(str).tolist()
    seq_x = featurizer.transform_many(sequences)

    env_df = df[env_cols].copy() if env_cols else df.iloc[:, 0:0].copy()

    if env_medians is None:
        env_medians = {c: float(pd.to_numeric(env_df[c], errors="coerce").median()) for c in env_cols}

    for c in env_cols:
        env_df[c] = pd.to_numeric(env_df[c], errors="coerce")
        env_df[c] = env_df[c].fillna(env_medians[c])

    env_x = env_df.to_numpy(dtype=np.float32) if env_cols else np.zeros((len(df), 0), dtype=np.float32)

    x = np.concatenate([seq_x, env_x], axis=1)
    y = df[target_col].to_numpy(dtype=np.float32)

    feature_names = featurizer.feature_names() + [f"env_{c}" for c in env_cols]
    return x, y, env_medians, feature_names


def train_bundle(
    df,
    sequence_col: str,
    target_col: str,
    env_cols: List[str],
    model_name: ModelName = "hgb",
    test_size: float = 0.2,
    random_state: int = 42,
    featurizer_version: int = 2,
    mlp_alpha: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_patience: int = 10,
    sgd_alpha: float = 1e-4,
    sgd_l1_ratio: float = 0.15,
    sgd_early_stopping: bool = True,
    hgb_l2: float = 0.0,
    sample_weight_col: Optional[str] = None,
) -> Tuple[EpitopeModelBundle, Dict[str, float]]:
    if df is None or len(df) == 0:
        raise ValueError("训练数据为空，请检查上传的 CSV 是否包含有效样本。")
    df = df.copy()
    df[sequence_col] = df[sequence_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    # Prepare sample weights if provided (must be aligned with filtered df)
    sample_weights: Optional[np.ndarray]
    if sample_weight_col and sample_weight_col in df.columns:
        sample_weights = pd.to_numeric(df[sample_weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    else:
        sample_weights = None

    x, y, env_medians, feature_names = make_xy(
        df,
        sequence_col=sequence_col,
        target_col=target_col,
        env_cols=env_cols,
        featurizer=SequenceFeatures(version=int(featurizer_version)),
        env_medians=None,
    )

    if x.shape[0] == 0:
        raise ValueError("训练样本数为 0，请检查序列列是否为空或列名是否正确。")

    # Split arrays; include sample weights in the same split if present
    if sample_weights is not None:
        x_train, x_val, y_train, y_val, sw_train, sw_val = train_test_split(
            x,
            y,
            sample_weights,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    model = build_model(
        model_name=model_name,
        random_state=random_state,
        mlp_alpha=float(mlp_alpha),
        mlp_early_stopping=bool(mlp_early_stopping),
        mlp_patience=int(mlp_patience),
        sgd_alpha=float(sgd_alpha),
        sgd_l1_ratio=float(sgd_l1_ratio),
        sgd_early_stopping=bool(sgd_early_stopping),
        hgb_l2=float(hgb_l2),
    )
    # Fit model with sample weights if available. For Pipeline models, forward
    # the sample weight to the final estimator via fit params (e.g. "mlp__sample_weight").
    try:
        if sample_weights is not None:
            if hasattr(model, "named_steps") and isinstance(model, Pipeline):
                # Get name of last step
                last_name = list(model.named_steps.keys())[-1]
                fit_kwargs = {f"{last_name}__sample_weight": sw_train}
                model.fit(x_train, y_train, **fit_kwargs)
            else:
                model.fit(x_train, y_train, sample_weight=sw_train)
        else:
            model.fit(x_train, y_train)
    except TypeError:
        # Some estimators/pipeline combos may not accept sample_weight; fall back
        model.fit(x_train, y_train)

    y_pred = np.asarray(model.predict(x_val)).reshape(-1)
    history: Dict[str, List[float]] = {}
    try:
        if hasattr(model, "named_steps") and "mlp" in model.named_steps:
            mlp = model.named_steps.get("mlp")
            if hasattr(mlp, "loss_curve_"):
                history["train_loss"] = list(getattr(mlp, "loss_curve_", []))
            if hasattr(mlp, "validation_scores_"):
                val_scores = list(getattr(mlp, "validation_scores_", []))
                history["val_loss"] = [float(1.0 - v) for v in val_scores]
        if hasattr(model, "named_steps") and "sgd" in model.named_steps:
            sgd = model.named_steps.get("sgd")
            if hasattr(sgd, "loss_function") and hasattr(sgd, "t_"):
                history.setdefault("train_loss", [])
    except Exception:
        history = {}

    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = list(model.feature_importances_)
    elif hasattr(model, "coef_"):
        feature_importances = list(np.abs(model.coef_))
    elif hasattr(model, "named_steps") and "sgd" in model.named_steps:
        feature_importances = list(np.abs(model.named_steps["sgd"].coef_))

    metrics = {
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": _rmse(y_val, y_pred),
        "r2": float(r2_score(y_val, y_pred)),
        "explained_variance": float(explained_variance_score(y_val, y_pred)),
        "max_error": float(max_error(y_val, y_pred)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(x.shape[1]),
        "feature_importances": feature_importances,
        "history": history,
        "suggestions": make_training_suggestions(history) if history else make_metric_suggestions({
            "r2": float(r2_score(y_val, y_pred)),
            "rmse": _rmse(y_val, y_pred),
        }),
        "y_val": y_val.tolist(),
        "y_pred": y_pred.tolist(),
    }

    bundle = EpitopeModelBundle(
        model=model,
        env_cols=list(env_cols),
        sequence_col=sequence_col,
        target_col=target_col,
        env_medians=env_medians,
        feature_names=feature_names,
        created_at=datetime.now().isoformat(timespec="seconds"),
        featurizer_version=int(featurizer_version),
    )

    return bundle, metrics


def predict_one(
    bundle: EpitopeModelBundle,
    sequence: str,
    env_params: Optional[Dict[str, float]] = None,
) -> float:
    feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
    featurizer = SequenceFeatures(version=feat_v)
    seq_x = featurizer.transform_one(sequence).reshape(1, -1)

    env_params = env_params or {}
    env_vec = []
    for c in bundle.env_cols:
        if c in env_params:
            env_vec.append(float(env_params[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.array(env_vec, dtype=np.float32).reshape(1, -1) if bundle.env_cols else np.zeros((1, 0), dtype=np.float32)
    x = np.concatenate([seq_x, env_x], axis=1)

    y_pred = bundle.model.predict(x)
    return float(np.asarray(y_pred).reshape(-1)[0])


def suggest_env_by_de_epitope(
    bundle: EpitopeModelBundle,
    sequence: str,
    env_bounds: Sequence[Tuple[float, float]],
    maximize: bool = True,
    de_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    使用差分进化搜索表位模型的数值环境参数以最大化/最小化预测值。

    Returns: (best_env_vector, best_prediction)
    """
    if not bundle.env_cols:
        raise ValueError("模型没有环境变量（bundle.env_cols 为空）")
    if de_kwargs is None:
        de_kwargs = {"pop_size": max(20, 5 * len(env_bounds)), "max_iter": 100, "F": 0.8, "CR": 0.9}

    feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
    featurizer = SequenceFeatures(version=feat_v)
    seq_x = featurizer.transform_one(sequence).reshape(1, -1)

    def objective(env_vec: np.ndarray) -> float:
        x = np.concatenate([seq_x, env_vec.reshape(1, -1).astype(np.float32)], axis=1)
        pred = bundle.model.predict(x)
        return float(np.asarray(pred).reshape(-1)[0])

    best_env, best_val = de_optimize(objective, env_bounds, maximize=maximize, **de_kwargs)
    return np.asarray(best_env, dtype=float), float(best_val)


def cross_validate(
    df,
    *,
    sequence_col: str,
    target_col: str,
    env_cols: Optional[List[str]] = None,
    model_name: ModelName = "hgb",
    n_splits: int = 5,
    random_state: int = 42,
    featurizer_version: int = 2,
) -> Dict[str, object]:
    """KFold cross-validation for epitope predictor."""

    env_cols = infer_env_cols(df, sequence_col=sequence_col, target_col=target_col, env_cols=env_cols)

    x, y, _, feature_names = make_xy(
        df,
        sequence_col=sequence_col,
        target_col=target_col,
        env_cols=list(env_cols),
        featurizer=SequenceFeatures(version=int(featurizer_version)),
        env_medians=None,
    )

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    per_fold: List[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(x), start=1):
        x_tr, x_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = build_model(model_name=model_name, random_state=int(random_state) + fold)
        model.fit(x_tr, y_tr)
        y_pred = np.asarray(model.predict(x_va)).reshape(-1)

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
        "n_samples": int(len(y)),
        "n_features": int(x.shape[1]),
    }

    return {
        "params": {
            "model": str(model_name),
            "sequence_col": str(sequence_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "featurizer_version": int(featurizer_version),
        },
        "feature_names": feature_names,
        "per_fold": per_fold,
        "summary": summary,
    }
