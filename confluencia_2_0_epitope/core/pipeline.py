from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureSpec, build_feature_matrix, ensure_columns
from .moe import MOERegressor, choose_compute_profile
from .sensitivity import neighborhood_importance, numerical_input_gradient, top_features
from .torch_mamba import (
    TorchMambaConfig,
    predict_torch_mamba,
    real_mamba_available,
    sensitivity_torch_mamba,
    torch_available,
    train_torch_mamba,
)


def _build_legacy_regressor(name: str, random_state: int = 42):
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=random_state)
    if name == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=1)
    if name == "ridge":
        return Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=random_state))])
    if name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        early_stopping=True,
                        max_iter=1200,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported legacy backend: {name}")


@dataclass
class EpitopeArtifacts:
    compute_profile: str
    moe_weights: Dict[str, float]
    moe_metrics: Dict[str, float]
    used_proxy_label: bool
    feature_dim: int
    env_cols: list[str]
    model_backend: str
    used_real_mamba: bool


@dataclass
class SensitivityArtifacts:
    sample_index: int
    prediction: float
    top_rows: pd.DataFrame
    neighborhood_rows: pd.DataFrame


def _resolve_label(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    if name not in df.columns:
        return None
    vals = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return vals


def _proxy_objective(work: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    base = 0.25 * work["dose"].to_numpy(dtype=np.float32)
    base += 0.18 * work["freq"].to_numpy(dtype=np.float32)
    base += 0.12 * work["circ_expr"].to_numpy(dtype=np.float32)
    base += 0.10 * work["ifn_score"].to_numpy(dtype=np.float32)

    seq_term = np.zeros((len(work),), dtype=np.float32)
    if X.shape[1] > 0:
        seq_term = 0.35 * X[:, : min(96, X.shape[1])].mean(axis=1).astype(np.float32)

    return (base + seq_term).astype(np.float32)


def run_pipeline(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    sensitivity_sample_idx: int = 0,
    model_backend: str = "torch-mamba",
    torch_cfg: TorchMambaConfig | None = None,
    feature_spec: FeatureSpec | None = None,
) -> Tuple[pd.DataFrame, EpitopeArtifacts, SensitivityArtifacts]:
    work = ensure_columns(df)
    spec = feature_spec if feature_spec is not None else FeatureSpec(use_mhc=True)
    X, feature_names, env_cols = build_feature_matrix(work, spec)

    n = X.shape[0]
    if n == 0:
        empty = work.copy()
        art = EpitopeArtifacts(
            compute_profile="low",
            moe_weights={},
            moe_metrics={},
            used_proxy_label=True,
            feature_dim=int(X.shape[1]),
            env_cols=list(env_cols),
            model_backend="sklearn-moe",
            used_real_mamba=False,
        )
        sens = SensitivityArtifacts(
            sample_index=0,
            prediction=0.0,
            top_rows=pd.DataFrame(columns=["feature", "importance", "grad"]),
            neighborhood_rows=pd.DataFrame(columns=["group", "importance"]),
        )
        return empty, art, sens

    y = _resolve_label(work, "efficacy")
    used_proxy = False
    if y is None:
        y = _proxy_objective(work, X)
        used_proxy = True

    use_torch_backend = model_backend == "torch-mamba" and torch_available()
    use_legacy_backend = model_backend in {"hgb", "gbr", "rf", "ridge", "mlp"}

    if use_torch_backend:
        cfg = torch_cfg or TorchMambaConfig()
        bundle = train_torch_mamba(
            work,
            y,
            env_cols=list(env_cols),
            cfg=cfg,
            prefer_real_mamba=True,
        )
        pred = predict_torch_mamba(bundle, work)
        out = work.copy()
        out["efficacy_pred"] = pred.astype(np.float32)

        residual = np.abs(pred - y).astype(np.float32)
        out["pred_uncertainty"] = residual / max(float(np.std(y)) + 1e-6, 1e-6)

        sid = int(np.clip(sensitivity_sample_idx, 0, max(n - 1, 0)))
        sens = sensitivity_torch_mamba(bundle, out.iloc[sid], sample_index=sid, top_k=20)

        top_records = [{"feature": name, "importance": val, "grad": 0.0} for name, val, _ in sens.token_rows]
        top_records.extend([{"feature": name, "importance": val, "grad": 0.0} for name, val in sens.env_rows])
        top_df = pd.DataFrame(top_records, columns=["feature", "importance", "grad"]).sort_values(
            "importance", ascending=False
        )

        ng_df = (
            pd.DataFrame({"group": list(sens.neighborhood.keys()), "importance": list(sens.neighborhood.values())})
            .sort_values("importance", ascending=False)
        )

        artifacts = EpitopeArtifacts(
            compute_profile=compute_mode,
            moe_weights={},
            moe_metrics={
                "train_loss_last": float(bundle.history.get("train_loss", [0.0])[-1]),
                "val_loss_last": float(bundle.history.get("val_loss", [0.0])[-1]),
            },
            used_proxy_label=used_proxy,
            feature_dim=int(X.shape[1]),
            env_cols=list(env_cols),
            model_backend="torch-mamba",
            used_real_mamba=bundle.used_real_mamba,
        )

        sens_art = SensitivityArtifacts(
            sample_index=sid,
            prediction=float(sens.prediction),
            top_rows=top_df,
            neighborhood_rows=ng_df,
        )
        return out, artifacts, sens_art

    prof = choose_compute_profile(n_samples=n, requested=compute_mode)

    if use_legacy_backend:
        model = _build_legacy_regressor(str(model_backend), random_state=42)
        model.fit(X, y)
        pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
        unc = np.abs(pred - y).astype(np.float32) / max(float(np.std(y)) + 1e-6, 1e-6)

        out = work.copy()
        out["efficacy_pred"] = pred
        out["pred_uncertainty"] = unc

        sid = int(np.clip(sensitivity_sample_idx, 0, max(n - 1, 0)))
        p0, grad = numerical_input_gradient(model, X[sid], eps=1e-3)
        imp = np.abs(grad)

        top = top_features(feature_names, imp, grad, k=20)
        top_df = pd.DataFrame(top, columns=["feature", "importance", "grad"])

        ng = neighborhood_importance(feature_names, imp)
        ng_df = pd.DataFrame({"group": list(ng.keys()), "importance": list(ng.values())}).sort_values("importance", ascending=False)

        mse = float(np.mean((pred - y) ** 2))
        mae = float(np.mean(np.abs(pred - y)))
        denom = max(float(np.sum((y - float(np.mean(y))) ** 2)), 1e-8)
        r2 = float(1.0 - np.sum((pred - y) ** 2) / denom)

        artifacts = EpitopeArtifacts(
            compute_profile=prof.level,
            moe_weights={},
            moe_metrics={"mae": mae, "rmse": float(np.sqrt(mse)), "r2": r2},
            used_proxy_label=used_proxy,
            feature_dim=int(X.shape[1]),
            env_cols=list(env_cols),
            model_backend=str(model_backend),
            used_real_mamba=False,
        )

        sens_art = SensitivityArtifacts(
            sample_index=sid,
            prediction=float(p0),
            top_rows=top_df,
            neighborhood_rows=ng_df,
        )
        return out, artifacts, sens_art

    moe = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds)
    moe.fit(X, y)

    pred = moe.predict(X)
    unc = moe.predict_uncertainty(X)

    out = work.copy()
    out["efficacy_pred"] = pred.astype(np.float32)
    out["pred_uncertainty"] = unc.astype(np.float32)

    sid = int(np.clip(sensitivity_sample_idx, 0, max(n - 1, 0)))
    p0, grad = numerical_input_gradient(moe, X[sid], eps=1e-3)
    imp = np.abs(grad)

    top = top_features(feature_names, imp, grad, k=20)
    top_df = pd.DataFrame(top, columns=["feature", "importance", "grad"])

    ng = neighborhood_importance(feature_names, imp)
    ng_df = pd.DataFrame({"group": list(ng.keys()), "importance": list(ng.values())}).sort_values("importance", ascending=False)

    artifacts = EpitopeArtifacts(
        compute_profile=prof.level,
        moe_weights=moe.explain_weights(),
        moe_metrics=moe.metrics,
        used_proxy_label=used_proxy,
        feature_dim=int(X.shape[1]),
        env_cols=list(env_cols),
        model_backend="sklearn-moe",
        used_real_mamba=False,
    )

    sens_art = SensitivityArtifacts(
        sample_index=sid,
        prediction=float(p0),
        top_rows=top_df,
        neighborhood_rows=ng_df,
    )

    return out, artifacts, sens_art
