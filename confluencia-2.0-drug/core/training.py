from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .legacy_algorithms import LegacyAlgorithmConfig
from .pipeline import ConfluenciaArtifacts, ConfluenciaModelBundle, predict_pipeline_with_bundle, run_pipeline, train_pipeline_bundle
from confluencia_shared.metrics import reg_metrics as _shared_reg_metrics


@dataclass
class DrugTrainingReport:
    sample_count: int
    used_labels: Dict[str, bool]
    metrics: Dict[str, float]


@dataclass
class DrugTrainedModel:
    bundle: ConfluenciaModelBundle
    artifacts: ConfluenciaArtifacts
    metadata: Dict[str, str] = field(default_factory=dict)


MODEL_MAGIC = b"CF2_DRUG_MODEL_V1"


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return _shared_reg_metrics(y_true, y_pred, prefix=prefix)


def _build_training_report(df: pd.DataFrame, result_df: pd.DataFrame) -> DrugTrainingReport:
    metric_map: Dict[str, float] = {}
    used_labels: Dict[str, bool] = {}

    targets = {
        "efficacy": "efficacy_pred",
        "target_binding": "target_binding_pred",
        "immune_activation": "immune_activation_pred",
        "immune_cell_activation": "immune_cell_activation_pred",
        "inflammation_risk": "inflammation_risk_pred",
        "toxicity_risk": "toxicity_risk_pred",
    }

    for y_col, pred_col in targets.items():
        has_label = y_col in df.columns and y_col in result_df.columns and pred_col in result_df.columns
        used_labels[y_col] = bool(has_label)
        if not has_label:
            metric_map.update(_reg_metrics(np.array([]), np.array([]), y_col))
            continue

        y_true = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_pred = pd.to_numeric(result_df[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        metric_map.update(_reg_metrics(y_true, y_pred, y_col))

    return DrugTrainingReport(
        sample_count=int(len(result_df)),
        used_labels=used_labels,
        metrics=metric_map,
    )


def train_drug_model(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    model_backend: str = "moe",
    dynamics_model: str = "ctm",
    legacy_cfg: LegacyAlgorithmConfig | None = None,
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
    tune_hyperparams: bool = False,
    tune_strategy: str = "random",
    tune_n_iter: int = 20,
    tune_cv: int = 3,
) -> DrugTrainedModel:
    _ = legacy_cfg
    if str(model_backend).lower() != "moe":
        raise ValueError("Split train/predict currently supports model_backend='moe' only")

    bundle, artifacts = train_pipeline_bundle(
        df,
        compute_mode=compute_mode,
        dynamics_model=dynamics_model,
        adaptive_enabled=adaptive_enabled,
        adaptive_strength=adaptive_strength,
        tune_hyperparams=tune_hyperparams,
        tune_strategy=tune_strategy,
        tune_n_iter=tune_n_iter,
        tune_cv=tune_cv,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_backend": "moe",
        "compute_mode": str(compute_mode),
        "compute_profile": str(artifacts.compute_profile),
        "dynamics_model": str(artifacts.dynamics_model),
        "sample_count": str(int(len(df))),
        "smiles_backend": str(artifacts.smiles_backend),
        "ctm_param_source": str(artifacts.ctm_param_source),
        "adaptive_enabled": str(bool(artifacts.adaptive_enabled)).lower(),
        "adaptive_strength": f"{float(artifacts.adaptive_strength):.4f}",
    }
    return DrugTrainedModel(bundle=bundle, artifacts=artifacts, metadata=metadata)


def predict_drug_with_model(
    df: pd.DataFrame,
    trained_model: DrugTrainedModel,
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, ConfluenciaArtifacts, DrugTrainingReport]:
    result_df, curve_df, artifacts = predict_pipeline_with_bundle(
        df=df,
        bundle=trained_model.bundle,
        adaptive_enabled=adaptive_enabled,
        adaptive_strength=adaptive_strength,
    )
    report = _build_training_report(df, result_df)
    return result_df, curve_df, artifacts, report


def export_drug_model_bytes(trained_model: DrugTrainedModel) -> bytes:
    payload = {
        "magic": MODEL_MAGIC,
        "model": trained_model,
    }
    return gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)


def import_drug_model_bytes(payload: bytes, allow_unsafe_deserialization: bool = False) -> DrugTrainedModel:
    if not allow_unsafe_deserialization:
        raise ValueError(
            "Model import is disabled by default for security reasons. "
            "Set allow_unsafe_deserialization=True only for trusted files."
        )

    try:
        obj = pickle.loads(gzip.decompress(payload))
    except Exception as e:
        raise ValueError(f"Invalid model payload: {e}") from e

    if not isinstance(obj, dict) or obj.get("magic") != MODEL_MAGIC:
        raise ValueError("Unsupported model file format")

    model = obj.get("model")
    if not isinstance(model, DrugTrainedModel):
        raise ValueError("Model payload type mismatch")

    if not hasattr(model, "metadata") or not isinstance(model.metadata, dict):
        model.metadata = {}

    if not model.metadata:
        model.metadata = {
            "model_backend": str(getattr(model.artifacts, "model_backend", "moe")),
            "compute_profile": str(getattr(model.artifacts, "compute_profile", "unknown")),
            "dynamics_model": str(getattr(model.artifacts, "dynamics_model", "ctm")),
            "smiles_backend": str(getattr(model.artifacts, "smiles_backend", "unknown")),
            "ctm_param_source": str(getattr(model.artifacts, "ctm_param_source", "unknown")),
        }
    return model


def get_drug_model_metadata(trained_model: DrugTrainedModel) -> Dict[str, str]:
    md = dict(trained_model.metadata)
    if "model_backend" not in md:
        md["model_backend"] = str(getattr(trained_model.artifacts, "model_backend", "moe"))
    if "compute_profile" not in md:
        md["compute_profile"] = str(getattr(trained_model.artifacts, "compute_profile", "unknown"))
    if "dynamics_model" not in md:
        md["dynamics_model"] = str(getattr(trained_model.artifacts, "dynamics_model", "ctm"))
    return md


def train_and_predict_drug(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    model_backend: str = "moe",
    dynamics_model: str = "ctm",
    legacy_cfg: LegacyAlgorithmConfig | None = None,
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, ConfluenciaArtifacts, DrugTrainingReport]:
    if str(model_backend).lower() == "moe":
        trained = train_drug_model(
            df=df,
            compute_mode=compute_mode,
            model_backend=model_backend,
            dynamics_model=dynamics_model,
            legacy_cfg=legacy_cfg,
            adaptive_enabled=adaptive_enabled,
            adaptive_strength=adaptive_strength,
        )
        return predict_drug_with_model(
            df=df,
            trained_model=trained,
            adaptive_enabled=adaptive_enabled,
            adaptive_strength=adaptive_strength,
        )

    result_df, curve_df, artifacts = run_pipeline(
        df,
        compute_mode=compute_mode,
        model_backend=model_backend,
        dynamics_model=dynamics_model,
        legacy_cfg=legacy_cfg,
        adaptive_enabled=adaptive_enabled,
        adaptive_strength=adaptive_strength,
    )

    report = _build_training_report(df, result_df)
    return result_df, curve_df, artifacts, report
