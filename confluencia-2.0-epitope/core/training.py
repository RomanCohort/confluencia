from __future__ import annotations

import hashlib
import io
import json
import logging
import pickle
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from confluencia_shared.utils.logging import get_logger
from confluencia_shared.metrics import reg_metrics as _shared_reg_metrics
from confluencia_shared.data_utils import resolve_label as _resolve_label

logger = get_logger(__name__)
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    FEATURE_SCHEMA_VERSION,
    KMER_HASH_VERSION,
    FeatureSpec,
    build_feature_matrix,
    ensure_columns,
    feature_schema_id,
)
from .moe import MOERegressor, choose_compute_profile
from .pipeline import EpitopeArtifacts, SensitivityArtifacts
from .report_template import generate_experiment_report, save_report_csv
from .sensitivity import neighborhood_importance, numerical_input_gradient, top_features
from .torch_mamba import (
    MambaSequenceRegressor,
    TorchMambaBundle,
    TorchMambaConfig,
    predict_torch_mamba,
    sensitivity_torch_mamba,
    torch,
    torch_available,
    train_torch_mamba,
)


@dataclass
class EpitopeTrainingReport:
    sample_count: int
    used_label: bool
    metrics: Dict[str, float]
    model_backend: str
    hyperparam_tuning: Dict[str, Any] | None = None  # tuning results if enabled


@dataclass
class EpitopeModelBundle:
    model_backend: str
    compute_profile: str
    model: Any
    feature_names: list[str]
    feature_dim: int
    env_cols: list[str]
    used_proxy_label: bool
    y_std: float
    used_real_mamba: bool = False
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_schema_id: str = ""
    kmer_hash_version: str = KMER_HASH_VERSION
    moe_weights: Dict[str, float] = field(default_factory=dict)
    moe_metrics: Dict[str, float] = field(default_factory=dict)


def _bundle_common_meta(model_bundle: EpitopeModelBundle) -> dict[str, Any]:
    return {
        "model_backend": str(model_bundle.model_backend),
        "compute_profile": str(model_bundle.compute_profile),
        "feature_names": list(model_bundle.feature_names),
        "feature_dim": int(model_bundle.feature_dim),
        "env_cols": list(model_bundle.env_cols),
        "used_proxy_label": bool(model_bundle.used_proxy_label),
        "y_std": float(model_bundle.y_std),
        "used_real_mamba": bool(model_bundle.used_real_mamba),
        "feature_schema_version": str(model_bundle.feature_schema_version),
        "feature_schema_id": str(model_bundle.feature_schema_id),
        "kmer_hash_version": str(model_bundle.kmer_hash_version),
        "moe_weights": {str(k): float(v) for k, v in model_bundle.moe_weights.items()},
        "moe_metrics": {str(k): float(v) for k, v in model_bundle.moe_metrics.items()},
    }


def _rebuild_bundle_from_meta(meta: dict[str, Any], model_obj: Any) -> EpitopeModelBundle:
    return EpitopeModelBundle(
        model_backend=str(meta.get("model_backend", "unknown")),
        compute_profile=str(meta.get("compute_profile", "auto")),
        model=model_obj,
        feature_names=[str(x) for x in meta.get("feature_names", [])],
        feature_dim=int(meta.get("feature_dim", 0)),
        env_cols=[str(x) for x in meta.get("env_cols", [])],
        used_proxy_label=bool(meta.get("used_proxy_label", False)),
        y_std=float(meta.get("y_std", 1.0)),
        used_real_mamba=bool(meta.get("used_real_mamba", False)),
        feature_schema_version=str(meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
        feature_schema_id=str(meta.get("feature_schema_id", feature_schema_id(FeatureSpec()))),
        kmer_hash_version=str(meta.get("kmer_hash_version", KMER_HASH_VERSION)),
        moe_weights={str(k): float(v) for k, v in dict(meta.get("moe_weights", {})).items()},
        moe_metrics={str(k): float(v) for k, v in dict(meta.get("moe_metrics", {})).items()},
    )


def export_epitope_model_bytes(model_bundle: EpitopeModelBundle) -> bytes:
    """Serialize a trained model bundle into a portable zip blob for download.

    Safe export is currently implemented for `ridge` and `torch-mamba` backends.
    """
    meta = {
        "format": "confluencia-epitope-model",
        "version": 2,
        **_bundle_common_meta(model_bundle),
    }

    blob = io.BytesIO()
    with zipfile.ZipFile(blob, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if model_bundle.model_backend == "ridge":
            model = model_bundle.model
            if not isinstance(model, Pipeline) or "scaler" not in model.named_steps or "ridge" not in model.named_steps:
                raise ValueError("当前 ridge 模型结构不受支持，无法进行安全导出。")

            scaler = model.named_steps["scaler"]
            ridge = model.named_steps["ridge"]
            payload = {
                "type": "ridge-pipeline-v1",
                "scaler": {
                    "mean": np.asarray(getattr(scaler, "mean_", []), dtype=np.float64).tolist(),
                    "scale": np.asarray(getattr(scaler, "scale_", []), dtype=np.float64).tolist(),
                    "var": np.asarray(getattr(scaler, "var_", []), dtype=np.float64).tolist(),
                    "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
                    "n_samples_seen": int(getattr(scaler, "n_samples_seen_", 0)),
                },
                "ridge": {
                    "alpha": float(getattr(ridge, "alpha", 1.0)),
                    "coef": np.asarray(getattr(ridge, "coef_", []), dtype=np.float64).tolist(),
                    "intercept": float(getattr(ridge, "intercept_", 0.0)),
                    "n_features_in": int(getattr(ridge, "n_features_in_", 0)),
                },
            }
            meta["serialization"] = "safe-ridge-v1"
            zf.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
            zf.writestr("safe_model.json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        elif model_bundle.model_backend == "torch-mamba":
            torch_bundle = model_bundle.model
            if not isinstance(torch_bundle, TorchMambaBundle):
                raise ValueError("torch-mamba 模型对象类型不匹配，无法导出。")

            state = torch_bundle.model.state_dict()
            state_buf = io.BytesIO()
            np.savez_compressed(state_buf, **{k: v.detach().cpu().numpy() for k, v in state.items()})

            payload = {
                "type": "torch-mamba-v1",
                "env_cols": list(torch_bundle.env_cols),
                "env_mean": np.asarray(torch_bundle.env_mean, dtype=np.float32).tolist(),
                "env_std": np.asarray(torch_bundle.env_std, dtype=np.float32).tolist(),
                "max_len": int(torch_bundle.max_len),
                "history": {str(k): [float(x) for x in v] for k, v in dict(torch_bundle.history).items()},
                "used_real_mamba": bool(torch_bundle.used_real_mamba),
                "cfg": asdict(torch_bundle.model.cfg),
                "env_dim": int(torch_bundle.model.env_dim),
            }
            meta["serialization"] = "safe-torch-state-npz-v1"
            zf.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
            zf.writestr("safe_model.json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            zf.writestr("torch_state.npz", state_buf.getvalue())
        else:
            raise ValueError(
                f"当前后端 {model_bundle.model_backend} 暂不支持安全导出。"
                "请改用 ridge 或 torch-mamba 后端进行模型导出。"
            )
    return blob.getvalue()


def import_epitope_model_bytes(data: bytes, *, allow_unsafe: bool = False) -> EpitopeModelBundle:
    """Load a model bundle from exported zip bytes.

    By default, unsafe pickle deserialization is blocked. Set allow_unsafe=True only
    when importing models from fully trusted internal sources.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
            meta_raw = zf.read("metadata.json")
            names = set(zf.namelist())
            safe_raw = zf.read("safe_model.json") if "safe_model.json" in names else b""
            state_raw = zf.read("torch_state.npz") if "torch_state.npz" in names else b""
            bundle_raw = zf.read("model_bundle.pkl") if "model_bundle.pkl" in names else b""
    except Exception as e:
        raise ValueError(f"模型文件读取失败: {e}") from e

    try:
        meta = json.loads(meta_raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"模型元信息损坏: {e}") from e

    if str(meta.get("format", "")) != "confluencia-epitope-model":
        raise ValueError("不是受支持的 Confluencia 表位模型文件。")

    serialization = str(meta.get("serialization", "pickle-unsafe")).strip().lower()
    if serialization == "safe-ridge-v1":
        try:
            payload = json.loads(safe_raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"安全模型内容损坏: {e}") from e

        scaler = StandardScaler()
        s = dict(payload.get("scaler", {}))
        scaler.mean_ = np.asarray(s.get("mean", []), dtype=np.float64)
        scaler.scale_ = np.asarray(s.get("scale", []), dtype=np.float64)
        scaler.var_ = np.asarray(s.get("var", []), dtype=np.float64)
        scaler.n_features_in_ = int(s.get("n_features_in", len(scaler.mean_)))
        scaler.n_samples_seen_ = int(s.get("n_samples_seen", 0))

        r = dict(payload.get("ridge", {}))
        ridge = Ridge(alpha=float(r.get("alpha", 1.0)))
        ridge.coef_ = np.asarray(r.get("coef", []), dtype=np.float64)
        ridge.intercept_ = float(r.get("intercept", 0.0))
        ridge.n_features_in_ = int(r.get("n_features_in", ridge.coef_.shape[-1] if ridge.coef_.size > 0 else 0))

        model = Pipeline(steps=[("scaler", scaler), ("ridge", ridge)])
        bundle = _rebuild_bundle_from_meta(meta, model)
    elif serialization == "safe-torch-state-npz-v1":
        if not torch_available() or torch is None:
            raise ValueError("当前环境缺少 PyTorch，无法导入 torch-mamba 安全模型。")
        try:
            payload = json.loads(safe_raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"安全模型内容损坏: {e}") from e

        cfg = TorchMambaConfig(**dict(payload.get("cfg", {})))
        env_dim = int(payload.get("env_dim", 0))
        use_real_mamba = bool(payload.get("used_real_mamba", False))
        model = MambaSequenceRegressor(env_dim=env_dim, cfg=cfg, use_real_mamba=use_real_mamba)

        npz = np.load(io.BytesIO(state_raw), allow_pickle=False)
        ref_state = model.state_dict()
        loaded_state = {}
        for k, ref in ref_state.items():
            if k not in npz:
                raise ValueError(f"模型权重缺失: {k}")
            loaded_state[k] = torch.as_tensor(npz[k], dtype=ref.dtype)
        model.load_state_dict(loaded_state, strict=True)

        torch_bundle = TorchMambaBundle(
            model=model,
            env_cols=[str(x) for x in payload.get("env_cols", [])],
            env_mean=np.asarray(payload.get("env_mean", []), dtype=np.float32),
            env_std=np.asarray(payload.get("env_std", []), dtype=np.float32),
            max_len=int(payload.get("max_len", cfg.max_len)),
            history={str(k): [float(x) for x in v] for k, v in dict(payload.get("history", {})).items()},
            used_real_mamba=use_real_mamba,
        )
        bundle = _rebuild_bundle_from_meta(meta, torch_bundle)
    elif serialization.startswith("pickle"):
        if not allow_unsafe:
            raise ValueError(
                "当前模型文件使用不安全反序列化格式（pickle），默认已禁用导入。"
                "若确认文件来源完全可信，请勾选“允许不安全导入”后重试。"
            )
        if not bundle_raw:
            raise ValueError("未找到旧版模型数据 model_bundle.pkl。")
        bundle = pickle.loads(bundle_raw)
        if not isinstance(bundle, EpitopeModelBundle):
            raise ValueError("模型对象类型不匹配，导入失败。")
    else:
        raise ValueError(f"不支持的模型序列化格式: {serialization}")

    bundle_schema = str(getattr(bundle, "feature_schema_version", "legacy-unknown"))
    if bundle_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"模型特征版本不兼容: model={bundle_schema}, runtime={FEATURE_SCHEMA_VERSION}。"
            "请使用同版本运行环境，或重新训练后导出模型。"
        )

    bundle_hash = str(getattr(bundle, "kmer_hash_version", "legacy-unknown"))
    if bundle_hash != KMER_HASH_VERSION:
        raise ValueError(
            f"模型哈希版本不兼容: model={bundle_hash}, runtime={KMER_HASH_VERSION}。"
            "请使用同版本运行环境，或重新训练后导出模型。"
        )

    if not getattr(bundle, "feature_schema_id", ""):
        bundle.feature_schema_id = feature_schema_id(FeatureSpec())
    return bundle


def _assert_feature_compat(bundle: EpitopeModelBundle, feature_names: list[str], x_dim: int) -> None:
    if int(bundle.feature_dim) != int(x_dim):
        raise ValueError(
            f"特征维度不一致: model={bundle.feature_dim}, runtime={x_dim}。"
            "请使用匹配版本模型，或重新训练后预测。"
        )

    if list(bundle.feature_names) != list(feature_names):
        raise ValueError("特征名签名不一致，当前运行时与模型训练环境不兼容。请重新训练模型。")

    if str(getattr(bundle, "feature_schema_version", "legacy-unknown")) != FEATURE_SCHEMA_VERSION:
        raise ValueError("模型特征版本与当前运行时不一致，请重新训练模型。")

    if str(getattr(bundle, "kmer_hash_version", "legacy-unknown")) != KMER_HASH_VERSION:
        raise ValueError("模型哈希版本与当前运行时不一致，请重新训练模型。")


def _proxy_objective(work: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """Construct a proxy efficacy label when no measured efficacy is available.

    Weights are designed to reflect known determinants of circRNA vaccine efficacy:
    - 25% dose: primary determinant of antigen availability (linear dose-response)
    - 18% frequency: repeated exposure boosts immune memory (diminishing returns)
    - 12% circRNA expression: transcript level correlates with protein output
    - 10% IFN response score: innate immune activation indicates adjuvant activity
    - 35% sequence features (first 96 dims): captures MHC-binding and immunogenicity signal
    Weights sum to 1.0 and are weighted toward sequence features because epitope-level
    immunogenicity is the strongest predictor of vaccine efficacy in practice.
    """
    base = 0.25 * work["dose"].to_numpy(dtype=np.float32)
    base += 0.18 * work["freq"].to_numpy(dtype=np.float32)
    base += 0.12 * work["circ_expr"].to_numpy(dtype=np.float32)
    base += 0.10 * work["ifn_score"].to_numpy(dtype=np.float32)

    seq_term = np.zeros((len(work),), dtype=np.float32)
    if X.shape[1] > 0:
        seq_term = 0.35 * X[:, : min(96, X.shape[1])].mean(axis=1).astype(np.float32)

    return (base + seq_term).astype(np.float32)


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


def build_artifacts_from_model(model_bundle: EpitopeModelBundle) -> EpitopeArtifacts:
    return EpitopeArtifacts(
        compute_profile=model_bundle.compute_profile,
        moe_weights=dict(model_bundle.moe_weights),
        moe_metrics=dict(model_bundle.moe_metrics),
        used_proxy_label=model_bundle.used_proxy_label,
        feature_dim=model_bundle.feature_dim,
        env_cols=list(model_bundle.env_cols),
        model_backend=model_bundle.model_backend,
        used_real_mamba=bool(model_bundle.used_real_mamba),
    )


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return _shared_reg_metrics(y_true, y_pred)


def _split_train_val_indices(n: int, seed: int = 42, val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Random split for backward compatibility."""
    idx = np.arange(max(n, 0), dtype=np.int64)
    if n <= 1:
        return idx, idx

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(float(n) * float(val_ratio)))
    n_val = max(1, min(n - 1, n_val))

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size == 0:
        tr_idx = val_idx.copy()
    return tr_idx, val_idx


def _split_train_val_by_sequence(
    df: pd.DataFrame,
    seq_col: str = "epitope_seq",
    seed: int = 42,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence-aware split: ensures same sequence doesn't appear in both train and val.

    This prevents data leakage when the same peptide sequence appears with different
    experimental conditions (dose, freq, etc.).
    """
    if seq_col not in df.columns:
        # Fallback to random split if sequence column not found
        return _split_train_val_indices(len(df), seed, val_ratio)

    unique_seqs = df[seq_col].dropna().unique()
    if len(unique_seqs) < 5:
        # Too few unique sequences, use random split
        return _split_train_val_indices(len(df), seed, val_ratio)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)

    n_val_seqs = max(1, int(len(unique_seqs) * val_ratio))
    val_seqs = set(unique_seqs[:n_val_seqs])

    val_idx = np.array([i for i, s in enumerate(df[seq_col]) if s in val_seqs])
    train_idx = np.array([i for i, s in enumerate(df[seq_col]) if s not in val_seqs])

    if len(train_idx) == 0 or len(val_idx) == 0:
        return _split_train_val_indices(len(df), seed, val_ratio)

    return train_idx, val_idx


def _hash_dataframe(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _snapshot_env_deps() -> dict[str, str]:
    deps: dict[str, str] = {}
    for pkg in ["python", "numpy", "pandas", "scikit-learn", "streamlit", "torch"]:
        if pkg == "python":
            deps[pkg] = platform.python_version()
            continue
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except Exception:
            deps[pkg] = "not-installed"
    return deps


def _auto_save_repro(
    module: str,
    data_df: pd.DataFrame,
    config: dict,
    metrics: dict,
    log_dir: str | Path | None = None,
) -> str:
    """Auto-save reproducibility bundle after each training run.

    Saves config, data hash, environment dependencies, and core metrics
    to logs/reproduce/ as both CSV row and Markdown report.
    Returns the run_id.
    """
    if log_dir is None:
        base = Path(__file__).resolve().parents[1] / "logs" / "reproduce"
    else:
        base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_hash = _hash_dataframe(data_df)
    run_id = f"{module}_{ts}_{data_hash[:8]}"
    env_deps = _snapshot_env_deps()

    save_report_csv(
        module=module,
        config=config,
        metrics=metrics,
        data_hash=data_hash,
        n_rows=int(len(data_df)),
        env_deps=env_deps,
        log_dir=base,
    )

    report = generate_experiment_report(
        module=module,
        config=config,
        metrics=metrics,
        data_hash=data_hash,
        n_rows=int(len(data_df)),
        env_deps=env_deps,
        python_executable=sys.executable,
    )
    md_path = base / f"{run_id}.md"
    md_path.write_text(report, encoding="utf-8")
    return run_id


def _tune_sklearn_regressor(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "random",
    n_iter: int = 20,
    cv: int = 3,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Tune sklearn regressor hyperparameters using cross-validation.

    Returns (best_model, tuning_results).
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

    param_grids: Dict[str, Dict[str, Any]] = {
        "hgb": {
            "max_depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "min_samples_leaf": [10, 20, 30],
        },
        "gbr": {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.03, 0.05, 0.1],
        },
        "rf": {
            "n_estimators": [150, 200, 260, 300],
            "max_depth": [8, 10, 12, 15],
            "min_samples_leaf": [1, 2, 4],
        },
        "ridge": {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "mlp": {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "mlp__alpha": [0.0001, 0.001, 0.01],
            "mlp__learning_rate_init": [0.001, 0.0005],
        },
    }

    base_model = _build_legacy_regressor(model_name, random_state=seed)
    grid = param_grids.get(model_name, {})

    if not grid:
        return base_model, {"best_params": {}, "method": "none"}

    if strategy == "grid":
        search = GridSearchCV(base_model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            base_model, grid, n_iter=min(n_iter, 20), cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=seed
        )

    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": float(-search.best_score_),
        "method": strategy,
    }


def train_epitope_model(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    model_backend: str = "torch-mamba",
    torch_cfg: TorchMambaConfig | None = None,
    # 特征编码器配置
    feature_spec: FeatureSpec | None = None,
    # 检查点参数
    checkpoint_dir: Optional[str] = None,
    checkpoint_save_every: int = 5,
    checkpoint_keep_last: int = 3,
    resume_from: Optional[str] = None,
    # 回调函数
    on_epoch_end: Optional[Callable[[int, float, float, float], None]] = None,
    # 超参数调优
    tune_hyperparams: bool = False,
    tune_strategy: str = "random",
    tune_n_iter: int = 20,
    tune_cv: int = 3,
) -> Tuple[EpitopeModelBundle, EpitopeTrainingReport]:
    start_time = time.time()
    tune_result: Dict[str, Any] | None = None  # hyperparameter tuning result
    work = ensure_columns(df)
    _feat_spec = feature_spec or FeatureSpec()
    X, feature_names, env_cols = build_feature_matrix(work, _feat_spec)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("输入数据为空，无法训练模型。")

    logger.info(f"Training epitope model: n_samples={n}, n_features={X.shape[1]}, backend={model_backend}")

    y = _resolve_label(work, "efficacy")
    used_proxy = False
    if y is None:
        y = _proxy_objective(work, X)
        used_proxy = True

    tr_idx, va_idx = _split_train_val_by_sequence(work, seq_col="epitope_seq", seed=42, val_ratio=0.2)
    work_tr = work.iloc[tr_idx].reset_index(drop=True)
    work_va = work.iloc[va_idx].reset_index(drop=True)
    y_tr = np.asarray(y[tr_idx], dtype=np.float32).reshape(-1)
    y_va = np.asarray(y[va_idx], dtype=np.float32).reshape(-1)
    eval_work = work_va if len(work_va) > 0 else work_tr
    eval_y = y_va if y_va.size > 0 else y_tr

    y_std = max(float(np.std(y_tr)), 1e-6)
    backend = str(model_backend)
    schema_id = feature_schema_id(_feat_spec)

    if backend == "torch-mamba":
        if not torch_available():
            raise RuntimeError("PyTorch 不可用，无法使用 torch-mamba 后端。")
        cfg = torch_cfg or TorchMambaConfig()
        torch_bundle = train_torch_mamba(
            work_tr, y_tr, env_cols=list(env_cols), cfg=cfg, prefer_real_mamba=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_save_every=checkpoint_save_every,
            checkpoint_keep_last=checkpoint_keep_last,
            resume_from=resume_from,
            on_epoch_end=on_epoch_end,
        )
        eval_pred = predict_torch_mamba(torch_bundle, eval_work)
        metrics = _reg_metrics(eval_y, eval_pred)
        model_bundle = EpitopeModelBundle(
            model_backend="torch-mamba",
            compute_profile=compute_mode,
            model=torch_bundle,
            feature_names=list(feature_names),
            feature_dim=int(X.shape[1]),
            env_cols=list(env_cols),
            used_proxy_label=used_proxy,
            y_std=y_std,
            used_real_mamba=bool(torch_bundle.used_real_mamba),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_schema_id=schema_id,
            kmer_hash_version=KMER_HASH_VERSION,
            moe_weights={},
            moe_metrics={
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                "train_loss_last": float(torch_bundle.history.get("train_loss", [0.0])[-1]),
                "val_loss_last": float(torch_bundle.history.get("val_loss", [0.0])[-1]),
            },
        )
    elif backend in {"hgb", "gbr", "rf", "ridge", "mlp"}:
        x_tr, _, _ = build_feature_matrix(work_tr, _feat_spec)
        x_va, _, _ = build_feature_matrix(eval_work, _feat_spec)
        tune_result = None
        if tune_hyperparams and x_tr.shape[0] >= 30:
            model, tune_result = _tune_sklearn_regressor(
                backend, x_tr, y_tr,
                strategy=tune_strategy, n_iter=tune_n_iter, cv=tune_cv, seed=42,
            )
        else:
            model = _build_legacy_regressor(backend, random_state=42)
            model.fit(x_tr, y_tr)
        eval_pred = np.asarray(model.predict(x_va), dtype=np.float32).reshape(-1)
        metrics = _reg_metrics(eval_y, eval_pred)
        model_bundle = EpitopeModelBundle(
            model_backend=backend,
            compute_profile=compute_mode,
            model=model,
            feature_names=list(feature_names),
            feature_dim=int(x_tr.shape[1]),
            env_cols=list(env_cols),
            used_proxy_label=used_proxy,
            y_std=y_std,
            used_real_mamba=False,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_schema_id=schema_id,
            kmer_hash_version=KMER_HASH_VERSION,
            moe_weights={},
            moe_metrics={
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
            },
        )
    else:
        x_tr, _, _ = build_feature_matrix(work_tr, _feat_spec)
        x_va, _, _ = build_feature_matrix(eval_work, _feat_spec)
        prof = choose_compute_profile(n_samples=int(len(work_tr)), requested=compute_mode)

        # Hyperparameter tuning for MOE experts
        from confluencia_shared.moe import ExpertConfig
        tuned_config: ExpertConfig | None = None
        tune_result = None
        if tune_hyperparams and x_tr.shape[0] >= 30:
            # Tune each enabled expert individually and collect best params
            best_params: Dict[str, Any] = {}
            for expert_name in prof.enabled_experts:
                if expert_name in {"hgb", "gbr", "rf", "ridge", "mlp"}:
                    _, result = _tune_sklearn_regressor(
                        expert_name, x_tr, y_tr,
                        strategy=tune_strategy, n_iter=tune_n_iter, cv=tune_cv, seed=42,
                    )
                    best_params[expert_name] = result
            if best_params:
                # Build tuned ExpertConfig from best params
                tuned_config = ExpertConfig()
                for name, res in best_params.items():
                    bp = res.get("best_params", {})
                    if name == "ridge" and "alpha" in bp:
                        tuned_config.ridge_alpha = float(bp["alpha"])
                    elif name in {"hgb", "gbr"}:
                        if "max_depth" in bp:
                            tuned_config.hgb_max_depth = int(bp["max_depth"])
                        if "learning_rate" in bp:
                            tuned_config.hgb_learning_rate = float(bp["learning_rate"])
                    elif name == "rf":
                        if "n_estimators" in bp:
                            tuned_config.rf_n_estimators = int(bp["n_estimators"])
                        if "max_depth" in bp:
                            tuned_config.rf_max_depth = int(bp["max_depth"])
                    elif name == "mlp":
                        if "mlp__hidden_layer_sizes" in bp:
                            tuned_config.mlp_hidden_sizes = bp["mlp__hidden_layer_sizes"]
                tune_result = {"expert_params": best_params, "method": tune_strategy}
            logger.info(f"Hyperparameter tuning: {tune_result}")

        moe = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds, config=tuned_config)
        moe.fit(x_tr, y_tr)
        eval_pred = moe.predict(x_va)
        metrics = _reg_metrics(eval_y, eval_pred)
        model_bundle = EpitopeModelBundle(
            model_backend="sklearn-moe",
            compute_profile=prof.level,
            model=moe,
            feature_names=list(feature_names),
            feature_dim=int(x_tr.shape[1]),
            env_cols=list(env_cols),
            used_proxy_label=used_proxy,
            y_std=y_std,
            used_real_mamba=False,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_schema_id=schema_id,
            kmer_hash_version=KMER_HASH_VERSION,
            moe_weights=moe.explain_weights(),
            moe_metrics={
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                **moe.metrics,
            },
        )

    report = EpitopeTrainingReport(
        sample_count=n,
        used_label=not used_proxy,
        metrics={
            "mae": float(model_bundle.moe_metrics.get("mae", 0.0)),
            "rmse": float(model_bundle.moe_metrics.get("rmse", 0.0)),
            "r2": float(model_bundle.moe_metrics.get("r2", 0.0)),
        },
        model_backend=model_bundle.model_backend,
        hyperparam_tuning=tune_result,
    )

    try:
        _auto_save_repro(
            module="epitope-training",
            data_df=df,
            config={
                "model_backend": model_bundle.model_backend,
                "compute_mode": compute_mode,
                "rows": n,
                "feature_schema_id": schema_id,
            },
            metrics={
                "mae": float(report.metrics.get("mae", 0.0)),
                "rmse": float(report.metrics.get("rmse", 0.0)),
                "r2": float(report.metrics.get("r2", 0.0)),
                "used_proxy_label": used_proxy,
            },
        )
    except Exception as exc:
        logger.debug(f"Auto-save failed (non-critical): {exc}")

    elapsed = time.time() - start_time
    logger.info(
        f"Training complete: mae={report.metrics.get('mae', 0):.4f}, "
        f"rmse={report.metrics.get('rmse', 0):.4f}, r2={report.metrics.get('r2', 0):.4f}, "
        f"duration={elapsed:.2f}s"
    )

    return model_bundle, report


def predict_epitope_model(
    model_bundle: EpitopeModelBundle,
    df: pd.DataFrame,
    sensitivity_sample_idx: int = 0,
) -> Tuple[pd.DataFrame, SensitivityArtifacts]:
    logger.debug(f"Predicting epitope model: n_samples={len(df)}, backend={model_bundle.model_backend}")
    work = ensure_columns(df)
    X, feature_names, _ = build_feature_matrix(work, FeatureSpec())
    _assert_feature_compat(model_bundle, feature_names, int(X.shape[1]))
    n = int(X.shape[0])

    if n == 0:
        out = work.copy()
        out["efficacy_pred"] = np.zeros((0,), dtype=np.float32)
        out["pred_uncertainty"] = np.zeros((0,), dtype=np.float32)
        sens = SensitivityArtifacts(
            sample_index=0,
            prediction=0.0,
            top_rows=pd.DataFrame(columns=["feature", "importance", "grad"]),
            neighborhood_rows=pd.DataFrame(columns=["group", "importance"]),
        )
        return out, sens

    sid = int(np.clip(sensitivity_sample_idx, 0, max(n - 1, 0)))
    y_known = _resolve_label(work, "efficacy")

    if model_bundle.model_backend == "torch-mamba":
        torch_bundle = model_bundle.model
        if not isinstance(torch_bundle, TorchMambaBundle):
            raise TypeError("torch-mamba 模型包类型不匹配。")

        pred = predict_torch_mamba(torch_bundle, work).astype(np.float32)
        out = work.copy()
        out["efficacy_pred"] = pred
        if y_known is not None:
            out["pred_uncertainty"] = np.abs(pred - y_known).astype(np.float32) / model_bundle.y_std
        else:
            out["pred_uncertainty"] = np.zeros_like(pred, dtype=np.float32)

        sens_t = sensitivity_torch_mamba(torch_bundle, out.iloc[sid], sample_index=sid, top_k=20)
        top_records = [{"feature": name, "importance": val, "grad": 0.0} for name, val, _ in sens_t.token_rows]
        top_records.extend([{"feature": name, "importance": val, "grad": 0.0} for name, val in sens_t.env_rows])
        top_df = pd.DataFrame(top_records, columns=["feature", "importance", "grad"]).sort_values(
            "importance", ascending=False
        )
        ng_df = (
            pd.DataFrame({"group": list(sens_t.neighborhood.keys()), "importance": list(sens_t.neighborhood.values())})
            .sort_values("importance", ascending=False)
        )
        sens = SensitivityArtifacts(
            sample_index=sid,
            prediction=float(sens_t.prediction),
            top_rows=top_df,
            neighborhood_rows=ng_df,
        )
        return out, sens

    model = model_bundle.model
    pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
    out = work.copy()
    out["efficacy_pred"] = pred

    if model_bundle.model_backend == "sklearn-moe" and hasattr(model, "predict_uncertainty"):
        out["pred_uncertainty"] = np.asarray(model.predict_uncertainty(X), dtype=np.float32).reshape(-1)
    elif y_known is not None:
        out["pred_uncertainty"] = np.abs(pred - y_known).astype(np.float32) / model_bundle.y_std
    else:
        out["pred_uncertainty"] = np.zeros_like(pred, dtype=np.float32)

    p0, grad = numerical_input_gradient(model, X[sid], eps=1e-3)
    imp = np.abs(grad)
    top = top_features(model_bundle.feature_names, imp, grad, k=20)
    top_df = pd.DataFrame(top, columns=["feature", "importance", "grad"])
    ng = neighborhood_importance(model_bundle.feature_names, imp)
    ng_df = pd.DataFrame({"group": list(ng.keys()), "importance": list(ng.values())}).sort_values("importance", ascending=False)
    sens = SensitivityArtifacts(
        sample_index=sid,
        prediction=float(p0),
        top_rows=top_df,
        neighborhood_rows=ng_df,
    )
    return out, sens


def train_and_predict_epitope(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    sensitivity_sample_idx: int = 0,
    model_backend: str = "torch-mamba",
    torch_cfg: TorchMambaConfig | None = None,
) -> Tuple[pd.DataFrame, EpitopeArtifacts, SensitivityArtifacts, EpitopeTrainingReport]:
    model_bundle, report = train_epitope_model(
        df,
        compute_mode=compute_mode,
        model_backend=model_backend,
        torch_cfg=torch_cfg,
    )
    result_df, sens = predict_epitope_model(
        model_bundle,
        df,
        sensitivity_sample_idx=sensitivity_sample_idx,
    )
    artifacts = build_artifacts_from_model(model_bundle)
    return result_df, artifacts, sens, report
