from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.model_selection import train_test_split

from .featurizer import MoleculeFeatures
from .predictor import infer_env_cols, make_xy
from confluencia_shared.training import EarlyStopping, build_scheduler, make_training_suggestions
from confluencia_shared.metrics import rmse as _rmse
try:
    from confluencia_shared.optim.differential_evolution import de_optimize
except Exception:
    from common.optim.differential_evolution import de_optimize


@dataclass
class TorchDrugModelBundle:
    model_state: Dict[str, Any]
    input_dim: int
    hidden_sizes: List[int]
    dropout: float
    env_cols: List[str]
    smiles_col: str
    target_col: str
    env_medians: Dict[str, float]
    feature_names: List[str]
    created_at: str
    featurizer_version: int = 2
    radius: int = 2
    n_bits: int = 2048
    version: int = 1


def build_torch_model(input_dim: int, hidden_sizes: Sequence[int], dropout: float) -> nn.Module:
    layers: List[nn.Module] = []
    in_dim = int(input_dim)
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, int(h)))
        layers.append(nn.ReLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(h)
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _bundle_to_dict(bundle: TorchDrugModelBundle) -> Dict[str, Any]:
    return {
        "model_state": bundle.model_state,
        "input_dim": int(bundle.input_dim),
        "hidden_sizes": list(bundle.hidden_sizes),
        "dropout": float(bundle.dropout),
        "env_cols": list(bundle.env_cols),
        "smiles_col": str(bundle.smiles_col),
        "target_col": str(bundle.target_col),
        "env_medians": {k: float(v) for k, v in bundle.env_medians.items()},
        "feature_names": list(bundle.feature_names),
        "created_at": str(bundle.created_at),
        "featurizer_version": int(bundle.featurizer_version),
        "radius": int(bundle.radius),
        "n_bits": int(bundle.n_bits),
        "version": int(bundle.version),
    }


def _bundle_from_dict(d: Dict[str, Any]) -> TorchDrugModelBundle:
    return TorchDrugModelBundle(
        model_state=cast_state_dict(d.get("model_state")),
        input_dim=int(d.get("input_dim", 0)),
        hidden_sizes=list(d.get("hidden_sizes", [])),
        dropout=float(d.get("dropout", 0.0)),
        env_cols=list(d.get("env_cols", [])),
        smiles_col=str(d.get("smiles_col", "smiles")),
        target_col=str(d.get("target_col", "efficacy")),
        env_medians={k: float(v) for k, v in cast(dict, d.get("env_medians", {})).items()},
        feature_names=list(d.get("feature_names", [])),
        created_at=str(d.get("created_at", "")),
        featurizer_version=int(d.get("featurizer_version", 2)),
        radius=int(d.get("radius", 2)),
        n_bits=int(d.get("n_bits", 2048)),
        version=int(d.get("version", 1)),
    )


def cast_state_dict(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        return state
    return {}


def save_torch_bundle(bundle: TorchDrugModelBundle, path: str) -> None:
    torch.save(_bundle_to_dict(bundle), path)


def dump_torch_bundle(bundle: TorchDrugModelBundle, fileobj) -> None:
    torch.save(_bundle_to_dict(bundle), fileobj)


def load_torch_bundle(path: str) -> TorchDrugModelBundle:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("模型文件格式不正确")
    return _bundle_from_dict(obj)


def load_torch_bundle_from_bytes(data: bytes) -> TorchDrugModelBundle:
    import io

    buf = io.BytesIO(data)
    obj = torch.load(buf, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("模型文件格式不正确")
    return _bundle_from_dict(obj)


def _select_device(use_cuda: bool) -> torch.device:
    if bool(use_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model_from_bundle(bundle: TorchDrugModelBundle, device: torch.device) -> nn.Module:
    model = build_torch_model(
        input_dim=int(bundle.input_dim),
        hidden_sizes=bundle.hidden_sizes,
        dropout=float(bundle.dropout),
    ).to(device)
    model.load_state_dict(bundle.model_state)
    model.eval()
    return model


def train_torch_bundle(
    df,
    smiles_col: str = "smiles",
    target_col: str = "efficacy",
    env_cols: Optional[List[str]] = None,
    hidden_sizes: Sequence[int] = (512, 256),
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
    featurizer_version: int = 2,
    radius: int = 2,
    n_bits: int = 2048,
    drop_invalid_smiles: bool = True,
    use_cuda: bool = False,
    amp: bool = True,
    early_stopping_patience: int = 10,
    weight_decay: float = 1e-4,
    lr_schedule: str = "cosine",
    step_size: int = 20,
    gamma: float = 0.5,
    min_lr: float = 1e-6,
    max_grad_norm: float = 5.0,
    teacher_bundle: Optional[TorchDrugModelBundle] = None,
    distill_weight: float = 0.2,
) -> Tuple[TorchDrugModelBundle, Dict[str, float]]:
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

    device = _select_device(use_cuda)
    model = build_torch_model(input_dim=int(x.shape[1]), hidden_sizes=hidden_sizes, dropout=float(dropout)).to(device)

    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
    x_val_t = torch.from_numpy(x_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=int(batch_size), shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    scheduler = build_scheduler(optimizer, lr_schedule, epochs=int(epochs), step_size=int(step_size), gamma=float(gamma), min_lr=float(min_lr))
    stopper = EarlyStopping(patience=int(early_stopping_patience), mode="min")

    teacher_model: Optional[nn.Module] = None
    if teacher_bundle is not None:
        teacher_model = _build_model_from_bundle(teacher_bundle, device)
        teacher_model.eval()

    scaler = torch.amp.GradScaler(enabled=bool(amp) and device.type == "cuda")
    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(int(epochs)):
        model.train()
        run = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device.type), enabled=bool(amp) and device.type == "cuda"):
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if teacher_model is not None and float(distill_weight) > 0:
                    with torch.no_grad():
                        t_pred = teacher_model(xb)
                    distill = loss_fn(pred, t_pred)
                    loss = loss + float(distill_weight) * distill
            loss = loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            run += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        history["train_loss"].append(run / max(1, count))

        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val_t.to(device)).detach().cpu().numpy().reshape(-1)
        val_loss = float(mean_squared_error(y_val, y_val_pred))
        history["val_loss"].append(val_loss)
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_val_t.to(device)).detach().cpu().numpy().reshape(-1)

    metrics = {
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": _rmse(y_val, y_pred),
        "r2": float(r2_score(y_val, y_pred)),
        "explained_variance_score": float(explained_variance_score(y_val, y_pred)),
        "max_error": float(max_error(y_val, y_pred)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(x.shape[1]),
        "invalid_smiles": int(invalid_smiles),
        "dropped_invalid_smiles": int(invalid_smiles if bool(drop_invalid_smiles) else 0),
        "device": str(device.type),
        "history": history,
        "suggestions": make_training_suggestions(history),
        # Return raw data for plotting
        "y_val": y_val,
        "y_pred": y_pred,
        "feature_names": feature_names,
    }

    bundle = TorchDrugModelBundle(
        model_state=model.state_dict(),
        input_dim=int(x.shape[1]),
        hidden_sizes=list(hidden_sizes),
        dropout=float(dropout),
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

    return bundle, metrics


def predict_torch_one(
    bundle: TorchDrugModelBundle,
    smiles: str,
    env_params: Optional[Dict[str, float]] = None,
    use_cuda: bool = False,
) -> float:
    featurizer = MoleculeFeatures(version=int(bundle.featurizer_version), radius=int(bundle.radius), n_bits=int(bundle.n_bits))
    mol_x, _ = featurizer.transform_one(smiles)

    env_params = env_params or {}
    env_vec = []
    for c in bundle.env_cols:
        if c in env_params:
            env_vec.append(float(env_params[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.array(env_vec, dtype=np.float32) if bundle.env_cols else np.zeros((0,), dtype=np.float32)
    x = np.concatenate([mol_x, env_x], axis=0).astype(np.float32).reshape(1, -1)

    device = _select_device(use_cuda)
    model = _build_model_from_bundle(bundle, device)
    with torch.no_grad():
        y = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)[0]
    return float(y)


def suggest_env_by_de_torch(
    bundle: TorchDrugModelBundle,
    smiles: str,
    env_bounds: Sequence[Tuple[float, float]],
    maximize: bool = True,
    de_kwargs: Optional[dict] = None,
    use_cuda: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    使用差分进化搜索 Torch 模型的数值环境参数以最大化/最小化预测值。

    Returns: (best_env_vector, best_prediction)
    """
    if de_kwargs is None:
        de_kwargs = {"pop_size": max(20, 5 * len(env_bounds)), "max_iter": 100, "F": 0.8, "CR": 0.9}

    featurizer = MoleculeFeatures(version=int(bundle.featurizer_version), radius=int(bundle.radius), n_bits=int(bundle.n_bits))
    mol_x, _ = featurizer.transform_one(smiles)

    device = _select_device(bool(use_cuda))
    model = _build_model_from_bundle(bundle, device)

    def objective(env_vec: np.ndarray) -> float:
        x = np.concatenate([mol_x, env_vec.reshape(-1).astype(np.float32)], axis=0).astype(np.float32).reshape(1, -1)
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)[0]
        return float(y)

    best_env, best_val = de_optimize(objective, env_bounds, maximize=maximize, **de_kwargs)
    return np.asarray(best_env, dtype=float), float(best_val)


def predict_torch_batch(
    bundle: TorchDrugModelBundle,
    smiles_list: Sequence[str],
    env_params_list: Optional[Sequence[Dict[str, float]]] = None,
    env_matrix: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    use_cuda: bool = False,
) -> np.ndarray:
    featurizer = MoleculeFeatures(version=int(bundle.featurizer_version), radius=int(bundle.radius), n_bits=int(bundle.n_bits))
    mol_x, _ = featurizer.transform_many(smiles_list)

    env_cols = list(bundle.env_cols)
    n = len(smiles_list)
    if env_matrix is not None:
        env_x = np.asarray(env_matrix, dtype=np.float32)
    elif env_cols:
        env_x = np.zeros((n, len(env_cols)), dtype=np.float32)
        if env_params_list is None:
            env_params_list = [{} for _ in range(n)]
        for i, params in enumerate(env_params_list):
            for j, c in enumerate(env_cols):
                env_x[i, j] = float(params.get(c, bundle.env_medians.get(c, 0.0)))
    else:
        env_x = np.zeros((n, 0), dtype=np.float32)

    x = np.concatenate([mol_x, env_x], axis=1).astype(np.float32)

    device = _select_device(use_cuda)
    model = _build_model_from_bundle(bundle, device)

    preds = np.empty((n,), dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        xb = torch.from_numpy(x[start:end]).to(device)
        with torch.no_grad():
            preds[start:end] = model(xb).detach().cpu().numpy().reshape(-1)
    return preds
