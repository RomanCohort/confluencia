from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.model_selection import train_test_split

from .predictor import infer_env_cols
from confluencia_shared.training import EarlyStopping, build_scheduler, make_training_suggestions
from confluencia_shared.utils import ema as ema_utils
from confluencia_shared.metrics import rmse as _rmse
from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from confluencia_shared.optim.differential_evolution import de_optimize
except Exception:
    # fallback when running as package
    from common.optim.differential_evolution import de_optimize


_SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<CLS>"]


def build_char_vocab(smiles_list: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in smiles_list:
        for ch in str(s):
            counts[ch] = counts.get(ch, 0) + 1

    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
    for ch, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= int(min_freq):
            vocab.setdefault(ch, len(vocab))
    return vocab


def encode_smiles(smiles: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    pad = vocab.get("<PAD>", 0)
    eos = vocab.get("<EOS>", 2)
    unk = vocab.get("<UNK>", 3)
    cls = vocab.get("<CLS>", 4)

    tokens = [cls] + [vocab.get(ch, unk) for ch in str(smiles)] + [eos]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens = tokens + [pad] * (max_len - len(tokens))
    return tokens


class SmilesTransformerRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        env_dim: int,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=0)
        self.pos = nn.Embedding(int(max_len), int(emb_dim))

        self.use_lstm = bool(use_lstm)
        self.lstm_bidirectional = bool(lstm_bidirectional)
        if self.use_lstm:
            lstm_out = int(lstm_hidden) * (2 if self.lstm_bidirectional else 1)
            self.lstm = nn.LSTM(
                input_size=int(emb_dim),
                hidden_size=int(lstm_hidden),
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            self.lstm_proj = nn.Linear(lstm_out, int(emb_dim))
        else:
            self.lstm = None
            self.lstm_proj = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(emb_dim),
            nhead=int(n_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.reg_head = nn.Sequential(
            nn.Linear(int(emb_dim) + int(env_dim), int(emb_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(emb_dim), 1),
        )

    def forward(self, token_ids: torch.Tensor, env_x: torch.Tensor) -> torch.Tensor:
        device = token_ids.device
        positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(token_ids.size(0), -1)
        x = self.emb(token_ids) + self.pos(positions)

        if self.lstm is not None:
            x, _ = self.lstm(x)
            x = self.lstm_proj(x)

        pad_mask = token_ids.eq(0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        if pad_mask.any():
            mask = (~pad_mask).unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            pooled = x.mean(dim=1)

        if env_x is None or env_x.numel() == 0:
            feats = pooled
        else:
            feats = torch.cat([pooled, env_x], dim=1)
        return self.reg_head(feats)


@dataclass
class TransformerDrugModelBundle:
    model_state: Dict[str, Any]
    vocab: Dict[str, int]
    max_len: int
    emb_dim: int
    n_heads: int
    n_layers: int
    ff_dim: int
    dropout: float
    use_lstm: bool
    lstm_hidden: int
    lstm_layers: int
    lstm_bidirectional: bool
    env_cols: List[str]
    env_medians: Dict[str, float]
    smiles_col: str
    target_col: str
    created_at: str
    version: int = 1


def _bundle_to_dict(bundle: TransformerDrugModelBundle) -> Dict[str, Any]:
    return {
        "model_state": bundle.model_state,
        "vocab": dict(bundle.vocab),
        "max_len": int(bundle.max_len),
        "emb_dim": int(bundle.emb_dim),
        "n_heads": int(bundle.n_heads),
        "n_layers": int(bundle.n_layers),
        "ff_dim": int(bundle.ff_dim),
        "dropout": float(bundle.dropout),
        "use_lstm": bool(bundle.use_lstm),
        "lstm_hidden": int(bundle.lstm_hidden),
        "lstm_layers": int(bundle.lstm_layers),
        "lstm_bidirectional": bool(bundle.lstm_bidirectional),
        "env_cols": list(bundle.env_cols),
        "env_medians": {k: float(v) for k, v in bundle.env_medians.items()},
        "smiles_col": str(bundle.smiles_col),
        "target_col": str(bundle.target_col),
        "created_at": str(bundle.created_at),
        "version": int(bundle.version),
    }


def _bundle_from_dict(d: Dict[str, Any]) -> TransformerDrugModelBundle:
    return TransformerDrugModelBundle(
        model_state=cast_state_dict(d.get("model_state")),
        vocab=cast(dict, d.get("vocab", {})),
        max_len=int(d.get("max_len", 128)),
        emb_dim=int(d.get("emb_dim", 128)),
        n_heads=int(d.get("n_heads", 4)),
        n_layers=int(d.get("n_layers", 2)),
        ff_dim=int(d.get("ff_dim", 256)),
        dropout=float(d.get("dropout", 0.1)),
        use_lstm=bool(d.get("use_lstm", False)),
        lstm_hidden=int(d.get("lstm_hidden", 128)),
        lstm_layers=int(d.get("lstm_layers", 1)),
        lstm_bidirectional=bool(d.get("lstm_bidirectional", True)),
        env_cols=list(d.get("env_cols", [])),
        env_medians={k: float(v) for k, v in cast(dict, d.get("env_medians", {})).items()},
        smiles_col=str(d.get("smiles_col", "smiles")),
        target_col=str(d.get("target_col", "efficacy")),
        created_at=str(d.get("created_at", "")),
        version=int(d.get("version", 1)),
    )


def cast_state_dict(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        return state
    return {}


def save_transformer_bundle(bundle: TransformerDrugModelBundle, path: str) -> None:
    torch.save(_bundle_to_dict(bundle), path)


def dump_transformer_bundle(bundle: TransformerDrugModelBundle, fileobj) -> None:
    torch.save(_bundle_to_dict(bundle), fileobj)


def load_transformer_bundle(path: str) -> TransformerDrugModelBundle:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("模型文件格式不正确")
    return _bundle_from_dict(obj)


def load_transformer_bundle_from_bytes(data: bytes) -> TransformerDrugModelBundle:
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


def _build_model_from_bundle(bundle: TransformerDrugModelBundle, env_dim: int, device: torch.device) -> nn.Module:
    model = SmilesTransformerRegressor(
        vocab_size=len(bundle.vocab),
        max_len=int(bundle.max_len),
        emb_dim=int(bundle.emb_dim),
        n_heads=int(bundle.n_heads),
        n_layers=int(bundle.n_layers),
        ff_dim=int(bundle.ff_dim),
        dropout=float(bundle.dropout),
        env_dim=int(env_dim),
        use_lstm=bool(bundle.use_lstm),
        lstm_hidden=int(bundle.lstm_hidden),
        lstm_layers=int(bundle.lstm_layers),
        lstm_bidirectional=bool(bundle.lstm_bidirectional),
    ).to(device)
    model.load_state_dict(bundle.model_state)
    model.eval()
    return model


def _prepare_env_matrix(df, env_cols: List[str], env_medians: Dict[str, float]) -> np.ndarray:
    if not env_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    env_df = df[env_cols].copy()
    for c in env_cols:
        env_df[c] = pd.to_numeric(env_df[c], errors="coerce")
        if c in env_medians:
            env_df[c] = env_df[c].fillna(float(env_medians[c]))
        else:
            env_df[c] = env_df[c].fillna(float(env_df[c].median()))
    return env_df.to_numpy(dtype=np.float32)


def suggest_env_by_de(
    bundle: TransformerDrugModelBundle,
    smiles: str,
    env_bounds: Sequence[Tuple[float, float]],
    maximize: bool = True,
    de_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, float]:
    """
    使用差分进化搜索最优环境条件（env vector）以最大化/最小化模型预测值。

    - `bundle`: 训练好的 `TransformerDrugModelBundle`。
    - `smiles`: 单条 SMILES 字符串。
    - `env_bounds`: 每个环境变量的 (low, high) 范围。
    - 返回 (best_env_vector, best_prediction)

    注意：这是一个示例集成，适用于数值型环境变量优化。
    """
    if de_kwargs is None:
        de_kwargs = {"pop_size": max(20, 5 * len(env_bounds)), "max_iter": 100, "F": 0.8, "CR": 0.9}
    device = device if device is not None else _select_device(False)

    # Build model
    model = _build_model_from_bundle(bundle, env_dim=len(env_bounds), device=device)
    model.eval()

    # Build token ids
    vocab = bundle.vocab
    token_ids = np.array([encode_smiles(smiles, vocab, int(bundle.max_len))], dtype=np.int64)

    def objective(env_vec: np.ndarray) -> float:
        # env_vec is 1D numpy array
        with torch.no_grad():
            t_ids = torch.from_numpy(token_ids).to(device)
            env_t = torch.from_numpy(env_vec.astype(np.float32).reshape(1, -1)).to(device)
            out = model(t_ids, env_t)
            val = float(out.cpu().numpy().reshape(-1)[0])
            return val

    best_env, best_val = de_optimize(objective, env_bounds, maximize=maximize, **de_kwargs)
    return np.asarray(best_env, dtype=float), float(best_val)


def train_transformer_bundle(
    df,
    smiles_col: str = "smiles",
    target_col: str = "efficacy",
    env_cols: Optional[List[str]] = None,
    max_len: int = 128,
    min_char_freq: int = 1,
    emb_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.1,
    use_lstm: bool = False,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    lstm_bidirectional: bool = True,
    lr: float = 2e-4,
    batch_size: int = 64,
    epochs: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    use_cuda: bool = False,
    amp: bool = True,
    early_stopping_patience: int = 8,
    weight_decay: float = 1e-4,
    lr_schedule: str = "cosine",
    step_size: int = 20,
    gamma: float = 0.5,
    min_lr: float = 1e-6,
    max_grad_norm: float = 5.0,
    teacher_bundle: Optional[TransformerDrugModelBundle] = None,
    distill_weight: float = 0.2,
    use_ema: bool = False,
    ema_decay: float = 0.99,
) -> Tuple[TransformerDrugModelBundle, Dict[str, float]]:
    env_cols = infer_env_cols(df, smiles_col=smiles_col, target_col=target_col, env_cols=env_cols)

    df = df.copy()
    df[smiles_col] = df[smiles_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    env_medians: Dict[str, float] = {}
    for c in env_cols:
        med = pd.to_numeric(df[c], errors="coerce").median()
        env_medians[c] = float(med) if np.isfinite(med) else 0.0

    smiles_list = df[smiles_col].astype(str).tolist()
    y = df[target_col].to_numpy(dtype=np.float32)

    vocab = build_char_vocab(smiles_list, min_freq=int(min_char_freq))
    token_mat = np.array(
        [encode_smiles(s, vocab, int(max_len)) for s in smiles_list],
        dtype=np.int64,
    )
    env_mat = _prepare_env_matrix(df, list(env_cols), env_medians)

    device = _select_device(use_cuda)
    x_train, x_val, env_train, env_val, y_train, y_val, seq_train, seq_val = train_test_split(
        token_mat,
        env_mat,
        y,
        smiles_list,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    model = SmilesTransformerRegressor(
        vocab_size=len(vocab),
        max_len=int(max_len),
        emb_dim=int(emb_dim),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        env_dim=int(env_train.shape[1]),
        use_lstm=bool(use_lstm),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        lstm_bidirectional=bool(lstm_bidirectional),
    ).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(env_train),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(env_val),
        torch.from_numpy(y_val.reshape(-1, 1)),
    )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    scheduler = build_scheduler(optimizer, lr_schedule, epochs=int(epochs), step_size=int(step_size), gamma=float(gamma), min_lr=float(min_lr))
    stopper = EarlyStopping(patience=int(early_stopping_patience), mode="min")

    teacher_model: Optional[nn.Module] = None
    if teacher_bundle is not None:
        teacher_model = _build_model_from_bundle(teacher_bundle, env_dim=int(env_train.shape[1]), device=device)
        teacher_model.eval()

    # Parameter-level EMA teacher (updated during training)
    ema_model: Optional[nn.Module] = None
    if use_ema:
        try:
            ema_model = ema_utils.clone_model_for_ema(model)
            ema_model.to(device)
            ema_model.eval()
            # if no external teacher bundle provided, use EMA as distillation teacher
            if teacher_model is None:
                teacher_model = ema_model
        except Exception as exc:
            logger.debug(f"EMA model setup failed: {exc}")

    scaler = torch.amp.GradScaler(enabled=bool(amp) and device.type == "cuda")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_state: Optional[Dict[str, Any]] = None
    best_val = float("inf")
    for _ in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, envb, yb in train_loader:
            xb = xb.to(device)
            envb = envb.to(device)
            yb = yb.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device.type), enabled=bool(amp) and device.type == "cuda"):
                preds = model(xb, envb).view(-1)
                loss = loss_fn(preds, yb)
                if teacher_model is not None and float(distill_weight) > 0:
                    with torch.no_grad():
                        t_pred = teacher_model(xb, envb).view(-1)
                    distill = loss_fn(preds, t_pred)
                    loss = loss + float(distill_weight) * distill
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            # Update EMA parameters after optimizer step
            if ema_model is not None:
                try:
                    ema_utils.update_ema(ema_model, model, decay=float(ema_decay))
                except Exception as exc:
                    logger.debug(f"EMA update failed: {exc}")
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        history["train_loss"].append(total / max(1, count))

        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, envb, yb in val_loader:
                xb = xb.to(device)
                envb = envb.to(device)
                yb = yb.to(device).view(-1)
                preds = model(xb, envb).view(-1)
                vloss = loss_fn(preds, yb)
                vtotal += float(vloss.item()) * xb.size(0)
                vcount += xb.size(0)
        val_loss = vtotal / max(1, vcount)
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
        y_pred = model(
            torch.from_numpy(x_val).to(device),
            torch.from_numpy(env_val).to(device),
        ).detach().cpu().numpy().reshape(-1)

    y_true = np.array(y_val, dtype=np.float32)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance_score": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "n_train": int(len(seq_train)),
        "n_val": int(len(seq_val)),
        "vocab_size": int(len(vocab)),
        "max_len": int(max_len),
        "device": str(device),
        "history": history,
        "suggestions": make_training_suggestions(history),
        "val_seqs": list(seq_val),
        "val_targets": list(y_true),
        "y_val": y_true,
        "y_pred": y_pred,
    }

    bundle = TransformerDrugModelBundle(
        model_state=model.state_dict(),
        vocab=dict(vocab),
        max_len=int(max_len),
        emb_dim=int(emb_dim),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        use_lstm=bool(use_lstm),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        lstm_bidirectional=bool(lstm_bidirectional),
        env_cols=list(env_cols),
        env_medians=env_medians,
        smiles_col=str(smiles_col),
        target_col=str(target_col),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    return bundle, metrics


def predict_transformer_one(
    bundle: TransformerDrugModelBundle,
    smiles: str,
    env_params: Optional[Dict[str, float]] = None,
    use_cuda: bool = False,
) -> float:
    env_params = env_params or {}
    env_vec: List[float] = []
    for c in bundle.env_cols:
        if c in env_params:
            env_vec.append(float(env_params[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.array(env_vec, dtype=np.float32).reshape(1, -1) if bundle.env_cols else np.zeros((1, 0), dtype=np.float32)
    token_ids = np.array([encode_smiles(smiles, bundle.vocab, int(bundle.max_len))], dtype=np.int64)

    device = _select_device(use_cuda)
    model = _build_model_from_bundle(bundle, env_dim=int(env_x.shape[1]), device=device)
    with torch.no_grad():
        y = model(torch.from_numpy(token_ids).to(device), torch.from_numpy(env_x).to(device)).cpu().numpy().reshape(-1)[0]
    return float(y)


def predict_transformer_batch(
    bundle: TransformerDrugModelBundle,
    smiles_list: Sequence[str],
    env_params_list: Optional[Sequence[Dict[str, float]]] = None,
    env_matrix: Optional[np.ndarray] = None,
    batch_size: int = 256,
    use_cuda: bool = False,
) -> np.ndarray:
    n = len(smiles_list)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    token_mat = np.array([encode_smiles(s, bundle.vocab, int(bundle.max_len)) for s in smiles_list], dtype=np.int64)

    env_cols = list(bundle.env_cols)
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

    device = _select_device(use_cuda)
    model = _build_model_from_bundle(bundle, env_dim=int(env_x.shape[1]), device=device)

    preds = np.empty((n,), dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        xb = torch.from_numpy(token_mat[start:end]).to(device)
        envb = torch.from_numpy(env_x[start:end]).to(device)
        with torch.no_grad():
            preds[start:end] = model(xb, envb).detach().cpu().numpy().reshape(-1)
    return preds
