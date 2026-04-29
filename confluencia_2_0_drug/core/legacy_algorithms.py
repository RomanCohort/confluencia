from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from confluencia_shared.metrics import rmse as _rmse

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = cast(Any, None)
    nn = cast(Any, None)
    DataLoader = cast(Any, None)
    TensorDataset = cast(Any, None)


@dataclass(frozen=True)
class LegacyAlgorithmConfig:
    random_state: int = 42
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    torch_hidden_1: int = 256
    torch_hidden_2: int = 128
    max_len: int = 128
    emb_dim: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1


def _build_sklearn_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, max_depth=6)
    if name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    if name == "rf":
        # Keep RF single-threaded to avoid joblib ThreadPool issues in some Python 3.13 environments.
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed, n_jobs=1)
    if name == "ridge":
        return Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))])
    if name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        max_iter=1200,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported legacy sklearn model: {name}")


def _torch_available() -> bool:
    return bool(torch is not None and nn is not None and DataLoader is not None and TensorDataset is not None)


def _train_predict_torch_mlp(X: np.ndarray, y: np.ndarray, cfg: LegacyAlgorithmConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    if not _torch_available():
        return np.zeros((len(y),), dtype=np.float32), {"fallback": 1.0}

    rng = np.random.default_rng(int(cfg.random_state))
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(round(0.2 * len(y)))) if len(y) > 1 else 1
    va = idx[:n_val]
    tr = idx[n_val:] if len(y) > 1 else idx
    if tr.size == 0:
        tr = va.copy()

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (X - mu) / sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(Xn.shape[1], int(cfg.torch_hidden_1)),
        nn.ReLU(),
        nn.Dropout(float(cfg.dropout)),
        nn.Linear(int(cfg.torch_hidden_1), int(cfg.torch_hidden_2)),
        nn.ReLU(),
        nn.Dropout(float(cfg.dropout)),
        nn.Linear(int(cfg.torch_hidden_2), 1),
    ).to(device)

    x_t = torch.as_tensor(Xn, dtype=torch.float32)
    y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
    tr_t = torch.as_tensor(tr, dtype=torch.long)
    va_t = torch.as_tensor(va, dtype=torch.long)

    ds_tr = TensorDataset(x_t.index_select(0, tr_t), y_t.index_select(0, tr_t))
    ds_va = TensorDataset(x_t.index_select(0, va_t), y_t.index_select(0, va_t))
    dl_tr = DataLoader(ds_tr, batch_size=min(int(cfg.batch_size), len(ds_tr)), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=min(int(cfg.batch_size), len(ds_va)), shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    bad = 0
    history: Dict[str, float] = {"train_loss_last": 0.0, "val_loss_last": 0.0}

    for _ in range(int(cfg.epochs)):
        model.train()
        tr_losses: List[float] = []
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                va_losses.append(float(torch.mean((pred - yb) ** 2).item()))

        tr_l = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_l = float(np.mean(va_losses)) if va_losses else tr_l
        history["train_loss_last"] = tr_l
        history["val_loss_last"] = va_l

        if va_l < best_val - 1e-6:
            best_val = va_l
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
    return pred.astype(np.float32), history


def _build_vocab(smiles_list: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<EOS>": 3}
    for s in smiles_list:
        for ch in str(s):
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def _encode_smiles(smiles: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab["<CLS>"]] + [vocab.get(ch, vocab["<UNK>"]) for ch in str(smiles)] + [vocab["<EOS>"]]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab["<PAD>"]] * (max_len - len(ids)))
    return ids


def _train_predict_transformer(
    smiles_list: List[str],
    env_x: np.ndarray,
    y: np.ndarray,
    cfg: LegacyAlgorithmConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not _torch_available():
        return np.zeros((len(y),), dtype=np.float32), {"fallback": 1.0}

    rng = np.random.default_rng(int(cfg.random_state))
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(round(0.2 * len(y)))) if len(y) > 1 else 1
    va = idx[:n_val]
    tr = idx[n_val:] if len(y) > 1 else idx
    if tr.size == 0:
        tr = va.copy()

    vocab = _build_vocab(smiles_list)
    tok = np.asarray([_encode_smiles(s, vocab, int(cfg.max_len)) for s in smiles_list], dtype=np.int64)

    env_mu = env_x.mean(axis=0, keepdims=True) if env_x.size else np.zeros((1, 0), dtype=np.float32)
    env_sigma = env_x.std(axis=0, keepdims=True) if env_x.size else np.ones((1, 0), dtype=np.float32)
    env_sigma = np.where(env_sigma < 1e-6, 1.0, env_sigma)
    env_n = (env_x - env_mu) / env_sigma if env_x.size else env_x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = nn.Embedding(len(vocab), int(cfg.emb_dim), padding_idx=0)
    pos = nn.Embedding(int(cfg.max_len), int(cfg.emb_dim))
    enc_layer = nn.TransformerEncoderLayer(
        d_model=int(cfg.emb_dim),
        nhead=int(cfg.n_heads),
        dim_feedforward=int(cfg.ff_dim),
        dropout=float(cfg.dropout),
        batch_first=True,
    )
    enc = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.n_layers))
    in_dim = int(cfg.emb_dim) + int(env_n.shape[1])
    head = nn.Sequential(
        nn.Linear(in_dim, int(cfg.ff_dim)),
        nn.ReLU(),
        nn.Dropout(float(cfg.dropout)),
        nn.Linear(int(cfg.ff_dim), 1),
    )
    model = nn.ModuleDict({"emb": emb, "pos": pos, "enc": enc, "head": head}).to(device)

    t_tok = torch.as_tensor(tok, dtype=torch.long)
    t_env = torch.as_tensor(env_n, dtype=torch.float32)
    t_y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
    tr_t = torch.as_tensor(tr, dtype=torch.long)
    va_t = torch.as_tensor(va, dtype=torch.long)

    ds_tr = TensorDataset(t_tok.index_select(0, tr_t), t_env.index_select(0, tr_t), t_y.index_select(0, tr_t))
    ds_va = TensorDataset(t_tok.index_select(0, va_t), t_env.index_select(0, va_t), t_y.index_select(0, va_t))

    dl_tr = DataLoader(ds_tr, batch_size=min(int(cfg.batch_size), len(ds_tr)), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=min(int(cfg.batch_size), len(ds_va)), shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    bad = 0
    history: Dict[str, float] = {"train_loss_last": 0.0, "val_loss_last": 0.0}

    pos_ids = torch.arange(int(cfg.max_len), dtype=torch.long).unsqueeze(0)

    def _forward(b_tok, b_env):
        bsz = b_tok.shape[0]
        p = pos_ids.to(b_tok.device).expand(bsz, -1)
        x = model["emb"](b_tok) + model["pos"](p)
        pad_mask = b_tok.eq(0)
        x = model["enc"](x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        feats = torch.cat([pooled, b_env], dim=1) if b_env.numel() else pooled
        return model["head"](feats)

    for _ in range(int(cfg.epochs)):
        model.train()
        tr_losses: List[float] = []
        for b_tok, b_env, b_y in dl_tr:
            b_tok = b_tok.to(device)
            b_env = b_env.to(device)
            b_y = b_y.to(device)
            pred = _forward(b_tok, b_env)
            loss = torch.mean((pred - b_y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for b_tok, b_env, b_y in dl_va:
                b_tok = b_tok.to(device)
                b_env = b_env.to(device)
                b_y = b_y.to(device)
                pred = _forward(b_tok, b_env)
                va_losses.append(float(torch.mean((pred - b_y) ** 2).item()))

        tr_l = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_l = float(np.mean(va_losses)) if va_losses else tr_l
        history["train_loss_last"] = tr_l
        history["val_loss_last"] = va_l

        if va_l < best_val - 1e-6:
            best_val = va_l
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = _forward(t_tok.to(device), t_env.to(device)).detach().cpu().numpy().reshape(-1)
    return pred.astype(np.float32), history


def train_predict_legacy_backend(
    work_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    env_cols: List[str],
    model_backend: str,
    cfg: LegacyAlgorithmConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cfg = cfg or LegacyAlgorithmConfig()
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if model_backend in {"hgb", "gbr", "rf", "ridge", "mlp"}:
        model = _build_sklearn_model(model_backend, int(cfg.random_state))
        model.fit(X, y)
        pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
        return pred, {
            "train_rmse": _rmse(y, pred),
            "train_mae": float(np.mean(np.abs(pred - y))),
            "train_r2": float(1.0 - np.sum((pred - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-8)),
        }

    if model_backend == "torch_mlp":
        pred, history = _train_predict_torch_mlp(X=X, y=y, cfg=cfg)
        if float(history.get("fallback", 0.0)) > 0:
            model = _build_sklearn_model("hgb", int(cfg.random_state))
            model.fit(X, y)
            pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
            history = {"fallback_to_hgb": 1.0}
        history.update(
            {
                "train_rmse": _rmse(y, pred),
                "train_mae": float(np.mean(np.abs(pred - y))),
                "train_r2": float(1.0 - np.sum((pred - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-8)),
            }
        )
        return pred, history

    if model_backend == "transformer":
        smiles = work_df["smiles"].astype(str).tolist()
        if env_cols:
            env_x = work_df[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        else:
            env_x = np.zeros((len(work_df), 0), dtype=np.float32)
        pred, history = _train_predict_transformer(smiles_list=smiles, env_x=env_x, y=y, cfg=cfg)
        if float(history.get("fallback", 0.0)) > 0:
            model = _build_sklearn_model("hgb", int(cfg.random_state))
            model.fit(X, y)
            pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
            history = {"fallback_to_hgb": 1.0}
        history.update(
            {
                "train_rmse": _rmse(y, pred),
                "train_mae": float(np.mean(np.abs(pred - y))),
                "train_r2": float(1.0 - np.sum((pred - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-8)),
            }
        )
        return pred, history

    raise ValueError(f"Unsupported model backend: {model_backend}")
