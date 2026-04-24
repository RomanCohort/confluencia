from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast, Callable

import numpy as np

from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    DataLoader = cast(Any, None)
    TensorDataset = cast(Any, None)

try:
    from mamba_ssm import Mamba as _MambaBlock  # type: ignore[import-not-found]
    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover
    _MambaBlock = None
    HAS_MAMBA_SSM = False


AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
PAD_ID = 0


@dataclass(frozen=True)
class TorchMambaConfig:
    d_model: int = 96
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 64
    max_len: int = 1024
    seed: int = 42


class _FallbackMambaBlock(nn.Module):  # type: ignore[misc]
    """Fallback block used when mamba-ssm is unavailable."""

    def __init__(self, d_model: int, d_conv: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv // 2, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.dw(h).transpose(1, 2)
        h = self.pw(h)
        g = torch.sigmoid(self.gate(x))
        y = self.norm(x + self.drop(g * h))
        return y


class MambaSequenceRegressor(nn.Module):  # type: ignore[misc]
    def __init__(self, env_dim: int, cfg: TorchMambaConfig, use_real_mamba: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.env_dim = int(env_dim)
        self.use_real_mamba = bool(use_real_mamba)

        self.embedding = nn.Embedding(len(AA_ORDER) + 1, cfg.d_model, padding_idx=PAD_ID)
        self.in_norm = nn.LayerNorm(cfg.d_model)

        blocks = []
        for _ in range(cfg.n_layers):
            if self.use_real_mamba:
                blocks.append(
                    _MambaBlock(  # type: ignore[misc]
                        d_model=cfg.d_model,
                        d_state=cfg.d_state,
                        d_conv=cfg.d_conv,
                        expand=cfg.expand,
                    )
                )
            else:
                blocks.append(_FallbackMambaBlock(cfg.d_model, d_conv=max(3, cfg.d_conv + 1), dropout=cfg.dropout))

        self.blocks = nn.ModuleList(blocks)
        self.block_norm = nn.LayerNorm(cfg.d_model)

        self.env_proj = nn.Sequential(
            nn.Linear(self.env_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        ) if self.env_dim > 0 else None

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 4 + (cfg.d_model if self.env_dim > 0 else 0), cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, 1),
        )

    def _pool(self, h, pad_mask):
        # h: [B, L, D], pad_mask: [B, L] True means padding
        valid = (~pad_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)

        mean_pool = (h * valid).sum(dim=1) / denom

        x = h.transpose(1, 2)
        local = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1).transpose(1, 2)
        meso = F.avg_pool1d(x, kernel_size=11, stride=1, padding=5).transpose(1, 2)
        global_ = F.avg_pool1d(x, kernel_size=33, stride=1, padding=16).transpose(1, 2)

        local_pool = (local * valid).sum(dim=1) / denom
        meso_pool = (meso * valid).sum(dim=1) / denom
        global_pool = (global_ * valid).sum(dim=1) / denom

        return {
            "mean": mean_pool,
            "local": local_pool,
            "meso": meso_pool,
            "global": global_pool,
        }

    def forward(self, seq_tokens, env, return_parts: bool = False):
        pad_mask = seq_tokens.eq(PAD_ID)
        h = self.embedding(seq_tokens)
        h = self.in_norm(h)

        for block in self.blocks:
            h = block(h)
        h = self.block_norm(h)

        p = self._pool(h, pad_mask)
        parts = [p["mean"], p["local"], p["meso"], p["global"]]

        env_vec = None
        if self.env_proj is not None:
            env_vec = self.env_proj(env)
            parts.append(env_vec)

        z = torch.cat(parts, dim=-1)
        y = self.head(z).squeeze(-1)

        if return_parts:
            return y, p, env_vec
        return y


@dataclass
class TorchMambaBundle:
    model: MambaSequenceRegressor
    env_cols: List[str]
    env_mean: np.ndarray
    env_std: np.ndarray
    max_len: int
    history: Dict[str, List[float]]
    used_real_mamba: bool


@dataclass
class TorchSensitivity:
    prediction: float
    token_rows: List[Tuple[str, float, int]]
    env_rows: List[Tuple[str, float]]
    neighborhood: Dict[str, float]


def torch_available() -> bool:
    return torch is not None and nn is not None and F is not None


def real_mamba_available() -> bool:
    return bool(torch_available() and HAS_MAMBA_SSM)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_batch(seqs: List[str], max_len: int) -> np.ndarray:
    arr = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        s = str(seq or "").strip().upper().replace(" ", "")
        ids = [AA_TO_IDX.get(ch, PAD_ID) for ch in s[:max_len]]
        if ids:
            arr[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return arr


def _build_env(df, env_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not env_cols:
        z = np.zeros((len(df), 0), dtype=np.float32)
        return z, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)

    x = np.zeros((len(df), len(env_cols)), dtype=np.float32)
    for j, c in enumerate(env_cols):
        col = np.asarray(df[c], dtype=np.float32)
        x[:, j] = col

    mu = x.mean(axis=0).astype(np.float32)
    sigma = x.std(axis=0).astype(np.float32)
    sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)
    x = (x - mu) / sigma
    return x, mu, sigma


def _split_idx(n: int, seed: int, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 1
    val = idx[:n_val]
    tr = idx[n_val:] if n > 1 else idx
    if tr.size == 0:
        tr = val.copy()
    return tr, val


def train_torch_mamba(
    df,
    y: np.ndarray,
    env_cols: List[str],
    cfg: TorchMambaConfig,
    prefer_real_mamba: bool = True,
    # 检查点相关参数
    checkpoint_dir: Optional[str] = None,
    checkpoint_save_every: int = 5,
    checkpoint_keep_last: int = 3,
    resume_from: Optional[str] = None,
    # 回调函数
    on_epoch_end: Optional[Callable[[int, float, float, float], None]] = None,
):
    if not torch_available():
        raise RuntimeError("PyTorch is not available in this environment.")

    _set_seed(cfg.seed)
    use_real = bool(prefer_real_mamba and real_mamba_available())

    seqs = df["epitope_seq"].astype(str).tolist()
    seq_tok = _tokenize_batch(seqs, max_len=cfg.max_len)
    env_x, env_mu, env_std = _build_env(df, env_cols)

    y = np.asarray(y, dtype=np.float32).reshape(-1)

    tr_idx, va_idx = _split_idx(len(df), cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaSequenceRegressor(env_dim=env_x.shape[1], cfg=cfg, use_real_mamba=use_real).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_seq_t = torch.as_tensor(seq_tok, dtype=torch.long)
    x_env_t = torch.as_tensor(env_x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)

    tr_idx_t = torch.as_tensor(tr_idx, dtype=torch.long)
    va_idx_t = torch.as_tensor(va_idx, dtype=torch.long)

    ds_tr = TensorDataset(
        x_seq_t.index_select(0, tr_idx_t),
        x_env_t.index_select(0, tr_idx_t),
        y_t.index_select(0, tr_idx_t),
    )
    ds_va = TensorDataset(
        x_seq_t.index_select(0, va_idx_t),
        x_env_t.index_select(0, va_idx_t),
        y_t.index_select(0, va_idx_t),
    )

    dl_tr = DataLoader(ds_tr, batch_size=min(cfg.batch_size, len(ds_tr)), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=min(cfg.batch_size, len(ds_va)), shuffle=False)

    best_state = None
    best_val = float("inf")
    bad_rounds = 0
    patience = 8
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    start_epoch = 0
    _ckpt_files: List[str] = []

    # 从检查点恢复
    if resume_from:
        _ckpt = np.load(resume_from, allow_pickle=True)
        start_epoch = int(_ckpt["epoch"]) + 1
        best_val = float(_ckpt.get("best_val", float("inf")))
        bad_rounds = int(_ckpt.get("bad_rounds", 0))
        _ms = {}
        for _k in _ckpt.files:
            if _k.startswith("model_state."):
                _ms[_k[len("model_state."):]] = torch.from_numpy(_ckpt[_k])
        if _ms:
            model.load_state_dict(_ms)
        if "history_train_loss" in _ckpt.files:
            history["train_loss"] = list(_ckpt["history_train_loss"].astype(float))
        if "history_val_loss" in _ckpt.files:
            history["val_loss"] = list(_ckpt["history_val_loss"].astype(float))

    def _save_ckpt(epoch: int, is_best: bool = False):
        if not checkpoint_dir:
            return None
        from pathlib import Path as _P
        _d = _P(checkpoint_dir)
        _d.mkdir(parents=True, exist_ok=True)
        _sfx = "_best" if is_best else ""
        _p = _d / f"mamba_epoch{epoch:04d}{_sfx}.npz"
        _sd = {
            "epoch": np.array(epoch),
            "best_val": np.array(best_val),
            "bad_rounds": np.array(bad_rounds),
            "history_train_loss": np.array(history["train_loss"], dtype=np.float32),
            "history_val_loss": np.array(history["val_loss"], dtype=np.float32),
        }
        for _k, _v in model.state_dict().items():
            _sd[f"model_state.{_k}"] = _v.cpu().numpy()
        np.savez_compressed(_p, **_sd)
        _ckpt_files.append(str(_p))
        # 清理旧检查点
        if not is_best and checkpoint_keep_last > 0:
            _normals = [f for f in _ckpt_files if "_best" not in f]
            for _old in _normals[:-checkpoint_keep_last]:
                try:
                    _P(_old).unlink()
                    _ckpt_files.remove(_old)
                except Exception as exc:
                    logger.debug(f"Failed to delete old checkpoint {_old}: {exc}")
        return str(_p)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tr_losses: List[float] = []
        for b_seq, b_env, b_y in dl_tr:
            b_seq = b_seq.to(device)
            b_env = b_env.to(device)
            b_y = b_y.to(device)

            pred = model(b_seq, b_env)
            loss = F.mse_loss(pred, b_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for b_seq, b_env, b_y in dl_va:
                b_seq = b_seq.to(device)
                b_env = b_env.to(device)
                b_y = b_y.to(device)
                pred = model(b_seq, b_env)
                loss = F.mse_loss(pred, b_y)
                va_losses.append(float(loss.item()))

        tr_m = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_m = float(np.mean(va_losses)) if va_losses else tr_m
        history["train_loss"].append(tr_m)
        history["val_loss"].append(va_m)

        if va_m < best_val - 1e-6:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
            is_best = True
        else:
            bad_rounds += 1
            is_best = False

        # 回调通知（用于前端显示训练进度）
        if on_epoch_end:
            on_epoch_end(epoch, tr_m, va_m, best_val)

        # 保存检查点
        if checkpoint_dir:
            if is_best:
                _save_ckpt(epoch, is_best=True)
            elif epoch % checkpoint_save_every == 0 or epoch == cfg.epochs - 1:
                _save_ckpt(epoch, is_best=False)

        if bad_rounds >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = TorchMambaBundle(
        model=model,
        env_cols=list(env_cols),
        env_mean=env_mu,
        env_std=env_std,
        max_len=cfg.max_len,
        history=history,
        used_real_mamba=use_real,
    )
    return bundle


def predict_torch_mamba(bundle: TorchMambaBundle, df) -> np.ndarray:
    model = bundle.model
    device = next(model.parameters()).device

    seqs = df["epitope_seq"].astype(str).tolist()
    seq_tok = _tokenize_batch(seqs, max_len=bundle.max_len)

    if bundle.env_cols:
        env = np.zeros((len(df), len(bundle.env_cols)), dtype=np.float32)
        for j, c in enumerate(bundle.env_cols):
            env[:, j] = np.asarray(df[c], dtype=np.float32)
        env = (env - bundle.env_mean) / bundle.env_std
    else:
        env = np.zeros((len(df), 0), dtype=np.float32)

    x_seq = torch.as_tensor(seq_tok, dtype=torch.long, device=device)
    x_env = torch.as_tensor(env, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        pred = model(x_seq, x_env).detach().cpu().numpy().astype(np.float32)
    return pred


def sensitivity_torch_mamba(bundle: TorchMambaBundle, df_row, sample_index: int = 0, top_k: int = 20) -> TorchSensitivity:
    model = bundle.model
    device = next(model.parameters()).device

    seq = str(df_row["epitope_seq"])
    seq_tok = _tokenize_batch([seq], max_len=bundle.max_len)

    if bundle.env_cols:
        env = np.array([[float(df_row[c]) for c in bundle.env_cols]], dtype=np.float32)
        env = (env - bundle.env_mean.reshape(1, -1)) / bundle.env_std.reshape(1, -1)
    else:
        env = np.zeros((1, 0), dtype=np.float32)

    x_seq = torch.as_tensor(seq_tok, dtype=torch.long, device=device)
    x_env = torch.as_tensor(env, dtype=torch.float32, device=device)
    x_env.requires_grad_(True)

    model.eval()
    pred, pools, env_vec = model(x_seq, x_env, return_parts=True)
    y = pred.squeeze(0)

    grads = torch.autograd.grad(y, [pools["local"], pools["meso"], pools["global"], pools["mean"], x_env], allow_unused=True)

    local_imp = float((grads[0].abs() * pools["local"].abs()).sum().detach().cpu().item()) if grads[0] is not None else 0.0
    meso_imp = float((grads[1].abs() * pools["meso"].abs()).sum().detach().cpu().item()) if grads[1] is not None else 0.0
    global_imp = float((grads[2].abs() * pools["global"].abs()).sum().detach().cpu().item()) if grads[2] is not None else 0.0
    mean_imp = float((grads[3].abs() * pools["mean"].abs()).sum().detach().cpu().item()) if grads[3] is not None else 0.0

    env_rows: List[Tuple[str, float]] = []
    if grads[4] is not None and bundle.env_cols:
        g_env = grads[4].detach().cpu().numpy().reshape(-1)
        x_env_np = x_env.detach().cpu().numpy().reshape(-1)
        imp_env = np.abs(g_env * x_env_np)
        for c, v in zip(bundle.env_cols, imp_env.tolist()):
            env_rows.append((f"env_{c}", float(v)))
        env_rows = sorted(env_rows, key=lambda t: -t[1])

    # Token-level saliency on embeddings for interpretability.
    seq_ids = [AA_TO_IDX.get(ch, PAD_ID) for ch in seq.strip().upper().replace(" ", "")[: bundle.max_len]]
    token_rows: List[Tuple[str, float, int]] = []
    if seq_ids:
        emb = model.embedding(torch.as_tensor([seq_ids], dtype=torch.long, device=device)).detach().clone().requires_grad_(True)
        h = model.in_norm(emb)
        for block in model.blocks:
            h = block(h)
        h = model.block_norm(h)
        pad_mask = torch.zeros((1, len(seq_ids)), dtype=torch.bool, device=device)
        p = model._pool(h, pad_mask)
        parts = [p["mean"], p["local"], p["meso"], p["global"]]
        if model.env_proj is not None:
            parts.append(model.env_proj(x_env))
        z = torch.cat(parts, dim=-1)
        y2 = model.head(z).squeeze(-1)
        g_emb = torch.autograd.grad(y2, emb, retain_graph=False)[0]
        score = (g_emb.abs() * emb.abs()).sum(dim=-1).detach().cpu().numpy().reshape(-1)
        chars = seq.strip().upper().replace(" ", "")[: bundle.max_len]
        for i, (ch, sc) in enumerate(zip(chars, score.tolist())):
            token_rows.append((f"res_{i + 1}_{ch}", float(sc), i + 1))
        token_rows = sorted(token_rows, key=lambda t: -t[1])[: max(1, int(top_k))]

    neighborhood = {
        "token_mean": mean_imp,
        "local": local_imp,
        "meso": meso_imp,
        "global": global_imp,
    }

    return TorchSensitivity(
        prediction=float(y.detach().cpu().item()),
        token_rows=token_rows,
        env_rows=env_rows,
        neighborhood=neighborhood,
    )
