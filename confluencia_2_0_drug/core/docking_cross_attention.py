from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH = True
except Exception:
    TORCH = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
from sklearn.model_selection import train_test_split
from confluencia_shared.training import EarlyStopping, build_scheduler, make_training_suggestions
from confluencia_shared.metrics import rmse as _rmse


_SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def build_char_vocab(seqs: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in seqs:
        for ch in str(s):
            counts[ch] = counts.get(ch, 0) + 1
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
    for ch, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= int(min_freq):
            vocab.setdefault(ch, len(vocab))
    return vocab


def encode_sequence(seq: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    pad = vocab.get("<PAD>", 0)
    bos = vocab.get("<BOS>", 1)
    eos = vocab.get("<EOS>", 2)
    unk = vocab.get("<UNK>", 3)
    tokens = [bos] + [vocab.get(ch, unk) for ch in str(seq)] + [eos]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens = tokens + [pad] * (max_len - len(tokens))
    return tokens


class DockingDataset(Dataset):
    def __init__(
        self,
        ligands: Sequence[str],
        proteins: Sequence[str],
        targets: Optional[Sequence[float]],
        lig_vocab: Dict[str, int],
        prot_vocab: Dict[str, int],
        lig_max_len: int,
        prot_max_len: int,
    ) -> None:
        self.ligands = list(ligands)
        self.proteins = list(proteins)
        self.targets = list(targets) if targets is not None else None
        self.lig_vocab = lig_vocab
        self.prot_vocab = prot_vocab
        self.lig_max_len = int(lig_max_len)
        self.prot_max_len = int(prot_max_len)

    def __len__(self) -> int:
        return len(self.ligands)

    def __getitem__(self, idx: int):
        lig = encode_sequence(self.ligands[idx], self.lig_vocab, self.lig_max_len)
        prot = encode_sequence(self.proteins[idx], self.prot_vocab, self.prot_max_len)
        if self.targets is None:
            return torch.LongTensor(lig), torch.LongTensor(prot)
        return torch.LongTensor(lig), torch.LongTensor(prot), float(self.targets[idx])


def _collate(batch):
    if len(batch[0]) == 3:
        lig, prot, y = zip(*batch)
        return torch.stack(lig), torch.stack(prot), torch.tensor(y, dtype=torch.float32)
    lig, prot = zip(*batch)
    return torch.stack(lig), torch.stack(prot)


class CrossAttentionDockingRegressor(nn.Module):
    def __init__(
        self,
        lig_vocab_size: int,
        prot_vocab_size: int,
        lig_max_len: int,
        prot_max_len: int,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lig_max_len = int(lig_max_len)
        self.prot_max_len = int(prot_max_len)

        self.lig_emb = nn.Embedding(int(lig_vocab_size), int(emb_dim), padding_idx=0)
        self.prot_emb = nn.Embedding(int(prot_vocab_size), int(emb_dim), padding_idx=0)
        self.lig_pos = nn.Embedding(int(lig_max_len), int(emb_dim))
        self.prot_pos = nn.Embedding(int(prot_max_len), int(emb_dim))

        self.use_lstm = bool(use_lstm)
        self.lstm_bidirectional = bool(lstm_bidirectional)
        if self.use_lstm:
            lstm_out = int(lstm_hidden) * (2 if self.lstm_bidirectional else 1)
            self.lig_lstm = nn.LSTM(
                input_size=int(emb_dim),
                hidden_size=int(lstm_hidden),
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            self.prot_lstm = nn.LSTM(
                input_size=int(emb_dim),
                hidden_size=int(lstm_hidden),
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            self.lig_proj = nn.Linear(lstm_out, int(emb_dim))
            self.prot_proj = nn.Linear(lstm_out, int(emb_dim))
        else:
            self.lig_lstm = None
            self.prot_lstm = None
            self.lig_proj = None
            self.prot_proj = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(emb_dim),
            nhead=int(n_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.lig_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.prot_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(emb_dim),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(int(emb_dim), int(emb_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(emb_dim), 1),
        )

    def forward(self, lig_ids: torch.Tensor, prot_ids: torch.Tensor) -> torch.Tensor:
        device = lig_ids.device
        lig_pos = torch.arange(self.lig_max_len, device=device).unsqueeze(0).expand(lig_ids.size(0), -1)
        prot_pos = torch.arange(self.prot_max_len, device=device).unsqueeze(0).expand(prot_ids.size(0), -1)

        lig_x = self.lig_emb(lig_ids) + self.lig_pos(lig_pos)
        prot_x = self.prot_emb(prot_ids) + self.prot_pos(prot_pos)

        if self.lig_lstm is not None and self.prot_lstm is not None:
            lig_x, _ = self.lig_lstm(lig_x)
            prot_x, _ = self.prot_lstm(prot_x)
            lig_x = self.lig_proj(lig_x)
            prot_x = self.prot_proj(prot_x)

        lig_pad = lig_ids.eq(0)
        prot_pad = prot_ids.eq(0)

        lig_x = self.lig_encoder(lig_x, src_key_padding_mask=lig_pad)
        prot_x = self.prot_encoder(prot_x, src_key_padding_mask=prot_pad)

        cross_out, _ = self.cross_attn(lig_x, prot_x, prot_x, key_padding_mask=prot_pad)

        if lig_pad.any():
            mask = (~lig_pad).unsqueeze(-1)
            pooled = (cross_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            pooled = cross_out.mean(dim=1)

        return self.head(pooled).squeeze(-1)


@dataclass
class DockingModelBundle:
    model_state: Dict[str, Any]
    lig_vocab: Dict[str, int]
    prot_vocab: Dict[str, int]
    lig_max_len: int
    prot_max_len: int
    emb_dim: int
    n_heads: int
    n_layers: int
    ff_dim: int
    dropout: float
    use_lstm: bool
    lstm_hidden: int
    lstm_layers: int
    lstm_bidirectional: bool
    ligand_col: str
    protein_col: str
    target_col: str
    created_at: str
    version: int = 1


def _bundle_to_dict(bundle: DockingModelBundle) -> Dict[str, Any]:
    return {
        "model_state": bundle.model_state,
        "lig_vocab": dict(bundle.lig_vocab),
        "prot_vocab": dict(bundle.prot_vocab),
        "lig_max_len": int(bundle.lig_max_len),
        "prot_max_len": int(bundle.prot_max_len),
        "emb_dim": int(bundle.emb_dim),
        "n_heads": int(bundle.n_heads),
        "n_layers": int(bundle.n_layers),
        "ff_dim": int(bundle.ff_dim),
        "dropout": float(bundle.dropout),
        "use_lstm": bool(bundle.use_lstm),
        "lstm_hidden": int(bundle.lstm_hidden),
        "lstm_layers": int(bundle.lstm_layers),
        "lstm_bidirectional": bool(bundle.lstm_bidirectional),
        "ligand_col": str(bundle.ligand_col),
        "protein_col": str(bundle.protein_col),
        "target_col": str(bundle.target_col),
        "created_at": str(bundle.created_at),
        "version": int(bundle.version),
    }


def _bundle_from_dict(d: Dict[str, Any]) -> DockingModelBundle:
    return DockingModelBundle(
        model_state=d.get("model_state", {}),
        lig_vocab=dict(d.get("lig_vocab", {})),
        prot_vocab=dict(d.get("prot_vocab", {})),
        lig_max_len=int(d.get("lig_max_len", 128)),
        prot_max_len=int(d.get("prot_max_len", 512)),
        emb_dim=int(d.get("emb_dim", 128)),
        n_heads=int(d.get("n_heads", 4)),
        n_layers=int(d.get("n_layers", 2)),
        ff_dim=int(d.get("ff_dim", 256)),
        dropout=float(d.get("dropout", 0.1)),
        use_lstm=bool(d.get("use_lstm", False)),
        lstm_hidden=int(d.get("lstm_hidden", 128)),
        lstm_layers=int(d.get("lstm_layers", 1)),
        lstm_bidirectional=bool(d.get("lstm_bidirectional", True)),
        ligand_col=str(d.get("ligand_col", "smiles")),
        protein_col=str(d.get("protein_col", "protein")),
        target_col=str(d.get("target_col", "docking_score")),
        created_at=str(d.get("created_at", "")),
        version=int(d.get("version", 1)),
    )


def save_docking_bundle(bundle: DockingModelBundle, path: str) -> None:
    if not TORCH:
        raise RuntimeError("PyTorch is required to save model bundles.")
    torch.save(_bundle_to_dict(bundle), path)


def dump_docking_bundle(bundle: DockingModelBundle, fileobj) -> None:
    if not TORCH:
        raise RuntimeError("PyTorch is required to save model bundles.")
    torch.save(_bundle_to_dict(bundle), fileobj)


def load_docking_bundle(path: str) -> DockingModelBundle:
    if not TORCH:
        raise RuntimeError("PyTorch is required to load model bundles.")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("模型文件格式不正确")
    return _bundle_from_dict(obj)


def load_docking_bundle_from_bytes(data: bytes) -> DockingModelBundle:
    if not TORCH:
        raise RuntimeError("PyTorch is required to load model bundles.")
    import io

    buf = io.BytesIO(data)
    obj = torch.load(buf, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("模型文件格式不正确")
    return _bundle_from_dict(obj)


def _select_device(use_cuda: bool) -> "torch.device":
    if bool(use_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_docking_bundle(
    df: pd.DataFrame,
    ligand_col: str = "smiles",
    protein_col: str = "protein",
    target_col: str = "docking_score",
    lig_max_len: int = 128,
    prot_max_len: int = 512,
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
    batch_size: int = 32,
    epochs: int = 30,
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
    teacher_bundle: Optional[DockingModelBundle] = None,
    distill_weight: float = 0.2,
) -> Tuple[DockingModelBundle, Dict[str, float]]:
    if not TORCH:
        raise RuntimeError("PyTorch is required for docking model training.")

    df = df.copy()
    df = df[[ligand_col, protein_col, target_col]].dropna()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    ligands = df[ligand_col].astype(str).tolist()
    proteins = df[protein_col].astype(str).tolist()
    targets = df[target_col].astype(float).tolist()

    lig_vocab = build_char_vocab(ligands, min_freq=int(min_char_freq))
    prot_vocab = build_char_vocab(proteins, min_freq=int(min_char_freq))

    idx = list(range(len(ligands)))
    train_idx, val_idx = train_test_split(idx, test_size=float(test_size), random_state=int(random_state))

    train_lig = [ligands[i] for i in train_idx]
    train_prot = [proteins[i] for i in train_idx]
    train_y = [targets[i] for i in train_idx]

    val_lig = [ligands[i] for i in val_idx]
    val_prot = [proteins[i] for i in val_idx]
    val_y = [targets[i] for i in val_idx]

    device = _select_device(bool(use_cuda))

    model = CrossAttentionDockingRegressor(
        lig_vocab_size=len(lig_vocab),
        prot_vocab_size=len(prot_vocab),
        lig_max_len=int(lig_max_len),
        prot_max_len=int(prot_max_len),
        emb_dim=int(emb_dim),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        use_lstm=bool(use_lstm),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        lstm_bidirectional=bool(lstm_bidirectional),
    ).to(device)

    train_ds = DockingDataset(train_lig, train_prot, train_y, lig_vocab, prot_vocab, lig_max_len, prot_max_len)
    val_ds = DockingDataset(val_lig, val_prot, val_y, lig_vocab, prot_vocab, lig_max_len, prot_max_len)
    train_dl = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=_collate)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    scheduler = build_scheduler(opt, lr_schedule, epochs=int(epochs), step_size=int(step_size), gamma=float(gamma), min_lr=float(min_lr))
    stopper = EarlyStopping(patience=int(early_stopping_patience), mode="min")

    teacher_model: Optional[nn.Module] = None
    if teacher_bundle is not None:
        teacher_model = _build_model_from_bundle(teacher_bundle).to(device)
        teacher_model.eval()

    scaler = torch.amp.GradScaler(enabled=bool(amp) and device.type == "cuda")

    best_state = None
    best_val = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(int(epochs)):
        model.train()
        trun = 0.0
        tcount = 0
        for lig_x, prot_x, y in train_dl:
            lig_x = lig_x.to(device)
            prot_x = prot_x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device.type), enabled=bool(amp) and device.type == "cuda"):
                pred = model(lig_x, prot_x)
                loss = loss_fn(pred, y)
                if teacher_model is not None and float(distill_weight) > 0:
                    with torch.no_grad():
                        t_pred = teacher_model(lig_x, prot_x)
                    distill = loss_fn(pred, t_pred)
                    loss = loss + float(distill_weight) * distill
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if float(max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            scaler.step(opt)
            scaler.update()
            trun += float(loss.item()) * lig_x.size(0)
            tcount += lig_x.size(0)

        history["train_loss"].append(trun / max(1, tcount))

        model.eval()
        vrun = 0.0
        vcount = 0
        with torch.no_grad():
            for lig_x, prot_x, y in val_dl:
                lig_x = lig_x.to(device)
                prot_x = prot_x.to(device)
                y = y.to(device)
                pred = model(lig_x, prot_x)
                vrun += loss_fn(pred, y).item() * lig_x.size(0)
                vcount += lig_x.size(0)
        val_loss = vrun / max(1, vcount)
        history["val_loss"].append(val_loss)
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        if stopper.step(val_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_preds = []
    model.eval()
    with torch.no_grad():
        for lig_x, prot_x, y in val_dl:
            lig_x = lig_x.to(device)
            prot_x = prot_x.to(device)
            pred = model(lig_x, prot_x).detach().cpu().numpy().tolist()
            val_preds.extend(pred)

    y_true = np.asarray(val_y, dtype=np.float32)
    y_pred = np.asarray(val_preds, dtype=np.float32)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance_score": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "val_size": float(len(val_y)),
        "history": history,
        "suggestions": make_training_suggestions(history),
        "y_val": y_true,
        "y_pred": y_pred,
    }

    bundle = DockingModelBundle(
        model_state=model.state_dict(),
        lig_vocab=lig_vocab,
        prot_vocab=prot_vocab,
        lig_max_len=int(lig_max_len),
        prot_max_len=int(prot_max_len),
        emb_dim=int(emb_dim),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        use_lstm=bool(use_lstm),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        lstm_bidirectional=bool(lstm_bidirectional),
        ligand_col=str(ligand_col),
        protein_col=str(protein_col),
        target_col=str(target_col),
        created_at=datetime.utcnow().isoformat(),
    )

    return bundle, metrics


def _build_model_from_bundle(bundle: DockingModelBundle) -> "CrossAttentionDockingRegressor":
    model = CrossAttentionDockingRegressor(
        lig_vocab_size=len(bundle.lig_vocab),
        prot_vocab_size=len(bundle.prot_vocab),
        lig_max_len=int(bundle.lig_max_len),
        prot_max_len=int(bundle.prot_max_len),
        emb_dim=int(bundle.emb_dim),
        n_heads=int(bundle.n_heads),
        n_layers=int(bundle.n_layers),
        ff_dim=int(bundle.ff_dim),
        dropout=float(bundle.dropout),
        use_lstm=bool(bundle.use_lstm),
        lstm_hidden=int(bundle.lstm_hidden),
        lstm_layers=int(bundle.lstm_layers),
        lstm_bidirectional=bool(bundle.lstm_bidirectional),
    )
    model.load_state_dict(bundle.model_state)
    model.eval()
    return model


def predict_docking_one(
    bundle: DockingModelBundle,
    smiles: str,
    protein: str,
    use_cuda: bool = False,
) -> float:
    if not TORCH:
        raise RuntimeError("PyTorch is required for docking prediction.")
    device = _select_device(bool(use_cuda))
    model = _build_model_from_bundle(bundle).to(device)

    lig = torch.LongTensor([encode_sequence(smiles, bundle.lig_vocab, bundle.lig_max_len)]).to(device)
    prot = torch.LongTensor([encode_sequence(protein, bundle.prot_vocab, bundle.prot_max_len)]).to(device)
    with torch.no_grad():
        pred = model(lig, prot)
    return float(pred.detach().cpu().item())


def predict_docking_batch(
    bundle: DockingModelBundle,
    ligands: Sequence[str],
    proteins: Sequence[str],
    batch_size: int = 64,
    use_cuda: bool = False,
) -> List[float]:
    if not TORCH:
        raise RuntimeError("PyTorch is required for docking prediction.")
    device = _select_device(bool(use_cuda))
    model = _build_model_from_bundle(bundle).to(device)

    ds = DockingDataset(ligands, proteins, None, bundle.lig_vocab, bundle.prot_vocab, bundle.lig_max_len, bundle.prot_max_len)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=_collate)

    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for lig_x, prot_x in dl:
            lig_x = lig_x.to(device)
            prot_x = prot_x.to(device)
            out = model(lig_x, prot_x).detach().cpu().numpy().tolist()
            preds.extend([float(v) for v in out])
    return preds
