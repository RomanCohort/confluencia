"""Train Protein-Ligand interaction model on synthetic data.

Generates simple synthetic pockets and ligands (no realistic physics) to exercise training loop.
Supports CLI args for LJ/electrostatic parameters which are passed into PhysicsMessageGNN.

Usage:
    python src/pl_train.py --epochs 20 --out pl_model.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .pl_interaction import ProteinLigandInteractionModel, protein_from_coords
from confluencia_shared.training import EarlyStopping, build_scheduler
from confluencia_shared.utils import ema as ema_utils
from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)
from .gnn import mol_to_graph, PhysicsMessageGNN

from rdkit import Chem
try:
    from rdkit.Chem import AllChem as _AllChem
    AllChem: Any = _AllChem
except Exception:  # pragma: no cover
    AllChem = None  # type: ignore


class SyntheticPLDataset(Dataset):
    """Creates synthetic protein-ligand pairs with a scalar interaction score.

    Score is a toy function: negative average distance between ligand atoms and nearest protein atoms
    plus a random small noise. This is only to validate training pipeline.
    """

    def __init__(self, smiles_list: List[str], n_prot_atoms: int = 30, box_size: float = 10.0, seed: int = 42):
        self.smiles = smiles_list
        self.n_prot = n_prot_atoms
        self.box = box_size
        self.rng = np.random.default_rng(int(seed))
        self._lig_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._build_lig_cache()

    def _build_lig_cache(self) -> None:
        for s in sorted(set(self.smiles)):
            X_l, A_l, mol = mol_to_graph(s)
            # try to embed conformer once
            try:
                if AllChem is None:
                    raise RuntimeError("RDKit AllChem 不可用")
                AllChem.EmbedMolecule(mol, randomSeed=42)
                conf = mol.GetConformer()
                lcoords = np.array(
                    [[float(conf.GetAtomPosition(i).x), float(conf.GetAtomPosition(i).y), float(conf.GetAtomPosition(i).z)] for i in range(mol.GetNumAtoms())],
                    dtype=np.float32,
                )
            except Exception as exc:
                logger.debug(f"RDKit 3D coords fallback for SMILES: {exc}")
                # fallback: small random ligand coords (fixed once per SMILES)
                lcoords = self.rng.random((max(1, X_l.shape[0]), 3), dtype=np.float32)
            self._lig_cache[s] = (X_l, A_l, lcoords)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        X_l, A_l, lcoords = self._lig_cache[s]

        # random pocket
        coords = self.rng.random((self.n_prot, 3), dtype=np.float32) * self.box
        atom_types = self.rng.choice(["C", "N", "O", "S", "H"], size=self.n_prot, replace=True).tolist()

        # compute score proxy
        prot_center = coords.mean(axis=0)
        dist = np.linalg.norm(lcoords - prot_center, axis=-1).mean()
        score = float(-dist + float(self.rng.normal(0.0, 0.1)))
        return X_l, A_l, lcoords, atom_types, coords, score


def collate_fn(batch):
    return batch


def build_tensors_from_item(item):
    X_l, A_l, lcoords, atom_types, coords, score = item
    # append ligand coords if available
    if lcoords is not None and len(lcoords) == X_l.shape[0]:
        X_l = np.hstack([X_l, lcoords])
    else:
        X_l = np.hstack([X_l, np.zeros((X_l.shape[0], 3), dtype=np.float32)])

    X_p, A_p, dmat_p = protein_from_coords(atom_types, coords)

    return (
        torch.from_numpy(X_l).float(),
        torch.from_numpy(A_l).float(),
        None,
        torch.from_numpy(X_p).float(),
        torch.from_numpy(A_p).float(),
        None,
        torch.tensor(float(score), dtype=torch.float32),
    )


def train(
    smiles_list: List[str],
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    out: str = "pl_model.pth",
    lj_epsilon: float = 0.1,
    lj_sigma: float = 3.5,
    dielectric: float = 80.0,
    dropout: float = 0.1,
    use_lstm: bool = False,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    lstm_bidirectional: bool = True,
    use_cuda: bool = False,
    seed: int = 42,
    weight_decay: float = 1e-4,
    lr_schedule: str = "cosine",
    step_size: int = 20,
    gamma: float = 0.5,
    min_lr: float = 1e-6,
    early_stopping_patience: int = 10,
    max_grad_norm: float = 5.0,
    teacher_path: Optional[str] = None,
    distill_weight: float = 0.2,
    use_ema: bool = False,
    ema_decay: float = 0.99,
) -> None:
    ds = SyntheticPLDataset(smiles_list, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # build model based on first sample shapes
    sample = ds[0]
    X_l, A_l, _, X_p, A_p, _, _ = build_tensors_from_item(sample)
    lig_in = X_l.shape[1]
    prot_in = X_p.shape[1]
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = ProteinLigandInteractionModel(
        lig_in,
        prot_in,
        hidden=64,
        dropout=float(dropout),
        use_lstm=bool(use_lstm),
        lstm_hidden=int(lstm_hidden),
        lstm_layers=int(lstm_layers),
        lstm_bidirectional=bool(lstm_bidirectional),
    ).to(device)

    teacher_model: Optional[ProteinLigandInteractionModel] = None
    if teacher_path:
        try:
            teacher_model = ProteinLigandInteractionModel(
                lig_in,
                prot_in,
                hidden=64,
                dropout=float(dropout),
                use_lstm=bool(use_lstm),
                lstm_hidden=int(lstm_hidden),
                lstm_layers=int(lstm_layers),
                lstm_bidirectional=bool(lstm_bidirectional),
            ).to(device)
            state = torch.load(str(teacher_path), map_location=device)
            if isinstance(state, dict):
                teacher_model.load_state_dict(state, strict=False)
            teacher_model.eval()
        except Exception as exc:
            logger.debug(f"Teacher model load failed: {exc}")
            teacher_model = None

    # Parameter-level EMA teacher: clone current student and update via EMA after each step
    ema_teacher: Optional[ProteinLigandInteractionModel] = None
    if use_ema:
        try:
            ema_teacher = ema_utils.clone_model_for_ema(model)
            # if a teacher_path was provided, try to load into ema_teacher as init
            if teacher_path:
                try:
                    state = torch.load(str(teacher_path), map_location=device)
                    if isinstance(state, dict):
                        ema_teacher.load_state_dict(state, strict=False)
                except Exception as exc:
                    logger.debug(f"EMA teacher state dict load failed: {exc}")
            ema_teacher.to(device)
            ema_teacher.eval()
        except Exception as exc:
            logger.debug(f"EMA teacher setup failed: {exc}")
            ema_teacher = None
    try:
        if isinstance(model.lig_enc, PhysicsMessageGNN):
            model.lig_enc.lj_epsilon = lj_epsilon
            model.lig_enc.lj_sigma = lj_sigma
            model.lig_enc.dielectric = dielectric
        if isinstance(model.prot_enc, PhysicsMessageGNN):
            model.prot_enc.lj_epsilon = lj_epsilon
            model.prot_enc.lj_sigma = lj_sigma
            model.prot_enc.dielectric = dielectric
    except Exception as exc:
        logger.debug(f"Physics param injection failed: {exc}")    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    scheduler = build_scheduler(opt, lr_schedule, epochs=int(epochs), step_size=int(step_size), gamma=float(gamma), min_lr=float(min_lr))
    stopper = EarlyStopping(patience=int(early_stopping_patience), mode="min")
    history = {"train_loss": [], "val_loss": []}

    # simple train/val split for early stopping
    idx = np.arange(len(ds))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:] if split < len(idx) else idx[:1]

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in dl:
            opt.zero_grad()
            batch_loss = torch.zeros((), device=device)
            for item in batch:
                X_l_t, A_l_t, _, X_p_t, A_p_t, _, y = build_tensors_from_item(item)
                X_l_t = X_l_t.to(device)
                A_l_t = A_l_t.to(device)
                X_p_t = X_p_t.to(device)
                A_p_t = A_p_t.to(device)
                y = y.to(device)
                prot_coords_t = torch.from_numpy(item[4]).float().to(device)
                pred = model(X_l_t, A_l_t, None, X_p_t, A_p_t, None, prot_coords=prot_coords_t)
                loss = loss_fn(pred.unsqueeze(0), y.unsqueeze(0))
                # Distillation can use a static teacher (loaded) or EMA teacher if enabled
                t_pred = None
                if ema_teacher is not None and float(distill_weight) > 0:
                    with torch.no_grad():
                        t_pred = ema_teacher(X_l_t, A_l_t, None, X_p_t, A_p_t, None, prot_coords=prot_coords_t)
                elif teacher_model is not None and float(distill_weight) > 0:
                    with torch.no_grad():
                        t_pred = teacher_model(X_l_t, A_l_t, None, X_p_t, A_p_t, None, prot_coords=prot_coords_t)

                if t_pred is not None:
                    distill = loss_fn(pred.unsqueeze(0), t_pred.unsqueeze(0))
                    loss = loss + float(distill_weight) * distill
                batch_loss += loss
                n += 1
            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            if float(max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            opt.step()
            # Update EMA teacher parameters after optimizer step
            if ema_teacher is not None:
                try:
                    ema_utils.update_ema(ema_teacher, model, decay=float(ema_decay))
                except Exception as exc:
                    logger.debug(f"EMA update failed: {exc}") += float(batch_loss.item())
        history["train_loss"].append(total / max(1, n))

        # quick val pass
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for i in val_idx:
                item = ds[int(i)]
                X_l_t, A_l_t, _, X_p_t, A_p_t, _, y = build_tensors_from_item(item)
                X_l_t = X_l_t.to(device)
                A_l_t = A_l_t.to(device)
                X_p_t = X_p_t.to(device)
                A_p_t = A_p_t.to(device)
                y = y.to(device)
                prot_coords_t = torch.from_numpy(item[4]).float().to(device)
                pred = model(X_l_t, A_l_t, None, X_p_t, A_p_t, None, prot_coords=prot_coords_t)
                loss = loss_fn(pred.unsqueeze(0), y.unsqueeze(0))
                vtotal += float(loss.item())
                vcount += 1
        val_loss = vtotal / max(1, vcount)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {ep+1}/{epochs} loss={history['train_loss'][-1]:.6f} val={val_loss:.6f}")
        if stopper.step(val_loss):
            break

    # Save student model
    torch.save(model.state_dict(), out)
    # Optionally save EMA teacher as well
    if ema_teacher is not None:
        try:
            torch.save(ema_teacher.state_dict(), str(Path(out).with_name(Path(out).stem + "_ema" + Path(out).suffix)))
        except Exception as exc:
            logger.debug(f"EMA model save failed: {exc}")
    print("Saved model to", out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='pl_model.pth')
    p.add_argument('--lj_eps', type=float, default=0.1)
    p.add_argument('--lj_sigma', type=float, default=3.5)
    p.add_argument('--dielectric', type=float, default=80.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--use_lstm', action='store_true')
    p.add_argument('--lstm_hidden', type=int, default=128)
    p.add_argument('--lstm_layers', type=int, default=1)
    p.add_argument('--lstm_bi', action='store_true')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lr_schedule', type=str, default='cosine')
    p.add_argument('--step_size', type=int, default=20)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--early_patience', type=int, default=10)
    p.add_argument('--max_grad_norm', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--teacher_path', type=str, default="", help='Optional teacher model state_dict path')
    p.add_argument('--distill_weight', type=float, default=0.2, help='Distillation loss weight')
    args = p.parse_args(argv)

    # example SMILES set (small)
    smiles_list = [
        'CCO', 'CC', 'CCC', 'CCN', 'CCOCC', 'c1ccccc1', 'CC(=O)O', 'CC(C)O', 'OCCO', 'CCS'
    ] * 10

    train(
        smiles_list,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        out=args.out,
        lj_epsilon=args.lj_eps,
        lj_sigma=args.lj_sigma,
        dielectric=args.dielectric,
        use_cuda=bool(args.cuda),
        seed=int(args.seed),
        dropout=float(args.dropout),
        use_lstm=bool(args.use_lstm),
        lstm_hidden=int(args.lstm_hidden),
        lstm_layers=int(args.lstm_layers),
        lstm_bidirectional=bool(args.lstm_bi),
        weight_decay=float(args.weight_decay),
        lr_schedule=str(args.lr_schedule),
        step_size=int(args.step_size),
        gamma=float(args.gamma),
        min_lr=float(args.min_lr),
        early_stopping_patience=int(args.early_patience),
        max_grad_norm=float(args.max_grad_norm),
        teacher_path=str(args.teacher_path) if str(args.teacher_path).strip() else None,
        distill_weight=float(args.distill_weight),
    )


if __name__ == '__main__':
    main()
