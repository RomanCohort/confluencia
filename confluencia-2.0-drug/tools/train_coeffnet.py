"""Train CoeffNet from CSV experimental data.

CSV must contain a SMILES column and numeric columns for D, Vmax, Km (names configurable).

Usage:
    python src/train_coeffnet.py --csv data/coeffs.csv --smiles smiles --cols D Vmax Km --out coeffnet.pth
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .gnn import mol_to_graph, EnhancedGNN, readout_mean
from .pinn import CoeffNet
from src.common import ema as ema_utils


class MolCoeffDataset(Dataset):
    def __init__(self, df: pd.DataFrame, smiles_col: str, coeff_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.coeff_cols = coeff_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = row[self.smiles_col]
        coeffs = row[self.coeff_cols].astype(float).values.astype('float32')
        return s, coeffs


def collate_fn(batch):
    smiles = [b[0] for b in batch]
    coeffs = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return smiles, coeffs


def build_encoder_from_smiles(sample_smiles: str, hidden_dim: int = 128, steps: int = 3) -> EnhancedGNN:
    X, A, _ = mol_to_graph(sample_smiles)
    in_feats = X.shape[1]
    enc = EnhancedGNN(in_feats, hidden_dim, steps=steps)
    return enc


def smiles_batch_to_embeddings(encoder: EnhancedGNN, smiles_list: List[str], device: torch.device):
    embs = []
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        for s in smiles_list:
            X_np, A_np, mol = mol_to_graph(s)
            x = torch.from_numpy(X_np).float().to(device)
            adj = torch.from_numpy(A_np).float().to(device)
            node_emb = encoder(x, adj)
            mol_emb = readout_mean(node_emb)
            embs.append(mol_emb.cpu())
    return torch.stack(embs, dim=0)


def train(df_path: str, smiles_col: str, coeff_cols: List[str], out_path: str, epochs: int = 50, batch_size: int = 32, device: str = 'cpu'):
    df = pd.read_csv(df_path)
    if len(df) == 0:
        raise ValueError('Empty CSV')
    ds = MolCoeffDataset(df, smiles_col, coeff_cols)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # build encoder from first SMILES
    sample_smiles = df.iloc[0][smiles_col]
    enc = build_encoder_from_smiles(sample_smiles)
    enc = enc.to(device)

    # build coeff net
    mol_dim = enc.hidden_dim
    coeff_net = CoeffNet(mol_dim).to(device)

    opt = torch.optim.Adam(list(coeff_net.parameters()) + list(enc.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    # EMA teacher for encoder + coeff_net
    ema_enc = None
    ema_coeff = None
    use_ema = False
    ema_decay = 0.99
    # Allow enabling via environment var for manual runs, or modify call signature as needed.
    try:
        import os
        use_ema = bool(int(os.environ.get('COEFFNET_USE_EMA', '0')))
        ema_decay = float(os.environ.get('COEFFNET_EMA_DECAY', '0.99'))
    except Exception:
        use_ema = False

    if use_ema:
        try:
            ema_enc = ema_utils.clone_model_for_ema(enc)
            ema_coeff = ema_utils.clone_model_for_ema(coeff_net)
            ema_enc.to(device)
            ema_coeff.to(device)
            ema_enc.eval()
            ema_coeff.eval()
        except Exception:
            ema_enc = None
            ema_coeff = None

    for ep in range(epochs):
        total = 0.0
        n = 0
        for smiles_batch, coeff_batch in dl:
            emb = smiles_batch_to_embeddings(enc, smiles_batch, torch.device(device))
            emb = emb.to(device)
            pred = coeff_net(emb)
            coeff_batch = coeff_batch.to(device)
            loss = loss_fn(pred, coeff_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # update EMA copies
            if ema_enc is not None and ema_coeff is not None:
                try:
                    ema_utils.update_ema(ema_enc, enc, decay=ema_decay)
                    ema_utils.update_ema(ema_coeff, coeff_net, decay=ema_decay)
                except Exception:
                    pass
            total += float(loss.item()) * coeff_batch.shape[0]
            n += coeff_batch.shape[0]
        print(f"Epoch {ep+1}/{epochs} loss={total / max(1,n):.6f}")

    torch.save({'encoder_state': enc.state_dict(), 'coeff_state': coeff_net.state_dict()}, out_path)
    if ema_enc is not None and ema_coeff is not None:
        try:
            torch.save({'encoder_state': ema_enc.state_dict(), 'coeff_state': ema_coeff.state_dict()}, str(Path(out_path).with_name(Path(out_path).stem + '_ema' + Path(out_path).suffix)))
        except Exception:
            pass
    print('Saved checkpoint to', out_path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--smiles', default='smiles')
    p.add_argument('--cols', nargs=3, metavar=('D','Vmax','Km'), required=True)
    p.add_argument('--out', default='coeffnet_checkpoint.pth')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(argv)
    train(args.csv, args.smiles, args.cols, args.out, epochs=args.epochs, batch_size=args.batch, device=args.device)


if __name__ == '__main__':
    main()
