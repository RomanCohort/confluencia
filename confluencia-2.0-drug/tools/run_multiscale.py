from __future__ import annotations

import argparse
import torch

from src.gnn import SimpleGNN, mol_to_graph
from src.multiscale import MultiScaleModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles')
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()

    X, A, mol = mol_to_graph(args.smiles)
    in_dim = X.shape[1]
    gnn = SimpleGNN(in_dim, hidden_dim=64, steps=args.steps)
    msm = MultiScaleModel(gnn)
    mol_emb = msm.encode_molecule(args.smiles)
    print('Molecule embedding dim:', mol_emb.shape)

    # build PINN for 1D spatial + time
    msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64)

    # toy training: random collocation pts
    collocation = torch.rand((128, 2))  # x,t
    D = 0.1
    Vmax = 0.5
    Km = 0.1
    optimizer = torch.optim.Adam(msm.pinn.parameters(), lr=1e-3)
    for epoch in range(30):
        loss = msm.pinn_step(optimizer, collocation, mol_emb, D, Vmax, Km)
        if epoch % 5 == 0:
            print('epoch', epoch, 'loss', loss)


if __name__ == '__main__':
    main()
