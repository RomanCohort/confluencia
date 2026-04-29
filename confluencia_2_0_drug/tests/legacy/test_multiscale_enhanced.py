import os
import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")

import torch
from src.gnn import mol_to_graph, EnhancedGNN
from src.multiscale import MultiScaleModel


def test_multiscale_quick():
    smiles = "CCO"
    X, A, mol = mol_to_graph(smiles)
    in_dim = X.shape[1]
    gnn = EnhancedGNN(in_dim, hidden_dim=64, steps=2)
    msm = MultiScaleModel(gnn)
    mol_emb = msm.encode_molecule(smiles)
    msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64)
    # quick train step
    pts = torch.rand((16, 2))
    losses = msm.train_pinn(epochs=3, collocation_sampler=lambda: pts, mol_emb=mol_emb, D=0.1, Vmax=0.5, Km=0.1, lr=1e-3)
    assert len(losses) == 3


if __name__ == '__main__':
    test_multiscale_quick()
    print('ok')
