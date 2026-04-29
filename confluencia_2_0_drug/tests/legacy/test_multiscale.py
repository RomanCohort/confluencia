import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")

import torch

from src.gnn import mol_to_graph, SimpleGNN
from src.multiscale import MultiScaleModel


def test_encode_and_build_pinn():
    smiles = "CCO"
    X, A, mol = mol_to_graph(smiles)
    in_dim = X.shape[1]
    gnn = SimpleGNN(in_dim, hidden_dim=32, steps=2)
    msm = MultiScaleModel(gnn)
    emb = msm.encode_molecule(smiles)
    assert emb.numel() > 0
    msm.build_pinn(spatial_dim=1, mol_emb_dim=emb.shape[0], hidden=32)
    assert msm.pinn is not None

if __name__ == '__main__':
    test_encode_and_build_pinn()
    print('test passed')
