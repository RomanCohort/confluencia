"""Simple example: build a PINN for 1D Poisson and register physics via MultiScaleModel.

This script demonstrates using the MultiScaleModel.register_physics API and the builtin
residuals in src.pinn.
"""
from pathlib import Path
import torch
from src.gnn import SimpleGNN
from src.multiscale import MultiScaleModel
from src.pinn import poisson_residual


def collocation_sampler(n=256):
    # x only (steady problem): here we assume spatial dim=1 and no time col
    x = torch.rand((n, 1)) * 2 - 1  # domain [-1,1]
    return x


if __name__ == "__main__":
    # tiny demo GNN (random weights) to satisfy MultiScaleModel constructor
    gnn = SimpleGNN(10, hidden_dim=16, steps=2)
    msm = MultiScaleModel(gnn)
    msm.build_pinn(spatial_dim=1, mol_emb_dim=16, hidden=64)

    # register poisson residual
    msm.register_physics(residual_fn=poisson_residual)

    # dummy mol embedding
    mol_emb = torch.randn(16)

    # short training
    optim = torch.optim.AdamW(msm.pinn.parameters(), lr=1e-3)
    for epoch in range(20):
        pts = collocation_sampler(128)
        loss = msm.pinn_step(optim, pts, mol_emb, D=0.1, Vmax=0.0, Km=0.0)
        if epoch % 5 == 0:
            print(epoch, loss)

    print('Done')
