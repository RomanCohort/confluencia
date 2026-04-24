from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from .gnn import SimpleGNN, mol_to_graph, readout_mean


def compute_embeddings_from_smiles(smiles: str, gnn: SimpleGNN, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (node_embeddings, mol_embedding) as torch tensors on `device`.

    Raises ValueError if SMILES invalid.
    """
    X_np, A_np, _ = mol_to_graph(smiles)
    if X_np.shape[0] == 0:
        raise ValueError("Empty molecule")
    x = torch.from_numpy(X_np).to(device)
    adj = torch.from_numpy(A_np).to(device)
    gnn = gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        node_emb = gnn(x, adj)
        mol_emb = readout_mean(node_emb)
    return node_emb, mol_emb


def sensitivity_masking(
    smiles: str,
    gnn: SimpleGNN,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cpu",
    mask_mode: str = "zero",
) -> Dict[int, float]:
    """Compute per-atom sensitivity by masking each atom and measuring prediction change.

    model_fn: function taking molecule embedding (1D torch tensor) and returning scalar tensor.
    mask_mode: 'zero' zeroes node features for the atom; 'drop' removes the atom (not implemented here).

    Returns dict atom_idx -> abs(delta prediction).
    """
    node_emb, mol_emb = compute_embeddings_from_smiles(smiles, gnn, device=device)
    baseline = float(model_fn(mol_emb).cpu().item())
    n = node_emb.shape[0]
    scores: Dict[int, float] = {}

    # To mask, we operate on original input features via re-computing embeddings
    # Recreate graph inputs
    X_np, A_np, _ = mol_to_graph(smiles)
    for i in range(n):
        X_mask = X_np.copy()
        if mask_mode == "zero":
            X_mask[i, :] = 0.0
        else:
            X_mask[i, :] = 0.0
        x = torch.from_numpy(X_mask).to(device)
        adj = torch.from_numpy(A_np).to(device)
        with torch.no_grad():
            node_emb_mask = gnn(x, adj)
            mol_emb_mask = readout_mean(node_emb_mask)
            pred = float(model_fn(mol_emb_mask).cpu().item())
        scores[i] = abs(pred - baseline)

    # normalize to sum=1 for convenience (if total > 0)
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] = scores[k] / total
    return scores


def example_model_fn_factory(embedding_dim: int) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]:
    """Return a simple linear model function and the module (so user can train/inspect).

    The returned model_fn accepts a 1D tensor (embedding) and returns a scalar tensor.
    """
    module = torch.nn.Sequential(torch.nn.Linear(embedding_dim, 1))

    def model_fn(emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        return module(emb).squeeze(1)

    return model_fn, module


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", help="SMILES string")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    # Build a GNN dynamically by inferring input dim from molecule
    from .gnn import mol_to_graph, SimpleGNN

    X, A, _ = mol_to_graph(args.smiles)
    in_dim = X.shape[1]
    gnn = SimpleGNN(in_dim, hidden_dim=64, steps=args.steps)
    model_fn, model_module = example_model_fn_factory(64)

    # Compute sensitivities
    scores = sensitivity_masking(args.smiles, gnn, model_fn)
    print("Per-atom normalized sensitivities:")
    for idx, s in scores.items():
        print(f"atom {idx}: {s:.4f}")
