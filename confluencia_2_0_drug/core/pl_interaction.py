"""Protein-Ligand interaction model using physics-guided message-passing.

This module provides a simple pipeline:
- featurize ligand via RDKit SMILES -> atom features + adjacency
- featurize protein pocket via coordinates + atom types (caller provides arrays)
- encode ligand & protein with PhysicsMessageGNN
- perform cross-attention modulated by distance-based potential to compute complex embedding
- predict interaction score with an MLP

This is a lightweight template for research/prototyping; for production use replace
protein featurizer with a robust PDB parser and proper residue-level features.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem

from .gnn import mol_to_graph, PhysicsMessageGNN


def protein_from_coords(atom_types: List[str], coords: np.ndarray, cutoff: float = 6.0):
    """Build node features X and adjacency A for protein pocket.

    - `atom_types`: list of element symbols (len N)
    - `coords`: (N,3) numpy array
    - adjacency: connect pairs within `cutoff` Å
    Returns (X, A)
    """
    assert coords.shape[0] == len(atom_types)
    elems = ["C", "N", "O", "S", "P", "H", "Fe", "Zn", "Mg", "Ca"]
    feats = []
    for i, t in enumerate(atom_types):
        one_hot = [1.0 if t == e else 0.0 for e in elems]
        feats.append(one_hot + list(coords[i]))
    X = np.array(feats, dtype=np.float32)
    n = X.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    dmat = np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=-1)
    A[dmat <= cutoff] = 1.0
    return X, A, dmat


class ProteinLigandInteractionModel(nn.Module):
    """Encodes ligand and protein pocket and predicts interaction score."""

    def __init__(
        self,
        ligand_in_dim: int,
        protein_in_dim: int,
        hidden: int = 128,
        dropout: float = 0.1,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_bidirectional: bool = True,
    ):
        super().__init__()
        self.lig_enc = PhysicsMessageGNN(ligand_in_dim, hidden_dim=hidden, steps=3)
        self.prot_enc = PhysicsMessageGNN(protein_in_dim, hidden_dim=hidden, steps=3)

        self.use_lstm = bool(use_lstm)
        self.lstm_bidirectional = bool(lstm_bidirectional)
        if self.use_lstm:
            lstm_out = int(lstm_hidden) * (2 if self.lstm_bidirectional else 1)
            self.lig_lstm = nn.LSTM(
                input_size=int(hidden),
                hidden_size=int(lstm_hidden),
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            self.prot_lstm = nn.LSTM(
                input_size=int(hidden),
                hidden_size=int(lstm_hidden),
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            self.lig_proj = nn.Linear(lstm_out, int(hidden))
            self.prot_proj = nn.Linear(lstm_out, int(hidden))
        else:
            self.lig_lstm = None
            self.prot_lstm = None
            self.lig_proj = None
            self.prot_proj = None

        # cross-attention readout (learned temperature)
        self.cross_proj = nn.Linear(hidden, hidden)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    def forward(self, lig_X: torch.Tensor, lig_A: torch.Tensor, lig_dist: Optional[torch.Tensor], prot_X: torch.Tensor, prot_A: torch.Tensor, prot_dist: Optional[torch.Tensor], prot_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        # node embeddings
        h_lig = self.lig_enc(lig_X, lig_A, lig_dist)
        h_prot = self.prot_enc(prot_X, prot_A, prot_dist)

        if self.lig_lstm is not None and self.prot_lstm is not None:
            h_lig, _ = self.lig_lstm(h_lig.unsqueeze(0))
            h_prot, _ = self.prot_lstm(h_prot.unsqueeze(0))
            h_lig = self.lig_proj(h_lig.squeeze(0))
            h_prot = self.prot_proj(h_prot.squeeze(0))

        # cross attention via distance between ligand atom coords (if available) and protein coords
        # if prot_coords provided and lig_coords encoded into last 3 entries of lig_X, use them
        if prot_coords is not None and lig_X.shape[1] >= 3:
            lig_coords = lig_X[:, -3:]
            # compute pairwise distances (L_atoms, P_atoms)
            dmat = torch.cdist(lig_coords, prot_coords)
            # attention weights
            attn = torch.softmax(-dmat, dim=1)  # ligand->protein normalized over protein
            # aggregate protein embedding to ligand frame
            weighted = torch.matmul(attn, h_prot)  # (L, hidden)
            # fuse
            fused = torch.cat([h_lig.mean(dim=0), weighted.mean(dim=0)], dim=-1)
        else:
            # fallback: mean-pool both
            fused = torch.cat([h_lig.mean(dim=0), h_prot.mean(dim=0)], dim=-1)

        out = self.pred(fused)
        return out.squeeze(-1)


def example_predict(smiles: str, prot_atom_types: List[str], prot_coords: np.ndarray) -> float:
    """Quick infer example: builds minimal features and returns a score (no trained weights).

    This is a demo using random initialization; for real use, train the model on labeled complexes.
    """
    # ligand graph
    X_l, A_l, mol = mol_to_graph(smiles)
    # add coords to ligand X if conformer available
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    if conf is not None:
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            coords[i] = [p.x, p.y, p.z]
        # append coords to X_l
        X_l = np.hstack([X_l, coords])

    # protein
    X_p, A_p, dmat_p = protein_from_coords(prot_atom_types, prot_coords)

    # build model
    lig_in = X_l.shape[1]
    prot_in = X_p.shape[1]
    model = ProteinLigandInteractionModel(lig_in, prot_in, hidden=64)

    # to tensors
    lig_X_t = torch.from_numpy(X_l).float()
    lig_A_t = torch.from_numpy(A_l).float()
    prot_X_t = torch.from_numpy(X_p).float()
    prot_A_t = torch.from_numpy(A_p).float()
    prot_coords_t = torch.from_numpy(prot_coords).float()

    with torch.no_grad():
        score = model(lig_X_t, lig_A_t, None, prot_X_t, prot_A_t, None, prot_coords=prot_coords_t)
    return float(score.item())
