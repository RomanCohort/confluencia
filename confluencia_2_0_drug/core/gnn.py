from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)

Chem: Any
try:
    from rdkit import Chem as _Chem
    Chem = _Chem
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    Chem = None  # type: ignore
    _RDKIT_AVAILABLE = False

RdAtom = Any
RdBond = Any
RdMol = Any


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE or Chem is None:
        raise RuntimeError("RDKit 未安装或不可用，请先安装 rdkit 才能使用分子图功能。")


def atom_features(atom: RdAtom) -> np.ndarray:
    # Basic atom features: element one-hot (common elements) + degree + formal charge + aromatic
    elems = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
    z = atom.GetSymbol()
    one_hot = [1.0 if z == e else 0.0 for e in elems]
    degree = float(atom.GetDegree())
    formal_charge = float(atom.GetFormalCharge())
    implicit_valence = float(atom.GetImplicitValence() or 0)
    num_hs = float(atom.GetTotalNumHs())
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    return np.array(one_hot + [degree, formal_charge, implicit_valence, num_hs, aromatic], dtype=np.float32)


def mol_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, RdMol]:
    """Convert SMILES to node features X (n, d) and adjacency A (n, n).

    Returns (X, A, mol). Raises ValueError if RDKit can't parse.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    feats = [atom_features(a) for a in mol.GetAtoms()]
    X = np.vstack(feats) if feats else np.zeros((0, len(feats[0]) if feats else 0), dtype=np.float32)
    A = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        A[i, j] = 1.0
        A[j, i] = 1.0
    return X, A, mol


class SimpleGNN(nn.Module):
    """A minimal message-passing GNN implemented with plain PyTorch.

    - Message: sum of neighbor features (A @ X) followed by a linear layer.
    - Update: MLP on concat(self, message) with residual connection.
    """

    def __init__(self, in_feats: int, hidden_dim: int = 128, steps: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.steps = steps

        # message projection now accepts hidden_dim (since we project input to hidden first)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

        # If input dim != hidden_dim, project initially
        if in_feats != hidden_dim:
            self.input_proj = nn.Linear(in_feats, hidden_dim)
        else:
            self.input_proj = nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (n, in_feats), adj: (n, n) (0/1 float)

        Returns node embeddings (n, hidden_dim).
        """
        h = self.input_proj(x)
        for _ in range(self.steps):
            m = torch.matmul(adj, h)  # aggregate neighbor hidden states
            m = self.msg_lin(m)
            concat = torch.cat([h, m], dim=-1)
            delta = self.update_mlp(concat)
            h = self.norm(h + self.dropout(delta))
        return h


def readout_mean(node_emb: torch.Tensor) -> torch.Tensor:
    """Mean-pool node embeddings to molecule embedding (1D tensor)."""
    if node_emb.numel() == 0:
        return torch.zeros((node_emb.shape[-1],), dtype=node_emb.dtype, device=node_emb.device)
    return node_emb.mean(dim=0)


class AttentionReadout(nn.Module):
    """Attention-based global readout: learnable scoring MLP produces attention weights over nodes.

    Usage: instantiate with node embedding dim then call on `(n, dim)` tensor to return `(dim,)` mol embedding.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        # node_emb: (n, dim)
        if node_emb.numel() == 0:
            return torch.zeros((node_emb.shape[-1],), dtype=node_emb.dtype, device=node_emb.device)
        scores = self.score(node_emb).squeeze(-1)  # (n,)
        alpha = torch.softmax(scores, dim=0).unsqueeze(-1)  # (n,1)
        return (alpha * node_emb).sum(dim=0)


def pairwise_distances(mol: RdMol) -> Optional[np.ndarray]:
    """Compute pairwise Euclidean distances using 3D coordinates if available.

    Returns None if no conformer coordinates present.
    """
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    if conf is None:
        return None
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i, 0] = float(p.x)
        coords[i, 1] = float(p.y)
        coords[i, 2] = float(p.z)
    d = np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=-1)
    return d


def compute_angle(a_pos, b_pos, c_pos):
    # angle at b between a-b-c
    ba = a_pos - b_pos
    bc = c_pos - b_pos
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosang = np.dot(ba, bc) / norm
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def bond_features(bond: RdBond) -> np.ndarray:
    """Simple bond feature vector: bond type one-hot + is_conjugated + is_in_ring."""
    types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bt = bond.GetBondType()
    one_hot = [1.0 if bt == t else 0.0 for t in types]
    conj = 1.0 if bond.GetIsConjugated() else 0.0
    ring = 1.0 if bond.IsInRing() else 0.0
    return np.array(one_hot + [conj, ring], dtype=np.float32)


def mol_edge_matrix(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RdMol]:
    """Return node features X and edge feature tensor E (n,n,ef) plus mol. Non-bond pairs are zeros."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    feats = [atom_features(a) for a in mol.GetAtoms()]
    X = np.vstack(feats) if feats else np.zeros((0, len(feats[0]) if feats else 0), dtype=np.float32)
    ef = len(bond_features(mol.GetBonds()[0])) if mol.GetBonds() else 2
    E = np.zeros((n, n, ef), dtype=np.float32)
    A = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        E[i, j, :] = bf
        E[j, i, :] = bf
        A[i, j] = 1.0
        A[j, i] = 1.0
    return X, E, A, mol


def mol_conformer_coords(mol: RdMol) -> Optional[np.ndarray]:
    """Return conformer coordinates as (n,3) float32 array or None if not available."""
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i, 0] = float(p.x)
        coords[i, 1] = float(p.y)
        coords[i, 2] = float(p.z)
    return coords


def physical_message(x_u: torch.Tensor, x_v: torch.Tensor, edge_feat: Optional[torch.Tensor], r_uv: Optional[float], angles: Optional[float]) -> torch.Tensor:
    """A small physics-guided message: combine feature similarity with distance/angle-based potentials.

    This function returns a vector (same dim as x_u) representing message from v to u.
    The implementation is intentionally simple and differentiable.
    """
    # feature-based part (dot product attention)
    feat_score = torch.sum(x_u * x_v, dim=-1, keepdim=True)
    # distance attenuation (if r_uv provided)
    if r_uv is None:
        r_term = 1.0
    else:
        r = float(r_uv) + 1e-6
        r_term = math.exp(-r / 3.0)  # simple exponential decay
    # angle modulation
    ang_term = 1.0
    if angles is not None:
        ang = float(angles)
        ang_term = max(0.0, math.cos(ang))

    scale = float(r_term * ang_term)
    # produce a message vector by scaling neighbor features by attention
    msg = x_v * feat_score * scale
    return msg


class GATLayer(nn.Module):
    """A lightweight Graph Attention layer (single-head) that can incorporate physical terms.

    Expects dense adjacency and optional distance/angle matrices to modulate attention.
    """

    def __init__(self, in_dim: int, out_dim: int, use_physics: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1)
        self.leaky = nn.LeakyReLU(0.2)
        self.use_physics = use_physics

    def forward(self, x: torch.Tensor, adj: torch.Tensor, dist_mat: Optional[torch.Tensor] = None, angle_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.W(x)  # (n, out_dim)
        n = h.shape[0]
        # prepare attention scores
        h_i = h.unsqueeze(1).repeat(1, n, 1)
        h_j = h.unsqueeze(0).repeat(n, 1, 1)
        cat = torch.cat([h_i, h_j], dim=-1)  # (n, n, 2*out_dim)
        e = self.leaky(self.attn(cat).squeeze(-1))  # (n, n)
        # mask by adjacency
        e = e.masked_fill(adj == 0, float('-inf'))
        alpha = torch.softmax(e, dim=1)  # normalize over neighbors

        # physics modulation
        if self.use_physics and dist_mat is not None:
            # attenuate by exp(-r) on alpha
            alpha = alpha * torch.exp(-dist_mat)
            if angle_mat is not None:
                alpha = alpha * (torch.clamp(torch.cos(angle_mat), min=0.0) + 1e-6)
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-9)

        out = torch.matmul(alpha, h)
        return out


class MultiHeadGAT(nn.Module):
    """Multi-head attention wrapper around GATLayer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, use_physics: bool = True, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads
        self.heads = nn.ModuleList([GATLayer(in_dim, self.head_dim, use_physics=use_physics) for _ in range(num_heads)])
        self.out_lin = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, dist_mat: Optional[torch.Tensor] = None, angle_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = [h(x, adj, dist_mat, angle_mat) for h in self.heads]
        cat = torch.cat(outs, dim=-1)
        cat = self.dropout(cat)
        return self.out_lin(cat)


class EnhancedGNN(nn.Module):
    """Enhanced GNN combining initial projection, several message-passing rounds, optional GAT refinement and residual connections."""

    def __init__(self, in_feats: int, hidden_dim: int = 128, steps: int = 3, gat_heads: int = 4, use_physics: bool = True, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_feats, hidden_dim) if in_feats != hidden_dim else nn.Identity()
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.steps = steps
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, num_heads=gat_heads, use_physics=use_physics, dropout=float(dropout))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor, dist_mat: Optional[torch.Tensor] = None, angle_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        for _ in range(self.steps):
            m = torch.matmul(adj, h)
            m = self.msg_lin(m)
            concat = torch.cat([h, m], dim=-1)
            delta = self.update(concat)
            h = self.norm(h + self.dropout(delta))
        # optional GAT refinement
        try:
            g = self.gat(h, adj, dist_mat, angle_mat)
            h = self.norm(h + g)
        except Exception as exc:
            logger.debug(f"GAT refinement skipped: {exc}")


class PhysicsMessageGNN(nn.Module):
    """GNN that incorporates simple physics-guided potential into message computation.

    The message from v->u is modulated by a potential term derived from pairwise distance
    (and optional edge features). By default uses an exponential decay exp(-r/scale) and
    feature similarity; a custom potential_fn(node_u, node_v, edge_feat, r) may be supplied.
    """

    def __init__(self, in_feats: int, hidden_dim: int = 128, steps: int = 3, scale: float = 3.0, potential_fn=None, potential_type: str = "auto", lj_epsilon: float = 0.1, lj_sigma: float = 3.5, dielectric: float = 80.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_feats, hidden_dim) if in_feats != hidden_dim else nn.Identity()
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.steps = steps
        self.scale = float(scale)
        self.potential_fn = potential_fn
        self.potential_type = potential_type
        # LJ params
        self.lj_epsilon = float(lj_epsilon)
        self.lj_sigma = float(lj_sigma)
        # electrostatic dielectric
        self.dielectric = float(dielectric)

    def default_potential(self, h_u: torch.Tensor, h_v: torch.Tensor, edge_feat: Optional[torch.Tensor], r: Optional[torch.Tensor]):
        # h_u/h_v: (..., dim), r: (...,) or scalar
        # feature similarity term (cosine-like) and distance attenuation
        feat_score = torch.sum(h_u * h_v, dim=-1)
        # distance attenuation: exp(-r/scale)
        if r is None:
            r_term = 1.0
        else:
            # ensure tensor
            r_term = torch.exp(-r / self.scale)
        # combine and clip
        score = feat_score * r_term
        # non-negative scaling
        return torch.relu(score).unsqueeze(-1)

    def lennard_jones(self, r: torch.Tensor) -> torch.Tensor:
        # LJ: V(r) = 4 * eps * [ (sigma/r)^12 - (sigma/r)^6 ]
        # return scalar potential (positive/negative), r shape (...,)
        eps = self.lj_epsilon
        sigma = self.lj_sigma
        # avoid division by zero
        r_safe = torch.clamp(r, min=1e-6)
        sr6 = (sigma / r_safe) ** 6
        v = 4.0 * eps * (sr6 * sr6 - sr6)
        return v

    def electrostatic(self, q1: torch.Tensor, q2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # Coulomb: k_e * q1*q2 / (epsilon * r)
        # Here we treat q1,q2 as scalar partial charges (if available), otherwise assume 1.
        eps = self.dielectric
        r_safe = torch.clamp(r, min=1e-6)
        # k_e in vacuum ~ 8.988e9, we can absorb constants into a learned scale; use 1.0 for simplicity
        k = 1.0
        return k * (q1 * q2) / (eps * r_safe)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, dist_mat: Optional[torch.Tensor] = None, edge_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (n, in_feats) or (batch, n, in_feats)
        adj: (n, n)
        dist_mat: (n, n) distances in same index order as adj, optional
        Returns node embeddings (n, hidden_dim)
        """
        h = self.input_proj(x)
        n = h.shape[0]
        for _ in range(self.steps):
            # linear transform neighbors
            h_lin = self.msg_lin(h)  # (n, d)

            # compute pairwise feature similarity matrix: (n,n)
            h_i = h.unsqueeze(1).repeat(1, n, 1)
            h_j = h.unsqueeze(0).repeat(n, 1, 1)
            # compute potential matrix
            if self.potential_fn is None:
                if dist_mat is None:
                    pot = torch.sum(h_i * h_j, dim=-1)
                    pot = torch.relu(pot).unsqueeze(-1)
                else:
                    pot_scalar = torch.sum(h_i * h_j, dim=-1)
                    # choose potential combination based on potential_type
                    if self.potential_type == 'lennard' or self.potential_type == 'lj':
                        lj = self.lennard_jones(dist_mat)
                        pot = torch.relu(pot_scalar) * torch.exp(-dist_mat / self.scale) + lj
                        pot = pot.unsqueeze(-1)
                    elif self.potential_type == 'electrostatic':
                        # no per-atom charges available here; fallback to distance-only attenuation
                        pot = torch.exp(-dist_mat / self.scale) * torch.relu(pot_scalar)
                        pot = pot.unsqueeze(-1)
                    else:
                        # auto: combine feature similarity attenuation + LJ soft term
                        lj = self.lennard_jones(dist_mat)
                        pot = torch.exp(-dist_mat / self.scale) * torch.relu(pot_scalar) + 0.1 * lj
                        pot = pot.unsqueeze(-1)
            else:
                # call user potential_fn elementwise (vectorized call expected)
                pot = self.potential_fn(h_i, h_j, edge_feat, dist_mat)

            # mask by adjacency
            adj_mask = adj.unsqueeze(-1)  # (n,n,1)
            if pot.dim() == 2:
                pot = pot.unsqueeze(-1)
            weighted = (pot * h_j) * adj_mask  # (n,n,d)
            # sum over neighbors (dim=1 -> neighbors of i)
            m = weighted.sum(dim=1)  # (n,d)

            concat = torch.cat([h, m], dim=-1)
            delta = self.update(concat)
            h = self.norm(h + self.dropout(delta))

        return h


class EGNNLayer(nn.Module):
    """A single EGNN layer (Satorras et al. style) implementing E(3)-equivariant updates.

    This layer updates node features `h` and coordinates `x` in an equivariant manner using
    messages that depend on feature pairs and inter-node distances.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.edge_mlp = nn.Sequential(nn.Linear(2 * feat_dim + 1, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
        self.coord_mlp = nn.Sequential(nn.Linear(feat_dim, 1), nn.Tanh())
        self.node_mlp = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU())

    def forward(self, h: torch.Tensor, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (n, d), x: (n,3), adj: (n,n)

        Returns updated (h, x)
        """
        n = h.shape[0]
        h_i = h.unsqueeze(1).repeat(1, n, 1)  # (n,n,d)
        h_j = h.unsqueeze(0).repeat(n, 1, 1)

        rel = x.unsqueeze(1) - x.unsqueeze(0)  # (n,n,3)
        dist = torch.norm(rel + 1e-8, dim=-1, keepdim=True)  # (n,n,1)

        edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # (n,n,2d+1)
        e_ij = self.edge_mlp(edge_input)  # (n,n,d)

        mask = adj.unsqueeze(-1)
        e_ij = e_ij * mask

        m_i = e_ij.sum(dim=1)  # (n,d)
        h_new = h + self.node_mlp(m_i)

        # coordinate update: scalar per edge to scale vector (x_i - x_j)
        coord_scalars = self.coord_mlp(e_ij).squeeze(-1)  # (n,n)
        coord_scalars = coord_scalars * (adj)
        dx = (coord_scalars.unsqueeze(-1) * rel).sum(dim=1)  # (n,3)
        x_new = x + dx

        return h_new, x_new


class EGNN(nn.Module):
    """Stacked EGNN layers to produce SE(3)-equivariant node embeddings.

    Example usage:
        g = EGNN(in_feats, hidden_dim=64, n_layers=4)
        h, x = g(x_feats, coords, adj)
    """

    def __init__(self, in_feats: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(in_feats, hidden_dim) if in_feats != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.hidden_dim = hidden_dim

    def forward(self, x_feats: torch.Tensor, adj: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """x_feats: (n, in_feats), adj: (n,n), coords: (n,3)

        Returns node embeddings (n, hidden_dim)."""
        h = self.in_proj(x_feats)
        x = coords
        for layer in self.layers:
            h, x = layer(h, x, adj)
        return h

