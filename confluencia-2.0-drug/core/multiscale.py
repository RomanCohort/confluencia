from __future__ import annotations

from typing import Tuple, Callable, Optional

import torch

from .gnn import mol_to_graph, readout_mean, SimpleGNN, GATLayer, AttentionReadout
from .pinn import PINN, pinn_loss


class MultiScaleModel:
    """Combines atom-level GNNs -> mid-level GAT -> top-level PINN.

    Usage: call `encode_molecule(smiles)` to get molecular embedding, then pass it to PINN.
    """

    def __init__(self, gnn: SimpleGNN, gat_hidden: int = 64, use_physics: bool = True, readout: str = "mean"):
        self.gnn = gnn
        self.gat = GATLayer(gnn.hidden_dim, gat_hidden, use_physics=use_physics)
        self.pinn: PINN = None  # to be set after knowing input dim
        self.coeff_net = None
        # readout selection: 'mean' or 'attention'
        self.readout_type = readout
        self.attn_readout: Optional[AttentionReadout] = None
        if self.readout_type == "attention":
            # instantiate attention readout with gnn hidden dim
            try:
                self.attn_readout = AttentionReadout(self.gnn.hidden_dim)
            except Exception:
                self.attn_readout = None
            # containers for user-provided physics: residual_fn(model, pts, mol_emb, **kwargs) and coeff_fn(mol_emb)
            self.physics_residual_fn: Optional[Callable] = None
            self.physics_coeff_fn: Optional[Callable] = None

    def encode_molecule(self, smiles: str, device: str = "cpu") -> torch.Tensor:
        X_np, A_np, mol = mol_to_graph(smiles)
        x = torch.from_numpy(X_np).float().to(device)
        adj = torch.from_numpy(A_np).float().to(device)
        # try to obtain coordinates for equivariant GNNs
        coords = None
        try:
            from .gnn import mol_conformer_coords

            coords_np = mol_conformer_coords(mol)
            if coords_np is not None:
                coords = torch.from_numpy(coords_np).float().to(device)
        except Exception:
            coords = None

        # call GNN; if it expects coords, pass them
        try:
            node_emb = self.gnn(x, adj, coords) if coords is not None else self.gnn(x, adj)
        except TypeError:
            # fallback: call without coords
            node_emb = self.gnn(x, adj)
        # optionally pass through GAT (requires distance matrix if 3D available)
        dist_np = None
        try:
            d = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            if d is not None:
                from .gnn import pairwise_distances

                dm = pairwise_distances(mol)
                if dm is not None:
                    dist_np = torch.from_numpy(dm).float().to(device)
        except Exception:
            dist_np = None

        node_emb2 = self.gat(node_emb, adj, dist_np, None)
        if self.readout_type == "attention" and self.attn_readout is not None:
            mol_emb = self.attn_readout(node_emb2)
        else:
            mol_emb = readout_mean(node_emb2)
        return mol_emb

    def encode_with_mask(self, smiles: str, mask_atoms: list, device: str = "cpu") -> torch.Tensor:
        """Encode molecule but zero-out features for atoms in mask_atoms before GNN.

        mask_atoms: list of atom indices to zero.
        """
        X_np, A_np, mol = mol_to_graph(smiles)
        if X_np.shape[0] == 0:
            raise ValueError("Empty molecule")
        X_mask = X_np.copy()
        for i in mask_atoms:
            if 0 <= i < X_mask.shape[0]:
                X_mask[i, :] = 0.0
        x = torch.from_numpy(X_mask).float().to(device)
        adj = torch.from_numpy(A_np).float().to(device)
        # try to obtain coordinates for equivariant GNNs
        coords = None
        try:
            from .gnn import mol_conformer_coords

            coords_np = mol_conformer_coords(mol)
            if coords_np is not None:
                # if masking atoms, zero their coordinates as well
                coords_np[mask_atoms, :] = 0.0
                coords = torch.from_numpy(coords_np).float().to(device)
        except Exception:
            coords = None

        try:
            node_emb = self.gnn(x, adj, coords) if coords is not None else self.gnn(x, adj)
        except TypeError:
            node_emb = self.gnn(x, adj)

        dist_np = None
        try:
            d = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            if d is not None:
                from .gnn import pairwise_distances

                dm = pairwise_distances(mol)
                if dm is not None:
                    dist_np = torch.from_numpy(dm).float().to(device)
        except Exception:
            dist_np = None

        node_emb2 = self.gat(node_emb, adj, dist_np, None)
        if self.readout_type == "attention" and self.attn_readout is not None:
            mol_emb = self.attn_readout(node_emb2)
        else:
            mol_emb = readout_mean(node_emb2)
        return mol_emb

    def build_pinn(self, spatial_dim: int, mol_emb_dim: int, hidden: int = 128, dropout: float = 0.0):
        # input: spatial coords + t + mol_emb
        input_dim = spatial_dim + 1 + mol_emb_dim
        self.pinn = PINN(input_dim, hidden=hidden, dropout=float(dropout))

    def build_coeff_net(self, mol_emb_dim: int, hidden: int = 64):
        try:
            from .pinn import CoeffNet

            self.coeff_net = CoeffNet(mol_emb_dim, hidden=hidden)
        except Exception:
            self.coeff_net = None

    def register_physics(self, residual_fn: Callable = None, coeff_fn: Callable = None) -> None:
        """Register user-provided physics functions.

        - `residual_fn(model, collocation_pts, mol_emb, **kwargs)` should return residual tensor shape (N,)
        - `coeff_fn(mol_emb)` should return per-sample coefficients (N,3) or (3,)
        """
        self.physics_residual_fn = residual_fn
        self.physics_coeff_fn = coeff_fn

    def save(self, path: str) -> None:
        """Save model state (GNN, GAT, PINN, coeff_net) to a checkpoint."""
        state = {}
        state['gnn'] = self.gnn.state_dict()
        state['gat'] = self.gat.state_dict()
        state['pinn'] = self.pinn.state_dict() if self.pinn is not None else None
        state['coeff'] = self.coeff_net.state_dict() if getattr(self, 'coeff_net', None) is not None else None
        # save attention readout if present
        if getattr(self, 'attn_readout', None) is not None:
            try:
                state['attn'] = self.attn_readout.state_dict()
            except Exception:
                state['attn'] = None
        import torch

        torch.save(state, path)

    def load(self, path: str, map_location='cpu') -> None:
        import torch

        state = torch.load(path, map_location=map_location)
        self.gnn.load_state_dict(state['gnn'])
        self.gat.load_state_dict(state['gat'])
        if state.get('pinn') is not None:
            if self.pinn is None:
                raise ValueError('PINN not constructed; call build_pinn before load()')
            self.pinn.load_state_dict(state['pinn'])
        if state.get('coeff') is not None and getattr(self, 'coeff_net', None) is not None:
            self.coeff_net.load_state_dict(state['coeff'])
        # load attention readout if available
        if state.get('attn') is not None and getattr(self, 'attn_readout', None) is not None:
            try:
                self.attn_readout.load_state_dict(state.get('attn'))
            except Exception:
                pass

    def train_pinn(self, epochs: int, collocation_sampler: Callable, mol_emb: torch.Tensor, D: float = 0.1, Vmax: float = 0.5, Km: float = 0.1, coeff_fn: Callable = None, lr: float = 1e-3, checkpoint: str = None):
        import torch
        from torch.optim import Adam

        assert self.pinn is not None, 'PINN not constructed. Call build_pinn first.'
        opt = Adam(self.pinn.parameters(), lr=lr)
        losses = []
        # use registered coeff_fn if provided, else use passed coeff_fn
        use_coeff = coeff_fn or getattr(self, 'physics_coeff_fn', None)
        for ep in range(int(epochs)):
            pts = collocation_sampler()
            # detach molecular embedding to avoid computing gradients through GNN during PINN training
            mol_for_pinn = mol_emb.detach() if isinstance(mol_emb, torch.Tensor) and mol_emb.requires_grad else mol_emb
            loss = pinn_loss(self.pinn, pts, mol_for_pinn, D, Vmax, Km, coeff_fn=use_coeff, residual_fn=getattr(self, 'physics_residual_fn', None))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if checkpoint and (ep % max(1, int(epochs // 5)) == 0):
                self.save(checkpoint)
        return losses

    def pinn_step(self, optimizer, collocation_pts: torch.Tensor, mol_emb: torch.Tensor, D: float, Vmax: float, Km: float, **kwargs):
        assert self.pinn is not None, "PINN not constructed. Call build_pinn first."
        optimizer.zero_grad()
        # prefer registered physics coeff_fn if present
        if 'coeff_fn' not in kwargs or kwargs.get('coeff_fn') is None:
            kwargs['coeff_fn'] = getattr(self, 'physics_coeff_fn', None)
        if 'residual_fn' not in kwargs or kwargs.get('residual_fn') is None:
            kwargs['residual_fn'] = getattr(self, 'physics_residual_fn', None)
        loss = pinn_loss(self.pinn, collocation_pts, mol_emb, D, Vmax, Km, **kwargs)
        loss.backward()
        optimizer.step()
        return float(loss.item())
