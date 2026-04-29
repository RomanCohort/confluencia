from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd


class PINN(nn.Module):
    """Physics-Informed Neural Network for concentration C(x,t) with molecular embedding input.

    The network approximates C(x,t; m) where m is a molecule embedding vector.
    """

    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + [hidden] * layers + [1]
        seq = []
        for i in range(len(dims) - 1):
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                seq.append(nn.Tanh())
                if float(dropout) > 0:
                    seq.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CoeffNet(nn.Module):
    """Small MLP to predict PDE coefficients (D, Vmax, Km) from molecule embedding."""

    def __init__(self, mol_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(mol_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # returns (N,3) or (3,) for a single embedding
        out = self.net(m)
        # enforce positivity for coefficients where appropriate
        out = torch.softplus(out)
        return out


def pinn_pde_residual(model: PINN, x_t: torch.Tensor, mol_emb: torch.Tensor, D, Vmax=None, Km=None, coeff_fn: Callable = None) -> torch.Tensor:
    """Compute PDE residual for Fick's second law with Michaelis-Menten sink:

    dC/dt - D * Laplacian_x(C) + Vmax * C/(Km + C) = 0

    x_t: tensor shape (N, x_dim+1) where last col is t. mol_emb broadcasted to each row.
    mol_emb: (m_dim,) or (N, m_dim)
    Returns residual shape (N,)
    """
    # Work on a detached copy to avoid reusing a computation graph across multiple backward calls
    x_t = x_t.clone().detach().requires_grad_(True)
    # concatenate mol_emb to inputs
    if mol_emb.ndim == 1:
        mol_b = mol_emb.unsqueeze(0).repeat(x_t.shape[0], 1)
    else:
        mol_b = mol_emb
    inp = torch.cat([x_t, mol_b], dim=1)
    C = model(inp).unsqueeze(-1)  # (N,1)

    # allow coefficients to be functions of mol_emb via coeff_fn(mol_emb) -> (N,3) or (3,)
    if coeff_fn is not None:
        coeffs = coeff_fn(mol_b)
        # coeffs: (N,3)
        Dv = coeffs[:, 0:1]
        Vmaxv = coeffs[:, 1:2]
        Kmv = coeffs[:, 2:3]
    else:
        # scalar or tensors provided directly
        Dv = torch.tensor(float(D), device=x_t.device).view(1, 1).repeat(x_t.shape[0], 1)
        Vmaxv = torch.tensor(float(Vmax or 0.0), device=x_t.device).view(1, 1).repeat(x_t.shape[0], 1)
        Kmv = torch.tensor(float(Km or 1e-3), device=x_t.device).view(1, 1).repeat(x_t.shape[0], 1)

    # time derivative dC/dt
    grads = autograd.grad(C, x_t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    dC_dt = grads[:, -1].unsqueeze(-1)

    # Laplacian: sum of second derivatives w.r.t spatial dims (assume first k cols are spatial)
    spatial = x_t[:, :-1]
    lap = torch.zeros_like(C)
    for i in range(spatial.shape[1]):
        dC_dx = autograd.grad(C, x_t, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, i].unsqueeze(-1)
        d2 = autograd.grad(dC_dx, x_t, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0][:, i].unsqueeze(-1)
        lap = lap + d2

    # reaction term
    # reaction term using possibly per-sample Vmax/Km
    reaction = Vmaxv * C / (Kmv + C + 1e-8)

    residual = dC_dt - Dv * lap + reaction
    return residual.squeeze(-1)


def heat_residual(model: PINN, x_t: torch.Tensor, mol_emb: torch.Tensor, D=None, **kwargs) -> torch.Tensor:
    """Heat equation residual: dC/dt - D * Laplacian(C) = 0"""
    x_t = x_t.clone().detach().requires_grad_(True)
    if mol_emb is not None and getattr(mol_emb, 'ndim', None) == 1:
        mol_b = mol_emb.unsqueeze(0).repeat(x_t.shape[0], 1)
    else:
        mol_b = mol_emb
    inp = torch.cat([x_t, mol_b], dim=1) if mol_b is not None else x_t
    C = model(inp).unsqueeze(-1)
    grads = autograd.grad(C, x_t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    dC_dt = grads[:, -1].unsqueeze(-1)
    spatial = x_t[:, :-1]
    lap = torch.zeros_like(C)
    for i in range(spatial.shape[1]):
        dC_dx = autograd.grad(C, x_t, grad_outputs=torch.ones_like(C), create_graph=True)[0][:, i].unsqueeze(-1)
        d2 = autograd.grad(dC_dx, x_t, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0][:, i].unsqueeze(-1)
        lap = lap + d2
    Dv = D if D is not None else float(kwargs.get('D', 0.1))
    return (dC_dt - Dv * lap).squeeze(-1)


def poisson_residual(model: PINN, x: torch.Tensor, mol_emb: torch.Tensor, D=None, f_fn: Callable = None, **kwargs) -> torch.Tensor:
    """Poisson residual (steady): -D * Laplacian(u) - f(x) = 0 -> residual = -D Laplacian - f"""
    x = x.clone().detach().requires_grad_(True)
    if mol_emb is not None and getattr(mol_emb, 'ndim', None) == 1:
        mol_b = mol_emb.unsqueeze(0).repeat(x.shape[0], 1)
    else:
        mol_b = mol_emb
    inp = torch.cat([x, mol_b], dim=1) if mol_b is not None else x
    U = model(inp).unsqueeze(-1)
    lap = torch.zeros_like(U)
    for i in range(x.shape[1]):
        dU_dx = autograd.grad(U, x, grad_outputs=torch.ones_like(U), create_graph=True)[0][:, i].unsqueeze(-1)
        d2 = autograd.grad(dU_dx, x, grad_outputs=torch.ones_like(dU_dx), create_graph=True)[0][:, i].unsqueeze(-1)
        lap = lap + d2
    Dv = D if D is not None else float(kwargs.get('D', 0.1))
    if callable(f_fn):
        f_val = f_fn(x)
    else:
        f_val = torch.zeros_like(U)
    return (-Dv * lap - f_val).squeeze(-1)


def burgers_residual(model: PINN, x_t: torch.Tensor, mol_emb: torch.Tensor, nu=None, **kwargs) -> torch.Tensor:
    """1D Burgers residual: u_t + u * u_x - nu * u_xx = 0"""
    x_t = x_t.clone().detach().requires_grad_(True)
    if mol_emb is not None and getattr(mol_emb, 'ndim', None) == 1:
        mol_b = mol_emb.unsqueeze(0).repeat(x_t.shape[0], 1)
    else:
        mol_b = mol_emb
    inp = torch.cat([x_t, mol_b], dim=1) if mol_b is not None else x_t
    u = model(inp).unsqueeze(-1)
    grads = autograd.grad(u, x_t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, -1].unsqueeze(-1)
    u_x = autograd.grad(u, x_t, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0].unsqueeze(-1)
    u_xx = autograd.grad(u_x, x_t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0].unsqueeze(-1)
    nu_v = nu if nu is not None else float(kwargs.get('nu', 0.1))
    return (u_t + u * u_x - nu_v * u_xx).squeeze(-1)


def default_residual(model: PINN, pts: torch.Tensor, mol_emb: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compatibility wrapper for default diffusion+Michaelis-Menten residual."""
    D = kwargs.get('D', None)
    Vmax = kwargs.get('Vmax', None)
    Km = kwargs.get('Km', None)
    coeff_fn = kwargs.get('coeff_fn', None)
    return pinn_pde_residual(model, pts, mol_emb, D, Vmax, Km, coeff_fn=coeff_fn)


def pinn_loss(model: PINN, collocation_pts: torch.Tensor, mol_emb: torch.Tensor, D: float = None, Vmax: float = None, Km: float = None, coeff_fn: Callable = None, residual_fn: Callable = None, bc_pts: torch.Tensor = None, bc_vals: torch.Tensor = None, ic_pts: torch.Tensor = None, ic_vals: torch.Tensor = None) -> torch.Tensor:
    # PDE residual term: allow user to provide a custom residual_fn. If not provided,
    # fall back to builtin `pinn_pde_residual` which implements diffusion+Michaelis-Menten.
    if residual_fn is None:
        res = pinn_pde_residual(model, collocation_pts, mol_emb, D, Vmax, Km, coeff_fn=coeff_fn)
    else:
        # residual_fn should have signature (model, collocation_pts, mol_emb, **kwargs) -> (N,)
        res = residual_fn(model, collocation_pts, mol_emb, D=D, Vmax=Vmax, Km=Km, coeff_fn=coeff_fn)
    loss_pde = torch.mean(res ** 2)

    loss = loss_pde
    if bc_pts is not None and bc_vals is not None:
        inp = torch.cat([bc_pts, mol_emb.unsqueeze(0).repeat(bc_pts.shape[0], 1)], dim=1)
        pred = model(inp)
        loss = loss + torch.mean((pred - bc_vals) ** 2)
    if ic_pts is not None and ic_vals is not None:
        inp = torch.cat([ic_pts, mol_emb.unsqueeze(0).repeat(ic_pts.shape[0], 1)], dim=1)
        pred = model(inp)
        loss = loss + torch.mean((pred - ic_vals) ** 2)
    return loss
