"""
Vectorized training functions for Scheme 3/5 — fixes CPU bottleneck.

The main issue: generate_helical_init uses Python for-loop over L positions
for each sample in each batch, blocking GPU.

This module provides cached + vectorized versions.
"""

import numpy as np
import torch
import math

# Cache for helical inits (computed once per unique length)
_init_cache = {}

def helical_init_np(L, bond_length=5.9):
    """Vectorized planar circular init using numpy (fast, no Python loop)."""
    if L in _init_cache:
        return _init_cache[L]
    angles = 2 * np.pi * np.arange(L, dtype=np.float32) / L
    radius = bond_length * L / (2 * np.pi) * 0.5
    coords = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(L, dtype=np.float32)
    ], axis=-1)
    coords = coords - coords.mean(axis=0)
    norm = max(np.linalg.norm(coords), 1e-6)
    coords = coords / norm
    _init_cache[L] = coords
    return coords

def build_init_batch(lengths, max_L, device):
    """Build batch of helical inits — uses cache, minimal Python loop."""
    B = len(lengths)
    coords_init = torch.zeros(B, max_L, 3, device=device)
    for b in range(B):
        L = int(lengths[b])
        ic = helical_init_np(L)
        coords_init[b, :L] = torch.tensor(ic, dtype=torch.float32, device=device)
        if L < max_L:
            coords_init[b, L:] = coords_init[b, L-1:L].expand(max_L - L, -1)
    return coords_init

def kabsch_rmsd_fast(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute RMSD after Kabsch optimal alignment."""
    p_c = (pred - pred.mean(dim=0)).detach().double()
    t_c = (target - target.mean(dim=0)).detach().double()
    H = t_c.T @ p_c
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.det(Vt.T @ U.T).sign()
        D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=pred.device, dtype=torch.float64))
        R = Vt.T @ D @ U.T
        p_aligned = (R @ p_c.T).T
        rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
    except Exception:
        rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
    result = rmsd.float().item()
    if math.isnan(result) or math.isinf(result) or result > 10000:
        return float('inf')
    return result
