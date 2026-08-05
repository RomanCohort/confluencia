"""Lightweight AF3-style physics refinement for inference.

After CoordDiffusion outputs (B,L,3) coordinates, run a short gradient-descent
relaxation driven by stereochemistry energy (bond/angle/clash/dihedral) +
BSJ closure. Optionally project bond lengths back to 5.9 Å each step
(rigid-ish) so the output has chemically exact backbone geometry.

This is NOT OpenMM — it reuses the existing stereochemistry_losses energy
terms as a differentiable potential and does cheap Adam steps. Millisecond-
scale per step on GPU. Trade-off vs full physics: no real thermodynamics,
but guarantees the 4 hard stereo constraints are minimized.

Public API:
    refine_coords(coords, lengths, n_steps=100, lr=0.5, project_bonds=True)
"""
from __future__ import annotations
import torch
from typing import Optional

# Local imports (same package)
from stereochemistry_losses import get_stereo_loss_breakdown


BOND_LENGTH = 5.9  # Å, P-P backbone (matches stereochemistry_losses default)


@torch.no_grad()
def _project_bond_lengths(coords: torch.Tensor, lengths: torch.Tensor,
                          bond_length: float = BOND_LENGTH,
                          circular: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rigid-ish bond-length projection: rescale each consecutive pair to bond_length.

    Greedy sequential projection (forward pass along the chain). Not a true
    rigid projection (that needs Lagrange/constraint solver) but converges
    fast and keeps bond error near zero after a few passes. Circular BSJ
    bond (0, L-1) is also projected — but only for circular samples
    (linear RNA 首尾不闭合, 投影会错误折叠).

    Args:
        coords: (B, L, 3) — modified in place (returns same tensor)
        lengths: (B,) valid lengths
        circular: (B,) 0/1 — None→全环化 (旧行为)
    """
    B = coords.shape[0]
    for b in range(B):
        vL = int(lengths[b].item())
        if vL < 2:
            continue
        c = coords[b, :vL]  # view, in-place edits propagate
        # Forward pass: i in 1..vL-1, pull c[i] toward/away from c[i-1]
        for i in range(1, vL):
            d = c[i] - c[i - 1]
            n = d.norm()
            if n > 1e-6:
                c[i] = c[i - 1] + d * (bond_length / n)
        # BSJ closure: only for circular samples (linear 首尾不闭合)
        if circular is None or circular[b] > 0:
            d_bsj = c[0] - c[vL - 1]
            n_bsj = d_bsj.norm()
            if n_bsj > 1e-6:
                half = d_bsj * 0.5 * (1.0 - bond_length / n_bsj)
                c[0] = c[0] - half
                c[vL - 1] = c[vL - 1] + half
    return coords


def _energy(coords: torch.Tensor, lengths: torch.Tensor,
            circular: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Differentiable stereo energy (bond/angle/clash/dihedral) — scalar.
    [v5] circular: 1=环化(BSJ惩罚), 0=线性(跳过BSJ)。None→全环化。"""
    breakdown = get_stereo_loss_breakdown(coords, lengths, circular=circular)
    return breakdown['total']


def refine_coords(coords: torch.Tensor,
                   lengths: torch.Tensor,
                   n_steps: int = 20,
                   lr: float = 0.5,
                   project_bonds: bool = True,
                   bond_length: float = BOND_LENGTH,
                   circular: Optional[torch.Tensor] = None,
                   return_history: bool = False) -> torch.Tensor:
    """AF3-style lightweight physics refinement.

    Runs n_steps of Adam on the coordinates to minimize stereochemistry energy,
    optionally projecting bond lengths back to target each step. Keeps the
    coordinates' global fold (no large moves — small lr, bounded steps).

    Args:
        coords: (B, L, 3) initial coordinates from CoordDiffusion
        lengths: (B,) valid lengths (padded positions ignored)
        n_steps: relaxation steps (default 20, matches diffusion step count)
        lr: Adam learning rate (default 0.5 Å — coarse, geometric-scale)
        project_bonds: if True, rigid-project bond lengths each step
        bond_length: target P-P distance (default 5.9 Å)
        circular: (B,) 0/1 — 1=环化(BSJ约束), 0=线性(跳过BSJ)。None→全环化。
        return_history: if True, returns (coords, [energy per step])

    Returns:
        refined coords (B, L, 3), or (coords, history) if return_history.
    """
    if coords.dim() != 3:
        raise ValueError(f"coords must be (B,L,3), got {coords.shape}")
    device = coords.device
    # Work on a clone so the input is untouched. Enable grad locally — callers
    # may invoke this from a no_grad() inference context.
    x = coords.clone().detach().to(torch.float32)
    B, L, _ = x.shape
    valid_mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(-1)
    mask_f = valid_mask.float().unsqueeze(-1)  # (B, L, 1)

    history = []
    with torch.enable_grad():
        x = x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for step in range(n_steps):
            opt.zero_grad()
            e = _energy(x, lengths, circular=circular)
            e.backward()
            if x.grad is not None:
                x.grad.mul_(mask_f)
            opt.step()

            if project_bonds:
                with torch.no_grad():
                    _project_bond_lengths(x, lengths, bond_length, circular=circular)
            x = x.detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)

            if return_history:
                history.append(e.item())


    if return_history:
        return x.detach(), history
    return x.detach()


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os, sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.insert(0, ROOT)

    torch.manual_seed(0)
    B, L = 2, 60
    # Garbage coords with bad bond lengths (random, ~1-2 Å bonds, wrong)
    coords = torch.randn(B, L, 3) * 3.0
    lengths = torch.tensor([L, L])

    # Measure bond error before
    def bond_err(c):
        errs = []
        for b in range(c.shape[0]):
            vL = int(lengths[b].item())
            cc = c[b, :vL]
            d = (cc[1:] - cc[:-1]).norm(dim=-1)
            errs.append(((d - BOND_LENGTH) ** 2).mean().item())
        return sum(errs) / len(errs)

    before = bond_err(coords)
    refined, hist = refine_coords(coords, lengths, n_steps=100, lr=0.5,
                                   return_history=True)
    after = bond_err(refined)

    print('=== physics refinement smoke test ===')
    print(f'energy: {hist[0]:.4f} -> {hist[-1]:.4f}  (should decrease)')
    print(f'bond MSE: {before:.4f} -> {after:.6f}  (should drop ~to 0 with projection)')
    print(f'energy history: {[f"{h:.3f}" for h in hist[:5]]} ... {[f"{h:.3f}" for h in hist[-3:]]}')

    assert hist[-1] < hist[0], 'energy did not decrease'
    assert after < before, 'bond error did not decrease'
    assert after < 0.1, f'bond MSE still high: {after}'
    print('\n[PASS] refinement decreases energy and enforces bond lengths.')
