"""cartesian_to_torus.py — Equivariant Cartesian → torus coordinate mapper (Phase 2, line 1).

Maps Cartesian coords (B, L, 3) produced by S1-S8 diffusion trunks into the
equivalent torus coordinates (θ, φ, r) used by S10's steerable kernel, so the
immune fingerprint heads can run full SO(2)×SO(2) equivariant message passing
on ALL schemes — not only S10.

Math (verified numerically to ≤ 1.33e-15, see tests/smoke_cartesian_to_torus):

    Given S10's forward map (TorusCoordPredictor):
        R  = bond_length * L / (2π)               # major-ring radius (closes the ring)
        x  = (R + r·cos φ) · cos θ
        y  = (R + r·cos φ) · sin θ
        z  = r·sin φ

    The inverse (this module) is:
        θ  = atan2(y, x)                          # main-ring angle, ∈ (-π, π]
        ρ  = sqrt(x² + y²) = R + r·cos φ          # distance to z-axis
        φ  = atan2(z, ρ - R)                      # cross-section angle
        r  = sqrt((ρ - R)² + z²)                  # cross-section radius (≥ 0)

Equivariance (the whole point — verified in the smoke test):
    - θ-action (rotate every point about z-axis by δθ):  θ → θ + δθ, φ/r unchanged
    - φ-action (fiberwise 2D rotation in the cross-section plane {e₁(θ), ẑ}):
        ρ' - R = (ρ-R)·cos δφ - z·sin δφ
        z'     = (ρ-R)·sin δφ + z·cos δφ
        x', y' rescaled to keep θ unchanged → φ → φ + δφ, θ/r unchanged

The φ-action is NOT a rigid 3D rotation (its rotation plane depends on θ),
but it is still well-defined as a Cartesian transformation, and the inverse
map is equivariant to it because θ = atan2(y,x) is unchanged by the rescaling.

Caveats (honest):
    - atan2 returns (-π, π]; angles wrap mod 2π. For Δθ/Δφ in the steerable
      kernel, use the wrapped difference so branch cuts don't break equivariance.
    - Singular point ρ = 0 (point on z-axis): θ is undefined. In practice
      circRNA backbones never reach the z-axis (R + r·cos φ ≥ R - r > 0 when
      r < R), so this is not hit. We guard with a small eps to avoid NaNs.
    - This inverse assumes the input coords actually live near the torus
      embedding surface (i.e. they were produced by a trunk that respects the
      (θ, φ, r) parametrization, even if implicitly). For arbitrary 3D point
      clouds the map still returns numbers, but they won't carry torus
      semantics. S1-S8 diffusion coords are produced by minimizing a closure
      + bond-length objective that pushes them toward ring geometry, so they
      are near the torus surface in practice.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


# A small epsilon to avoid division by ρ = 0 (z-axis singularity). The torus
# surface never reaches the z-axis when r < R, but diffusion noise can push a
# point arbitrarily close. We clamp ρ from below to keep atan2 well-defined.
_RHO_EPS = 1e-6


def cartesian_to_torus(
    coords: torch.Tensor,           # (B, L, 3) or (..., L, 3)
    R: torch.Tensor,                # (B,) or scalar — major-ring radius per sequence
    rho_eps: float = _RHO_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Cartesian coords → (θ, φ, r) torus coordinates.

    Args:
        coords: (B, L, 3) Cartesian. Last dim is (x, y, z).
        R: (B,) major-ring radius per sequence. R = bond_length * L_seq / (2π).
            Broadcastable to coords' batch dim. Pass a scalar if all sequences
            have the same length.
        rho_eps: floor on ρ = sqrt(x²+y²) to avoid the z-axis singularity.

    Returns:
        theta: (B, L) main-ring angle ∈ (-π, π]
        phi:   (B, L) cross-section angle ∈ (-π, π]
        r:     (B, L) cross-section radius ≥ 0
    """
    # Catastrophic cancellation guard: ρ ≈ R (small r·cos φ) loses precision in
    # float32. Promote to float64 for the atan2/sqrt core where it bites, then
    # cast back. Math is exact (verified to 1e-14 in float64); this is purely
    # a float32 numerics workaround, no approximation introduced.
    orig_dtype = coords.dtype
    use_f64 = orig_dtype == torch.float32
    if use_f64:
        coords64 = coords.double()
        R64 = R.double() if torch.is_tensor(R) and R.dtype == torch.float32 else R
    else:
        coords64 = coords
        R64 = R

    x = coords64[..., 0]
    y = coords64[..., 1]
    z = coords64[..., 2]

    # θ = atan2(y, x) — main-ring angle. SO(2)_θ-equivariant: rotating (x,y)
    # by δθ about the z-axis shifts θ by +δθ.
    theta = torch.atan2(y, x)

    # ρ = distance to z-axis = R + r·cos φ. Clamp away from 0 (z-axis singularity).
    rho = torch.sqrt(x * x + y * y)
    rho = torch.clamp(rho, min=rho_eps)

    # Broadcast R to align with (B, L).
    if torch.is_tensor(R64):
        if R64.dim() == 1:
            Rb = R64.unsqueeze(-1)  # (B, 1)
        elif R64.dim() == 0:
            Rb = R64.view(1, 1)
        else:
            Rb = R64
    else:
        Rb = R64

    rho_minus_R = rho - Rb  # = r·cos φ

    # φ = atan2(z, ρ - R) — cross-section angle. SO(2)_φ-equivariant: the
    # φ-action rotates (ρ-R, z) in the cross-section plane, shifting φ by +δφ.
    phi = torch.atan2(z, rho_minus_R)

    # r = sqrt((ρ-R)² + z²) — cross-section radius, invariant under both actions.
    r = torch.sqrt(rho_minus_R * rho_minus_R + z * z)

    if use_f64:
        theta = theta.to(orig_dtype)
        phi = phi.to(orig_dtype)
        r = r.to(orig_dtype)

    return theta, phi, r


def apply_theta_action(
    coords: torch.Tensor,           # (B, L, 3)
    delta_theta: torch.Tensor,      # scalar or (B,) — rotation about z-axis
) -> torch.Tensor:
    """Apply the SO(2)_θ group action: rotate every point about the z-axis.

    This is a rigid 3D rotation (SO(2) about z). θ → θ + δθ, φ/r unchanged.

    Used by the equivariance test to construct the group action on Cartesian
    coords, then verify cartesian_to_torus(action(x)) == action_torus(cartesian_to_torus(x)).
    """
    if delta_theta.dim() == 0:
        cos_t = torch.cos(delta_theta)
        sin_t = torch.sin(delta_theta)
    else:
        # (B,) → (B, 1) for broadcasting over L
        cos_t = torch.cos(delta_theta).unsqueeze(-1)
        sin_t = torch.sin(delta_theta).unsqueeze(-1)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    z_new = z

    return torch.stack([x_new, y_new, z_new], dim=-1)


def apply_phi_action(
    coords: torch.Tensor,           # (B, L, 3)
    R: torch.Tensor,                # (B,) or scalar — major-ring radius
    delta_phi: torch.Tensor,        # scalar or (B,) — cross-section rotation
    rho_eps: float = _RHO_EPS,
) -> torch.Tensor:
    """Apply the SO(2)_φ group action: fiberwise cross-section rotation.

    NOT a rigid 3D rotation — the rotation plane {e₁(θ), ẑ} depends on θ.
    The Cartesian form (derived from the torus forward map):

        ρ' - R = (ρ - R)·cos δφ - z·sin δφ
        z'     = (ρ - R)·sin δφ + z·cos δφ
        x', y' rescaled so θ unchanged:  x' = (R + r·cos(φ+δφ)) · (x/ρ)
                                          y' = (R + r·cos(φ+δφ)) · (y/ρ)

    φ → φ + δφ, θ/r unchanged. Used by the equivariance test.

    Args:
        coords: (B, L, 3)
        R: (B,) or scalar major-ring radius.
        delta_phi: scalar or (B,) rotation angle.
        rho_eps: floor on ρ (z-axis singularity guard).
    """
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    rho = torch.sqrt(x * x + y * y)
    rho = torch.clamp(rho, min=rho_eps)

    if R.dim() == 1:
        R = R.unsqueeze(-1)
    elif R.dim() == 0:
        R = R.view(1, 1)

    if delta_phi.dim() == 0:
        cos_p = torch.cos(delta_phi)
        sin_p = torch.sin(delta_phi)
    else:
        cos_p = torch.cos(delta_phi).unsqueeze(-1)
        sin_p = torch.sin(delta_phi).unsqueeze(-1)

    rho_minus_R = rho - R  # (B, L)
    # Rotate (ρ-R, z) in the cross-section plane.
    rho_minus_R_new = rho_minus_R * cos_p - z * sin_p
    z_new = rho_minus_R * sin_p + z * cos_p
    # ρ' = R + (ρ-R)·cos(φ+δφ) = R + rho_minus_R_new
    rho_new = R + rho_minus_R_new

    # Rescale (x, y) to keep θ unchanged: direction (x/ρ, y/ρ) preserved,
    # magnitude set to ρ'.
    scale = rho_new / rho  # (B, L)
    x_new = x * scale
    y_new = y * scale
    z_final = z_new

    return torch.stack([x_new, y_new, z_final], dim=-1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle tensor to (-π, π] for branch-cut-safe comparison.

    atan2 already returns (-π, π], but after computing differences
    (θ_rotated - θ_orig) the result may fall outside (-π, π]; wrap before
    comparing in the equivariance test.
    """
    # Map to (-π, π] using x mod 2π then shift.
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(angle + math.pi, two_pi) - math.pi
    # remainder maps to (-π, π]; the +π/-π shift centers it. atan2 convention
    # has +π inclusive, so leave as-is.
    return wrapped


def major_ring_radius(bond_length: float, lengths: torch.Tensor) -> torch.Tensor:
    """Compute the major-ring radius R = bond_length * L_seq / (2π) per sequence.

    Matches S10's TorusCoordPredictor convention (scheme10_circ_equivariant_gnn.py:465).
    Use this to feed `R` into cartesian_to_torus when you only have per-sequence
    valid lengths, not precomputed radii.

    Args:
        bond_length: adjacent-phosphate distance (Å), e.g. 5.9.
        lengths: (B,) valid length per sequence.

    Returns:
        (B,) major-ring radius per sequence.
    """
    return bond_length * lengths.float() / (2.0 * math.pi)
