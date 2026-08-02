"""
coord_diffusion.py — SO(2)-equivariant coordinate-level diffusion.

Replaces InvDiffusion + coord_head with direct 3D coordinate denoising.
The denoising target is (B, L, 3) atomic coordinates — same philosophy as
AlphaFold 3's structure module.

SO(2) equivariance for 3D coords:
  (x, y)  → degree-1 vector (rotates under SO(2) around z-axis)
  z       → degree-0 scalar (invariant under SO(2))

Architecture:
  x_noisy (B,L,3) → split (x,y) & z → eq/inv proj → coord_emb (B,L,D)
       ↓
  cond_inv (B,L,d_inv) + cond_eq (B,L,d_eq,2) → cond_emb (B,L,D)
       ↓
  MixedHybridAttention(query=coord_emb, key=value=cond_emb) → x_input
       ↓
  time_embed + x_input → denoiser → noise_pred (B,L,3)
       ↓
  Loss: MSE(noise_pred, real_noise)

Training:  coords_t = sqrt(ᾱ)·coords_0 + sqrt(1-ᾱ)·ε
Inference: x_T ~ N(0,I) → DDIM loop (20 steps) → x_0 (predicted coords)
CFG:      p=0.1 drop cond during training, cfg_scale=2.0 at inference
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equivariant import SO2EquivariantLinear
from mixed_attention import MixedHybridAttention


class CoordDiffusion(nn.Module):
    """SO(2)-equivariant 3D coordinate diffusion module.

    Directly denoises (B, L, 3) coordinates conditioned on structural
    representations (inv/eq latents from the encoder).

    Parameters
    ----------
    d_inv : int
        Dimension of degree-0 (invariant) structural condition.
    d_eq : int
        Dimension of degree-1 (equivariant) structural condition
        — each position carries a 2-vector.
    d_coord_hidden : int
        Internal feature dimension for the coordinate embedding & denoiser.
    n_steps : int
        Number of diffusion timesteps.
    cfg_dropout_prob : float
        Classifier-free guidance dropout probability during training.
    """

    def __init__(
        self,
        d_inv: int = 64,
        d_eq: int = 32,
        d_coord_hidden: int = 128,
        n_steps: int = 100,
        cfg_dropout_prob: float = 0.1,
        use_dynamic_anchors: bool = False,
        anchor_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_inv = d_inv
        self.d_eq = d_eq
        self.d_coord_hidden = d_coord_hidden
        self.n_steps = n_steps
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_dynamic_anchors = use_dynamic_anchors

        # ── Time embedding ────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # ── Coordinate embedding ──────────────────────────────────
        # (x, y) → degree-1 (eq): (B, L, 1, 2)
        self.coord_proj_eq = SO2EquivariantLinear(
            1, d_coord_hidden // 4, degree_in=1, degree_out=1, bias=False,
        )
        # z → degree-0 (inv): (B, L, 1)
        self.coord_proj_inv = nn.Linear(1, d_coord_hidden // 2)

        # ── Condition embeddings ──────────────────────────────────
        # coord_eq: (B,L,d_out,2) → flatten(-2) = (B,L,d_out*2)
        # cond_proj_eq outputs d_coord_hidden//4 → flatten = d_coord_hidden//2
        # cond_proj_inv outputs d_coord_hidden//2 → total = d_coord_hidden ✓
        self.cond_proj_inv = nn.Linear(d_inv, d_coord_hidden // 2)
        self.cond_proj_eq = SO2EquivariantLinear(
            d_eq, d_coord_hidden // 4, degree_in=1, degree_out=1, bias=False,
        )

        # ── Cross-attention (same interface as old cond_attn) ────
        self.cond_attn = MixedHybridAttention(
            d_model=d_coord_hidden,
            n_heads=4,
            window=256,
            n_anchors=128,
            bsj_flank=32,
            dropout=0.1,
            use_dynamic_anchors=use_dynamic_anchors,
            anchor_ratio=anchor_ratio,
        )

        # ── Denoiser ──────────────────────────────────────────────
        self.denoiser = nn.Sequential(
            nn.Linear(d_coord_hidden + 64, d_coord_hidden),
            nn.GELU(),
            nn.Linear(d_coord_hidden, d_coord_hidden),
            nn.GELU(),
            nn.Linear(d_coord_hidden, 3),
        )

        # ── Noise schedule ────────────────────────────────────────
        beta = torch.linspace(1e-4, 0.02, n_steps)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

    # ── public forward ──────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        cond_inv: torch.Tensor,
        cond_eq: torch.Tensor,
        n_steps: Optional[int] = None,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """DDIM inference: x_T ~ N(0,I) → x_0 (predicted coordinates).

        Parameters
        ----------
        cond_inv : (B, L, d_inv)
        cond_eq  : (B, L, d_eq, 2)
        n_steps  : number of DDIM steps (default = self.n_steps)
        cfg_scale : classifier-free guidance scale; 1.0 = no CFG.
        seed      : optional random seed for reproducibility.

        Returns
        -------
        coords : (B, L, 3)
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, L = cond_inv.shape[:2]
        device = cond_inv.device
        n_steps = n_steps or self.n_steps

        x = torch.randn(B, L, 3, device=device)

        for step in range(n_steps - 1, -1, -1):
            # Denoise with CFG: weighted sum of conditional + unconditional
            x_hat = self._denoise(x, step, cond_inv, cond_eq, device)
            if cfg_scale > 1.0:
                x_uncond = self._denoise(
                    x, step, cond_inv, cond_eq, device, dropout_cond=True,
                )
                x_hat = x_hat + cfg_scale * (x_hat - x_uncond)

            # DDIM deterministic update
            alpha_bar_t = self.alpha_bar[step]
            prev_step = step - 1
            alpha_bar_prev = self.alpha_bar[prev_step] if prev_step >= 0 else torch.tensor(1.0, device=device)
            beta_t = self.beta[step]

            sigma = 0.0  # deterministic DDIM (sigma=0)
            x = (
                (x - (1 - alpha_bar_t).sqrt() * x_hat / beta_t.sqrt())
                / alpha_bar_t.sqrt()
                + sigma * beta_t.sqrt() * torch.randn_like(x)
            )
            # Project onto α_prev · x_hat ball
            x_hat_proj = (1 - alpha_bar_prev - sigma ** 2 * beta_t).sqrt() * x_hat
            x = alpha_bar_prev.sqrt() * x + x_hat_proj + sigma * beta_t.sqrt() * torch.randn_like(x)

        return x

    def anchor_aux_loss(self, cond_inv: torch.Tensor, cond_eq: torch.Tensor,
                        pair_probs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss supervising dynamic anchor scorers against pair_probs."""
        if self.use_dynamic_anchors:
            cond_inv_emb = self.cond_proj_inv(cond_inv)
            cond_eq_emb = self.cond_proj_eq(cond_eq).flatten(-2)
            cond_emb = torch.cat([cond_inv_emb, cond_eq_emb], dim=-1)  # [B, L, D]
            # Sum aux losses from all dynamic layers (odd indices in cond_attn.layers)
            total_aux = torch.tensor(0.0, device=cond_inv.device)
            for layer in self.cond_attn.layers:
                if hasattr(layer, 'anchor_aux_loss'):
                    total_aux = total_aux + layer.anchor_aux_loss(cond_emb, pair_probs)
            return total_aux
        return torch.tensor(0.0, device=cond_inv.device)

    def forward(
        self,
        coords: torch.Tensor,
        cond_inv: torch.Tensor,
        cond_eq: torch.Tensor,
        return_noise_pred: bool = False,
        return_x0_pred: bool = False,
    ):
        """Training forward.

        Parameters
        ----------
        coords : (B, L, 3) clean target coordinates.
        cond_inv : (B, L, d_inv)
        cond_eq  : (B, L, d_eq, 2)
        return_noise_pred : if True, also return (B, L, 3) noise_pred.
        return_x0_pred : if True, also return (B, L, 3) x0_pred
            (single-step denoised coords, differentiable — for geometric losses).

        Returns
        -------
        loss : MSE(noise_pred, real_noise)
        noise_pred : only if return_noise_pred=True
        x0_pred : only if return_x0_pred=True

        Returns
        -------
        loss : MSE(noise_pred, real_noise)
        noise_pred : only if return_noise_pred=True
        """
        B, L, _ = coords.shape
        device = coords.device

        # Sample timestep per sample
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Vectorised forward diffusion
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)
        one_minus_bar = 1.0 - alpha_bar_t
        noise = torch.randn_like(coords)
        coords_noisy = alpha_bar_t.sqrt() * coords + one_minus_bar.sqrt() * noise

        # ── Coordinate embedding (split eq/inv by degree) ─────────
        # (x,y) → degree-1 (eq): need (B,L,1,2) for SO2EquivariantLinear
        coord_eq = self.coord_proj_eq(coords_noisy[:, :, :2].unsqueeze(-2))  # (B,L,d_out,2)
        # z → degree-0 (inv): plain Linear on (B,L,1)
        coord_inv = self.coord_proj_inv(coords_noisy[:, :, 2:])  # (B,L,d_out)
        coord_emb = torch.cat(
            [coord_eq.flatten(-2), coord_inv], dim=-1,
        )  # (B,L,D)

        # ── Condition embedding ───────────────────────────────────
        cond_inv_emb = self.cond_proj_inv(cond_inv)  # (B,L,D//4)
        cond_eq_emb = self.cond_proj_eq(cond_eq).flatten(-2)  # (B,L,D//4)
        cond_emb = torch.cat([cond_inv_emb, cond_eq_emb], dim=-1)  # (B,L,D)

        # ── Classifier-free guidance dropout (whole-batch) ────────
        cond_drop = cond_emb.clone()
        if self.training and self.cfg_dropout_prob > 0:
            if torch.rand(1, device=device).item() < self.cfg_dropout_prob:
                cond_drop = cond_drop * 0.0  # drop cond for uncond path

        # ── Cross-attention ───────────────────────────────────────
        x_cond = self.cond_attn(coord_emb, cond_drop, cond_drop)
        x_input = coord_emb + x_cond

        # ── Time embedding ────────────────────────────────────────
        t_frac = (t.float() / self.n_steps).unsqueeze(-1)  # (B,1)
        t_emb = self.time_embed(t_frac).unsqueeze(1).expand(B, L, -1)  # (B,L,64)

        # ── Denoise ───────────────────────────────────────────────
        noise_pred = self.denoiser(torch.cat([x_input, t_emb], dim=-1))  # (B,L,3)

        loss = F.mse_loss(noise_pred, noise)

        if return_noise_pred and return_x0_pred:
            # One-step x0 prediction (differentiable, single forward — fast)
            x0_pred = (coords_noisy - one_minus_bar.sqrt() * noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-6)
            return loss, noise_pred, x0_pred
        if return_noise_pred:
            return loss, noise_pred
        return loss

    # ── internal helpers ────────────────────────────────────────────

    def _denoise(
        self,
        x: torch.Tensor,
        t: int,
        cond_inv: torch.Tensor,
        cond_eq: torch.Tensor,
        device: torch.device,
        dropout_cond: bool = False,
    ) -> torch.Tensor:
        """Single-step noise prediction (shared by training & inference)."""
        B, L, _ = x.shape
        t_scalar = torch.full((B,), t, device=device, dtype=torch.long)

        # Coordinate embedding
        coord_eq = self.coord_proj_eq(x[:, :, :2].unsqueeze(-2))  # (B,L,1,2)
        coord_inv = self.coord_proj_inv(x[:, :, 2:])  # (B,L,1)
        coord_emb = torch.cat([coord_eq.flatten(-2), coord_inv], dim=-1)

        # Condition embedding
        cond_inv_emb = self.cond_proj_inv(cond_inv)
        cond_eq_emb = self.cond_proj_eq(cond_eq).flatten(-2)
        cond_emb = torch.cat([cond_inv_emb, cond_eq_emb], dim=-1)

        if dropout_cond:
            cond_emb = cond_emb * 0.0

        x_cond = self.cond_attn(coord_emb, cond_emb, cond_emb)
        x_input = coord_emb + x_cond

        t_frac = (t_scalar.float() / self.n_steps).unsqueeze(-1)
        t_emb = self.time_embed(t_frac).unsqueeze(1).expand(B, L, -1)

        return self.denoiser(torch.cat([x_input, t_emb], dim=-1))


# ── DDIM sampler helper (used by strict S10 inference) ──────────────

@torch.no_grad()
def ddim_sample(
    diffusion: CoordDiffusion,
    cond_inv: torch.Tensor,
    cond_eq: torch.Tensor,
    n_steps: Optional[int] = None,
    cfg_scale: float = 2.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Run the DDIM reverse process and return (B, L, 3) coordinates."""
    return diffusion.generate(
        cond_inv=cond_inv,
        cond_eq=cond_eq,
        n_steps=n_steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
