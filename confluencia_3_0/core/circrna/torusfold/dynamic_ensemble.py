"""
dynamic_ensemble.py — 构象系综生成器 (v4: 坐标扩散版)

v4 关键变化：
  - 不再使用 InvDiffusion + DDIMSampler + coord_head 三段式流程
  - 直接调用 CoordDiffusion.generate() 输出 (B,L,3) 坐标
  - 每次采样用不同 seed → 构象系综

流程：
  seq_tokens → encoder → s8_refine → latent (inv + eq)
            → [N] CoordDiffusion.generate(seed=i, cfg_scale=w) → coords (B,L,3)
            → RMSF / cluster
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class EnsembleResult:
    """构象系综结果"""
    coords_list: List[torch.Tensor]     # N 个构象 (L, 3)
    mean_coords: torch.Tensor           # 平均构象 (L, 3)
    b_factors: torch.Tensor             # 每个位置的 RMSF (L,)
    trajectories: Optional[List[torch.Tensor]] = None  # 折叠轨迹（可选）


class ConformationalEnsembleGenerator:
    """构象系综生成器 (v4: 坐标扩散)"""

    def __init__(self, model: nn.Module, d_inv: int, d_eq: int,
                 n_diffusion_steps: int = 100):
        self.model = model
        self.d_inv = d_inv
        self.d_eq = d_eq
        self.n_diffusion_steps = n_diffusion_steps

    @torch.no_grad()
    def _encode(self, seq_tokens: torch.Tensor):
        """编码序列 → (latent_inv, latent_eq)，同训练路径"""
        node_repr_inv, node_repr_eq, topk_idx = self.model.encoder(seq_tokens)
        if self.model.s8_refine is not None:
            node_repr_eq = self.model.s8_refine(
                node_repr_inv, node_repr_eq, topk_idx=topk_idx,
            )
        latent_inv, latent_eq = self.model.latent(node_repr_inv, node_repr_eq)
        return latent_inv, latent_eq

    @torch.no_grad()
    def generate_ensemble(
        self,
        seq_tokens: torch.Tensor,
        n_samples: int = 100,
        cfg_scale: float = 2.0,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
    ) -> EnsembleResult:
        """
        生成构象系综

        Args:
            seq_tokens : (B, L)
            n_samples  : 采样数量
            cfg_scale  : CFG 强度（0=无引导高动态，1=标准，2+=刚性）
            return_trajectory : 预留（当前不返回中间步）
            seed       : 随机种子（可选）

        Returns:
            EnsembleResult
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, L = seq_tokens.shape
        self.model.eval()
        diffusion = self.model.coord_diffusion
        if diffusion is None:
            raise RuntimeError("model.coord_diffusion is None; use use_coord_diffusion=True")

        # 编码只做一次
        latent_inv, latent_eq = self._encode(seq_tokens)

        coords_list = []
        for i in range(n_samples):
            # 每次用独立种子 → 不同构象
            per_seed = (seed + i + 1) if seed is not None else None
            coords = diffusion.generate(
                cond_inv=latent_inv,
                cond_eq=latent_eq,
                n_steps=min(self.n_diffusion_steps, 50),
                cfg_scale=cfg_scale,
                seed=per_seed,
            )  # (B, L, 3)
            coords_list.append(coords.squeeze(0))  # (L, 3)

        coords_stack = torch.stack(coords_list)  # (N, L, 3)
        mean_coords = coords_stack.mean(dim=0)
        deviations = torch.sqrt(
            ((coords_stack - mean_coords.unsqueeze(0)) ** 2).sum(dim=-1),
        )  # (N, L)
        b_factors = deviations.mean(dim=0)

        return EnsembleResult(
            coords_list=coords_list,
            mean_coords=mean_coords,
            b_factors=b_factors,
        )

    @torch.no_grad()
    def generate_temperature_sweep(
        self,
        seq_tokens: torch.Tensor,
        cfg_scales: List[float] = None,
        n_samples_per_temp: int = 20,
    ) -> dict:
        """温度扫描：不同 CFG 强度下的构象分布"""
        if cfg_scales is None:
            cfg_scales = [0.0, 1.0, 3.0, 5.0, 10.0]
        results = {}
        for w in cfg_scales:
            print(f"  Sampling with cfg_scale={w}...")
            results[w] = self.generate_ensemble(
                seq_tokens,
                n_samples=n_samples_per_temp,
                cfg_scale=w,
            )
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 分析工具
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    diff = coords1 - coords2
    return torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()


def compute_pairwise_rmsd_matrix(coords_list: List[torch.Tensor]) -> torch.Tensor:
    n = len(coords_list)
    rmsd_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            r = compute_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = r
            rmsd_matrix[j, i] = r
    return rmsd_matrix


def cluster_conformations(coords_list: List[torch.Tensor], n_clusters: int = 5) -> dict:
    n = len(coords_list)
    L = coords_list[0].shape[0]
    X = torch.stack([c.flatten() for c in coords_list])  # (N, L*3)
    centroids = X[torch.randperm(n)[:n_clusters]]

    for _ in range(100):
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=1)
        new_centroids = torch.stack(
            [X[labels == k].mean(dim=0) for k in range(n_clusters)],
        )
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    cluster_counts = torch.bincount(labels, minlength=n_clusters)
    return {"labels": labels, "counts": cluster_counts, "centroids": centroids}


if __name__ == "__main__":
    print("dynamic_ensemble.py (v4)")
    print("  Usage:  ConformationalEnsembleGenerator(model, d_inv, d_eq, n_diffusion_steps)")
    print("          generator.generate_ensemble(seq_tokens, n_samples=100, cfg_scale=2.0)")
