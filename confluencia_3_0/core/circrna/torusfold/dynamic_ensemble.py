"""
dynamic_ensemble.py — 构象系综生成器

核心能力：
1. 条件构象采样：同一序列，不同随机种子 → 多个合理构象
2. 温度/引导强度控制：w=0（高动态）→ w=10（刚性稳定）
3. 时间步插值：扩散反向过程的中间步骤 = 折叠轨迹可视化

本质：把扩散模型从"回归器"还原为"生成器"
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnsembleResult:
    """构象系综结果"""
    coords_list: List[torch.Tensor]        # N 个构象 (N, L, 3)
    mean_coords: torch.Tensor               # 平均构象 (L, 3)
    b_factors: torch.Tensor                 # 每个位置的 RMSF (L,)
    trajectories: Optional[List[torch.Tensor]] = None  # 折叠轨迹（可选）


class DDIMSampler:
    """DDIM 采样器（确定性，可加速）"""

    def __init__(self, diffusion_model, n_steps: int = 50):
        """
        Args:
            diffusion_model: InvDiffusion 模块
            n_steps: 采样步数（默认 50，比训练时的 100 快 2 倍）
        """
        self.diffusion = diffusion_model
        self.n_steps = n_steps

        # DDIM 时间步序列（均匀间隔）
        self.timesteps = torch.linspace(
            diffusion_model.n_steps - 1, 0, n_steps
        ).long()

        # 预计算 DDIM 系数
        self.alpha_bar = diffusion_model.alpha_bar

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        condition: torch.Tensor = None,
        cfg_scale: float = 1.0,
        return_trajectory: bool = False,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        DDIM 采样

        Args:
            shape: (B, L, d_inv)
            condition: 条件特征（用于 CFG）
            cfg_scale: Classifier-Free Guidance 强度
                - 0.0: 无引导，高多样性
                - 1.0: 标准引导
                - 5.0+: 强引导，趋向稳定构象
            return_trajectory: 是否返回中间步骤
            device: 设备

        Returns:
            x_0: 最终采样结果
            trajectory: 中间步骤列表（如果 return_trajectory=True）
        """
        if device is None:
            device = next(self.diffusion.parameters()).device

        B, L, D = shape

        # 初始噪声
        x_t = torch.randn(shape, device=device)

        trajectory = [] if return_trajectory else None

        for i, t in enumerate(self.timesteps):
            t_next = self.timesteps[i + 1] if i < len(self.timesteps) - 1 else torch.tensor(-1)

            # 预测噪声
            if cfg_scale > 0 and condition is not None:
                # Classifier-Free Guidance
                noise_cond = self._predict_noise(x_t, t, condition, device)
                noise_uncond = self._predict_noise(x_t, t, None, device)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self._predict_noise(x_t, t, condition, device)

            # DDIM 更新公式
            alpha_bar_t = self.alpha_bar[t].to(device)
            alpha_bar_next = self.alpha_bar[t_next].to(device) if t_next >= 0 else torch.tensor(1.0, device=device)

            # x_0 预测
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # 方向指向 x_t
            direction = torch.sqrt(1 - alpha_bar_next) * noise_pred

            # DDIM 更新
            x_t = torch.sqrt(alpha_bar_next) * x_0_pred + direction

            if return_trajectory:
                trajectory.append(x_t.clone())

        return x_t, trajectory

    def _predict_noise(self, x_t: torch.Tensor, t: int, condition: torch.Tensor, device: torch.device) -> torch.Tensor:
        """预测噪声（推理路径，不重复加噪）

        复刻 InvDiffusion.forward 里的 cond_attn + denoiser 逻辑，
        但跳过 add_noise 步骤（x_t 已经是加噪后的状态）。

        condition = latent_eq: 通过交叉注意力注入条件信息。
        condition = None: unconditional 分支（CFG 必需）。
        """
        B, L, D = x_t.shape

        # 时间步嵌入（与 InvDiffusion.forward 一致）
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        t_frac = (t_tensor.float() / self.diffusion.n_steps).unsqueeze(-1)
        t_emb = self.diffusion.time_embed(t_frac).unsqueeze(1).expand(B, L, -1)

        # 条件交叉注意力（与 InvDiffusion.forward 一致）
        if condition is not None:
            cond_flat = condition.reshape(B, L, -1)
            cond_emb = self.diffusion.cond_proj(cond_flat)
            x_cond, _ = self.diffusion.cond_attn(query=x_t, key=cond_emb, value=cond_emb)
            x_input = x_t + x_cond
        else:
            x_input = x_t

        x_input = torch.cat([x_input, t_emb], dim=-1)
        return self.diffusion.denoiser(x_input)


class ConformationalEnsembleGenerator:
    """构象系综生成器

    核心能力：
    1. 多构象采样：同一序列 → 多个合理构象
    2. 温度控制：调节引导强度 w
    3. 折叠轨迹：提取中间步骤
    """

    def __init__(self, model: nn.Module, d_inv: int, d_eq: int, n_diffusion_steps: int = 100):
        """
        Args:
            model: StrictlyEquivariantS10 模型
            d_inv: inv 通道维度
            d_eq: eq 通道维度
            n_diffusion_steps: 扩散步数
        """
        self.model = model
        self.d_inv = d_inv
        self.d_eq = d_eq

        # 创建 DDIM 采样器
        self.sampler = DDIMSampler(model.diffusion, n_steps=min(50, n_diffusion_steps))

    @torch.no_grad()
    def generate_ensemble(
        self,
        seq_tokens: torch.Tensor,
        n_samples: int = 100,
        cfg_scale: float = 1.0,
        return_trajectory: bool = False,
        seed: int = None,
    ) -> EnsembleResult:
        """
        生成构象系综

        Args:
            seq_tokens: (B, L) 序列 token
            n_samples: 采样数量
            cfg_scale: 引导强度（0=高动态，1=标准，5+=刚性）
            return_trajectory: 是否返回折叠轨迹
            seed: 随机种子（可选）

        Returns:
            EnsembleResult: 包含多个构象、平均构象、RMSF
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 1. 编码序列（只做一次）
        self.model.eval()
        node_repr_inv, node_repr_eq, topk_idx = self.model.encoder(seq_tokens)
        latent_inv, latent_eq = self.model.latent(node_repr_inv, node_repr_eq)

        if self.model.s8_refine is not None:
            latent_eq = self.model.s8_refine(latent_inv, latent_eq)

        # 2. 多次采样
        coords_list = []
        trajectories_list = [] if return_trajectory else None

        for i in range(n_samples):
            # 采样 latent_inv（通过 cond_attn 注入 latent_eq 条件）
            z_inv, trajectory = self.sampler.sample(
                shape=(B, L, self.d_inv),
                condition=latent_eq,
                cfg_scale=cfg_scale,
                return_trajectory=return_trajectory,
                device=device,
            )

            # 新版 SO2AxisAngleCoordHead 直接返回 coords
            coords, _, _, _ = self.model.coord_head(z_inv, latent_eq)
            coords_list.append(coords.squeeze(0))  # (L, 3)

            if return_trajectory:
                traj_coords = []
                for z_t in trajectory:
                    c_t, _, _, _ = self.model.coord_head(z_t, latent_eq)
                    traj_coords.append(c_t.squeeze(0))
                trajectories_list.append(traj_coords)

        coords_stack = torch.stack(coords_list)  # (N, L, 3)

        # 3. 计算平均构象和 RMSF
        mean_coords = coords_stack.mean(dim=0)  # (L, 3)
        deviations = torch.sqrt(((coords_stack - mean_coords.unsqueeze(0)) ** 2).sum(dim=-1))  # (N, L)
        b_factors = deviations.mean(dim=0)  # (L,)

        return EnsembleResult(
            coords_list=coords_list,
            mean_coords=mean_coords,
            b_factors=b_factors,
            trajectories=trajectories_list,
        )

    @torch.no_grad()
    def generate_temperature_sweep(
        self,
        seq_tokens: torch.Tensor,
        cfg_scales: List[float] = [0.0, 1.0, 3.0, 5.0, 10.0],
        n_samples_per_temp: int = 20,
    ) -> dict:
        """
        温度扫描：不同引导强度下的构象分布

        物理类比：
        - w=0: 高温 MD（高动态）
        - w=1: 室温 MD
        - w=5: 低温 MD
        - w=10: 能量最小化

        Returns:
            dict: {cfg_scale: EnsembleResult}
        """
        results = {}
        for w in cfg_scales:
            print(f"  Sampling with w={w}...")
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
    """计算 RMSD"""
    diff = coords1 - coords2
    return torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()


def compute_pairwise_rmsd_matrix(coords_list: List[torch.Tensor]) -> torch.Tensor:
    """计算构象间的成对 RMSD 矩阵"""
    n = len(coords_list)
    rmsd_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = compute_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    return rmsd_matrix


def cluster_conformations(coords_list: List[torch.Tensor], n_clusters: int = 5) -> dict:
    """构象聚类（简单 k-means）"""
    # 展平坐标
    n = len(coords_list)
    L = coords_list[0].shape[0]
    X = torch.stack([c.flatten() for c in coords_list])  # (N, L*3)

    # 简单 k-means
    centroids = X[torch.randperm(n)[:n_clusters]]

    for _ in range(100):
        # 分配
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=1)

        # 更新
        new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(n_clusters)])
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    # 统计
    cluster_counts = torch.bincount(labels, minlength=n_clusters)

    return {
        'labels': labels,
        'counts': cluster_counts,
        'centroids': centroids,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("构象系综生成器测试")
    print("=" * 60)
    print()
    print("核心功能：")
    print("  1. generate_ensemble(): 多构象采样")
    print("  2. generate_temperature_sweep(): 温度扫描")
    print("  3. cluster_conformations(): 构象聚类")
    print()
    print("物理类比：")
    print("  w=0  → 高温 MD（高动态）")
    print("  w=1  → 室温 MD")
    print("  w=5  → 低温 MD")
    print("  w=10 → 能量最小化")
    print()
    print("发表卖点：")
    print("  同一序列 → 构象系综 → 动态柔性 → Nature Methods 级别")
    print("=" * 60)