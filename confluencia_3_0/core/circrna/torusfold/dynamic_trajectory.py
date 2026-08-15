"""
dynamic_trajectory.py — 动态构象系综生成器

三层递进架构：
  Layer 1: 温度引导采样 (Temperature-Guided Sampling)
  Layer 2: 势能约束扩散 (Potential-Constrained Diffusion)
  Layer 3: 马尔可夫轨迹 (Markov Trajectory)

原理：
  扩散噪声强度 ↔ 温度 T ↔ Boltzmann 分布
  在扩散去噪的每一步加入物理势能梯度
  从构象系综构建转移矩阵，生成物理合理的时间轨迹

用法：
  generator = DynamicEnsembleGenerator(model, config)
  result = generator.generate(seq_tokens, ...)
  # result.trajectory: (T, L, 3) 动态轨迹
  # result.conformations: (N, L, 3) 静态系综
  # result.free_energy_surface: (N, N) 自由能面
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicEnsembleConfig:
    """动态构象系综配置"""
    # 温度引导
    temperatures: tuple = (0.0, 0.1, 0.3, 0.5, 1.0)
    n_samples_per_temp: int = 20
    cfg_scale: float = 2.0

    # 势能约束
    use_potential_guidance: bool = True
    potential_weight: float = 0.1
    potential_steps: int = 50

    # 马尔可夫轨迹
    use_markov_trajectory: bool = True
    n_trajectory_steps: int = 1000
    rmsd_kernel_sigma: float = 2.0
    interpolation_method: str = 'cubic'


# ══════════════════════════════════════════════════════════════════════════════
# 结果
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicEnsembleResult:
    """动态构象系综结果"""
    # 静态系综
    conformations: torch.Tensor          # (N, L, 3)
    temperatures: List[float]

    # 势能引导构象
    physically_refined: Optional[torch.Tensor] = None  # (N, L, 3)
    energy_scores: Optional[List[float]] = None

    # 动态轨迹
    trajectory: Optional[torch.Tensor] = None          # (T, L, 3)
    transition_matrix: Optional[torch.Tensor] = None   # (N, N)
    residence_times: Optional[List[float]] = None

    # 分析
    rmsf: Optional[torch.Tensor] = None                # (L,)
    free_energy_surface: Optional[torch.Tensor] = None  # (N, N)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: 温度引导采样
# ══════════════════════════════════════════════════════════════════════════════

def temperature_to_cfg_scale(temperature: float, base_cfg: float = 2.0) -> float:
    """温度 → CFG scale 映射

    T=0  → cfg=inf (确定性)
    T=0.3 → cfg=base_cfg (生理温度)
    T=1.0 → cfg=0 (高动态)
    """
    if temperature <= 0:
        return 100.0  # 近似确定性
    return base_cfg / (temperature + 0.01)


def sample_at_temperature(
    model,
    seq_tokens: torch.Tensor,
    temperature: float,
    n_samples: int = 20,
    base_cfg: float = 2.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """在指定温度下采样构象系综

    Args:
        model: 完整模型（含 encoder, s8_refine, latent, coord_diffusion）
        seq_tokens: (B, L) 序列 token
        temperature: 温度参数
        n_samples: 采样数量
        base_cfg: 基础 CFG scale
        seed: 随机种子

    Returns:
        coords: (N, B, L, 3) 构象系综
    """
    cfg_scale = temperature_to_cfg_scale(temperature, base_cfg)
    device = seq_tokens.device

    # 编码只做一次
    model.eval()
    with torch.no_grad():
        node_inv, node_eq, topk_idx = model.encoder(seq_tokens)
        if model.s8_refine is not None:
            node_eq = model.s8_refine(node_inv, node_eq, topk_idx=topk_idx)
        latent_inv, latent_eq = model.latent(node_inv, node_eq)

    # 多次采样
    coords_list = []
    for i in range(n_samples):
        per_seed = (seed + i + 1) if seed is not None else None
        B_l, L_l = latent_inv.shape
        eq_2d = latent_eq.reshape(B_l, L_l, -1, 2)
        coords = model.coord_diffusion.generate(
            cond_inv=latent_inv,
            cond_eq=eq_2d,
            n_steps=50,
            cfg_scale=cfg_scale,
            seed=per_seed,
        )
        coords_list.append(coords)

    return torch.cat(coords_list, dim=0)  # (N*B, L, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: 势能约束扩散
# ══════════════════════════════════════════════════════════════════════════════

class PotentialEnergy(nn.Module):
    """物理势能模块

    组合多种物理约束：
    - 立体化学 (stereo)
    - 空间位阻 (clash)
    - 碱基堆积 (stacking)
    - 圆环闭合 (closure)
    - 碱基配对 (base_pair)
    """

    def __init__(self, bond_length: float = 5.9):
        super().__init__()
        self.bond_length = bond_length

    def compute_bond_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """键长能量: U = sum((d_ij - d0)^2)"""
        # 相邻残基距离
        diff = coords[:, 1:, :] - coords[:, :-1, :]  # (B, L-1, 3)
        dist = torch.norm(diff, dim=-1)  # (B, L-1)
        energy = ((dist - self.bond_length) ** 2).mean(dim=-1)  # (B,)
        return energy

    def compute_clash_energy(self, coords: torch.Tensor,
                              min_dist: float = 3.0) -> torch.Tensor:
        """空间位阻能量: U = sum(max(0, min_dist - d_ij)^2)"""
        B, L, _ = coords.shape
        # 计算所有对距离
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
        dist = torch.norm(diff, dim=-1)  # (B, L, L)

        # 排除相邻残基（已有键长约束）
        mask = torch.ones(L, L, device=coords.device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(L - 1):
            mask[i, i + 1] = False
            mask[i + 1, i] = False

        # 位阻惩罚
        clash = F.relu(min_dist - dist)  # (B, L, L)
        clash = clash * mask.unsqueeze(0).float()
        energy = (clash ** 2).sum(dim=(-1, -2)) / (L * L)  # (B,)
        return energy

    def compute_stacking_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """碱基堆积能量: 相邻碱基的平行性"""
        if coords.shape[1] < 3:
            return torch.zeros(coords.shape[0], device=coords.device)

        # 相邻三个碱基的法向量
        v1 = coords[:, 1:-1, :] - coords[:, :-2, :]  # (B, L-2, 3)
        v2 = coords[:, 2:, :] - coords[:, 1:-1, :]   # (B, L-2, 3)

        # 叉积得到法向量
        normal = torch.cross(v1, v2, dim=-1)  # (B, L-2, 3)
        normal = F.normalize(normal, dim=-1)

        # 相邻法向量的平行性
        cos_sim = (normal[:, :-1, :] * normal[:, 1:, :]).sum(dim=-1)  # (B, L-3)
        energy = -cos_sim.mean(dim=-1)  # (B,) 负号：越平行能量越低
        return energy

    def compute_closure_energy(self, coords: torch.Tensor,
                                is_circular: bool = True) -> torch.Tensor:
        """圆环闭合能量: 首尾距离"""
        if not is_circular:
            return torch.zeros(coords.shape[0], device=coords.device)

        head = coords[:, 0, :]  # (B, 3)
        tail = coords[:, -1, :]  # (B, 3)
        dist = torch.norm(head - tail, dim=-1)  # (B,)
        energy = dist ** 2
        return energy

    def forward(self, coords: torch.Tensor,
                is_circular: bool = True) -> torch.Tensor:
        """计算总势能

        Args:
            coords: (B, L, 3) 坐标
            is_circular: 是否环化

        Returns:
            energy: (B,) 总势能
        """
        E_bond = self.compute_bond_energy(coords)
        E_clash = self.compute_clash_energy(coords)
        E_stacking = self.compute_stacking_energy(coords)
        E_closure = self.compute_closure_energy(coords, is_circular)

        # 加权组合
        total = E_bond + 0.5 * E_clash + 0.3 * E_stacking + 1.0 * E_closure
        return total


def potential_guided_denoise(
    model,
    seq_tokens: torch.Tensor,
    latent_inv: torch.Tensor,
    latent_eq: torch.Tensor,
    potential_weight: float = 0.1,
    n_steps: int = 50,
    n_refine_steps: int = 10,
    is_circular: bool = True,
) -> torch.Tensor:
    """势能约束扩散去噪

    在标准扩散去噪后，用势能梯度精炼坐标

    Args:
        model: 模型（含 coord_diffusion）
        seq_tokens: (B, L) 序列
        latent_inv: (B, L, d_inv)
        latent_eq: (B, L, d_eq)
        potential_weight: 势能权重 α
        n_steps: 扩散步数
        n_refine_steps: 势能精炼步数
        is_circular: 是否环化

    Returns:
        coords: (B, L, 3) 势能精炼后的坐标
    """
    # 1. 标准扩散采样
    B_l, L_l = latent_inv.shape
    eq_2d = latent_eq.reshape(B_l, L_l, -1, 2)
    coords = model.coord_diffusion.generate(
        cond_inv=latent_inv,
        cond_eq=eq_2d,
        n_steps=n_steps,
        cfg_scale=2.0,
    )

    # 2. 势能精炼
    if potential_weight > 0 and n_refine_steps > 0:
        potential = PotentialEnergy(bond_length=model.config.bond_length)
        coords = coords.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([coords], lr=0.01)

        for step in range(n_refine_steps):
            optimizer.zero_grad()
            energy = potential(coords, is_circular=is_circular)
            loss = potential_weight * energy.mean()
            loss.backward()
            optimizer.step()

        coords = coords.detach()

    return coords


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: 马尔可夫轨迹
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmsd_matrix(coords_list: List[torch.Tensor]) -> torch.Tensor:
    """计算 RMSD 矩阵

    Args:
        coords_list: N 个 (L, 3) 坐标

    Returns:
        rmsd_matrix: (N, N) RMSD 矩阵
    """
    n = len(coords_list)
    coords_stack = torch.stack(coords_list)  # (N, L, 3)
    rmsd_matrix = torch.zeros(n, n, device=coords_list[0].device)

    for i in range(n):
        for j in range(i + 1, n):
            diff = coords_stack[i] - coords_stack[j]
            rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    return rmsd_matrix


def build_markov_transition_matrix(
    rmsd_matrix: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    """构建马尔可夫转移矩阵

    使用 Gaussian kernel: T[i,j] = exp(-rmsd[i,j]^2 / 2σ²)

    Args:
        rmsd_matrix: (N, N) RMSD 矩阵
        sigma: Gaussian kernel 带宽

    Returns:
        transition_matrix: (N, N) 归一化转移矩阵
    """
    # Gaussian kernel
    T = torch.exp(-rmsd_matrix ** 2 / (2 * sigma ** 2))

    # 对角线置零（不自跳转）
    T.fill_diagonal_(0)

    # 归一化
    T = T / T.sum(dim=1, keepdim=True)

    return T


def generate_markov_trajectory(
    transition_matrix: torch.Tensor,
    coords_list: List[torch.Tensor],
    n_steps: int = 1000,
    start_state: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """生成马尔可夫轨迹

    Args:
        transition_matrix: (N, N) 转移矩阵
        coords_list: N 个 (L, 3) 坐标
        n_steps: 轨迹步数
        start_state: 起始状态（默认随机）
        seed: 随机种子

    Returns:
        trajectory: (T, L, 3) 平滑轨迹
        state_sequence: 状态序列
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = transition_matrix.shape[0]
    device = transition_matrix.device

    # 起始状态
    if start_state is None:
        start_state = torch.randint(0, N, (1,)).item()

    # 马尔可夫链采样
    state_sequence = [start_state]
    for _ in range(n_steps):
        current = state_sequence[-1]
        probs = transition_matrix[current]
        next_state = torch.multinomial(probs, 1).item()
        state_sequence.append(next_state)

    # 提取坐标序列
    trajectory = torch.stack([coords_list[s] for s in state_sequence])  # (T+1, L, 3)

    return trajectory, state_sequence


def compute_residence_times(state_sequence: List[int], n_states: int) -> List[float]:
    """计算每个态的驻留时间

    Args:
        state_sequence: 状态序列
        n_states: 状态总数

    Returns:
        residence_times: 每个态的平均驻留时间
    """
    residence = [0.0] * n_states
    count = [0] * n_states

    current_state = state_sequence[0]
    duration = 1

    for s in state_sequence[1:]:
        if s == current_state:
            duration += 1
        else:
            residence[current_state] += duration
            count[current_state] += 1
            current_state = s
            duration = 1

    # 平均驻留时间
    for i in range(n_states):
        if count[i] > 0:
            residence[i] /= count[i]

    return residence


def compute_free_energy_surface(
    state_sequence: List[int],
    n_states: int,
    temperature: float = 0.3,
) -> torch.Tensor:
    """计算自由能面

    F(i) = -kT * ln(P(i))
    """
    # 统计每个态的访问频率
    counts = torch.zeros(n_states)
    for s in state_sequence:
        counts[s] += 1

    # 概率分布
    probs = counts / counts.sum()

    # 自由能
    kT = temperature + 1e-6
    F = -kT * torch.log(probs + 1e-6)

    return F


# ══════════════════════════════════════════════════════════════════════════════
# 完整生成器
# ══════════════════════════════════════════════════════════════════════════════

class DynamicEnsembleGenerator:
    """动态构象系综生成器

    三层递进：
      Layer 1: 温度引导采样
      Layer 2: 势能约束扩散
      Layer 3: 马尔可夫轨迹
    """

    def __init__(self, model, config: Optional[DynamicEnsembleConfig] = None):
        self.model = model
        self.config = config or DynamicEnsembleConfig()

    def generate(
        self,
        seq_tokens: torch.Tensor,
        bpp_matrix: Optional[torch.Tensor] = None,
        is_circular: bool = True,
        seed: Optional[int] = None,
    ) -> DynamicEnsembleResult:
        """生成完整的动态构象系综

        Args:
            seq_tokens: (B, L) 序列 token
            bpp_matrix: (L, L) 或 (B, L, L) ViennaRNA bpp（可选）
            is_circular: 是否环化
            seed: 随机种子

        Returns:
            DynamicEnsembleResult
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device
        self.model.eval()

        # ── 编码 ──
        with torch.no_grad():
            fwd_result = self.model.forward(seq_tokens, bpp_matrix=bpp_matrix)
            latent_inv = fwd_result['latent_inv']  # (B, L, d_inv)
            latent_eq = fwd_result['latent_eq']    # (B, L, d_eq)

        # ── Layer 1: 温度引导采样 ──
        all_conformations = []
        all_temperatures = []
        coord_head = self.model.coord_head

        for T in self.config.temperatures:
            n_samples = self.config.n_samples_per_temp
            coords_list = []
            for i in range(n_samples):
                if T <= 0:
                    noisy_inv = latent_inv
                else:
                    noise = torch.randn_like(latent_inv) * T
                    noisy_inv = latent_inv + noise
                latent_final = torch.cat([noisy_inv, latent_eq], dim=-1)
                coords = coord_head(latent_final)
                coords_list.append(coords.squeeze(0).detach())

            all_conformations.extend(coords_list)
            all_temperatures.extend([T] * n_samples)

        conformations = torch.stack(all_conformations)  # (N, L, 3)

        # ── Layer 2: 势能约束精炼 ──（需要梯度）
        physically_refined = None
        energy_scores = None

        if self.config.use_potential_guidance:
            potential = PotentialEnergy(bond_length=getattr(self.model.config, 'bond_length', 5.9))
            refined_list = []
            energy_list = []

            for i in range(len(all_conformations)):
                coords = conformations[i:i+1].clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([coords], lr=0.01)

                for step in range(self.config.potential_steps):
                    optimizer.zero_grad()
                    energy = potential(coords, is_circular=is_circular)
                    loss = self.config.potential_weight * energy.mean()
                    loss.backward()
                    optimizer.step()

                refined_list.append(coords.detach().squeeze(0))
                energy_list.append(energy.item())

            physically_refined = torch.stack(refined_list)
            energy_scores = energy_list

        # ── Layer 3: 马尔可夫轨迹 ──
        trajectory = None
        transition_matrix = None
        residence_times = None
        free_energy = None

        if self.config.use_markov_trajectory and len(all_conformations) >= 2:
            coords_for_rmsd = physically_refined if physically_refined is not None else conformations
            rmsd_matrix = compute_rmsd_matrix([coords_for_rmsd[i] for i in range(len(coords_for_rmsd))])
            transition_matrix = build_markov_transition_matrix(
                rmsd_matrix, sigma=self.config.rmsd_kernel_sigma
            )
            trajectory, state_seq = generate_markov_trajectory(
                transition_matrix,
                [coords_for_rmsd[i] for i in range(len(coords_for_rmsd))],
                n_steps=self.config.n_trajectory_steps,
                seed=seed,
            )
            n_states = len(all_conformations)
            residence_times = compute_residence_times(state_seq, n_states)
            free_energy = compute_free_energy_surface(state_seq, n_states)

        # ── RMSF 分析 ──
        mean_coords = conformations.mean(dim=0)  # (L, 3)
        rmsf = torch.sqrt(((conformations - mean_coords) ** 2).sum(dim=-1).mean(dim=0))  # (L,)

        return DynamicEnsembleResult(
            conformations=conformations,
            temperatures=all_temperatures,
            physically_refined=physically_refined,
            energy_scores=energy_scores,
            trajectory=trajectory,
            transition_matrix=transition_matrix,
            residence_times=residence_times,
            rmsf=rmsf,
            free_energy_surface=free_energy,
        )
