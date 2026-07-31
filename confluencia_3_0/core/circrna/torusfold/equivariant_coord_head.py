"""
equivariant_coord_head.py — SO(2) 等变 + 轴角参数化坐标头

架构（替代原来的 SO(2) θ/φ 双投影方案）：
=============================================

问题（原方案）：
  theta_vec = SO2EquivariantLinear(d_eq→1)  # (cos θ, sin θ)
  phi_vec   = SO2EquivariantLinear(d_eq→1)  # (cos φ, sin φ)
  两个独立角度从同一个 degree-1 特征投影 → 线性耦合，3D 螺旋无法独立控制

修复（轴角参数化）：
  axis_xy = SO2EquivariantLinear(d_eq→1)  # (cos θ, sin θ) — 等变量
  axis_z  = Linear(d_inv→1)               # z 分量 — 不变量，可学习缩放
  axis    = normalize([axis_xy, axis_z])   # 3D 旋转轴
  angle   = Linear(d_inv→1)               # 旋转角度 — 不变量
  coords  = r · R(axis, angle) · e₀        # Rodrigues 旋转 → 3D 位置

⚠️ 对称性声明（重要）：
  整个网络的等变群是 SO(2)（绕 z 轴旋转），不是 SO(3)。
  axis-angle 参数化能表达任意 3D 方向，但网络对称性仍是 SO(2)。
  对环面 circRNA（天然绕环轴 SO(2) 对称），SO(2) 等变是充分的。
  文档/论文中应声明为 "SO(2) 等变 + 轴角参数化"，不要写 "SO(3) 等变"。

等变性证明：
  SO(2) 旋转 R(ψ) 绕 z 轴作用于 input：
    - axis_xy → R(ψ)·axis_xy（2D 旋转）
    - axis_z  → axis_z（不变，z 方向）
    - axis → (R(ψ)·axis_xy, axis_z) → 3D 轴绕 z 轴旋转
    - angle → angle（不变）
    - r → r（不变）
    - e₀ = [0,0,1] → 绕 z 轴不变
    - coords → R_z(ψ) · coords（3D 坐标绕 z 轴正确变换）
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equivariant import SO2EquivariantLinear


class SO2AxisAngleCoordHead(nn.Module):
    """SO(2) 等变坐标头，轴角参数化。

    关键改动 vs 旧版：
    1. axis_z 不加 tanh 限制，配合可学习缩放因子，让 z 分量能竞争
    2. e₀ = [0,0,1]（z 方向），避免 axis 在 xy 平面时交叉积退化
    3. 对称性声明正确：SO(2) 等变 + 轴角参数化（不是 SO(3) 等变）
    """

    def __init__(
        self,
        d_inv: int,
        d_eq: int,
        d_hidden: int = 64,
        dropout: float = 0.1,
        r_scale: float = 10.0,
    ):
        super().__init__()
        self.r_scale = r_scale

        # ── r 预测（degree-0 不变量）────────────────────────────
        self.r_mlp = nn.Sequential(
            nn.Linear(d_inv, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

        # ── 3D 旋转轴 ───────────────────────────────────────────
        # xy 分量：从 degree-1 投影（SO(2) 等变）
        self.axis_xy_proj = SO2EquivariantLinear(
            d_eq, 1, degree_in=1, degree_out=1, bias=False
        )
        # z 分量：从 degree-0 投影（SO(2) 不变）
        self.axis_z_proj = nn.Linear(d_inv, 1)
        # 可学习 z 缩放：让 axis_z 的量级能与 axis_xy 竞争，避免被归一化稀释
        self.axis_z_scale = nn.Parameter(torch.tensor(1.0))

        # ── 旋转角度（degree-0 不变量）────────────────────────────
        self.angle_proj = nn.Linear(d_inv, 1)

    def forward(
        self,
        latent_inv: torch.Tensor,
        latent_eq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = latent_inv.shape[:2]

        # 1. r 标量
        r_raw = self.r_mlp(latent_inv).squeeze(-1)
        r = (torch.tanh(r_raw) + 1) * 0.5 * self.r_scale

        # 2. 3D 旋转轴
        #    axis_xy 来自 degree-1（等变），axis_z 来自 degree-0（不变）
        #    Fix 1: axis_z 用可学习缩放（而非 tanh 限制），让 z 分量能竞争
        axis_xy = self.axis_xy_proj(latent_eq).squeeze(-2)  # (B, L, 2)
        axis_z = (self.axis_z_scale * self.axis_z_proj(latent_inv)).squeeze(-1)  # (B, L)
        axis = torch.cat([axis_xy, axis_z.unsqueeze(-1)], dim=-1)
        axis = F.normalize(axis, dim=-1, eps=1e-8)  # 归一化为单位向量

        # 3. 旋转角度
        angle = self.angle_proj(latent_inv).squeeze(-1)

        # 4. 轴角 → 3D 位置
        coords = self._axis_angle_to_position(axis, angle, r)

        return coords, r, axis, angle

    @staticmethod
    def _axis_angle_to_position(
        axis: torch.Tensor,
        angle: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Rodrigues 旋转应用于参考方向 e₀，得到 3D 位置。

        Rodrigues:  v_rot = v·cosθ + (axis×v)·sinθ + axis·(axis·v)·(1-cosθ)

        参考方向 e₀ = [0, 0, 1]（Fix 2：改用 z 方向，避免 axis 在 xy 平面时退化）
        当 axis 在 xy 平面（circRNA 常见情况）时，axis ⟂ e₀，
        cross term (axis×e₀) 最大 → 旋转表达力最强。
        """
        device = axis.device
        dtype = axis.dtype

        # e₀ = [0, 0, 1]（z 方向参考）
        ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3)
        v = ref.expand(axis.shape[0], axis.shape[1], 3)  # (B, L, 3)

        cos_a = torch.cos(angle).unsqueeze(-1)
        sin_a = torch.sin(angle).unsqueeze(-1)

        cross = torch.linalg.cross(axis, v, dim=-1)
        dot = torch.sum(axis * v, dim=-1, keepdim=True)

        v_rot = v * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
        coords = r.unsqueeze(-1) * v_rot

        return coords

    def extra_repr(self):
        return f"d_hidden=64, r_scale={self.r_scale}, axis_z_scale={self.axis_z_scale.item():.3f}"


# 向后兼容别名
StrictlyEquivariantCoordHead = SO2AxisAngleCoordHead
SO3AxisAngleCoordHead = SO2AxisAngleCoordHead
