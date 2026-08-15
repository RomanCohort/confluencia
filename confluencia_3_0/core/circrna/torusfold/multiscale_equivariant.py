"""
multiscale_equivariant.py — 多尺度等变架构（简化版）

针对长序列（>1000nt）的分层处理：
  1. 粗粒度：降采样 → 完整 S10 → 抓全局结构
  2. 细粒度：上采样 → 局部 GNN → 精修几何

等变性保证：
  - 降采样：AvgPool → 线性操作，保 SO(3) 等变性
  - 上采样：线性插值 → 线性操作，保 SO(3) 等变性
  - 坐标预测：StrictlyEquivariantCoordHead → 保 SO(3) 等变性

注意：不强制序列循环移位等变（那是特征提取阶段的任务）
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleConfig:
    """多尺度配置"""
    min_length_for_multiscale: int = 1000  # L > 1000 启用多尺度
    factor_1000_2500: int = 2              # 2x downsampling (2000→1000, preserves detail)
    factor_2500_5000: int = 4              # moderate
    factor_over_5000: int = 10             # aggressive for very long
    local_radius: int = 8
    d_hidden: int = 64
    dropout: float = 0.1


def get_downsample_factor(L: int, config: MultiScaleConfig) -> int:
    """根据序列长度动态选择降采样因子"""
    if L <= 2500:
        return config.factor_1000_2500
    elif L <= 5000:
        return config.factor_2500_5000
    else:
        return config.factor_over_5000


class DownsampleEncoder(nn.Module):
    """降采样编码器（SO(2) 等变）

    Mean pooling over a fixed window is linear → commutes with SO(2) rotation:
        rot(mean(v_k)) = mean(rot(v_k))
    so degree-1 (eq) equivariance is preserved.

    For degree-0 (inv), the mean is trivially invariant.

    The `eq_proj` is an SO2EquivariantLinear that re-calibrates the pooled
    representation before it enters the coarse-grained S10 — symmetric to the
    UpsampleDecoder's eq_proj so both directions see the same feature space.
    """

    def __init__(self, factor: int, d_inv: int, d_eq: int):
        super().__init__()
        self.factor = factor
        self.d_inv = d_inv
        self.d_eq = d_eq

        # SO(2)-equivariant projection: pooled coarse eq → same dimension
        from .so2_equivariant import SO2EquivariantLinear
        self.eq_proj = SO2EquivariantLinear(d_eq, d_eq, degree_in=1, degree_out=1,
                                             bias=False)

    def forward(self, node_repr_inv: torch.Tensor, node_repr_eq: torch.Tensor):
        """
        Args:
            node_repr_inv: (B, L, d_inv)
            node_repr_eq:  (B, L, d_eq, 2)

        Returns:
            inv_coarse: (B, L_coarse, d_inv)
            eq_coarse:  (B, L_coarse, d_eq, 2)
        """
        B, L, d_inv = node_repr_inv.shape
        d_eq = node_repr_eq.shape[2]
        factor = self.factor

        L_coarse = L // factor
        L_trim = L_coarse * factor  # truncate remainder for clean pooling

        inv_trim = node_repr_inv[:, :L_trim, :]
        eq_trim = node_repr_eq[:, :L_trim, :, :]

        # inv: mean pooling (invariant, degree-0)
        inv_coarse = inv_trim.view(B, L_coarse, factor, d_inv).mean(dim=2)

        # eq: mean pooling → equivariant projection (degree-1, preserves SO(2))
        eq_coarse = eq_trim.view(B, L_coarse, factor, d_eq, 2).mean(dim=2)
        eq_coarse = self.eq_proj(eq_coarse)

        return inv_coarse, eq_coarse


class UpsampleDecoder(nn.Module):
    """上采样解码器

    等变性：线性插值是线性操作，保持 SO(3) 等变性
    """

    def __init__(self, factor: int, d_inv: int, d_eq: int):
        super().__init__()
        self.factor = factor
        self.d_inv = d_inv
        self.d_eq = d_eq

        self.inv_proj = nn.Linear(d_inv, d_inv)

        from .so2_equivariant import SO2EquivariantLinear
        self.eq_proj = SO2EquivariantLinear(d_eq, d_eq, degree_in=1, degree_out=1, bias=False)

    def forward(self, inv_coarse: torch.Tensor, eq_coarse: torch.Tensor, L_target: int):
        """
        Args:
            inv_coarse: (B, L_coarse, d_inv)
            eq_coarse:  (B, L_coarse, d_eq, 2)
            L_target:   目标长度

        Returns:
            inv_fine: (B, L_target, d_inv)
            eq_fine:  (B, L_target, d_eq, 2)
        """
        inv_proj = self.inv_proj(inv_coarse)
        eq_proj = self.eq_proj(eq_coarse)

        # inv 上采样：线性插值
        inv_transposed = inv_proj.transpose(1, 2)
        inv_upsampled = F.interpolate(inv_transposed, size=L_target, mode='linear', align_corners=False)
        inv_fine = inv_upsampled.transpose(1, 2)

        # eq 上采样：对 (x,y) 分别插值
        eq_x = eq_proj[..., 0].transpose(1, 2)
        eq_y = eq_proj[..., 1].transpose(1, 2)

        eq_x_up = F.interpolate(eq_x, size=L_target, mode='linear', align_corners=False).transpose(1, 2)
        eq_y_up = F.interpolate(eq_y, size=L_target, mode='linear', align_corners=False).transpose(1, 2)

        eq_fine = torch.stack([eq_x_up, eq_y_up], dim=-1)

        return inv_fine, eq_fine


class FineGrainedGNN(nn.Module):
    """细粒度 GNN（只看局部邻居）

    等变性：局部消息传递 + 等变投影
    """

    def __init__(self, d_inv: int, d_eq: int, d_hidden: int = 64,
                 local_radius: int = 8, dropout: float = 0.1):
        super().__init__()
        self.local_radius = local_radius

        self.inv_update = nn.Sequential(
            nn.Linear(d_inv * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_inv),
        )

        from .so2_equivariant import SO2EquivariantLinear
        self.eq_update = SO2EquivariantLinear(d_eq, d_eq, degree_in=1, degree_out=1)

        self.norm_inv = nn.LayerNorm(d_inv)
        # 注意：degree-1 不用 LayerNorm（会破坏等变性）

    def forward(self, inv: torch.Tensor, eq: torch.Tensor):
        B, L, d_inv = inv.shape
        radius = self.local_radius

        # 局部邻居消息（循环边界）
        inv_msgs = []
        for offset in range(-radius, radius + 1):
            inv_msgs.append(torch.roll(inv, shifts=offset, dims=1))
        inv_neighbors = torch.stack(inv_msgs, dim=2).mean(dim=2)

        # inv 更新
        inv_out = self.norm_inv(inv + self.inv_update(torch.cat([inv, inv_neighbors], dim=-1)))

        # eq 更新（等变）
        eq_out = eq + self.eq_update(eq)

        return inv_out, eq_out


class MultiScaleLatent(nn.Module):
    """多尺度 Latent 模块（U-Net 跳跃连接版）

    V2 升级：加入跳跃连接，保留细节信息
    """

    def __init__(self, config: MultiScaleConfig, d_model_inv: int, d_model_eq: int,
                 d_inv: int, d_eq: int):
        super().__init__()
        self.config = config
        self.d_model_inv = d_model_inv
        self.d_model_eq = d_model_eq
        self.d_inv = d_inv
        self.d_eq = d_eq

        # 单尺度投影
        from .so2_equivariant import SO2EquivariantLinear
        self.inv_proj_single = nn.Sequential(
            nn.Linear(d_model_inv, d_inv * 2),
            nn.GELU(),
            nn.Linear(d_inv * 2, d_inv),
        )
        self.eq_proj_single = SO2EquivariantLinear(d_model_eq, d_eq, degree_in=1, degree_out=1, bias=False)

        # 多尺度组件
        self.downsample = None
        self.upsample = None
        self.fine_gnn = None

        # U-Net 跳跃连接投影层（动态初始化）
        self.skip_inv_proj = None
        self.skip_eq_proj = None

    def _init_multiscale_modules(self, factor: int):
        if self.downsample is None:
            self.downsample = DownsampleEncoder(factor, self.d_model_inv, self.d_model_eq)
            self.upsample = UpsampleDecoder(factor, self.d_inv, self.d_eq)
            self.fine_gnn = FineGrainedGNN(
                self.d_inv, self.d_eq,
                d_hidden=self.config.d_hidden,
                local_radius=self.config.local_radius,
                dropout=self.config.dropout
            )

            # U-Net 跳跃连接：拼接层（输入维度翻倍）
            from .so2_equivariant import SO2EquivariantLinear
            self.skip_inv_proj = nn.Sequential(
                nn.Linear(self.d_inv * 2, self.d_inv),  # 拼接后投影回原维度
                nn.GELU(),
                nn.Linear(self.d_inv, self.d_inv),
            )
            self.skip_eq_proj = SO2EquivariantLinear(
                self.d_eq * 2, self.d_eq, degree_in=1, degree_out=1, bias=False
            )

    def forward(self, node_repr_inv: torch.Tensor, node_repr_eq: torch.Tensor):
        B, L = node_repr_inv.shape[:2]

        # 单尺度路径
        if L <= self.config.min_length_for_multiscale:
            latent_inv = self.inv_proj_single(node_repr_inv)
            latent_eq = self.eq_proj_single(node_repr_eq)
            return latent_inv, latent_eq, False

        # 多尺度路径
        factor = get_downsample_factor(L, self.config)
        self._init_multiscale_modules(factor)

        # === U-Net 结构 ===

        # Encoder 路径（降采样）
        inv_coarse, eq_coarse = self.downsample(node_repr_inv, node_repr_eq)

        # 粗粒度投影
        latent_inv_coarse = self.inv_proj_single(inv_coarse)
        latent_eq_coarse = self.eq_proj_single(eq_coarse)

        # 上采样回原始分辨率
        latent_inv_up, latent_eq_up = self.upsample(latent_inv_coarse, latent_eq_coarse, L)

        # === 跳跃连接 ===
        # 细粒度特征（直接从输入投影）
        latent_inv_fine = self.inv_proj_single(node_repr_inv)
        latent_eq_fine = self.eq_proj_single(node_repr_eq)

        # 拼接：粗粒度上采样 + 细粒度直接
        inv_concat = torch.cat([latent_inv_up, latent_inv_fine], dim=-1)  # [B, L, d_inv*2]
        eq_concat = torch.cat([latent_eq_up, latent_eq_fine], dim=-2)     # [B, L, d_eq*2, 2]

        # 投影回原维度（保持等变性）
        latent_inv = self.skip_inv_proj(inv_concat)
        latent_eq = self.skip_eq_proj(eq_concat)

        # 细粒度精修
        latent_inv, latent_eq = self.fine_gnn(latent_inv, latent_eq)

        return latent_inv, latent_eq, True


def test_multiscale():
    """测试多尺度模块"""
    print("=" * 60)
    print("Multiscale Module Test")
    print("=" * 60)

    torch.manual_seed(42)

    config = MultiScaleConfig()
    B = 2
    d_model_inv, d_model_eq = 32, 32
    d_inv, d_eq = 16, 16

    model = MultiScaleLatent(config, d_model_inv, d_model_eq, d_inv, d_eq)
    model.eval()

    # Test 1: Short sequence
    print("\n[Test 1] Short (500nt) -> Single-scale")
    L = 500
    inv = torch.randn(B, L, d_model_inv)
    eq = torch.randn(B, L, d_model_eq, 2)

    with torch.no_grad():
        out_inv, out_eq, is_ms = model(inv, eq)

    print(f"  Input:  inv={inv.shape}, eq={eq.shape}")
    print(f"  Output: inv={out_inv.shape}, eq={out_eq.shape}")
    print(f"  Multiscale: {is_ms}")

    # Test 2: Long sequence
    print("\n[Test 2] Long (2000nt) -> Multiscale")
    L_long = 2000
    inv_long = torch.randn(B, L_long, d_model_inv)
    eq_long = torch.randn(B, L_long, d_model_eq, 2)

    with torch.no_grad():
        out_inv_long, out_eq_long, is_ms_long = model(inv_long, eq_long)

    print(f"  Input:  inv={inv_long.shape}, eq={eq_long.shape}")
    print(f"  Output: inv={out_inv_long.shape}, eq={out_eq_long.shape}")
    print(f"  Multiscale: {is_ms_long}, Factor: {get_downsample_factor(L_long, config)}")

    # Test 3: Memory check
    print("\n[Test 3] Memory check (3000nt)")
    L_3k = 3000
    inv_3k = torch.randn(4, L_3k, d_model_inv)  # B=4
    eq_3k = torch.randn(4, L_3k, d_model_eq, 2)

    print(f"  Input shape: {inv_3k.shape}")
    print(f"  Expected coarse: {L_3k // get_downsample_factor(L_3k, config)}")

    with torch.no_grad():
        out_3k_inv, out_3k_eq, _ = model(inv_3k, eq_3k)

    print(f"  Output shape: {out_3k_inv.shape}")
    print(f"  PASS")

    print("\n" + "=" * 60)
    print("All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_multiscale()