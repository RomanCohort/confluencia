"""
so2_equivariant.py — SO(2) 等变基础模块

核心约束：W @ R(θ) = R(θ) @ W

对于 SO(2) 群：
- degree 0（标量）：旋转不变，可用任意 Linear
- degree 1（向量）：(x, y) 在旋转下变换，权重必须满足特殊结构
- degree k（高阶）：cos(kθ), sin(kθ) 对

等变线性层的设计：
- degree 0 ↔ degree 0：普通 Linear
- degree 1 ↔ degree 1：块对角结构 [[a, -b], [b, a]]
- degree 0 → degree 1：零（标量不能生成向量，除非有常数偏移）
- degree 1 → degree 0：可以（向量内积得到标量）

参考：
- Cohen & Welling (2016) "Group Equivariant Convolutional Networks"
- Weiler et al. (2018) "3D Steerable CNNs"
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# SO(2) 等变线性层
# ══════════════════════════════════════════════════════════════════════════════

class SO2EquivariantLinear(nn.Module):
    """SO(2) 等变线性层

    对于 degree-1 特征（二维向量表示），权重矩阵必须满足：
    W @ R(θ) = R(θ) @ W

    这要求 W 具有形式：[[a, -b], [b, a]]（2x2 块）

    对于 degree-0 特征，权重可以是任意的。

    参数化：
    - 对于每个 2D 通道对，我们只需要两个参数 (a, b)
    - 而不是四个参数 (w00, w01, w10, w11)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        degree_in: int = 1,
        degree_out: int = 1,
        bias: bool = False,
    ):
        """
        Args:
            d_in: 输入通道数
            d_out: 输出通道数
            degree_in: 输入 irrep 的阶数（0 或 1）
            degree_out: 输出 irrep 的阶数（0 或 1）
            bias: 是否加偏移
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.degree_in = degree_in
        self.degree_out = degree_out

        if degree_in == 0 and degree_out == 0:
            # degree 0 → degree 0：普通 Linear
            self.weight = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
            self._forward = self._forward_00

        elif degree_in == 1 and degree_out == 1:
            # degree 1 → degree 1：等变结构
            # 输入形状：(B, L, d_in, 2)，输出形状：(B, L, d_out, 2)
            # 我们需要 d_out × d_in 个 2x2 等变块
            # 每个 2x2 块有 2 个参数 (a, b)

            # 参数化：real 和 imag 部分
            self.weight_real = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
            self.weight_imag = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
            self._forward = self._forward_11

        elif degree_in == 1 and degree_out == 0:
            # degree 1 → degree 0：向量到标量（严格旋转不变）
            # 等变约束：输入旋转 → 输出不变
            # 数学上唯一合法的方式：对每个 (x_i, y_i) 取 L2 范数 sqrt(x^2+y^2)
            # 得到 d_in 个不变量，再用 d_in → d_out 的普通 Linear 投影
            # （注意：sqrt 后加 bias 会破坏不变性，所以 bias=False）
            self.weight = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
            self._forward = self._forward_10

        elif degree_in == 0 and degree_out == 1:
            # degree 0 → degree 1：标量到向量
            # 标量不能生成向量（除非有外部方向信息）
            # 这里我们允许，但初始化为小值
            self.weight = nn.Parameter(torch.randn(d_out, d_in, 2) / math.sqrt(d_in))
            self._forward = self._forward_01

        else:
            raise NotImplementedError(f"degree {degree_in} -> {degree_out} not supported")

        if bias:
            if degree_out == 0:
                self.bias = nn.Parameter(torch.zeros(d_out))
            else:
                self.bias = nn.Parameter(torch.zeros(d_out, 2))
        else:
            self.register_parameter('bias', None)

    def _forward_00(self, x: torch.Tensor) -> torch.Tensor:
        """degree 0 → degree 0"""
        # x: (B, L, d_in)
        return F.linear(x, self.weight, self.bias)

    def _forward_11(self, x: torch.Tensor) -> torch.Tensor:
        """degree 1 → degree 1（核心等变层）

        输入 x: (B, L, d_in, 2)
        输出: (B, L, d_out, 2)

        等变矩阵：
        W = [[a, -b], [b, a]]

        等价于复数乘法：(a + ib) * (x0 + i*x1) = (a*x0 - b*x1) + i*(b*x0 + a*x1)
        """
        # x: (B, L, d_in, 2)
        x0, x1 = x[..., 0], x[..., 1]  # (B, L, d_in)

        # 复数乘法
        # out0 = real * x0 - imag * x1
        # out1 = real * x1 + imag * x0

        out0 = F.linear(x0, self.weight_real, None) - F.linear(x1, self.weight_imag, None)
        out1 = F.linear(x1, self.weight_real, None) + F.linear(x0, self.weight_imag, None)

        out = torch.stack([out0, out1], dim=-1)  # (B, L, d_out, 2)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)

        return out

    def _forward_10(self, x: torch.Tensor) -> torch.Tensor:
        """degree 1 → degree 0：严格旋转不变。

        输入 x: (B, L, d_in, 2)
        输出: (B, L, d_out)

        数学：先对每个 channel 取 L2 范数 sqrt(x_i^2 + y_i^2)，
        得到 d_in 个旋转不变量，再用普通 Linear(d_in → d_out) 投影。
        """
        norms = torch.norm(x, p=2, dim=-1)  # (B, L, d_in) 旋转不变
        return F.linear(norms, self.weight, self.bias)

    def _forward_01(self, x: torch.Tensor) -> torch.Tensor:
        """degree 0 → degree 1

        输入 x: (B, L, d_in)
        输出: (B, L, d_out, 2)

        标量生成向量：不满足严格等变性
        但如果输入已经是某种"方向编码"，可以接受
        """
        # self.weight: (d_out, d_in, 2)
        out = torch.einsum('...i,oi...->...o', x, self.weight)  # (B, L, d_out, 2)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


# ══════════════════════════════════════════════════════════════════════════════
# SO(2) 等变 MLP
# ══════════════════════════════════════════════════════════════════════════════

class SO2EquivariantMLP(nn.Module):
    """SO(2) 等变 MLP

    支持两种模式：
    1. 纯 degree-1 特征流：每层都是等变 Linear + GELU（等变）
    2. 混合 degree-0 和 degree-1：分别处理，degree-0 可以非线性
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        degree: int = 1,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_in: 输入通道数
            d_hidden: 隐藏层通道数
            d_out: 输出通道数
            degree: irrep 阶数（0 或 1）
            n_layers: 层数
            dropout: dropout 概率
        """
        super().__init__()
        self.degree = degree

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            d_prev = d_in if i == 0 else d_hidden
            d_next = d_out if i == n_layers - 1 else d_hidden

            self.layers.append(SO2EquivariantLinear(
                d_prev, d_next, degree, degree, bias=False
            ))

            if degree == 0:
                # degree-0 可以用 LayerNorm
                self.norms.append(nn.LayerNorm(d_next))
            else:
                # degree-1 用 GroupNorm（对通道归一化）
                self.norms.append(nn.GroupNorm(min(8, d_next // 2), d_next))

        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        degree 0: x: (B, L, d)
        degree 1: x: (B, L, d, 2)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)

            if self.degree == 0:
                # degree-0：LayerNorm + GELU
                x = norm(x)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
            else:
                # degree-1：GroupNorm（需要 reshape）
                B, L, d, _ = x.shape
                x_reshape = x.permute(0, 3, 1, 2).reshape(B * 2, d, L)  # (B*2, d, L)
                x_reshape = norm(x_reshape)
                x = x_reshape.reshape(B, 2, d, L).permute(0, 2, 3, 1)  # (B, d, L, 2)

                # 注意：degree-1 不加非线性（GELU 会破坏等变性）
                # 只有最后一层之后可以考虑加

            x = self.dropout(x)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# 混合 irrep 表示
# ══════════════════════════════════════════════════════════════════════════════

class SO2MixedProjection(nn.Module):
    """混合 degree-0 和 degree-1 的投影层

    输入：分开的 degree-0 和 degree-1 特征
    输出：分开的 degree-0 和 degree-1 特征

    规则：
    - degree-0 可以从 degree-0 和 degree-1 输入
    - degree-1 只能从 degree-0 和 degree-1 输入（但不能混合）
    """

    def __init__(
        self,
        d_inv_in: int,
        d_eq_in: int,
        d_inv_out: int,
        d_eq_out: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_inv_in: 输入 degree-0 通道数
            d_eq_in: 输入 degree-1 通道数
            d_inv_out: 输出 degree-0 通道数
            d_eq_out: 输出 degree-1 通道数
        """
        super().__init__()

        # degree-0 输出：可以从 degree-0 和 degree-1 输入
        self.inv_from_inv = nn.Linear(d_inv_in, d_inv_out, bias=False)
        self.inv_from_eq = SO2EquivariantLinear(d_eq_in, d_inv_out, degree_in=1, degree_out=0, bias=False)

        # degree-1 输出：只能从单一来源
        # 选择：优先从 degree-1 输入（保持等变），退化时从 degree-0
        if d_eq_in > 0:
            self.eq_from_eq = SO2EquivariantLinear(d_eq_in, d_eq_out, degree_in=1, degree_out=1, bias=False)
            self.eq_from_inv = None  # 不从 degree-0 生成 degree-1
        else:
            self.eq_from_eq = None
            self.eq_from_inv = SO2EquivariantLinear(d_inv_in, d_eq_out, degree_in=0, degree_out=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_repr_inv: torch.Tensor,
        node_repr_eq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_repr_inv: (B, L, d_inv_in) - degree-0
            node_repr_eq: (B, L, d_eq_in, 2) - degree-1

        Returns:
            node_repr_inv_out: (B, L, d_inv_out)
            node_repr_eq_out: (B, L, d_eq_out, 2)
        """
        # degree-0 输出
        inv_out = self.inv_from_inv(node_repr_inv)
        if node_repr_eq.shape[-1] == 2:  # 有 degree-1 输入
            inv_out = inv_out + self.inv_from_eq(node_repr_eq)

        # degree-1 输出
        if self.eq_from_eq is not None:
            eq_out = self.eq_from_eq(node_repr_eq)
        elif self.eq_from_inv is not None:
            eq_out = self.eq_from_inv(node_repr_inv)
        else:
            eq_out = torch.zeros(*node_repr_inv.shape[:2], self.d_eq_out, 2, device=node_repr_inv.device)

        return self.dropout(inv_out), self.dropout(eq_out)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """生成 2D 旋转矩阵

    Args:
        theta: (B,) 或标量，旋转角度

    Returns:
        R: (B, 2, 2) 或 (2, 2)
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    if theta.dim() == 0:
        return torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t]),
        ])

    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1),
    ], dim=-2)  # (B, 2, 2)

    return R


def apply_rotation_to_degree1(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """对 degree-1 特征应用旋转

    Args:
        x: (B, L, d, 2) - degree-1 特征
        R: (B, 2, 2) 或 (2, 2) - 旋转矩阵

    Returns:
        x_rot: (B, L, d, 2)
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)  # (1, 2, 2)

    # x @ R.T
    return torch.einsum('bldi,bji->bldj', x, R)


def check_equivariance(
    layer: nn.Module,
    degree_in: int = 1,
    B: int = 2,
    L: int = 10,
    d_in: int = 32,
    n_tests: int = 10,
    threshold: float = 1e-5,
) -> Tuple[float, bool]:
    """检查层的等变性

    Returns:
        max_error: 最大误差
        is_equivariant: 是否等变（error < threshold）
    """
    layer.eval()

    errors = []

    for _ in range(n_tests):
        theta = torch.rand(1).item() * 2 * math.pi
        R = rotation_matrix_2d(torch.tensor(theta))

        if degree_in == 1:
            x = torch.randn(B, L, d_in, 2)
            x_rot = apply_rotation_to_degree1(x, R)
        else:
            x = torch.randn(B, L, d_in)
            x_rot = x.clone()  # degree-0 不变

        with torch.no_grad():
            out_a = layer(x_rot)
            out_b = layer(x)

            if out_a.shape[-1] == 2:  # degree-1 输出
                out_b = apply_rotation_to_degree1(out_b, R)

        error = (out_a - out_b).abs().max().item()
        errors.append(error)

    max_error = max(errors)
    return max_error, max_error < threshold


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SO(2) Equivariant Modules Test")
    print("=" * 60)

    # 测试 degree-1 → degree-1
    print("\n1. Degree-1 → Degree-1 等变线性层")
    layer_11 = SO2EquivariantLinear(32, 64, degree_in=1, degree_out=1)

    max_err, is_eq = check_equivariance(layer_11, degree_in=1, B=2, L=50, d_in=32)
    print(f"   最大误差: {max_err:.2e}")
    print(f"   等变性: {'PASS' if is_eq else 'FAIL'}")

    # 测试 degree-0 → degree-0
    print("\n2. Degree-0 → Degree-0 等变线性层")
    layer_00 = SO2EquivariantLinear(32, 64, degree_in=0, degree_out=0)

    max_err, is_eq = check_equivariance(layer_00, degree_in=0, B=2, L=50, d_in=32)
    print(f"   最大误差: {max_err:.2e}")
    print(f"   等变性: {'PASS' if is_eq else 'FAIL'}")

    # 测试 degree-1 → degree-0
    print("\n3. Degree-1 → Degree-0 等变线性层")
    layer_10 = SO2EquivariantLinear(32, 64, degree_in=1, degree_out=0)

    max_err, is_eq = check_equivariance(layer_10, degree_in=1, B=2, L=50, d_in=32)
    print(f"   最大误差: {max_err:.2e}")
    print(f"   等变性: {'PASS' if is_eq else 'FAIL'}")

    # 测试混合投影
    print("\n4. 混合 irrep 投影")
    mixed_proj = SO2MixedProjection(32, 32, 64, 64)

    x_inv = torch.randn(2, 50, 32)
    x_eq = torch.randn(2, 50, 32, 2)

    theta = torch.rand(1).item() * 2 * math.pi
    R = rotation_matrix_2d(torch.tensor(theta))

    x_eq_rot = apply_rotation_to_degree1(x_eq, R)

    with torch.no_grad():
        inv_a, eq_a = mixed_proj(x_inv, x_eq_rot)
        inv_b, eq_b = mixed_proj(x_inv, x_eq)
        eq_b = apply_rotation_to_degree1(eq_b, R)

    inv_err = (inv_a - inv_b).abs().max().item()
    eq_err = (eq_a - eq_b).abs().max().item()

    print(f"   inv 误差: {inv_err:.2e}")
    print(f"   eq 误差: {eq_err:.2e}")
    print(f"   等变性: {'PASS' if max(inv_err, eq_err) < 1e-5 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)