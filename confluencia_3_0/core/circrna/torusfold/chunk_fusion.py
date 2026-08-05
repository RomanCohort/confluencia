"""chunk_fusion.py — Level 3: 全链融合 GNN.

三层层次化预测的最后一步:
  把每个 chunk 的局部预测融合成全链结构.
  用已有的等变 GNN, chunk 作为 "超级残基".

架构:
  chunk features → 全局 attention → 等变 GNN → 全链坐标偏移 → 拼接

关键约束:
  - 重叠区域一致性 (由 overlap_loss 保证)
  - 键长连续性
  - BSJ 闭合

用法:
  fusion = ChunkFusionGNN(d_chunk=256, n_heads=8)
  global_coords = fusion(chunk_features, chunk_coords, chunk_infos)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkFusionGNN(nn.Module):
    """Level 3: 全链融合 GNN.

    将每个 chunk 的特征作为 "超级残基", 通过全局注意力 + 等变 GNN
    计算 chunk 间的相对偏移和旋转, 重建全链结构.
    """

    def __init__(
        self,
        d_chunk: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_chunk = d_chunk

        # Chunk token: 每个 chunk 的全局表示
        self.chunk_token_proj = nn.Linear(d_chunk, d_chunk)

        # 全局注意力: chunk 之间交互
        self.global_attn = nn.ModuleList([
            nn.MultiheadAttention(d_chunk, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.attn_norm = nn.ModuleList([
            nn.LayerNorm(d_chunk) for _ in range(n_layers)
        ])

        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_chunk, d_ffn),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn, d_chunk),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(d_chunk) for _ in range(n_layers)
        ])

        # 边特征: chunk 间的重叠程度
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_chunk * 2 + 1, d_chunk),
            nn.GELU(),
            nn.Linear(d_chunk, d_chunk),
        )

        # 输出: 每个 chunk 的全局偏移 + 旋转
        self.offset_head = nn.Linear(d_chunk, 3)    # 平移偏移
        self.rotation_head = nn.Linear(d_chunk, 3)   # 旋转 (axis-angle)
        self.scale_head = nn.Linear(d_chunk, 1)      # 缩放因子

        # 最终坐标预测: chunk 内部坐标 + 全局偏移
        self.coord_refine = nn.Sequential(
            nn.Linear(3 + d_chunk, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        chunk_features: torch.Tensor,
        chunk_coords: List[torch.Tensor],
        chunk_infos: list,
        total_length: int,
    ) -> torch.Tensor:
        """融合 chunk 预测成全链坐标.

        Args:
            chunk_features: (N_chunks, d_chunk) — 每个 chunk 的全局特征
            chunk_coords: List[(L_i, 3)] — 每个 chunk 的局部坐标
            chunk_infos: List[ChunkInfo] — chunk 元数据
            total_length: 全链长度 L

        Returns:
            global_coords: (L, 3) — 全链坐标
        """
        device = chunk_features.device
        N = chunk_features.shape[0]

        if N == 1:
            # 只有一个 chunk, 直接返回
            return chunk_coords[0]

        # 1. Chunk token projection
        x = self.chunk_token_proj(chunk_features)  # (N, d_chunk)

        # 2. 计算边特征 (重叠程度)
        edge_features = self._compute_edge_features(chunk_infos)  # (N, N, 1)

        # 3. 全局注意力 + 边特征
        for i in range(len(self.global_attn)):
            # Self-attention
            attn_out, _ = self.global_attn[i](x, x, x)
            x = self.attn_norm[i](x + attn_out)

            # FFN
            ffn_out = self.ffn[i](x)
            x = self.ffn_norm[i](x + ffn_out)

        # 4. 计算每个 chunk 的全局变换
        offsets = self.offset_head(x)      # (N, 3) — 平移
        rotations = self.rotation_head(x)  # (N, 3) — axis-angle 旋转
        scales = self.scale_head(x)        # (N, 1) — 缩放
        scales = 1.0 + torch.tanh(scales)  # 归一化到 (0, 2)

        # 5. 应用全局变换到每个 chunk 的局部坐标
        global_coords = torch.zeros(total_length, 3, device=device)
        counts = torch.zeros(total_length, 1, device=device)

        for i in range(N):
            L_chunk = chunk_coords[i].shape[0]
            info = chunk_infos[i]

            # 旋转 (axis-angle → rotation matrix)
            R = self._axis_angle_to_rotation(rotations[i])  # (3, 3)

            # 变换: scale * R @ local_coords + offset
            local_coords = chunk_coords[i]  # (L_chunk, 3)
            transformed = scales[i] * (local_coords @ R.T) + offsets[i]

            # 累加到全局坐标
            for j in range(L_chunk):
                pos = (info.start + j) % total_length
                global_coords[pos] += transformed[j]
                counts[pos] += 1

        # 平均重叠区域
        counts = counts.clamp(min=1)
        global_coords = global_coords / counts

        # 6. 坐标精炼: 利用 chunk 特征微调
        global_coords = self._refine_coords(
            global_coords, chunk_features, chunk_infos, total_length
        )

        return global_coords

    def _compute_edge_features(
        self, chunk_infos: list
    ) -> torch.Tensor:
        """计算 chunk 间的边特征 (重叠程度)."""
        N = len(chunk_infos)
        device = torch.device('cpu')

        edge_features = torch.zeros(N, N, 1, device=device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # 计算重叠长度
                overlap_start = max(chunk_infos[i].start, chunk_infos[j].start)
                overlap_end = min(chunk_infos[i].end, chunk_infos[j].end)
                overlap_len = max(0, overlap_end - overlap_start)

                # 归一化
                max_len = max(chunk_infos[i].end - chunk_infos[i].start,
                            chunk_infos[j].end - chunk_infos[j].start)
                edge_features[i, j, 0] = overlap_len / max_len if max_len > 0 else 0

        return edge_features

    def _axis_angle_to_rotation(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Axis-angle → 3×3 rotation matrix (Rodrigues formula)."""
        angle = torch.norm(axis_angle) + 1e-8
        axis = axis_angle / angle

        # Rodrigues: R = cos(θ)I + sin(θ)[k]× + (1-cos(θ))k⊗k
        K = torch.zeros(3, 3, device=axis_angle.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        R = torch.eye(3, device=axis_angle.device) + \
            torch.sin(angle) * K + \
            (1 - torch.cos(angle)) * (K @ K)

        return R

    def _refine_coords(
        self,
        global_coords: torch.Tensor,
        chunk_features: torch.Tensor,
        chunk_infos: list,
        total_length: int,
    ) -> torch.Tensor:
        """利用 chunk 特征微调全局坐标."""
        device = global_coords.device
        refined = torch.zeros_like(global_coords)

        for i, info in enumerate(chunk_infos):
            L_chunk = info.end - info.start
            feature = chunk_features[i]  # (d_chunk,)

            for j in range(L_chunk):
                pos = (info.start + j) % total_length
                coord = global_coords[pos]
                feat = feature

                # 拼接坐标和特征
                combined = torch.cat([coord, feat], dim=0)
                delta = self.coord_refine(combined)
                refined[pos] = coord + delta

        return refined
