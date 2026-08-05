"""overlap_loss.py — 重叠区域一致性损失.

三层层次化预测的核心约束:
  相邻 chunk 的重叠区域必须几何一致.

损失项:
  1. coord_loss: 重叠区域坐标 MSE (Kabsch 对齐后)
  2. bond_loss: 重叠区域键长连续性
  3. tangent_loss: 切线方向平滑
  4. bsj_closure: BSJ 闭合约束

用法:
  overlap_criterion = OverlapConsistencyLoss()
  loss_dict = overlap_criterion(chunks_coords, chunk_infos, total_length)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapConsistencyLoss(nn.Module):
    """重叠区域一致性损失."""

    def __init__(
        self,
        w_coord: float = 10.0,
        w_bond: float = 5.0,
        w_tangent: float = 2.0,
        w_bsj_closure: float = 5.0,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.w_coord = w_coord
        self.w_bond = w_bond
        self.w_tangent = w_tangent
        self.w_bsj_closure = w_bsj_closure
        self.bond_length = bond_length

    def forward(
        self,
        chunks_coords: List[torch.Tensor],
        chunk_infos: list,
        total_length: int,
    ) -> Dict[str, torch.Tensor]:
        """计算重叠一致性损失.

        Args:
            chunks_coords: List[(chunk_len, 3)] — 每个 chunk 的预测坐标
            chunk_infos: List[ChunkInfo] — chunk 元数据 (start, end, overlap)
            total_length: 全链长度 L

        Returns:
            dict: {
                'total': total_loss,
                'coord': coord_mse,
                'bond': bond_violation,
                'tangent': tangent_loss,
                'bsj_closure': closure_dist,
            }
        """
        device = chunks_coords[0].device
        loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # 1. 相邻 chunk 重叠区域坐标一致性
        coord_losses = []
        for i in range(len(chunks_coords) - 1):
            c1 = chunks_coords[i]      # (L1, 3)
            c2 = chunks_coords[i + 1]  # (L2, 3)
            info1 = chunk_infos[i]
            info2 = chunk_infos[i + 1]

            # 计算重叠区域
            overlap_start = max(info1.start, info2.start)
            overlap_end = min(info1.end, info2.end)

            if overlap_start >= overlap_end:
                continue  # 无重叠

            # 转换为 chunk 内部索引
            local_start_1 = overlap_start - info1.start
            local_end_1 = overlap_end - info1.start
            local_start_2 = overlap_start - info2.start
            local_end_2 = overlap_end - info2.start

            # 提取重叠区域坐标
            overlap_1 = c1[local_start_1:local_end_1]  # (overlap_len, 3)
            overlap_2 = c2[local_start_2:local_end_2]  # (overlap_len, 3)

            # Kabsch 对齐后 MSE
            aligned_2 = self._kabsch_align(overlap_2, overlap_1)
            mse = F.mse_loss(overlap_1, aligned_2)
            coord_losses.append(mse)

        if coord_losses:
            coord_loss = torch.stack(coord_losses).mean()
            loss_dict['coord'] = coord_loss
            loss = loss + self.w_coord * coord_loss
        else:
            loss_dict['coord'] = torch.tensor(0.0, device=device)

        # 2. 键长连续性
        bond_losses = []
        for i in range(len(chunks_coords) - 1):
            c1 = chunks_coords[i]
            c2 = chunks_coords[i + 1]
            info1 = chunk_infos[i]
            info2 = chunk_infos[i + 1]

            # 重叠区域边界处的键长
            if info1.end > info1.start and info2.start < info2.end:
                # chunk i 的最后一个残基
                pos_i = info1.end - info1.start - 1
                if pos_i < c1.shape[0]:
                    last_i = c1[pos_i]

                    # chunk i+1 的第一个残基
                    pos_j = 0
                    if pos_j < c2.shape[0]:
                        first_j = c2[pos_j]

                        # 键长
                        bond_dist = torch.norm(last_i - first_j)
                        bond_loss = (bond_dist - self.bond_length) ** 2
                        bond_losses.append(bond_loss)

        if bond_losses:
            bond_loss = torch.stack(bond_losses).mean()
            loss_dict['bond'] = bond_loss
            loss = loss + self.w_bond * bond_loss
        else:
            loss_dict['bond'] = torch.tensor(0.0, device=device)

        # 3. 切线方向平滑
        tangent_losses = []
        for i in range(len(chunks_coords) - 1):
            c1 = chunks_coords[i]
            c2 = chunks_coords[i + 1]
            info1 = chunk_infos[i]
            info2 = chunk_infos[i + 1]

            # 重叠区域内, 取中间点计算切线
            overlap_start = max(info1.start, info2.start)
            overlap_end = min(info1.end, info2.end)

            if overlap_end - overlap_start < 3:
                continue

            # chunk i 的切线 (重叠区域中间)
            mid_1 = (overlap_start + overlap_end) // 2 - info1.start
            if 1 <= mid_1 < c1.shape[0] - 1:
                tangent_1 = c1[mid_1 + 1] - c1[mid_1 - 1]
                tangent_1 = F.normalize(tangent_1, dim=-1)

                # chunk i+1 的切线
                mid_2 = (overlap_start + overlap_end) // 2 - info2.start
                if 1 <= mid_2 < c2.shape[0] - 1:
                    tangent_2 = c2[mid_2 + 1] - c2[mid_2 - 1]
                    tangent_2 = F.normalize(tangent_2, dim=-1)

                    # 切线一致性: 1 - cos(angle)
                    cos_sim = (tangent_1 * tangent_2).sum()
                    tangent_loss = 1.0 - cos_sim
                    tangent_losses.append(tangent_loss)

        if tangent_losses:
            tangent_loss = torch.stack(tangent_losses).mean()
            loss_dict['tangent'] = tangent_loss
            loss = loss + self.w_tangent * tangent_loss
        else:
            loss_dict['tangent'] = torch.tensor(0.0, device=device)

        # 4. BSJ 闭合约束
        if chunks_coords:
            first_pos = chunk_infos[0].start
            last_pos = chunk_infos[-1].end - 1

            # 获取首尾坐标
            first_coord = chunks_coords[0][0] if chunks_coords[0].shape[0] > 0 else None
            last_coord = chunks_coords[-1][-1] if chunks_coords[-1].shape[0] > 0 else None

            if first_coord is not None and last_coord is not None:
                closure_dist = torch.norm(first_coord - last_coord)
                bsj_loss = (closure_dist - self.bond_length) ** 2
                loss_dict['bsj_closure'] = bsj_loss
                loss = loss + self.w_bsj_closure * bsj_loss
            else:
                loss_dict['bsj_closure'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['bsj_closure'] = torch.tensor(0.0, device=device)

        loss_dict['total'] = loss
        return loss_dict

    @staticmethod
    def _kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Kabsch 对齐: 将 P 旋转对齐到 Q.

        Args:
            P: (N, 3) 源坐标
            Q: (N, 3) 目标坐标

        Returns:
            P_aligned: (N, 3) 对齐后的坐标
        """
        # 中心化
        P_center = P - P.mean(dim=0)
        Q_center = Q - Q.mean(dim=0)

        # SVD
        H = P_center.T @ Q_center  # (3, 3)
        U, S, Vt = torch.linalg.svd(H)

        # 旋转矩阵
        R = Vt.T @ U.T

        # 确保右手系
        if torch.det(R) < 0:
            Vt_fixed = Vt.clone()
            Vt_fixed[-1, :] = -Vt_fixed[-1, :]
            R = Vt_fixed.T @ U.T

        # 应用旋转 + 平移
        P_aligned = P_center @ R.T + Q_center.mean(dim=0)

        return P_aligned
