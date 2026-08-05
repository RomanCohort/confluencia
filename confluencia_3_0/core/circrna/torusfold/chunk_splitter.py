"""chunk_splitter.py — 长序列 chunk 切分.

三层层次化预测的第一步:
  将长序列 (L > 500) 切成有重叠的 chunk, 每个 chunk 独立预测局部结构.

切分策略:
  - chunk_size: 500nt (与 Phase 2 数据长度匹配)
  - overlap: 50nt (10% 重叠)
  - BSJ 位置必须是 chunk 边界 (环状 RNA 特殊处理)

用法:
  splitter = ChunkSplitter(chunk_size=500, overlap=50)
  chunks = splitter.split(seq_tokens, is_circular=True, bsj_pos=0)
  # chunks: List[ChunkInfo] — 每个 chunk 的元数据
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class ChunkInfo:
    """一个 chunk 的元数据."""
    chunk_id: int                # chunk 编号
    start: int                   # 起始位置 (包含)
    end: int                     # 结束位置 (不包含)
    seq_tokens: torch.Tensor = None  # (chunk_len,) token IDs
    is_circular_chunk: bool = False   # 是否包含 BSJ junction
    bsj_local_pos: int = -1     # BSJ 在 chunk 内的局部位置
    overlap_left: int = 0       # 左侧重叠长度
    overlap_right: int = 0      # 右侧重叠长度


class ChunkSplitter:
    """长序列 chunk 切分器.

    将 (B, L) 的序列切分成多个 chunk, 每个 chunk 独立预测.
    支持环状 RNA 的 BSJ 闭合约束.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的目标长度
            overlap: 相邻 chunk 的重叠长度
            min_chunk_size: 最小 chunk 长度 (小于此的不分割)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk = min_chunk_size

    def split(
        self,
        seq_tokens: torch.Tensor,
        is_circular: bool = False,
        bsj_pos: int = 0,
    ) -> List[ChunkInfo]:
        """切分序列.

        Args:
            seq_tokens: (L,) token IDs — 单条序列
            is_circular: 是否为环状 RNA
            bsj_pos: BSJ junction 位置 (环状时, 默认 0)

        Returns:
            List[ChunkInfo] — 切分后的 chunk 列表
        """
        L = seq_tokens.shape[0]

        if L <= self.chunk_size:
            # 序列足够短, 不需要切分
            return [ChunkInfo(
                chunk_id=0,
                start=0,
                end=L,
                seq_tokens=seq_tokens,
                is_circular_chunk=is_circular,
                bsj_local_pos=bsj_pos if is_circular else -1,
            )]

        chunks = []
        step = self.chunk_size - self.overlap
        chunk_id = 0

        if is_circular:
            # 环状 RNA: 从 BSJ 位置开始, 顺时针切分
            # BSJ 位置成为第一个 chunk 的起点
            positions = list(range(bsj_pos, L)) + list(range(0, bsj_pos))
            reordered = seq_tokens[positions]

            # 按 chunk_size 切分, 最后一个 chunk 的末尾会回到 BSJ
            for i in range(0, L, step):
                chunk_end = min(i + self.chunk_size, L)
                chunk_tokens = reordered[i:chunk_end]

                # 计算原始位置
                orig_start = positions[i]
                orig_end = positions[min(chunk_end - 1, L - 1)]

                # 是否包含 BSJ junction
                has_bsj = (i <= bsj_pos < chunk_end) or (chunk_end >= L and bsj_pos < i)

                bsj_local = -1
                if has_bsj:
                    # BSJ 在 chunk 内的局部位置
                    bsj_local = (bsj_pos - i) % L

                c = ChunkInfo(
                    chunk_id=chunk_id,
                    start=orig_start,
                    end=orig_end,
                    seq_tokens=chunk_tokens,
                    is_circular_chunk=has_bsj,
                    bsj_local_pos=bsj_local,
                    overlap_left=min(self.overlap, i) if i > 0 else 0,
                    overlap_right=min(self.overlap, L - chunk_end) if chunk_end < L else 0,
                )
                chunks.append(c)
                chunk_id += 1

                if chunk_end >= L:
                    break
        else:
            # 线性 RNA: 简单切分
            for i in range(0, L, step):
                chunk_end = min(i + self.chunk_size, L)
                chunk_tokens = seq_tokens[i:chunk_end]

                c = ChunkInfo(
                    chunk_id=chunk_id,
                    start=i,
                    end=chunk_end,
                    seq_tokens=chunk_tokens,
                    is_circular_chunk=False,
                    bsj_local_pos=-1,
                    overlap_left=min(self.overlap, i) if i > 0 else 0,
                    overlap_right=min(self.overlap, L - chunk_end) if chunk_end < L else 0,
                )
                chunks.append(c)
                chunk_id += 1

                if chunk_end >= L:
                    break

        return chunks

    def split_batch(
        self,
        seq_tokens: torch.Tensor,
        lengths: torch.Tensor,
        circular_mask: Optional[torch.Tensor] = None,
    ) -> List[List[ChunkInfo]]:
        """批量切分.

        Args:
            seq_tokens: (B, L_max) token IDs (padded)
            lengths: (B,) 每条序列的实际长度
            circular_mask: (B,) 0/1 环状/线性

        Returns:
            List[List[ChunkInfo]] — 每个样本的 chunk 列表
        """
        B = seq_tokens.shape[0]
        all_chunks = []

        for b in range(B):
            L = int(lengths[b].item())
            seq = seq_tokens[b, :L]
            is_circ = circular_mask is not None and circular_mask[b] > 0

            chunks = self.split(seq, is_circular=is_circ, bsj_pos=0)
            all_chunks.append(chunks)

        return all_chunks

    @staticmethod
    def stitch_chunks(
        chunks_coords: List[torch.Tensor],
        chunk_infos: List[ChunkInfo],
        total_length: int,
    ) -> torch.Tensor:
        """将 chunk 坐标拼接成全链坐标.

        重叠区域取平均值.

        Args:
            chunks_coords: List[(chunk_len, 3)] — 每个 chunk 的预测坐标
            chunk_infos: List[ChunkInfo] — chunk 元数据
            total_length: 全链长度 L

        Returns:
            coords: (L, 3) — 全链坐标
        """
        device = chunks_coords[0].device
        coords = torch.zeros(total_length, 3, device=device)
        counts = torch.zeros(total_length, 1, device=device)

        for chunk_coords, info in zip(chunks_coords, chunk_infos):
            L_chunk = chunk_coords.shape[0]

            # 简单映射到全链位置
            # 对于环状, 需要根据 start 位置映射
            for i in range(L_chunk):
                pos = (info.start + i) % total_length
                coords[pos] += chunk_coords[i]
                counts[pos] += 1

        # 平均重叠区域
        counts = counts.clamp(min=1)
        coords = coords / counts

        return coords
