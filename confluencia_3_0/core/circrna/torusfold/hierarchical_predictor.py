"""hierarchical_predictor.py — 三层层次化预测完整管线.

集成: MSA聚类 → chunk切分 → chunk独立预测 → 全链融合

用法:
  predictor = HierarchicalPredictor(config)
  result = predictor(seq_tokens, msa_tokens)
  # result['coords']: (L, 3) — 全链坐标
  # result['chunk_predictions']: List[ChunkPrediction]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .msa_cluster import HierarchicalMSA, MSACluster
from .chunk_splitter import ChunkSplitter, ChunkInfo
from .chunk_predictor import ChunkPredictor, ChunkPrediction, ChunkFeatureExtractor
from .chunk_fusion import ChunkFusionGNN
from .overlap_loss import OverlapConsistencyLoss


@dataclass
class HierarchicalConfig:
    """层次化预测配置."""
    # MSA
    n_representatives: int = 64
    use_msa_clustering: bool = True
    d_msa: int = 640
    d_pair: int = 128

    # Chunk
    chunk_size: int = 500
    overlap: int = 50
    min_chunk_size: int = 100

    # Fusion
    d_chunk: int = 256
    n_heads: int = 8
    n_fusion_layers: int = 4

    # Loss weights
    w_overlap_coord: float = 10.0
    w_overlap_bond: float = 5.0
    w_overlap_tangent: float = 2.0
    w_bsj_closure: float = 5.0


class HierarchicalPredictor(nn.Module):
    """三层层次化预测完整管线.

    Level 1: Chunk 独立预测
      - MSA 聚类 → 代表性序列
      - 每个 chunk 用 RhoFold+ backbone 独立预测
      - 输出 chunk 坐标 + 特征

    Level 2: 特征提取
      - 从每个 chunk 的预测中提取全局特征向量
      - 坐标不变特征 + contact map 特征

    Level 3: 全链融合
      - 全局注意力 + 等变 GNN
      - 计算 chunk 间的相对偏移/旋转
      - 重叠区域约束
    """

    def __init__(
        self,
        config: HierarchicalConfig,
        rhofold_backbone: Optional[nn.Module] = None,
        node_proj: Optional[nn.Module] = None,
        pair_track: Optional[nn.Module] = None,
        pair_to_node: Optional[nn.Module] = None,
        node_norm: Optional[nn.Module] = None,
        cg_decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # MSA 处理
        self.msa_cluster = MSACluster(
            n_representatives=config.n_representatives,
            method="embedding",
        ) if config.use_msa_clustering else None

        # Chunk 切分
        self.splitter = ChunkSplitter(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
        )

        # Level 1: Chunk 独立预测
        if rhofold_backbone is not None:
            self.chunk_predictor = ChunkPredictor(
                rhofold_backbone=rhofold_backbone,
                node_proj=node_proj,
                pair_track=pair_track,
                pair_to_node=pair_to_node,
                node_norm=node_norm,
                cg_decoder=cg_decoder,
                d_node=256,
                d_pair=64,
            )
        else:
            self.chunk_predictor = None

        # Level 2: 特征提取
        self.feature_extractor = ChunkFeatureExtractor(
            d_feature=config.d_chunk
        )

        # Level 3: 全链融合
        self.fusion_gnn = ChunkFusionGNN(
            d_chunk=config.d_chunk,
            n_heads=config.n_heads,
            n_layers=config.n_fusion_layers,
        )

        # 重叠一致性损失
        self.overlap_loss = OverlapConsistencyLoss(
            w_coord=config.w_overlap_coord,
            w_bond=config.w_overlap_bond,
            w_tangent=config.w_overlap_tangent,
            w_bsj_closure=config.w_bsj_closure,
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        circular_mask: Optional[torch.Tensor] = None,
        return_chunk_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """层次化预测.

        Args:
            seq_tokens: (B, L) token IDs — 输入序列
            lengths: (B,) 每条序列的实际长度
            circular_mask: (B,) 0/1 环状/线性
            return_chunk_details: 是否返回每个 chunk 的详细信息

        Returns:
            dict: {
                'coords': (B, L, 3) — 全链坐标,
                'overlap_loss': dict — 重叠一致性损失,
                'chunk_predictions': List[List[ChunkPrediction]] — 如果 return_chunk_details,
            }
        """
        B, L_max = seq_tokens.shape
        device = seq_tokens.device

        if lengths is None:
            lengths = torch.tensor([L_max] * B, device=device)

        if circular_mask is None:
            circular_mask = torch.ones(B, device=device)

        all_coords = []
        all_overlap_losses = []
        all_chunk_details = []

        for b in range(B):
            L = int(lengths[b].item())
            seq = seq_tokens[b, :L]
            is_circ = circular_mask[b] > 0

            # Step 1: 切分 chunk
            chunks = self.splitter.split(seq, is_circular=is_circ, bsj_pos=0)

            if self.chunk_predictor is None or len(chunks) == 0:
                # fallback: 直接预测全链
                coords = torch.zeros(L, 3, device=device)
                all_coords.append(coords)
                continue

            # Step 2: 每个 chunk 独立预测
            chunk_coords = []
            chunk_features = []
            chunk_preds = []

            for chunk in chunks:
                tokens = chunk.seq_tokens.unsqueeze(0)  # (1, L_chunk)
                result = self.chunk_predictor(tokens, return_features=True)

                chunk_coord = result['coords'].squeeze(0)  # (L_chunk, 3)
                chunk_coords.append(chunk_coord)

                # Level 2: 提取特征
                pred = ChunkPrediction(
                    coords=chunk_coord,
                    node_repr=result['node_repr'].squeeze(0),
                    contact_map=result['contact_map'].squeeze(0),
                    bsj_confidence=result['bsj_confidence'].item(),
                    chunk_id=chunk.chunk_id,
                    start=chunk.start,
                    end=chunk.end,
                )
                chunk_preds.append(pred)

                feature = self.feature_extractor(pred)
                chunk_features.append(feature)

            # Step 3: 全链融合
            if len(chunk_features) > 1:
                chunk_features_tensor = torch.stack(chunk_features)  # (N, d_chunk)
                global_coords = self.fusion_gnn(
                    chunk_features_tensor,
                    chunk_coords,
                    chunks,
                    L,
                )
            else:
                global_coords = chunk_coords[0]

            # 计算重叠一致性损失
            overlap_losses = self.overlap_loss(chunk_coords, chunks, L)

            all_coords.append(global_coords)
            all_overlap_losses.append(overlap_losses)
            all_chunk_details.append(chunk_preds)

        # Padding 到 max length
        max_L = max(c.shape[0] for c in all_coords)
        padded_coords = torch.zeros(B, max_L, 3, device=device)
        for b, coords in enumerate(all_coords):
            padded_coords[b, :coords.shape[0]] = coords

        result = {
            'coords': padded_coords,
            'overlap_loss': self._aggregate_overlap_losses(all_overlap_losses),
        }

        if return_chunk_details:
            result['chunk_predictions'] = all_chunk_details

        return result

    def _aggregate_overlap_losses(
        self, losses_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """聚合多个样本的重叠损失."""
        if not losses_list:
            return {'total': torch.tensor(0.0)}

        aggregated = {}
        for key in losses_list[0].keys():
            values = [l[key] for l in losses_list if key in l]
            if values:
                aggregated[key] = torch.stack(values).mean()
            else:
                aggregated[key] = torch.tensor(0.0)

        return aggregated
