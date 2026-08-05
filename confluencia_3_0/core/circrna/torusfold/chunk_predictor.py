"""chunk_predictor.py — Level 1: chunk 独立预测.

三层层次化预测的核心:
  每个 chunk 用 RhoFold+ backbone + S10 等变层独立预测局部结构.
  输出 chunk 坐标 + 特征, 供 Level 3 融合使用.

架构:
  chunk seq → RhoFold+ RNA FM → PairTrack → CG Decoder → chunk coords
  同时输出: node_repr, contact_map (供 Level 3 融合)

用法:
  predictor = ChunkPredictor(rhofold_backbone, s10_encoder, cg_decoder)
  result = predictor(chunk_tokens)
  # result['coords']: (chunk_len, 3)
  # result['node_repr']: (chunk_len, d_model)
  # result['contact_map']: (chunk_len, chunk_len)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChunkPrediction:
    """单个 chunk 的预测结果."""
    coords: torch.Tensor          # (L_chunk, 3) — 局部 3D 坐标
    node_repr: torch.Tensor       # (L_chunk, d_model) — 等变 GNN 特征
    contact_map: torch.Tensor     # (L_chunk, L_chunk) — 接触图
    bsj_confidence: float = 0.0   # BSJ 闭合置信度
    chunk_id: int = 0
    start: int = 0                # 全链中的起始位置
    end: int = 0                  # 全链中的结束位置


class ChunkPredictor(nn.Module):
    """Level 1: chunk 独立预测器.

    复用已有的 RhoFold+ backbone 和 S10 组件, 但:
    1. 接受 chunk (500nt) 作为输入
    2. 输出 chunk 坐标 + 融合特征
    3. 不做全局计算 (O(chunk²) 而非 O(L²))
    """

    def __init__(
        self,
        rhofold_backbone: nn.Module,
        node_proj: nn.Module,
        pair_track: nn.Module,
        pair_to_node: nn.Module,
        node_norm: nn.Module,
        cg_decoder: nn.Module,
        d_node: int = 256,
        d_pair: int = 64,
    ):
        """
        Args:
            rhofold_backbone: RhoFoldBackbone (冻结)
            node_proj: Linear(640→256) + LayerNorm + GELU
            pair_track: PairTrack 模块
            pair_to_node: pair→node 融合 MLP
            node_norm: LayerNorm after fusion
            cg_decoder: CG Decoder (MLP 256→3)
            d_node: node 维度
            d_pair: pair 维度
        """
        super().__init__()
        self.backbone = rhofold_backbone
        self.node_proj = node_proj
        self.pair_track = pair_track
        self.pair_to_node = pair_to_node
        self.node_norm = node_norm
        self.cg_decoder = cg_decoder
        self.d_node = d_node
        self.d_pair = d_pair

    def forward(
        self,
        seq_tokens: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """预测单个 chunk.

        Args:
            seq_tokens: (B, L_chunk) token IDs
            return_features: 是否返回融合特征 (Level 3 需要)

        Returns:
            dict: {
                'coords': (B, L_chunk, 3),
                'node_repr': (B, L_chunk, d_node) — 如果 return_features,
                'contact_map': (B, L_chunk, L_chunk) — 如果 return_features,
                'bsj_confidence': (B,),
            }
        """
        B, L = seq_tokens.shape

        # 1. RhoFold+ 编码
        node_640, pair_128 = self.backbone(seq_tokens)

        # 2. 投影
        node_feat = self.node_proj(node_640)  # (B, L, d_node)

        # 3. PairTrack
        if pair_128 is not None:
            pair_feat = self.pair_track.init_from_rna_fm_pair(pair_128)
        else:
            pair_feat = self.pair_track.init_from_node(node_feat)

        pair_feat = self.pair_track(pair_feat)  # (B, L, K, d_pair)

        # 4. Pair → Node 融合
        pair_enhance = pair_feat.mean(dim=2)  # (B, L, d_pair)
        pair_enhance = self.pair_to_node(pair_enhance)  # (B, L, d_node)
        node_feat = self.node_norm(node_feat + pair_enhance)

        # 5. CG Decoder
        coords = self.cg_decoder(node_feat)  # (B, L, 3)

        # 6. BSJ 闭合
        closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)
        bsj_conf = torch.exp(-closure_dist / 10.0)

        result = {
            'coords': coords,
            'bsj_confidence': bsj_conf,
        }

        if return_features:
            # 接触图: 基于距离阈值
            dist_matrix = torch.cdist(coords, coords)  # (B, L, L)
            contact_map = (dist_matrix < 8.0).float()  # 8Å 阈值

            result['node_repr'] = node_feat
            result['contact_map'] = contact_map

        return result

    def predict_chunk_list(
        self,
        chunks: list,
    ) -> List[ChunkPrediction]:
        """批量预测多个 chunk.

        Args:
            chunks: List[ChunkInfo] — 切分后的 chunk 列表

        Returns:
            List[ChunkPrediction] — 每个 chunk 的预测结果
        """
        predictions = []

        for chunk in chunks:
            tokens = chunk.seq_tokens.unsqueeze(0)  # (1, L_chunk)
            result = self.forward(tokens, return_features=True)

            pred = ChunkPrediction(
                coords=result['coords'].squeeze(0),        # (L_chunk, 3)
                node_repr=result['node_repr'].squeeze(0),   # (L_chunk, d_node)
                contact_map=result['contact_map'].squeeze(0), # (L_chunk, L_chunk)
                bsj_confidence=result['bsj_confidence'].item(),
                chunk_id=chunk.chunk_id,
                start=chunk.start,
                end=chunk.end,
            )
            predictions.append(pred)

        return predictions


class ChunkFeatureExtractor(nn.Module):
    """Level 2: 从 chunk 预测中提取融合特征.

    每个 chunk 输出:
      - coords (L_chunk, 3) → 旋转平移不变特征
      - node_repr (L_chunk, d_model) → 等变 GNN 特征
      - contact_map (L_chunk, L_chunk) → 局部密度特征
    """

    def __init__(self, d_feature: int = 256):
        super().__init__()
        self.d_feature = d_feature

        # 坐标 → 特征
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, d_feature),
        )

        # contact map → 特征: (1,1,L,L) → pool → (1,1,8,8) → flatten → d_feature
        self.contact_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, d_feature),
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(d_feature * 3, d_feature),
            nn.LayerNorm(d_feature),
            nn.GELU(),
        )

    def forward(
        self,
        chunk_pred: ChunkPrediction,
    ) -> torch.Tensor:
        """提取单个 chunk 的全局特征向量.

        Args:
            chunk_pred: ChunkPrediction

        Returns:
            feature: (d_feature,) — chunk 的全局特征
        """
        coords = chunk_pred.coords  # (L, 3)
        node_repr = chunk_pred.node_repr  # (L, d_model)
        contact_map = chunk_pred.contact_map  # (L, L)

        # 坐标特征: 平均池化
        coord_feat = self.coord_encoder(coords).mean(dim=0)  # (d_feature,)

        # contact map 特征: (1, 1, L, L)
        contact = contact_map.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        contact_feat = self.contact_encoder(contact).squeeze()  # (d_feature,)

        # node repr 特征
        node_feat = node_repr.mean(dim=0)  # (d_model)
        if node_feat.shape[0] != self.d_feature:
            # 投影到统一维度
            node_feat = F.adaptive_avg_pool1d(
                node_feat.unsqueeze(0).unsqueeze(0), self.d_feature
            ).squeeze()

        # 融合
        combined = torch.cat([coord_feat, contact_feat, node_feat], dim=0)
        feature = self.fusion(combined)

        return feature
