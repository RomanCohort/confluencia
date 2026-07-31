"""
contact_map_aux_head.py — 接触图辅助任务头

从 latent_inv 投影预测接触图，作为辅助任务：
- 强制网络学习有意义的中间表征
- 主路径仍端到端（接触图只是辅助监督）

接触图定义：两个残基的 C4' 距离 < 8Å 则为接触（1），否则为 0。

用法：
    from contact_map_aux_head import ContactMapAuxHead

    head = ContactMapAuxHead(d_inv=32)
    contact_pred = head(latent_inv)  # [B, L, L]
    aux_loss = F.binary_cross_entropy(contact_pred, contact_target)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactMapAuxHead(nn.Module):
    """接触图辅助任务头

    从 latent_inv 预测残基间接触图。

    设计：
    1. 投影 inv 到 contact embedding
    2. 用外积 + MLP 预测接触概率
    3. 保持对称性（contact[i,j] == contact[j,i]）

    Args:
        d_inv: inv 特征维度
        d_hidden: 隐藏层维度
        contact_threshold: 接触距离阈值（Å）
    """

    def __init__(
        self,
        d_inv: int = 32,
        d_hidden: int = 64,
        contact_threshold: float = 8.0,
    ):
        super().__init__()
        self.d_inv = d_inv
        self.d_hidden = d_hidden
        self.contact_threshold = contact_threshold

        # 投影层：inv → contact embedding
        self.contact_proj = nn.Sequential(
            nn.Linear(d_inv, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # 接触预测 MLP（输入：拼接的 pair embedding）
        # 对称设计：用 (e_i + e_j) 和 (e_i * e_j) 两个特征
        self.contact_mlp = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden),  # e_i, e_j, e_i*e_j → hidden
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, latent_inv: torch.Tensor) -> torch.Tensor:
        """预测接触图

        Args:
            latent_inv: [B, L, d_inv]

        Returns:
            contact_pred: [B, L, L] 接触概率（对称）
        """
        B, L, _ = latent_inv.shape

        # 投影到 contact embedding
        emb = self.contact_proj(latent_inv)  # [B, L, d_hidden]

        # 构建 pair features（对称）
        emb_i = emb.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, d_hidden]
        emb_j = emb.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, d_hidden]
        emb_prod = emb_i * emb_j                          # [B, L, L, d_hidden]

        # 拼接特征
        pair_feat = torch.cat([emb_i, emb_j, emb_prod], dim=-1)  # [B, L, L, d_hidden*3]

        # 预测接触概率
        contact_logit = self.contact_mlp(pair_feat).squeeze(-1)  # [B, L, L]

        # 强制对称性：(logit[i,j] + logit[j,i]) / 2
        contact_logit = (contact_logit + contact_logit.transpose(-1, -2)) / 2

        # Sigmoid 转概率
        contact_pred = torch.sigmoid(contact_logit)

        return contact_pred

    def compute_aux_loss(
        self,
        latent_inv: torch.Tensor,
        coords_target: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """计算辅助 loss

        Args:
            latent_inv: [B, L, d_inv]
            coords_target: [B, L, 3] 真实坐标（用于生成接触图标签）
            lengths: [B] 有效长度

        Returns:
            loss: 标量
        """
        B, L, _ = latent_inv.shape
        device = latent_inv.device

        # 预测接触图
        contact_pred = self.forward(latent_inv)  # [B, L, L]

        # 从真实坐标生成接触标签
        dist = torch.cdist(coords_target, coords_target)  # [B, L, L]
        contact_target = (dist < self.contact_threshold).float()

        # 排除对角线（自己和自己）
        eye = torch.eye(L, device=device).unsqueeze(0)
        contact_target = contact_target * (1 - eye)
        contact_pred = contact_pred * (1 - eye)

        # 加权 BCE（接触比非接触少，正样本加权）
        n_positive = contact_target.sum(dim=(-1, -2)).clamp(min=1)
        n_negative = (1 - contact_target).sum(dim=(-1, -2)).clamp(min=1)
        pos_weight = n_negative / n_positive  # [B]

        # BCE loss
        bce = F.binary_cross_entropy(contact_pred, contact_target, reduction='none')  # [B, L, L]

        # 加权
        weighted_bce = bce * (contact_target * pos_weight.unsqueeze(-1).unsqueeze(-1) + (1 - contact_target))

        # 归一化
        if lengths is not None:
            # 只计算有效区域
            mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, L, L]
            pair_mask = pair_mask * (1 - eye)
            weighted_bce = weighted_bce * pair_mask
            loss = weighted_bce.sum(dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp(min=1)
        else:
            loss = weighted_bce.mean(dim=(-1, -2))

        return loss.mean()


def generate_contact_map(coords: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """从坐标生成接触图（用于生成标签）

    Args:
        coords: [B, L, 3]
        threshold: 接触距离阈值

    Returns:
        contact: [B, L, L] 二值接触图
    """
    dist = torch.cdist(coords, coords)
    contact = (dist < threshold).float()
    # 排除对角线
    L = coords.shape[1]
    eye = torch.eye(L, device=coords.device).unsqueeze(0)
    contact = contact * (1 - eye)
    return contact