"""
contrastive_circrna.py — circRNA 对比学习框架

核心思想：
  - 同一 circRNA 序列的不同扰动 → 应该预测相似的几何特征
  - 无需真实 3D 坐标标签
  - 学习序列内在的几何一致性

框架：
  1. SimCLR: 对比损失（InfoNCE）
  2. BYOL: 自蒸馏（无需负样本）
  3. SwAV: 聚类对比

关键设计：
  - 增强：序列突变、Dropout、坐标噪声
  - 投影头：将几何特征映射到对比空间
  - 温度参数：控制分布锐度
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveConfig:
    """对比学习配置"""
    # 框架选择
    framework: str = 'simclr'  # 'simclr', 'byol', 'swav'

    # 投影头
    d_proj: int = 128          # 投影空间维度
    proj_hidden: int = 256     # 投影头隐藏层维度

    # SimCLR 参数
    temperature: float = 0.07  # InfoNCE 温度
    n_negatives: int = 1024    # 负样本数（内存库）

    # BYOL 参数
    momentum: float = 0.996    # 动量更新系数
    use_predictor: bool = True # 是否使用预测头

    # 增强参数
    mutation_rate: float = 0.1     # 序列突变率
    dropout_rate: float = 0.1      # Dropout 率
    coord_noise_scale: float = 1.0 # 坐标噪声尺度 (Å)

    # 批次大小
    batch_size: int = 64


class CircRNAAugmentation:
    """circRNA 数据增强"""

    def __init__(self, config: ContrastiveConfig):
        self.config = config

    def augment_sequence(
        self,
        seq_tokens: torch.Tensor,  # (B, L)
        augment_type: str = 'mutation',
    ) -> torch.Tensor:
        """
        序列增强

        Args:
            seq_tokens: (B, L) 序列 token
            augment_type: 'mutation', 'dropout', 'mask'
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        if augment_type == 'mutation':
            # 随机突变部分碱基
            mutation_mask = torch.rand(B, L, device=device) < self.config.mutation_rate
            random_tokens = torch.randint(0, 4, (B, L), device=device, dtype=seq_tokens.dtype)
            augmented = torch.where(mutation_mask, random_tokens, seq_tokens)

        elif augment_type == 'dropout':
            # Dropout 部分 token（设为 padding）
            dropout_mask = torch.rand(B, L, device=device) < self.config.dropout_rate
            augmented = torch.where(dropout_mask, torch.full_like(seq_tokens, 4), seq_tokens)

        elif augment_type == 'mask':
            # 随机 mask 连续片段
            augmented = seq_tokens.clone()
            for b in range(B):
                # 随机选择 mask 起点
                start = torch.randint(0, max(L - 10, 1), (1,), device=device).item()
                length = torch.randint(3, min(10, L - start + 1), (1,), device=device).item()
                augmented[b, start:start+length] = 4  # Padding token

        else:
            augmented = seq_tokens

        return augmented

    def augment_coords(
        self,
        coords: torch.Tensor,  # (B, L, 3)
        augment_type: str = 'noise',
    ) -> torch.Tensor:
        """
        坐标增强

        Args:
            coords: (B, L, 3) 3D 坐标
            augment_type: 'noise', 'rotation', 'translation'
        """
        B, L, _ = coords.shape
        device = coords.device

        if augment_type == 'noise':
            # 添加高斯噪声
            noise = torch.randn_like(coords) * self.config.coord_noise_scale
            augmented = coords + noise

        elif augment_type == 'rotation':
            # 随机旋转
            # 生成随机旋转矩阵（Rodrigues 公式）
            axis = torch.randn(B, 3, device=device)
            axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
            angle = torch.rand(B, 1, 1, device=device) * 2 * math.pi

            # Rodrigues 旋转矩阵
            K = torch.zeros(B, 3, 3, device=device)
            K[:, 0, 1] = -axis[:, 2]
            K[:, 0, 2] = axis[:, 1]
            K[:, 1, 0] = axis[:, 2]
            K[:, 1, 2] = -axis[:, 0]
            K[:, 2, 0] = -axis[:, 1]
            K[:, 2, 1] = axis[:, 0]

            R = torch.eye(3, device=device).unsqueeze(0) + \
                torch.sin(angle) * K + \
                (1 - torch.cos(angle)) * (K @ K)

            augmented = torch.bmm(coords, R)

        elif augment_type == 'translation':
            # 随机平移
            translation = torch.randn(B, 1, 3, device=device) * 5.0  # ±5 Å
            augmented = coords + translation

        else:
            augmented = coords

        return augmented


class ProjectionHead(nn.Module):
    """投影头：将特征映射到对比空间"""

    def __init__(self, d_in: int, d_proj: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_in) 或 (B, d_in)
        Returns:
            (B, d_proj) 归一化的投影向量
        """
        if x.dim() == 3:
            # 全局池化
            x = x.mean(dim=1)

        proj = self.net(x)
        # L2 归一化
        return F.normalize(proj, dim=-1)


class PredictorHead(nn.Module):
    """预测头（BYOL 专用）"""

    def __init__(self, d_in: int, d_out: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRLoss(nn.Module):
    """SimCLR InfoNCE 损失"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,  # (B, d) 视角 1
        z_j: torch.Tensor,  # (B, d) 视角 2
    ) -> torch.Tensor:
        """
        InfoNCE 损失

        正样本对：(z_i[b], z_j[b])
        负样本对：所有其他组合
        """
        B, d = z_i.shape
        device = z_i.device

        # 拼接
        z = torch.cat([z_i, z_j], dim=0)  # (2B, d)

        # 相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # 构建标签：正样本对
        labels = torch.cat([
            torch.arange(B, 2*B, device=device),
            torch.arange(0, B, device=device),
        ], dim=0)

        # Mask 掉自相似
        mask = torch.eye(2*B, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # 交叉熵损失
        loss = F.cross_entropy(sim, labels)

        return loss


class BYOLLoss(nn.Module):
    """BYOL 自蒸馏损失"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        online_pred: torch.Tensor,  # (B, d) 在线网络预测
        target_proj: torch.Tensor,  # (B, d) 目标网络投影
    ) -> torch.Tensor:
        """
        BYOL 损失 = 2 - 2 * cos_similarity(p, z)
        """
        # 归一化
        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)

        # 余弦相似度
        cos_sim = (online_pred * target_proj).sum(dim=-1)

        # 损失
        loss = 2 - 2 * cos_sim.mean()

        return loss


class ContrastiveTrainer:
    """对比学习训练器"""

    def __init__(
        self,
        encoder: nn.Module,
        config: ContrastiveConfig,
    ):
        self.encoder = encoder
        self.config = config
        self.augmentor = CircRNAAugmentation(config)

        # 投影头
        d_encoder = getattr(encoder, 'd_model', 128)
        self.projector = ProjectionHead(d_encoder, config.d_proj, config.proj_hidden)

        # 根据框架选择损失
        if config.framework == 'simclr':
            self.loss_fn = SimCLRLoss(config.temperature)

        elif config.framework == 'byol':
            self.loss_fn = BYOLLoss()
            # 目标网络（动量更新）
            self.target_encoder = self._copy_model(encoder)
            self.target_projector = self._copy_model(self.projector)
            self.predictor = PredictorHead(config.d_proj, config.d_proj, config.proj_hidden)

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """复制模型（用于 BYOL 目标网络）"""
        import copy
        return copy.deepcopy(model)

    @torch.no_grad()
    def _momentum_update(self):
        """动量更新目标网络（BYOL）"""
        if self.config.framework != 'byol':
            return

        m = self.config.momentum
        for param, param_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = m * param_target.data + (1 - m) * param.data

        for param, param_target in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_target.data = m * param_target.data + (1 - m) * param.data

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """对比学习训练步骤"""
        seq_tokens = batch['seq_ids']  # (B, L)

        # 1. 生成两个增强视角
        seq_i = self.augmentor.augment_sequence(seq_tokens, augment_type='mutation')
        seq_j = self.augmentor.augment_sequence(seq_tokens, augment_type='dropout')

        # 2. 编码器前向
        if self.config.framework == 'byol':
            # 在线网络
            feat_i = self.encoder(seq_i)
            z_i = self.projector(feat_i['single_repr'] if isinstance(feat_i, dict) else feat_i)
            p_i = self.predictor(z_i)

            # 目标网络（不计算梯度）
            with torch.no_grad():
                feat_j = self.target_encoder(seq_j)
                z_j = self.target_projector(feat_j['single_repr'] if isinstance(feat_j, dict) else feat_j)

            # 损失
            loss = self.loss_fn(p_i, z_j)

            # 对称损失
            with torch.no_grad():
                feat_i_target = self.target_encoder(seq_i)
                z_i_target = self.target_projector(feat_i_target['single_repr'] if isinstance(feat_i_target, dict) else feat_i_target)

            feat_j_online = self.encoder(seq_j)
            z_j_online = self.projector(feat_j_online['single_repr'] if isinstance(feat_j_online, dict) else feat_j_online)
            p_j = self.predictor(z_j_online)

            loss = (loss + self.loss_fn(p_j, z_i_target)) / 2

            # 动量更新
            self._momentum_update()

        else:  # SimCLR
            feat_i = self.encoder(seq_i)
            feat_j = self.encoder(seq_j)

            z_i = self.projector(feat_i['single_repr'] if isinstance(feat_i, dict) else feat_i)
            z_j = self.projector(feat_j['single_repr'] if isinstance(feat_j, dict) else feat_j)

            loss = self.loss_fn(z_i, z_j)

        return {
            'loss': loss,
            'z_i': z_i,
            'z_j': z_j,
        }


# ═══════════════════════════════════════════════════════════════
# 几何一致性对比学习（Geometry-Aware Contrastive Learning）
# ═══════════════════════════════════════════════════════════════

class GeometryContrastiveLoss(nn.Module):
    """
    几何一致性对比损失

    核心思想：
      - 同一序列的两个增强版本 → 预测的 3D 结构应该几何一致
      - 不需要真实标签，只需内部一致性
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        coords_i: torch.Tensor,  # (B, L, 3) 视角 1 预测坐标
        coords_j: torch.Tensor,  # (B, L, 3) 视角 2 预测坐标
        lengths: torch.Tensor,   # (B,) 实际长度
    ) -> torch.Tensor:
        """
        几何一致性损失

        要求：两个视角预测的接触图相似
        """
        B, L, _ = coords_i.shape
        device = coords_i.device

        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            valid_L = lengths[b]

            # 计算接触图（距离矩阵）
            dist_i = torch.cdist(coords_i[b, :valid_L].unsqueeze(0),
                                  coords_i[b, :valid_L].unsqueeze(0)).squeeze(0)
            dist_j = torch.cdist(coords_j[b, :valid_L].unsqueeze(0),
                                  coords_j[b, :valid_L].unsqueeze(0)).squeeze(0)

            # 转换为接触概率（距离越近，概率越高）
            contact_i = torch.exp(-dist_i / self.temperature)
            contact_j = torch.exp(-dist_j / self.temperature)

            # 一致性损失：KL 散度
            loss = F.kl_div(
                F.log_softmax(contact_i.view(-1), dim=0),
                F.softmax(contact_j.view(-1), dim=0),
                reduction='sum',
            )

            total_loss = total_loss + loss

        return total_loss / B


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

def train_with_contrastive_learning(
    model: nn.Module,
    train_loader,
    args,
    device,
):
    """对比学习训练主函数"""
    print("\n" + "="*60)
    print("  Training with Contrastive Learning")
    print("  Framework: SimCLR / BYOL")
    print("="*60)

    config = ContrastiveConfig(framework='byol')
    trainer = ContrastiveTrainer(model, config)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(trainer.projector.parameters()),
        lr=args.lr
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            loss_dict = trainer.train_step(batch)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            epoch_loss += loss_dict['loss'].item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    return model
