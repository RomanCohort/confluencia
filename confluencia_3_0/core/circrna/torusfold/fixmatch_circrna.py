"""
fixmatch_circrna.py — FixMatch 一致性混合训练（针对 circRNA）

核心思想：
  - 弱增强（保守） → 生成硬伪标签（高置信度）
  - 强增强（激进） → 用弱增强的伪标签监督
  - 强制模型学习"不管怎么变，本质不变"

关键设计：
  1. 弱增强：ViennaRNA 预测 + 小幅坐标扰动
  2. 强增强：序列突变 + 大幅几何扰动 + Dropout
  3. 置信度阈值：只使用高置信（>0.9）的伪标签
  4. 分层监督：局部结构（高置信）+ 全局拓扑（低置信）

优势：
  - 无需真实 3D 结构
  - 对噪声标签鲁棒
  - 自适应置信度过滤
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FixMatchConfig:
    """FixMatch 配置"""
    # 置信度阈值
    confidence_threshold_weak: float = 0.9    # 弱增强伪标签阈值
    confidence_threshold_strong: float = 0.5  # 强增强损失计算阈值

    # 增强强度
    weak_augment_scale: float = 0.1      # 弱增强扰动尺度
    strong_augment_scale: float = 1.0    # 强增强扰动尺度

    # 序列增强
    weak_mutation_rate: float = 0.01     # 弱序列突变率
    strong_mutation_rate: float = 0.15   # 强序列突变率

    # 几何增强
    weak_coord_noise: float = 0.5        # 弱坐标噪声 (Å)
    strong_coord_noise: float = 3.0      # 强坐标噪声 (Å)

    # Dropout
    weak_dropout: float = 0.0            # 弱增强 Dropout
    strong_dropout: float = 0.3          # 强增强 Dropout

    # 物理约束
    bond_length: float = 5.9             # Å
    pair_distance: float = 10.6          # Å

    # 损失权重
    w_structure: float = 1.0             # 结构损失权重
    w_closure: float = 2.0               # BSJ 闭合权重


class WeakAugmenter:
    """弱增强：保守扰动，生成高质量伪标签"""

    def __init__(self, config: FixMatchConfig):
        self.config = config

    def augment_sequence(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """弱序列增强：极低突变率"""
        B, L = seq_tokens.shape
        device = seq_tokens.device

        mutation_mask = torch.rand(B, L, device=device) < self.config.weak_mutation_rate
        random_tokens = torch.randint(0, 4, (B, L), device=device, dtype=seq_tokens.dtype)

        return torch.where(mutation_mask, random_tokens, seq_tokens)

    def augment_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """弱坐标增强：小幅噪声"""
        noise = torch.randn_like(coords) * self.config.weak_coord_noise
        return coords + noise


class StrongAugmenter:
    """强增强：激进扰动，测试模型鲁棒性"""

    def __init__(self, config: FixMatchConfig):
        self.config = config

    def augment_sequence(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """强序列增强：高突变率 + Dropout"""
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 突变
        mutation_mask = torch.rand(B, L, device=device) < self.config.strong_mutation_rate
        random_tokens = torch.randint(0, 4, (B, L), device=device, dtype=seq_tokens.dtype)
        seq_mutated = torch.where(mutation_mask, random_tokens, seq_tokens)

        # Dropout（mask）
        dropout_mask = torch.rand(B, L, device=device) < self.config.strong_dropout
        seq_augmented = torch.where(dropout_mask, torch.full_like(seq_mutated, 4), seq_mutated)

        return seq_augmented

    def augment_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """强坐标增强：大幅噪声 + 随机旋转"""
        B, L, _ = coords.shape
        device = coords.device

        # 噪声
        noise = torch.randn_like(coords) * self.config.strong_coord_noise
        coords_noisy = coords + noise

        # 随机旋转
        axis = torch.randn(B, 3, device=device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(B, 1, 1, device=device) * 2 * math.pi

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

        coords_rotated = torch.bmm(coords_noisy, R)

        return coords_rotated


class FixMatchTrainer:
    """FixMatch 一致性训练器"""

    def __init__(
        self,
        model: nn.Module,
        config: FixMatchConfig,
        vienna_teacher: Optional['ViennaRNATeacher'] = None,
    ):
        self.model = model
        self.config = config
        self.weak_augmenter = WeakAugmenter(config)
        self.strong_augmenter = StrongAugmenter(config)

        # ViennaRNA 教师（可选）
        if vienna_teacher is None:
            from .physics_distillation import ViennaRNATeacher
            self.teacher = ViennaRNATeacher()
        else:
            self.teacher = vienna_teacher

    def generate_weak_pseudo_labels(
        self,
        seq_weak: torch.Tensor,
        coords_weak: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        从弱增强生成硬伪标签

        Returns:
            pseudo_contact: (B, L, L) 高置信接触图
            pseudo_coords: (B, L, 3) 弱增强后的坐标
            confidence_mask: (B, L, L) 高置信区域 mask
        """
        B, L = seq_weak.shape
        device = seq_weak.device

        # 模型前向（弱增强）
        with torch.no_grad():
            out_weak = self.model(seq_weak)

        pred_contact = out_weak.get('pair_probs', torch.zeros(B, L, L, device=device))
        pred_coords = out_weak['coords']

        # ViennaRNA 教师（如果可用）
        # 这里简化处理，实际应该从 ViennaRNA 获取配对概率

        # 生成硬伪标签（只保留高置信）
        pseudo_contact = torch.zeros_like(pred_contact)
        confidence_mask = pred_contact > self.config.confidence_threshold_weak

        # 锐化为硬标签
        pseudo_contact[confidence_mask] = 1.0

        # 坐标伪标签：直接使用弱增强预测
        pseudo_coords = pred_coords

        return {
            'pseudo_contact': pseudo_contact,
            'pseudo_coords': pseudo_coords,
            'confidence_mask': confidence_mask,
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        coords_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """FixMatch 训练步骤"""
        seq_tokens = batch['seq_ids']  # (B, L)
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 1. 弱增强 + 生成伪标签
        seq_weak = self.weak_augmenter.augment_sequence(seq_tokens)

        if coords_target is not None:
            coords_weak = self.weak_augmenter.augment_coords(coords_target)
        else:
            coords_weak = None

        pseudo_labels = self.generate_weak_pseudo_labels(seq_weak, coords_weak)

        # 2. 强增强
        seq_strong = self.strong_augmenter.augment_sequence(seq_tokens)

        # 3. 模型前向（强增强）
        out_strong = self.model(seq_strong)
        pred_contact_strong = out_strong.get('pair_probs', torch.zeros(B, L, L, device=device))
        pred_coords_strong = out_strong['coords']

        # 4. 计算一致性损失（只在高置信区域）
        # 接触图一致性
        contact_diff = (pred_contact_strong - pseudo_labels['pseudo_contact']) ** 2

        # 只在高置信区域计算损失
        masked_contact_loss = contact_diff * pseudo_labels['confidence_mask'].float()
        contact_loss = masked_contact_loss.sum() / (pseudo_labels['confidence_mask'].sum() + 1e-8)

        # 坐标一致性（如果有）
        if coords_weak is not None:
            # 强增强坐标需要反变换（旋转逆）
            # 这里简化处理，直接比较归一化坐标

            # 归一化
            pred_norm = pred_coords_strong - pred_coords_strong.mean(dim=1, keepdim=True)
            pseudo_norm = pseudo_labels['pseudo_coords'] - pseudo_labels['pseudo_coords'].mean(dim=1, keepdim=True)

            pred_scale = torch.norm(pred_norm, dim=(1,2), keepdim=True).clamp(min=1.0)
            pseudo_scale = torch.norm(pseudo_norm, dim=(1,2), keepdim=True).clamp(min=1.0)

            pred_normalized = pred_norm / pred_scale
            pseudo_normalized = pseudo_norm / pseudo_scale

            coords_loss = F.mse_loss(pred_normalized, pseudo_normalized)

        else:
            coords_loss = torch.tensor(0.0, device=device)

        # 5. BSJ 闭合损失
        closure_dist = torch.norm(pred_coords_strong[:, 0] - pred_coords_strong[:, -1], dim=-1)
        closure_loss = ((closure_dist - self.config.bond_length) ** 2).mean()

        # 6. 总损失
        total_loss = (
            self.config.w_structure * (contact_loss + coords_loss) +
            self.config.w_closure * closure_loss
        )

        return {
            'total_loss': total_loss,
            'contact_loss': contact_loss,
            'coords_loss': coords_loss,
            'closure_loss': closure_loss,
            'confidence_ratio': pseudo_labels['confidence_mask'].sum() / (B * L * L),
        }


# ═══════════════════════════════════════════════════════════════
# ReMixMatch：分布对齐 + 增强锚定
# ═══════════════════════════════════════════════════════════════

class ReMixMatchTrainer(FixMatchTrainer):
    """ReMixMatch：FixMatch + 分布对齐"""

    def __init__(self, model: nn.Module, config: FixMatchConfig):
        super().__init__(model, config)

    def distribution_alignment(
        self,
        pseudo_labels: torch.Tensor,   # (B, L, L) 伪标签分布
        target_distribution: torch.Tensor,  # (L, L) 目标分布（如 ViennaRNA 统计）
    ) -> torch.Tensor:
        """
        分布对齐：强制伪标签分布与目标分布一致

        Args:
            pseudo_labels: 当前批次的伪标签
            target_distribution: 目标分布（从大量数据统计）

        Returns:
            aligned_labels: 对齐后的伪标签
        """
        # 当前分布
        current_dist = pseudo_labels.mean(dim=0)  # (L, L)

        # 对齐：乘以目标/当前比率
        alignment_ratio = target_distribution / (current_dist + 1e-8)

        # 软对齐（避免极端值）
        alignment_ratio = torch.clamp(alignment_ratio, 0.5, 2.0)

        # 应用对齐
        aligned_labels = pseudo_labels * alignment_ratio

        # 重新归一化
        aligned_labels = torch.clamp(aligned_labels, 0, 1)

        return aligned_labels

    def augmentation_anchoring(
        self,
        seq_tokens: torch.Tensor,
        pseudo_labels: Dict[str, torch.Tensor],
        n_augments: int = 4,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        增强锚定：多次强增强向弱增强结果靠拢

        Args:
            seq_tokens: 原始序列
            pseudo_labels: 弱增强伪标签（锚点）
            n_augments: 强增强次数

        Returns:
            多个强增强结果，都向锚点靠拢
        """
        augment_results = []

        for _ in range(n_augments):
            # 强增强
            seq_strong = self.strong_augmenter.augment_sequence(seq_tokens)

            out_strong = self.model(seq_strong)

            # 计算与锚点的距离
            anchor_loss = F.mse_loss(
                out_strong['coords'],
                pseudo_labels['pseudo_coords'],
            )

            augment_results.append({
                'seq_strong': seq_strong,
                'output': out_strong,
                'anchor_loss': anchor_loss,
            })

        return augment_results

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        target_dist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """ReMixMatch 训练步骤"""
        seq_tokens = batch['seq_ids']

        # 1. FixMatch 基础步骤
        fixmatch_result = super().train_step(batch)

        # 2. 分布对齐（如果有目标分布）
        if target_dist is not None:
            pseudo_labels_aligned = self.distribution_alignment(
                fixmatch_result['pseudo_contact'],
                target_dist,
            )
            fixmatch_result['pseudo_contact'] = pseudo_labels_aligned

        # 3. 增强锚定
        pseudo_labels = {
            'pseudo_coords': fixmatch_result['pseudo_coords'],
            'pseudo_contact': fixmatch_result['pseudo_contact'],
        }

        anchor_results = self.augmentation_anchoring(seq_tokens, pseudo_labels, n_augments=2)

        # 添加锚定损失
        anchor_loss_total = sum(r['anchor_loss'] for r in anchor_results) / len(anchor_results)
        fixmatch_result['total_loss'] = fixmatch_result['total_loss'] + 0.5 * anchor_loss_total
        fixmatch_result['anchor_loss'] = anchor_loss_total

        return fixmatch_result


# ═══════════════════════════════════════════════════════════════
# MixMatch：伪标签锐化 + MixUp
# ═══════════════════════════════════════════════════════════════

class MixMatchTrainer:
    """MixMatch：伪标签锐化 + MixUp"""

    def __init__(self, model: nn.Module, temperature: float = 0.5, alpha: float = 0.75):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha  # MixUp 比例

    def sharpen_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """锐化伪标签"""
        labels_pow = labels ** (1.0 / self.temperature)
        return labels_pow / labels_pow.sum(dim=-1, keepdim=True)

    def mixup(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MixUp: 线性插值

        Args:
            x1, x2: 两个输入（序列或坐标）
            y1, y2: 对应标签

        Returns:
            x_mix, y_mix: 混合后的输入和标签
        """
        # Beta 分布采样
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        # 混合
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2

        return x_mix, y_mix

    def train_step(
        self,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """MixMatch 训练步骤"""
        # 生成伪标签（锐化）
        with torch.no_grad():
            out_unlabeled = self.model(unlabeled_batch['seq_ids'])
            pseudo_labels = self.sharpen_labels(out_unlabeled['pair_probs'])

        # MixUp
        x_mix, y_mix = self.mixup(
            labeled_batch['seq_ids'],
            unlabeled_batch['seq_ids'],
            labeled_batch['pair_probs'],
            pseudo_labels,
        )

        # 模型前向
        out_mix = self.model(x_mix)

        # 损失
        loss = F.mse_loss(out_mix['pair_probs'], y_mix)

        return {
            'loss': loss,
            'x_mix': x_mix,
            'y_mix': y_mix,
        }


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

def train_with_fixmatch(
    model: nn.Module,
    train_loader,
    args,
    device,
):
    """FixMatch 训练主函数"""
    print("\n" + "="*60)
    print("  Training with FixMatch (Weak-Strong Consistency)")
    print("  Weak augment → Hard pseudo labels")
    print("  Strong augment → Consistency training")
    print("="*60)

    config = FixMatchConfig()
    trainer = FixMatchTrainer(model, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        confidence_ratios = []

        for batch in train_loader:
            loss_dict = trainer.train_step(batch)

            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()

            epoch_loss += loss_dict['total_loss'].item()
            confidence_ratios.append(loss_dict['confidence_ratio'].item())

        avg_loss = epoch_loss / len(train_loader)
        avg_conf = sum(confidence_ratios) / len(confidence_ratios)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, "
              f"Confidence: {avg_conf:.2%}")

    return model