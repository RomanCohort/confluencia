"""
unified_training_strategy.py — TorusFold 统一训练策略

整合所有训练策略：
  1. 物理蒸馏（Physics Distillation）—— 最推荐
  2. 对比学习（Contrastive Learning）
  3. FixMatch 一致性训练（Weak-Strong Consistency）
  4. 迁移学习（Transfer Learning）

推荐组合：
  - 基础方案：物理蒸馏 + FixMatch
  - 进阶方案：物理蒸馏 + 对比学习 + 渐进解冻
  - 最高质量：预训练 + 物理蒸馏 + FixMatch + 等变约束

使用示例：
  python unified_training_strategy.py --strategy physics_distillation --epochs 50
  python unified_training_strategy.py --strategy contrastive --framework byol
  python unified_training_strategy.py --strategy fixmatch --weak-aug 0.1 --strong-aug 1.0
  python unified_training_strategy.py --strategy transfer --pretrained model.pt
  python unified_training_strategy.py --strategy combined --epochs 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# 导入各个策略模块
from .physics_distillation import (
    PhysicsDistillationConfig,
    PhysicsDistillationTrainer,
    ViennaRNATeacher,
)
from .contrastive_circrna import (
    ContrastiveConfig,
    ContrastiveTrainer,
)
from .fixmatch_circrna import (
    FixMatchConfig,
    FixMatchTrainer,
    ReMixMatchTrainer,
)
from .transfer_circrna import (
    TransferConfig,
    ProgressiveTransferTrainer,
    AnchorGenerator,
)


@dataclass
class UnifiedConfig:
    """统一配置"""
    # 策略选择
    strategy: str = 'physics_distillation'  # 'physics_distillation', 'contrastive', 'fixmatch', 'transfer', 'combined'

    # 物理蒸馏配置
    physics: PhysicsDistillationConfig = None

    # 对比学习配置
    contrastive: ContrastiveConfig = None

    # FixMatch 配置
    fixmatch: FixMatchConfig = None

    # 迁移学习配置
    transfer: TransferConfig = None

    # 训练参数
    epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 8
    device: str = 'cuda'

    # 数据路径
    data_path: str = 'data/circrna_3d_merged'
    output_path: str = 'outputs/unified_training'

    def __post_init__(self):
        if self.physics is None:
            self.physics = PhysicsDistillationConfig()
        if self.contrastive is None:
            self.contrastive = ContrastiveConfig()
        if self.fixmatch is None:
            self.fixmatch = FixMatchConfig()
        if self.transfer is None:
            self.transfer = TransferConfig()


class UnifiedTrainer:
    """统一训练器：整合所有策略"""

    def __init__(self, model: nn.Module, config: UnifiedConfig):
        self.model = model
        self.config = config

        # 根据策略初始化子训练器
        self.trainers = self._setup_trainers()

    def _setup_trainers(self) -> Dict:
        """设置训练器"""
        trainers = {}

        if 'physics' in self.config.strategy or self.config.strategy == 'combined':
            trainers['physics'] = PhysicsDistillationTrainer(
                self.model, self.config.physics
            )

        if 'contrastive' in self.config.strategy or self.config.strategy == 'combined':
            trainers['contrastive'] = ContrastiveTrainer(
                self.model, self.config.contrastive
            )

        if 'fixmatch' in self.config.strategy or self.config.strategy == 'combined':
            trainers['fixmatch'] = FixMatchTrainer(
                self.model, self.config.fixmatch
            )

        if 'transfer' in self.config.strategy:
            trainers['transfer'] = ProgressiveTransferTrainer(
                self.model, self.config.transfer
            )

        return trainers

    def train(self, train_loader, val_loader=None):
        """主训练循环"""
        print(f"\n{'='*60}")
        print(f"  Unified Training Strategy: {self.config.strategy.upper()}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')

        for epoch in range(self.config.epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader, optimizer, epoch)

            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, best_val_loss)

                print(f"Epoch {epoch+1}/{self.config.epochs} "
                      f"train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.config.epochs} train={train_loss:.4f}")

        return self.model

    def _train_epoch(self, train_loader, optimizer, epoch):
        """单 epoch 训练"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            loss = self._compute_loss(batch, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _compute_loss(self, batch, epoch):
        """根据策略计算损失"""
        losses = []

        # 物理蒸馏损失
        if 'physics' in self.trainers:
            sequences = batch.get('sequences', [])
            pseudo_labels = self.trainers['physics'].generate_pseudo_labels(sequences)
            loss_dict = self.trainers['physics'].train_step(batch, pseudo_labels)
            losses.append(loss_dict['total_loss'])

        # 对比学习损失
        if 'contrastive' in self.trainers:
            loss_dict = self.trainers['contrastive'].train_step(batch)
            losses.append(loss_dict['loss'])

        # FixMatch 损失
        if 'fixmatch' in self.trainers:
            loss_dict = self.trainers['fixmatch'].train_step(batch)
            losses.append(loss_dict['total_loss'])

        # 迁移学习损失
        if 'transfer' in self.trainers:
            # 需要预先生成锚点
            loss_dict = self.trainers['transfer'].train_step(batch, [], epoch)
            losses.append(loss_dict['loss'])

        # 组合损失
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.config.device)

        return sum(losses) / len(losses)

    def _validate_epoch(self, val_loader):
        """验证阶段"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_loss(batch, -1)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }

        path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════
# 预设策略配置
# ═══════════════════════════════════════════════════════════════

def get_preset_config(preset: str) -> UnifiedConfig:
    """获取预设配置"""

    if preset == 'quick':
        # 快速训练：仅物理蒸馏
        return UnifiedConfig(
            strategy='physics_distillation',
            epochs=30,
            lr=1e-4,
            physics=PhysicsDistillationConfig(
                w_contact=1.0,
                w_closure=2.0,
            ),
        )

    elif preset == 'standard':
        # 标准训练：物理蒸馏 + FixMatch
        return UnifiedConfig(
            strategy='combined',
            epochs=50,
            lr=1e-4,
            physics=PhysicsDistillationConfig(
                w_contact=1.0,
                w_kl=0.5,
                w_closure=2.0,
            ),
            fixmatch=FixMatchConfig(
                confidence_threshold_weak=0.9,
                w_structure=1.0,
                w_closure=2.0,
            ),
        )

    elif preset == 'advanced':
        # 进阶训练：物理蒸馏 + 对比学习 + 渐进解冻
        return UnifiedConfig(
            strategy='combined',
            epochs=100,
            lr=5e-5,
            physics=PhysicsDistillationConfig(
                w_contact=1.0,
                w_kl=0.5,
                w_energy=0.3,
                w_closure=2.0,
            ),
            contrastive=ContrastiveConfig(
                framework='byol',
                temperature=0.1,
                momentum=0.996,
            ),
            transfer=TransferConfig(
                progressive_unfreeze=True,
            ),
        )

    elif preset == 'production':
        # 生产级训练：全策略组合
        return UnifiedConfig(
            strategy='combined',
            epochs=150,
            lr=3e-5,
            batch_size=4,
            physics=PhysicsDistillationConfig(
                w_contact=1.5,
                w_kl=0.7,
                w_energy=0.5,
                w_closure=3.0,
                temperature_sharpen=0.5,
            ),
            contrastive=ContrastiveConfig(
                framework='byol',
                d_proj=256,
                temperature=0.07,
                momentum=0.998,
            ),
            fixmatch=FixMatchConfig(
                confidence_threshold_weak=0.95,
                strong_mutation_rate=0.1,
                w_structure=1.2,
                w_closure=3.0,
            ),
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")


# ═══════════════════════════════════════════════════════════════
# 命令行接口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TorusFold Unified Training')

    # 策略选择
    parser.add_argument('--strategy', type=str, default='physics_distillation',
                       choices=['physics_distillation', 'contrastive', 'fixmatch',
                               'transfer', 'combined'],
                       help='Training strategy')

    parser.add_argument('--preset', type=str, default=None,
                       choices=['quick', 'standard', 'advanced', 'production'],
                       help='Use preset configuration')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    # 数据路径
    parser.add_argument('--data-path', type=str, default='data/circrna_3d_merged')
    parser.add_argument('--output-path', type=str, default='outputs/unified_training')

    # 物理蒸馏参数
    parser.add_argument('--w-contact', type=float, default=1.0)
    parser.add_argument('--w-kl', type=float, default=0.5)
    parser.add_argument('--w-energy', type=float, default=0.3)
    parser.add_argument('--w-closure', type=float, default=2.0)

    # 对比学习参数
    parser.add_argument('--contrastive-framework', type=str, default='byol',
                       choices=['simclr', 'byol'])
    parser.add_argument('--temperature', type=float, default=0.07)

    # FixMatch 参数
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    parser.add_argument('--weak-aug', type=float, default=0.1)
    parser.add_argument('--strong-aug', type=float, default=1.0)

    # 迁移学习参数
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model')

    args = parser.parse_args()

    # 构建配置
    if args.preset:
        config = get_preset_config(args.preset)
    else:
        config = UnifiedConfig(
            strategy=args.strategy,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            data_path=args.data_path,
            output_path=args.output_path,
            physics=PhysicsDistillationConfig(
                w_contact=args.w_contact,
                w_kl=args.w_kl,
                w_energy=args.w_energy,
                w_closure=args.w_closure,
            ),
            contrastive=ContrastiveConfig(
                framework=args.contrastive_framework,
                temperature=args.temperature,
            ),
            fixmatch=FixMatchConfig(
                confidence_threshold_weak=args.confidence_threshold,
                weak_augment_scale=args.weak_aug,
                strong_augment_scale=args.strong_aug,
            ),
            transfer=TransferConfig(
                pretrained_model_path=args.pretrained,
            ),
        )

    # 打印配置
    print(f"\n{'='*60}")
    print(f"  TorusFold Unified Training Configuration")
    print(f"{'='*60}")
    print(f"Strategy: {config.strategy}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    # 这里应该加载模型和数据
    # 示例代码（实际使用需要完整实现）
    print("To use this trainer, implement model loading and data loading.")
    print("Example:")
    print("  from torusfold import Scheme8Model")
    print("  model = Scheme8Model()")
    print("  trainer = UnifiedTrainer(model, config)")
    print("  trainer.train(train_loader, val_loader)")


if __name__ == '__main__':
    main()
