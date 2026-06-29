"""
transfer_circrna.py — 跨家族迁移学习（部分伪标签策略）

核心思想：
  - 线性 RNA (mRNA/tRNA) 有大量 PDB 结构
  - circRNA 缺少实验结构，但局部片段与线性 RNA 相似
  - 用预训练模型生成"高置信局部锚点"，只训练 BSJ 区域

策略：
  1. 预训练：在 PDB 线性 RNA 上训练基础模型
  2. 伪标签生成：预测 circRNA，筛选高置信局部片段
  3. 锚定迁移：冻结高置信区域参数，只训练 BSJ 连接区
  4. 渐进解冻：逐步释放更多参数进行微调

优势：
  - 利用现有 RNA 结构数据（PDB 有数千个）
  - 降低 circRNA 训练的不确定性
  - 物理保证：局部结构从已知数据迁移
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


@dataclass
class TransferConfig:
    """迁移学习配置"""
    # 置信度阈值
    confidence_anchor: float = 0.99    # 锚点置信度阈值（极高）
    confidence_fine_tune: float = 0.7  # 微调置信度阈值

    # 锚点策略
    anchor_min_length: int = 10        # 锚点最小长度
    anchor_max_length: int = 50        # 锚点最大长度
    freeze_anchor_layers: bool = True  # 是否冻结锚点相关层

    # BSJ 区域定义
    bsj_flank_size: int = 30           # BSJ 两侧区域大小
    bsj_train_weight: float = 3.0      # BSJ 区域损失权重

    # 渐进解冻
    progressive_unfreeze: bool = True
    unfreeze_schedule: List[int] = [10, 20, 30]  # 每 N epoch 解冻一层

    # 预训练模型路径
    pretrained_model_path: Optional[str] = None


class PDBLinearRNAPreTrainer:
    """在 PDB 线性 RNA 上预训练"""

    def __init__(self, model: nn.Module, pdb_data_path: str):
        self.model = model
        self.pdb_data_path = Path(pdb_data_path)

    def load_pdb_rna_data(self) -> List[Dict]:
        """加载 PDB 线性 RNA 结构数据"""
        # PDB RNA 类型：
        # - tRNA (数百个)
        # - ribozyme (数百个)
        # - mRNA fragments
        # - ribosomal RNA fragments

        data_files = list(self.pdb_data_path.glob("*.json"))
        all_data = []

        for f in data_files:
            with open(f) as fp:
                data = json.load(fp)
                all_data.extend(data.get('structures', []))

        print(f"Loaded {len(all_data)} linear RNA structures from PDB")
        return all_data

    def pretrain(
        self,
        epochs: int = 50,
        lr: float = 1e-4,
        device: str = 'cuda',
    ):
        """预训练循环"""
        data = self.load_pdb_rna_data()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0

            for item in data:
                # 构建 batch
                seq = item['sequence']
                coords = torch.tensor(item['coords'], dtype=torch.float32)
                seq_tokens = self._tokenize(seq)

                # 前向
                seq_tokens = seq_tokens.unsqueeze(0).to(device)
                coords_target = coords.unsqueeze(0).to(device)

                out = self.model(seq_tokens)

                # 损失（标准 3D 结构预测）
                loss = self._compute_loss(out['coords'], coords_target)

                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data)
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.model

    def _tokenize(self, seq: str) -> torch.Tensor:
        """序列 tokenize"""
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        return torch.tensor([mapping.get(b, 4) for b in seq], dtype=torch.long)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算结构预测损失"""
        # 归一化
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)

        pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
        target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)

        pred_norm = pred_centered / pred_scale
        target_norm = target_centered / target_scale

        return F.mse_loss(pred_norm, target_norm)


class AnchorGenerator:
    """生成高置信局部锚点"""

    def __init__(self, pretrained_model: nn.Module, config: TransferConfig):
        self.model = pretrained_model
        self.config = config

    def generate_anchors(
        self,
        circRNA_sequences: List[str],
        device: str = 'cuda',
    ) -> List[Dict]:
        """
        为 circRNA 序列生成局部锚点

        Returns:
            anchors: 每个序列的锚点列表
                [{
                    'region': (start, end),
                    'coords': anchor_coords,
                    'confidence': confidence_score,
                    'frozen': True,
                }, ...]
        """
        anchors_all = []

        for seq in circRNA_sequences:
            seq_tokens = self._tokenize(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                out = self.model(seq_tokens)

            pred_coords = out['coords'][0].cpu()  # (L, 3)
            pred_confidence = out.get('confidence', torch.ones(len(seq))).cpu()

            # 识别高置信局部区域
            anchors = self._identify_anchor_regions(
                pred_coords,
                pred_confidence,
                len(seq),
            )

            anchors_all.append({
                'sequence': seq,
                'anchors': anchors,
                'full_coords': pred_coords,
                'bsj_region': self._define_bsj_region(len(seq)),
            })

        return anchors_all

    def _identify_anchor_regions(
        self,
        coords: torch.Tensor,
        confidence: torch.Tensor,
        L: int,
    ) -> List[Dict]:
        """识别高置信区域作为锚点"""
        anchors = []

        # 滑动窗口检测连续高置信区域
        window = self.config.anchor_min_length

        for start in range(0, L - window, window // 2):
            end = min(start + window, L)

            # 区域平均置信度
            region_conf = confidence[start:end].mean().item()

            if region_conf >= self.config.confidence_anchor:
                # 高置信区域
                anchors.append({
                    'region': (start, end),
                    'coords': coords[start:end],
                    'confidence': region_conf,
                    'frozen': True,
                })

            elif region_conf >= self.config.confidence_fine_tune:
                # 中置信区域（可微调）
                anchors.append({
                    'region': (start, end),
                    'coords': coords[start:end],
                    'confidence': region_conf,
                    'frozen': False,
                })

        return anchors

    def _define_bsj_region(self, L: int) -> Tuple[int, int]:
        """定义 BSJ 区域（需要重点训练）"""
        flank = self.config.bsj_flank_size
        # BSJ 连接点：位置 0 和位置 L-1
        # 训练区域：两端 flank 区域
        return (0, min(flank, L // 4), max(L - flank, L * 3 // 4), L)

    def _tokenize(self, seq: str) -> torch.Tensor:
        """序列 tokenize"""
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        return torch.tensor([mapping.get(b, 4) for b in seq], dtype=torch.long)


class PartialPseudoLabelTrainer:
    """部分伪标签训练器"""

    def __init__(
        self,
        model: nn.Module,
        config: TransferConfig,
        anchors: List[Dict],
    ):
        self.model = model
        self.config = config
        self.anchors = anchors

        # 标记冻结参数
        self._setup_freezing()

    def _setup_freezing(self):
        """根据锚点冻结模型参数"""
        if not self.config.freeze_anchor_layers:
            return

        # 统计所有冻结锚点区域
        frozen_positions = set()
        for seq_data in self.anchors:
            for anchor in seq_data['anchors']:
                if anchor['frozen']:
                    start, end = anchor['region']
                    frozen_positions.update(range(start, end))

        print(f"Frozen positions: {len(frozen_positions)} / total")

        # 实际冻结策略：
        # - 方案 A：冻结特定层的参数（更简单）
        # - 方案 B：冻结特定位置的梯度（需要梯度 mask）

        # 这里采用方案 A：冻结前几层（底层特征对所有序列通用）
        n_layers_to_freeze = 2  # 冻结前 2 层

        for i, layer in enumerate(list(self.model.children())[:n_layers_to_freeze]):
            for param in layer.parameters():
                param.requires_grad = False

        print(f"Frozen {n_layers_to_freeze} bottom layers")

    def progressive_unfreeze(self, epoch: int):
        """渐进解冻"""
        if not self.config.progressive_unfreeze:
            return

        # 检查是否达到解冻时间点
        for i, threshold in enumerate(self.config.unfreeze_schedule):
            if epoch == threshold:
                # 解冻第 i+1 层
                layers = list(self.model.children())
                if i + 2 < len(layers):
                    layer = layers[i + 2]
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Epoch {epoch}: Unfrozen layer {i+2}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        anchor_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        训练步骤：锚点监督 + BSJ 重点训练

        Args:
            batch: 当前批次数据
            anchor_idx: 对应的锚点索引
        """
        seq_tokens = batch['seq_ids']
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # 模型前向
        out = self.model(seq_tokens)
        pred_coords = out['coords']

        # 1. 锚点监督损失（冻结区域）
        anchor_loss = torch.tensor(0.0, device=device)

        anchor_data = self.anchors[anchor_idx]
        for anchor in anchor_data['anchors']:
            if anchor['frozen']:
                start, end = anchor['region']
                anchor_coords = anchor['coords'].to(device)

                # 强制预测与锚点一致
                anchor_loss = anchor_loss + F.mse_loss(
                    pred_coords[0, start:end],
                    anchor_coords,
                )

        # 2. BSJ 重点损失（非冻结区域）
        bsj_region = anchor_data['bsj_region']
        head_start, head_end, tail_start, tail_end = bsj_region

        # BSJ 闭合约束
        closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
        closure_loss = (closure_dist - 5.9) ** 2

        # BSJ 区域结构约束
        bsj_loss = torch.tensor(0.0, device=device)

        # 头部区域
        if head_end > head_start:
            head_pred = pred_coords[:, head_start:head_end]
            # 与锚点平滑过渡（如果有）
            if len(anchor_data['anchors']) > 0:
                first_anchor = anchor_data['anchors'][0]
                if first_anchor['region'][0] < head_end:
                    # 过渡区域
                    transition_start = first_anchor['region'][0]
                    transition_end = min(head_end, first_anchor['region'][1])

                    # 梯度惩罚：确保平滑过渡
                    transition_pred = pred_coords[:, transition_start:transition_end]
                    transition_anchor = first_anchor['coords'][transition_start-first_anchor['region'][0]:transition_end-first_anchor['region'][0]].to(device)

                    bsj_loss = bsj_loss + F.mse_loss(transition_pred, transition_anchor.unsqueeze(0))

        # 尾部区域类似处理...

        # 3. 总损失
        total_loss = (
            anchor_loss +
            self.config.bsj_train_weight * (closure_loss + bsj_loss)
        )

        return {
            'total_loss': total_loss,
            'anchor_loss': anchor_loss,
            'closure_loss': closure_loss.mean(),
            'bsj_loss': bsj_loss,
        }


# ═══════════════════════════════════════════════════════════════
# 渐进式训练策略（Progressive Training）
# ═══════════════════════════════════════════════════════════════

class ProgressiveTransferTrainer:
    """渐进式迁移训练"""

    def __init__(self, model: nn.Module, config: TransferConfig):
        self.model = model
        self.config = config
        self.stage = 0  # 0: 锚定, 1: 微调, 2: 全量

    def get_current_stage(self, epoch: int) -> int:
        """根据 epoch 确定训练阶段"""
        if epoch < 10:
            return 0  # 锚定阶段：只训练 BSJ
        elif epoch < 30:
            return 1  # 微调阶段：训练中置信区域
        else:
            return 2  # 全量阶段：全部训练

    def configure_stage(self, stage: int):
        """配置当前阶段"""
        if stage == 0:
            # 锚定：冻结大部分，只训练 BSJ 相关层
            self._freeze_except_bsj()

        elif stage == 1:
            # 微调：解冻中置信区域
            self._unfreeze_medium_confidence()

        elif stage == 2:
            # 全量：全部可训练
            self._unfreeze_all()

    def _freeze_except_bsj(self):
        """冻结除 BSJ 外的所有层"""
        # 保留最后几层（负责 BSJ）可训练
        layers = list(self.model.children())
        n_trainable = 2

        for i, layer in enumerate(layers[:-n_trainable]):
            for param in layer.parameters():
                param.requires_grad = False

        for layer in layers[-n_trainable:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Stage 0: Frozen {len(layers) - n_trainable} layers, training BSJ")

    def _unfreeze_medium_confidence(self):
        """解冻中等置信度区域"""
        layers = list(self.model.children())
        n_to_unfreeze = len(layers) // 2

        for i, layer in enumerate(layers[:n_to_unfreeze]):
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Stage 1: Unfrozen {n_to_unfreeze} layers")

    def _unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True

        print("Stage 2: All layers unfrozen")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        anchors: List[Dict],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """根据阶段调整训练"""
        stage = self.get_current_stage(epoch)

        if stage != self.stage:
            self.configure_stage(stage)
            self.stage = stage

        # 针对不同阶段的损失设计
        seq_tokens = batch['seq_ids']
        out = self.model(seq_tokens)
        pred_coords = out['coords']

        if stage == 0:
            # 只计算 BSJ 闭合损失
            closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
            loss = (closure_dist - 5.9) ** 2

        elif stage == 1:
            # BSJ + 中置信区域损失
            closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
            closure_loss = (closure_dist - 5.9) ** 2

            # 添加中置信区域监督（如果有锚点）
            anchor_loss = torch.tensor(0.0, device=seq_tokens.device)
            for anchor in anchors:
                if not anchor['frozen'] and anchor['confidence'] > 0.7:
                    start, end = anchor['region']
                    anchor_coords = anchor['coords'].to(seq_tokens.device)
                    anchor_loss = anchor_loss + F.mse_loss(
                        pred_coords[0, start:end],
                        anchor_coords,
                    )

            loss = closure_loss.mean() + 0.5 * anchor_loss

        else:  # stage == 2
            # 全量损失（标准训练）
            # 这里可以添加标准 3D 结构预测损失
            # 或者与物理蒸馏、对比学习结合
            loss = self._full_training_loss(batch, out)

        return {'loss': loss}

    def _full_training_loss(self, batch: Dict, out: Dict) -> torch.Tensor:
        """全量训练损失"""
        # 可以集成之前定义的物理蒸馏、对比学习等策略
        # 这里简化为标准损失
        pred_coords = out['coords']

        # 闭合损失
        closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
        closure_loss = (closure_dist - 5.9) ** 2

        # 键长一致性
        bond_loss = torch.tensor(0.0, device=pred_coords.device)
        for b in range(pred_coords.shape[0]):
            bonds = torch.norm(pred_coords[b, 1:] - pred_coords[b, :-1], dim=-1)
            bond_loss = bond_loss + F.mse_loss(bonds, torch.full_like(bonds, 5.9))

        return closure_loss.mean() + 0.5 * bond_loss.mean()


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

def train_with_transfer_learning(
    model: nn.Module,
    circRNA_sequences: List[str],
    args,
    device,
):
    """迁移学习训练主函数"""
    print("\n" + "="*60)
    print("  Training with Transfer Learning")
    print("  Stage 0: Anchor BSJ (freeze most layers)")
    print("  Stage 1: Fine-tune medium confidence regions")
    print("  Stage 2: Full training")
    print("="*60)

    config = TransferConfig()

    # 1. 预训练（如果需要）
    if config.pretrained_model_path:
        print(f"Loading pretrained model from {config.pretrained_model_path}")
        model.load_state_dict(torch.load(config.pretrained_model_path))
    else:
        print("No pretrained model provided, using current model")

    # 2. 生成锚点
    anchor_generator = AnchorGenerator(model, config)
    anchors = anchor_generator.generate_anchors(circRNA_sequences, device)

    # 3. 渐进式训练
    trainer = ProgressiveTransferTrainer(model, config)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for i, seq in enumerate(circRNA_sequences):
            seq_tokens = anchor_generator._tokenize(seq).unsqueeze(0).to(device)
            batch = {'seq_ids': seq_tokens}

            anchor_data = anchors[i]

            loss_dict = trainer.train_step(batch, anchor_data['anchors'], epoch)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            epoch_loss += loss_dict['loss'].item()

        avg_loss = epoch_loss / len(circRNA_sequences)
        stage = trainer.get_current_stage(epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Stage {stage}, Loss: {avg_loss:.4f}")

    return model