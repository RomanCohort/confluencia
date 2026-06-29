"""
physics_distillation.py — 物理蒸馏训练策略

核心思想：
  - ViennaRNA 二级结构 = 物理先验（能量最低原理）
  - SimRNA 粗粒度 3D 结构 = 低分辨率几何约束
  - 深度学习模型作为"学生"，物理软件作为"教师"

损失函数设计：
  1. 接触图蒸馏损失（Contact Map Distillation）
     L_contact = MSE(P_model, P_vienna)

  2. 配对概率 KL 散度
     L_kl = KL(P_vienna || P_model)

  3. 能量一致性损失
     L_energy = |E_model - E_vienna|

  4. 置信度加权（高置信区域权重大）
     L_total = Σ confidence_i × L_i

优势：
  - 无需真实 3D 结构标签
  - 物理规律是"绝对真理"
  - 碱基配对关系确定性高
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False


@dataclass
class PhysicsDistillationConfig:
    """物理蒸馏配置"""
    # 蒸馏权重
    w_contact: float = 1.0          # 接触图损失权重
    w_kl: float = 0.5               # KL 散度损失权重
    w_energy: float = 0.3           # 能量一致性损失权重
    w_closure: float = 2.0          # BSJ 闭合损失权重

    # 温度参数
    temperature_sharpen: float = 0.5  # 锐化温度（越低越自信）
    temperature_softmax: float = 1.0  # Softmax 温度

    # 置信度阈值
    confidence_threshold: float = 0.7  # 低于此阈值的配对不参与蒸馏

    # 物理参数
    bond_length: float = 5.9        # Å, P-P 骨架距离
    pair_distance: float = 10.6     # Å, WC C1'-C1' 距离

    # 数据增强
    use_augmentation: bool = True
    n_augment_samples: int = 3      # 每个序列增强样本数


class ViennaRNATeacher:
    """ViennaRNA 教师模型：生成物理先验标签"""

    def __init__(self, use_partition_function: bool = True):
        self.use_pf = use_partition_function and HAS_VIENNA

    def predict_secondary_structure(
        self,
        sequence: str,
        circ_mode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        预测二级结构

        Returns:
            contact_map: (L, L) 接触概率矩阵
            dot_bracket: 二级结构点括号表示
            mfe: 最小自由能
            base_pairs: 配对列表 [(i, j, prob), ...]
        """
        L = len(sequence)

        if not HAS_VIENNA:
            # 回退：启发式配对
            return self._heuristic_pairing(sequence)

        try:
            # ViennaRNA circ 模式
            md = RNA.md()
            md.circ = circ_mode
            md.max_bp_span = L
            md.window_size = L

            fc = RNA.fold_compound(sequence, md)

            # MFE 结构
            structure, mfe = fc.mfe()

            # 配分函数（配对概率）
            if self.use_pf:
                fc.pf()

            # 构建接触图
            contact_map = torch.zeros(L, L, dtype=torch.float32)
            base_pairs = []

            # 解析点括号
            stack = []
            for pos, char in enumerate(structure):
                if char == '(':
                    stack.append(pos)
                elif char == ')' and stack:
                    i = stack.pop()
                    j = pos
                    # 从配分函数获取概率
                    if self.use_pf:
                        prob = fc.get_pair_probs(i + 1, j + 1)  # 1-indexed
                    else:
                        prob = 0.85  # MFE 默认高置信

                    contact_map[i, j] = prob
                    contact_map[j, i] = prob
                    base_pairs.append((i, j, prob))

            return {
                'contact_map': contact_map,
                'dot_bracket': structure,
                'mfe': mfe,
                'base_pairs': base_pairs,
            }

        except Exception as e:
            print(f"ViennaRNA failed: {e}")
            return self._heuristic_pairing(sequence)

    def _heuristic_pairing(self, sequence: str) -> Dict[str, torch.Tensor]:
        """启发式配对（无 ViennaRNA 时）"""
        L = len(sequence)
        contact_map = torch.zeros(L, L, dtype=torch.float32)
        base_pairs = []

        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        wobble = {'G': 'U', 'U': 'G'}

        for i in range(L):
            for j in range(i + 4, min(i + 20, L)):  # 窗口约束
                b1, b2 = sequence[i], sequence[j]

                if complement.get(b1) == b2:
                    prob = 0.7
                elif wobble.get(b1) == b2 or wobble.get(b2) == b1:
                    prob = 0.5
                else:
                    continue

                contact_map[i, j] = prob
                contact_map[j, i] = prob
                base_pairs.append((i, j, prob))

        return {
            'contact_map': contact_map,
            'dot_bracket': '',
            'mfe': 0.0,
            'base_pairs': base_pairs,
        }


class ContactMapDistillationLoss(nn.Module):
    """接触图蒸馏损失"""

    def __init__(self, config: PhysicsDistillationConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        pred_contact: torch.Tensor,      # (B, L, L) 模型预测
        teacher_contact: torch.Tensor,   # (B, L, L) ViennaRNA 教师标签
        confidence: Optional[torch.Tensor] = None,  # (B, L, L) 置信度
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失

        Args:
            pred_contact: 模型预测的接触概率
            teacher_contact: ViennaRNA 预测的接触概率
            confidence: 每个配对的置信度（可选）
        """
        B, L, _ = pred_contact.shape

        # 1. MSE 损失（接触图）
        mse_loss = F.mse_loss(pred_contact, teacher_contact, reduction='none')

        # 置信度加权
        if confidence is not None:
            mse_loss = mse_loss * confidence
            weight_sum = confidence.sum()
        else:
            # 自适应权重：高概率配对权重大
            weight = torch.where(
                teacher_contact > self.config.confidence_threshold,
                teacher_contact * 2.0,  # 高置信区域加权
                torch.ones_like(teacher_contact) * 0.1  # 低置信区域降权
            )
            mse_loss = mse_loss * weight
            weight_sum = weight.sum()

        contact_loss = mse_loss.sum() / (weight_sum + 1e-8)

        # 2. KL 散度损失（配对概率分布）
        # 锐化教师标签
        teacher_sharpen = self._sharpen(teacher_contact, self.config.temperature_sharpen)

        # 展平为 (B*L, L)
        pred_flat = pred_contact.view(B * L, L)
        teacher_flat = teacher_sharpen.view(B * L, L)

        kl_loss = F.kl_div(
            F.log_softmax(pred_flat / self.config.temperature_softmax, dim=-1),
            F.softmax(teacher_flat / self.config.temperature_softmax, dim=-1),
            reduction='batchmean',
        )

        return {
            'contact_loss': contact_loss,
            'kl_loss': kl_loss,
            'total_loss': (
                self.config.w_contact * contact_loss +
                self.config.w_kl * kl_loss
            ),
        }

    def _sharpen(self, probs: torch.Tensor, temperature: float) -> torch.Tensor:
        """锐化概率分布（温度越低越自信）"""
        if temperature == 1.0:
            return probs

        # 避免数值问题
        probs_clamped = probs.clamp(min=1e-8)
        sharpened = probs_clamped ** (1.0 / temperature)

        # 重新归一化
        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True)

        return sharpened


class EnergyConsistencyLoss(nn.Module):
    """能量一致性损失"""

    def __init__(self, config: PhysicsDistillationConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        pred_coords: torch.Tensor,      # (B, L, 3) 预测坐标
        pair_constraints: List[List[Tuple[int, int, float, float]]],
    ) -> torch.Tensor:
        """
        计算能量一致性损失

        Args:
            pred_coords: 预测的 3D 坐标
            pair_constraints: 配对约束 [(i, j, target_dist, weight), ...]
        """
        B, L, _ = pred_coords.shape
        device = pred_coords.device

        energy_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for b in range(B):
            if b >= len(pair_constraints):
                continue

            for (i, j, target_dist, weight) in pair_constraints[b]:
                if i >= L or j >= L:
                    continue

                # 预测距离
                pred_dist = torch.norm(pred_coords[b, i] - pred_coords[b, j])

                # 调和势能：E = k * (r - r0)^2
                # 这里用软惩罚
                dist_error = (pred_dist - target_dist) ** 2
                energy_loss = energy_loss + weight * dist_error
                n_pairs += 1

        if n_pairs > 0:
            energy_loss = energy_loss / n_pairs

        return energy_loss


class PhysicsDistillationTrainer:
    """物理蒸馏训练器"""

    def __init__(
        self,
        student_model: nn.Module,
        config: PhysicsDistillationConfig,
    ):
        self.student = student_model
        self.config = config
        self.teacher = ViennaRNATeacher()
        self.contact_loss_fn = ContactMapDistillationLoss(config)
        self.energy_loss_fn = EnergyConsistencyLoss(config)

    def generate_pseudo_labels(
        self,
        sequences: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """为序列生成物理先验伪标签"""
        pseudo_labels = []

        for seq in sequences:
            # ViennaRNA 预测二级结构
            ss_result = self.teacher.predict_secondary_structure(seq, circ_mode=True)

            # 构建配对约束
            pair_constraints = []
            for (i, j, prob) in ss_result['base_pairs']:
                if prob > self.config.confidence_threshold:
                    pair_constraints.append((
                        i, j,
                        self.config.pair_distance,
                        prob  # 权重 = 置信度
                    ))

            pseudo_labels.append({
                'contact_map': ss_result['contact_map'],
                'dot_bracket': ss_result['dot_bracket'],
                'mfe': ss_result['mfe'],
                'pair_constraints': pair_constraints,
            })

        return pseudo_labels

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        pseudo_labels: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """单步训练"""
        seq_tokens = batch['seq_ids']  # (B, L)
        B, L = seq_tokens.shape

        # 1. 学生模型前向
        student_out = self.student(seq_tokens)
        pred_coords = student_out['coords']          # (B, L, 3)
        pred_contact = student_out.get('pair_probs', torch.zeros(B, L, L, device=seq_tokens.device))

        # 2. 构建教师标签批次
        teacher_contact = torch.zeros(B, L, L, device=seq_tokens.device)
        pair_constraints_batch = []

        for b in range(B):
            if b < len(pseudo_labels):
                pl = pseudo_labels[b]
                pl_L = pl['contact_map'].shape[0]
                teacher_contact[b, :pl_L, :pl_L] = pl['contact_map'].to(seq_tokens.device)
                pair_constraints_batch.append(pl['pair_constraints'])

        # 3. 接触图蒸馏损失
        contact_loss_dict = self.contact_loss_fn(pred_contact, teacher_contact)

        # 4. 能量一致性损失
        energy_loss = self.energy_loss_fn(pred_coords, pair_constraints_batch)

        # 5. BSJ 闭合损失
        closure_dist = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
        closure_loss = ((closure_dist - self.config.bond_length) ** 2).mean()

        # 6. 总损失
        total_loss = (
            contact_loss_dict['total_loss'] +
            self.config.w_energy * energy_loss +
            self.config.w_closure * closure_loss
        )

        return {
            'total_loss': total_loss,
            'contact_loss': contact_loss_dict['contact_loss'],
            'kl_loss': contact_loss_dict['kl_loss'],
            'energy_loss': energy_loss,
            'closure_loss': closure_loss,
            'pred_coords': pred_coords,
        }


# ═══════════════════════════════════════════════════════════════
# 示例：如何集成到现有训练脚本
# ═══════════════════════════════════════════════════════════════

def train_with_physics_distillation(
    model: nn.Module,
    train_loader,
    args,
    device,
):
    """物理蒸馏训练主函数"""
    print("\n" + "="*60)
    print("  Training with Physics Distillation")
    print("  Teacher: ViennaRNA (thermodynamic equilibrium)")
    print("="*60)

    config = PhysicsDistillationConfig()
    trainer = PhysicsDistillationTrainer(model, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            # 生成伪标签
            sequences = batch.get('sequences', [])
            pseudo_labels = trainer.generate_pseudo_labels(sequences)

            # 训练步骤
            loss_dict = trainer.train_step(batch, pseudo_labels)

            # 反向传播
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_dict['total_loss'].item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    return model
