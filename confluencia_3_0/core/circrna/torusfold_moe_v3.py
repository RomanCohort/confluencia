"""
torusfold_moe_v3.py — MOE 路由与动态权重深度融合

将 TorusFold MOE 的 SeqTopK 路由机制集成到：
  1. 免疫评分：根据序列特征自动路由到不同免疫通路专家
  2. 四维目标权重：Gate 输出直接作为权重
  3. 序列进化：不同专家给出不同优化策略

版本演进：
  V1: 纯启发式（immune_sensing.py）
  V2: TorusFold 结构信号（immune_sensing_v2.py）
  V3: MOE 路由 + 结构信号 + 动态专家融合

架构：
  circRNA 序列
      ↓
  Feature Extractor (length, GC, dsRNA, ...)
      ↓
  SeqTopK Gate → Select K Experts
      ↓
  ┌──────────────────────────────────────┐
  │ Expert Pool (免疫通路 + 目标权重)     │
  │ - RIG-I Expert (dsRNA-focused)        │
  │ - TLR Expert (motif-focused)          │
  │ - PKR Expert (long-dsRNA-focused)     │
  │ - Stability Expert (BSJ-focused)      │
  │ - Translation Expert (IRES-focused)   │
  └──────────────────────────────────────┘
      ↓
  Confidence-weighted Fusion → Final Scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class MOEIntegratedConfig:
    """MOE + V2 集成配置。"""
    # 专家数量
    n_immunogenicity_experts: int = 4  # RIG-I, TLR7, TLR8, PKR
    n_objective_experts: int = 4       # Stability, Translation, Immune, Delivery
    top_k: int = 2                      # SeqTopK

    # 路由特征
    routing_features: List[str] = field(default_factory=lambda: [
        'seq_length',
        'gc_content',
        'dsRNA_fraction',      # ← TorusFold 信号
        'bsj_stability',       # ← TorusFold 信号
        'sasa_mean',           # ← TorusFold 信号
        'motif_count',
    ])

    # 融合模式
    fusion_mode: str = 'confidence_weighted'  # 使用专家置信度加权

    # 训练参数
    freeze_experts: bool = True          # 专家权重冻结（预训练）
    gate_lr: float = 1e-3
    load_balance_weight: float = 0.01

    # 启用 TorusFold
    use_torusfold: bool = True


# ═══════════════════════════════════════════════════════════════
# Expert Definitions
# ═══════════════════════════════════════════════════════════════

@dataclass
class ImmunogenicityExpertOutput:
    """单个免疫通路专家的输出。"""
    pathway: str           # "rig_i", "tlr7", "tlr8", "pkr"
    score: float           # [0, 1]
    confidence: float      # [0, 1]
    features_used: Dict[str, float]  # 该专家关注的关键特征
    rationale: str         # 为什么激活这个专家


class ImmunogenicityExpert(nn.Module):
    """单个免疫通路专家模块。

    每个专家专注于特定的 circRNA 特征：
      - RIG-I Expert: dsRNA structure, BSJ stability
      - TLR7 Expert: GU-rich motifs, uridine content
      - TLR8 Expert: AU-rich motifs, GUUG patterns
      - PKR Expert: long dsRNA (>33bp), modification
    """

    def __init__(self, pathway: str, d_input: int = 64):
        super().__init__()
        self.pathway = pathway

        # 专家的特征权重（可学习）
        self.feature_attention = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.ReLU(),
            nn.Linear(d_input // 2, d_input),
            nn.Softmax(dim=-1),  # 注意力权重
        )

        # 专家评分网络
        self.score_net = nn.Sequential(
            nn.Linear(d_input, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 置信度网络
        self.confidence_net = nn.Sequential(
            nn.Linear(d_input, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> ImmunogenicityExpertOutput:
        """计算专家评分。

        Args:
            features: (d_input) 融合特征向量

        Returns:
            ImmunogenicityExpertOutput
        """
        # 注意力加权
        attn_weights = self.feature_attention(features)  # (d_input)
        weighted_features = features * attn_weights

        # 评分
        score = self.score_net(weighted_features).squeeze().item()

        # 置信度
        confidence = self.confidence_net(weighted_features).squeeze().item()

        # 关键特征（根据 pathway）
        features_used = self._get_key_features(attn_weights.detach().cpu().numpy())

        # 理由
        rationale = self._generate_rationale(score, features_used)

        return ImmunogenicityExpertOutput(
            pathway=self.pathway,
            score=float(score),
            confidence=float(confidence),
            features_used=features_used,
            rationale=rationale,
        )

    def _get_key_features(self, attn_weights: np.ndarray) -> Dict[str, float]:
        """提取该专家关注的关键特征。"""
        feature_names = [
            'seq_length', 'gc', 'dsRNA', 'bsj_stab',
            'sasa', 'motif', 'ires_access', 'm6a_access'
        ]

        # 确保 attn_weights 长度匹配
        n_features = min(len(attn_weights), len(feature_names))

        # 取 top-3 高权重特征
        top_indices = np.argsort(attn_weights[:n_features])[-min(3, n_features):]

        return {feature_names[i]: float(attn_weights[i]) for i in top_indices if i < len(feature_names)}

    def _generate_rationale(self, score: float, features: Dict[str, float]) -> str:
        """生成专家激活理由。"""
        top_feature = max(features.keys(), key=lambda k: features[k])

        rationales = {
            'rig_i': f"dsRNA fraction high ({top_feature}) → RIG-I activation",
            'tlr7': f"GU-rich motifs detected ({top_feature}) → TLR7 sensing",
            'tlr8': f"AU-rich regions ({top_feature}) → TLR8 activation",
            'pkr': f"long dsRNA (>33bp) ({top_feature}) → PKR phosphorylation",
        }

        return rationales.get(self.pathway, f"pathway-specific activation ({top_feature})")


@dataclass
class ObjectiveWeightExpertOutput:
    """四维目标权重专家输出。"""
    weights: Dict[str, float]  # {stability, translation, immune, delivery}
    confidence: float
    rationale: str


class ObjectiveWeightExpert(nn.Module):
    """四维目标权重专家。

    不同场景需要不同的权重配置：
      - Short seq Expert: stability-focused
      - Long seq Expert: delivery-focused
      - High dsRNA Expert: immune_evasion-focused
      - High IRES access Expert: translation-focused
    """

    def __init__(self, expert_type: str, d_input: int = 64):
        super().__init__()
        self.expert_type = expert_type

        # 权重预测网络
        self.weight_net = nn.Sequential(
            nn.Linear(d_input, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 weights
        )

        # 置信度
        self.confidence_net = nn.Sequential(
            nn.Linear(d_input, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> ObjectiveWeightExpertOutput:
        """预测四维权重。"""
        # 原始权重
        raw_weights = self.weight_net(features).squeeze()  # (4)

        # 归一化（总和 = 1）
        weights = F.softmax(raw_weights, dim=0)

        # 置信度
        confidence = self.confidence_net(features).squeeze().item()

        # 转换为 dict
        weight_dict = {
            'stability': float(weights[0]),
            'translation': float(weights[1]),
            'immune_evasion': float(weights[2]),
            'delivery': float(weights[3]),
        }

        rationale = self._generate_rationale(weight_dict)

        return ObjectiveWeightExpertOutput(
            weights=weight_dict,
            confidence=float(confidence),
            rationale=rationale,
        )

    def _generate_rationale(self, weights: Dict[str, float]) -> str:
        """生成权重理由。"""
        top_dim = max(weights.keys(), key=lambda k: weights[k])

        rationales = {
            'short_seq': f"Short sequence → stability priority ({top_dim}: {weights[top_dim]:.2f})",
            'long_seq': f"Long sequence → delivery challenge ({top_dim}: {weights[top_dim]:.2f})",
            'high_dsRNA': f"High dsRNA → immune evasion critical ({top_dim}: {weights[top_dim]:.2f})",
            'high_ires': f"IRES exposed → translation priority ({top_dim}: {weights[top_dim]:.2f})",
        }

        return rationales.get(self.expert_type, f"adaptive weights ({top_dim}: {weights[top_dim]:.2f})")


# ═══════════════════════════════════════════════════════════════
# SeqTopK Routing (Integrated)
# ═══════════════════════════════════════════════════════════════

class IntegratedSeqTopKGating(nn.Module):
    """集成版 SeqTopK 路由。

    同时为免疫评分和目标权重选择专家。
    """

    def __init__(
        self,
        d_input: int = 64,
        n_immunogenicity_experts: int = 4,
        n_objective_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.n_imm_experts = n_immunogenicity_experts
        self.n_obj_experts = n_objective_experts
        self.top_k = top_k

        # 免疫评分 Gate
        self.imm_gate = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Linear(d_input, n_immunogenicity_experts),
        )

        # 目标权重 Gate
        self.obj_gate = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Linear(d_input, n_objective_experts),
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算两个路由。

        Args:
            features: (B, d_input) or (d_input)

        Returns:
            (imm_gate_logits, obj_gate_logits) or single sample
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        B = features.shape[0]

        # 免疫评分 gate
        imm_logits = self.imm_gate(features) / self.temperature  # (B, n_imm)

        # 目标权重 gate
        obj_logits = self.obj_gate(features) / self.temperature  # (B, n_obj)

        return imm_logits, obj_logits


# ═══════════════════════════════════════════════════════════════
# Feature Extractor (TorusFold-integrated)
# ═══════════════════════════════════════════════════════════════

class IntegratedFeatureExtractor(nn.Module):
    """集成 TorusFold 信号的特征提取器。

    V2: 从 torusfold_signals 直接提取
    V3: 结合 handcrafted + TorusFold + learned features
    """

    def __init__(self, d_out: int = 64):
        super().__init__()
        self.d_out = d_out

        # Learned encoder (optional)
        self.seq_embed = nn.Embedding(5, 16)
        self.pos_embed = nn.Embedding(2048, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 融合层：handcrafted(6) + TorusFold(5) + learned(32)
        self.fusion = nn.Sequential(
            nn.Linear(6 + 5 + 32, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(
        self,
        sequence: str,
        torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
    ) -> torch.Tensor:
        """提取路由特征。

        Args:
            sequence: circRNA 序列
            torusfold_signals: V2 扩展信号

        Returns:
            (d_out) 特征向量
        """
        L = len(sequence)

        # === Handcrafted features (6) ===
        gc = sum(1 for c in sequence.upper() if c in "GC") / max(L, 1)
        motif_count = sum(sequence.upper().count(m) for m in ["GUUG", "AUUA", "GUU"])

        handcrafted = torch.tensor([
            L / 1000.0,                # normalized length
            gc,                        # GC content
            motif_count / max(L/100, 1),  # motif density
            0.0 if torusfold_signals is None else torusfold_signals.dsRNA_fraction,
            0.5 if torusfold_signals is None else torusfold_signals.bsj_stability,
            0.5 if torusfold_signals is None else torusfold_signals.sasa_mean,
        ], dtype=torch.float32)

        # === TorusFold signals (5) ===
        if torusfold_signals and torusfold_signals.available:
            torusfeat = torch.tensor([
                torusfold_signals.dsRNA_fraction,
                torusfold_signals.bsj_stability,
                torusfold_signals.sasa_mean,
                torusfold_signals.motif_accessibility.get('ires', 0.5),
                torusfold_signals.motif_accessibility.get('m6a', 0.5),
            ], dtype=torch.float32)
        else:
            torusfeat = torch.zeros(5, dtype=torch.float32)

        # === Learned features (32) ===
        seq_ids = self._encode_sequence(sequence)
        seq_emb = self.seq_embed(seq_ids)  # (L, 16)
        pos = torch.arange(L).clamp(max=2047)
        pos_emb = self.pos_embed(pos)  # (L, 16)
        combined = torch.cat([seq_emb, pos_emb], dim=-1).transpose(0, 1).unsqueeze(0)  # (1, 32, L)
        learned = self.conv(combined).squeeze()  # (32)

        # === 融合 ===
        fused = torch.cat([handcrafted, torusfeat, learned], dim=-1)  # (6+5+32)
        return self.fusion(fused)  # (d_out)

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """序列编码。"""
        base_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        seq_ids = torch.tensor([base_map.get(c.upper(), 4) for c in sequence], dtype=torch.long)
        return seq_ids


# ═══════════════════════════════════════════════════════════════
# MOE Integrated Model
# ═══════════════════════════════════════════════════════════════

class TorusFoldMOEIntegrated(nn.Module):
    """TorusFold MOE 深度集成模型。

    V3 = MOE Routing + TorusFold Signals + Dynamic Fusion
    """

    def __init__(self, config: Optional[MOEIntegratedConfig] = None):
        super().__init__()
        self.config = config or MOEIntegratedConfig()

        # Feature Extractor
        self.feature_extractor = IntegratedFeatureExtractor(d_out=64)

        # SeqTopK Gating
        self.gating = IntegratedSeqTopKGating(
            d_input=64,
            n_immunogenicity_experts=self.config.n_immunogenicity_experts,
            n_objective_experts=self.config.n_objective_experts,
            top_k=self.config.top_k,
        )

        # Immunogenicity Experts
        self.imm_experts = nn.ModuleList([
            ImmunogenicityExpert(pathway=p, d_input=64)
            for p in ['rig_i', 'tlr7', 'tlr8', 'pkr']
        ])

        # Objective Weight Experts
        self.obj_experts = nn.ModuleList([
            ObjectiveWeightExpert(expert_type=t, d_input=64)
            for t in ['short_seq', 'long_seq', 'high_dsRNA', 'high_ires']
        ])

    def forward(
        self,
        sequence: str,
        torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
    ) -> Dict:
        """MOE 预测。

        Args:
            sequence: circRNA 序列
            torusfold_signals: V2 扩展信号

        Returns:
            {
                'immunogenicity': {pathway: score, ...},
                'objective_weights': {dim: weight, ...},
                'selected_experts': {'imm': [...], 'obj': [...]},
                'rationales': {...}
            }
        """
        # === 提取特征 ===
        features = self.feature_extractor(sequence, torusfold_signals)  # (64)

        # === SeqTopK Routing ===
        imm_logits, obj_logits = self.gating(features)  # (1, n_experts)

        # Select Top-K
        imm_topk_vals, imm_topk_idx = torch.topk(imm_logits.squeeze(), k=self.config.top_k)
        obj_topk_vals, obj_topk_idx = torch.topk(obj_logits.squeeze(), k=self.config.top_k)

        # Normalized weights
        imm_weights = F.softmax(imm_topk_vals, dim=0)
        obj_weights = F.softmax(obj_topk_vals, dim=0)

        # === 专家推理 ===
        # Immunogenicity
        imm_outputs = []
        for idx in imm_topk_idx:
            expert_out = self.imm_experts[idx](features)
            imm_outputs.append(expert_out)

        # Objective weights
        obj_outputs = []
        for idx in obj_topk_idx:
            expert_out = self.obj_experts[idx](features)
            obj_outputs.append(expert_out)

        # === 融合 ===
        # Immunogenicity: confidence-weighted fusion
        imm_scores = {}
        pathways = ['rig_i', 'tlr7', 'tlr8', 'pkr']

        # 首先初始化所有通路为 0
        for p in pathways:
            imm_scores[p] = 0.0

        total_conf = 0.0
        for i, expert_out in enumerate(imm_outputs):
            w = imm_weights[i].item()
            conf = expert_out.confidence
            pathway = expert_out.pathway

            # 累加（confidence weighted）
            imm_scores[pathway] = expert_out.score * w * conf
            total_conf += w * conf

        # 归一化
        if total_conf > 0:
            for p in pathways:
                imm_scores[p] /= total_conf

        # Overall immunogenicity
        overall_imm = sum(imm_scores.values()) / len(imm_scores)

        # Objective weights: weighted average
        obj_weights_final = {
            'stability': 0.0,
            'translation': 0.0,
            'immune_evasion': 0.0,
            'delivery': 0.0,
        }

        total_obj_conf = 0.0
        for i, expert_out in enumerate(obj_outputs):
            w = obj_weights[i].item()
            conf = expert_out.confidence

            for dim, weight in expert_out.weights.items():
                obj_weights_final[dim] += weight * w * conf

            total_obj_conf += w * conf

        # 归一化
        if total_obj_conf > 0:
            for dim in obj_weights_final:
                obj_weights_final[dim] /= total_obj_conf

        # === 返回结果 ===
        return {
            'immunogenicity': {
                'overall': float(overall_imm),
                'pathways': {p: float(imm_scores[p]) for p in pathways},
            },
            'objective_weights': obj_weights_final,
            'selected_experts': {
                'imm': [self.imm_experts[idx].pathway for idx in imm_topk_idx],
                'obj': [self.obj_experts[idx].expert_type for idx in obj_topk_idx],
            },
            'expert_outputs': {
                'imm': imm_outputs,
                'obj': obj_outputs,
            },
            'rationales': {
                'imm': [e.rationale for e in imm_outputs],
                'obj': [e.rationale for e in obj_outputs],
            },
            'gate_logits': {
                'imm': imm_logits.squeeze().detach().cpu().numpy(),
                'obj': obj_logits.squeeze().detach().cpu().numpy(),
            },
        }


# ═══════════════════════════════════════════════════════════════
# Quick Interface
# ═══════════════════════════════════════════════════════════════

def predict_with_moe_v3(
    sequence: str,
    torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
    use_torusfold: bool = True,
) -> Dict:
    """MOE V3 快速预测接口。"""
    model = TorusFoldMOEIntegrated()

    # 如果需要 TorusFold 信号但未提供，尝试获取
    if use_torusfold and torusfold_signals is None:
        try:
            from torusfold_scorer_v2 import extract_extended_signals
            # 启发式特征（无真实 coords）
            torusfold_signals = extract_extended_signals(
                sequence,
                coords=None,
                pair_probs=None,
            )
        except Exception:
            torusfold_signals = None

    return model(sequence, torusfold_signals)


def compute_adaptive_weights_moe_v3(
    sequence: str,
    torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
) -> Dict[str, float]:
    """MOE 动态四维权重。"""
    result = predict_with_moe_v3(sequence, torusfold_signals)
    return result['objective_weights']


def compute_immunogenicity_moe_v3(
    sequence: str,
    torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
) -> Dict[str, float]:
    """MOE 免疫原性评分。"""
    result = predict_with_moe_v3(sequence, torusfold_signals)
    return result['immunogenicity']['pathways']


# ═══════════════════════════════════════════════════════════════
# Evolution Integration
# ═══════════════════════════════════════════════════════════════

def evaluate_sequence_moe_v3(
    sequence: str,
    torusfold_signals: Optional["TorusFoldSignalsExtended"] = None,
    modification: str = "none",
) -> Tuple[float, Dict]:
    """MOE V3 序列评估（用于进化）。"""
    result = predict_with_moe_v3(sequence, torusfold_signals)

    weights = result['objective_weights']
    imm = result['immunogenicity']['pathways']

    # 四维评分
    stability = 0.3 + 0.4 * (1 - (torusfold_signals.bsj_closure / 10.0 if torusfold_signals else 0.5))
    translation = 0.4 + 0.3 * (torusfold_signals.motif_accessibility.get('ires', 0.5) if torusfold_signals else 0.5)
    immune_evasion = 1.0 - imm['overall']

    gc = sum(1 for c in sequence.upper() if c in "GC") / max(len(sequence), 1)
    delivery = 0.5 - 0.3 * abs(gc - 0.5)

    # Weighted total
    total = (
        weights['stability'] * stability +
        weights['translation'] * translation +
        weights['immune_evasion'] * immune_evasion +
        weights['delivery'] * delivery
    )

    return total, {
        'stability': stability,
        'translation': translation,
        'immune_evasion': immune_evasion,
        'delivery': delivery,
        'weights': weights,
        'selected_experts': result['selected_experts'],
        'rationales': result['rationales'],
    }