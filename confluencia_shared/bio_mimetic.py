"""
Bio-Mimetic AIDD Architecture

脑科学仿生架构 for Drug Discovery:
1. Topology Pharmacophore Network - 拓扑药效团网络
2. Tissue-Specific Dynamic Attention - 动态微环境门控
3. Adversarial Synaptic Pruning - 对抗性突触修剪进化
4. Neuroplastic Closed-loop - 神经可塑性闭环
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════
# 1. 拓扑药效团网络 (Topology Pharmacophore Network)
# ═══════════════════════════════════════════════════════════════════════════════════


class PharmacophoreNodeType(Enum):
    """药效团节点类型"""
    HBD = "hydrogen_bond_donor"      # 氢键供体
    HBA = "hydrogen_bond_acceptor"   # 氢键受体
    HYD = "hydrophobic"                # 疏水区域
    ARO = "aromatic"                 # 芳香环
    POS = "positive_charge"           # 正电荷
    NEG = "negative_charge"          # 负电荷
    CAT = "cation"                    # 阳离子pi相互作用
    ANI = "anion"                    # 阴离子pi相互作用


@dataclass
class PharmacophoreNode:
    """药效团节点"""
    node_type: PharmacophoreNodeType
    position: np.ndarray  # 3D坐标
    importance: float = 1.0  # 节点重要性
    connections: List[int] = field(default_factory=list)

    def degree_centrality(self, total_nodes: int) -> float:
        """度中心性"""
        return len(self.connections) / max(total_nodes - 1, 1)


@dataclass
class PharmacophoreEdge:
    """药效团边"""
    source: int
    target: int
    weight: float
    interaction_type: Optional[str] = None


class TopologyPharmacophoreNetwork:
    """拓扑药效团网络

    将分子表示为药效团特征的无标度网络
    通过对比学习提取"药理拓扑骨架"

    Attributes:
        nodes: 药效团节点列表
        edges: 边列表
        topology_features: 拓扑特征
    """

    def __init__(
        self,
        decay_alpha: float = 1.5,
        distance_cutoff: float = 10.0,
    ) -> None:
        self.decay_alpha = decay_alpha
        self.distance_cutoff = distance_cutoff
        self.nodes: List[PharmacophoreNode] = []
        self.edges: List[PharmacophoreEdge] = []

    def build_from_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """从特征构建网络

        Args:
            features: 特征向量 (n_features,)
            feature_names: 特征名称列表
        """
        self.nodes.clear()
        self.edges.clear()

        # 特征类型映射
        type_mapping = {
            'hbd': PharmacophoreNodeType.HBD,
            'hba': PharmacophoreNodeType.HBA,
            'hydrophobic': PharmacophoreNodeType.HYD,
            'aromatic': PharmacophoreNodeType.ARO,
            'positive': PharmacophoreNodeType.POS,
            'negative': PharmacophoreNodeType.NEG,
        }

        # 从特征推断节点
        for i, (fname, fval) in enumerate(zip(feature_names, features)):
            fname_lower = fname.lower()

            for prefix, ptype in type_mapping.items():
                if prefix in fname_lower and fval > 0:
                    # 创建节点 (位置使用特征的一维表示)
                    position = np.array([i, fval, 0.0])
                    node = PharmacophoreNode(
                        node_type=ptype,
                        position=position,
                        importance=fval,
                    )
                    self.nodes.append(node)
                    break

        # 构建边 (基于特征索引的距离)
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                # 位置差异作为"距离"
                dist = abs(self.nodes[i].position[0] - self.nodes[j].position[0])

                if dist < self.distance_cutoff:
                    # 边权重 = 距离衰减
                    weight = 1.0 / (dist ** self.decay_alpha + 1e-6)
                    edge = PharmacophoreEdge(i, j, weight)
                    self.edges.append(edge)

                    # 更新节点的连接
                    self.nodes[i].connections.append(j)
                    self.nodes[j].connections.append(i)

    def get_topology_features(self) -> Dict[str, float]:
        """提取拓扑特征

        Returns:
            拓扑统计特征
        """
        if not self.nodes:
            return {}

        # 度分布
        degrees = [len(n.connections) for n in self.nodes]

        # 节点类型统计
        type_counts: Dict[str, int] = {}
        for node in self.nodes:
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # 计算特征
        features = {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'mean_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'degree_std': np.std(degrees) if degrees else 0,
            'network_density': len(self.edges) / max(len(self.nodes) * (len(self.nodes) - 1) / 2, 1),
        }

        # 添加类型分布
        for ptype in PharmacophoreNodeType:
            features[f'type_{ptype.value}'] = type_counts.get(ptype.value, 0)

        return features

    def get_embedding(self, dim: int = 64) -> np.ndarray:
        """生成拓扑嵌入向量

        Args:
            dim: 嵌入维度

        Returns:
            拓扑嵌入向量
        """
        topo_feats = self.get_topology_features()

        # 填充到固定维度
        embedding = np.zeros(dim, dtype=np.float32)
        keys = list(topo_feats.keys())
        for i, k in enumerate(keys[:dim]):
            embedding[i] = topo_feats[k]

        return embedding


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. 动态微环境门控 (Tissue-Specific Dynamic Attention)
# ═══════════════════════════════════════════════════════════════════════════════════


class TissueType(Enum):
    """组织类型"""
    LIVER = "liver"
    KIDNEY = "kidney"
    INTESTINE = "intestine"
    BRAIN = "brain"
    PLASMA = "plasma"


@dataclass
class PhysiologicalState:
    """生理状态

    患者特定的生理指标
    """
    liver_function: float = 0.8      # 肝功能 (0-1)
    kidney_function: float = 0.8     # 肾功能
    inflammation: float = 0.0        # 炎症水平
    ph: float = 7.4                  # 血液pH
    oxidative_stress: float = 0.0    # 氧化应激水平
    enzyme_activity: float = 0.5     # 代谢酶活性
    transporter_expression: float = 0.5  # 转运蛋白表达
    protein_binding: float = 0.9     # 蛋白结合率
    age: float = 0.5                 # 年龄 (归一化)
    bmi: float = 0.5                 # BMI (归一化)

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.liver_function,
            self.kidney_function,
            self.inflammation,
            self.ph - 7.4,  # 偏离度
            self.oxidative_stress,
            self.enzyme_activity,
            self.transporter_expression,
            self.protein_binding,
            self.age,
            self.bmi,
        ], dtype=np.float32)


class TissueSpecificAttention:
    """组织特异性动态注意力

    根据患者生理状态生成"门控权重"
    动态调制主模型对不同官能团的关注度

    Attributes:
        n_features: 特征维度
        tissue_type: 主要组织类型
    """

    def __init__(
        self,
        n_features: int,
        tissue_type: TissueType = TissueType.LIVER,
    ) -> None:
        self.n_features = n_features
        self.tissue_type = tissue_type

        # 默认注意力权重
        self.base_attention = np.ones(n_features) / n_features

    def get_dynamic_weights(
        self,
        phys_state: PhysiologicalState,
    ) -> np.ndarray:
        """获取动态门控权重

        Args:
            phys_state: 患者生理状态

        Returns:
            动态调制的注意力权重
        """
        # 基础注意力
        weights = self.base_attention.copy()

        # 根据生理状态的调制
        state_vec = phys_state.to_vector()

        # 1. 肝功能影响
        if self.tissue_type == TissueType.LIVER:
            if phys_state.oxidative_stress > 0.3:
                # 高氧化应激 → 提高对代谢敏感位的关注
                weights *= (1 + phys_state.oxidative_stress * 0.5)
            if phys_state.enzyme_activity < 0.5:
                # 低酶活性 → 关注代谢产物
                weights *= (1 + (0.5 - phys_state.enzyme_activity) * 0.3)

        # 2. 肾功能影响
        elif self.tissue_type == TissueType.KIDNEY:
            if phys_state.protein_binding < 0.8:
                # 低蛋白结合 → 关注游离药物清除
                weights *= (1 + (0.8 - phys_state.protein_binding) * 0.5)

        # 3. 肠道吸收
        elif self.tissue_type == TissueType.INTESTINE:
            if phys_state.ph < 6.0:
                # 低pH → 关注电离状态
                weights *= (1 + (6.0 - phys_state.ph) * 0.3)

        # 4. 全身暴露
        elif self.tissue_type == TissueType.PLASMA:
            if phys_state.protein_binding > 0.95:
                # 高蛋白结合 → 关注结合/游离平衡
                weights *= (1 + (phys_state.protein_binding - 0.95) * 0.5)

        # 炎症调制
        if phys_state.inflammation > 0.5:
            weights *= (1 + phys_state.inflammation * 0.2)

        # 归一化
        weights = weights / weights.sum()

        return weights

    def predict_admet_modulation(
        self,
        base_prediction: float,
        phys_state: PhysiologicalState,
    ) -> float:
        """预测ADMET调制

        Args:
            base_prediction: 基础预测值
            phys_state: 生理状态

        Returns:
            调制的预测值
        """
        weights = self.get_dynamic_weights(phys_state)

        # 加权调制因子
        modulation_factor = np.mean(weights) / np.mean(self.base_attention)

        # 根据组织类型调整
        if self.tissue_type == TissueType.LIVER:
            # 肝脏代谢
            if phys_state.oxidative_stress > 0.5:
                return base_prediction * 1.3  # 增强清除
            elif phys_state.enzyme_activity < 0.3:
                return base_prediction * 0.7  # 降低清除

        elif self.tissue_type == TissueType.KIDNEY:
            # 肾脏清除
            if phys_state.kidney_function < 0.5:
                return base_prediction * 0.5  # 大幅降低清除

        return base_prediction * modulation_factor


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. 对抗性突触修剪 (Adversarial Synaptic Pruning)
# ════��══════════════════════════════════════════════════════════════════════════════


@dataclass
class MoleculeCandidate:
    """分子候选"""
    smiles: str
    features: np.ndarray
    scores: Dict[str, float] = field(default_factory=dict)
    is_alive: bool = True

    def fitness(self) -> float:
        """综合适应度"""
        if not self.scores:
            return 0.0
        return np.mean(list(self.scores.values()))


@dataclass
class ParetoRecord:
    """帕累托档案记录"""
    candidate: MoleculeCandidate
    scores: np.ndarray

    def dominates(self, other: 'ParetoRecord') -> bool:
        """是否支配另一个解"""
        return all(self.scores >= other.scores) and any(self.scores > other.scores)


class AdversarialPruningOptimizer:
    """对抗性突触修剪优化器

    结合GAN和进化算法的思想:
    - 帕累托最优选择
    - 突触修剪: 淘汰差的分子

    Attributes:
        objectives: 优化目标列表
        n_population: 种群大小
    """

    def __init__(
        self,
        objectives: List[str] = None,
        n_population: int = 100,
    ) -> None:
        self.objectives = objectives or [
            'activity', 'solubility', 'permeability', 'metabolic_stability', 'toxicity'
        ]
        self.n_population = n_population

        # 帕累托档案
        self.pareto_front: List[ParetoRecord] = []

        # 历史最佳
        self.best_fitness_history: List[float] = []

    def optimize(
        self,
        evaluate_fn,
        n_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ) -> List[ParetoRecord]:
        """进化优化

        Args:
            evaluate_fn: 评估函数 (candidate) -> Dict[str, float]
            n_generations: 迭代次数
            mutation_rate: 变异率
            crossover_rate: 杂交率

        Returns:
            帕累托最优解列表
        """
        # 初始化种群 (这里应该调用分子生成器)
        population = self._init_population()

        for gen in range(n_generations):
            # 1. 评估所有候选
            for candidate in population:
                if candidate.is_alive:
                    candidate.scores = evaluate_fn(candidate)

            # 2. 识别帕累托最优
            current_pareto = self._identify_pareto_front(population)

            # 3. 更新帕累托档案
            self._update_pareto_front(current_pareto)

            # 4. 突触修剪 (淘汰差的候选)
            self._synaptic_pruning(population)

            # 5. 选择+交叉+变异生成新一代
            population = self._generate_next_generation(
                population, mutation_rate, crossover_rate
            )

            # 记录最佳适应度
            if current_pareto:
                best = max(r.candidate.fitness() for r in current_pareto)
                self.best_fitness_history.append(best)

                if gen % 10 == 0:
                    logger.info(f"Gen {gen}: {len(current_pareto)} Pareto, best fitness: {best:.3f}")

        return self.pareto_front

    def _init_population(self) -> List[MoleculeCandidate]:
        """初始化种群"""
        # 实际应该调用分子生成器
        return []

    def _identify_pareto_front(
        self,
        population: List[MoleculeCandidate],
    ) -> List[ParetoRecord]:
        """识别帕累托最优"""
        # 过滤活着的候选
        alive = [c for c in population if c.is_alive]

        if not alive:
            return []

        scores = np.array([[c.scores.get(o, 0) for o in self.objectives] for c in alive])

        n = len(alive)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # j支配i?
                    if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                        is_pareto[i] = False
                        break

        return [
            ParetoRecord(alive[i], scores[i])
            for i in range(n) if is_pareto[i]
        ]

    def _update_pareto_front(self, new_front: List[ParetoRecord]) -> None:
        """更新帕累托档案"""
        # 合并
        all_records = self.pareto_front + new_front

        if not all_records:
            return

        scores = np.array([r.scores for r in all_records])

        # 重新识别帕累托最优
        n = len(all_records)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j and is_pareto[i]:
                    if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                        is_pareto[i] = False
                        break

        self.pareto_front = [all_records[i] for i in range(n) if is_pareto[i]]

        # 限制大小
        if len(self.pareto_front) > self.n_population:
            # 按综合分数排序，保留最好的
            self.pareto_front.sort(
                key=lambda r: np.mean(r.scores), reverse=True
            )
            self.pareto_front = self.pareto_front[:self.n_population]

    def _synaptic_pruning(self, population: List[MoleculeCandidate]) -> None:
        """突触修剪

        淘汰差的候选:
        - 完全不满足目标 → 死亡
        - 偏科严重 → 降低权重
        """
        for candidate in population:
            if not candidate.is_alive:
                continue

            fitness = candidate.fitness()

            # 检查偏科程度
            scores = [candidate.scores.get(o, 0) for o in self.objectives]
            score_std = np.std(scores) if len(scores) > 1 else 0

            # 死亡条件
            if fitness < 0.1:
                # 严重不满足 → 死亡
                candidate.is_alive = False
            elif fitness < 0.3 and score_std > 0.4:
                # 偏科严重 → 大幅削弱
                pass  # 实际应该削弱该分子的表示

    def _generate_next_generation(
        self,
        population: List[MoleculeCandidate],
        mutation_rate: float,
        crossover_rate: float,
    ) -> List[MoleculeCandidate]:
        """生成下一代"""
        # 简化版本: 应该在真实实现中调用分子生成器
        alive = [c for c in population if c.is_alive]

        if not alive:
            return []

        # 精英选择
        n_keep = min(len(alive), self.n_population // 2)
        alive.sort(key=lambda c: c.fitness(), reverse=True)
        selected = alive[:n_keep]

        # 补充新分子 (实际应调用生成器)
        new_candidates = []  # placeholder

        return selected + new_candidates


# ═══════════════════════════════════════════════════════════════════════════════════
# 4. 神经可塑性闭环 (Neuroplastic Closed-loop)
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class ClinicalFeedback:
    """临床反馈"""
    patient_id: str
    genotype: Optional[str] = None
    metabolizer_status: Optional[str] = None
    predicted_outcome: float = 0.0
    actual_outcome: float = 0.0
    error: float = 0.0

    def __post_init__(self) -> None:
        self.error = self.actual_outcome - self.predicted_outcome


class NeuroplasticClosedLoop:
    """神经可塑性闭环系统

    当新的临床数据到来时:
    1. 不是简单更新参数
    2. 而是调整网络"厚度"和连接方式

    Attributes:
        adaptation_rate: 适应速率
        plasticity_threshold: 可塑性阈值
    """

    def __init__(
        self,
        adaptation_rate: float = 0.1,
        plasticity_threshold: float = 0.3,
        memory_size: int = 100,
    ) -> None:
        self.adaptation_rate = adaptation_rate
        self.plasticity_threshold = plasticity_threshold
        self.memory_size = memory_size

        # 历史反馈
        self.feedback_memory: List[ClinicalFeedback] = []

        # 层可塑性
        self.layer_plasticity: Dict[str, float] = {}

    def incorporate_feedback(
        self,
        feedback: ClinicalFeedback,
    ) -> Dict[str, Any]:
        """整合临床反馈

        Args:
            feedback: 临床反馈

        Returns:
            调整信息
        """
        # 存储反馈
        self.feedback_memory.append(feedback)

        # 限制大小
        if len(self.feedback_memory) > self.memory_size:
            self.feedback_memory.pop(0)

        # 分析误差
        adjustment = self._analyze_and_adjust(feedback)

        return adjustment

    def _analyze_and_adjust(self, feedback: ClinicalFeedback) -> Dict[str, Any]:
        """分析并调整

        Returns:
            调整信息
        """
        error = feedback.error
        abs_error = abs(error)

        adjustment = {
            'type': 'fine_tune',
            'magnitude': 0.0,
            'affected_layers': [],
        }

        if abs_error > self.plasticity_threshold * 2:
            # 大误差 → 结构可塑性
            adjustment['type'] = 'structural_plasticity'
            adjustment['magnitude'] = self.adaptation_rate * 1.5

            # 识别重要特征
            affected = self._identify_important_features(feedback)
            adjustment['affected_layers'] = affected

            # 更新可塑性权重
            for layer in affected:
                self.layer_plasticity[layer] = self.layer_plasticity.get(layer, 1.0) * 1.2

        elif abs_error > self.plasticity_threshold:
            # 中等误差 → 权重调整
            adjustment['type'] = 'weight_reweight'
            adjustment['magnitude'] = self.adaptation_rate

            affected = self._identify_important_features(feedback)
            adjustment['affected_layers'] = affected

        else:
            # 小误差 → 微调
            adjustment['type'] = 'fine_tune'
            adjustment['magnitude'] = self.adaptation_rate * 0.1

        # 归一化可塑性
        if self.layer_plasticity:
            total = sum(self.layer_plasticity.values())
            for k in self.layer_plasticity:
                self.layer_plasticity[k] /= total

        return adjustment

    def _identify_important_features(
        self,
        feedback: ClinicalFeedback,
    ) -> List[str]:
        """识别重要特征

        基于患者特征类型确定需要调整的层
        """
        affected = []

        if feedback.genotype:
            affected.append('genomic_features')

        if feedback.metabolizer_status:
            affected.append('metabolic_features')

        # 默认层
        if not affected:
            affected = ['default']

        return affected

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """获取适应摘要"""
        if not self.feedback_memory:
            return {'status': 'no_feedback'}

        errors = [f.error for f in self.feedback_memory]

        return {
            'n_feedback': len(self.feedback_memory),
            'mean_error': np.mean(errors),
            'error_trend': 'improving' if errors[-5:] < errors[:5] else 'stable',
            'layer_plasticity': dict(self.layer_plasticity),
        }


# ═══════════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # 1. 拓扑药效团网络
    'PharmacophoreNodeType',
    'PharmacophoreNode',
    'PharmacophoreEdge',
    'TopologyPharmacophoreNetwork',
    # 2. 动态微环境门控
    'TissueType',
    'PhysiologicalState',
    'TissueSpecificAttention',
    # 3. 对抗性突触修剪
    'MoleculeCandidate',
    'ParetoRecord',
    'AdversarialPruningOptimizer',
    # 4. 神经可塑性闭环
    'ClinicalFeedback',
    'NeuroplasticClosedLoop',
]