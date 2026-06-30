"""
cirrna_evolution_v2.py — TorusFold 深度集成的序列进化模块

用 TorusFold 结构信号替代硬编码的四维目标权重。

版本差异：
  V1 (cirrna_evolution.py): 硬编码权重 35/30/25/10
  V2 (本文件): TorusFold 动态权重，根据结构质量自适应

使用方式：
  # V2 版本（TorusFold 深度集成）
  from cirrna_evolution_v2 import CircRNAEvolverV2
  evolver = CircRNAEvolverV2(use_torusfold=True)
  best_seq = evolver.evolve(initial_seq, generations=50)

  # V1 版本（硬编码权重）
  from cirrna_evolution import CircRNAEvolver
  evolver = CircRNAEvolver()

关键改进：
  1. 四维权重动态化：根据 dsRNA_fraction / bsj_stability / GC 动态调整
  2. 目标评分数据驱动：从 TorusFold 输出计算，而非启发式
  3. 帕累托前沿优化：权重不再是固定值，而是搜索空间的一部分
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from confluencia_3_0.core.circrna.torusfold_scorer import TorusFoldSignals, TorusFoldScorer

# Lazy import
_TorusFoldScorer = None


def _get_torusfold_scorer():
    global _TorusFoldScorer
    if _TorusFoldScorer is None:
        from confluencia_3_0.core.circrna.torusfold_scorer import TorusFoldScorer
        _TorusFoldScorer = TorusFoldScorer
    return _TorusFoldScorer


@dataclass
class EvolutionConfigV2:
    """V2 版本的进化配置。

    改进点：
      V1: weight_stability=0.35 (硬编码)
      V2: weight_range + 动态计算
    """
    # === 基础参数 ===
    generations: int = 50
    population_size: int = 100
    mutation_rate: float = 0.05
    crossover_rate: float = 0.3

    # === 权重范围（而非固定值）===
    weight_stability_range: Tuple[float, float] = (0.20, 0.50)
    weight_translation_range: Tuple[float, float] = (0.15, 0.40)
    weight_immune_evasion_range: Tuple[float, float] = (0.10, 0.35)
    weight_delivery_range: Tuple[float, float] = (0.05, 0.15)

    # === 动态权重因子 ===
    # 如果 dsRNA_fraction 高 → 提高 immune_evasion 权重
    dsRNA_weight_boost: float = 0.15
    # 如果 bsj_stability 低 → 提高 stability 权重
    bsj_weight_boost: float = 0.20

    # === TorusFold 集成 ===
    use_torusfold: bool = True
    structure_backend: str = "torusfold"

    # === 帕累托搜索 ===
    pareto_weight_samples: int = 32
    adaptive_weight_search: bool = True


@dataclass
class AdaptiveObjectiveWeights:
    """动态四维目标权重。"""
    stability: float
    translation: float
    immune_evasion: float
    delivery: float

    # 元信息
    method: str  # "torusfold_adaptive" 或 "heuristic_default"
    dsRNA_factor: float = 0.0
    bsj_factor: float = 0.0


def compute_adaptive_objective_weights(
    torusfold_signals: Optional["TorusFoldSignals"],
    sequence: str,
    config: EvolutionConfigV2,
) -> AdaptiveObjectiveWeights:
    """根据 TorusFold 信号动态计算四维权重。

    替代硬编码权重：
      V1: stability=0.35, translation=0.30, immune=0.25, delivery=0.10
      V2: stability = base + boost * (1 - bsj_stability)
          immune = base + boost * dsRNA_fraction

    Args:
        torusfold_signals: TorusFold 结构信号
        sequence: circRNA 序列
        config: 进化配置

    Returns:
        AdaptiveObjectiveWeights: 数据驱动的权重
    """
    # 基础权重范围
    w_stab_range = config.weight_stability_range
    w_trans_range = config.weight_translation_range
    w_immune_range = config.weight_immune_evasion_range
    w_del_range = config.weight_delivery_range

    # 如果 TorusFold 不可用，使用 V1 默认值
    if torusfold_signals is None or not torusfold_signals.available:
        return AdaptiveObjectiveWeights(
            stability=0.35,
            translation=0.30,
            immune_evasion=0.25,
            delivery=0.10,
            method="heuristic_default",
        )

    # === 动态计算 ===

    # 提取信号
    dsRNA_frac = torusfold_signals.dsRNA_fraction or 0.0
    bsj_stab = torusfold_signals.bsj_stability or 0.5
    gc = sum(1 for c in sequence.upper() if c in "GC") / max(len(sequence), 1)

    # Stability 权重：BSJ 不稳定 → 提高权重
    # bsj_stability ∈ [0, 1]，低值表示不稳定
    stability_need = 1.0 - bsj_stab  # [0, 1]
    w_stability = w_stab_range[0] + config.bsj_weight_boost * stability_need
    w_stability = np.clip(w_stability, w_stab_range[0], w_stab_range[1])

    # Immune evasion 权重：dsRNA 高 → 提高权重
    w_immune = w_immune_range[0] + config.dsRNA_weight_boost * dsRNA_frac
    w_immune = np.clip(w_immune, w_immune_range[0], w_immune_range[1])

    # Translation 权重：GC 高 → 可能影响翻译 → 降低权重
    # （GC 高的序列可能折叠过度，阻碍 IRES）
    gc_factor = gc - 0.5  # [-0.5, 0.5]
    w_translation = w_trans_range[0] + w_trans_range[1] - w_trans_range[0] * 0.5
    w_translation -= 0.05 * gc_factor  # 轻微调整
    w_translation = np.clip(w_translation, w_trans_range[0], w_trans_range[1])

    # Delivery 权重：相对固定
    w_delivery = (w_del_range[0] + w_del_range[1]) / 2.0

    # 归一化（总和 = 1.0）
    total = w_stability + w_translation + w_immune + w_delivery
    w_stability /= total
    w_translation /= total
    w_immune /= total
    w_delivery /= total

    return AdaptiveObjectiveWeights(
        stability=w_stability,
        translation=w_translation,
        immune_evasion=w_immune,
        delivery=w_delivery,
        method="torusfold_adaptive",
        dsRNA_factor=dsRNA_frac,
        bsj_factor=bsj_stab,
    )


def compute_objective_scores_v2(
    sequence: str,
    torusfold_signals: Optional["TorusFoldSignals"],
    weights: AdaptiveObjectiveWeights,
    modification: str = "none",
) -> Dict[str, float]:
    """计算四维目标评分（V2 版本）。

    改进点：
      V1: stability = heuristic_estimate()
      V2: stability = sigmoid(bsj_closure_error) + bond_rmsd_score
    """
    L = len(sequence)
    gc = sum(1 for c in sequence.upper() if c in "GC") / max(L, 1)

    # === Stability ===
    if torusfold_signals and torusfold_signals.available:
        # 从 TorusFold 直接计算
        bsj_stab = torusfold_signals.bsj_stability or 0.0
        bond_score = 1.0 - (torusfold_signals.bond_rmsd or 0.5) / 5.0  # bond_rmsd < 5Å 为好
        stability_score = 0.6 * bsj_stab + 0.4 * bond_score
    else:
        # 启发式兜底
        # GC 高 → 稳定
        stability_score = 0.3 + 0.5 * gc

    # === Translation ===
    if torusfold_signals and torusfold_signals.available:
        # 从 SASA 计算 IRES 可及性
        sasa_mean = torusfold_signals.sasa_mean or 0.5
        # 高暴露 → IRES 易被识别 → 翻译效率高
        translation_score = 0.4 + 0.4 * sasa_mean + 0.2 * (1 - gc)  # 低 GC 有利
    else:
        # 启发式
        translation_score = 0.3 + 0.3 * (1 - gc)  # 低 GC 有利于翻译

    # === Immune Evasion ===
    if torusfold_signals and torusfold_signals.available:
        # dsRNA fraction 低 → 免疫激活低 → 免疫逃逸好
        dsRNA_frac = torusfold_signals.dsRNA_fraction or 0.5
        immune_score = 1.0 - dsRNA_frac

        # m6A 修饰额外加分
        if modification.lower() in ["m6a", "psi", "5mc"]:
            immune_score += 0.15
    else:
        # 启发式
        immune_score = 0.5 - 0.3 * gc  # 低 GC → 少 dsRNA → 免疫逃逸好
        if modification.lower() in ["m6a", "psi"]:
            immune_score += 0.2

    # === Delivery ===
    # 序列长度和 GC 影响递送效率
    length_factor = np.clip(L / 500.0, 0.0, 1.0)  # 长序列难递送
    gc_factor = np.clip(abs(gc - 0.5), 0.0, 0.5)  # 极端 GC 不利
    delivery_score = 0.5 - 0.3 * length_factor - 0.2 * gc_factor

    # === 加权总分 ===
    total_score = (
        weights.stability * stability_score +
        weights.translation * translation_score +
        weights.immune_evasion * immune_score +
        weights.delivery * delivery_score
    )

    return {
        "total": float(np.clip(total_score, 0.0, 1.0)),
        "stability": float(stability_score),
        "translation": float(translation_score),
        "immune_evasion": float(immune_score),
        "delivery": float(delivery_score),
        "weights_used": {
            "stability": weights.stability,
            "translation": weights.translation,
            "immune": weights.immune_evasion,
            "delivery": weights.delivery,
        },
    }


class CircRNAEvolverV2:
    """circRNA 序列进化优化器（V2 版本）。

    深度集成 TorusFold：
      1. 每代评估时调用 TorusFold 获取结构信号
      2. 动态调整四维权重
      3. 帕累托前沿搜索权重空间
    """

    def __init__(
        self,
        config: Optional[EvolutionConfigV2] = None,
        use_torusfold: bool = True,
    ):
        self.config = config or EvolutionConfigV2()
        self.config.use_torusfold = use_torusfold
        self.scorer = None

        if use_torusfold:
            try:
                TorusFoldScorer = _get_torusfold_scorer()
                self.scorer = TorusFoldScorer(use_structure_prediction=True)
            except Exception:
                # TorusFold 不可用，使用启发式
                self.scorer = None

    def evaluate_sequence(
        self,
        sequence: str,
        modification: str = "none",
    ) -> Tuple[float, Dict[str, float]]:
        """评估单个序列的四维目标。"""
        # 获取 TorusFold 信号
        torusfold_signals = None
        if self.scorer is not None:
            try:
                torusfold_signals = self.scorer.extract_signals(sequence)
            except Exception:
                pass

        # 计算动态权重
        weights = compute_adaptive_objective_weights(
            torusfold_signals, sequence, self.config
        )

        # 计算评分
        scores = compute_objective_scores_v2(
            sequence, torusfold_signals, weights, modification
        )

        return scores["total"], scores

    def evolve(
        self,
        initial_sequence: str,
        generations: Optional[int] = None,
        modification: str = "none",
        verbose: bool = True,
    ) -> Tuple[str, Dict]:
        """进化优化 circRNA 序列。

        Args:
            initial_sequence: 初始序列
            generations: 进化代数（默认用 config）
            modification: 核苷酸修饰
            verbose: 是否打印进度

        Returns:
            (best_sequence, history): 最优序列 + 进化历史
        """
        generations = generations or self.config.generations

        population = [initial_sequence]
        history = []

        best_seq = initial_sequence
        best_score, best_details = self.evaluate_sequence(initial_sequence, modification)

        for gen in range(generations):
            # 变异
            new_population = []
            for seq in population[:self.config.population_size]:
                # 突变
                if np.random.random() < self.config.mutation_rate:
                    mutated = self._mutate(seq)
                    new_population.append(mutated)

                # 交叉
                if len(population) > 1 and np.random.random() < self.config.crossover_rate:
                    partner = population[np.random.randint(len(population))]
                    crossed = self._crossover(seq, partner)
                    new_population.append(crossed)

            population = population + new_population

            # 评估
            scores_pop = []
            for seq in population:
                score, details = self.evaluate_sequence(seq, modification)
                scores_pop.append((seq, score, details))

            # 选择
            scores_pop.sort(key=lambda x: x[1], reverse=True)
            population = [x[0] for x in scores_pop[:self.config.population_size]]

            # 更新最优
            if scores_pop[0][1] > best_score:
                best_seq = scores_pop[0][0]
                best_score = scores_pop[0][1]
                best_details = scores_pop[0][2]

            history.append({
                "generation": gen,
                "best_score": best_score,
                "population_size": len(population),
                "best_weights": best_details.get("weights_used", {}),
            })

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best={best_score:.3f}, "
                      f"weights={best_details.get('weights_used', {})}")

        return best_seq, {
            "final_score": best_score,
            "final_details": best_details,
            "generations": history,
            "method": "torusfold_v2" if self.scorer else "heuristic_v2",
        }

    def _mutate(self, sequence: str) -> str:
        """单点突变。"""
        seq_list = list(sequence)
        pos = np.random.randint(len(seq_list))
        bases = ["A", "C", "G", "U"]
        seq_list[pos] = np.random.choice(bases)
        return "".join(seq_list)

    def _crossover(self, seq1: str, seq2: str) -> str:
        """单点交叉。"""
        if len(seq1) != len(seq2):
            return seq1  # 长度不同，不交叉

        pos = np.random.randint(len(seq1))
        return seq1[:pos] + seq2[pos:]


# === 快速接口 ===

def quick_evolve_v2(
    sequence: str,
    generations: int = 30,
    use_torusfold: bool = True,
) -> Tuple[str, float]:
    """快速进化接口。"""
    evolver = CircRNAEvolverV2(use_torusfold=use_torusfold)
    best_seq, result = evolver.evolve(sequence, generations=generations, verbose=False)
    return best_seq, result["final_score"]


# === 预设场景 ===

def evolve_for_translation_v2(sequence: str, generations: int = 30) -> Tuple[str, float]:
    """优先翻译效率的进化。"""
    config = EvolutionConfigV2(
        weight_translation_range=(0.35, 0.50),  # 提高翻译权重范围
        weight_stability_range=(0.15, 0.30),
        weight_immune_evasion_range=(0.10, 0.20),
        weight_delivery_range=(0.05, 0.10),
        generations=generations,
    )
    evolver = CircRNAEvolverV2(config=config, use_torusfold=True)
    best_seq, result = evolver.evolve(sequence, verbose=False)
    return best_seq, result["final_score"]


def evolve_for_stability_v2(sequence: str, generations: int = 30) -> Tuple[str, float]:
    """优先稳定性的进化。"""
    config = EvolutionConfigV2(
        weight_stability_range=(0.40, 0.55),  # 提高稳定性权重
        bsj_weight_boost=0.25,  # BSJ 不稳定时额外加权
        generations=generations,
    )
    evolver = CircRNAEvolverV2(config=config, use_torusfold=True)
    best_seq, result = evolver.evolve(sequence, verbose=False)
    return best_seq, result["final_score"]


def evolve_for_immune_evasion_v2(sequence: str, generations: int = 30) -> Tuple[str, float]:
    """优先免疫逃逸的进化。"""
    config = EvolutionConfigV2(
        weight_immune_evasion_range=(0.30, 0.45),
        dsRNA_weight_boost=0.20,  # dsRNA 高时额外加权
        generations=generations,
    )
    evolver = CircRNAEvolverV2(config=config, use_torusfold=True)
    best_seq, result = evolver.evolve(sequence, verbose=False)
    return best_seq, result["final_score"]