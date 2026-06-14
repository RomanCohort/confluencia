"""肿瘤异质性 (Tumor Heterogeneity)

模拟亚克隆演化、耐药突变和克隆竞争。

亚克隆模型:
  - 每个亚克隆有 fitness, fraction, resistance_level, mutation_signature
  - 每步：亚克隆按 fitness 竞争生长，突变产生新亚克隆
  - 药物压力下：耐药突变概率增加
  - Shannon 多样性指数追踪异质性
"""
import math
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from ..event_bus import EventBus, Event
from ..events import TUMOR_HETEROGENEITY, DRUG_RESISTANCE_EMERGED
from ..config import HeterogeneityConfig


@dataclass
class Subclone:
    """亚克隆"""
    id: int
    fitness: float = 1.0          # 相对适应度
    fraction: float = 0.25        # 占总肿瘤的比例
    resistance_level: float = 0.0  # 耐药水平 (0=敏感, 1=完全耐药)
    mutation_count: int = 0       # 突变数
    parent_id: int = -1           # 父克隆ID
    birth_step: int = 0           # 产生步数


class TumorHeterogeneity:
    """肿瘤异质性引擎

    维护亚克隆列表，追踪 Shannon 多样性指数和耐药克隆比例。
    """

    def __init__(self, config: HeterogeneityConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus
        self._rng = random.Random(42)

        # 初始亚克隆
        self._subclones: List[Subclone] = []
        dominant_fraction = 0.6
        remaining = 1.0 - dominant_fraction
        n_others = config.n_initial_subclones - 1

        # 优势克隆
        self._subclones.append(Subclone(
            id=0, fitness=1.0, fraction=dominant_fraction,
            resistance_level=0.0, mutation_count=0, parent_id=-1, birth_step=0
        ))

        # 其他亚克隆
        for i in range(1, config.n_initial_subclones):
            frac = remaining / n_others if n_others > 0 else remaining
            fitness = 0.8 + 0.2 * self._rng.random()  # 0.8-1.0
            self._subclones.append(Subclone(
                id=i, fitness=fitness, fraction=frac,
                resistance_level=0.0, mutation_count=self._rng.randint(1, 5),
                parent_id=0, birth_step=0
            ))

        self._next_id = config.n_initial_subclones

        if self.bus:
            self.bus.subscribe(DRUG_RESISTANCE_EMERGED, self._on_resistance, priority=0, name="heterogeneity")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步亚克隆演化"""
        # 药物压力增加耐药突变概率
        drug_concentration = state.get("drg_concentration", 0.0)
        drug_active = state.get("drg_active_drug", "")
        kill_fraction = state.get("drg_kill_fraction", 0.0)

        # 突变和选择
        new_subclones = []
        for sc in self._subclones:
            # 药物压力下的适应度修正
            if drug_active and sc.resistance_level < 0.5:
                # 敏感克隆在药物压力下适应度下降
                effective_fitness = sc.fitness * (1 - kill_fraction * 2.0)
            else:
                # 耐药克隆不受药物影响（甚至有优势）
                effective_fitness = sc.fitness * (1 + sc.resistance_level * 0.1)

            # 药物压力下现有克隆也累积耐药性（表观遗传适应）
            if drug_active and kill_fraction > 0.001:
                # 小概率逐步增加耐药水平（模拟表观遗传适应）
                if self._rng.random() < 0.02:
                    sc.resistance_level = min(1.0, sc.resistance_level + 0.02)

            # 突变产生新亚克隆
            # 药物压力大幅增加突变率（临床现实：化疗诱导基因组不稳定）
            mutation_prob = 0.01  # 基础突变概率1%/步（每亚克隆）
            if drug_active and kill_fraction > 0.001:
                mutation_prob *= (1 + kill_fraction * 50)  # 药物压力下突变率大幅增加

            if self._rng.random() < mutation_prob and len(self._subclones) + len(new_subclones) < self.config.max_subclones:
                # 耐药突变概率
                # 临床现实: 化疗选择压力下耐药突变概率显著增加
                if drug_active and kill_fraction > 0.001:
                    resistance_prob = 0.02 + kill_fraction * 0.3  # 2-15%
                else:
                    resistance_prob = 0.005  # 基础0.5%
                resistance_mutation = self._rng.random() < resistance_prob
                new_fitness = effective_fitness + self._rng.gauss(0, 0.1)
                new_fitness = max(0.1, min(2.0, new_fitness))
                new_resistance = sc.resistance_level + (0.5 if resistance_mutation else 0.05)
                new_resistance = min(1.0, new_resistance)

                new_sc = Subclone(
                    id=self._next_id,
                    fitness=new_fitness,
                    fraction=0.001,  # 新克隆起始很小
                    resistance_level=new_resistance,
                    mutation_count=sc.mutation_count + 1,
                    parent_id=sc.id,
                    birth_step=state.get("exp_step_count", 0),
                )
                self._next_id += 1
                new_subclones.append(new_sc)

        self._subclones.extend(new_subclones)

        # 基于适应度的竞争（比例重新分配）
        total_adjusted = 0.0
        adjusted_fitnesses = []
        for sc in self._subclones:
            # 药物压力下耐药克隆适应度更高
            if drug_active and kill_fraction > 0.001:
                # 敏感克隆被杀伤，耐药克隆存活
                adjusted = sc.fitness * (1 + sc.resistance_level * kill_fraction * 20)
            else:
                adjusted = sc.fitness
            adjusted_fitnesses.append(adjusted)
            total_adjusted += adjusted

        if total_adjusted > 0:
            for sc, adj in zip(self._subclones, adjusted_fitnesses):
                sc.fraction = adj / total_adjusted

        # 清除极小克隆（<0.001）
        self._subclones = [sc for sc in self._subclones if sc.fraction >= 0.001]

        # 重新归一化
        total_frac = sum(sc.fraction for sc in self._subclones)
        if total_frac > 0:
            for sc in self._subclones:
                sc.fraction /= total_frac

        # 计算多样性指数
        diversity = self._compute_shannon_diversity()

        # 耐药克隆比例（阈值0.2：resistance>=0.2视为耐药）
        resistance_fraction = sum(sc.fraction for sc in self._subclones if sc.resistance_level >= 0.2)

        # 优势克隆
        dominant = max(self._subclones, key=lambda sc: sc.fraction) if self._subclones else None

        return {
            "het_n_subclones": len(self._subclones),
            "het_diversity_index": diversity,
            "het_dominant_clone_fraction": dominant.fraction if dominant else 0,
            "het_resistance_clone_fraction": resistance_fraction,
            "drg_resistance_level": resistance_fraction,
        }

    def _compute_shannon_diversity(self) -> float:
        """计算 Shannon 多样性指数"""
        if not self._subclones:
            return 0.0
        H = 0.0
        for sc in self._subclones:
            if sc.fraction > 0:
                H -= sc.fraction * math.log(sc.fraction)
        return H

    def _on_resistance(self, event: Event) -> Dict[str, Any]:
        """耐药事件处理"""
        return {"resistance_detected": True}

    def get_subclones(self) -> List[Subclone]:
        """获取当前亚克隆列表"""
        return self._subclones