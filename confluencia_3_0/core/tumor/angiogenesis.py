"""血管生成引擎 (Angiogenesis Engine)

模拟 VEGF 驱动的肿瘤血管生成和血管正常化窗口。

反馈环路:
  hypoxia → VEGF分泌 → 微血管密度增加 → 氧合改善 → VEGF下降
  抗VEGF药物 → 暂正常化窗口 → 灌注改善 → 药物递送增强
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TUMOR_ANGIOGENESIS
from ..config import AngiogenesisConfig


class AngiogenesisEngine:
    """血管生成引擎

    VEGF 由肿瘤细胞在缺氧条件下分泌，驱动微血管密度增加。
    抗 VEGGF 治疗（如贝伐珠单抗）创造短暂的血管正常化窗口，
    在此期间灌注改善，有利于药物递送。
    """

    def __init__(self, config: AngiogenesisConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TUMOR_ANGIOGENESIS, self._on_angiogenesis, priority=0, name="angiogenesis")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步血管生成更新"""
        volume = state.get("tum_volume", 50.0)
        carrying_capacity = 1000.0

        # 当前血管状态
        mvd = state.get("vasc_microvessel_density", 0.3)  # 微血管密度
        oxygenation = state.get("vasc_oxygenation", 0.7)
        perfusion = state.get("vasc_perfusion", 0.5)
        leakiness = state.get("vasc_leakiness", 0.3)
        normalization = state.get("vasc_normalization_window", 0.0)

        # 缺氧驱动 VEGF 分泌
        hypoxia = 1.0 - oxygenation
        tumor_factor = min(1.0, volume / carrying_capacity)  # 大肿瘤分泌更多VEGF
        vegf = hypoxia * tumor_factor * self.config.vegf_production_rate + 0.05  # 基线VEGF

        # VEGF 驱动微血管密度增加
        mvd_growth = vegf * self.config.vegf_threshold * 0.1
        mvd_decay = 0.01  # 血管自然退化
        new_mvd = mvd + mvd_growth - mvd_decay
        new_mvd = max(0.05, min(self.config.max_microvessel_density, new_mvd))

        # 氧合 = f(mvd, perfusion)
        # 高MVD和灌注 → 好氧合；肿瘤体积大 → 氧需求增加
        oxygen_demand = min(1.0, volume / 200.0)  # 小肿瘤氧需求低
        new_oxygenation = new_mvd * perfusion * 0.8 - oxygen_demand * 0.3
        new_oxygenation = max(0.0, min(1.0, new_oxygenation + 0.3))  # 基线氧合

        # 灌注受血管正常化窗口影响
        if normalization > 0:
            # 正常化窗口期间：灌注改善，渗漏减少
            new_perfusion = perfusion + normalization * 0.2
            new_leakiness = leakiness - normalization * 0.15
            normalization -= 0.05  # 正常化窗口逐渐消退
        else:
            new_perfusion = perfusion * 0.99  # 微小退化
            new_leakiness = leakiness + 0.002 * (1 - mvd)  # 不成熟血管渗漏

        new_perfusion = max(0.1, min(1.0, new_perfusion))
        new_leakiness = max(0.05, min(0.8, new_leakiness))
        normalization = max(0.0, min(1.0, normalization))

        return {
            "vasc_vegf_level": vegf,
            "vasc_microvessel_density": new_mvd,
            "vasc_oxygenation": new_oxygenation,
            "vasc_perfusion": new_perfusion,
            "vasc_leakiness": new_leakiness,
            "vasc_normalization_window": normalization,
        }

    def _on_angiogenesis(self, event) -> Dict[str, Any]:
        return {"angiogenesis_updated": True}

    def apply_anti_vegf(self, duration: float = 7.0) -> Dict[str, Any]:
        """应用抗VEGF治疗，开启血管正常化窗口"""
        return {"vasc_normalization_window": min(1.0, duration / self.config.normalization_duration)}