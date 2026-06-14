"""癌干细胞池 (Cancer Stem Cell Pool)

模拟 CSC 自我更新、分化和化疗抗性。

模型:
  - CSC_fraction 变化: d(f)/dt = self_renewal * f - differentiation * f
  - CSC 对化疗有高抗性（由 csc_chemo_resistance_factor 控制）
  - CSC 可在治疗后再生肿瘤
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TUMOR_CSC_UPDATE, DRUG_PD_EFFECT
from ..config import CSCConfig


class CancerStemCellPool:
    """癌干细胞池

    CSC 是肿瘤内一小部分具有自我更新和分化能力的细胞，
    对化疗和放疗有天然抗性，是肿瘤复发的主要来源。
    """

    def __init__(self, config: CSCConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TUMOR_CSC_UPDATE, self._on_csc_update, priority=0, name="csc_pool")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步 CSC 动态更新"""
        csc_fraction = state.get("csc_fraction", self.config.initial_fraction)
        self_renewal = state.get("csc_self_renewal", self.config.self_renewal_rate)
        differentiation = state.get("csc_differentiation_rate", self.config.differentiation_rate)

        # 药物杀伤对 CSC 的影响（CSC 抗性更高）
        kill_fraction = state.get("drg_kill_fraction", 0.0)
        csc_resistance = state.get("csc_chemo_resistance", self.config.chemo_resistance_factor)
        csc_kill = kill_fraction / csc_resistance  # CSC 受杀伤更少

        # CSC 比例变化
        # 自我更新增加 CSC 比例，分化减少
        # 化疗杀伤非CSC细胞 → CSC比例相对增加
        net_change = (self_renewal - differentiation) * csc_fraction

        # 化疗后 CSC 比例相对上升（非CSC被杀伤更多）
        if kill_fraction > 0:
            relative_increase = kill_fraction * (1 - 1 / csc_resistance) * csc_fraction * 0.1
            net_change += relative_increase

        # CSC 自身被杀伤
        net_change -= csc_kill * csc_fraction * 0.5

        new_fraction = csc_fraction + net_change
        new_fraction = max(0.001, min(0.5, new_fraction))  # CSC比例有上下界

        # CD44/CD24 表达（CSC标志物）
        cd44 = 0.5 + 0.5 * new_fraction  # CSC比例越高，CD44越高
        cd24 = 0.5 - 0.3 * new_fraction  # CSC比例越高，CD24越低

        return {
            "csc_fraction": new_fraction,
            "csc_cd44_expression": cd44,
            "csc_cd24_expression": cd24,
        }

    def _on_csc_update(self, event) -> Dict[str, Any]:
        return {"csc_updated": True}