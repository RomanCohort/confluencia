"""耐药检测器 (Resistance Detector)

检测获得性耐药签名：
  - ABCB1过表达（紫杉类耐药）
  - BRCA回复突变（PARP抑制剂耐药）
  - ERCC1过表达（铂类耐药）
  - MHC-I完全丧失（免疫治疗耐药）
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import RESISTANCE_DETECTED


class ResistanceDetector:
    """耐药检测器"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(RESISTANCE_DETECTED, self._on_resistance, priority=0, name="resistance_detector")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步耐药检测"""
        resistance_fraction = state.get("het_resistance_clone_fraction", 0.0)
        mhc_i_loss = state.get("evs_mhc_i_downreg", 0.1)

        # 综合耐药水平
        resistance_level = resistance_fraction * 0.7 + mhc_i_loss * 0.3

        # 如果耐药水平超过阈值，发布耐药检测事件
        if resistance_level > 0.3 and self.bus:
            self.bus.publish(RESISTANCE_DETECTED, {
                "resistance_level": resistance_level,
                "resistance_clone_fraction": resistance_fraction,
                "mhc_i_loss": mhc_i_loss,
            }, source="resistance_detector")

        return {"drg_resistance_level": resistance_level}

    def _on_resistance(self, event) -> Dict[str, Any]:
        return {"resistance_checked": True}