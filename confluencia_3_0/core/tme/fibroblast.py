"""肿瘤相关成纤维细胞 (CAF) 和 ECM

模拟 CAF 激活、ECM 重塑和物理屏障效应。

CAF 亚型:
  - myCAFs: 肌成纤维型，产生胶原和ECM
  - iCAFs: 炎症型，分泌细胞因子
  - apCAFs: 抗原呈递型，呈递抗原给T细胞

ECM效应:
  - 高密度ECM阻碍药物递送
  - 高硬度ECM促进肿瘤侵袭
  - 透明质酸增加组织间压
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TME_FIBROBLAST_UPDATE
from ..config import CAFConfig


class FibroblastActivation:
    """CAF激活和ECM重塑引擎"""

    def __init__(self, config: CAFConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TME_FIBROBLAST_UPDATE, self._on_fibroblast, priority=0, name="fibroblast")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步CAF/ECM动态更新"""
        caf_activation = state.get("caf_activation", 0.2)
        caf_count = state.get("caf_count", 50.0)
        ecm_density = state.get("caf_ecm_density", 0.3)
        ecm_stiffness = state.get("caf_ecm_stiffness", 0.3)
        collagen = state.get("caf_collagen_density", 0.4)
        hyaluronan = state.get("caf_hyaluronan", 0.3)

        # 驱动因素
        tgf_beta = state.get("evs_tgf_beta", 0.15)
        volume = state.get("tum_volume", 50.0)
        tumor_signal = min(1.0, volume / 200.0)

        # CAF激活：TGF-β和肿瘤因子驱动
        activation_drive = (tgf_beta * 0.5 + tumor_signal * 0.3) * self.config.activation_rate
        new_activation = caf_activation + activation_drive - 0.002  # 微小自然失活
        new_activation = max(0.05, min(1.0, new_activation))

        # CAF数量
        new_caf = caf_count * (1 + new_activation * 0.01 - 0.005)
        new_caf = max(10.0, min(500.0, new_caf))

        # ECM密度：CAF产生，酶降解
        ecm_production = new_caf * self.config.ecm_production_rate * 0.001
        ecm_degradation = self.config.ecm_degradation_rate * ecm_density
        new_ecm = ecm_density + ecm_production - ecm_degradation
        new_ecm = max(0.05, min(self.config.max_ecm_density, new_ecm))

        # 胶原密度
        new_collagen = collagen + caf_activation * 0.005 - 0.002
        new_collagen = max(0.05, min(1.0, new_collagen))

        # ECM硬度：与胶原密度正相关
        new_stiffness = 0.3 + 0.5 * new_collagen + 0.2 * new_ecm
        new_stiffness = max(0.1, min(1.0, new_stiffness))

        # 透明质酸
        new_hyaluronan = hyaluronan + new_activation * 0.003 - 0.001
        new_hyaluronan = max(0.05, min(1.0, new_hyaluronan))

        return {
            "caf_activation": new_activation,
            "caf_count": new_caf,
            "caf_ecm_density": new_ecm,
            "caf_ecm_stiffness": new_stiffness,
            "caf_collagen_density": new_collagen,
            "caf_hyaluronan": new_hyaluronan,
        }

    def _on_fibroblast(self, event) -> Dict[str, Any]:
        return {"fibroblast_updated": True}