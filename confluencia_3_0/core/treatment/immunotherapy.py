"""免疫治疗引擎 (Immunotherapy Engine)

实现检查点阻断、CAR-T 和细胞因子治疗。

治疗类型:
  - anti_pd1/anti_pd_l1: PD-1/PD-L1检查点阻断
  - car_t: 嵌合抗原受体T细胞
  - cytokine: IL-2, IFN-α 细胞因子治疗
"""
from typing import Dict, Any, Optional
from .drug_pipeline.pkpd import hill_equation
from .drug_pipeline.drug_registry import get_drug
from ..event_bus import EventBus, Event
from ..events import DRUG_ADMINISTERED, IMMUNOTHERAPY_UPDATE


class ImmunotherapyEngine:
    """免疫治疗引擎"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        self._active_therapies: Dict[str, Dict[str, Any]] = {}

        if self.bus:
            self.bus.subscribe(DRUG_ADMINISTERED, self._on_drug_administered, priority=1, name="immunotherapy")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步免疫治疗更新"""
        result = {}

        for therapy_name, therapy_info in self._active_therapies.items():
            therapy_type = therapy_info["type"]
            intensity = therapy_info.get("intensity", 0.5)

            if therapy_type == "anti_pd1":
                # PD-1/PD-L1 阻断
                # 效应：减少T细胞耗竭，增加激活
                pd_l1 = state.get("evs_pd_l1_expression", 0.2)
                blockade_efficiency = intensity * 0.7  # 阻断效率

                # 耗竭减少
                exhaustion_reduction = blockade_efficiency * pd_l1 * 0.1
                # 激活增加
                activation_boost = blockade_efficiency * 0.05

                result["imm_t_cell_exhaustion"] = max(0.0,
                    state.get("imm_t_cell_exhaustion", 0.1) - exhaustion_reduction)
                result["imm_t_cell_activation"] = min(1.0,
                    state.get("imm_t_cell_activation", 0.3) + activation_boost)

                # 免疫杀伤增加
                cd8_count = state.get("imm_cd8_count", 100.0)
                immune_kill_boost = cd8_count * activation_boost * 0.0001
                result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + immune_kill_boost

            elif therapy_type == "car_t":
                # CAR-T 直接杀伤
                antigen_expression = state.get("sub_ck5_6_expression", 0.5)  # 靶抗原
                car_t_count = therapy_info.get("car_t_count", 1000.0)
                killing = car_t_count * antigen_expression * 0.0001 * intensity
                result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + killing

            elif therapy_type == "cytokine_il2":
                # IL-2 扩增T细胞
                cd8 = state.get("imm_cd8_count", 100.0)
                cd4 = state.get("imm_cd4_count", 150.0)
                expansion = intensity * 0.02
                result["imm_cd8_count"] = cd8 * (1 + expansion)
                result["imm_cd4_count"] = cd4 * (1 + expansion * 0.5)

            elif therapy_type == "cytokine_ifn":
                # IFN-α 增强 M1 极化和抗原呈递
                m1_frac = state.get("imm_m1_fraction", 0.5)
                result["imm_m1_fraction"] = min(0.95, m1_frac + intensity * 0.02)
                result["imm_ifn_gamma"] = min(1.0,
                    state.get("imm_ifn_gamma", 0.2) + intensity * 0.05)

        return result

    def _on_drug_administered(self, event: Event) -> Dict[str, Any]:
        """给药事件处理"""
        data = event.data
        drug_name = data.get("drug_name", "").lower()

        drug_def = get_drug(drug_name)
        if drug_def is None or drug_def.drug_class != "immunotherapy":
            return {}

        # 确定免疫治疗类型
        if "pd" in drug_name or "atezolizumab" in drug_name or "pembrolizumab" in drug_name:
            therapy_type = "anti_pd1"
        else:
            therapy_type = "anti_pd1"  # 默认

        self._active_therapies[drug_name] = {
            "type": therapy_type,
            "intensity": 0.7,
            "definition": drug_def,
        }

        return {"immunotherapy_added": drug_name}

    def add_therapy(self, therapy_type: str, intensity: float = 0.7, **kwargs):
        """直接添加免疫治疗"""
        self._active_therapies[therapy_type] = {
            "type": therapy_type,
            "intensity": intensity,
            **kwargs,
        }