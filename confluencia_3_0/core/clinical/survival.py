"""生存模型 (Survival Model)

PFS/OS 估计，基于当前肿瘤动态和治疗响应。

模型:
  - PFS: 从治疗开始到进展的时间
  - OS: PFS + 进展后生存时间
  - 风险率: f(肿瘤体积, 转移负荷, 毒性级别)
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import SURVIVAL_UPDATE


class SurvivalModel:
    """生存模型"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(SURVIVAL_UPDATE, self._on_survival, priority=0, name="survival")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步生存更新"""
        day = state.get("exp_step_count", 0) if "exp_step_count" in state else 0

        # PFS：如果已经PD，PFS就是PD发生的时间
        recist = state.get("cli_recist_response", "SD")
        if recist == "PD":
            pfs = day / 30.0  # 转换为月
        else:
            pfs = day / 30.0  # 当前无进展月数

        # OS估计：基于风险率
        volume = state.get("tum_volume", 50.0)
        met_burden = state.get("met_metastatic_burden", 0.0)
        toxicity = state.get("cli_toxicity_grade", 0)

        # 风险率（简化Weibull模型）
        hazard = 0.01 + volume * 0.00001 + met_burden * 0.001 + toxicity * 0.005
        # 治疗响应降低风险
        if recist in ("CR", "PR"):
            hazard *= 0.3
        elif recist == "SD":
            hazard *= 0.7

        # OS估计（简化）
        if hazard > 0:
            estimated_os = 1.0 / hazard + pfs  # 月
        else:
            estimated_os = pfs + 60.0  # 默认5年

        return {
            "cli_pfs_months": pfs,
            "cli_os_months": min(200.0, estimated_os),
        }

    def _on_survival(self, event) -> Dict[str, Any]:
        return {"survival_updated": True}