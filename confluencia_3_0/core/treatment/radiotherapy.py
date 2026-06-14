"""放疗引擎 (Radiotherapy Engine)

分次放疗和远隔效应 (Abscopal Effect)。

分次方案:
  - 常规分割: 2 Gy/次, 5次/周, 总量50-60 Gy
  - 大分割: 3-4 Gy/次
  - SBRT: 8-20 Gy/次
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import RADIOTHERAPY_UPDATE


class RadiotherapyEngine:
    """放疗引擎"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        self._active = False
        self._total_dose = 0.0       # 累积剂量 (Gy)
        self._dose_per_fraction = 2.0  # 每次剂量 (Gy)
        self._fractions_per_week = 5   # 每周次数
        self._prescribed_dose = 50.0   # 处方剂量 (Gy)
        self._abscopal_strength = 0.1  # 远隔效应强度

        if self.bus:
            self.bus.subscribe(RADIOTHERAPY_UPDATE, self._on_radiotherapy, priority=0, name="radiotherapy")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步放疗更新"""
        if not self._active:
            return {}

        result = {}

        # 每周给药 _fractions_per_week 次
        day_of_week = state.get("exp_step_count", 0) % 7
        is_fraction_day = day_of_week < self._fractions_per_week

        if is_fraction_day and self._total_dose < self._prescribed_dose:
            # 放疗杀伤
            # 线性二次模型: SF = exp(-α*D - β*D²)
            alpha = 0.3   # Gy⁻¹
            beta = 0.03   # Gy⁻²
            D = self._dose_per_fraction
            sf = math.exp(-alpha * D - beta * D * D) if D > 0 else 1.0
            kill_fraction = 1.0 - sf

            # CSC 对放疗有抗性
            csc_fraction = state.get("csc_fraction", 0.02)
            csc_rt_resistance = 3.0  # CSC放疗抗性倍数
            effective_kill = kill_fraction * (1 - csc_fraction * (1 - 1 / csc_rt_resistance))

            self._total_dose += D
            result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + effective_kill

            # 远隔效应：放疗诱导免疫激活
            ifn_gamma = state.get("imm_ifn_gamma", 0.2)
            abscopal = self._abscopal_strength * D * ifn_gamma
            result["imm_t_cell_activation"] = min(1.0,
                state.get("imm_t_cell_activation", 0.3) + abscopal * 0.01)
            result["imm_ifn_gamma"] = min(1.0,
                state.get("imm_ifn_gamma", 0.2) + abscopal * 0.02)

        # 检查是否完成
        if self._total_dose >= self._prescribed_dose:
            self._active = False

        return result

    def _on_radiotherapy(self, event) -> Dict[str, Any]:
        return {"radiotherapy_updated": True}

    def start_radiotherapy(
        self,
        prescribed_dose: float = 50.0,
        dose_per_fraction: float = 2.0,
        fractions_per_week: int = 5,
        abscopal_strength: float = 0.1,
    ):
        """开始放疗"""
        self._active = True
        self._total_dose = 0.0
        self._prescribed_dose = prescribed_dose
        self._dose_per_fraction = dose_per_fraction
        self._fractions_per_week = fractions_per_week
        self._abscopal_strength = abscopal_strength