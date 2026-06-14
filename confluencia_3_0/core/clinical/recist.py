"""RECIST 评估 (RECIST Tracker)

实体瘤反应评估标准 (Response Evaluation Criteria In Solid Tumors):
  - CR (Complete Response): 所有目标病灶消失
  - PR (Partial Response): 目标病灶直径总和减少 ≥30%
  - SD (Stable Disease): 既不满足PR也不满足PD
  - PD (Progressive Disease): 目标病灶直径总和增加 ≥20%
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import RECIST_EVALUATION


class RECISTTracker:
    """RECIST评估追踪器"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(RECIST_EVALUATION, self._on_recist, priority=0, name="recist")

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RECIST评估

        每6周（42天）进行一次正式评估。
        """
        current_volume = state.get("tum_volume", 50.0)
        baseline_volume = state.get("cli_baseline_volume", 50.0)
        nadir_volume = state.get("cli_nadir_volume", 50.0)

        # 肿瘤变化百分比（相对于基线）
        change_pct = ((current_volume - baseline_volume) / baseline_volume) * 100.0

        # RECIST分类
        if current_volume <= 0.01:  # 近乎消失
            response = "CR"
        elif change_pct <= -30.0:
            response = "PR"
        elif change_pct >= 20.0:
            # PD需要确认：相对于最低点增加20%且绝对增加5mm
            response = "PD"
        else:
            response = "SD"

        # 更新最低点
        new_nadir = min(nadir_volume, current_volume)

        return {
            "cli_recist_response": response,
            "cli_tumor_change_pct": change_pct,
            "cli_nadir_volume": new_nadir,
        }

    def _on_recist(self, event) -> Dict[str, Any]:
        return {"recist_evaluated": True}