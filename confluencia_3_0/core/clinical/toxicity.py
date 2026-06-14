"""毒性分级器 (Toxicity Grader)

CTCAE (Common Terminology Criteria for Adverse Events) 毒性分级。

级别:
  0: 无毒性
  1: 轻度
  2: 中度
  3: 重度/严重（需干预）
  4: 危及生命
  5: 死亡
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TOXICITY_UPDATE


class ToxicityGrader:
    """CTCAE毒性分级器"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(TOXICITY_UPDATE, self._on_toxicity, priority=0, name="toxicity")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步毒性评估"""
        drug_conc = state.get("drg_concentration", 0.0)
        drug_name = state.get("drg_active_drug", "")

        # 基于药物浓度的毒性计算
        # 简化模型：毒性级别与浓度成正比
        neutropenia = self._compute_neutropenia(state, drug_conc)
        cardiotoxicity = self._compute_cardiotoxicity(state, drug_name, drug_conc)
        neuropathy = self._compute_neuropathy(state, drug_name, drug_conc)
        fatigue = self._compute_fatigue(state, drug_conc)
        nausea = self._compute_nausea(state, drug_conc)

        # 最高毒性级别
        max_grade = max(neutropenia, cardiotoxicity, neuropathy, fatigue, nausea)

        # 3级以上毒性触发剂量减少
        treatment_days = state.get("cli_days_on_treatment", 0) + 1

        return {
            "cli_neutropenia_grade": neutropenia,
            "cli_cardiotoxicity_grade": cardiotoxicity,
            "cli_neuropathy_grade": neuropathy,
            "cli_fatigue_grade": fatigue,
            "cli_nausea_grade": nausea,
            "cli_toxicity_grade": max_grade,
            "cli_days_on_treatment": treatment_days,
        }

    def _compute_neutropenia(self, state: Dict, drug_conc: float) -> int:
        """中性粒细胞减少级别"""
        # 化疗药物常见毒性
        if drug_conc > 100:
            return 4
        elif drug_conc > 50:
            return 3
        elif drug_conc > 10:
            return 2
        elif drug_conc > 1:
            return 1
        return 0

    def _compute_cardiotoxicity(self, state: Dict, drug_name: str, drug_conc: float) -> int:
        """心脏毒性级别（蒽环类特有）"""
        if "doxorubicin" in drug_name.lower():
            if drug_conc > 50:
                return 3
            elif drug_conc > 10:
                return 2
            elif drug_conc > 1:
                return 1
        return 0

    def _compute_neuropathy(self, state: Dict, drug_name: str, drug_conc: float) -> int:
        """神经病变级别（紫杉类/铂类特有）"""
        if "paclitaxel" in drug_name.lower() or "cisplatin" in drug_name.lower():
            if drug_conc > 50:
                return 3
            elif drug_conc > 10:
                return 2
            elif drug_conc > 1:
                return 1
        return 0

    def _compute_fatigue(self, state: Dict, drug_conc: float) -> int:
        """疲劳级别"""
        if drug_conc > 100:
            return 3
        elif drug_conc > 20:
            return 2
        elif drug_conc > 5:
            return 1
        return 0

    def _compute_nausea(self, state: Dict, drug_conc: float) -> int:
        """恶心级别"""
        if drug_conc > 100:
            return 3
        elif drug_conc > 30:
            return 2
        elif drug_conc > 5:
            return 1
        return 0

    def _on_toxicity(self, event) -> Dict[str, Any]:
        return {"toxicity_updated": True}