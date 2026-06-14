"""化疗引擎 (Chemotherapy Engine)

实现细胞周期特异性和非特异性化疗药物效应。

细胞周期分类:
  - 周期特异性 (cell_cycle_specific): 紫杉类 (S/M期)
  - 周期非特异性 (cell_cycle_nonspecific): 蒽环类、铂类
"""
from typing import Dict, Any, Optional
from .drug_pipeline.pkpd import (
    PKPDParams, PKPDState, hill_equation,
    simulate_pkpd_step, infer_pkpd_params_from_drug,
)
from .drug_pipeline.drug_registry import get_drug, OncologyDrugDefinition
from ..event_bus import EventBus, Event
from ..events import DRUG_ADMINISTERED, DRUG_PK_UPDATE, DRUG_PD_EFFECT
from ..config import DrugPipelineConfig


# 细胞周期分类
CELL_CYCLE_SPECIFIC = {"paclitaxel", "docetaxel", "vinorelbine"}
CELL_CYCLE_NONSPECIFIC = {"doxorubicin", "carboplatin", "cisplatin", "cyclophosphamide"}


class ChemotherapyEngine:
    """化疗引擎

    管理化疗药物的 PK/PD 模拟和杀伤分数计算。
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        self._active_drugs: Dict[str, Dict[str, Any]] = {}  # {drug_name: {params, state, ...}}

        if self.bus:
            self.bus.subscribe(DRUG_ADMINISTERED, self._on_drug_administered, priority=0, name="chemotherapy")

    # kill_fraction缩放: Hill方程Emax代表相对效应强度(0-1)
    # 临床参考: 21天周期缩小30-50% -> 日均kill约0.015-0.025
    # 缩放因子将Hill effect映射到每日杀伤率(最大约0.04/day)
    KILL_SCALE = 0.04  # 最大每日杀伤4%

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步化疗更新"""
        total_kill = 0.0
        total_effect = 0.0
        total_concentration = 0.0
        active_drug_name = ""

        for drug_name, drug_info in self._active_drugs.items():
            pkpd_state = drug_info["state"]
            params = drug_info["params"]
            is_dose_time = drug_info.get("next_dose_day", 0) <= 0

            # PK更新（1天 = 24小时，分24步模拟）
            for h in range(24):
                is_dose = is_dose_time and h == 0
                pkpd_state = simulate_pkpd_step(pkpd_state, params, dt_h=1.0, is_dose_time=is_dose)

            drug_info["state"] = pkpd_state

            # PD效应 (Hill方程, 0-1范围)
            effect = hill_equation(pkpd_state.effect_conc, params.ec50, params.emax, params.hill)
            total_effect += effect

            # 将Hill效应缩放为每日杀伤率
            # effect=1.0 -> kill=0.04/day, effect=0.5 -> kill=0.02/day
            resistance = state.get("drg_resistance_level", 0.0)
            effective_kill = effect * self.KILL_SCALE * (1 - resistance)

            # CSC 抗性修正
            csc_fraction = state.get("csc_fraction", 0.02)
            csc_resistance = state.get("csc_chemo_resistance", 5.0)
            effective_kill *= (1 - csc_fraction * (1 - 1 / csc_resistance))

            total_kill += effective_kill
            total_concentration += pkpd_state.central_conc
            active_drug_name = drug_name

            # 更新下次给药日
            if is_dose_time:
                freq_days = drug_info.get("freq_days", 21)
                drug_info["next_dose_day"] = freq_days
            else:
                drug_info["next_dose_day"] = drug_info.get("next_dose_day", 1) - 1

        # 清除已完成的药物（浓度极低且无后续给药）
        to_remove = []
        for name, info in self._active_drugs.items():
            if info["state"].central_conc < 0.001 and info.get("next_dose_day", 0) < -100:
                to_remove.append(name)
        for name in to_remove:
            del self._active_drugs[name]

        return {
            "drg_kill_fraction": min(0.1, total_kill),  # 上限10%/day
            "drg_concentration": total_concentration,
            "drg_effect": min(1.0, total_effect),
            "drg_active_drug": active_drug_name,
        }

    def _on_drug_administered(self, event: Event) -> Dict[str, Any]:
        """给药事件处理"""
        data = event.data
        drug_name = data.get("drug_name", "").lower()
        dose = data.get("dose", 0.0)

        drug_def = get_drug(drug_name)
        if drug_def is None:
            return {"error": f"Unknown drug: {drug_name}"}

        if drug_def.drug_class != "chemo":
            return {}  # 非化疗药物，忽略

        # 推断PK/PD参数
        params = infer_pkpd_params_from_drug(drug_def)
        if dose > 0:
            params.dose_mg = dose * 1.7  # BSA调整

        # 初始化PK/PD状态
        pkpd_state = PKPDState()

        self._active_drugs[drug_name] = {
            "params": params,
            "state": pkpd_state,
            "freq_days": drug_def.frequency_days,
            "next_dose_day": 0,  # 立即给药
            "definition": drug_def,
        }

        return {"drug_added": drug_name, "dose": dose}

    def add_drug(self, drug_name: str, dose: float = 0.0):
        """直接添加化疗药物"""
        if self.bus:
            self.bus.publish(DRUG_ADMINISTERED, {
                "drug_name": drug_name,
                "dose": dose,
            }, source="chemotherapy")
        else:
            # 无EventBus时直接处理
            drug_def = get_drug(drug_name)
            if drug_def and drug_def.drug_class == "chemo":
                params = infer_pkpd_params_from_drug(drug_def)
                if dose > 0:
                    params.dose_mg = dose * 1.7
                self._active_drugs[drug_name.lower()] = {
                    "params": params,
                    "state": PKPDState(),
                    "freq_days": drug_def.frequency_days,
                    "next_dose_day": 0,
                    "definition": drug_def,
                }