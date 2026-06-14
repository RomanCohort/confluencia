"""靶向治疗引擎 (Targeted Therapy Engine)

PARP抑制剂、AKT抑制剂、AR拮抗剂。

药物-标志物对应:
  - olaparib/rucaparib: BRCA1/2突变 (合成致死)
  - ipatasertib/capivasertib: PIK3CA/AKT1突变
  - enzalutamide: LAR亚型 (AR阳性)
"""
from typing import Dict, Any, Optional
from .drug_pipeline.pkpd import hill_equation
from .drug_pipeline.drug_registry import get_drug
from ..event_bus import EventBus, Event
from ..events import DRUG_ADMINISTERED


class TargetedTherapyEngine:
    """靶向治疗引擎"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        self._active_drugs: Dict[str, Dict[str, Any]] = {}

        if self.bus:
            self.bus.subscribe(DRUG_ADMINISTERED, self._on_drug_administered, priority=2, name="targeted")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步靶向治疗更新"""
        result = {}

        for drug_name, drug_info in self._active_drugs.items():
            drug_type = drug_info["type"]
            intensity = drug_info.get("intensity", 0.5)

            if drug_type == "parp_inhibitor":
                # PARP抑制剂：BRCA突变肿瘤中合成致死
                brca_status = state.get("bio_brca_status", 0)  # 0=WT, 1=BRCA1, 2=BRCA2
                hrd_status = state.get("bio_hr_status", 0)

                if brca_status > 0:
                    # BRCA突变：高效
                    sensitivity = 0.8
                elif hrd_status == 1:
                    # HRD阳性：中等
                    sensitivity = 0.5
                else:
                    # 野生型：低效
                    sensitivity = 0.1

                kill = sensitivity * intensity * 0.03
                result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + kill

            elif drug_type == "akt_inhibitor":
                # AKT抑制剂：PIK3CA/AKT1突变
                pi3k_mut = state.get("bio_pi3k_mutation", 0)

                if pi3k_mut == 1:
                    sensitivity = 0.6
                else:
                    sensitivity = 0.15

                kill = sensitivity * intensity * 0.02
                result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + kill

            elif drug_type == "ar_antagonist":
                # AR拮抗剂：LAR亚型
                ar_expression = state.get("bio_androgen_receptor", 0.1)
                subtype = state.get("sub_molecular_subtype", "BLIS")

                if subtype == "LAR" and ar_expression > 0.5:
                    sensitivity = 0.6
                else:
                    sensitivity = 0.05

                kill = sensitivity * intensity * 0.02
                result["drg_kill_fraction"] = state.get("drg_kill_fraction", 0.0) + kill

        return result

    def _on_drug_administered(self, event: Event) -> Dict[str, Any]:
        """给药事件处理"""
        data = event.data
        drug_name = data.get("drug_name", "").lower()

        drug_def = get_drug(drug_name)
        if drug_def is None or drug_def.drug_class != "targeted":
            return {}

        # 确定靶向类型
        if "olaparib" in drug_name or "rucaparib" in drug_name or "niraparib" in drug_name:
            drug_type = "parp_inhibitor"
        elif "ipatasertib" in drug_name or "capivasertib" in drug_name:
            drug_type = "akt_inhibitor"
        elif "enzalutamide" in drug_name or "bicalutamide" in drug_name:
            drug_type = "ar_antagonist"
        else:
            drug_type = "unknown"

        self._active_drugs[drug_name] = {
            "type": drug_type,
            "intensity": 0.7,
            "definition": drug_def,
        }

        return {"targeted_therapy_added": drug_name}

    def add_drug(self, drug_name: str, intensity: float = 0.7):
        """直接添加靶向药物"""
        drug_def = get_drug(drug_name)
        if drug_def:
            if "olaparib" in drug_name.lower():
                drug_type = "parp_inhibitor"
            elif "ipatasertib" in drug_name.lower():
                drug_type = "akt_inhibitor"
            elif "enzalutamide" in drug_name.lower():
                drug_type = "ar_antagonist"
            else:
                drug_type = "unknown"

            self._active_drugs[drug_name.lower()] = {
                "type": drug_type,
                "intensity": intensity,
                "definition": drug_def,
            }