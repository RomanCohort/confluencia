"""转移引擎 (Metastasis Engine)

模拟 EMT/MET 转化、肿瘤细胞扩散和器官趋向性转移。

模型:
  EMT: 上皮→间充质转化，由 TGF-β、缺氧和炎症驱动
  MET: 间充质→上皮逆转，在远处器官发生
  器官趋向性: "种子与土壤"理论，不同TNBC亚型有不同器官偏好
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TUMOR_METASTASIS
from ..config import MetastasisConfig


class MetastasisEngine:
    """转移引擎

    EMT 进度受 TGF-β、缺氧和炎症驱动。
    当 EMT 超过阈值时，肿瘤细胞开始扩散。
    扩散的细胞按器官趋向性分布到远处器官。
    """

    def __init__(self, config: MetastasisConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TUMOR_METASTASIS, self._on_metastasis, priority=0, name="metastasis")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步转移动态更新"""
        emt = state.get("met_emt_progress", 0.0)
        met = state.get("met_met_progress", 0.0)
        volume = state.get("tum_volume", 50.0)

        # EMT 驱动因素
        tgf_beta = state.get("evs_tgf_beta", 0.15)
        hypoxia = 1.0 - state.get("vasc_oxygenation", 0.7)
        inflammation = state.get("imm_tnf_alpha", 0.15) + state.get("imm_il6", 0.1)

        # EMT 进度增加
        emt_drive = self.config.emt_rate * (tgf_beta + hypoxia * 0.5 + inflammation * 0.3)
        new_emt = emt + emt_drive
        new_emt = max(0.0, min(1.0, new_emt))

        # MET 进度（远处器官的逆转）
        met_drive = self.config.met_rate * met * 0.1  # MET缓慢
        new_met = met + met_drive if met > 0 else 0
        new_met = max(0.0, min(1.0, new_met))

        # 扩散率：EMT超过阈值时开始扩散
        emt_threshold = 0.3
        if new_emt > emt_threshold:
            dissemination = self.config.dissemination_rate * volume * (new_emt - emt_threshold)
        else:
            dissemination = 0.0

        # 器官趋向性分布
        organ_burdens = {}
        total_metastatic = 0.0
        for organ, weight in self.config.organotropism.items():
            current = state.get(f"met_{organ}_burden", 0.0) if organ != "distant_lymph" else 0.0
            key = organ.replace("distant_lymph", "lymph")
            new_burden = current + dissemination * weight * 0.001  # 极小增量
            new_burden = max(0.0, new_burden)
            organ_burdens[f"met_{key}_burden"] = new_burden
            total_metastatic += new_burden

        # 转移灶数
        n_sites = sum(1 for v in organ_burdens.values() if v > 0.01)

        return {
            "met_emt_progress": new_emt,
            "met_met_progress": new_met,
            "met_dissemination_rate": dissemination,
            "met_metastatic_burden": total_metastatic,
            "met_n_metastatic_sites": n_sites,
            **organ_burdens,
        }

    def _on_metastasis(self, event) -> Dict[str, Any]:
        return {"metastasis_updated": True}