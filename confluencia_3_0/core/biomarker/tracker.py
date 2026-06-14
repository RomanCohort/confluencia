"""生物标志物追踪器 (Biomarker Tracker)

动态追踪 PD-L1 CPS、TIL密度、ctDNA、TMB 等标志物。
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import BIOMARKER_UPDATE


class BiomarkerTracker:
    """生物标志物追踪器"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(BIOMARKER_UPDATE, self._on_biomarker, priority=0, name="biomarker_tracker")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步标志物更新"""
        # PD-L1 CPS：与肿瘤PD-L1表达和IFN-γ相关
        pd_l1_expression = state.get("evs_pd_l1_expression", 0.2)
        ifn_gamma = state.get("imm_ifn_gamma", 0.2)
        pd_l1_cps = pd_l1_expression * 50 + ifn_gamma * 20  # CPS评分估算
        pd_l1_cps = max(0, min(100, pd_l1_cps))

        # TIL密度：直接从免疫状态读取
        til_density = state.get("imm_til_density", 0.2)

        # ctDNA：与肿瘤体积和转移负荷相关
        volume = state.get("tum_volume", 50.0)
        met_burden = state.get("met_metastatic_burden", 0.0)
        ctdna = min(1.0, (volume * 0.001 + met_burden * 0.01) * 0.5)
        # 治疗后ctDNA下降
        kill_fraction = state.get("drg_kill_fraction", 0.0)
        ctdna *= (1 - kill_fraction * 0.3)

        # TMB：随时间缓慢累积（治疗增加突变）
        tmb = state.get("bio_tmb", 5.0)
        drug_conc = state.get("drg_concentration", 0.0)
        tmb_increase = drug_conc * 0.0001  # 化疗增加突变负荷
        new_tmb = tmb + tmb_increase

        return {
            "bio_pd_l1_cps": pd_l1_cps,
            "bio_til_density": til_density,
            "bio_ctdna_level": ctdna,
            "bio_tmb": new_tmb,
        }

    def _on_biomarker(self, event) -> Dict[str, Any]:
        return {"biomarker_updated": True}