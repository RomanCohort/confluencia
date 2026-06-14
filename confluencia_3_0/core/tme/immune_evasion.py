"""免疫逃逸 (Immune Evasion)

模拟肿瘤免疫逃逸机制：
  - PD-L1上调（适应性抵抗：IFN-γ驱动）
  - MHC-I下调（基因组缺失或表观沉默）
  - TGF-β分泌（免疫抑制性细胞因子）
  - IDO活性（色氨酸耗竭抑制T细胞）
  - Galectin-9（诱导T细胞凋亡）
  - B7-H3（抑制T细胞共刺激）
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import TME_EVASION_UPDATE, IMMUNOTHERAPY_UPDATE
from ..config import EvasionConfig


class ImmuneEvasion:
    """免疫逃逸引擎"""

    def __init__(self, config: EvasionConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TME_EVASION_UPDATE, self._on_evasion, priority=0, name="evasion")
            self.bus.subscribe(IMMUNOTHERAPY_UPDATE, self._on_immunotherapy, priority=5, name="evasion")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步免疫逃逸更新"""
        pd_l1 = state.get("evs_pd_l1_expression", 0.2)
        mhc_i_down = state.get("evs_mhc_i_downreg", 0.1)
        tgf_beta = state.get("evs_tgf_beta", 0.15)
        ido = state.get("evs_ido_activity", 0.05)
        gal9 = state.get("evs_gal3_expression", 0.2)
        b7h3 = state.get("evs_b7_h3_expression", 0.3)

        # 驱动因素
        ifn_gamma = state.get("imm_ifn_gamma", 0.2)
        volume = state.get("tum_volume", 50.0)
        tumor_burden = min(1.0, volume / 500.0)

        # ===== PD-L1上调 =====
        # 适应性抵抗：IFN-γ驱动PD-L1表达上调
        # 本质性抵抗：肿瘤固有PD-L1高表达（某些亚型）
        pd_l1_adaptive = ifn_gamma * self.config.pd_l1_upregulation_rate
        pd_l1_constitutive = tumor_burden * 0.002  # 肿瘤负荷驱动
        new_pd_l1 = pd_l1 + pd_l1_adaptive + pd_l1_constitutive - 0.001  # 微小自然下降
        new_pd_l1 = max(0.05, min(1.0, new_pd_l1))

        # ===== MHC-I下调 =====
        # 基因组缺失或表观沉默，在免疫选择压力下增加
        immune_pressure = state.get("ied_immune_pressure", 0.5)
        mhc_i_drive = immune_pressure * self.config.mhc_i_downreg_rate
        new_mhc_i_down = mhc_i_down + mhc_i_drive - 0.001
        new_mhc_i_down = max(0.0, min(0.9, new_mhc_i_down))  # 不完全丧失

        # ===== TGF-β =====
        # 肿瘤细胞和CAF分泌
        caf_activation = state.get("caf_activation", 0.2)
        tgf_drive = (tumor_burden * 0.5 + caf_activation * 0.3) * self.config.tgf_beta_secretion_rate
        new_tgf_beta = tgf_beta + tgf_drive - 0.001
        new_tgf_beta = max(0.05, min(1.0, new_tgf_beta))

        # ===== IDO =====
        # IFN-γ诱导IDO表达
        ido_drive = ifn_gamma * self.config.ido_activation_rate
        new_ido = ido + ido_drive - 0.001
        new_ido = max(0.0, min(1.0, new_ido))

        # ===== Galectin-9 =====
        new_gal9 = gal9 + tumor_burden * 0.002 - 0.001
        new_gal9 = max(0.05, min(1.0, new_gal9))

        # ===== B7-H3 =====
        new_b7h3 = b7h3 + tumor_burden * 0.001 - 0.001
        new_b7h3 = max(0.05, min(1.0, new_b7h3))

        # 综合逃逸评分
        evasion_score = (
            new_pd_l1 * 0.3 +
            new_mhc_i_down * 0.25 +
            new_tgf_beta * 0.2 +
            new_ido * 0.1 +
            new_gal9 * 0.08 +
            new_b7h3 * 0.07
        )

        return {
            "evs_pd_l1_expression": new_pd_l1,
            "evs_mhc_i_downreg": new_mhc_i_down,
            "evs_tgf_beta": new_tgf_beta,
            "evs_ido_activity": new_ido,
            "evs_gal3_expression": new_gal9,
            "evs_b7_h3_expression": new_b7h3,
            "ied_evasion_pressure": evasion_score,
        }

    def _on_evasion(self, event) -> Dict[str, Any]:
        return {"evasion_updated": True}

    def _on_immunotherapy(self, event) -> Dict[str, Any]:
        """免疫治疗对逃逸的影响"""
        data = event.data
        therapy_type = data.get("therapy_type", "")
        if therapy_type == "anti_pd1":
            # 抗PD-1阻断PD-L1/PD-1交互
            return {"pd_l1_blockade": data.get("intensity", 0.5)}
        return {}