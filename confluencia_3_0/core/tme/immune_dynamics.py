"""免疫细胞动力学 (Immune Cell Dynamics)

模拟肿瘤微环境中的免疫细胞群体动态和癌症-免疫循环。

细胞类型:
  - CD8+ T细胞: 抗肿瘤效应细胞，受PD-L1/PD-1抑制导致耗竭
  - CD4+ T细胞: 辅助性T细胞，包含Th1(抗肿瘤)和Treg(免疫抑制)
  - NK细胞: 自然杀伤，不依赖MHC-I，受MDSC抑制
  - M1巨噬细胞: 抗肿瘤型，IFN-γ驱动极化
  - M2巨噬细胞: 促肿瘤型，IL-4/IL-13驱动极化
  - MDSC: 髓源性抑制细胞，抑制T和NK细胞
  - Treg: 调节性T细胞，抑制效应T细胞
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus, Event
from ..events import TME_IMMUNE_UPDATE, IMMUNOTHERAPY_UPDATE
from ..config import ImmuneConfig


class ImmuneCellDynamics:
    """免疫细胞动力学引擎

    模拟癌症-免疫循环中的关键细胞群体动态。
    """

    def __init__(self, config: ImmuneConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        if self.bus:
            self.bus.subscribe(TME_IMMUNE_UPDATE, self._on_immune_update, priority=0, name="immune")
            self.bus.subscribe(IMMUNOTHERAPY_UPDATE, self._on_immunotherapy, priority=0, name="immune")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步免疫细胞动态更新"""
        # 当前免疫状态
        cd8 = state.get("imm_cd8_count", 100.0)
        cd4 = state.get("imm_cd4_count", 150.0)
        activation = state.get("imm_t_cell_activation", 0.3)
        exhaustion = state.get("imm_t_cell_exhaustion", 0.1)
        nk = state.get("imm_nk_count", 50.0)
        nk_cyto = state.get("imm_nk_cytotoxicity", 0.3)
        m1_frac = state.get("imm_m1_fraction", 0.5)
        m2_frac = state.get("imm_m2_fraction", 0.5)
        macro = state.get("imm_macrophage_count", 80.0)
        treg = state.get("imm_treg_count", 20.0)
        mdsc = state.get("imm_mdsc_count", 30.0)

        # 肿瘤和逃逸状态
        volume = state.get("tum_volume", 50.0)
        pd_l1 = state.get("evs_pd_l1_expression", 0.2)
        mhc_i = 1.0 - state.get("evs_mhc_i_downreg", 0.1)  # MHC-I表达
        tgf_beta = state.get("evs_tgf_beta", 0.15)
        ido = state.get("evs_ido_activity", 0.05)

        # 免疫编辑修正: 逃逸阶段免疫抑制，消除阶段免疫增强
        ied_immune_mod = state.get("ied_immune_modifier", 1.0)

        # circRNA免疫刺激效应
        cfr_ifn_boost = state.get("cfr_ifn_gamma_boost", 0.0)
        cfr_immune_act_boost = state.get("cfr_immune_activation_boost", 0.0)

        # ===== CD8+ T细胞 =====
        # 激活：MHC-I抗原呈递
        activation_drive = mhc_i * self.config.cd8_activation_rate
        # 耗竭：PD-L1/PD-1交互 + 慢性刺激 + TGF-beta + 基线补充
        exhaustion_drive = (
            pd_l1 * self.config.t_cell_exhaustion_rate
            + (1 - activation) * 0.02  # 低激活促耗竭
            + tgf_beta * 0.04  # TGF-beta促耗竭
            + 0.005  # 基线耗竭补充（防止永远降到0）
        )
        # Treg抑制
        treg_suppression = treg / (treg + 100) * self.config.treg_suppression

        new_activation = activation + activation_drive - exhaustion_drive - treg_suppression + cfr_immune_act_boost
        new_activation = max(0.05, min(1.0, new_activation))

        # 耗竭恢复：高激活可部分逆转耗竭
        exhaustion_recovery = activation * 0.015
        new_exhaustion = exhaustion + exhaustion_drive - exhaustion_recovery
        new_exhaustion = max(0.05, min(0.95, new_exhaustion))  # 下限0.05（不会完全消除）

        # CD8数量变化：Logistic增长（容量限制）
        # 临床参考: 肿瘤浸润CD8+上限约500-1500/mm3
        cd8_capacity = state.get("imm_cd8_capacity", 800.0)
        cd8_growth_rate = new_activation * 0.008 * (1 - cd8 / cd8_capacity) * ied_immune_mod
        cd8_death_rate = new_exhaustion * 0.003 + 0.002  # 耗竭+自然衰减
        cd8_change = cd8 * (cd8_growth_rate - cd8_death_rate)
        new_cd8 = max(10.0, cd8 + cd8_change)

        # ===== NK细胞 =====
        # NK不依赖MHC-I，但受MDSC和TGF-β抑制
        nk_capacity = state.get("imm_nk_capacity", 300.0)
        nk_growth = 0.003 * (1 - nk / nk_capacity)
        nk_change = nk * (nk_growth - 0.001)
        new_nk = max(5.0, nk + nk_change)
        mdsc_suppression = mdsc / (mdsc + 100) * self.config.mdsc_suppression
        nk_inhibition = mdsc_suppression + tgf_beta * 0.3
        new_nk_cyto = nk_cyto * (1 - nk_inhibition * 0.1)
        new_nk_cyto = max(0.05, min(1.0, new_nk_cyto))

        # ===== 巨噬细胞极化 =====
        ifn_gamma = state.get("imm_ifn_gamma", 0.2)
        # M1极化：IFN-γ驱动
        m1_drive = ifn_gamma * self.config.m1_polarization_rate
        # M2极化：IL-10, TGF-β, 肿瘤因子驱动
        il10 = state.get("imm_il10", 0.1)
        m2_drive = (il10 + tgf_beta) * self.config.m2_polarization_rate

        new_m1_frac = m1_frac + m1_drive - m2_drive
        new_m2_frac = 1.0 - new_m1_frac
        new_m1_frac = max(0.05, min(0.95, new_m1_frac))
        new_m2_frac = max(0.05, min(0.95, new_m2_frac))

        # ===== 细胞因子 =====
        # IFN-γ: 由激活的T和NK细胞产生 + circRNA免疫刺激增强
        new_ifn_gamma = (new_cd8 * new_activation + nk * new_nk_cyto) * self.config.ifn_gamma_production * 0.01 + cfr_ifn_boost
        new_ifn_gamma = max(0.05, min(1.0, new_ifn_gamma))

        # IL-10: 由M2和Treg产生（免疫抑制）
        new_il10 = (new_m2_frac * macro * 0.001 + treg * 0.002) * 0.1
        new_il10 = max(0.05, min(1.0, new_il10))

        # TNF-α: 由M1产生
        new_tnf_alpha = new_m1_frac * macro * 0.001 * 0.1
        new_tnf_alpha = max(0.05, min(1.0, new_tnf_alpha))

        # ===== TIL密度 =====
        # TIL = f(CD8, CD4, NK) / 肿瘤体积
        total_lymphocytes = new_cd8 + cd4 + nk
        til_density = min(1.0, total_lymphocytes / (volume * 0.1 + 1))

        # ===== MDSC =====
        # 肿瘤因子和缺氧驱动MDSC扩增
        hypoxia = 1.0 - state.get("vasc_oxygenation", 0.7)
        mdsc_drive = (volume * 0.0001 + hypoxia * 0.01) * mdsc
        new_mdsc = mdsc + mdsc_drive * 0.01 - mdsc * 0.005  # 增殖-自然死亡
        new_mdsc = max(5.0, min(500.0, new_mdsc))

        return {
            "imm_cd8_count": new_cd8,
            "imm_t_cell_activation": new_activation,
            "imm_t_cell_exhaustion": new_exhaustion,
            "imm_nk_count": new_nk,
            "imm_nk_cytotoxicity": new_nk_cyto,
            "imm_m1_fraction": new_m1_frac,
            "imm_m2_fraction": new_m2_frac,
            "imm_ifn_gamma": new_ifn_gamma,
            "imm_il10": new_il10,
            "imm_tnf_alpha": new_tnf_alpha,
            "imm_til_density": til_density,
            "imm_mdsc_count": new_mdsc,
            "imm_mdsc_suppression": mdsc_suppression,
        }

    def _on_immune_update(self, event: Event) -> Dict[str, Any]:
        return {"immune_updated": True}

    def _on_immunotherapy(self, event: Event) -> Dict[str, Any]:
        """免疫治疗事件处理"""
        data = event.data
        therapy_type = data.get("therapy_type", "")
        if therapy_type == "anti_pd1":
            # 抗PD-1减少耗竭
            return {"exhaustion_reduction": data.get("intensity", 0.5)}
        return {}