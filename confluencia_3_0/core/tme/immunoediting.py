"""免疫编辑三阶段模型 (Cancer Immunoediting)

三阶段: Elimination -> Equilibrium -> Escape
双向转换: 免疫治疗可将 Escape -> Equilibrium -> Elimination

临床现实:
- BLIS亚型: 低PD-L1但高免疫抑制(TGF-β, MDSC), 容易逃逸
- IM亚型: 高PD-L1, 高TIL, 在免疫治疗下可从逃逸回到消除
- 无治疗TNBC通常在数月内从消除进入逃逸
"""
from __future__ import annotations
from typing import Dict, Any


class Immunoediting:
    """免疫编辑三阶段模型"""

    PHASES = ["elimination", "equilibrium", "escape"]

    def __init__(self, config=None):
        self.config = config

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步免疫编辑更新"""
        phase = state.get("ied_phase", "elimination")
        progress = state.get("ied_phase_progress", 0.0)

        # === 计算免疫压力 ===
        cd8 = state.get("imm_cd8_count", 100)
        activation = state.get("imm_t_cell_activation", 0.3)
        exhaustion = state.get("imm_t_cell_exhaustion", 0.1)
        nk = state.get("imm_nk_count", 50)
        nk_cyto = state.get("imm_nk_cytotoxicity", 0.3)
        ifn_gamma = state.get("imm_ifn_gamma", 0.3)
        m1_frac = state.get("imm_m1_fraction", 0.5)

        # 免疫压力: 综合CD8激活、NK杀伤、M1极化
        immune_pressure = (
            cd8 * activation * (1 - exhaustion) * 0.0005  # CD8效应
            + nk * nk_cyto * 0.0003  # NK效应
            + ifn_gamma * 0.2  # IFN-γ促免疫
            + m1_frac * 0.1  # M1促免疫
        )

        # === 计算逃逸压力 ===
        pd_l1 = state.get("evs_pd_l1_expression", 0.15)
        mhc_i = state.get("evs_mhc_i_downreg", 0.1)
        tgf_beta = state.get("evs_tgf_beta", 0.2)
        ido = state.get("evs_ido_activity", 0.1)
        mdsc_sup = state.get("imm_mdsc_suppression", 0.1)
        treg_frac = state.get("imm_treg_fraction", 0.05)
        tumor_volume = state.get("tum_volume", 50.0)

        # 逃逸压力: 综合免疫抑制因子 + 肿瘤负荷
        # 肿瘤负荷越大，逃逸压力越高（临床现实：大肿瘤更易免疫逃逸）
        volume_factor = min(1.0, tumor_volume / 500.0)  # 500mm3以上饱和

        evasion_pressure = (
            pd_l1 * 0.3  # PD-L1免疫检查点
            + mhc_i * 0.25  # MHC-I下调
            + tgf_beta * 0.25  # TGF-β免疫抑制
            + ido * 0.15  # IDO色氨酸耗竭
            + mdsc_sup * 0.2  # MDSC抑制
            + treg_frac * 0.15  # Treg抑制
            + volume_factor * 0.3  # 肿瘤负荷驱动逃逸
        )

        # === 阶段转换 ===
        # 转换阈值（基于临床现实调整）
        # 免疫压力 > 逃逸压力 + margin -> 向消除方向移动
        # 逃逸压力 > 免疫压力 + margin -> 向逃逸方向移动
        ELIM_TO_EQUI = 0.02   # 消除->平衡阈值（低阈值：稍有失衡即转换）
        EQUI_TO_ESCA = 0.05   # 平衡->逃逸阈值
        ESCA_TO_EQUI = 0.08   # 逃逸->平衡阈值（需要更强的免疫压力）
        EQUI_TO_ELIM = 0.10   # 平衡->消除阈值

        new_phase = phase
        progress_rate = 0.05  # 每步进度变化率（加快转换速度）

        if phase == "elimination":
            # 消除阶段: 免疫占优
            # 如果逃逸压力升高，进入平衡
            if evasion_pressure > immune_pressure + ELIM_TO_EQUI:
                progress += progress_rate * (evasion_pressure - immune_pressure)
                if progress >= 1.0:
                    new_phase = "equilibrium"
                    progress = 0.0
            else:
                # 免疫反超时progress缓慢下降（但不完全重置）
                progress = max(0.0, progress - progress_rate * 0.1)

        elif phase == "equilibrium":
            # 平衡阶段: 免疫与逃逸大致相当
            if evasion_pressure > immune_pressure + EQUI_TO_ESCA:
                progress += progress_rate * (evasion_pressure - immune_pressure)
                if progress >= 1.0:
                    new_phase = "escape"
                    progress = 0.0
            elif immune_pressure > evasion_pressure + EQUI_TO_ELIM:
                progress += progress_rate * (immune_pressure - evasion_pressure)
                if progress >= 1.0:
                    new_phase = "elimination"
                    progress = 0.0
            else:
                # 保持平衡
                progress = max(0.0, min(1.0, progress))

        elif phase == "escape":
            # 逃逸阶段: 肿瘤逃避免疫
            # 免疫治疗可能逆转
            if immune_pressure > evasion_pressure + ESCA_TO_EQUI:
                progress += progress_rate * (immune_pressure - evasion_pressure)
                if progress >= 1.0:
                    new_phase = "equilibrium"
                    progress = 0.0
            else:
                progress = max(0.0, progress - progress_rate * 0.3)

        # === 阶段对免疫和肿瘤的反馈 ===
        # 消除阶段: 免疫增强，肿瘤受控
        # 平衡阶段: 免疫与逃逸拉锯
        # 逃逸阶段: 免疫被抑制，肿瘤失控
        if new_phase == "elimination":
            immune_modifier = 1.2  # 免疫增强20%
            growth_modifier = 0.85  # 生长减缓15%
        elif new_phase == "equilibrium":
            immune_modifier = 1.0
            growth_modifier = 1.0
        else:  # escape
            immune_modifier = 0.7  # 免疫抑制30%
            growth_modifier = 1.15  # 生长加速15%

        return {
            "ied_phase": new_phase,
            "ied_phase_progress": min(1.0, max(0.0, progress)),
            "ied_immune_pressure": immune_pressure,
            "ied_evasion_pressure": evasion_pressure,
            "ied_immune_modifier": immune_modifier,
            "ied_growth_modifier": growth_modifier,
        }
