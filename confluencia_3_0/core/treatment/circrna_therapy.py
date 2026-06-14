"""circRNA治疗引擎 (CircRNA Therapy Engine)

三种机制:
  1. miRNA海绵: 抑制致癌miRNA -> 减缓生长
  2. 蛋白编码: 编码抑癌蛋白 -> 直接杀伤
  3. 免疫刺激: 激活RIG-I/TLR -> IFN-γ增强免疫

增强：当 pk_backend=="rnactm" 时，使用内化 RNACTM 替代硬编码衰减模型。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class CircRNATherapy:
    """circRNA治疗方案"""
    mechanism: str  # mirna_sponge / protein_coding / immune_stimulation
    dose: float = 1.0
    target: str = ""
    efficacy: float = 0.0
    days_active: int = 0
    modification: str = "none"
    delivery_vector: str = "LNP_standard"


class CircRNATherapyEngine:
    """circRNA治疗引擎

    当 config.pk_backend == "rnactm" 时，使用内化 RNACTM 模型计算 PK 曲线，
    替代硬编码指数衰减模型，获得更精确的蛋白表达窗口和 RNA 半衰期。
    """

    def __init__(self, config=None):
        self.config = config
        self._therapies: List[CircRNATherapy] = []
        self._pk_cache: Dict[str, Any] = {}  # 缓存 PK 结果

    def add_therapy(self, mechanism: str, dose: float = 1.0, target: str = "",
                    modification: str = "none", delivery_vector: str = "LNP_standard"):
        """添加circRNA治疗方案"""
        self._therapies.append(CircRNATherapy(
            mechanism=mechanism, dose=dose, target=target,
            modification=modification, delivery_vector=delivery_vector,
        ))

    def _on_circrna(self, event_data: Dict[str, Any]):
        """处理CIRCRNA_THERAPY_UPDATE事件"""
        mechanism = event_data.get("mechanism", "mirna_sponge")
        dose = event_data.get("dose", 1.0)
        target = event_data.get("target", "")
        modification = event_data.get("modification", "none")
        delivery_vector = event_data.get("delivery_vector", "LNP_standard")
        self.add_therapy(mechanism, dose, target, modification, delivery_vector)

    def _get_pk_decay(self, therapy: CircRNATherapy) -> float:
        """获取基于 RNACTM 的衰减因子。

        如果 pk_backend=="rnactm" 且内化模块可用，使用 RNACTM 计算的
        蛋白表达窗口和 RNA 半衰期来估计衰减。
        否则使用硬编码衰减模型。
        """
        circrna_cfg = getattr(self.config, 'circrna', None)
        if circrna_cfg is None or circrna_cfg.pk_backend != "rnactm":
            # 硬编码衰减 (原逻辑)
            return max(0.1, 1.0 / (1 + therapy.days_active * 0.005))

        try:
            from ..pk.rnactm import infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve

            # 缓存键
            cache_key = f"{therapy.modification}_{therapy.delivery_vector}_{therapy.dose}"
            if cache_key not in self._pk_cache:
                params = infer_rna_ctm_params(
                    modification=therapy.modification,
                    delivery_vector=therapy.delivery_vector,
                    route="IV",
                )
                curve = simulate_rna_ctm(
                    dose=therapy.dose, freq=1.0, params=params,
                    horizon=min(getattr(circrna_cfg, 'pk_default_horizon', 168), 168),
                )
                summary = summarize_rna_ctm_curve(curve)
                self._pk_cache[cache_key] = summary

            summary = self._pk_cache[cache_key]
            rna_half_life = summary.get("rna_ctm_rna_half_life_h", 6.0)
            expression_window = summary.get("rna_ctm_protein_expression_window_h", 48.0)

            # 基于 RNACTM 半衰期的衰减
            # circRNA 半衰期约 6h (未修饰) 到 18h (修饰后)
            # 转换为天: hours / 24
            half_life_days = max(rna_half_life / 24.0, 0.1)
            # 指数衰减: decay = 0.5^(days / half_life_days)
            decay = 0.5 ** (therapy.days_active / half_life_days)
            # 但蛋白表达窗口内效果较强，窗口外快速衰减
            if therapy.days_active > expression_window / 24.0:
                decay *= 0.5  # 表达窗口外额外衰减

            return max(0.05, decay)

        except Exception:
            # 降级到硬编码衰减
            return max(0.1, 1.0 / (1 + therapy.days_active * 0.005))

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步circRNA治疗更新"""
        if not self._therapies:
            return {}

        total_growth_reduction = 0.0
        total_kill = 0.0
        total_ifn_boost = 0.0
        total_immune_activation = 0.0

        for therapy in self._therapies:
            therapy.days_active += 1

            # 基于 RNACTM 的衰减 (替代硬编码)
            decay = self._get_pk_decay(therapy)
            base_efficacy = therapy.dose * decay

            if therapy.mechanism == "mirna_sponge":
                # miRNA海绵: 抑制致癌miRNA -> 减缓肿瘤生长
                growth_reduction = base_efficacy * 0.15
                total_growth_reduction += growth_reduction
                therapy.efficacy = growth_reduction

            elif therapy.mechanism == "protein_coding":
                # 蛋白编码: 编码抑癌蛋白(p53等) -> 直接诱导凋亡
                kill_rate = base_efficacy * 0.012
                total_kill += kill_rate
                therapy.efficacy = kill_rate

            elif therapy.mechanism == "immune_stimulation":
                # 免疫刺激: 激活RIG-I/TLR通路 -> IFN-γ增强
                ifn_boost = base_efficacy * 0.3
                immune_act = base_efficacy * 0.05
                total_ifn_boost += ifn_boost
                total_immune_activation += immune_act
                therapy.efficacy = ifn_boost

        return {
            "cfr_growth_reduction": total_growth_reduction,
            "cfr_kill_fraction": total_kill,
            "cfr_ifn_gamma_boost": total_ifn_boost,
            "cfr_immune_activation_boost": total_immune_activation,
            "cfr_active_therapies": len(self._therapies),
        }
