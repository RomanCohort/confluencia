"""肿瘤生长引擎 (Tumor Growth Engine)

实现 Logistic 和 Gompertz 生长模型，支持药物杀伤和 CSC 抗性。

模型:
  Logistic: dV/dt = r * V * (1 - V/K) - death * V - kill * V
  Gompertz: dV/dt = r * V * ln(K/V) - death * V - kill * V

其中:
  V = 肿瘤体积 (mm³)
  r = 生长率
  K = 携带容量 (mm³)
  death = 凋亡率
  kill = 药物/免疫杀伤率
"""
import math
from typing import Dict, Any, Optional
from ..event_bus import EventBus, Event
from ..events import TUMOR_GROWTH, STEP_START, DRUG_PD_EFFECT
from ..config import TumorConfig


class TumorGrowthEngine:
    """肿瘤生长引擎

    支持 Logistic/Gompertz 生长模型，药物杀伤分数直接减少体积。
    CSC 分数提供化疗抗性亚群（杀伤分数乘以 (1 - csc_fraction * csc_resistance)）。
    """

    def __init__(self, config: TumorConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.bus = event_bus

        # 内部状态
        self._volume = config.initial_volume
        self._growth_model = config.growth_model
        self._carrying_capacity = config.carrying_capacity

        # 订阅事件
        if self.bus:
            self.bus.subscribe(STEP_START, self._on_step_start, priority=0, name="growth_engine")
            self.bus.subscribe(DRUG_PD_EFFECT, self._on_drug_effect, priority=5, name="growth_engine")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """计算一步生长

        Args:
            state: 当前全局状态字典

        Returns:
            更新后的肿瘤状态键值
        """
        volume = state.get("tum_volume", self._volume)
        growth_rate = state.get("tum_growth_rate", self.config.growth_rate)
        apoptosis_rate = state.get("tum_apoptosis_rate", self.config.apoptosis_rate)
        carrying_capacity = self._carrying_capacity

        # 药物杀伤分数（来自化疗/免疫治疗）
        kill_fraction = state.get("drg_kill_fraction", 0.0)

        # CSC 抗性修正：CSC部分对化疗有抗性
        csc_fraction = state.get("csc_fraction", 0.02)
        csc_resistance = state.get("csc_chemo_resistance", 5.0)
        effective_kill = kill_fraction * (1.0 - csc_fraction * (1.0 - 1.0 / csc_resistance))

        # 缺氧修正：低氧降低生长率
        oxygenation = state.get("vasc_oxygenation", 0.7)
        hypoxia_factor = 0.5 + 0.5 * oxygenation  # 氧合0→0.5倍生长, 氧合1→1倍生长

        # circRNA治疗效应
        cfr_growth_reduction = state.get("cfr_growth_reduction", 0.0)  # miRNA海绵减缓生长
        cfr_kill = state.get("cfr_kill_fraction", 0.0)  # 蛋白编码直接杀伤

        # 免疫编辑生长修正
        ied_growth_modifier = state.get("ied_growth_modifier", 1.0)

        # 综合生长率修正
        growth_modifier = hypoxia_factor * (1 - cfr_growth_reduction) * ied_growth_modifier

        # 免疫杀伤（CD8+ T细胞和NK细胞）
        immune_kill = 0.0
        cd8_activation = state.get("imm_t_cell_activation", 0.3)
        cd8_count = state.get("imm_cd8_count", 100.0)
        nk_cytotoxicity = state.get("imm_nk_cytotoxicity", 0.3)
        nk_count = state.get("imm_nk_count", 50.0)

        # T细胞杀伤（受耗竭抑制）
        exhaustion = state.get("imm_t_cell_exhaustion", 0.1)
        immune_kill += cd8_count * cd8_activation * (1 - exhaustion) * 0.0001
        # NK杀伤（不受PD-L1抑制，但受MDSC抑制）
        mdsc_suppression = state.get("imm_mdsc_suppression", 0.1)
        immune_kill += nk_count * nk_cytotoxicity * (1 - mdsc_suppression) * 0.0001

        # 计算体积变化
        if self._growth_model == "logistic":
            # dV/dt = r * V * (1 - V/K) * modifier - death * V - kill * V - immune * V - cfr_kill * V
            dV = growth_rate * volume * (1 - volume / carrying_capacity) * growth_modifier \
                 - apoptosis_rate * volume \
                 - effective_kill * volume \
                 - immune_kill * volume \
                 - cfr_kill * volume
        elif self._growth_model == "gompertz":
            # dV/dt = r * V * ln(K/V) * modifier - death * V - kill * V - immune * V - cfr_kill * V
            if volume > 0 and carrying_capacity > 0:
                log_ratio = math.log(carrying_capacity / max(volume, 0.01))
                dV = growth_rate * volume * log_ratio * growth_modifier \
                     - apoptosis_rate * volume \
                     - effective_kill * volume \
                     - immune_kill * volume \
                     - cfr_kill * volume
            else:
                dV = 0
        else:  # exponential
            dV = growth_rate * volume * growth_modifier \
                 - apoptosis_rate * volume \
                 - effective_kill * volume \
                 - immune_kill * volume \
                 - cfr_kill * volume

        new_volume = max(0.0, volume + dV)

        # 坏死分数（体积超过阈值时）
        necrosis_threshold = self.config.necrosis_threshold
        necrosis_fraction = max(0, min(1, (new_volume - necrosis_threshold) / carrying_capacity))

        # 细胞数估算（1 mm³ ≈ 1e6 细胞）
        cell_count = new_volume * 1e6

        # 增殖指数
        net_growth_rate = dV / max(volume, 0.01)
        proliferation_index = max(0, min(1, net_growth_rate / 0.1))

        return {
            "tum_volume": new_volume,
            "tum_necrosis_fraction": necrosis_fraction,
            "tum_cell_count": cell_count,
            "tum_proliferation_index": proliferation_index,
            "tum_oxygenation": oxygenation,
        }

    def _on_step_start(self, event: Event) -> Dict[str, Any]:
        """STEP_START 事件处理"""
        return {"volume_at_start": self._volume}

    def _on_drug_effect(self, event: Event) -> Dict[str, Any]:
        """药物PD效应事件处理"""
        data = event.data
        kill_fraction = data.get("kill_fraction", 0.0)
        return {"kill_fraction_received": kill_fraction}

    def set_growth_model(self, model: str):
        """切换生长模型"""
        if model in ("logistic", "gompertz", "exponential"):
            self._growth_model = model