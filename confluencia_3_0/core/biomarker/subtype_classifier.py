"""分子亚型分类器 (Molecular Subtype Classifier)

TNBC 四种分子亚型:
  - BLIS (Basal-Like Immune-Suppressed): 低免疫浸润，高基底标志物
  - IM (Immunomodulatory): 高免疫浸润，高PD-L1
  - M (Mesenchymal): 高EMT，高CAF激活
  - LAR (Luminal Androgen Receptor): AR阳性，低免疫

亚型影响:
  - 生长率
  - 免疫逃逸
  - 药物敏感性
  - 转移器官趋向性
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import SUBTYPE_RECLASSIFIED


# 亚型特征权重
SUBTYPE_FEATURES = {
    "BLIS": {
        "ck5_6": 0.8, "egfr": 0.7, "ar": 0.05,
        "pd_l1": 0.15, "til": 0.1, "emt": 0.1, "caf": 0.2,
    },
    "IM": {
        "ck5_6": 0.6, "egfr": 0.5, "ar": 0.1,
        "pd_l1": 0.5, "til": 0.6, "emt": 0.2, "caf": 0.2,
    },
    "M": {
        "ck5_6": 0.4, "egfr": 0.3, "ar": 0.1,
        "pd_l1": 0.2, "til": 0.2, "emt": 0.5, "caf": 0.4,
    },
    "LAR": {
        "ck5_6": 0.2, "egfr": 0.2, "ar": 0.8,
        "pd_l1": 0.1, "til": 0.15, "emt": 0.1, "caf": 0.2,
    },
}


class MolecularSubtypeClassifier:
    """分子亚型分类器"""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.bus = event_bus
        if self.bus:
            self.bus.subscribe(SUBTYPE_RECLASSIFIED, self._on_subtype, priority=0, name="subtype_classifier")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """一步亚型评估（每30天重新分类一次）"""
        # 大多数情况下亚型稳定，只在治疗压力下可能转变
        current_subtype = state.get("sub_molecular_subtype", "BLIS")
        subtype_progress = state.get("sub_subtype_progress", 1.0)

        # 治疗压力下亚型可能漂移
        drug_conc = state.get("drg_concentration", 0.0)
        if drug_conc > 0:
            # 治疗压力降低亚型稳定性
            subtype_progress -= drug_conc * 0.001
            subtype_progress = max(0.3, subtype_progress)

        # 低稳定性时可能重新分类
        if subtype_progress < 0.5:
            new_subtype = self._classify(state)
            if new_subtype != current_subtype:
                return {
                    "sub_molecular_subtype": new_subtype,
                    "sub_subtype_progress": 0.5,  # 新亚型起始稳定性低
                }

        # 稳定性自然恢复
        subtype_progress += 0.001
        subtype_progress = min(1.0, subtype_progress)

        return {"sub_subtype_progress": subtype_progress}

    def _classify(self, state: Dict[str, Any]) -> str:
        """基于当前状态特征分类亚型"""
        # 提取特征
        features = {
            "ck5_6": state.get("sub_ck5_6_expression", 0.5),
            "egfr": state.get("sub_egfr_expression", 0.4),
            "ar": state.get("bio_androgen_receptor", 0.1),
            "pd_l1": state.get("evs_pd_l1_expression", 0.2),
            "til": state.get("imm_til_density", 0.2),
            "emt": state.get("met_emt_progress", 0.0),
            "caf": state.get("caf_activation", 0.2),
        }

        # 计算与每个亚型的相似度
        scores = {}
        for subtype, weights in SUBTYPE_FEATURES.items():
            score = 0.0
            for feature, weight in weights.items():
                observed = features.get(feature, 0.0)
                score += weight * observed
            scores[subtype] = score

        # 选择最高分的亚型
        best_subtype = max(scores, key=scores.get)
        return best_subtype

    def _on_subtype(self, event) -> Dict[str, Any]:
        return {"subtype_updated": True}