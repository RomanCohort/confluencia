"""联合治疗探索器 (Combination Therapy Explorer)

系统探索药物组合，计算协同指数。

参考模型:
  - Bliss独立性: E_AB = E_A + E_B - E_A * E_B
  - Loewe可加性: D_A/IC50_A + D_B/IC50_B = 1
  - 协同指数 CI < 1 (协同), CI = 1 (可加), CI > 1 (拮抗)
"""
from typing import Dict, Any, List, Tuple
from .sandbox import TNBCPharmacologySandbox, TreatmentArm, ExperimentMode


class CombinationTherapyExplorer:
    """联合治疗探索器"""

    def __init__(self):
        self._drug_pairs: List[Tuple[str, str]] = []
        self._results: List[Dict[str, Any]] = []

    def add_pair(self, drug_a: str, drug_b: str):
        """添加药物对"""
        self._drug_pairs.append((drug_a, drug_b))

    def run(self, n_days: int = 180, subtype: str = "BLIS") -> List[Dict[str, Any]]:
        """运行联合筛选"""
        self._results = []

        for drug_a, drug_b in self._drug_pairs:
            result = self._evaluate_pair(drug_a, drug_b, n_days, subtype)
            self._results.append(result)

        return self._results

    def _evaluate_pair(self, drug_a: str, drug_b: str,
                       n_days: int, subtype: str) -> Dict[str, Any]:
        """评估一对药物的协同性"""
        # 单药A
        sandbox_a = TNBCPharmacologySandbox()
        sandbox_a.add_arm(TreatmentArm(
            name=f"{drug_a}_alone",
            drugs=[{"name": drug_a, "dose": 60.0, "start_day": 0, "end_day": n_days}],
        ))
        result_a = sandbox_a.run(n_days, subtype)
        effect_a = max(0, -result_a[0].tumor_change_pct / 100.0) if result_a else 0

        # 单药B
        sandbox_b = TNBCPharmacologySandbox()
        sandbox_b.add_arm(TreatmentArm(
            name=f"{drug_b}_alone",
            drugs=[{"name": drug_b, "dose": 60.0, "start_day": 0, "end_day": n_days}],
        ))
        result_b = sandbox_b.run(n_days, subtype)
        effect_b = max(0, -result_b[0].tumor_change_pct / 100.0) if result_b else 0

        # 联合
        sandbox_ab = TNBCPharmacologySandbox()
        sandbox_ab.add_arm(TreatmentArm(
            name=f"{drug_a}+{drug_b}",
            drugs=[
                {"name": drug_a, "dose": 60.0, "start_day": 0, "end_day": n_days},
                {"name": drug_b, "dose": 60.0, "start_day": 0, "end_day": n_days},
            ],
        ))
        result_ab = sandbox_ab.run(n_days, subtype)
        effect_ab = max(0, -result_ab[0].tumor_change_pct / 100.0) if result_ab else 0

        # Bliss独立性参考
        bliss_expected = effect_a + effect_b - effect_a * effect_b

        # 协同指数
        if bliss_expected > 0:
            synergy_index = effect_ab / bliss_expected
        else:
            synergy_index = 1.0

        # 分类
        if synergy_index > 1.1:
            synergy_type = "synergistic"
        elif synergy_index < 0.9:
            synergy_type = "antagonistic"
        else:
            synergy_type = "additive"

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "effect_a": effect_a,
            "effect_b": effect_b,
            "effect_ab": effect_ab,
            "bliss_expected": bliss_expected,
            "synergy_index": synergy_index,
            "synergy_type": synergy_type,
        }