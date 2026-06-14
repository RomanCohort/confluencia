"""临床试验模拟器 (Clinical Trial Simulator)

Phase I/II/III 试验设计。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .sandbox import TNBCPharmacologySandbox, TreatmentArm, ArmResult, ExperimentMode


@dataclass
class TrialDesign:
    """试验设计"""
    phase: int = 2                     # I/II/III
    n_patients_per_arm: int = 30       # 每臂患者数
    duration_days: int = 365           # 试验持续时间
    arms: List[TreatmentArm] = field(default_factory=list)
    primary_endpoint: str = "recist"   # recist / pfs / os
    significance_level: float = 0.05


class ClinicalTrialSimulator:
    """临床试验模拟器"""

    def __init__(self, design: TrialDesign):
        self.design = design
        self._results: Dict[str, List[ArmResult]] = {}

    def run(self, subtype: str = "BLIS", brca_mutation: bool = False) -> Dict[str, Any]:
        """运行试验"""
        all_results = {}

        for arm in self.design.arms:
            arm_results = []
            for patient_id in range(self.design.n_patients_per_arm):
                # 每个患者用不同种子
                sandbox = TNBCPharmacologySandbox(ExperimentMode.CONTROLLED)
                sandbox.add_arm(arm)

                results = sandbox.run(
                    n_days=self.design.duration_days,
                    subtype=subtype,
                    brca_mutation=brca_mutation,
                    seed=42 + patient_id,
                )

                if results:
                    arm_results.append(results[0])

            all_results[arm.name] = arm_results
            self._results[arm.name] = arm_results

        # 汇总统计
        return self._compute_statistics(all_results)

    def _compute_statistics(self, all_results: Dict[str, List[ArmResult]]) -> Dict[str, Any]:
        """计算试验统计"""
        stats = {}
        for arm_name, results in all_results.items():
            if not results:
                continue

            n = len(results)
            # RECIST响应率
            cr_count = sum(1 for r in results if r.recist_response == "CR")
            pr_count = sum(1 for r in results if r.recist_response == "PR")
            orr = (cr_count + pr_count) / n  # 客观缓解率

            # 中位PFS
            pfs_values = sorted(r.pfs_months for r in results)
            median_pfs = pfs_values[n // 2] if n > 0 else 0

            # 平均毒性
            avg_toxicity = sum(r.toxicity_grade for r in results) / n

            # 3级以上毒性比例
            grade3_plus = sum(1 for r in results if r.toxicity_grade >= 3) / n

            stats[arm_name] = {
                "n": n,
                "orr": orr,
                "cr_rate": cr_count / n,
                "pr_rate": pr_count / n,
                "median_pfs": median_pfs,
                "avg_toxicity": avg_toxicity,
                "grade3_plus_rate": grade3_plus,
            }

        return stats