"""生物标志物分层 (Biomarker Stratification)

按生物标志物分层测试不同治疗方案。
"""
from typing import Dict, Any, List
from .sandbox import TNBCPharmacologySandbox, TreatmentArm


class BiomarkerStratification:
    """生物标志物分层"""

    def __init__(self):
        self._strata: List[Dict[str, Any]] = []

    def add_stratum(self, name: str, filter_fn=None, treatments: List[TreatmentArm] = None):
        """添加分层"""
        self._strata.append({
            "name": name,
            "filter": filter_fn,
            "treatments": treatments or [],
        })

    def run(self, n_days: int = 180) -> Dict[str, Any]:
        """运行分层分析"""
        results = {}

        for stratum in self._strata:
            stratum_name = stratum["name"]
            stratum_results = {}

            for arm in stratum["treatments"]:
                sandbox = TNBCPharmacologySandbox()
                sandbox.add_arm(arm)
                arm_results = sandbox.run(n_days, subtype=stratum_name)

                if arm_results:
                    stratum_results[arm.name] = {
                        "recist": arm_results[0].recist_response,
                        "change_pct": arm_results[0].tumor_change_pct,
                        "pfs": arm_results[0].pfs_months,
                    }

            results[stratum_name] = stratum_results

        return results