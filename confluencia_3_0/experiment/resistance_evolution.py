"""耐药演化追踪器 (Resistance Evolution Tracker)"""
from typing import Dict, Any, List
from .sandbox import TNBCPharmacologySandbox, TreatmentArm, ExperimentMode


class ResistanceEvolutionTracker:
    """耐药演化追踪器"""

    def __init__(self):
        self._timeline: List[Dict[str, Any]] = []

    def run(self, drug_name: str, dose: float = 60.0,
            n_days: int = 365, subtype: str = "BLIS") -> Dict[str, Any]:
        """运行耐药追踪实验"""
        from core.config import TNBCSimulacrumConfig
        from core.agent import TNBCSimulacrum

        config = TNBCSimulacrumConfig()
        config.molecular_subtype = subtype
        config.experiment.seed = 42

        agent = TNBCSimulacrum(config)

        # 基线
        for _ in range(30):
            agent.step()

        # 给药
        agent.administer_drug(drug_name, dose)

        # 追踪
        timeline = []
        resistance_emerged_day = None

        for day in range(n_days - 30):
            agent.step()

            if day % 7 == 0:
                s = agent.state
                entry = {
                    "day": agent.day,
                    "volume": s["tum_volume"],
                    "resistance_level": s["drg_resistance_level"],
                    "n_subclones": s["het_n_subclones"],
                    "diversity": s["het_diversity_index"],
                    "recist": s["cli_recist_response"],
                }
                timeline.append(entry)

                if s["drg_resistance_level"] > 0.3 and resistance_emerged_day is None:
                    resistance_emerged_day = agent.day

        self._timeline = timeline

        return {
            "drug": drug_name,
            "resistance_emerged_day": resistance_emerged_day,
            "final_resistance": agent.state["drg_resistance_level"],
            "final_subclones": agent.state["het_n_subclones"],
            "timeline": timeline,
        }