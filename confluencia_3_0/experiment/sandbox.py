"""TNBC 药理学沙盒 (TNBC Pharmacology Sandbox)

仿照 CLF PsychopharmacologySandbox，适配肿瘤学实验设计。

5种实验模式:
  1. Controlled: 单药对照试验
  2. Temporal: 时序探索（化疗先 vs 免疫先）
  3. DoseFrequency: 剂量-频率矩阵
  4. Combination: 联合用药筛选
  5. ResistancePrevention: 耐药预防策略
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.config import TNBCSimulacrumConfig
from core.agent import TNBCSimulacrum


class ExperimentMode(Enum):
    CONTROLLED = "controlled"
    TEMPORAL = "temporal"
    DOSE_FREQUENCY = "dose_frequency"
    COMBINATION = "combination"
    RESISTANCE_PREVENTION = "resistance_prevention"


@dataclass
class TreatmentArm:
    """治疗臂"""
    name: str
    drugs: List[Dict[str, Any]] = field(default_factory=list)
    # [{"name": "doxorubicin", "dose": 60.0, "start_day": 0, "end_day": 126}]
    immunotherapies: List[Dict[str, Any]] = field(default_factory=list)
    targeted_therapies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ArmResult:
    """治疗臂结果"""
    name: str
    final_volume: float = 0.0
    recist_response: str = "SD"
    tumor_change_pct: float = 0.0
    pfs_months: float = 0.0
    toxicity_grade: int = 0
    resistance_level: float = 0.0
    volume_trajectory: List[float] = field(default_factory=list)
    recist_trajectory: List[str] = field(default_factory=list)


class TNBCPharmacologySandbox:
    """TNBC药理学沙盒"""

    def __init__(self, mode: ExperimentMode = ExperimentMode.CONTROLLED):
        self.mode = mode
        self._arms: List[TreatmentArm] = []
        self._results: List[ArmResult] = []

    def add_arm(self, arm: TreatmentArm):
        """添加治疗臂"""
        self._arms.append(arm)

    def run(self, n_days: int = 180, subtype: str = "BLIS",
            brca_mutation: bool = False, seed: int = 42) -> List[ArmResult]:
        """运行实验"""
        self._results = []

        for arm in self._arms:
            result = self._run_arm(arm, n_days, subtype, brca_mutation, seed)
            self._results.append(result)

        return self._results

    def _run_arm(self, arm: TreatmentArm, n_days: int,
                 subtype: str, brca_mutation: bool, seed: int) -> ArmResult:
        """运行单个治疗臂"""
        config = TNBCSimulacrumConfig()
        config.molecular_subtype = subtype
        config.brca_mutation = brca_mutation
        config.experiment.seed = seed

        agent = TNBCSimulacrum(config)

        volume_trajectory = []
        recist_trajectory = []

        for day in range(n_days):
            # 给药检查
            for drug in arm.drugs:
                start = drug.get("start_day", 0)
                end = drug.get("end_day", n_days)
                freq = drug.get("frequency_days", 21)

                if start <= day < end:
                    if (day - start) % freq == 0:
                        agent.administer_drug(
                            drug["name"],
                            drug.get("dose", 60.0),
                        )

            # 推进模拟
            agent.step()

            # 记录轨迹
            if day % 7 == 0:
                volume_trajectory.append(agent.state["tum_volume"])
                recist_trajectory.append(agent.state["cli_recist_response"])

        # 收集结果
        s = agent.state
        return ArmResult(
            name=arm.name,
            final_volume=s["tum_volume"],
            recist_response=s["cli_recist_response"],
            tumor_change_pct=s["cli_tumor_change_pct"],
            pfs_months=s["cli_pfs_months"],
            toxicity_grade=s["cli_toxicity_grade"],
            resistance_level=s["drg_resistance_level"],
            volume_trajectory=volume_trajectory,
            recist_trajectory=recist_trajectory,
        )

    def get_summary(self) -> str:
        """获取实验摘要"""
        lines = [f"TNBC Pharmacology Sandbox - {self.mode.value}"]
        lines.append("=" * 50)

        for r in self._results:
            lines.append(f"\n{r.name}:")
            lines.append(f"  Final volume: {r.final_volume:.1f} mm3")
            lines.append(f"  RECIST: {r.recist_response}")
            lines.append(f"  Change: {r.tumor_change_pct:.1f}%")
            lines.append(f"  PFS: {r.pfs_months:.1f} months")
            lines.append(f"  Toxicity: Grade {r.toxicity_grade}")
            lines.append(f"  Resistance: {r.resistance_level:.3f}")

        return "\n".join(lines)