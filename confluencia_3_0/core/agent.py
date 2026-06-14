"""TNBC Simulacrum Agent

三阴性乳腺癌模拟环境主编排器，遵循 CLF EventBus-first 架构。
"""
import random
import numpy as np
from typing import Dict, List, Any, Optional

from .event_bus import EventBus, Event
from .events import *
from .state_schema import StateSchema
from .config import TNBCSimulacrumConfig


class TNBCSimulacrum:
    """三阴性乳腺癌模拟 Agent

    编排肿瘤生长、TME交互、治疗效应、生物标志物动态和临床评估。
    每个模拟步代表一天。

    用法:
        config = TNBCSimulacrumConfig()
        agent = TNBCSimulacrum(config)
        agent.step()  # 推进一天
    """

    def __init__(self, config: Optional[TNBCSimulacrumConfig] = None):
        self.config = config or TNBCSimulacrumConfig()

        # 随机种子
        self._rng = random.Random(self.config.experiment.seed)
        np.random.seed(self.config.experiment.seed)

        # EventBus
        self.bus = EventBus(log_enabled=True)

        # StateSchema + 初始化状态
        self._schema = StateSchema()
        self._internal_state: Dict[str, Any] = self._schema.init_defaults()

        # 应用配置到初始状态
        self._apply_config_to_state()

        # 模拟计数器
        self._step_count: int = 0
        self._day: int = 0

        # ===== 创建子系统模块 =====
        # 每个模块在 __init__ 中自行订阅 EventBus

        # 肿瘤模块
        from .tumor.growth_engine import TumorGrowthEngine
        from .tumor.heterogeneity import TumorHeterogeneity
        from .tumor.cancer_stem_cell import CancerStemCellPool
        from .tumor.angiogenesis import AngiogenesisEngine
        from .tumor.metastasis import MetastasisEngine

        self.growth_engine = TumorGrowthEngine(self.config.tumor, event_bus=self.bus)
        self.heterogeneity = TumorHeterogeneity(self.config.heterogeneity, event_bus=self.bus)
        self.csc_pool = CancerStemCellPool(self.config.csc, event_bus=self.bus)
        self.angiogenesis = AngiogenesisEngine(self.config.angiogenesis, event_bus=self.bus)
        self.metastasis_engine = MetastasisEngine(self.config.metastasis, event_bus=self.bus)

        # TME模块
        from .tme.immune_dynamics import ImmuneCellDynamics
        from .tme.fibroblast import FibroblastActivation
        from .tme.immune_evasion import ImmuneEvasion
        from .tme.immunoediting import Immunoediting

        self.immune = ImmuneCellDynamics(self.config.immune, event_bus=self.bus)
        self.fibroblast = FibroblastActivation(self.config.caf, event_bus=self.bus)
        self.evasion = ImmuneEvasion(self.config.evasion, event_bus=self.bus)
        self.immunoediting = Immunoediting(config=None)

        # 治疗模块
        from .treatment.chemotherapy import ChemotherapyEngine
        from .treatment.immunotherapy import ImmunotherapyEngine
        from .treatment.targeted import TargetedTherapyEngine
        from .treatment.radiotherapy import RadiotherapyEngine

        self.chemotherapy = ChemotherapyEngine(event_bus=self.bus)
        self.immunotherapy = ImmunotherapyEngine(event_bus=self.bus)
        self.targeted = TargetedTherapyEngine(event_bus=self.bus)
        self.radiotherapy = RadiotherapyEngine(event_bus=self.bus)

        # circRNA治疗
        from .treatment.circrna_therapy import CircRNATherapyEngine

        self.circrna_therapy = CircRNATherapyEngine(config=None)

        # 生物标志物
        from .biomarker.tracker import BiomarkerTracker
        from .biomarker.subtype_classifier import MolecularSubtypeClassifier

        self.biomarker_tracker = BiomarkerTracker(event_bus=self.bus)
        self.subtype_classifier = MolecularSubtypeClassifier(event_bus=self.bus)

        # 临床评估
        from .clinical.recist import RECISTTracker
        from .clinical.survival import SurvivalModel
        from .clinical.toxicity import ToxicityGrader

        self.recist = RECISTTracker(event_bus=self.bus)
        self.survival_model = SurvivalModel(event_bus=self.bus)
        self.toxicity_grader = ToxicityGrader(event_bus=self.bus)

        # ===== Subsystem Managers =====
        from .subsystem_managers import (
            TumorManager, TMEManager, TreatmentManager,
            BiomarkerManager, ClinicalManager,
        )

        self.mgr_tumor = TumorManager(self)
        self.mgr_tme = TMEManager(self)
        self.mgr_treatment = TreatmentManager(self)
        self.mgr_biomarker = BiomarkerManager(self)
        self.mgr_clinical = ClinicalManager(self)

        # 记录基线体积
        self._internal_state["cli_baseline_volume"] = self._internal_state["tum_volume"]
        self._internal_state["cli_nadir_volume"] = self._internal_state["tum_volume"]

    def _apply_config_to_state(self):
        """将配置参数映射到初始状态"""
        s = self._internal_state
        c = self.config

        # 肿瘤
        s["tum_volume"] = c.tumor.initial_volume
        s["tum_growth_rate"] = c.tumor.growth_rate
        s["tum_apoptosis_rate"] = c.tumor.apoptosis_rate

        # 亚型
        s["sub_molecular_subtype"] = c.molecular_subtype

        # CSC
        s["csc_fraction"] = c.csc.initial_fraction
        s["csc_self_renewal"] = c.csc.self_renewal_rate
        s["csc_differentiation_rate"] = c.csc.differentiation_rate
        s["csc_chemo_resistance"] = c.csc.chemo_resistance_factor

        # BRCA
        if c.brca_mutation:
            s["bio_brca_status"] = 1  # BRCA1突变
            s["bio_hr_status"] = 1    # HRD+

        # 亚型特定参数
        self._apply_subtype_preset(c.molecular_subtype)

    def _apply_subtype_preset(self, subtype: str):
        """应用分子亚型预设参数"""
        s = self._internal_state
        presets = {
            "BLIS": {  # Basal-Like Immune-Suppressed
                "tum_growth_rate": 0.035,
                "evs_pd_l1_expression": 0.15,
                "imm_til_density": 0.1,
                "sub_ck5_6_expression": 0.8,
                "sub_egfr_expression": 0.7,
                "sub_ar_expression": 0.05,
                "bio_tmb": 8.0,
            },
            "IM": {  # Immunomodulatory
                "tum_growth_rate": 0.025,
                "evs_pd_l1_expression": 0.5,
                "imm_til_density": 0.6,
                "sub_ck5_6_expression": 0.6,
                "sub_egfr_expression": 0.5,
                "sub_ar_expression": 0.1,
                "bio_tmb": 12.0,
            },
            "M": {  # Mesenchymal
                "tum_growth_rate": 0.03,
                "evs_pd_l1_expression": 0.2,
                "imm_til_density": 0.2,
                "sub_ck5_6_expression": 0.4,
                "sub_egfr_expression": 0.3,
                "sub_ar_expression": 0.1,
                "bio_tmb": 6.0,
                "met_emt_progress": 0.3,
                "caf_activation": 0.4,
            },
            "LAR": {  # Luminal Androgen Receptor
                "tum_growth_rate": 0.02,
                "evs_pd_l1_expression": 0.1,
                "imm_til_density": 0.15,
                "sub_ck5_6_expression": 0.2,
                "sub_egfr_expression": 0.2,
                "sub_ar_expression": 0.8,
                "bio_tmb": 3.0,
                "bio_androgen_receptor": 0.7,
            },
        }
        if subtype in presets:
            for key, value in presets[subtype].items():
                s[key] = value

    @property
    def state(self) -> Dict[str, Any]:
        """只读状态访问"""
        return self._internal_state

    @property
    def day(self) -> int:
        return self._day

    @property
    def step_count(self) -> int:
        return self._step_count

    def step(self) -> Dict[str, Any]:
        """推进一个模拟步（一天）

        Pipeline:
        1. STEP_START
        2. 肿瘤生长
        3. TME交互
        4. 治疗效应
        5. 生物标志物更新
        6. 临床评估
        7. STEP_END

        Returns:
            本步汇总信息
        """
        self._step_count += 1
        self._day += 1

        # 1. STEP_START
        self.bus.publish(STEP_START, {"day": self._day}, source="agent")

        # 2. 肿瘤更新
        tumor_result = self.mgr_tumor.step()

        # 3. TME更新
        tme_result = self.mgr_tme.step()

        # 4. 治疗更新
        treatment_result = self.mgr_treatment.step()

        # 5. 生物标志物更新
        biomarker_result = self.mgr_biomarker.step()

        # 6. 临床评估
        clinical_result = self.mgr_clinical.step()

        # 7. STEP_END
        self.bus.publish(STEP_END, {"day": self._day}, source="agent")

        return {
            "day": self._day,
            "tumor": tumor_result,
            "tme": tme_result,
            "treatment": treatment_result,
            "biomarker": biomarker_result,
            "clinical": clinical_result,
        }

    def run(self, n_days: int = 365) -> List[Dict[str, Any]]:
        """运行模拟 n_days 天"""
        results = []
        for _ in range(n_days):
            result = self.step()
            results.append(result)
        return results

    def administer_drug(self, drug_name: str, dose: float, **kwargs):
        """给药接口"""
        self.bus.publish(DRUG_ADMINISTERED, {
            "drug_name": drug_name,
            "dose": dose,
            "day": self._day,
            **kwargs,
        }, source="agent")

    def get_summary(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        s = self._internal_state
        return {
            "day": self._day,
            "volume_mm3": s.get("tum_volume", 0),
            "subtype": s.get("sub_molecular_subtype", "unknown"),
            "recist": s.get("cli_recist_response", "unknown"),
            "tumor_change_pct": s.get("cli_tumor_change_pct", 0),
            "toxicity_grade": s.get("cli_toxicity_grade", 0),
            "immunoediting_phase": s.get("ied_phase", "unknown"),
            "pd_l1_cps": s.get("bio_pd_l1_cps", 0),
            "brca_status": s.get("bio_brca_status", 0),
        }