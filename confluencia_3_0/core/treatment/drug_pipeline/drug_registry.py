"""肿瘤药物注册表 (Oncology Drug Registry)

定义 TNBC 常用药物的 PK/PD 参数和受体靶点。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class OncologyDrugDefinition:
    """肿瘤药物定义"""
    name: str
    drug_class: str              # chemo, immunotherapy, targeted, anti_angiogenic
    smiles: str = ""             # SMILES 结构式
    dose_mg_m2: float = 0.0     # 标准剂量 (mg/m²)
    frequency_days: int = 21    # 给药间隔 (天)
    half_life_h: float = 12.0   # 半衰期 (h)
    ec50: float = 1.0           # EC50 (ng/mL)
    emax: float = 0.8           # 最大效应
    hill_coeff: float = 1.0     # Hill系数
    cyp_interactions: List[str] = field(default_factory=list)  # CYP450相互作用
    resistance_mechanisms: List[str] = field(default_factory=list)
    toxicity_profile: Dict[str, int] = field(default_factory=dict)  # {毒性类型: 最高CTCAE级别}
    receptor_targets: Dict[str, float] = field(default_factory=dict)  # {受体: 亲和力}


# ===== TNBC 常用药物 =====

DRUG_REGISTRY: Dict[str, OncologyDrugDefinition] = {
    # ===== 化疗药物 =====
    "doxorubicin": OncologyDrugDefinition(
        name="doxorubicin",
        drug_class="chemo",
        smiles="CC1C(C(CC(O1)OC2CC3C4=CC=CC=C4C(=O)C5=C(C3=C2O)C(=O)C6=C5C=CC=C6O)O)N",
        dose_mg_m2=60.0,
        frequency_days=21,
        half_life_h=30.0,
        ec50=0.05,
        emax=0.7,
        hill_coeff=1.5,
        cyp_interactions=["3A4", "2D6"],
        resistance_mechanisms=["ABCB1_overexpression", "TOP2A_mutation"],
        toxicity_profile={"cardiotoxicity": 4, "neutropenia": 4, "nausea": 3, "fatigue": 2},
        receptor_targets={"topoisomerase_II": 0.9},
    ),
    "paclitaxel": OncologyDrugDefinition(
        name="paclitaxel",
        drug_class="chemo",
        smiles="CC1=C2C(=O)C3C4C(C5C(C(O5)C(=O)OC6C(C7C(C6O)OC(=O)C6=CC=CC=C6)O)O)OC(=O)C",
        dose_mg_m2=175.0,
        frequency_days=21,
        half_life_h=6.0,
        ec50=0.01,
        emax=0.65,
        hill_coeff=1.2,
        cyp_interactions=["3A4", "2C8"],
        resistance_mechanisms=["ABCB1_overexpression", "TUBB_mutation"],
        toxicity_profile={"neuropathy": 3, "neutropenia": 4, "fatigue": 2, "nausea": 2},
        receptor_targets={"beta_tubulin": 0.95},
    ),
    "carboplatin": OncologyDrugDefinition(
        name="carboplatin",
        drug_class="chemo",
        smiles="C1CC1N[Pt](OC(=O)C(=O)O)(Cl)Cl",
        dose_mg_m2=400.0,  # AUC-based dosing simplified
        frequency_days=21,
        half_life_h=3.0,
        ec50=10.0,
        emax=0.5,
        hill_coeff=1.0,
        cyp_interactions=[],
        resistance_mechanisms=["ERCC1_overexpression", "BRCA_reversion"],
        toxicity_profile={"neutropenia": 3, "thrombocytopenia": 3, "nausea": 2},
        receptor_targets={"DNA": 0.8},
    ),
    "cisplatin": OncologyDrugDefinition(
        name="cisplatin",
        drug_class="chemo",
        smiles="N[Pt](Cl)(Cl)N",
        dose_mg_m2=75.0,
        frequency_days=21,
        half_life_h=2.0,
        ec50=5.0,
        emax=0.55,
        hill_coeff=1.0,
        cyp_interactions=[],
        resistance_mechanisms=["ERCC1_overexpression", "BRCA_reversion"],
        toxicity_profile={"nephrotoxicity": 3, "neuropathy": 3, "neutropenia": 3, "nausea": 4},
        receptor_targets={"DNA": 0.85},
    ),

    # ===== 免疫治疗 =====
    "atezolizumab": OncologyDrugDefinition(
        name="atezolizumab",
        drug_class="immunotherapy",
        smiles="",
        dose_mg_m2=1200.0,  # 1200mg fixed dose
        frequency_days=14,
        half_life_h=408.0,  # ~17天
        ec50=5.0,
        emax=0.6,
        hill_coeff=1.0,
        cyp_interactions=[],
        resistance_mechanisms=["low_pd_l1", "low_til", "high_tgf_beta"],
        toxicity_profile={"immune_colitis": 2, "immune_hepatitis": 2, "fatigue": 2},
        receptor_targets={"pd_l1": 0.95},
    ),
    "pembrolizumab": OncologyDrugDefinition(
        name="pembrolizumab",
        drug_class="immunotherapy",
        smiles="",
        dose_mg_m2=200.0,  # 200mg fixed dose
        frequency_days=21,
        half_life_h=552.0,  # ~23天
        ec50=3.0,
        emax=0.65,
        hill_coeff=1.0,
        cyp_interactions=[],
        resistance_mechanisms=["low_pd_l1", "low_tmb", "mhc_i_loss"],
        toxicity_profile={"immune_colitis": 2, "immune_pneumonitis": 2, "fatigue": 2},
        receptor_targets={"pd_1": 0.95},
    ),

    # ===== 靶向治疗 =====
    "olaparib": OncologyDrugDefinition(
        name="olaparib",
        drug_class="targeted",
        smiles="C1=CC=C2C(=C1)C(=O)NC(=O)N2C3=CC=C(C=C3)C(=O)N4CCN(CC4)C",
        dose_mg_m2=300.0,  # 300mg BID
        frequency_days=1,
        half_life_h=5.0,
        ec50=0.05,
        emax=0.75,  # BRCA突变中高效应
        hill_coeff=1.0,
        cyp_interactions=["3A4"],
        resistance_mechanisms=["BRCA_reversion", "PARP1_mutation", "53BP1_loss"],
        toxicity_profile={"nausea": 2, "fatigue": 2, "anemia": 3},
        receptor_targets={"parp1": 0.9, "parp2": 0.8},
    ),
    "ipatasertib": OncologyDrugDefinition(
        name="ipatasertib",
        drug_class="targeted",
        smiles="",
        dose_mg_m2=400.0,
        frequency_days=1,
        half_life_h=8.0,
        ec50=0.1,
        emax=0.5,
        hill_coeff=1.0,
        cyp_interactions=["3A4"],
        resistance_mechanisms=["AKT_mutation", "PTEN_loss"],
        toxicity_profile={"diarrhea": 3, "rash": 2, "fatigue": 2},
        receptor_targets={"akt": 0.9},
    ),
    "enzalutamide": OncologyDrugDefinition(
        name="enzalutamide",
        drug_class="targeted",
        smiles="",
        dose_mg_m2=160.0,  # 160mg daily
        frequency_days=1,
        half_life_h=336.0,  # ~14天
        ec50=0.02,
        emax=0.4,  # LAR亚型中有效
        hill_coeff=1.0,
        cyp_interactions=["2C8", "3A4"],
        resistance_mechanisms=["ar_mutation", "ar_splice_variant"],
        toxicity_profile={"fatigue": 2, "rash": 2},
        receptor_targets={"androgen_receptor": 0.95},
    ),

    # ===== 抗血管生成 =====
    "bevacizumab": OncologyDrugDefinition(
        name="bevacizumab",
        drug_class="anti_angiogenic",
        smiles="",
        dose_mg_m2=15.0,  # 15 mg/kg
        frequency_days=21,
        half_life_h=480.0,  # ~20天
        ec50=10.0,
        emax=0.3,
        hill_coeff=1.0,
        cyp_interactions=[],
        resistance_mechanisms=["vegfr_mutation", "alternative_angiogenesis"],
        toxicity_profile={"hypertension": 3, "proteinuria": 2, "bleeding": 2},
        receptor_targets={"vegf": 0.95},
    ),
}


def get_drug(name: str) -> Optional[OncologyDrugDefinition]:
    """获取药物定义"""
    return DRUG_REGISTRY.get(name.lower())


def list_drugs_by_class(drug_class: str) -> List[OncologyDrugDefinition]:
    """按类别列出药物"""
    return [d for d in DRUG_REGISTRY.values() if d.drug_class == drug_class]


def list_all_drugs() -> List[str]:
    """列出所有药物名"""
    return list(DRUG_REGISTRY.keys())