"""TNBC 状态模式定义

使用 KeyDef 定义所有状态键的类型、默认值、范围和元数据。
StateSchema 负责初始化默认值和运行时验证。

移植自 CLF core/state_schema.py，适配 TNBC 域。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


@dataclass
class KeyDef:
    """状态键定义

    Attributes:
        key: 状态键名（如 "tum_volume"）
        dtype: Python 类型
        default: 默认值
        range_: 允许范围 (min, max)，None 表示无限制
        description: 人类可读描述
        subsystem: 所属子系统标签
        aliases: 别名列表
    """
    key: str
    dtype: type = float
    default: Any = 0.0
    range_: Optional[Tuple[float, float]] = None
    description: str = ""
    subsystem: str = ""
    aliases: List[str] = field(default_factory=list)


class StateSchema:
    """TNBC 状态模式

    定义 ~170 个状态键，按前缀命名空间组织：
    - tum_* : 肿瘤生物学
    - sub_* : 分子亚型
    - het_* : 异质性
    - csc_* : 癌干细胞
    - vasc_* : 血管生成
    - met_* : 转移
    - imm_* : 免疫
    - evs_* : 免疫逃逸
    - ied_* : 免疫编辑
    - caf_* : 成纤维细胞/ECM
    - drg_* : 药物
    - bio_* : 生物标志物
    - cli_* : 临床结局
    - cfl_* : Confluencia
    """

    # ===== 肿瘤生物学 (tum_*) =====
    TUMOR_KEYS = [
        KeyDef("tum_volume", float, 50.0, (0, 1e6), "肿瘤体积 (mm³)", "tumor"),
        KeyDef("tum_growth_rate", float, 0.027, (0, 1), "净生长率", "tumor"),
        KeyDef("tum_apoptosis_rate", float, 0.005, (0, 1), "凋亡率", "tumor"),
        KeyDef("tum_necrosis_fraction", float, 0.0, (0, 1), "坏死分数", "tumor"),
        KeyDef("tum_proliferation_index", float, 0.3, (0, 1), "Ki-67增殖指数", "tumor"),
        KeyDef("tum_cell_count", float, 5e7, (0, 1e12), "肿瘤细胞数", "tumor"),
        KeyDef("tum_oxygenation", float, 0.7, (0, 1), "氧合水平", "tumor"),
        KeyDef("tum_glucose_level", float, 0.8, (0, 1), "葡萄糖水平", "tumor"),
        KeyDef("tum_lactate_level", float, 0.3, (0, 1), "乳酸水平 (Warburg效应)", "tumor"),
        KeyDef("tum_ph", float, 7.2, (6.0, 7.6), "肿瘤pH (酸性微环境)", "tumor"),
    ]

    # ===== 分子亚型 (sub_*) =====
    SUBTYPE_KEYS = [
        KeyDef("sub_molecular_subtype", str, "BLIS", None, "分子亚型: BLIS/IM/M/LAR", "subtype"),
        KeyDef("sub_subtype_progress", float, 1.0, (0, 1), "亚型稳定性", "subtype"),
        KeyDef("sub_er_expression", float, 0.0, (0, 1), "ER表达 (TNBC=0)", "subtype"),
        KeyDef("sub_pr_expression", float, 0.0, (0, 1), "PR表达 (TNBC=0)", "subtype"),
        KeyDef("sub_her2_expression", float, 0.0, (0, 1), "HER2表达 (TNBC=0)", "subtype"),
        KeyDef("sub_ar_expression", float, 0.1, (0, 1), "AR表达 (LAR亚型高)", "subtype"),
        KeyDef("sub_ck5_6_expression", float, 0.5, (0, 1), "CK5/6基底标志物", "subtype"),
        KeyDef("sub_egfr_expression", float, 0.4, (0, 1), "EGFR表达", "subtype"),
    ]

    # ===== 异质性 (het_*) =====
    HETEROGENEITY_KEYS = [
        KeyDef("het_n_subclones", int, 4, (1, 50), "亚克隆数", "heterogeneity"),
        KeyDef("het_diversity_index", float, 0.3, (0, 1), "Shannon多样性指数", "heterogeneity"),
        KeyDef("het_dominant_clone_fraction", float, 0.6, (0, 1), "优势克隆比例", "heterogeneity"),
        KeyDef("het_resistance_clone_fraction", float, 0.0, (0, 1), "耐药克隆比例", "heterogeneity"),
        KeyDef("het_genomic_instability", float, 0.5, (0, 1), "基因组不稳定性", "heterogeneity"),
    ]

    # ===== 癌干细胞 (csc_*) =====
    CSC_KEYS = [
        KeyDef("csc_fraction", float, 0.02, (0, 1), "CSC比例", "csc"),
        KeyDef("csc_self_renewal", float, 0.5, (0, 1), "自我更新率", "csc"),
        KeyDef("csc_differentiation_rate", float, 0.5, (0, 1), "分化率", "csc"),
        KeyDef("csc_chemo_resistance", float, 5.0, (1, 100), "化疗抗性倍数", "csc"),
        KeyDef("csc_cd44_expression", float, 0.8, (0, 1), "CD44表达", "csc"),
        KeyDef("csc_cd24_expression", float, 0.2, (0, 1), "CD24表达", "csc"),
    ]

    # ===== 血管生成 (vasc_*) =====
    ANGIOGENESIS_KEYS = [
        KeyDef("vasc_vegf_level", float, 0.2, (0, 1), "VEGF水平", "angiogenesis"),
        KeyDef("vasc_microvessel_density", float, 0.3, (0, 1), "微血管密度 (MVD)", "angiogenesis"),
        KeyDef("vasc_oxygenation", float, 0.7, (0, 1), "血管氧合", "angiogenesis"),
        KeyDef("vasc_normalization_window", float, 0.0, (0, 1), "血管正常化窗口", "angiogenesis"),
        KeyDef("vasc_perfusion", float, 0.5, (0, 1), "灌注水平", "angiogenesis"),
        KeyDef("vasc_leakiness", float, 0.3, (0, 1), "血管渗漏", "angiogenesis"),
    ]

    # ===== 转移 (met_*) =====
    METASTASIS_KEYS = [
        KeyDef("met_emt_progress", float, 0.0, (0, 1), "EMT进展", "metastasis"),
        KeyDef("met_met_progress", float, 0.0, (0, 1), "MET逆转进展", "metastasis"),
        KeyDef("met_dissemination_rate", float, 0.001, (0, 1), "肿瘤细胞扩散率", "metastasis"),
        KeyDef("met_metastatic_burden", float, 0.0, (0, 1e6), "转移负荷 (mm³)", "metastasis"),
        KeyDef("met_n_metastatic_sites", int, 0, (0, 20), "转移灶数", "metastasis"),
        KeyDef("met_lung_burden", float, 0.0, (0, 1e6), "肺转移负荷", "metastasis"),
        KeyDef("met_liver_burden", float, 0.0, (0, 1e6), "肝转移负荷", "metastasis"),
        KeyDef("met_bone_burden", float, 0.0, (0, 1e6), "骨转移负荷", "metastasis"),
        KeyDef("met_brain_burden", float, 0.0, (0, 1e6), "脑转移负荷", "metastasis"),
    ]

    # ===== 免疫 (imm_*) =====
    IMMUNE_KEYS = [
        KeyDef("imm_cd8_count", float, 100.0, (0, 1e4), "CD8+ T细胞数/mm³", "immune"),
        KeyDef("imm_cd4_count", float, 150.0, (0, 1e4), "CD4+ T细胞数/mm³", "immune"),
        KeyDef("imm_t_cell_activation", float, 0.3, (0, 1), "T细胞激活水平", "immune"),
        KeyDef("imm_t_cell_exhaustion", float, 0.1, (0, 1), "T细胞耗竭", "immune"),
        KeyDef("imm_nk_cytotoxicity", float, 0.3, (0, 1), "NK细胞毒性", "immune"),
        KeyDef("imm_nk_count", float, 50.0, (0, 1e4), "NK细胞数/mm³", "immune"),
        KeyDef("imm_m1_fraction", float, 0.5, (0, 1), "M1巨噬细胞比例", "immune"),
        KeyDef("imm_m2_fraction", float, 0.5, (0, 1), "M2巨噬细胞比例", "immune"),
        KeyDef("imm_macrophage_count", float, 80.0, (0, 1e4), "巨噬细胞数/mm³", "immune"),
        KeyDef("imm_treg_fraction", float, 0.1, (0, 1), "Treg比例", "immune"),
        KeyDef("imm_treg_count", float, 20.0, (0, 1e4), "Treg数/mm³", "immune"),
        KeyDef("imm_mdsc_count", float, 30.0, (0, 1e4), "MDSC数/mm³", "immune"),
        KeyDef("imm_mdsc_suppression", float, 0.1, (0, 1), "MDSC抑制活性", "immune"),
        KeyDef("imm_til_density", float, 0.2, (0, 1), "肿瘤浸润淋巴细胞密度", "immune"),
        KeyDef("imm_ifn_gamma", float, 0.2, (0, 1), "IFN-γ水平", "immune"),
        KeyDef("imm_il2", float, 0.15, (0, 1), "IL-2水平", "immune"),
        KeyDef("imm_il10", float, 0.1, (0, 1), "IL-10水平 (免疫抑制)", "immune"),
        KeyDef("imm_tnf_alpha", float, 0.15, (0, 1), "TNF-α水平", "immune"),
    ]

    # ===== 免疫逃逸 (evs_*) =====
    EVASION_KEYS = [
        KeyDef("evs_pd_l1_expression", float, 0.2, (0, 1), "PD-L1表达 (CPS)", "evasion"),
        KeyDef("evs_mhc_i_downreg", float, 0.1, (0, 1), "MHC-I下调程度", "evasion"),
        KeyDef("evs_tgf_beta", float, 0.15, (0, 1), "TGF-β水平", "evasion"),
        KeyDef("evs_ido_activity", float, 0.05, (0, 1), "IDO活性", "evasion"),
        KeyDef("evs_gal3_expression", float, 0.2, (0, 1), "Galectin-9表达", "evasion"),
        KeyDef("evs_b7_h3_expression", float, 0.3, (0, 1), "B7-H3表达", "evasion"),
    ]

    # ===== 免疫编辑 (ied_*) =====
    IMMUNOEDITING_KEYS = [
        KeyDef("ied_phase", str, "elimination", None, "免疫编辑阶段: elimination/equilibrium/escape", "immunoediting"),
        KeyDef("ied_phase_progress", float, 0.3, (0, 1), "当前阶段进展", "immunoediting"),
        KeyDef("ied_immune_pressure", float, 0.5, (0, 1), "免疫压力", "immunoediting"),
        KeyDef("ied_evasion_pressure", float, 0.3, (0, 1), "逃逸压力", "immunoediting"),
    ]

    # ===== 成纤维/ECM (caf_*) =====
    CAF_KEYS = [
        KeyDef("caf_activation", float, 0.2, (0, 1), "CAF激活水平", "caf"),
        KeyDef("caf_count", float, 50.0, (0, 1e4), "CAF数/mm³", "caf"),
        KeyDef("caf_ecm_density", float, 0.3, (0, 1), "ECM密度", "caf"),
        KeyDef("caf_ecm_stiffness", float, 0.3, (0, 1), "ECM硬度", "caf"),
        KeyDef("caf_collagen_density", float, 0.4, (0, 1), "胶原密度", "caf"),
        KeyDef("caf_hyaluronan", float, 0.3, (0, 1), "透明质酸水平", "caf"),
    ]

    # ===== 药物 (drg_*) =====
    DRUG_KEYS = [
        KeyDef("drg_active_drug", str, "", None, "当前活跃药物名", "drug"),
        KeyDef("drg_concentration", float, 0.0, (0, 1e6), "中央室药物浓度 (ng/mL)", "drug"),
        KeyDef("drg_effect", float, 0.0, (-1, 1), "药物净效应", "drug"),
        KeyDef("drg_kill_fraction", float, 0.0, (0, 1), "杀伤分数", "drug"),
        KeyDef("drg_resistance_level", float, 0.0, (0, 1), "耐药水平", "drug"),
        KeyDef("drg_dose", float, 0.0, (0, 1e4), "给药剂量 (mg/m²)", "drug"),
        KeyDef("drg_time_since_admin", float, 0.0, (0, 1e6), "距上次给药时间 (h)", "drug"),
        KeyDef("drg_auc", float, 0.0, (0, 1e6), "AUC曲线下面积", "drug"),
        KeyDef("drg_cmax", float, 0.0, (0, 1e6), "Cmax峰浓度", "drug"),
        KeyDef("drg_half_life", float, 12.0, (0, 1e4), "半衰期 (h)", "drug"),
        KeyDef("drg_ec50", float, 1.0, (0, 1e6), "EC50 (ng/mL)", "drug"),
        KeyDef("drg_emax", float, 0.8, (0, 1), "最大效应", "drug"),
    ]

    # ===== 生物标志物 (bio_*) =====
    BIOMARKER_KEYS = [
        KeyDef("bio_pd_l1_cps", float, 10.0, (0, 100), "PD-L1 CPS评分", "biomarker"),
        KeyDef("bio_til_density", float, 0.2, (0, 1), "TIL密度", "biomarker"),
        KeyDef("bio_brca_status", int, 0, (0, 2), "BRCA状态: 0=野生型, 1=BRCA1突变, 2=BRCA2突变", "biomarker"),
        KeyDef("bio_tmb", float, 5.0, (0, 100), "肿瘤突变负荷 (mut/Mb)", "biomarker"),
        KeyDef("bio_ctdna_level", float, 0.1, (0, 1), "ctDNA水平", "biomarker"),
        KeyDef("bio_msi_status", float, 0.05, (0, 1), "MSI-H概率", "biomarker"),
        KeyDef("bio_hr_status", int, 0, (0, 1), "HR状态: 0=HRD-, 1=HRD+", "biomarker"),
        KeyDef("bio_pi3k_mutation", int, 0, (0, 1), "PIK3CA突变: 0=否, 1=是", "biomarker"),
        KeyDef("bio_androgen_receptor", float, 0.1, (0, 1), "AR表达水平", "biomarker"),
    ]

    # ===== 临床 (cli_*) =====
    CLINICAL_KEYS = [
        KeyDef("cli_recist_response", str, "SD", None, "RECIST响应: CR/PR/SD/PD", "clinical"),
        KeyDef("cli_tumor_change_pct", float, 0.0, (-100, 100), "肿瘤变化百分比", "clinical"),
        KeyDef("cli_baseline_volume", float, 50.0, (0, 1e6), "基线体积 (mm³)", "clinical"),
        KeyDef("cli_nadir_volume", float, 50.0, (0, 1e6), "最低点体积", "clinical"),
        KeyDef("cli_pfs_months", float, 0.0, (0, 200), "无进展生存月数", "clinical"),
        KeyDef("cli_os_months", float, 0.0, (0, 200), "总生存月数", "clinical"),
        KeyDef("cli_toxicity_grade", int, 0, (0, 5), "最高CTCAE毒性级别", "clinical"),
        KeyDef("cli_neutropenia_grade", int, 0, (0, 5), "中性粒细胞减少级别", "clinical"),
        KeyDef("cli_cardiotoxicity_grade", int, 0, (0, 5), "心脏毒性级别", "clinical"),
        KeyDef("cli_neuropathy_grade", int, 0, (0, 5), "神经病变级别", "clinical"),
        KeyDef("cli_fatigue_grade", int, 0, (0, 5), "疲劳级别", "clinical"),
        KeyDef("cli_nausea_grade", int, 0, (0, 5), "恶心级别", "clinical"),
        KeyDef("cli_days_on_treatment", int, 0, (0, 1e4), "治疗天数", "clinical"),
        KeyDef("cli_treatment_discontinued", int, 0, (0, 1), "是否停药", "clinical"),
    ]

    # ===== Confluencia (cfl_*) =====
    CONFLUENCIA_KEYS = [
        KeyDef("cfl_drug_prediction_score", float, 0.0, (0, 1), "Confluencia药物预测评分", "confluencia"),
        KeyDef("cfl_pk_simulation_available", int, 0, (0, 1), "PK模拟是否可用", "confluencia"),
        KeyDef("cfl_joint_score", float, 0.0, (0, 1), "联合评估分数", "confluencia"),
        KeyDef("cfl_binding_score", float, 0.0, (0, 1), "结合评分", "confluencia"),
        KeyDef("cfl_kinetics_score", float, 0.0, (0, 1), "动力学评分", "confluencia"),
        KeyDef("cfl_epitope_score", float, 0.0, (0, 1), "表位评分", "confluencia"),
    ]

    # ===== circRNA (crna_*) =====
    CIRCRNA_KEYS = [
        KeyDef("crna_immunogenicity_score", float, 0.0, (0, 1), "circRNA免疫原性评分", "circrna"),
        KeyDef("crna_ips_score", float, 0.0, (0, 10), "IPS免疫表型评分", "circrna"),
        KeyDef("crna_structure_method", str, "none", None, "结构预测方法 (none/viennarna/fallback)", "circrna"),
        KeyDef("crna_mfe_kcal", float, 0.0, None, "MFE最小自由能 (kcal/mol)", "circrna"),
        KeyDef("crna_pkr_score", float, 0.0, (0, 1), "PKR通路评分", "circrna"),
        KeyDef("crna_rig_i_score", float, 0.0, (0, 1), "RIG-I通路评分", "circrna"),
        KeyDef("crna_tlr_score", float, 0.0, (0, 1), "TLR通路评分", "circrna"),
        KeyDef("crna_vaccine_therapeutic_window", float, 0.0, (0, 1), "疫苗治疗窗口", "circrna"),
        KeyDef("crna_evolution_generation", int, 0, (0, 1e4), "序列进化代数", "circrna"),
        KeyDef("crna_evolution_best_score", float, 0.0, (0, 1), "当前最优序列评分", "circrna"),
        KeyDef("crna_folding_method", str, "none", None, "折叠方法", "circrna"),
        KeyDef("crna_backend_tier", str, "heuristic", None, "当前后端层级 (heuristic/vienna/esm2)", "circrna"),
        KeyDef("crna_pk_auc_efficacy", float, 0.0, (0, 1e6), "RNACTM AUC 疗效", "circrna"),
        KeyDef("crna_pk_peak_protein", float, 0.0, (0, 1e6), "RNACTM 蛋白峰值", "circrna"),
        KeyDef("crna_pk_rna_half_life", float, 0.0, (0, 1e4), "RNACTM RNA 半衰期 (h)", "circrna"),
        # TorusFold DL 信号
        KeyDef("crna_torusfold_method", str, "none", None, "TorusFold方法 (none/torusfold_dl/unavailable)", "circrna"),
        KeyDef("crna_closure_score", float, 0.5, (0, 1), "TorusFold闭合约束评分", "circrna"),
        KeyDef("crna_bsj_stability", float, 0.5, (0, 1), "TorusFold BSJ稳定性", "circrna"),
        KeyDef("crna_dsRNA_fraction_dl", float, 0.0, (0, 1), "TorusFold dsRNA比例", "circrna"),
        KeyDef("crna_translation_efficiency_dl", float, 0.5, (0, 1), "TorusFold翻译效率", "circrna"),
        KeyDef("crna_circ_stability_dl", float, 0.5, (0, 1), "TorusFold环稳定性", "circrna"),
        KeyDef("crna_rig_i_score_dl", float, 0.0, (0, 1), "TorusFold RIG-I (DL覆盖)", "circrna"),
        KeyDef("crna_pkr_score_dl", float, 0.0, (0, 1), "TorusFold PKR (DL覆盖)", "circrna"),
        KeyDef("crna_tlr_score_dl", float, 0.0, (0, 1), "TorusFold TLR (DL覆盖)", "circrna"),
        KeyDef("crna_obj_stability", float, 0.0, (0, 1), "四维目标: 稳定性", "circrna"),
        KeyDef("crna_obj_translation", float, 0.0, (0, 1), "四维目标: 翻译", "circrna"),
        KeyDef("crna_obj_immune_evasion", float, 0.0, (0, 1), "四维目标: 免疫逃逸", "circrna"),
        KeyDef("crna_obj_delivery", float, 0.0, (0, 1), "四维目标: 递送", "circrna"),
        KeyDef("molecule_evolution_best_score", float, 0.0, (0, 1), "分子进化最优评分", "evolution"),
    ]

    def __init__(self):
        self._all_keys: Dict[str, KeyDef] = {}
        self._alias_map: Dict[str, str] = {}
        self._subsystem_keys: Dict[str, List[str]] = {}

        all_groups = [
            self.TUMOR_KEYS, self.SUBTYPE_KEYS, self.HETEROGENEITY_KEYS,
            self.CSC_KEYS, self.ANGIOGENESIS_KEYS, self.METASTASIS_KEYS,
            self.IMMUNE_KEYS, self.EVASION_KEYS, self.IMMUNOEDITING_KEYS,
            self.CAF_KEYS, self.DRUG_KEYS, self.BIOMARKER_KEYS,
            self.CLINICAL_KEYS, self.CONFLUENCIA_KEYS, self.CIRCRNA_KEYS,
        ]

        for group in all_groups:
            for kd in group:
                self._all_keys[kd.key] = kd
                for alias in kd.aliases:
                    self._alias_map[alias] = kd.key
                if kd.subsystem:
                    self._subsystem_keys.setdefault(kd.subsystem, []).append(kd.key)

    def init_defaults(self) -> Dict[str, Any]:
        """创建包含所有默认值的状态字典"""
        return {kd.key: kd.default for kd in self._all_keys.values()}

    def validate(self, state: Dict[str, Any]) -> List[str]:
        """验证状态字典，返回警告列表"""
        warnings = []
        for key, value in state.items():
            kd = self._all_keys.get(key)
            if kd is None:
                continue
            if not isinstance(value, kd.dtype):
                try:
                    value = kd.dtype(value)
                except (ValueError, TypeError):
                    warnings.append(f"Key '{key}': cannot convert {type(value).__name__} to {kd.dtype.__name__}")
            if kd.range_ is not None and isinstance(value, (int, float)):
                lo, hi = kd.range_
                if value < lo or value > hi:
                    warnings.append(f"Key '{key}': value {value} outside range [{lo}, {hi}]")
        return warnings

    def get_key(self, key: str) -> Optional[KeyDef]:
        """获取键定义"""
        return self._all_keys.get(key) or self._all_keys.get(self._alias_map.get(key, ""))

    def get_subsystem_keys(self, subsystem: str) -> List[str]:
        """获取子系统的所有键"""
        return self._subsystem_keys.get(subsystem, [])

    @property
    def all_keys(self) -> Dict[str, KeyDef]:
        return self._all_keys

    @property
    def key_count(self) -> int:
        return len(self._all_keys)