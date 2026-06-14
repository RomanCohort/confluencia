"""TNBC 事件类型定义

所有事件类型常量，按肿瘤模拟生命周期分组。
"""

# ===== 生命周期事件 =====
STEP_START = "step_start"             # 每步开始
STEP_END = "step_end"                 # 每步结束

# ===== 肿瘤生物学事件 =====
TUMOR_GROWTH = "tumor_growth"                # 生长引擎更新
TUMOR_HETEROGENEITY = "tumor_heterogeneity"  # 亚克隆演化
TUMOR_ANGIOGENESIS = "tumor_angiogenesis"    # VEGF驱动血管生成
TUMOR_METASTASIS = "tumor_metastasis"        # EMT/MET转移
TUMOR_CSC_UPDATE = "tumor_csc_update"        # 癌干细胞动态

# ===== 肿瘤微环境事件 =====
TME_IMMUNE_UPDATE = "tme_immune_update"       # 免疫细胞动态
TME_FIBROBLAST_UPDATE = "tme_fibroblast_update"  # CAF激活
TME_ENDOTHELIAL_UPDATE = "tme_endothelial_update"  # 血管动态
TME_EVASION_UPDATE = "tme_evasion_update"     # 免疫逃逸机制
TME_IMMUNOEDITING = "tme_immunoediting"       # 三阶段转换

# ===== 治疗事件 =====
DRUG_ADMINISTERED = "drug_administered"       # 给药
DRUG_PK_UPDATE = "drug_pk_update"            # PK浓度更新
DRUG_PD_EFFECT = "drug_pd_effect"            # PD效应应用
DRUG_RESISTANCE_EMERGED = "drug_resistance_emerged"  # 耐药检测
IMMUNOTHERAPY_UPDATE = "immunotherapy_update"  # 免疫治疗（PD-1/PD-L1, CAR-T）
RADIOTHERAPY_UPDATE = "radiotherapy_update"    # 放疗
CIRCRNA_THERAPY_UPDATE = "circrna_therapy_update"  # circRNA治疗

# ===== 生物标志物事件 =====
BIOMARKER_UPDATE = "biomarker_update"          # 标志物动态
SUBTYPE_RECLASSIFIED = "subtype_reclassified"  # 分子亚型变更
RESISTANCE_DETECTED = "resistance_detected"    # 耐药签名检测

# ===== 临床结局事件 =====
RECIST_EVALUATION = "recist_evaluation"        # RECIST评估
SURVIVAL_UPDATE = "survival_update"            # PFS/OS估计
TOXICITY_UPDATE = "toxicity_update"            # CTCAE毒性分级

# ===== Confluencia集成事件 =====
CONFLUENCIA_DRUG_PREDICTION = "confluencia_drug_prediction"
CONFLUENCIA_PK_SIMULATION = "confluencia_pk_simulation"
CONFLUENCIA_EPITOPE_PREDICTION = "confluencia_epitope_prediction"
CONFLUENCIA_JOINT_EVAL = "confluencia_joint_eval"

# ===== circRNA 子系统事件 =====
CIRCRNA_IMMUNE_EVAL = "circrna_immune_eval"          # 免疫原性评估
CIRCRNA_STRUCTURE_PREDICT = "circrna_structure_predict"  # 结构预测
CIRCRNA_SEQUENCE_EVOLVE = "circrna_sequence_evolve"    # 序列进化优化
CIRCRNA_VACCINE_ASSESS = "circrna_vaccine_assess"      # 疫苗综合评估
CIRCRNA_FOLDING_KINETICS = "circrna_folding_kinetics"  # 折叠动力学
CIRCRNA_DRUG_RESPONSE = "circrna_drug_response"        # 药物响应预测
CIRCRNA_PK_SIMULATE = "circrna_pk_simulate"            # circRNA PK 模拟请求
MOLECULE_EVOLUTION_REQUEST = "molecule_evolution_request"  # 药物分子进化请求

# ===== 实验事件 =====
EXPERIMENT_START = "experiment_start"
EXPERIMENT_END = "experiment_end"
TRIAL_PHASE_TRANSITION = "trial_phase_transition"
COMBINATION_SYNERGY_CALCULATED = "combination_synergy_calculated"


# 所有事件类型列表
ALL_EVENTS = [
    STEP_START, STEP_END,
    TUMOR_GROWTH, TUMOR_HETEROGENEITY, TUMOR_ANGIOGENESIS, TUMOR_METASTASIS, TUMOR_CSC_UPDATE,
    TME_IMMUNE_UPDATE, TME_FIBROBLAST_UPDATE, TME_ENDOTHELIAL_UPDATE, TME_EVASION_UPDATE, TME_IMMUNOEDITING,
    DRUG_ADMINISTERED, DRUG_PK_UPDATE, DRUG_PD_EFFECT, DRUG_RESISTANCE_EMERGED,
    IMMUNOTHERAPY_UPDATE, RADIOTHERAPY_UPDATE, CIRCRNA_THERAPY_UPDATE,
    BIOMARKER_UPDATE, SUBTYPE_RECLASSIFIED, RESISTANCE_DETECTED,
    RECIST_EVALUATION, SURVIVAL_UPDATE, TOXICITY_UPDATE,
    CONFLUENCIA_DRUG_PREDICTION, CONFLUENCIA_PK_SIMULATION, CONFLUENCIA_EPITOPE_PREDICTION, CONFLUENCIA_JOINT_EVAL,
    CIRCRNA_IMMUNE_EVAL, CIRCRNA_STRUCTURE_PREDICT, CIRCRNA_SEQUENCE_EVOLVE, CIRCRNA_VACCINE_ASSESS,
    CIRCRNA_FOLDING_KINETICS, CIRCRNA_DRUG_RESPONSE, CIRCRNA_PK_SIMULATE, MOLECULE_EVOLUTION_REQUEST,
    EXPERIMENT_START, EXPERIMENT_END, TRIAL_PHASE_TRANSITION, COMBINATION_SYNERGY_CALCULATED,
]