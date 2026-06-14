"""TNBC Simulacrum 配置

可变运行时参数，实验期间可动态修改。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TumorConfig:
    """肿瘤生物学参数"""
    growth_model: str = "logistic"        # "logistic" | "gompertz"
    growth_rate: float = 0.027            # 日生长率 (TNBC中位值)
    carrying_capacity: float = 1000.0     # mm^3 最大体积
    initial_volume: float = 50.0          # mm^3 初始体积
    apoptosis_rate: float = 0.005         # 日凋亡率
    necrosis_threshold: float = 800.0     # 缺氧坏死阈值 (mm^3)
    doubling_time: float = 25.0           # 天 (TNBC中位倍增时间)


@dataclass
class HeterogeneityConfig:
    """肿瘤异质性参数"""
    n_initial_subclones: int = 4          # 初始亚克隆数
    mutation_rate: float = 1e-6           # 每细胞每代突变率
    max_subclones: int = 50               # 最大亚克隆数
    resistance_mutation_prob: float = 0.01  # 耐药突变概率


@dataclass
class CSCConfig:
    """癌干细胞参数"""
    initial_fraction: float = 0.02        # 初始CSC比例
    self_renewal_rate: float = 0.5        # 自我更新概率
    differentiation_rate: float = 0.5     # 分化概率
    chemo_resistance_factor: float = 5.0  # CSC化疗抗性倍数


@dataclass
class AngiogenesisConfig:
    """血管生成参数"""
    vegf_production_rate: float = 0.1     # VEGF产生率
    vegf_threshold: float = 0.3           # 血管生成VEGF阈值
    max_microvessel_density: float = 1.0  # 最大微血管密度
    normalization_duration: float = 7.0   # 血管正常化窗口 (天)


@dataclass
class MetastasisConfig:
    """转移参数"""
    emt_rate: float = 0.01                # EMT转化率
    met_rate: float = 0.005               # MET逆转率
    dissemination_rate: float = 0.001     # 扩散率
    organotropism: Dict[str, float] = field(default_factory=lambda: {
        "lung": 0.35, "liver": 0.25, "bone": 0.20, "brain": 0.15, "distant_lymph": 0.05
    })


@dataclass
class ImmuneConfig:
    """免疫参数"""
    cd8_activation_rate: float = 0.05     # CD8+ T细胞激活率
    cd8_killing_rate: float = 0.02        # CD8+ 杀伤率
    nk_cytotoxicity: float = 0.015        # NK细胞毒性
    m1_polarization_rate: float = 0.03    # M1极化率
    m2_polarization_rate: float = 0.02    # M2极化率
    treg_suppression: float = 0.01        # Treg抑制率
    mdsc_suppression: float = 0.015       # MDSC抑制率
    t_cell_exhaustion_rate: float = 0.005  # T细胞耗竭率
    ifn_gamma_production: float = 0.02    # IFN-γ产生率


@dataclass
class EvasionConfig:
    """免疫逃逸参数"""
    pd_l1_upregulation_rate: float = 0.01  # PD-L1上调率
    mhc_i_downreg_rate: float = 0.005      # MHC-I下调率
    tgf_beta_secretion_rate: float = 0.008  # TGF-β分泌率
    ido_activation_rate: float = 0.003      # IDO激活率


@dataclass
class CAFConfig:
    """肿瘤相关成纤维细胞参数"""
    activation_rate: float = 0.01          # CAF激活率
    ecm_production_rate: float = 0.02      # ECM产生率
    ecm_degradation_rate: float = 0.01     # ECM降解率
    max_ecm_density: float = 1.0           # 最大ECM密度


@dataclass
class DrugPipelineConfig:
    """药物管线参数"""
    pk_model: str = "three_compartment"    # PK模型类型
    absorption_rate: float = 1.0           # 吸收率
    central_volume: float = 10.0           # 中央室体积 (L)
    elimination_rate: float = 0.1          # 消除率
    distribution_rate: float = 0.5         # 分布率
    hill_coefficient: float = 1.0          # Hill方程系数
    bioavailability: float = 1.0           # 生物利用度


@dataclass
class ClinicalConfig:
    """临床评估参数"""
    recist_evaluation_interval: int = 42   # RECIST评估间隔 (天, ~6周)
    ctcae_version: str = "5.0"             # CTCAE版本
    baseline_measurement_day: int = 0      # 基线测量日


@dataclass
class ConfluenciaConfig:
    """Confluencia集成参数"""
    enabled: bool = False                   # 是否启用Confluencia集成
    confluencia_path: str = ""              # Confluencia项目路径
    drug_prediction_model: str = "moe_ensemble"  # 药物预测模型
    pk_model_type: str = "rnactm"           # PK模型类型
    joint_eval_weights: Dict[str, float] = field(default_factory=lambda: {
        "clinical": 0.4, "binding": 0.35, "kinetics": 0.25
    })


@dataclass
class CircRNAConfig:
    """circRNA 子系统参数"""
    enabled: bool = True                          # 是否启用circRNA子系统
    immunogenicity_backend: str = "heuristic"     # heuristic/vienna/esm2
    mhc_backend: str = "local"                    # local/netmhcpan
    drug_backend: str = "local"                   # local/chembl_api
    pk_backend: str = "rnactm"                    # rnactm（内化）/2.0-bridge/fallback
    enable_structure_prediction: bool = True      # 启用ViennaRNA结构预测
    enable_folding_kinetics: bool = False         # 启用折叠动力学
    enable_torusfold: bool = False                # 启用TorusFold DL评估 (需GPU)
    # 进化配置
    evolution_backend: str = "internal"           # internal/2.0-bridge
    evolution_default_rounds: int = 5             # 默认进化轮数
    evolution_default_objective: str = "ips"      # ips/stability/translation/immune_safety
    # PK 配置
    pk_default_horizon: int = 168                 # 默认 PK 模拟时长 (h)
    pk_default_dt: float = 1.0                    # 默认 PK 时间步长 (h)
    viennarna_timeout_ms: int = 5000              # ViennaRNA超时


@dataclass
class ExperimentConfig:
    """实验参数"""
    max_steps: int = 365                   # 最大步数 (天)
    step_size: float = 1.0                 # 步长 (天)
    n_groups: int = 1                      # 实验组数
    seed: int = 42                         # 随机种子
    log_interval: int = 10                 # 日志间隔 (步)


@dataclass
class Confluencia3Config:
    """Confluencia 3.0 统一平台总配置（TNBC模拟 + circRNA子系统）"""
    tumor: TumorConfig = field(default_factory=TumorConfig)
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)
    csc: CSCConfig = field(default_factory=CSCConfig)
    angiogenesis: AngiogenesisConfig = field(default_factory=AngiogenesisConfig)
    metastasis: MetastasisConfig = field(default_factory=MetastasisConfig)
    immune: ImmuneConfig = field(default_factory=ImmuneConfig)
    evasion: EvasionConfig = field(default_factory=EvasionConfig)
    caf: CAFConfig = field(default_factory=CAFConfig)
    drug_pipeline: DrugPipelineConfig = field(default_factory=DrugPipelineConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    confluencia: ConfluenciaConfig = field(default_factory=ConfluenciaConfig)
    circrna: CircRNAConfig = field(default_factory=CircRNAConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # 分子亚型预设
    molecular_subtype: str = "BLIS"        # BLIS | IM | M | LAR

    # BRCA状态
    brca_mutation: bool = False            # 是否有BRCA1/2突变

    # 患者基本信息
    patient_age: int = 55                  # 患者年龄
    tumor_stage: str = "II"                # I | II | III | IV