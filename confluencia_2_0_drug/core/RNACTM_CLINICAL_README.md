# RNACTM 临床级升级文档

## 概述

RNACTM (circRNA Six-Compartment Pharmacokinetic Model) 已升级至临床级标准，具备以下能力：

1. **数据层** (`pk_data_layer.py`): 标准化 PK 数据格式、文献数据挖掘、合成数据生成
2. **模型层** (`pk_model_layer.py`): PopPK 非线性混合效应模型、个体间变异、协变量分析
3. **验证层** (`pk_validation_layer.py`): 内部/外部验证、VPC、监管合规检查
4. **工程层** (`pk_engineering_layer.py`): 可视化诊断、报告生成、模型比较

## 文件清单

```
confluencia-2.0-drug/core/
├── pk_data_layer.py       # PK 数据标准化 (Phase 1)
├── pk_model_layer.py      # PopPK 模型 (Phase 2)
├── pk_validation_layer.py # 验证框架 (Phase 3)
└── pk_engineering_layer.py # 可视化与报告 (Phase 4)
```

## 快速开始

### 1. 数据准备

```python
from pk_data_layer import (
    PKSample, PopulationPKData,
    SyntheticPKGenerator, LiteraturePKExtractor,
    create_synthetic_dataset, create_literature_dataset,
)

# 方式1: 使用合成数据（开发测试）
synth_data = create_synthetic_dataset(n_subjects=30, seed=42)

# 方式2: 从文献编译数据
lit_data = create_literature_dataset()

# 方式3: 从文件加载
from pk_data_layer import load_pk_dataset
data = load_pk_dataset('path/to/your/pk_data.json')

# 转为 DataFrame
df = data.to_dataframe()
```

### 2. 模型拟合

```python
from pk_model_layer import (
    OneCompartmentModel, RNACTMModel,
    PKParameters, PopPKFitter,
    fit_population_pk
)

# 方式1: 便捷函数
fit_result = fit_population_pk(
    df,
    model_type='1cmt',      # 或 'rnactm_6cmt'
    bootstrap=True,
    n_bootstrap=200,
    verbose=True,
)

# 方式2: 详细控制
model = RNACTMModel(extended=False)
initial_params = PKParameters(
    tv_ka=0.1, tv_ke=0.12, tv_v=2.0, tv_f=0.02,
    omega_ka=0.4, omega_ke=0.3, omega_v=0.25, omega_f=0.5,
    sigma_prop=0.2, sigma_add=0.05,
)

fitter = PopPKFitter(model)
fit_result = fitter.fit(df, initial_params=initial_params)
```

### 3. 验证

```python
from pk_validation_layer import (
    run_full_validation,
    InternalValidator, ExternalValidator,
    VPCAnalyzer, RegulatoryComplianceChecker,
    ValidationLevel,
)

# 完整验证流程
validation_results = run_full_validation(
    fit_result,
    model=model,
    params=params,
    external_data=external_df,  # 可选
    level=ValidationLevel.EXTERNAL,
    n_bootstrap=500,
    n_vpc_sim=1000,
    verbose=True,
)

# 验证标准
REGULATORY_THRESHOLDS = {
    ValidationLevel.INTERNAL: {
        'peak_time_error': 25,  # %
        'cmax_error': 30,
        'auc_error': 30,
        'r2': 0.70,
    },
    ValidationLevel.EXTERNAL: {
        'peak_time_error': 15,
        'cmax_error': 20,
        'auc_error': 20,
        'r2': 0.85,
    },
    ValidationLevel.REGULATORY: {
        'peak_time_error': 10,
        'cmax_error': 15,
        'auc_error': 15,
        'r2': 0.90,
    },
}
```

### 4. 可视化与报告

```python
from pk_engineering_layer import (
    PKPlotter, PKReportGenerator,
    ModelComparator, SensitivityAnalyzer,
)

# 绘图
plotter = PKPlotter(figsize=(10, 8), dpi=150)

# 观测 vs 预测
img = plotter.plot_observed_vs_predicted(obs, pred)

# VPC 图
img = plotter.plot_vpc(time_bins, time_labels, obs_pcts, sim_pcts)

# 参数分布
img = plotter.plot_parameter_distributions(param_names, param_values)

# 生成报告
report_gen = PKReportGenerator(
    fit_result=fit_result,
    validation_results=validation_results,
    model_name="RNACTM Six-Compartment Model",
)

# HTML 报告
html = report_gen.generate_html_report(
    output_path='pk_report.html',
    include_plots=True,
    include_tables=True,
)

# JSON 报告
json_report = report_gen.generate_json_report()

# 模型比较
comparator = ModelComparator()
comparator.add_model('Model A', fit_result_A, val_result_A)
comparator.add_model('Model B', fit_result_B, val_result_B)
comparison_df = comparator.compare()
```

## 数据格式

### PKSample

```python
@dataclass
class PKSample:
    sample_id: str           # 样本 ID
    subject_id: str         # 受试者 ID
    dose: float             # 剂量 (μg/kg)
    route: DeliveryRoute    # 给药途径 (IV/IM/SC)
    modification: NucleotideModification  # 核苷酸修饰
    delivery_vector: str    # 递送载体 (LNP_standard/AAV/naked)

    # 可选
    gc_content: float = 0.5       # GC 含量
    weight_kg: float = 20.0       # 体重
    species: str = "mouse"         # 物种
    observations: List[PKObservation]  # 观察数据

    def add_observation(self, time_h, concentration, tissue='plasma'):
        ...

    def compute_auc(self, tissue='plasma') -> float:
        ...

    def compute_cmax_tmax(self, tissue='plasma') -> Tuple[float, float]:
        ...
```

### PopulationPKData

```python
@dataclass
class PopulationPKData:
    study_id: str
    study_title: str
    samples: List[PKSample]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为长格式 DataFrame"""

    def to_nca_summary(self) -> pd.DataFrame:
        """NCA 汇总表"""

    def save(self, path: str):
        """保存为 JSON"""

    @classmethod
    def load(cls, path: str) -> 'PopulationPKData':
        """从 JSON 加载"""
```

## 模型结构

### 1. 单室模型 (OneCompartmentModel)

```
单次给药:
  C(t) = (Dose/V) × exp(-ke × t)          [IV]
  C(t) = (F×ka×Dose/V)/(ka-ke) × [exp(-ke×t) - exp(-ka×t)]  [血管外]

参数:
  ka: 吸收速率 (1/h)
  ke: 消除速率 (1/h)
  V:  分布容积 (L/kg)
  F:  生物利用度分数
```

### 2. RNACTM 六室模型 (RNACTMModel)

```
房室结构:
  Inj → LNP → Endo → Cyto → Prot → Clear
           ↓
        Liver/Spleen (组织分布)

ODE 系统:
  dInj/dt = -k_release × Inj
  dLNP/dt = k_release × Inj - k_release × LNP
  dEndo/dt = k_release × LNP - k_escape × Endo
  dCyto/dt = k_escape × Endo - (k_degrade + k_translate + k_immune_clear) × Cyto
  dProt/dt = k_translate × Cyto - k_protein_deg × Prot
  dClear/dt = k_degrade × Cyto + k_protein_deg × Prot
```

## 参数说明

### 群体参数 (PKParameters)

| 参数 | 典型值 | 单位 | 说明 |
|------|--------|------|------|
| tv_ka | 0.10 | 1/h | 吸收速率常数 |
| tv_ke | 0.12 | 1/h | 消除速率常数 |
| tv_v | 2.0 | L/kg | 分布容积 |
| tv_f | 0.02 | - | 生物利用度分数 |
| omega_ka | 0.40 | CV% | ka 个体间变异 |
| omega_ke | 0.30 | CV% | ke 个体间变异 |
| omega_v | 0.25 | CV% | V 个体间变异 |
| omega_f | 0.50 | CV% | F 个体间变异 |
| sigma_prop | 0.20 | CV% | 比例残差变异 |
| sigma_add | 0.05 | ng/mL | 加法残差变异 |

### 协变量模型

```python
# 体重影响 (异速缩放)
V = TV_V × (weight/70)^1.0
ke = TV_ke × (weight/70)^0.75

# 修饰效应
ke_psi = ke × 0.40    # Ψ 修饰使 ke 降低 60%
ke_m6a = ke × 0.56    # m6A 修饰使 ke 降低 44%

# 给药途径影响
ka_iv = TV_ka × 1.0   # IV
ka_im = TV_ka × 0.5   # IM
ka_sc = TV_ka × 0.4   # SC
```

## 验证标准

### 内部验证 (Internal)

| 指标 | 阈值 | 说明 |
|------|------|------|
| R² | ≥ 0.70 | 拟合优度 |
| RMSE | - | 均方根误差 |
| AUC 误差 | < 30% | 与观测 AUC 对比 |
| Cmax 误差 | < 30% | 与观测 Cmax 对比 |
| 参数 CV% | < 30% | Bootstrap 参数稳定性 |

### 外部验证 (External)

| 指标 | 阈值 | 说明 |
|------|------|------|
| R² | ≥ 0.85 | 外部数据拟合优度 |
| AUC 误差 | < 20% | 外部数据 AUC 对比 |
| Cmax 误差 | < 20% | 外部数据 Cmax 对比 |
| 组织分布相关性 | r ≥ 0.85 | 生物分布数据相关性 |

### 监管验证 (Regulatory)

| 指标 | 阈值 | 说明 |
|------|------|------|
| R² | ≥ 0.90 | 最高标准 |
| VPC 一致率 | > 90% | 模拟与观测百分位重叠率 |
| 95% CI 覆盖率 | > 95% | 个体预测区间覆盖率 |

## 下一步

### 获取真实数据

1. **合作伙伴**: 联系 circRNA 公司（Laronde、Orna Therapeutics）
2. **文献挖掘**: 从已发表论文中提取 PK 曲线数据
3. **湿实验**: 荧光素酶报告基因 + 活体成像

### 模型扩展

```python
# 添加新的房室（如缓释、靶向）
class ExtendedRNACTM(PKModel):
    def __init__(self):
        self.compartments = [
            'injection', 'lnp', 'endosome',
            'cytoplasm', 'lysosome',  # 新增溶酶体
            'protein', 'clearance',
        ]

# 添加新的协变量
class PKParameters:
    # ... 现有参数 ...

    # 新增协变量
    beta_age_cl: float = -0.3     # 年龄对 CL 的影响
    beta_sex_v: float = 0.1        # 性别对 V 的影响
```

### 法规合规

```python
from pk_validation_layer import RegulatoryComplianceChecker

checker = RegulatoryComplianceChecker(fit_result)

# FDA 合规检查
fda_compliance = checker.check_fda_compliance()
print(f"FDA Compliant: {fda_compliance['overall_compliance']}")

# EMA 合规检查
ema_compliance = checker.check_ema_compliance()
print(f"EMA Compliant: {ema_compliance['overall_compliance']}")

# 生成合规报告
report = checker.generate_compliance_report(agency='FDA')
```

## 参考文献

1. Wesselhoeft et al. (2018) Nat Commun 9:2629 - circRNA 半衰期
2. Liu et al. (2023) Nat Commun 14:2548 - 修饰 circRNA 疗效
3. Chen et al. (2019) Nature 586:651-655 - m6A 效应
4. Gilleron et al. (2013) Nat Biotechnol 31:638-646 - 内体逃逸
5. Paunovska et al. (2018) ACS Nano 12:8307-8320 - 组织分布
6. Hassett et al. (2019) Mol Ther 27:1885-1897 - LNP 动力学
7. FDA Guidance on Population Pharmacokinetics (2022)
8. ICH E4: Dose-Response Information to Support Drug Registration

## 联系方式

- GitHub: https://github.com/IGEM-FBH/confluencia
- Email: igem@fbh-china.org
