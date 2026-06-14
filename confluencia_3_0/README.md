# Confluencia 3.0 — 统一架构：circRNA + TNBC Simulacrum

> EventBus-first 多子系统仿真：肿瘤/TME/治疗/circRNA/生物标志物/临床 六维联合评估

## 定位

Confluencia 3.0 将 TNBC simulacrum（肿瘤微环境仿真）与 circRNA 模块完全整合，以 **EventBus + SubsystemManager** 架构为骨架，统一 6 个子系统的计算管线。与 2.0 版完全区分 — 2.0 是单任务预测模块，3.0 是多维度联合仿真平台。

## 核心架构

```
confluencia_3_0/
├── main.py                        # CLI 入口
├── app.py                         # 统一 Streamlit 前端
│
├── core/
│   ├── agent.py                   # Confluencia3Agent 主编排器
│   ├── config.py                  # Confluencia3Config + CircRNAConfig
│   ├── event_bus.py               # EventBus pub/sub (34+ 事件类型)
│   ├── events.py                  # 事件常量定义
│   ├── state_schema.py            # 状态模式 (~180 键，前缀命名空间)
│   ├── subsystem_managers.py      # 6 个 Manager 协调子模块
│   ├── backend_architecture.py    # ConfluenciaEvaluator + Backend 插件
│   ├── external_backends.py       # ViennaRNA/ESM2/NetMHCpan 外部后端
│   │
│   ├── tumor/                     # 肿瘤生物学 (5 子模块)
│   │   ├── growth_engine.py       # 生长动力学
│   │   ├── heterogeneity.py       # 异质性建模
│   │   ├── cancer_stem_cell.py    # CSC 模拟
│   │   ├── angiogenesis.py        # 血管生成
│   │   └── metastasis.py          # 转移动力学
│   │
│   ├── tme/                       # 肿瘤微环境 (4 子模块)
│   │   ├── immune_dynamics.py     # 免疫细胞动力学
│   │   ├── fibroblast.py          # CAF 建模
│   │   ├── immune_evasion.py      # 免疫逃逸机制
│   │   └── immunoediting.py       # 免疫编辑三阶段
│   │
│   ├── treatment/                 # 治疗引擎 (6 子模块)
│   │   ├── chemotherapy.py        # 化疗
│   │   ├── immunotherapy.py       # 免疫治疗 (PD-1/CTLA-4)
│   │   ├── targeted.py            # 靶向治疗
│   │   ├── radiotherapy.py        # 放疗
│   │   ├── circrna_therapy.py     # circRNA 治疗 (增强版)
│   │   └── drug_pipeline/         # 药物管线
│   │       ├── drug_registry.py   # 药物注册表
│   │       └── pkpd.py            # PK/PD 模拟
│   │
│   ├── circrna/                   # circRNA 子系统 (16 子模块)
│   │   ├── immune_sensing.py      # 免疫感知 (PKR/RIG-I/TLR)
│   │   ├── structure_prediction.py # 结构预测 (ViennaRNA/fallback)
│   │   ├── folding_kinetics.py    # 折叠动力学
│   │   ├── cotrans_folding.py     # 共转录折叠
│   │   ├── folding_pathways.py    # 折叠路径
│   │   ├── drug_response.py       # 药物响应预测
│   │   ├── rna_docking.py         # RNA-药物对接
│   │   ├── rna_modifications.py   # RNA 修饰
│   │   ├── clinical_prediction.py # 临床预测
│   │   ├── cirrna_evolution.py    # 序列进化优化
│   │   ├── bsj_features.py        # BSJ 特征提取
│   │   ├── circrna_rl_abm.py      # RL-ABM 策略学习
│   │   ├── tme_simulation.py      # TME 模拟
│   │   ├── multi_drug_combination.py # 多药组合
│   │   ├── patient_stratification.py # 患者分层
│   │   └── adaptive_dosing.py     # 自适应给药
│   │
│   ├── encoder/                   # 序列编码器 (7 文件)
│   │   ├── tokenizer.py           # RNA 序列 tokenizer
│   │   ├── model.py               # 编码模型
│   │   ├── adapter.py             # 适配层
│   │   ├── train.py               # 训练脚本
│   │   └── config.py              # 编码器配置
│   │
│   ├── pipeline/                  # 统一管线
│   │   └── circrna_pipeline.py    # 通过 Backend 调度
│   │
│   ├── biomarker/                 # 生物标志物 (3 子模块)
│   │   ├── tracker.py             # 标志物追踪
│   │   ├── subtype_classifier.py  # TNBC 亚型分类
│   │   └ resistance_detector.py   # 耐药检测
│   │
│   ├── clinical/                  # 临床评估 (3 子模块)
│   │   ├── recist.py              # RECIST 评估
│   │   ├── survival.py            # 生存分析
│   │   └── toxicity.py            # 毒性评估
│   │
│   └── confluencia/               # 2.0 桥接 (4 个懒加载)
│       ├── drug_bridge.py         # DrugPredictionBridge
│       ├── pk_bridge.py           # PKModelBridge
│       ├── epitope_bridge.py      # EpitopePredictionBridge
│       └── joint_bridge.py        # JointEvaluationBridge
│
├── frontend/                      # Streamlit 前端 (10 个 tab)
│   ├── tabs/
│   │   ├── tumor_dashboard.py     # 肿瘤仪表盘
│   │   ├── tme_immune.py          # TME/免疫
│   │   ├── treatment.py           # 治疗方案
│   │   ├── circrna_analysis.py    # circRNA 分析 ← 新
│   │   ├── circrna_design.py      # circRNA 设计 ← 新
│   │   ├── circrna_vaccine.py     # circRNA 疫苗 ← 新
│   │   ├── biomarker.py           # 生物标志物
│   │   ├── clinical.py            # 临床评估
│   │   ├── experiments.py         # 实验框架
│   │   └── confluencia_bridge.py  # 2.0 桥接界面
│   └── svgs/                      # 可视化 SVG
│
├── experiment/                    # 实验框架
│   ├── sandbox.py                 # Sandbox 模式
│   ├── clinical_trial.py          # 临床试验模拟
│   ├── combination.py             # 组合疗法
│   └ resistance_evolution.py      # 耐药演化
│   └ biomarker_stratification.py  # 标志物分层
│
├── experiments/                   # 预定义实验
│
├── utils/
│   └ confluencia_loader.py        # 2.0 模块加载
│
├── tests/
│   ├── test_agent.py              # Agent 测试
│   ├── test_circrna_manager.py    # CircRNAManager 测试
│   ├── test_backend_architecture.py # Backend 测试
│   └ test_pipeline_unified.py     # 管线测试
│   └ ...
│
└── data/
    └ reference/
        └ scoring_weights_literature.json
```

## 核心创新

| 创新点 | 说明 |
|--------|------|
| **EventBus-first 架构** | 34+ 事件类型，pub/sub 解耦子系统，能量高效激活 |
| **SubsystemManager 模式** | 6 个 Manager (Tumor/TME/Treatment/CircRNA/Biomarker/Clinical) 协调 37+ 子模块 |
| **状态前缀命名空间** | ~180 状态键，`t_*`, `tme_*`, `tx_*`, `crna_*`, `bm_*`, `cl_*` 分组 |
| **Backend 三层降级** | esm2 → vienna → heuristic，离线优先原则 |
| **统一管线到 Backend** | CircRNAPipeline 通过 ConfluenciaEvaluator 调度，消除架构断层 |
| **懒加载 2.0 桥接** | Drug/Epitope/PK/Joint 四桥接静默失败降级 |

## 事件类型

### circRNA 新增事件 (6 个)

| 事件常量 | 值 | 触发场景 |
|---|---|---|
| `CIRCRNA_IMMUNE_EVAL` | `"circrna_immune_eval"` | 免疫原性评估请求 |
| `CIRCRNA_STRUCTURE_PREDICT` | `"circrna_structure_predict"` | 结构预测请求 |
| `CIRCRNA_SEQUENCE_EVOLVE` | `"circrna_sequence_evolve"` | 序列进化优化请求 |
| `CIRCRNA_VACCINE_ASSESS` | `"circrna_vaccine_assess"` | 疫苗综合评估请求 |
| `CIRCRNA_FOLDING_KINETICS` | `"circrna_folding_kinetics"` | 折叠动力学请求 |
| `CIRCRNA_DRUG_RESPONSE` | `"circrna_drug_response"` | 药物响应预测请求 |

### circRNA 状态键 (12 个)

| 键 | 类型 | 默认值 | 范围 | 描述 |
|---|---|---|---|---|
| `crna_immunogenicity_score` | float | 0.0 | [0, 1] | circRNA 免疫原性评分 |
| `crna_ips_score` | float | 0.0 | [0, 10] | IPS 评分 |
| `crna_structure_method` | str | "none" | — | 结构预测方法 |
| `crna_mfe_kcal` | float | 0.0 | — | MFE 自由能 |
| `crna_pkr_score` | float | 0.0 | [0, 1] | PKR 评分 |
| `crna_rig_i_score` | float | 0.0 | [0, 1] | RIG-I 评分 |
| `crna_tlr_score` | float | 0.0 | [0, 1] | TLR 评分 |
| `crna_vaccine_therapeutic_window` | float | 0.0 | [0, 1] | 治疗窗口 |
| `crna_evolution_generation` | int | 0 | — | 进化代数 |
| `crna_evolution_best_score` | float | 0.0 | [0, 1] | 当前最优序列评分 |
| `crna_folding_method` | str | "none" | — | 折叠方法 |
| `crna_backend_tier` | str | "heuristic" | — | 当前后端层级 |

## 配置

```python
@dataclass
class CircRNAConfig:
    enabled: bool = True
    immunogenicity_backend: str = "heuristic"  # heuristic/vienna/esm2
    mhc_backend: str = "local"                  # local/netmhcpan
    drug_backend: str = "local"                 # local/chembl_api
    pk_backend: str = "rnactm"                  # rnactm/external
    enable_structure_prediction: bool = True
    enable_folding_kinetics: bool = False
    evolution_generations: int = 50
    evolution_objective: str = "ips"            # ips/immunogenicity/translation
    viennarna_timeout_ms: int = 5000

@dataclass
class Confluencia3Config:
    # ... TNBC simulacrum 配置 ...
    circrna: CircRNAConfig = field(default_factory=CircRNAConfig)
    confluencia: ConfluenciaConfig = field(default_factory=ConfluenciaConfig)
```

## 快速启动

### CLI 入口
```bash
cd confluencia_3_0
pip install -r requirements.txt
python main.py --steps 100 --subtype basal-like --circrna-backend heuristic
```

### Streamlit 前端
```bash
streamlit run app.py
```

### 禁用 circRNA
```bash
python main.py --no-circrna --steps 50
```

## CircRNAManager

```python
class CircRNAManager:
    """管理 circRNA 免疫感知、结构预测、序列设计、疫苗评估子系统。"""

    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config.circrna
        self.event_bus = agent._event_bus
        self.backend = agent.backend_manager
        self._current_sequence = ""
        self._subscribe_events()

    def step(self):
        """每步执行 circRNA 相关计算，更新 crna_* 状态键。"""
        if not self.config.enabled:
            return {}
        # ... 计算逻辑 ...

    def assess_immunogenicity(self, sequence, backend="heuristic"):
        """通过 Backend 架构调度免疫原性评估。"""
        evaluator = self.backend.get_immunogenicity_backend(backend)
        return evaluator.predict(sequence)

    def predict_structure(self, sequence):
        """通过 Backend 调度结构预测 (ViennaRNA/fallback)。"""
        ...

    def evolve_sequence(self, sequence, objective):
        """序列进化优化 (cirrna_evolution)。"""
        ...
```

## Backend 架构

### 三层降级

```
esm2 (最高精度，需 GPU)
  ↓ 失败/超时
vienna (中等精度，CPU 可用)
  ↓ 失败/超时
heuristic (最低精度，纯 Python)
```

### Backend 类型

| Backend | 用途 | 实现位置 |
|---|---|---|
| `ImmunogenicityBackend` | 免疫原性预测 | `heuristic`, `vienna`, `esm2` |
| `MHCBackend` | MHC 结合预测 | `local`, `netmhcpan` |
| `DrugBackend` | 药物响应预测 | `local`, `chembl_api` |
| `PKBackend` | PK 模拟 | `rnactm`, `external` |

## 与 2.0 版关系

| 方面 | 2.0 | 3.0 |
|------|-----|-----|
| **定位** | 单任务预测 | 多维度联合仿真 |
| **架构** | Pipeline 直接调用 | EventBus + Manager |
| **circRNA** | 独立模块 | 统一子系统 |
| **TNBC** | 无 | simulacrum 核心 |
| **桥接** | 无 | 4 个懒加载桥接 |

```
confluencia_3_0
    ├── core/confluencia/
    │   ├── drug_bridge.py      → confluencia-2.0-drug
    │   ├── epitope_bridge.py   → confluencia-2.0-epitope
    │   ├── pk_bridge.py        → confluencia-2.0-drug/core/ctm.py
    │   └ joint_bridge.py       → confluencia_joint
    │
    └── 2.0 模块独立运行，3.0 通过桥接调用
```

## 依赖

```toml
[核心]
numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, scipy>=1.10
streamlit>=1.30, joblib>=1.3, rich>=13.0

[可选]
torch>=2.0           # ESM-2 编码器
ViennaRNA            # RNA 结构预测 (外部 CLI)
```

## 测试

```bash
pytest confluencia_3_0/tests/ -v
```

覆盖：
- CircRNAManager 基本功能
- EventBus 事件流
- Backend 降级
- 状态键注册
- 配置集成

---

**整合计划详见**：`~/.claude/plans/vivid-percolating-rainbow.md`