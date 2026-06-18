# Confluencia 3.0 完整架构图

**版本**: 3.0.0
**日期**: 2026-06-17
**作者**: 颜子壹 | 吉林大学计算机科学与技术学院 | 吉林大学第一白求恩临床医学院

---

## 一、系统总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CONFLUENCIA 3.0 统一计算平台                                    │
│                    circRNA药物发现 × TNBC数字孪生 × 智能实验设计                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│   🧬 TNBC Simulacrum  │    │  🔬 circRNA Subsystem │    │   🧪 Experiment Lab   │
│    (肿瘤数字孪生)      │    │    (circRNA分析)      │    │    (实验设计)         │
└───────────────────────┘    └───────────────────────┘    └───────────────────────┘
```

---

## 二、目录结构与文件映射

### 根目录结构

```
D:\IGEM集成方案\
│
├── confluencia_3_0/                    # ★ 主模块 (Python包)
│   ├── core/                           # 核心引擎
│   ├── frontend/                       # Streamlit前端
│   ├── experiments/                    # 实验模板
│   ├── experiment/                     # 实验框架
│   ├── utils/                          # 工具函数
│   ├── tests/                          # 单元测试
│   ├── scripts/                        # 辅助脚本
│   └── main.py                         # CLI入口
│
├── confluencia-studio/                 # ★ Streamlit Studio (无代码UI)
│   └── streamlit_app/
│       ├── Home.py
│       └── pages/
│
├── confluencia-2.0-drug/               # 药物预测模块 (后端)
├── confluencia-2.0-epitope/            # 表位预测模块 (后端)
├── confluencia-shared/                 # 共享库
├── confluencia-joint/                  # 联合分析
├── confluencia-circrna/                # circRNA工具
│
├── data/                               # 数据目录
├── output/                             # 输出目录
├── manuscripts/                        # 论文草稿
├── docs/                               # 文档
└── pyproject.toml                      # 项目配置
```

---

## 三、核心架构 (confluencia_3_0/core/)

### 3.1 核心编排层

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CORE ORCHESTRATION                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│  │   agent.py      │    │  event_bus.py   │    │  events.py      │                     │
│  │ TNBCSimulacrum  │◄──►│   EventBus      │    │  事件常量定义    │                     │
│  │   主编排器       │    │   pub/sub       │    │  (40+事件)      │                     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│          │                      │                                                       │
│          ▼                      ▼                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│  │ state_schema.py │    │   config.py     │    │external_backends│                     │
│  │ StateSchema     │    │ Confluencia3    │    │    .py          │                     │
│  │ 170+状态键定义   │    │ Config          │    │ 外部API桥接      │                     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 文件映射表

| 文件路径 | 功能 | 关键类/函数 |
|---------|------|------------|
| `core/agent.py` | TNBC模拟主编排器 | `TNBCSimulacrum` |
| `core/event_bus.py` | 事件总线 (pub/sub) | `EventBus`, `Event`, `Subscription` |
| `core/events.py` | 事件类型常量 | `STEP_START`, `DRUG_ADMINISTERED`, `CIRCRNA_IMMUNE_EVAL` 等40+ |
| `core/state_schema.py` | 状态模式定义 | `StateSchema`, `KeyDef` (170+状态键) |
| `core/config.py` | 运行时配置 | `Confluencia3Config`, `CircRNAConfig`, `TumorConfig` 等12个dataclass |
| `core/backend_architecture.py` | 可插拔后端架构 | `ConfluenciaEvaluator`, `ImmunogenicityBackendBase` |
| `core/external_backends.py` | 外部API集成 | NetMHCpan, ESM-2, ViennaRNA桥接 |

---

## 四、子系统管理器 (Subsystem Managers)

### 4.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SUBSYSTEM MANAGERS (subsystem_managers.py)                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   SubsystemManager (基类)                                                               │
│   ├── agent: TNBCSimulacrum 引用                                                        │
│   ├── schema: StateSchema 引用                                                          │
│   ├── state: Dict[str, Any] 状态访问                                                    │
│   └── step() -> Dict                                                                    │
│                                                                                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│   │TumorManager │  │ TMEManager  │  │TreatmentMgr │  │BiomarkerMgr │  │ ClinicalMgr │  │
│   │   tumor     │  │    tme      │  │  treatment  │  │  biomarker  │  │  clinical   │  │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│          │                │                │                │                │          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                        │
│   │CircRNAManager│  │             │  │             │                                        │
│   │  circrna    │  │   (未来)     │  │   (未来)     │                                        │
│   └─────────────┘  └─────────────┘  └─────────────┘                                        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 文件映射

| 文件路径 | 功能 | 管理器类 |
|---------|------|---------|
| `core/subsystem_managers.py` | 子系统管理器集合 | `TumorManager`, `TMEManager`, `TreatmentManager`, `BiomarkerManager`, `ClinicalManager`, `CircRNAManager` |

---

## 五、肿瘤子系统 (core/tumor/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    TUMOR SUBSYSTEM                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   tum_* 状态键命名空间 (10个)                                                            │
│   ├── tum_volume, tum_growth_rate, tum_apoptosis_rate                                   │
│   ├── tum_necrosis_fraction, tum_proliferation_index                                    │
│   └── tum_cell_count, tum_oxygenation, tum_glucose_level, tum_lactate_level, tum_ph     │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │ growth_engine.py│    │ heterogeneity.py│    │cancer_stem_cell │                     │
│   │ TumorGrowth     │    │ TumorHetero     │    │ CancerStemCell  │                     │
│   │ Engine          │    │ geneity         │    │ Pool            │                     │
│   │ Logistic/       │    │ 亚克隆演化       │    │ CSC动态         │                     │
│   │ Gompertz        │    │ Shannon多样性    │    │ CD44/CD24       │                     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐                                           │
│   │ angiogenesis.py │    │  metastasis.py  │                                           │
│   │ Angiogenesis    │    │ Metastasis      │                                           │
│   │ Engine          │    │ Engine          │                                           │
│   │ VEGF/MVD        │    │ EMT/MET         │                                           │
│   │ 血管正常化       │    │ 器官趋向性       │                                           │
│   └─────────────────┘    └─────────────────┘                                           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 文件映射

| 文件路径 | 功能 | 关键类 |
|---------|------|-------|
| `core/tumor/growth_engine.py` | 肿瘤生长引擎 | `TumorGrowthEngine` |
| `core/tumor/heterogeneity.py` | 肿瘤异质性 | `TumorHeterogeneity` |
| `core/tumor/cancer_stem_cell.py` | 癌干细胞池 | `CancerStemCellPool` |
| `core/tumor/angiogenesis.py` | 血管生成 | `AngiogenesisEngine` |
| `core/tumor/metastasis.py` | 转移引擎 | `MetastasisEngine` |

---

## 六、肿瘤微环境子系统 (core/tme/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      TME SUBSYSTEM                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   imm_* 状态键 (19个) + evs_* (6个) + ied_* (4个) + caf_* (6个)                          │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │immune_dynamics.py│   │ immune_evasion.py│   │immunoediting.py │                     │
│   │ ImmuneCell      │    │ ImmuneEvasion   │    │ Immunoediting    │                     │
│   │ Dynamics        │    │ PD-L1/MHC-I     │    │ 三阶段转换       │                     │
│   │ CD8/CD4/NK/Treg │    │ TGF-β/IDO       │    │ Elimination→    │                     │
│   │ M1/M2/MDSC      │    │ Galectin-9      │    │ Equilibrium→    │                     │
│   └─────────────────┘    └─────────────────┘    │ Escape           │                     │
│                          ┌─────────────────┐    └─────────────────┘                     │
│                          │  fibroblast.py  │                                            │
│                          │ Fibroblast      │                                            │
│                          │ Activation      │                                            │
│                          │ CAF/ECM/胶原    │                                            │
│                          └─────────────────┘                                            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 文件映射

| 文件路径 | 功能 | 关键类 |
|---------|------|-------|
| `core/tme/immune_dynamics.py` | 免疫细胞动态 | `ImmuneCellDynamics` |
| `core/tme/immune_evasion.py` | 免疫逃逸机制 | `ImmuneEvasion` |
| `core/tme/immunoediting.py` | 免疫编辑三阶段 | `Immunoediting` |
| `core/tme/fibroblast.py` | 成纤维细胞/ECM | `FibroblastActivation` |

---

## 七、治疗子系统 (core/treatment/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TREATMENT SUBSYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   drg_* 状态键 (12个)                                                                    │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │ chemotherapy.py │    │ immunotherapy.py│    │  targeted.py    │                     │
│   │ Chemotherapy    │    │ Immunotherapy   │    │ TargetedTherapy │                     │
│   │ Engine          │    │ Engine          │    │ Engine          │                     │
│   │ 蒽环类/紫杉类   │    │ PD-1/PD-L1      │    │ PARP/AKT/PI3K   │                     │
│   │ 剂量密度        │    │ CAR-T           │    │ 靶向抑制剂       │                     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐             │
│   │ radiotherapy.py │    │circrna_therapy.py│   │ drug_pipeline/          │             │
│   │ Radiotherapy    │    │ CircRNATherapy  │    │ ├── drug_registry.py    │             │
│   │ Engine          │    │ Engine          │    │ └── pkpd.py             │             │
│   │ 外照射/近距离   │    │ circRNA疫苗     │    │ 药物注册/PKPD模型       │             │
│   │ 远隔效应        │    │ 序列优化        │    │                         │             │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────┘             │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 文件映射

| 文件路径 | 功能 | 关键类 |
|---------|------|-------|
| `core/treatment/chemotherapy.py` | 化疗引擎 | `ChemotherapyEngine` |
| `core/treatment/immunotherapy.py` | 免疫治疗引擎 | `ImmunotherapyEngine` |
| `core/treatment/targeted.py` | 靶向治疗引擎 | `TargetedTherapyEngine` |
| `core/treatment/radiotherapy.py` | 放疗引擎 | `RadiotherapyEngine` |
| `core/treatment/circrna_therapy.py` | circRNA治疗引擎 | `CircRNATherapyEngine` |
| `core/treatment/drug_pipeline/drug_registry.py` | 药物注册表 | `DrugRegistry` |
| `core/treatment/drug_pipeline/pkpd.py` | PK/PD模型 | `PKPDModel` |

---

## 八、circRNA子系统 (core/circrna/) ★ 核心

### 8.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              CIRCRNA SUBSYSTEM (四大支柱)                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   crna_* 状态键 (30+个)                                                                  │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           CircRNAManager (subsystem_managers.py)                 │   │
│   │  - assess_immunogenicity()  - predict_structure()  - evolve_sequence()          │   │
│   │  - assess_with_torusfold()  - simulate_pk()        - evolve_molecules()          │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                              │
│        ┌─────────────────┬───────────────┼───────────────┬─────────────────┐           │
│        ▼                 ▼               ▼               ▼                 ▼           │
│   ┌─────────┐      ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│   │ RNACTM  │      │ViennaRNA │    │TorusFold │    │Simulacrum│    │ Evolution│       │
│   │ PK/PD   │      │ 二级结构  │    │ DL结构   │    │ TNBC响应 │    │ 序列进化 │       │
│   └─────────┘      └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 circRNA模块详细

```
core/circrna/
│
├── __init__.py
├── immune_sensing.py          # ★ 免疫原性评估 (RIG-I/TLR/PKR)
├── structure_prediction.py    # ★ 结构预测 (ViennaRNA/fallback)
├── torusfold_scorer.py        # ★ TorusFold评分器
├── folding_kinetics.py        # 折叠动力学
├── folding_pathways.py        # 折叠路径
├── cotrans_folding.py         # 共转录折叠
├── bsj_features.py            # BSJ特征提取
├── rna_modifications.py       # RNA修饰 (m6A等)
├── rna_docking.py             # RNA对接
├── drug_response.py           # 药物响应预测
├── clinical_prediction.py     # 临床预测
├── patient_stratification.py  # 患者分层
├── adaptive_dosing.py         # 自适应剂量
├── multi_drug_combination.py  # 多药联合
├── tme_simulation.py          # TME模拟
├── circrna_rl_abm.py          # 强化学习ABM
├── cirrna_evolution.py        # circRNA进化
│
└── torusfold/                  # ★ TorusFold深度学习模块
    ├── __init__.py
    ├── torusfold.py           # 主模块 (TorusFoldEncoder)
    ├── diffusion_structure.py # AF3风格扩散模型
    ├── equivariant_backbone.py# 等变骨干网络
    ├── physics_structure_head.py # 物理约束结构头
    ├── constraint_solver.py   # 几何约束求解器
    ├── physics_bridge.py      # OpenMM桥接
    ├── cgmd_refiner.py        # 粗粒化MD精修
    ├── structure_validator.py # 结构验证
    ├── tertiary_interaction.py# 三级相互作用
    ├── irs_pair.py            # IRS配对
    ├── tpe.py                 # TPE位置编码
    └── triangle_update.py     # 三角形更新
```

### 8.3 TorusFold架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    TORUSFOLD DL                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   结构预测模式 (structure_mode):                                                         │
│   ├── "heuristic"  → 不使用TorusFold，走Backend三层降级                                  │
│   ├── "simple"     → SimpleStructureHead (MDS快速推断)                                  │
│   ├── "diffusion"  → CircDiffusionStructure (AF3风格扩散)                               │
│   ├── "physics_b"  → PhysicsStructureHead (几何约束求解器，零训练)                       │
│   └── "physics_ba" → PhysicsStructureHead + OpenMM MD精修                              │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        TorusFoldEncoder (torusfold.py)                           │   │
│   │  输入: RNA序列 (AUGC)                                                            │   │
│   │  输出: 结构信号 + 免疫评分覆盖 + 四维目标                                          │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                              │
│        ┌─────────────────┬───────────────┼───────────────┬─────────────────┐           │
│        ▼                 ▼               ▼               ▼                 ▼           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │Equivariant│    │ Diffusion│    │ Physics  │    │Tertiary  │    │  IRS     │         │
│   │ Backbone  │    │ Structure│    │ Structure│    │Interaction│   │  Pair    │         │
│   │ 等变网络  │    │ 扩散模型 │    │ 几何约束 │    │ 三级相互作用│   │ 内环配对 │         │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 九、生物标志物与临床子系统

### 9.1 生物标志物 (core/biomarker/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  BIOMARKER SUBSYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   bio_* 状态键 (9个)                                                                     │
│   ├── bio_pd_l1_cps, bio_til_density, bio_brca_status                                   │
│   ├── bio_tmb, bio_ctdna_level, bio_msi_status                                          │
│   └── bio_hr_status, bio_pi3k_mutation, bio_androgen_receptor                           │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │   tracker.py    │    │subtype_classifier│   │resistance_      │                     │
│   │ Biomarker       │    │ .py             │    │ detector.py     │                     │
│   │ Tracker         │    │ Molecular       │    │ Resistance      │                     │
│   │ 标志物动态追踪   │    │ Subtype         │    │ Detector        │                     │
│   │                 │    │ Classifier      │    │ 耐药签名检测     │                     │
│   └─────────────────┘    │ BLIS/IM/M/LAR   │    └─────────────────┘                     │
│                          └─────────────────┘                                            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 临床评估 (core/clinical/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLINICAL SUBSYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   cli_* 状态键 (14个)                                                                    │
│   ├── cli_recist_response, cli_tumor_change_pct, cli_baseline_volume                    │
│   ├── cli_pfs_months, cli_os_months, cli_toxicity_grade                                 │
│   └── cli_neutropenia_grade, cli_cardiotoxicity_grade 等                                │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │   recist.py     │    │  survival.py    │    │  toxicity.py    │                     │
│   │ RECISTTracker   │    │ SurvivalModel   │    │ ToxicityGrader  │                     │
│   │ CR/PR/SD/PD     │    │ PFS/OS估计      │    │ CTCAE 5.0       │                     │
│   │ 肿瘤缓解评估     │    │ Kaplan-Meier    │    │ 毒性分级        │                     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十、进化与PK模块

### 10.1 进化模块 (core/evolution/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  EVOLUTION MODULE                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │cirrna_evolution │    │molecule_        │    │   pareto.py     │                     │
│   │ .py             │    │ evolution.py    │    │ ParetoFront     │                     │
│   │ evolve_cirrna   │    │ evolve_         │    │ 多目标优化       │                     │
│   │ 四维目标优化     │    │ molecules_      │    │                 │                     │
│   │ 稳定性/翻译/     │    │ with_reflection │    │                 │                     │
│   │ 免疫逃逸/递送    │    │ 药物分子进化     │    │                 │                     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 PK模块 (core/pk/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PK MODULE                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐                                           │
│   │   rnactm.py     │    │  legacy_ctm.py  │                                           │
│   │ infer_rna_ctm   │    │ LegacyCTM       │                                           │
│   │ _params         │    │ 旧版CTM模型     │                                           │
│   │ simulate_rna    │    │                 │                                           │
│   │ _ctm            │    │                 │                                           │
│   │ circRNA PK模拟  │    │                 │                                           │
│   └─────────────────┘    └─────────────────┘                                           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十一、编码器模块 (core/encoder/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ENCODER MODULE                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   circRNA序列编码器 (预训练/微调)                                                         │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                     │
│   │   model.py      │    │  tokenizer.py   │    │   train.py      │                     │
│   │ RNATransformer  │    │ RNATokenizer    │    │ 训练脚本        │                     │
│   │ BERT-style      │    │ K-mer tokenization│   │ 预训练/微调     │                     │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                     │
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐                                           │
│   │   adapter.py    │    │   config.py     │                                           │
│   │ 下游任务适配器   │    │ EncoderConfig   │                                           │
│   └─────────────────┘    └─────────────────┘                                           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十二、前端与实验模块

### 12.1 前端 (frontend/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND MODULE                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐                                           │
│   │   app.py        │    │  app_core.py    │                                           │
│   │ Streamlit主应用 │    │ 共享基础设施     │                                           │
│   │ 多标签页布局    │    │ Tokyo Night主题  │                                           │
│   └─────────────────┘    │ 状态管理        │                                           │
│                          └─────────────────┘                                           │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      tabs/                                       │   │
│   │  ├── biomarker.py      # 生物标志物标签页                                        │   │
│   │  ├── clinical.py       # 临床评估标签页                                          │   │
│   │  ├── confluencia.py    # Confluencia集成标签页                                   │   │
│   │  ├── experiments.py    # 实验标签页                                              │   │
│   │  ├── tme_immune.py     # TME/免疫标签页                                          │   │
│   │  ├── treatment.py      # 治疗标签页                                              │   │
│   │  └── tumor_dashboard.py# 肿瘤仪表盘                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 实验模板 (experiments/)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXPERIMENT TEMPLATES                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   experiments/                                                                          │
│   ├── experiment_akt_pi3k.py          # AKT/PI3K通路实验                                │
│   ├── experiment_ar_lar.py            # AR/LAR亚型实验                                  │
│   ├── experiment_biomarker_stratified.py # 生物标志物分层                               │
│   ├── experiment_chemo_response.py    # 化疗响应                                        │
│   ├── experiment_circrna_therapy.py   # circRNA治疗                                     │
│   ├── experiment_combination_chemo_immuno.py # 联合化疗免疫                             │
│   ├── experiment_combination_screening.py # 组合筛选                                    │
│   ├── experiment_confluencia_pkpd.py  # Confluencia PK/PD                              │
│   ├── experiment_dose_frequency_matrix.py # 剂量频率矩阵                                │
│   ├── experiment_immune_checkpoint.py # 免疫检查点                                      │
│   ├── experiment_parp_brca.py         # PARP/BRCA实验                                   │
│   ├── experiment_radiotherapy_abscopal.py # 放疗远隔效应                                │
│   ├── experiment_resistance_evolution.py # 耐药演化                                     │
│   └── experiment_subtype_comparison.py # 亚型比较                                       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十三、Confluencia Studio (Streamlit UI)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              CONFLUENCIA STUDIO (无代码UI)                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   confluencia-studio/streamlit_app/                                                     │
│   │                                                                                     │
│   ├── Home.py                       # ★ 主页 - 模块选择                                 │
│   │                                                                                     │
│   └── pages/                                                                            │
│       ├── 1_CircRNA_Analysis.py     # circRNA免疫原性分析                               │
│       ├── 2_Drug_Prediction.py      # 药物ADMET预测                                     │
│       ├── 3_Epitope_Screening.py    # MHC表位筛选                                       │
│       ├── 4_TNBC_Simulator.py       # TNBC数字孪生仿真                                  │
│       ├── 5_Report_Export.py        # HTML报告导出                                      │
│       └── 6_Joint_Analysis.py       # circRNA+药物联合分析                              │
│                                                                                         │
│   功能:                                                                                 │
│   ├── 无需编程的图形化界面                                                               │
│   ├── 序列粘贴 → 一键分析                                                                │
│   ├── 可视化HTML报告生成                                                                 │
│   └── 数据导出 (JSON)                                                                    │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十四、事件总线架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  EVENT BUS ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   EventBus (event_bus.py)                                                               │
│   ├── subscribe(event_type, handler, priority)                                          │
│   ├── publish(event_type, data, source) -> collected                                    │
│   └── 同步调用，返回值收集                                                               │
│                                                                                         │
│   事件类型 (events.py) - 40+ 事件常量:                                                   │
│                                                                                         │
│   生命周期:        STEP_START, STEP_END                                                 │
│   肿瘤:          TUMOR_GROWTH, TUMOR_HETEROGENEITY, TUMOR_ANGIOGENESIS, TUMOR_METASTASIS│
│   TME:           TME_IMMUNE_UPDATE, TME_FIBROBLAST_UPDATE, TME_EVASION_UPDATE           │
│   治疗:          DRUG_ADMINISTERED, DRUG_PK_UPDATE, DRUG_PD_EFFECT, IMMUNOTHERAPY_UPDATE│
│   生物标志物:     BIOMARKER_UPDATE, SUBTYPE_RECLASSIFIED, RESISTANCE_DETECTED           │
│   临床:          RECIST_EVALUATION, SURVIVAL_UPDATE, TOXICITY_UPDATE                    │
│   circRNA:       CIRCRNA_IMMUNE_EVAL, CIRCRNA_STRUCTURE_PREDICT, CIRCRNA_SEQUENCE_EVOLVE│
│   Confluencia:   CONFLUENCIA_DRUG_PREDICTION, CONFLUENCIA_PK_SIMULATION                 │
│                                                                                         │
│   流量示例:                                                                             │
│   agent.step()                                                                          │
│       ├── bus.publish(STEP_START, {day})                                                │
│       ├── mgr_tumor.step() → TUMOR_GROWTH, TUMOR_HETEROGENEITY...                       │
│       ├── mgr_tme.step() → TME_IMMUNE_UPDATE, TME_EVASION_UPDATE...                     │
│       ├── mgr_treatment.step() → DRUG_PK_UPDATE, DRUG_PD_EFFECT...                      │
│       └── bus.publish(STEP_END, {day})                                                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十五、Backend架构 (可插拔后端)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               PLUGGABLE BACKEND ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ConfluenciaEvaluator (backend_architecture.py)                                        │
│   ├── immunogenicity_backend: "heuristic" | "vienna" | "esm2"                          │
│   ├── mhc_backend: "local" (AUC=0.80) | "netmhcpan" (AUC=0.92-0.96)                    │
│   ├── drug_backend: "local" | "external"                                                │
│   └── pk_backend: "rnactm" | "external"                                                 │
│                                                                                         │
│   免疫原性后端对比:                                                                       │
│   ┌──────────────┬───────────┬────────────┬──────────┐                                 │
│   │ Backend      │ Speed     │ Accuracy   │ Offline  │                                 │
│   ├──────────────┼───────────┼────────────┼──────────┤                                 │
│   │ heuristic    │ ~85ms     │ Medium     │ Yes      │                                 │
│   │ vienna       │ ~150ms    │ High       │ Yes      │                                 │
│   │ esm2         │ 2-5s      │ Highest    │ GPU      │                                 │
│   └──────────────┴───────────┴────────────┴──────────┘                                 │
│                                                                                         │
│   MHC后端对比:                                                                          │
│   ┌──────────────┬───────────┬────────────┬──────────┐                                 │
│   │ Backend      │ Speed     │ AUC        │ Offline  │                                 │
│   ├──────────────┼───────────┼────────────┼──────────┤                                 │
│   │ local        │ ~50ms     │ 0.80       │ Yes      │                                 │
│   │ netmhcpan    │ ~200ms    │ 0.92-0.96  │ No       │                                 │
│   └──────────────┴───────────┴────────────┴──────────┘                                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十六、完整文件索引

### 16.1 核心文件清单

| 模块 | 文件 | 行数 | 功能 |
|------|------|------|------|
| **编排层** | `core/agent.py` | 289 | TNBC模拟主编排器 |
| | `core/event_bus.py` | 140 | 事件总线 |
| | `core/events.py` | 79 | 事件类型定义 |
| | `core/state_schema.py` | 320 | 状态模式 (170+键) |
| | `core/config.py` | 210 | 配置dataclass |
| | `core/backend_architecture.py` | 455 | 可插拔后端架构 |
| | `core/subsystem_managers.py` | 641 | 子系统管理器 |
| **肿瘤** | `core/tumor/growth_engine.py` | ~200 | 生长引擎 |
| | `core/tumor/heterogeneity.py` | ~250 | 异质性 |
| | `core/tumor/cancer_stem_cell.py` | ~100 | CSC池 |
| | `core/tumor/angiogenesis.py` | ~120 | 血管生成 |
| | `core/tumor/metastasis.py` | ~100 | 转移 |
| **TME** | `core/tme/immune_dynamics.py` | ~240 | 免疫动态 |
| | `core/tme/fibroblast.py` | ~100 | 成纤维细胞 |
| | `core/tme/immune_evasion.py` | ~140 | 免疫逃逸 |
| | `core/tme/immunoediting.py` | ~190 | 免疫编辑 |
| **治疗** | `core/treatment/chemotherapy.py` | ~190 | 化疗 |
| | `core/treatment/immunotherapy.py` | ~145 | 免疫治疗 |
| | `core/treatment/targeted.py` | ~150 | 靶向治疗 |
| | `core/treatment/radiotherapy.py` | ~105 | 放疗 |
| | `core/treatment/circrna_therapy.py` | ~200 | circRNA治疗 |
| **circRNA** | `core/circrna/immune_sensing.py` | ~640 | 免疫感知 |
| | `core/circrna/structure_prediction.py` | ~480 | 结构预测 |
| | `core/circrna/torusfold_scorer.py` | ~550 | TorusFold评分 |
| | `core/circrna/cirrna_evolution.py` | ~800 | circRNA进化 |
| | `core/circrna/bsj_features.py` | ~1150 | BSJ特征 |
| **TorusFold** | `core/circrna/torusfold/torusfold.py` | ~800 | TorusFold主模块 |
| | `core/circrna/torusfold/diffusion_structure.py` | ~970 | 扩散模型 |
| | `core/circrna/torusfold/equivariant_backbone.py` | ~435 | 等变骨干 |
| | `core/circrna/torusfold/physics_structure_head.py` | ~305 | 物理结构头 |
| **进化** | `core/evolution/cirrna_evolution.py` | ~480 | circRNA进化 |
| | `core/evolution/molecule_evolution.py` | ~385 | 分子进化 |
| **PK** | `core/pk/rnactm.py` | ~410 | RNACTM模型 |
| **编码器** | `core/encoder/model.py` | ~470 | RNA Transformer |
| **前端** | `frontend/app_core.py` | ~465 | 前端核心 |
| | `frontend/app.py` | ~130 | 主应用 |
| **实验** | `experiments/*.py` | 14文件 | 实验模板 |
| **Studio** | `confluencia-studio/streamlit_app/Home.py` | ~205 | Studio主页 |

### 16.2 状态键统计

| 命名空间 | 前缀 | 数量 | 示例 |
|---------|------|------|------|
| 肿瘤 | `tum_*` | 10 | `tum_volume`, `tum_growth_rate` |
| 亚型 | `sub_*` | 8 | `sub_molecular_subtype`, `sub_ar_expression` |
| 异质性 | `het_*` | 5 | `het_n_subclones`, `het_diversity_index` |
| CSC | `csc_*` | 6 | `csc_fraction`, `csc_self_renewal` |
| 血管 | `vasc_*` | 6 | `vasc_vegf_level`, `vasc_microvessel_density` |
| 转移 | `met_*` | 9 | `met_emt_progress`, `met_metastatic_burden` |
| 免疫 | `imm_*` | 19 | `imm_cd8_count`, `imm_t_cell_activation` |
| 逃逸 | `evs_*` | 6 | `evs_pd_l1_expression`, `evs_tgf_beta` |
| 编辑 | `ied_*` | 4 | `ied_phase`, `ied_immune_pressure` |
| CAF | `caf_*` | 6 | `caf_activation`, `caf_ecm_density` |
| 药物 | `drg_*` | 12 | `drg_concentration`, `drg_effect` |
| 标志物 | `bio_*` | 9 | `bio_pd_l1_cps`, `bio_brca_status` |
| 临床 | `cli_*` | 14 | `cli_recist_response`, `cli_pfs_months` |
| Confluencia | `cfl_*` | 6 | `cfl_drug_prediction_score` |
| circRNA | `crna_*` | 30+ | `crna_immunogenicity_score`, `crna_torusfold_method` |
| **总计** | | **~170** | |

---

## 十七、启动方式

```bash
# CLI模式
python -m confluencia_3_0 --steps 365 --subtype BLIS --circrna-backend vienna

# Studio模式 (无代码UI)
cd D:\IGEM集成方案
streamlit run confluencia-studio/streamlit_app/Home.py

# 或使用批处理
.\启动Studio.bat
```

---

**文档版本**: 3.0.0
**最后更新**: 2026-06-17
**作者**: 颜子壹
