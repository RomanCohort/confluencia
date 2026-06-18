# Confluencia 3.0 — 完整架构图 (Markdown)

> TNBC Simulacrum + circRNA Therapy Unified Computing Platform
> EventBus-First Architecture | Offline-First Design | Graceful Degradation

---

## 1. 系统入口

```
main.py — CLI 入口
├── --subtype BLIS|IM|M|LAR          分子亚型选择
├── --steps 365                       模拟天数
├── --circrna-backend heuristic|vienna|esm2    免疫原性后端
└── --structure-mode heuristic|simple|diffusion|physics_b|physics_ba  结构模式
```

---

## 2. 核心架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│                    main.py (CLI 入口)                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              core/agent.py — TNBCSimulacrum                     │
│              Agent 基类, 持有所有 Manager 引用                   │
│              每日 step() 调用各 Manager.step()                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────────┐
│ event_bus.py  │   │ state_schema.py │   │ events.py         │
│ EventBus      │   │ StateSchema     │   │ 18 种事件类型     │
│ 事件中枢      │   │ 200+ 状态键     │   │ CIRCRNA_* 等      │
└───────────────┘   └─────────────────┘   └───────────────────┘
```

---

## 3. 子系统管理器层

| Manager | 子系统名 | 核心职责 |
|---------|----------|----------|
| **TumorManager** | `tumor` | 肿瘤生长、异质性、CSC、血管生成、转移 |
| **TMEManager** | `tme` | 免疫细胞、CAF/ECM、免疫逃逸、免疫编辑 |
| **TreatmentManager** | `treatment` | 化疗、免疫治疗、靶向、放疗、circRNA治疗 |
| **CircRNAManager** | `circrna` | 免疫评估、结构预测、PK模拟、序列进化 |
| **ClinicalManager** | `clinical` | RECIST评估、生存分析、毒性分级 |
| **BiomarkerManager** | `biomarker` | 亚型分类、耐药检测、标志物追踪 |

---

## 4. 各子系统内部模块 (文件级)

### 4.1 肿瘤子系统 (`core/tumor/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `growth_engine.py` | 肿瘤生长动力学 | Gompertz/Logistic生长模型, 亚型特异性增长率 |
| `heterogeneity.py` | 细胞亚群异质性 | 克隆演化, 空间异质性建模 |
| `cancer_stem_cell.py` | CSC 干细胞池 | 自我更新/分化平衡, 肿瘤起始能力 |
| `angiogenesis.py` | 血管生成 | VEGF信号驱动, 血管成熟度, 缺氧诱导 |
| `metastasis.py` | 转移级联 | EMT/MET切换, 器官趋向性 |

### 4.2 微环境子系统 (`core/tme/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `immune_dynamics.py` | 免疫细胞动力学 | CD8+/CD4+/Treg/NK细胞, 细胞因子梯度 |
| `fibroblast.py` | CAF/ECM 重塑 | CAF激活/分化, 基质沉积/降解 |
| `immune_evasion.py` | 免疫逃逸机制 | MHC-I下调, PD-L1过表达, 免疫排斥 |
| `immunoediting.py` | 免疫编辑 | 消除→平衡→逃逸三阶段 |

### 4.3 治疗子系统 (`core/treatment/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `chemotherapy.py` | 化疗 | 蒽环/紫杉烷, 剂量-效应曲线, 周期调度 |
| `immunotherapy.py` | 免疫治疗 | PD-1/PD-L1抑制, CTLA-4阻断, 免疫检查点 |
| `targeted.py` | 靶向治疗 | HER2/EGFR抑制, PI3K/AKT靶向, PARP抑制 |
| `radiotherapy.py` | 放疗 | 分次照射模型, DNA损伤修复, 放射敏感性 |
| `circrna_therapy.py` | circRNA治疗 | 免疫原性调控, 疫苗评估 |

**药物管道 (`drug_pipeline/`)**:
- `drug_registry.py` — 药物注册表
- `pkpd.py` — PK/PD 模型

### 4.4 circRNA 子系统 (`core/circrna/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `immune_sensing.py` | 四通路免疫评估 | RIG-I/TLR7/TLR8/PKR, dsRNA检测 |
| `structure_prediction.py` | 二级结构预测 | ViennaRNA MFE, 配对概率矩阵 |
| `bsj_features.py` | BSJ位点特征 | 反向剪接检测, 环化信号评分 |
| `folding_kinetics.py` | 折叠动力学 | 折叠路径采样, 动力学速率常数 |
| `cotrans_folding.py` | 共转录折叠 | 合成中折叠, 环化前构象 |
| `rna_modifications.py` | RNA修饰 | m6A/Psi/5mC修饰, 免疫原性影响 |
| `tme_simulation.py` | TME响应模拟 | 免疫微环境交互 |
| `drug_response.py` | 药物响应预测 | 敏感性/耐药性评估 |
| `clinical_prediction.py` | 临床预后预测 | 生存期估计 |
| `adaptive_dosing.py` | 自适应剂量 | PK引导给药调整 |
| `multi_drug_combination.py` | 联合用药 | 协同/拮抗评分 |
| `rna_docking.py` | RNA-蛋白对接 | RIG-I/TLR结合预测 |
| `patient_stratification.py` | 患者分层 | 亚型特异性方案 |

### 4.5 临床子系统 (`core/clinical/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `recist.py` | 疗效评估 | RECIST 1.1标准, CR/PR/SD/PD分类 |
| `survival.py` | 生存分析 | Kaplan-Meier, Cox回归, 中位生存期 |
| `toxicity.py` | 毒性分级 | CTCAE分级, 器官毒性评估 |

### 4.6 标志物子系统 (`core/biomarker/`)

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `subtype_classifier.py` | 分子亚型分类 | BLIS/IM/M/LAR, 随机森林/基因表达特征 |
| `resistance_detector.py` | 耐药检测 | 获得性耐药, 药物泵过表达, 靶点突变 |
| `tracker.py` | 标志物追踪 | 时序追踪, 动态阈值预警 |

---

## 5. TorusFold 深度学习架构

**路径**: `core/circrna/torusfold/`

```
输入: circRNA 序列 (A,C,G,U) + 基因表达字典
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ tpe.py — Torus Positional Encoding                          │
│ • TPE[0] = TPE[L] 周期性位置编码                            │
│ • sin(2πi/L) + 谐波分量                                     │
│ • 保证环形拓扑连续性                                        │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ equivariant_backbone.py — 旋转等变骨干网络                  │
│ • ESM2 (frozen) — 预训练蛋白质LLM (650M参数)                │
│ • TPE — 环形拓扑感知                                        │
│ • Torus Transformer — 环形自注意力                          │
└─────────────────────────────┬───────────────────────────────┘
                              │ sequence_repr (B, L, d_model)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Pair Initialization + Circular Distance                     │
│ • z[i,j] = left(seq[i]) + right(seq[j])                    │
│ • d_circ(i,j) 环形距离特征                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │ pair_repr (B, L, L, c_z=128)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ triangle_update.py — CircPairformer 核心 (x4 blocks)        │
│                                                             │
│  ├─ TriangleMulUpdate (outgoing)  行聚合→列更新            │
│  ├─ TriangleMulUpdate (incoming)  列聚合→行更新            │
│  ├─ TriangleAttention (starting)  行方向三角注意力          │
│  ├─ TriangleAttention (ending)    列方向三角注意力          │
│  └─ PairTransition               2层MLP过渡                │
│                                                             │
│  + 环形距离 bias: d_circ(i,j) → 注意力偏置                  │
└─────────────────────────────┬───────────────────────────────┘
                              │ refined pair_repr
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐   ┌─────────────────────────────────┐
│ irs_pair.py              │   │ 结构预测模块 (4种模式)          │
│ IRS 配对预测头            │   │                                 │
│ • pair_probs[L,L]        │   │ ┌─────────────────────────────┐ │
│ • BSJ 配对检测            │   │ │ diffusion_structure.py      │ │
│ • circular_distance_matrix│   │ │ AF3-style 扩散              │ │
└──────────────────────────┘   │ │ 闭合约束损失                │ │
                               │ │ ||x₀-xₗ||² → min           │ │
                               │ └─────────────────────────────┘ │
                               │                                 │
                               │ ┌─────────────────────────────┐ │
                               │ │ physics_bridge.py           │ │
                               │ │ DL → 几何约束提取           │ │
                               │ │ ConstraintSet:              │ │
                               │ │  • bond constraints         │ │
                               │ │  • pair constraints         │ │
                               │ │  • clash constraints        │ │
                               │ └──────────────┬──────────────┘ │
                               │                ▼                │
                               │ ┌─────────────────────────────┐ │
                               │ │ constraint_solver.py        │ │
                               │ │ Plan B: 纯几何求解          │ │
                               │ │ 多构象采样 + 闭环约束       │ │
                               │ └──────────────┬──────────────┘ │
                               │                │                │
                               │ ┌──────────────┴──────────────┐ │
                               │ │ cgmd_refiner.py             │ │
                               │ │ Plan A: OpenMM MD精修       │ │
                               │ │ 能量最小化 + MD弛豫         │ │
                               │ │ DL bias 注入力场            │ │
                               │ └──────────────┬──────────────┘ │
                               │                ▼                │
                               │ ┌─────────────────────────────┐ │
                               │ │ structure_validator.py      │ │
                               │ │ ValidationMetrics:          │ │
                               │ │  • closure_distance         │ │
                               │ │  • bond_rmsd                │ │
                               │ │  • pair_satisfaction        │ │
                               │ │  • clash_count              │ │
                               │ │  • energy_score             │ │
                               │ │  • stability_score          │ │
                               │ └─────────────────────────────┘ │
                               └─────────────────────────────────┘
```

### 5.1 TorusFold 主入口

**文件**: `torusfold.py`

```python
class TorusFoldConfig:
    structure_mode: str = "simple"  # simple | diffusion | physics_b | physics_ba
    n_diffusion_steps: int = 100
    n_solver_samples: int = 20
    n_minimize_steps: int = 500   # OpenMM
    n_md_steps: int = 5000        # OpenMM
```

### 5.2 TorusFold 多任务头

| 头 | 输出维度 | 功能 |
|----|----------|------|
| Composite Score | 8 | 稳定性/翻译/免疫/递送等综合评分 |
| Report Card | 4 | 安全性/有效性/可制造性/可递送性 |
| Drug Response | 3 | 敏感性/耐药性/协同性 |
| Translation Efficiency | 1 | IRES活性 + 翻译效率 |
| Circ Stability | 1 | 环稳定性评分 |
| Immune Pathway | 3 | RIG-I / TLR / PKR 独立预测 |

---

## 6. TorusFold 评分桥接

**文件**: `core/circrna/torusfold_scorer.py`

```python
@dataclass
class TorusFoldSignals:
    # 闭合约束
    closure_distance: float      # 首-末核苷酸 3D 距离 (Å)
    closure_score: float         # 归一化闭合评分 [0,1]

    # BSJ 配对
    bsj_stability: float         # BSJ 位点稳定性
    bsj_pair_count: float        # 跨 BSJ 配对数

    # pair_map 衍生
    dsRNA_fraction: float        # 配对碱基比例
    long_range_pair_fraction: float  # 长程配对比例

    # 多任务头
    translation_efficiency: float
    circ_stability: float
    immune_rig_i: float
    immune_tlr: float
    immune_pkr: float

    # 物理约束
    energy_score: float
    bond_rmsd: float
    pair_satisfaction: float
```

**核心方法**:

```python
def compute_objectives() -> np.ndarray:
    """返回四维目标向量 [stability, translation, immune_evasion, delivery]"""

def compute_immune_override() -> Dict[str, float]:
    """用 TorusFold 免疫头替代启发式评分"""
```

---

## 7. Backend 三层降级架构

**文件**: `core/backend_architecture.py`, `core/external_backends.py`

```
┌──────────────────┐              ┌──────────────────┐              ┌──────────────────┐
│   Tier 0: ESM2   │   不可用时   │  Tier 1: Vienna  │   不可用时   │ Tier 2: Heuristic│
│   (最高精度)     │ ──────────→  │   (中等精度)     │ ──────────→  │  (保底降级)      │
├──────────────────┤              ├──────────────────┤              ├──────────────────┤
│ • 650M 参数 LLM  │              │ • 热力学折叠     │              │ • GC含量计算     │
│ • 需 GPU + 在线  │              │ • 本地 CPU 运行  │              │ • IRES基序检测   │
│ • RNA嵌入+微调   │              │ • MFE结构预测    │              │ • 零依赖纯Python │
└──────────────────┘              └──────────────────┘              └──────────────────┘
```

### 7.1 Encoder 模块 (`core/encoder/`)

| 文件 | 功能 |
|------|------|
| `model.py` | Transformer 编码器模型 |
| `tokenizer.py` | RNA 序列分词器 (A/C/G/U 字母表) |
| `adapter.py` | 外部模型适配器 (ESM2 接口) |
| `config.py` | 编码器配置 |
| `train.py` | 编码器训练脚本 |

---

## 8. PK/PD 模块

**文件**: `core/pk/rnactm.py`

### 六室 circRNA 药代动力学模型

```
┌─────────┐   k_ab   ┌─────────┐   k_dt   ┌──────────┐
│  Depot  │ ───────→ │  Blood  │ ───────→ │  Tissue  │
│ 给药部位 │          │ 血液循环 │          │ 组织分布  │
└─────────┘          └────┬────┘          └──────────┘
                          │ k_be
                          ▼
                   ┌──────────┐   k_ec   ┌───────────┐   k_cp   ┌───────────┐
                   │ Endosome │ ───────→ │ Cytoplasm │ ───────→ │  Protein  │
                   │   内体   │          │  细胞质   │          │ 表达蛋白   │
                   └──────────┘          └───────────┘          └───────────┘
```

**输出指标**:
- AUC (药效面积)
- Cmax (峰浓度)
- t½ (半衰期)
- 峰蛋白量

**核心函数**:
```python
infer_rna_ctm_params()   # 从序列特征推断 PK 参数
simulate_rna_ctm()       # ODE 数值积分
summarize_rna_ctm_curve() # 曲线特征提取
```

---

## 9. 进化优化模块

**路径**: `core/evolution/`

### 9.1 circRNA 序列进化 (`cirrna_evolution.py`)

```
┌──────────────────────────────────────────────────────────────┐
│              REINFORCE 策略梯度 + Pareto 多目标优化          │
├──────────────────────────────────────────────────────────────┤
│  目标权重 (IPS 默认):                                        │
│    • stability:       0.35  (环稳定性 + 闭合约束)            │
│    • translation:     0.30  (IRES活性 + 翻译效率)            │
│    • immune_evasion:  0.25  (四通路免疫逃逸)                 │
│    • delivery:        0.10  (长度/GC/修饰)                   │
├──────────────────────────────────────────────────────────────┤
│  输出: best_sequence, best_reward, rounds_ran                │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 药物分子进化 (`molecule_evolution.py`)

- SMILES 字符串优化
- 反射学习 (Self-Reflection)
- 输出: best_reward, rounds_ran, reflections

### 9.3 辅助模块

| 文件 | 功能 |
|------|------|
| `actions.py` | 进化动作空间 (突变/交叉/重组) |
| `pareto.py` | Pareto 前沿计算 + 拥挤距离排序 |

---

## 10. Confluencia 桥接层

**路径**: `core/confluencia/`

> circRNA ↔ TNBC Simulacrum 双向耦合

| 文件 | 功能 |
|------|------|
| `drug_bridge.py` | circRNA 治疗药物 ↔ 化疗方案协同/拮抗检测 |
| `epitope_bridge.py` | circRNA 疫苗抗原表位 → TME 免疫激活桥接 |
| `pk_bridge.py` | RNACTM 药代曲线 → TME 药物浓度场耦合 |
| `joint_bridge.py` | 联合桥接: 多通路信号汇总 + 综合响应预测 |

---

## 11. Pipeline 模块

**文件**: `core/pipeline/circrna_pipeline.py`

```
circRNA 设计全流程管道:

序列输入 → 免疫评估 → 结构预测 → PK模拟 → 序列进化 → 疫苗评估 → 临床预测
    │          │           │          │          │           │           │
    ▼          ▼           ▼          ▼          ▼           ▼           ▼
Backend    TorusFold   ViennaRNA   RNACTM    REINFORCE   epitope    survival
三层降级   四种模式    MFE折叠     六室PK    多目标优化  桥接       预测
```

---

## 12. Experiment 模块

**路径**: `experiments/`

| 文件 | 功能 |
|------|------|
| `experiment_circrna_therapy.py` | circRNA 治疗方案评估 |
| `experiment_confluencia_pkpd.py` | PK/PD 桥接验证 |
| `experiment_biomarker_stratified.py` | 分层分析 (BLIS/IM/M/LAR) |
| `experiment_chemo_response.py` | 化疗响应预测 |
| `experiment_akt_pi3k.py` | AKT/PI3K 通路靶向 |
| `experiment_ar_lar.py` | AR/LAR 亚型靶向 |
| `experiment_combination_chemo_immuno.py` | 联合化疗+免疫 |
| `experiment_combination_screening.py` | 联合方案筛选 |

**高级封装 (`experiment/`)**:
- `sandbox.py` — 沙盒实验环境
- `clinical_trial.py` — 临床试验模拟
- `biomarker_stratification.py` — 标志物分层
- `combination.py` — 组合实验框架
- `resistance_evolution.py` — 耐药演化

---

## 13. Config 模块

**文件**: `core/config.py`

```python
@dataclass
class CircRNAConfig:
    enabled: bool = True
    immunogenicity_backend: str = "heuristic"  # heuristic | vienna | esm2
    structure_mode: str = "heuristic"          # heuristic | simple | diffusion | physics_b | physics_ba
    enable_structure_prediction: bool = True
    enable_torusfold: bool = False             # 自动由 structure_mode 决定

    # 结构模式参数
    diffusion_steps: int = 100
    solver_samples: int = 20
    openmm_minimize_steps: int = 500
    openmm_md_steps: int = 5000

    # PK 默认参数
    pk_default_horizon: int = 168  # 小时
    pk_default_dt: float = 1.0     # 小时
```

---

## 14. Tests

**路径**: `tests/`

| 文件 | 功能 |
|------|------|
| `test_circrna_manager.py` | CircRNAManager 单元测试 (免疫评估/结构预测/事件处理) |
| `test_rnactm.py` | RNACTM 六室 PK 模型单元测试 |
| `test_molecule_evolution.py` | 分子进化单元测试 |

---

## 15. 数据流总览

```
用户输入 (CLI)
    │
    ▼
main.py → Confluencia3Config → TNBCSimulacrum.__init__()
    │
    ├── 初始化 EventBus (event_bus.py)
    ├── 初始化 StateSchema (state_schema.py)
    ├── 初始化 6 个 Manager
    │
    └── 每日循环:
        │
        ├── TumorManager.step()
        │   ├── growth_engine.step()      → tumor_volume, growth_rate
        │   ├── heterogeneity.step()      → clone_frequencies
        │   ├── csc_pool.step()           → csc_fraction
        │   ├── angiogenesis.step()       → vessel_density
        │   └── metastasis_engine.step()  → metastatic_sites
        │
        ├── TMEManager.step()
        │   ├── immune.step()             → cd8_count, treg_ratio
        │   ├── fibroblast.step()         → caf_activation, ecm_density
        │   ├── evasion.step()            → pd_l1_expression
        │   └── immunoediting.step()      → editing_phase
        │
        ├── TreatmentManager.step()
        │   ├── chemotherapy.step()       → chemo_response
        │   ├── immunotherapy.step()      → checkpoint_inhibition
        │   ├── targeted.step()           → pathway_inhibition
        │   ├── radiotherapy.step()       → radiation_damage
        │   └── circrna_therapy.step()    → circrna_effect
        │
        ├── CircRNAManager.step()
        │   ├── assess_immunogenicity()   → immune_sensing.py → Backend降级
        │   ├── predict_structure()       → structure_prediction.py
        │   ├── assess_with_torusfold()   → torusfold_scorer.py
        │   └── simulate_pk()             → rnactm.py
        │
        ├── ClinicalManager.step()
        │   ├── recist.evaluate()         → CR/PR/SD/PD
        │   ├── survival_model.step()     → survival_probability
        │   └── toxicity_grader.step()    → ctcae_grade
        │
        └── BiomarkerManager.step()
            ├── biomarker_tracker.step()   → biomarker_trajectory
            └── subtype_classifier.step()  → subtype_scores
```

---

## 16. EventBus 事件类型

**文件**: `core/events.py`

| 事件 | 用途 |
|------|------|
| `CIRCRNA_IMMUNE_EVAL` | circRNA 免疫原性评估请求 |
| `CIRCRNA_STRUCTURE_PREDICT` | circRNA 结构预测请求 |
| `CIRCRNA_SEQUENCE_EVOLVE` | circRNA 序列进化请求 |
| `CIRCRNA_VACCINE_ASSESS` | circRNA 疫苗评估请求 |
| `CIRCRNA_FOLDING_KINETICS` | circRNA 折叠动力学请求 |
| `CIRCRNA_DRUG_RESPONSE` | circRNA 药物响应请求 |
| `CIRCRNA_PK_SIMULATE` | circRNA PK 模拟请求 |
| `MOLECULE_EVOLUTION_REQUEST` | 药物分子进化请求 |
| *(10+ 其他)* | 肿瘤/TME/治疗/临床模块间通信 |

---

## 17. 统计

| 指标 | 数值 |
|------|------|
| Python 文件 | 75+ |
| 子系统管理器 | 6 |
| 事件类型 | 18 |
| 状态键 | 200+ |
| TorusFold 模式 | 4 (simple / diffusion / physics_b / physics_ba) |
| Backend 层级 | 3 (ESM2 → ViennaRNA → Heuristic) |
| PK 室数 | 6 |
| 免疫通路 | 4 (RIG-I / TLR7 / TLR8 / PKR) |
| 进化目标维度 | 4 (stability / translation / immune_evasion / delivery) |

---

## 18. 架构设计原则

### 18.1 EventBus-First

- 所有跨模块通信通过 EventBus
- 订阅者动态注册, 发布者无需感知
- 解耦模块依赖, 支持独立测试

### 18.2 Offline-First

- Tier 2 Heuristic 零依赖, 保底可用
- 隔离网络环境 (医院内网) 可正常运行
- 在线 API 仅作为精度增强选项

### 18.3 Graceful Degradation

- Backend 三层降级: ESM2 → ViennaRNA → Heuristic
- TorusFold 四种模式: 按资源/精度需求选择
- OpenMM 不可用时自动降级到 physics_b

### 18.4 Research-Friendly

- 每个模块独立可测试
- 清晰的文件边界和职责划分
- 配置驱动, 无需修改代码即可切换模式
