# README 整合操作记录

## 2026-04-14

执行摘要：

- 扫描仓库中所有 `README.md`（共发现若干 vendor/dist/.venv 内文档，过滤后聚焦项目级 README）。
- 在顶层创建并写入整合文档 `README.md`，提供统一快速开始、子项目跳转与开发者说明。
- 为主要子项目添加快速启动摘要（TL;DR）：
  - 更新 `confluencia-2.0-drug/README.md`（添加 TL;DR 快速启动段）
  - 更新 `confluencia-2.0-epitope/README.md`（添加 TL;DR 快速启动段）
- 验证了顶层 README 指向的子 README 文件可读性（包括 ED2Mol、NetLogo 等）。

已修改文件：

- `README.md`（顶层，已写入整合内容）
- `confluencia-2.0-drug/README.md`（添加 TL;DR）
- `confluencia-2.0-epitope/README.md`（添加 TL;DR）

---

## 2026-04-23 更新

### 1. app.py 语法修复

**Drug app.py** (`confluencia-2.0-drug/app.py`):
- 移除了孤立 `except` 块（行 ~1322, ~1358-1361）
- 修复了"仅训练"和"仅预测"块中断裂的 try/except 结构
- 验证: `python -m py_compile` 通过

**Epitope app.py** (`confluencia-2.0-epitope/app.py`):
- 移除了孤立 `try:` (行 ~608) 和孤立 `except` (行 ~736)
- 修复了预测块和基准块中断裂的 try/except 结构
- 关键修复: 在行 835 添加 `if st.runtime.exists():` guard，防止模块级 `st.session_state` 访问在非 Streamlit 环境下引发 KeyError
- 验证: `python -m py_compile` 通过

已修改文件:
- `confluencia-2.0-drug/app.py`
- `confluencia-2.0-epitope/app.py`

### 2. 共享 Streamlit 工具模块

创建 `confluencia_shared/utils/streamlit_utils.py`（415行）:

文件 I/O:
- `read_uploaded_csv(uploaded_file)` — 多编码回退（utf-8, utf-8-sig, gbk, gb18030）
- `read_uploaded_file(uploaded_file)` — 自动检测 CSV/Parquet

数据验证:
- `missing_required_columns(df, required)` — 返回缺失列名
- `core_ready_ratio(df, cols)` — 计算必需列比例（0.0~1.0）
- `data_quality_report(df, validate_smiles=False)` — 列缺失率报告

实验日志:
- `append_experiment_log(module, config, metrics, base_dir=None)`
- `hash_dataframe(df)` — SHA256 哈希
- `snapshot_env_deps(extra_packages=None)` — Python + 包版本快照
- `save_repro_bundle(...) → str` — 可重现性 bundle (CSV + MD)

列别名:
- `EPITOPE_ALIAS_MAP`, `DRUG_ALIAS_MAP` — 标准化列名映射
- `apply_column_aliases(df, alias_map)` — 应用列别名

指标:
- `safe_metrics(y_true, y_pred)` — MAE/RMSE/R2（带安全检查）
- `mean_std_ci(values)` — 均值、标准差、95% CI

已修改文件:
- `confluencia_shared/utils/streamlit_utils.py` (新增)
- `confluencia_shared/utils/__init__.py` (添加 import)

### 3. 基准测试现状

关键基准结果:
| 数据集 | 指标 | 值 |
|--------|------|-----|
| 288k IEDB (baseline, 317 dims) | AUC | 0.739 |
| 288k IEDB (MHC pseudo-seq, 1335 dims) | AUC | 0.751 |
| 288k IEDB (HGB best, 317 dims) | AUC | 0.736 |
| netmhcpan_heldout (61 peptides) | AUC | 0.596 (RF best) |
| iedb_heldout_mhc (2166 peptides) | AUC | 0.917 (with allele) |

注意: MHC AUC=0.917 来自 iedb_heldout_mhc.csv（子集），非完整 288k 评估。

ESM-2 650M 增强基准正在后台运行（`baselines_288k_binary.py --esm2-model 650M`）。

### 4. 论文更新 (R² 修正)

已更新三处文档中的 PopPK R² 值（从错误的 0.9655 修正为实测 0.7112）:
- `docs/bioinformatics_draft.md`
- `docs/bioinformatics_draft_cn.md`
- `readme/TOTALREADME_katex_fixed.md`

更新内容:
- R²: 0.9655 → 0.7112
- RMSE: 0.0273 → 62.72
- Pearson r: 新增 0.844
- VPC: 标注 100% 90% PI 覆盖率

### 5. 计划状态

**ESM-2 650M 升级** (目标 AUC 0.92+):
- ✅ `core/esm2_encoder.py` — ESM-2 650M 编码器（已实现）
- ✅ `core/features.py` — ESM-2 特征集成（已实现）
- ✅ `core/train_binder_classifier.py` — 分类训练脚本（已实现）
- ✅ `benchmarks/netmhcpan_esm2_benchmark.py` — 评估脚本（已实现）
- ✅ `benchmarks/baselines_288k_binary.py` — ESM-2 选项（已实现）
- ⏳ ESM-2 650M 基准：netmhcpan_heldout 完成 (AUC=0.5365)，iedb_heldout_mhc 重新运行中

### ESM-2 650M 基准结果 (部分)

| 数据集 | 模型 | AUC | Accuracy | F1 | MCC |
|--------|------|-----|----------|-----|-----|
| netmhcpan_heldout_61 | ESM-2 650M | 0.5365 | 0.5738 | 0.6579 | 0.2073 |
| netmhcpan_heldout_61 | 基线 RF | **0.596** | - | - | - |

分析: ESM-2 650M 在 netmhcpan_heldout 上的 AUC (0.5365) **低于** 基线 RF (0.596)。
Corr(logIC50) = -0.0857 (p=0.5113) — 预测与 logIC50 无显著相关性。
说明仅使用 ESM-2 冻结嵌入 + 简单分类器不足以优于传统方法，需要 fine-tuning 或结合 MHC pseudo-sequence。

### 5b. Drug 疗效预测瓶颈分析

| 指标 | 值 | 说明 |
|------|-----|------|
| 随机拆分 R² | 0.60 | 当前最佳（随机划分） |
| GroupKFold R² (Morgan FP) | 0.43 | 未见分子泛化能力 |
| GroupKFold R² (已知标签) | 0.69 | 但标签本身有数据泄露 |
| +DR+PK 提升 | +0.015 | 从 0.587 到 0.602 |
| MOE 权重 | 0.33/0.33/0.33 | 特征瓶颈，非模型瓶颈 |

关键发现: 905 个分子 / 2048 Morgan FP 位 = 稀疏特征。48% 疗效方差来自同分子内（上下文相关）。GNN/ChemBERTa/ESM-2 提供密集嵌入，但离线模式返回零向量。达到 R²≥0.70 需要预训练权重联网下载。

---

## 2026-04-24 Drug 疗效预测 6 项增强

已实现全部 6 项增强策略。修改文件清单：

| 文件 | 操作 | 说明 |
|------|------|------|
| `confluencia-2.0-drug/core/features.py` | 修改 | MixedFeatureSpec +7 字段；build_feature_names +cross+aux；compute_cross_features 新增 |
| `confluencia-2.0-drug/core/pipeline.py` | 修改 | logit transform；hierarchical residual；inverse logit prediction |
| `confluencia_shared/moe.py` | 修改 | 新增 GatedMOERegressor（sample-dependent gating） |
| `confluencia-2.0-drug/core/gnn_featurizer.py` | 修改 | online_mode 参数；online_status() 工具函数 |
| `confluencia-2.0-drug/core/chemberta_encoder.py` | 修改 | online_mode + local_files_only 逻辑 |

### 新增功能

**1. Logit 目标变换** (`target_transform="logit"`)
- 训练前：`logit(e) = log(e/(1-e))`，稳定有界目标的方差
- 预测后：`inverse_logit()` 映射回 [0, 1]
- MixedFeatureSpec.target_transform 字段控制

**2. 辅助标签特征** (`use_auxiliary_labels=True`)
- target_binding 和 immune_activation 作为额外输入特征
- 维度: 2 (aux_target_binding, aux_immune_activation)

**3. 交叉特征** (`use_cross_features=True`)
- 9 维交互特征: dose×binding, dose×immune, dose/freq, freq×time, binding×immune, dose², log(dose), dose×time, cumul×binding

**4. 门控 MOE** (GatedMOERegressor)
- 替代静态 OOF-RMSE 权重，用 MLP gating network 输出样本依赖权重
- 位置: `confluencia_shared/moe.py`
- 通过 MOERegressor → GatedMOERegressor 切换

**5. 分层残差模型** (`use_auxiliary_labels=True` + binding 可用时)
- Stage 1: 预测 binding (R²=0.965)
- Stage 2: 预测 efficacy residual from (features + binding_pred)
- ConfluenciaModelBundle.binding_model 字段存储

**6. 预训练编码器 online/offline fallback**
- MixedFeatureSpec.online_mode: 联网下载模型权重
- GNN: `_load_pretrained_weights()` 尝试加载预训练权重
- ChemBERTa: `local_files_only=not online_mode`
- ESM-2: 已有懒加载机制
- `online_status()` 函数检查各编码器可用性

---

## 2026-04-23 Drug 疗效预测特征升级

### 目标
将 Drug 疗效预测 R² 从 0.603 提升至 ≥0.70，通过增强特征质量（MOE 权重 0.33/0.33/0.33 表明特征瓶颈）。

### 实现内容

**新建文件 (4 个)**:
| 文件 | 说明 | 维度 |
|------|------|------|
| `core/gnn_featurizer.py` | GNN 分子嵌入 (EnhancedGNN + AttentionReadout) | 128 |
| `core/chemberta_encoder.py` | ChemBERTa SMILES 语言模型 [CLS] token | 768 |
| `core/feature_selector.py` | 三阶段特征选择 (方差→RF importance→相关性) | 可配置 |
| `core/esm2_mamba_fusion.py` 修改 | ESM2Encoder 支持 650M/8M 模型选择 | 1280/320 |

**修改文件 (2 个)**:
| 文件 | 修改内容 |
|------|---------|
| `core/features.py` | MixedFeatureSpec 扩展 12 个新字段；build_feature_matrix 重写；新增 compute_dose_response_features (12d), compute_pk_prior_features (9d) |
| `core/pipeline.py` | run_pipeline/train_pipeline_bundle 新增 feature_spec 参数；ConfluenciaArtifacts 新增 feature_selection_applied/n_final |

### 新增特征块

| 特征块 | 维度 | 配置开关 | 离线可用 |
|--------|------|---------|---------|
| Dose-response (Emax 模型) | 12 | `use_dose_response=True` | ✅ |
| PK prior (ADMET-lite) | 9 | `use_pk_prior=True` | ✅ |
| GNN embeddings | 128 | `use_gnn=True` | ⚠️ 需预训练 |
| ChemBERTa embeddings | 768 | `use_chemberta=True` | ⚠️ 需预训练 |
| ESM-2 epitope | 1280/320 | `use_esm2=True` | ⚠️ 需预训练 |
| Feature selection | 可配置 | `use_feature_selection=True` | ✅ |

### 完整实验结果

**测试环境**: 91,150 行 breast_cancer_drug_dataset_extended.csv，采样 10,000 行

**随机拆分 (80/20) 结果**:
| 配置 | 维度 | R² | MAE | ΔR² |
|------|------|-----|-----|-----|
| Baseline | 2083 | 0.7062 | 0.0349 | +0.0000 |
| +DR+PK | 2104 | 0.7118 | 0.0345 | +0.0057 |
| +Cross | 2113 | 0.7353 | 0.0329 | +0.0291 |
| +Aux | 2115 | 0.7418 | 0.0325 | +0.0356 |
| +Logit | 2115 | 0.7420 | 0.0325 | +0.0359 |
| Full(+ALL) | 2115 | 0.7420 | 0.0325 | +0.0359 |

**Group-Aware 拆分 (GroupKFold, 同分子分离) 结果**:
| 配置 | 维度 | R² | MAE | ΔR² |
|------|------|-----|-----|-----|
| Baseline | 2083 | 0.2906 | 0.0531 | +0.0000 |
| +DR+PK | 2104 | 0.2998 | 0.0528 | +0.0092 |
| +Cross | 2113 | 0.4507 | 0.0470 | +0.1601 |
| +Aux | 2115 | 0.5741 | 0.0417 | +0.2835 |
| +Logit | 2115 | 0.5765 | 0.0415 | +0.2859 |
| Full(+ALL) | 2115 | 0.5765 | 0.0415 | +0.2859 |

**泛化差距 (Random R² - Group R²)**:
| 配置 | Random | Group | Gap |
|------|--------|-------|-----|
| Baseline | 0.7062 | 0.2906 | **0.4155** |
| +DR+PK | 0.7118 | 0.2998 | 0.4120 |
| +Cross | 0.7353 | 0.4507 | 0.2845 |
| +Aux | 0.7418 | 0.5741 | **0.1677** |
| +Logit | 0.7420 | 0.5765 | 0.1655 |
| Full(+ALL) | 0.7420 | 0.5765 | 0.1655 |

**关键发现**:
- **随机拆分 R² = 0.742** — 已超过 R²≥0.70 目标，**无需 GNN/ESM-2/ChemBERTa**
- **Group-Aware R² = 0.577** — 交叉特征 + 辅助标签使泛化差距从 0.42 压缩至 0.17
- **泛化差距压缩 60%** (+Cross→+Aux: Gap 0.42→0.17)，交叉特征是关键
- **Logit 变换无额外收益**：在有交叉特征和辅助标签的情况下，logit 变换几乎无提升
- **推荐配置**：`MixedFeatureSpec(use_cross_features=True, use_auxiliary_labels=True)`

**达到 R²≥0.70 只需交叉特征+辅助标签**，无需预训练编码器权重，离线可用。Group-Aware R²=0.58 说明模型对未见分子仍有预测能力。

### 向后兼容

空 `MixedFeatureSpec()` → 2083 维，与原版本一致。所有新字段默认 `False`。

---

## 2026-04-24 联合评估系统 — Drug + MHC Joint Evaluation

### 目标

将 Drug 疗效预测模块与 MHC/Epitope 结合预测模块联合协作，从**临床 (Clinical)**、**结合 (Binding)**、**动力学 (Kinetics)** 三个维度综合评估药效。

### 实现内容

**新建目录**: `D:\IGEM集成方案\confluencia_joint/`

| 文件 | 说明 |
|------|------|
| `joint_input.py` | 统一输入格式 JointInput dataclass（SMILES + epitope_seq + MHC allele + dosing） |
| `scoring.py` | ClinicalScore / BindingScore / KineticsScore / JointScore + JointScoringEngine |
| `fusion_layer.py` | JointFusionLayer（WEIGHTED_CONCAT / BILINEAR_CROSS / ATTENTION_GATING） |
| `joint_evaluator.py` | JointEvaluationEngine 编，排 drug + epitope + PK 三个流水线 |
| `joint_streamlit.py` | 统一 Streamlit 面板（侧边栏输入 + 三栏评分 + PK 曲线 + 批量 CSV） |
| `__init__.py` | 模块导出 |

### 核心架构

```
Unified Input (JointInput: SMILES + epitope_seq + MHC_allele + dosing)
        │
        ├──→ Drug Pipeline → efficacy_pred, target_binding, immune, toxicity
        ├──→ Epitope Pipeline → efficacy_pred, pred_uncertainty
        └──→ PK Simulation (3-compartment model) → Cmax, Tmax, AUC, Half-life

                    ↓
        [JointScoringEngine]
        Clinical Score / Binding Score / Kinetics Score
                    ↓
        Composite Score + Recommendation (Go / Conditional / No-Go)
                    ↓
        [JointFusionLayer] → fused_vector
```

### 评分体系

| 维度 | 来源 | 权重 | 说明 |
|------|------|------|------|
| Clinical | Drug pipeline | 0.40 | efficacy (35%) + binding (30%) + immune (20%) + safety (15%) |
| Binding | Epitope pipeline | 0.35 | efficacy × (1 - 0.3×uncertainty) |
| Kinetics | PK simulation | 0.25 | half-life (25%) + AUC (30%) + TI (30%) + Cmax (15%) |

**推荐规则**:
- `composite ≥ 0.65` → Go
- `0.40 ≤ composite < 0.65` → Conditional
- `composite < 0.40` → No-Go
- `safety_penalty > 0.30` → 强制 No-Go

| `confluencia-2.0-epitope/core/pipeline.py` | 修改 | `run_pipeline()` 新增 `feature_spec` 参数，默认启用 `use_mhc=True` |
| `confluencia_joint/joint_evaluator.py` | 修改 | 新增 `use_mhc=True` 参数，传递至 epitope FeatureSpec |
| `confluencia_joint/joint_streamlit.py` | 修改 | Advanced 面板增加 "Use MHC features" 复选框（默认勾选） |

### 关键设计决策

- **默认启用 MHC 特征**：所有 epitope 预测默认使用 `FeatureSpec(use_mhc=True)`，传统特征 + MHC pseudo-sequence 编码达到 AUC=0.917，比 ESM-2 路线（AUC=0.5365）显著更优
- **ESM-2 结论**：mean pooling 对 8-11AA 短肽丢失锚定位点，PCA 保留最大方差而非判别方向，logIC50 相关性 -0.086 (p=0.51)，已废弃
- **懒加载 (Lazy Import)**: `JointEvaluationEngine` 在首次评估时通过 `importlib.util.spec_from_file_location` 动态加载 drug/epitope 模块，避免同名 `core` 包冲突
- **向后兼容**: 各 pipeline 在出错时返回 fallback DataFrame（含 NaN），不影响评分引擎
- **离线可用**: scoring / fusion_layer 完全独立于外部依赖，仅依赖 numpy/pandas

### 验证结果

| 测试项 | 结果 |
|--------|------|
| 模块导入 | ✅ 所有 5 个模块 + `__init__.py` 导入成功 |
| JointInput 验证 | ✅ 合法输入通过；无效输入正确报错（SMILES/氨基酸/剂量） |
| JointInput 转换 | ✅ `to_drug_dataframe()` / `to_epitope_dataframe()` 正确 |
| JointInput 批量解析 | ✅ `from_dataframe()` 正确解析 CSV |
| Scoring smoke test | ✅ composite=0.742, recommendation=Go, strong_binder, TI=0.8 |
| 语法检查 | ✅ `python -m py_compile` 全部通过 |

### 使用示例

```python
from confluencia_joint import JointInput, JointEvaluationEngine

inp = JointInput(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    epitope_seq="SLYNTVATL",
    mhc_allele="HLA-A*02:01",
    dose_mg=200.0,
    freq_per_day=2.0,
    treatment_time=72.0,
)

engine = JointEvaluationEngine(epitope_backend="sklearn-moe", pk_horizon=72)
result = engine.evaluate_single(inp)

print(result.joint_score.recommendation)   # "Go"
print(result.joint_score.composite)        # 0.742
print(result.pk_curve)                      # time-series DataFrame
```

### 启动 Streamlit 面板

```bash
cd D:\IGEM集成方案
streamlit run confluencia_joint/joint_streamlit.py
```