
# Confluencia — circRNA 药物发现多任务计算平台

> **Adaptive Mixture-of-Experts with Pharmacokinetic Dynamics for Small-Sample circRNA Drug Discovery**

Confluencia 是一个面向 circRNA 药物发现的多任务计算平台，集成了样本量自适应 MOE 集成学习、RNACTM 六室药代动力学模型和 Mamba3Lite 多尺度序列编码器，专为小样本 (N<300) 场景设计。

## 目录

- [项目简介](#项目简介)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [共享库 (confluencia_shared)](#共享库-confluencia_shared)
- [可信评估框架](#可信评估框架)
- [测试](#测试)
- [开发者指南](#开发者指南)
- [论文与引用](#论文与引用)
- [许可证](#许可证)

## 项目简介

circRNA 治疗药物是新兴的药物模态，但其计算预测面临三大挑战：(1) 湿实验室样本量小 (N<300)；(2) 需要跨疗效、毒性、免疫激活的多维预测；(3) 缺乏时间分辨的药代动力学建模。

Confluencia 通过以下创新解决这些问题：

| 创新点 | 说明 |
|--------|------|
| **样本量自适应 MOE 集成** | 根据数据量自动选择和加权回归专家 (Ridge/HGB/RF/MLP) |
| **RNACTM 药代动力学模型** | 首个针对 circRNA 的六室 PK 模型（注射→LNP→内吞→胞质释放→翻译→清除） |
| **Mamba3Lite 序列编码器** | 三时间常数自适应状态空间递归 + 四尺度池化 + 自注意力增强，轻量高效 |
| **可选超参数调优** | 支持 RandomizedSearchCV/GridSearchCV 对专家模型参数优化 |
| **Bootstrap CI 置信区间** | 小样本使用 t 分布，大样本使用 bootstrap percentile，更可靠的统计推断 |
| **分层交叉验证** | 按目标变量分位数分箱，确保各 fold 效能分布均衡 |

### 核心实验结果

| 指标 | 数值 | 说明 |
|------|------|------|
| **288K IEDB AUC (allele-aware)** | **0.80** | HGB，MHC 等位基因特征编码 |
| 288K IEDB AUC (allele-agnostic) | 0.73 | HGB/RF，序列感知分割 |
| **Drug Ridge R²** | **0.984** | 小样本药物预测最优 |
| MOE MAE (表位) | 0.389 | 比 Ridge 降低 39.2% (p<0.001) |
| MOE R² (表位) | 0.819 | 5折交叉验证 |
| **Mamba3Lite+Attn(d=16)** | **MAE=0.395, R²=0.802** | 注意力增强最佳单编码器配置 |
| **Drug 无 FP R²** | **0.960** | 移除过拟合的 Morgan FP 后 |
| TCCIA circRNA 验证 | r=0.888 | N=75 |

### 真实数据端到端验证 (2026-04-27 更新)

| 模块 | 样本量 | 方法 | 核心指标 | 数值 |
|------|--------|------|----------|------|
| **A. Drug Binding** | 4,774 | 5-Fold OOF Ensemble | **AUC** | **0.9252** (95% CI: 0.917–0.933) |
| **B. Target Binding** | 70,583 | 5-Fold OOF + MHC Distillation | **AUC** | **0.8245** (95% CI: 0.821–0.828) |
| **C. Immune Activation** | 126,259 | 5-Fold OOF Ensemble | **AUC** | **0.9054** (95% CI: 0.901–0.909) |
| D. Drug Toxicity | 21 | LOO Bayesian Ridge | Pearson r | 0.41 (toxicity), 0.25 (inflammation) |
| **E. GDSC Sensitivity** | 50 | LOO | **Pearson r** | **0.9155**, **AUC 0.94** |

**详细性能指标：**

| 模块 | XGB | LGB | HGB | Ensemble | 说明 |
|------|-----|-----|-----|----------|------|
| Drug Binding | 0.9238 | 0.9228 | 0.9235 | **0.9252** | ChEMBL 持有集验证 |
| Target Binding | 0.8344 | 0.8171 | 0.813 | **0.8245** | 368 等位基因，序列感知分割 |
| Immune Activation | 0.9046 | 0.9045 | 0.9029 | **0.9054** | 标签清洗 + 特征增强 |

### 288K IEDB 大规模训练 (2026-04-28)

| 模型 | AUC | F1 | MCC | 训练时间 |
|------|-----|----|----|----------|
| **RF** | **0.7343** | 0.3193 | 0.2337 | 134s |
| **HGB** | 0.7334 | **0.5783** | **0.3456** | 27s |
| LR | 0.6630 | 0.4569 | 0.2321 | 43s |
| MLP | 0.6627 | 0.5190 | 0.2390 | 244s |

**配置：** 288,135 样本，序列感知分割 (231K train / 57K test)，325 维特征，40.6% binder 率。
**模型已缓存：** `data/cache/epitope_model_288k.joblib`

### MHC 等位基因特征增强 (2026-05-31 新增)

| 配置 | 维度 | AUC | 说明 |
|------|------|-----|------|
| Baseline (no allele) | 317 | 0.7406 | 通用序列特征 |
| **+ MHC allele features** | **1335** | **0.8037** | 等位基因特异性编码 (+0.063) |

**Per-allele AUC (HGB, 52K IEDB binary):**

| 等位基因 | AUC | 样本量 | 说明 |
|----------|-----|--------|------|
| **HLA-A\*33:03** | **0.9495** | 315 | NetMHCpan 级别 |
| **HLA-A\*33:01** | **0.9242** | 318 | NetMHCpan 级别 |
| HLA-A\*68:01 | 0.8556 | 312 | |
| HLA-A\*24:02 | 0.7047 | 604 | |
| HLA-A\*02:01 | 0.6720 | 2144 | 最常见等位基因，需更多训练 |
| Overall | 0.8295 | 52942 | 246 等位基因 |

**关键发现：** MHC 等位基因编码器 (1018-dim MHC-I + 947-dim MHC-II) 从未在默认流水线中使用。当数据包含 `mhc_allele` 列时，训练管道自动启用 MHC 特征。

### Drug 消融实验 (2026-04-28 新增)

| 配置 | 维度 | MAE | R² | 说明 |
|------|------|-----|-----|------|
| Full | 2083 | 0.201 | 0.668 | 完整特征 |
| **- Morgan FP** | **35** | **0.076** | **0.960** | 最佳！FP 在小样本过拟合 |
| - Descriptors | 2075 | 0.648 | -2.057 | 描述符单独无预测力 |
| Only context (baseline) | 3 | 0.463 | -0.731 | 环境参数无预测力 |

**关键发现：** Morgan FP 在小样本 (n=200) 上严重过拟合 (R² 0.668 → 0.960 移除后)。Ridge 等线性模型更适合小样本药物数据。

### Drug 基线对比

| 模型 | MAE | R² | 说明 |
|------|-----|----|-----|
| **Ridge** | **0.037** | **0.984** | 线性模型最优 |
| MOE | 0.039 | 0.982 | 自适应集成 |
| RF | 0.042 | 0.979 | 随机森林 |
| GBR | 0.046 | 0.974 | 梯度提升 |
| HGB | 0.047 | 0.966 | 直方图梯度提升 |
| MLP | 0.082 | 0.900 | 神经网络 |

MOE 对比基线：比 MLP MAE 降低 52.7%，比 HGB 降低 17.7%。

### 消融实验结果 (Epitope, 2026-04-27)

| 配置 | 维度 | MAE | R² | 说明 |
|------|------|-----|-----|------|
| Full (all components) | 317 | 0.332 | **0.828** | 完整模型 |
| - Mamba local pool | 293 | **0.315** | **0.844** | 最佳！局部池化可能过拟合 |
| - Biochem stats | 301 | 0.537 | 0.547 | 生化统计重要 |
| - Environment | 312 | 0.567 | 0.520 | 环境特征关键 |
| Only env (baseline) | 5 | 0.799 | -0.016 | 环境单独无预测力 |

**关键发现：** 移除 Mamba local pool 后性能反而提升 (R² 0.828→0.844)，建议后续版本移除此组件。

### SOTA 工具对比 (2026-05-31 更新)

| 对比 | Confluencia (allele-aware) | Confluencia (allele-agnostic) | SOTA | 差距 (allele-aware) | 基准数据 |
|------|---------------------------|-------------------------------|------|---------------------|----------|
| vs NetMHCpan-4.1 | AUC 0.80 | AUC 0.74 | 0.92-0.96 | -0.12~-0.16 | 288K peptides |
| vs MHCflurry | AUC 0.83 (per-allele) | AUC 0.77 | 0.85-0.90 | -0.02~-0.07 | 52K binary |

**差距原因分析：** (1) 常见等位基因 (HLA-A*02:01) 内多样性大，需更多等位基因特异训练 (2) Confluencia 多任务而非专门做 MHC binding (3) 部分 alleles 已达 SOTA 级别 (A*33:01/03: 0.92-0.95)。**价值定位：** Confluencia 提供 RNACTM 药代动力学仿真、剂量优化、免疫原性预测等多任务能力，专业工具无法覆盖。

## 版本架构说明

Confluencia v1.0（`src/`目录）和 v2.0 共享同一套计算后端。v1.0 提供统一的 Streamlit 前端（`src/frontend.py`），通过 `sys.path` 导入 v2.0 核心模块（PINN、GNN、multiscale、docking 等），作为单页面多标签应用运行。v2.0 将各模块拆分为独立前端（Drug `app.py`、Epitope `epitope_frontend.py`、circRNA `app.py`），保留所有 v1.0 计算能力并新增 circRNA 评估维度（RNACTM、5D评分、跨模态决策）。推荐使用 v2.0 前端；v1.0 统一前端仍可运行但不再主动维护。

## 架构概览

```
Confluencia/
├── confluencia-2.0-drug/         # 药物预测模块
│   ├── core/
│   │   ├── predictor.py          # 模型训练与预测（使用共享 ModelFactory）
│   │   ├── pipeline.py           # 完整流水线（CTM/PKPD 模拟已提取为共享函数）
│   │   ├── features.py           # 分子特征工程（RDKit/Morgan FP）
│   │   ├── featurizer.py         # MoleculeFeatures 编码器
│   │   ├── ctm.py                # CTM 六室药代动力学模型
│   │   └── ...
│   ├── tests/
│   └── app.py                    # Streamlit 前端
│
├── confluencia-2.0-epitope/      # 表位预测模块
│   ├── core/
│   │   ├── training.py           # 训练主入口（含日志/时间记录）
│   │   ├── predictor.py          # 模型训练与预测
│   │   ├── features.py           # 序列特征工程（向量化实现）
│   │   ├── mamba3.py             # Mamba3Lite 编码器（支持状态序列化）
│   │   ├── torch_mamba.py        # PyTorch Mamba 深度模型
│   │   └── ...
│   ├── tests/
│   └── epitope_frontend.py       # Streamlit 前端
│
├── confluencia_shared/           # 共享库（统一基础设施）
│   ├── __init__.py
│   ├── models.py                 # ModelFactory + ModelConfig + ModelName
│   ├── metrics.py                # 统一指标计算 (RMSE, MAE, R2)
│   ├── moe.py                    # MOE 集成学习（ExpertConfig + MOERegressor）
│   ├── protocols.py              # PredictableRegressor 协议
│   ├── data_utils.py             # resolve_label 等数据工具
│   ├── training.py               # 训练辅助（EarlyStopping, 学习率调度）
│   ├── features/
│   │   └── bioseq.py             # 氨基酸常量与生物序列工具
│   ├── optim/                    # 差分进化优化器
│   └── utils/
│       ├── logging.py            # 统一日志框架
│       └── ema.py                # 指数移动平均
│
├── tests/
│   └── test_shared_modules.py    # 共享库单元测试（28 tests）
│
├── benchmarks/results/           # JSON 格式实验结果
├── docs/                         # 论文草稿（中英文）
├── figures/                      # 出版级图表（PNG 300 DPI + PDF）
├── pyproject.toml                # 项目配置
└── Dockerfile                    # Docker 多阶段构建
```

## 快速开始

### 1. 环境配置

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1     # Windows

pip install -r requirements-shared.txt
# 完整依赖：pip install -r requirements-shared-full.txt
```

### 2. 运行测试

```bash
# 从项目根目录执行（需正确解析 confluencia_shared）
python -m pytest confluencia-2.0-epitope/tests/ tests/test_shared_modules.py -v
```

### 3. 启动前端

```bash
# 表位模块
cd confluencia-2.0-epitope && PYTHONPATH=.. streamlit run epitope_frontend.py

# 药物模块
cd confluencia-2.0-drug && PYTHONPATH=.. streamlit run app.py

# 统一前端 (v1.0, 不再主动维护)
PYTHONPATH=. streamlit run src/frontend.py
```

### 4. 命令行

```bash
pip install -e .                              # 安装入口点
confluencia version                            # Confluencia CLI v2.1.0
confluencia drug props --smiles CC(=O)Oc1ccccc1C(=O)O  # 分子属性
confluencia interactive                        # 启动交互式 REPL
```

### 5. Docker

```bash
docker build -t confluencia .
docker run -p 8501:8501 -p 8502:8502 confluencia
```

### 6. R 包

R 包通过 [reticulate](https://github.com/rstudio/reticulate) 桥接 Python 后端，所有计算由 Confluencia Python 完成，R 侧只做输入输出转换。

**前提条件：** 需要一个安装了 Confluencia 依赖的 Python（pandas, numpy, scikit-learn, rdkit, scipy, joblib）。

```r
# 安装
devtools::install_github("IGEM-FBH/confluencia", subdir = "confluencia-rpkg")

# 指定 Python 环境（首次使用必须设置）
library(confluencia)
cf_use_python("/path/to/python")    # 或设置环境变量 CONFLUENCIA_PYTHON

# PK 仿真
params <- cf_ctm_params(binding = 0.72, immune = 0.65, inflammation = 0.12)
pk <- cf_ctm_simulate(dose = 200, freq = 2, binding = 0.72, horizon = 72)
rna_pk <- cf_rna_ctm_simulate(dose = 5, freq = 1, modification = "pseudouridine")

# 免疫原性
scores <- cf_circrna_immunogenicity("ACGUACGUACGUACGU")

# 联合评估
result <- cf_joint_evaluate(smiles = "CC(=O)Oc1ccccc1C(=O)O",
                             epitope_seq = "SLYNTVATL",
                             mhc_allele = "HLA-A*02:01",
                             dose_mg = 200, freq_per_day = 2, treatment_time = 72)

# 更多函数
cf_mhc_encode("SLYNTVATL", "HLA-A*02:01")      # 1018-dim MHC-I 编码
cf_mhc_detect_class("HLA-A*02:01")               # → "I"
cf_mamba3_encode("ACGUACGUACGU")                  # 多尺度序列编码
cf_reg_metrics(c(1,2,3), c(1.1,2,2.9))            # MAE/RMSE/R²
```

**完整函数列表 (22个)：**

| 函数 | 功能 | 返回类型 |
|------|------|----------|
| `cf_ctm_params()` | CTM 参数推导 | named numeric vector |
| `cf_ctm_simulate()` | 小分子 PK 仿真 | data.frame |
| `cf_rna_ctm_params()` | RNA-CTM 参数推导 | named numeric vector |
| `cf_rna_ctm_simulate()` | circRNA PK 仿真 | data.frame |
| `cf_circrna_immunogenicity()` | 免疫原性预测 | named numeric vector |
| `cf_circrna_pipeline()` | circRNA 全流水线 | list |
| `cf_joint_evaluate()` | 5D 联合评估 | list (composite + 5 dimensions) |
| `cf_drug_predict()` | 药物疗效预测 | numeric scalar |
| `cf_drug_train()` | **药物模型训练** | list (mae, rmse, r2, bundle_path) |
| `cf_epitope_predict()` | 表位结合预测 | numeric scalar |
| `cf_epitope_train()` | **表位模型训练** | list (mae, rmse, r2, bundle_path) |
| `cf_mhc_encode()` | MHC 特征编码 | numeric vector (1018/947 dim) |
| `cf_mhc_detect_class()` | MHC 类型检测 | character ("I"/"II") |
| `cf_mamba3_encode()` | Mamba3Lite 序列编码 | named list of vectors |
| `cf_reg_metrics()` | 回归指标计算 | named numeric vector |
| `cf_hub_push_model()` | **共享模型 (联邦式, 不暴露数据)** | list (model_id) |
| `cf_hub_pull_model()` | **下载社区模型** | list (bundle_path) |
| `cf_hub_list_models()` | **列出社区模型** | data.frame |
| `cf_hub_push_data()` | **贡献数据 (可选, 需指定许可)** | list (dataset_id) |
| `cf_hub_data_stats()` | **查看社区数据统计** | named list |
| `cf_use_python()` | 指定 Python 路径 | NULL (side effect) |
| `cf_find_python()` | 自动发现 Python | character path (invisible) |

```r
# 用自己的数据训练模型
result <- cf_drug_train("my_data.csv", model_name = "ridge")
print(result$r2)  # 0.914
pred <- cf_drug_predict(result$bundle_path, "CC(=O)Oc1ccccc1C(=O)O")

result <- cf_epitope_train("epitope_data.csv", target_col = "efficacy")
pred <- cf_epitope_predict(result$bundle_path, "SLYNTVATL")
```

```r
# 共享模型到社区（联邦式 — 不暴露原始数据）
cf_hub_push_model(result$bundle_path, metadata = list(r2 = result$r2))

# 下载并使用社区模型
models <- cf_hub_list_models(task = "drug")
community <- cf_hub_pull_model("hub:drug:anonymous:abc123")
pred <- cf_drug_predict(community$bundle_path, "CC(=O)Oc1ccccc1C(=O)O")

# 可选：贡献数据到社区池（匿名，需指定许可）
cf_hub_push_data("my_drug_data.csv", license = "CC-BY-4.0", anonymous = TRUE)
cf_hub_data_stats()  # 查看社区数据量
```

### 7. VS Code 扩展

VS Code 扩展通过 JSON-RPC 调用 Python 后端，提供命令面板交互和 PK 曲线可视化。

**前提条件：** 需要安装了 Confluencia 依赖的 Python 环境。

```bash
# 从 VSIX 安装（GitHub Releases 下载）
code --install-extension confluencia-vscode-0.1.0.vsix

# 或从源码编译
cd confluencia-vscode && npm install && npx tsc
# F5 启动扩展开发宿主
```

**命令面板命令 (11个)：**

| Command | 功能 |
|---------|------|
| `Confluencia: Simulate PK (CTM)` | 小分子 PK 仿真 + Plotly 曲线 |
| `Confluencia: Simulate circRNA PK` | circRNA PK 仿真 |
| `Confluencia: Predict Immunogenicity` | 免疫原性预测 |
| `Confluencia: Joint Evaluate (5D)` | 5D 联合评估 + 分数树 |
| `Confluencia: Predict Drug Efficacy` | 药物疗效预测 |
| `Confluencia: Predict Epitope Binding` | 表位结合预测 |
| `Confluencia: Train Drug Model` | **药物模型训练 (CSV → .joblib)** |
| `Confluencia: Train Epitope Model` | **表位模型训练 (CSV → .joblib)** |
| `Confluencia: Encode MHC Features` | MHC 特征编码 |
| `Confluencia: Compute Metrics` | 回归指标计算 |
| `Confluencia: Select Python Environment` | 设置 Python 路径 |

联合评估完成后，侧边栏显示 5D 分数树：
```
Composite: 0.73 [PROCEED]
  Clinical (0.68)    Binding (0.71)    Kinetics (0.62)
  Gene Signature (0.75)    CircRNA (0.77)
```

## 命令行界面 (CLI)

Confluencia 提供功能完整的命令行终端界面，支持交互式 REPL 和 60+ 子命令。

```bash
pip install -e .        # 安装入口点 confluencia
confluencia --help      # 查看所有命令
confluencia interactive # 交互式 REPL
```

主要模块：`drug` (分子属性/训练/筛选/PK仿真), `epitope` (表位预测/ESM-2编码/MHC编码), `circrna` (免疫原性/多组学/生存分析), `joint` (联合评估), `bench` (基准测试), `chart` (可视化)。

### Python API

Confluencia 提供统一的 Python API，R 包和 VS Code 扩展都通过它桥接后端：

```python
from confluencia_cli.bridge import ConfluenciaBridge

bridge = ConfluenciaBridge()

# PK 仿真
params = bridge.ctm_params(binding=0.72, immune=0.65, inflammation=0.12)
pk = bridge.ctm_simulate(dose=200, freq=2, binding=0.72, horizon=72)

# circRNA 免疫原性
scores = bridge.circrna_immunogenicity("ACGUACGUACGUACGU")

# 5D 联合评估
result = bridge.joint_evaluate({
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "epitope_seq": "SLYNTVATL", "mhc_allele": "HLA-A*02:01",
    "dose_mg": 200, "freq_per_day": 2, "treatment_time": 72
})

# MHC 编码
enc = bridge.mhc_encode("SLYNTVATL", "HLA-A*02:01")  # 1018-dim vector

# 训练模型（用自己的数据）
result = bridge.drug_train("my_drug_data.csv", model_name="ridge")
print(f"R²={result['r2']:.3f}, saved to {result['bundle_path']}")
pred = bridge.drug_predict(result["bundle_path"], "CC(=O)Oc1ccccc1C(=O)O")

result = bridge.epitope_train("epitope_data.csv", target_col="efficacy")
pred = bridge.epitope_predict(result["bundle_path"], "SLYNTVATL")

# 回归指标
metrics = bridge.reg_metrics([1,2,3], [1.1,2,2.9])
```

### 插件系统 (可扩展平台)

Confluencia 不是固定工具，而是可扩展平台——用户可以注册自定义算法替换任何阶段：

```python
import confluencia_cli.plugins as cf

# 注册 XGBoost 作为新模型类型
@cf.register_model("xgboost")
def create_xgb(**kwargs):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=6, **kwargs)

# 现在可以用 xgboost 训练
from confluencia_cli.bridge import ConfluenciaBridge
bridge = ConfluenciaBridge()
result = bridge.drug_train("data.csv", model_name="xgboost")

# 注册自定义评估维度
cf.register_dimension("manufacturability", weight=0.10,
                      description="Ease of large-scale circRNA production")

# 自定义评估权重
cf.set_weights(clinical=0.25, binding=0.20, kinetics=0.15,
               gene_signature=0.15, circrna=0.15, manufacturability=0.10)

# 查看所有注册组件
cf.list_registry()
# {'models': ['xgboost'], 'dimensions': ['manufacturability'], 'weights': {...}}
```

R 中同样可以注册：

```r
# 注册 XGBoost
cf_register_model("xgboost", "xgboost", "XGBRegressor")
result <- cf_drug_train("data.csv", model_name = "xgboost")

# 添加自定义维度
cf_register_dimension("manufacturability", weight = 0.10)
cf_set_weights(c(clinical=0.25, binding=0.20, kinetics=0.15,
                 gene_signature=0.15, circrna=0.15, manufacturability=0.10))
```

## 核心模块

### MOE 集成学习 (confluencia_shared/moe.py)

样本量自适应专家选择：
- **Low (N<80):** Ridge + HGB
- **Medium (80<=N<300):** Ridge + HGB + RF
- **High (N>=300):** Ridge + HGB + RF + MLP

权重公式：$w_e = \frac{1/RMSE_{OOF,e}}{\sum_{e'} 1/RMSE_{OOF,e'}}$

### Mamba3Lite 序列编码器 (mamba3.py)

三时间常数自适应状态空间递归 + 四尺度池化（residue/local/meso/global）+ 轻量自注意力增强，
输出 96 维摘要向量。支持 `get_state()`/`set_state()` 状态序列化。

注意力增强设计：
- QKV 投影降维（d_attn = max(8, d/2)）
- 因果注意力掩码 + 保守残差权重 (0.1)
- 最佳配置：d=16 时 MAE=0.395 (R²=0.802)，注意力补偿小模型容量损失

### RNACTM 药代动力学模型 (ctm.py)

六室 ODE 模型模拟 circRNA 从注射到清除的完整动力学过程，支持 5 种核苷酸修饰参数（m6A, Psi, 5mC, ms2m6A, 未修饰）。

## 共享库 (confluencia_shared)

| 模块 | 功能 | 被使用位置 |
|------|------|------------|
| `models.py` | ModelFactory, ModelConfig, ModelName | epitope/drug predictor.py |
| `metrics.py` | rmse(), reg_metrics() | 8+ 个核心文件 |
| `moe.py` | MOERegressor, ExpertConfig, **GatedMOERegressor** | epitope/drug moe.py |
| `protocols.py` | PredictableRegressor | epitope/drug predictor.py |
| `data_utils.py` | resolve_label() | epitope training.py, drug pipeline.py |
| `features/bioseq.py` | AA 常量, 序列分析工具 | mamba3.py, features.py |
| `utils/logging.py` | get_logger() | 所有核心模块 |
| `training.py` | EarlyStopping, 学习率调度 | 深度学习训练器 |
| `optim/hyperopt.py` | run_hyper_search(), RandomizedSearchCV | drug pipeline.py, epitope training.py |

## 可信评估框架

为小样本场景提供可靠的统计推断能力：

### Bootstrap CI 置信区间

对于交叉验证结果 (n=5 fold)，传统的 z 分布假设 (`1.96 * std/sqrt(n)`) 会导致置信区间过窄。
本框架采用自适应策略：

| 样本量 | 方法 | 说明 |
|--------|------|------|
| n < 10 | **t 分布** | 更准确的尾部估计 (t₀.₀₂₅,₄=2.776 vs z=1.96) |
| n ≥ 10 | **Bootstrap percentile** | 1000 次重采样，2.5%-97.5% 百分位区间 |

### 分层 K-Fold 交叉验证

普通 KFold 对可能有双峰分布的疗效指标无法保证各 fold 分布均衡。
本框架实现基于分位数分箱的分层交叉验证：

1. 将连续目标变量按分位数划分为 n_bins 个类别
2. 使用 `StratifiedKFold` 确保各类别在各 fold 中均衡分布
3. 样本不足时自动回退到普通 KFold

### GatedMOE 集成器

`GatedMOERegressor` 通过可学习的门控网络动态分配样本到不同专家：
- 使用 relu 激活的 MLP 门控 + 数值稳定的 softmax 权重归一化
- 每个样本的预测是各专家输出的加权和，权重由特征决定

### 可选超参数调优

支持 RandomizedSearchCV 和 GridSearchCV 对各专家模型进行调优：
- Ridge: `alpha` ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
- HGB: `max_depth` ∈ [4, 6, 8, 10], `learning_rate` ∈ [0.05, 0.1, 0.15, 0.2]
- RF: `n_estimators` ∈ [100, 200, 300], `max_depth` ∈ [6, 10, 14]
- MLP: `hidden_layer_sizes` ∈ [64], [128], [64, 32]

调优通过 Streamlit 前端一键启用，自动应用于 MOE 集成中的各专家。

## 测试

```bash
# 完整测试套件（32 tests）
python -m pytest confluencia-2.0-epitope/tests/smoke_test.py tests/test_shared_modules.py -v

# 4 个集成测试 + 28 个共享库单元测试
# 全部通过即表示核心功能正常
```

## 开发者指南

### 运行基准测试

```bash
# 完整基准测试套件
python -m benchmarks.run_all --data-epitope data/example_epitope.csv --data-drug data/example_drug.csv

# 快速基准
python -m benchmarks.quick_benchmark

# 消融实验
python -m benchmarks.ablation

# 真实数据端到端验证
python benchmarks/validate_all_real_data.py
```

### 代码风格

```bash
black --line-length 100 .
isort --profile black .
```
## 许可证

MIT License. 本仓库为研究/原型用途，代码与模型仅用于研究演示，不构成临床建议。

---

**Contact:** igem@fbh-china.org | **Repository:** https://github.com/IGEM-FBH/confluencia
