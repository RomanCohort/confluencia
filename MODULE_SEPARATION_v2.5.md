# Confluencia v2.5 模块分离更新

## 重要架构变更 (2026-05)

### 模块独立化

项目现已重构为两个完全独立的专业模块：

| 模块 | 专注领域 | 代码路径 | 前端文件 |
|------|---------|---------|---------|
| **Drug 2.0** | 小分子药物发现 | `confluencia-2.0-drug/` | `app_drug.py` |
| **circRNA** | circRNA疫苗开发 | `confluencia_circrna/` | `app.py` |

---

## 版本迭代更新

| 版本 | 代号 | 定位 | 时间线 |
|------|------|------|--------|
| v0.6.x | 早期集成版 | 全功能多模块原型 | 2026-01 |
| v2.0 | Drug 模块 | 药物疗效 + PK/PD 动力学 | 2026-04 |
| v2.0 | Epitope 模块 | 表位免疫疗效 + Mamba序列 | 2026-04 |
| v2.1 | 项目重构 | 共享库提取 + 目录整理 | 2026-04 |
| v2.2 | 真实数据验证 | Drug AUC=0.9252, Target AUC=0.8245 | 2026-04 |
| v2.3 | 五基因Signature | FiveGeneMOE + 生存分析 | 2026-04 |
| v2.4 | Bio-Mimetic | 拓扑药效团 + 突触修剪 | 2026-05 |
| **v2.5** | **模块分离** | **Drug/circRNA独立前端+后端** | **2026-05** |

---

## Drug 2.0 模块

### 专注领域
小分子药物发现

### 前端功能 (`app_drug.py`)

| 页面 | 功能 |
|------|------|
| 🧪 Molecule Input | SMILES输入、验证、批量上传 |
| 📊 ADMET Prediction | 吸收/分布/代谢/排泄/毒性 |
| 🧬 ED2Mol Generation | 结构导向分子设计 |
| 📈 PK/PD Simulation | 药代/药效动力学模拟 |
| 🧪 Molecule Evolution | 进化优化 + Pareto筛选 |
| 🎯 Target Docking | 分子-靶点对接预测 |

### 核心模块

```
confluencia-2.0-drug/core/
├── ed2mol_adapter.py      # ED2Mol分子生成
├── ed2mol_templates.py    # 配置模板
├── evolution.py           # 分子进化优化
├── admet.py               # ADMET预测
├── pkpd.py                # PK/PD模拟
├── docking.py             # 靶点对接
├── ctm.py                 # 四房室动力学
├── moe.py                 # MOE集成学习
└── features.py            # 分子特征工程
```

### 运行方式

```bash
cd confluencia-2.0-drug
streamlit run app_drug.py
```

---

## circRNA 模块

### 专注领域
circRNA疫苗设计与开发

### 前端功能 (`app.py`)

| 页面 | 功能 |
|------|------|
| 📊 Sequence Analysis | RIG-I/TLR/PKR评分、结构预测、修饰分析 |
| 🧪 Sequence Design | 进化优化、IRES设计、修饰选择 |
| 💉 Vaccine Development | IPS评分、药效预测、治疗方案 |
| 📋 Clinical Report | 生存分析、不良反应、报告生成 |

### 核心模块

```
confluencia_circrna/core/
├── immune_sensing.py      # RIG-I/TLR/PKR评分 (文献权重)
├── structure_prediction.py # ViennaRNA二级结构
├── folding_kinetics.py    # 折叠动力学
├── cotrans_folding.py     # 共转录折叠
├── folding_pathways.py    # 折叠路径分析
├── cirrna_evolution.py    # circRNA序列进化
├── rna_modifications.py   # m6A/IRES/miRNA/RBP
├── drug_response.py       # circRNA疫苗药效
├── clinical_prediction.py # 临床预后预测
└── rna_docking.py         # RNA-药物对接
```

### 运行方式

```bash
cd confluencia_circrna
streamlit run app.py
```

---

## 文献支持

### Drug模块核心文献

| 功能 | 文献 |
|------|------|
| MOE集成 | Chernoff, 1952 - 样本量自适应 |
| CTM动力学 | Gibaldi & Perrier, 1982 |
| ED2Mol | pineappleK/ED2Mol GitHub |
| PK/PD | Gabrielsson & Weiner, 2000 |

### circRNA模块核心文献

| 功能 | 文献 |
|------|------|
| RIG-I评分 | Schlee et al., 2009 (Nature) |
| TLR7/8 | Forsbach et al., 2008 |
| PKR-dsRNA | Nallagatla et al., 2007 |
| m6A免疫影响 | Liu et al., 2022 (Nature) |
| circRNA翻译 | Yang et al., 2017 |
| miRNA海绵 | Hansen et al., 2013 |
| IPS评分 | Cristescu et al., 2018 |
| TIDE评分 | Jiang et al., 2018 |

---

## 快速启动

### 方式1: 启动器

```bash
# Windows
start_confluencia.bat

# Linux/Mac
./run.sh
```

### 方式2: 直接运行

```bash
# Drug模块
streamlit run confluencia-2.0-drug/app_drug.py

# circRNA模块
streamlit run confluencia_circrna/app.py
```

---

## 依赖安装

### Drug模块

```bash
pip install streamlit rdkit pandas numpy plotly scikit-learn
```

### circRNA模块

```bash
pip install streamlit pandas numpy plotly

# 可选: ViennaRNA (Linux)
apt-get install vienna-rna
```

---

## AutoDL部署

```bash
# 拉取最新代码
git pull origin main

# 安装依赖
pip install streamlit pandas numpy plotly

# 运行
streamlit run confluencia_circrna/app.py
# 或
streamlit run confluencia-2.0-drug/app_drug.py
```

---

## 功能对比

| 特性 | Drug Module | circRNA Module |
|------|-------------|----------------|
| **输入类型** | SMILES分子 | RNA序列 |
| **核心分析** | ADMET, PK/PD | 免疫评分, 结构 |
| **生成方式** | ED2Mol (结构) | 进化 (序列) |
| **预测目标** | 结合, 毒性 | 免疫原性, 生存 |
| **临床应用** | 药效预测 | 疫苗响应 |
| **代码规模** | 774行前端 + 5核心模块 | 958行前端 + 10核心模块 |

---

## 迁移说明

### 从旧版app.py迁移

**Drug 2.0:**
- 旧: `confluencia-2.0-drug/app.py` (2628行，混合circRNA)
- 新: `confluencia-2.0-drug/app_drug.py` (774行，专注药物)

**circRNA:**
- 旧: 无独立前端
- 新: `confluencia_circrna/app.py` (958行，专用疫苗)

### 已移除

- Drug模块中的circRNA进化功能 → 迁移至circRNA模块
- Drug模块中的CircRNAFeatureSpec → 迁移至circRNA模块
- Drug模块中的RNACTM六房室 → 迁移至circRNA模块

---

## 更新日志

### 2026-05 (v2.5)

**新增:**
- `confluencia_circrna/app.py` - circRNA专用前端
- `confluencia-2.0-drug/app_drug.py` - Drug专用前端
- `start_confluencia.bat` / `run.sh` - 启动器
- `README_modules.md` - 模块文档

**重构:**
- `rna_evolution.py` → `cirrna_evolution.py` (仅保留circRNA)
- 移除Drug模块中的ED2Mol导入 (保留在本地)
- 分离分子进化和circRNA进化

**移除:**
- Drug模块中的circRNA相关代码
- 混合功能页面

---

## 贡献指南

请根据模块专注领域提交代码：

- **Drug相关**: `confluencia-2.0-drug/`
- **circRNA相关**: `confluencia_circrna/`
- **共享工具**: `confluencia_shared/`