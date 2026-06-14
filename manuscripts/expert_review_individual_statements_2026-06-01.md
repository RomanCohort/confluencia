# Bioinformatics Application Note - 四领域专家独立阐述

## 稿件信息

**标题:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**评审日期:** 2026-06-01

---

# 方法论专家独立阐述

## 我的专业视角

我是方法论专家，专注于评估算法设计的科学性、参数设置的合理性、以及验证实验的严谨性。我的评审标准是：**算法是否有坚实的理论基础？参数是否有实证依据？验证是否充分？**

---

## 我的评分与建议

- **方法论评分: 3/5**
- **建议: Major Revision**

---

## 我为什么给3分

### 扣分项 (共-2分)

| 问题 | 扣分 | 原因 |
|------|------|------|
| 权重参数无经验校准 | -0.5 | 所有免疫评分权重为"author-informed heuristics" |
| 验证样本量严重不足 | -0.75 | N=10完全无法支持统计结论 |
| MOE命名误导 | -0.25 | 实际是WMA，非真正的Mixture-of-Experts |
| 缺乏独立验证和对照比较 | -0.5 | 无外部数据集、无工具对比 |

---

## 我的核心理由

### 问题1: 权重参数是"猜"的，不是"测"的

我仔细检查了`immune_sensing.py`的代码，发现了这个注释：

```python
# NOTE: These weights are author-informed heuristics, NOT empirically calibrated.
# Direction consistency with literature IFN data has been validated (ρ=0.93),
# but quantitative accuracy is not claimed.
```

**我的理解:** 你们诚实承认权重是启发式设定，这很好。但问题是：

1. 论文Methods部分说"literature-backed scoring"，暗示有文献支持
2. 实际上文献只提供了"方向一致性"，没有提供具体数值
3. RIG-I=0.35, TLR7=0.20, TLR8=0.15, PKR=0.30这些数字从哪来？

**我的计算:**

我检查了权重敏感性测试结果(`rig_i_weight_optimization_results.json`)：

```
所有20种权重组合的agreement率: 均为66.7%
```

这意味着：**无论怎么调整权重，结果都一样。** 这说明：
- 要么验证数据太少(N=7)，无法区分
- 要么各通路评分高度相关，权重不重要

**我的判断:** 权重参数科学依据不足，需要实验校准或更充分的敏感性分析。

---

### 问题2: N=10的验证毫无统计学意义

论文Results部分报告：
> "r=0.85 (GC content与免疫原性的相关性)"

我计算了N=10时这个相关系数的置信区间：

**r=0.85, N=10 的95% CI: [0.47, 0.96]**

置信区间宽度是0.49！这意味着真实相关性可能是"中等"(0.47)也可能是"极强"(0.96)，你无法得出任何可靠结论。

**更严重的问题:** 

我检查了你们的IFN相关性验证：
```json
"literature_cases": {
  "n": 7,
  "spearman_r": 0.135,
  "pearson_r": -0.056
}
```

只有7个案例，且相关性极弱。这怎么支撑"免疫原性预测"的声称？

**我的建议:** 

你们有`circbase_large_scale_validation.py`可以跑N=5000的验证，为什么不用？验证数据已经在代码里准备好了，只需要运行并报告结果。

---

### 问题3: MOE命名不准确

论文声称使用"MOE (Mixture-of-Experts)"，但我检查`moe.py`后发现：

```python
# NOTE: Despite the "MOE" naming, this module implements 
# Weighted Model Averaging (WMA), not a learnable Mixture 
# of Experts with gating networks.
```

**WMA vs 真正的MoE:**

| 特征 | 真正的MoE | 本项目实现 |
|------|-----------|------------|
| 门控网络 | 可学习，输入依赖 | 无 |
| 专家选择 | 动态路由 | 固定权重 |
| 权重来源 | 训练学习 | inverse-RMSE启发式 |

你们实现的是加权模型平均，不是真正的Mixture-of-Experts。命名有误导性。

**我的建议:** 改名为"Weighted Ensemble"或明确说明是"WMA而非gated MoE"。

---

## 我认可的部分

### ✓ 算法设计有文献基础

你们正确引用了关键文献：
- RIG-I: Schlee 2009, Zhang 2016
- TLR7/8: Forsbach 2008
- PKR: Nallagatla 2007
- m6A: Liu 2022

### ✓ circRNA特异性处理正确

```python
# circRNA is a covalently closed loop with NO 5' or 3' ends.
# RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing
```

这个理解是正确的。你们区分了circRNA与线性RNA的RIG-I激活机制差异。

### ✓ 代码诚实透明

你们在代码注释中明确承认限制，这是良好的科学实践。

---

## 我的具体要求

### 必须完成 (Major Revision接受条件)

1. **扩展验证至N≥50**
   - 使用现有的`circbase_large_scale_validation.py`
   - 报告完整的统计指标(95% CI, p值)

2. **权重敏感性分析**
   - 如果验证数据无法区分权重，说明权重不重要
   - 或者找到有实验IFN数据的数据集进行校准

3. **MOE命名修正**
   - 改为"Weighted Ensemble"或明确说明差异

### 建议完成

4. 与现有RNA免疫预测工具定量比较
5. 进化算法rounds增至10-20轮(解决收敛问题)
6. 报告ViennaRNA回退模式的精度损失

---

## 我的结论

> "算法框架设计合理，生物学理解正确，但**方法论的科学严谨性不足**——权重无校准、验证不充分、命名不准确。这些问题可以通过补充实验和数据解决，但需要较大工作量。因此我建议Major Revision。"

---

# RNA生物学专家独立阐述

## 我的专业视角

我是RNA生物学专家，专注于评估免疫传感器机制、circRNA特性处理、以及RNA修饰的生物学准确性。我的评审标准是：**生物学机制是否正确？circRNA特异性是否考虑？文献引用是否准确？**

---

## 我的评分与建议

- **生物学准确性评分: 4/5**
- **建议: Minor Revision**

---

## 我为什么给4分

### 加分项 (共+4分)

| 方面 | 加分 | 原因 |
|------|------|------|
| circRNA特性正确处理 | +1 | 明确区分无5'/3'端 |
| RIG-I机制正确 | +1 | dsRNA backbone替代blunt-end |
| 文献引用准确 | +1 | Schlee/Zhang/Nallagatla等恰当 |
| m6A双向效应识别 | +1 | 考虑了免疫逃逸和翻译增强 |

### 扣分项 (共-1分)

| 问题 | 扣分 | 原因 |
|------|------|------|
| 论文描述与代码不符 | -0.5 | Methods说"blunt-end detection"但代码正确实现dsRNA机制 |
| 权重缺乏实验验证 | -0.5 | TLR7/8分离评分等创新点无定量验证 |

---

## 我的核心理由

### 为什么生物学评价高：代码正确，论文描述有瑕疵

我检查了核心免疫评分代码，发现了非常专业的circRNA处理：

```python
# immune_sensing.py 第9-17行

# IMPORTANT: circRNA is a covalently closed loop with NO 5' or 3' ends.
# Therefore:
# - RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing
#   (the canonical linear RNA pathway; Schlee et al., 2009)
# 
# - RIG-I may be INDIRECTLY activated by circRNA through:
#   * dsRNA structures (backbone-forming inverted repeats)
#   * Short 5' overhangs from incomplete back-splicing (rare)
#
# Reference: Zhang et al., Nature Immunology 2016
```

**这是完全正确的生物学理解！**

你们明确区分了：
- **线性RNA RIG-I激活:** 5'-三磷酸 + blunt-end dsRNA
- **circRNA RIG-I激活:** dsRNA backbone结构(间接)

### 但论文描述有问题

论文Methods第34行：
> "RIG-I recognition is predicted using blunt-end detection and GU-rich content analysis"

**这个描述对circRNA是误导的。** "blunt-end detection"对circRNA不适用。代码正确，但论文描述会让读者误解。

**我的建议:** 改为：
> "RIG-I recognition is predicted via dsRNA backbone structure detection, as circRNAs lack 5' termini required for canonical blunt-end sensing."

---

## 我的详细评估

### RIG-I机制: 4/5 ✓

| 评估项 | 代码实现 | 文献依据 | 正确性 |
|--------|----------|----------|--------|
| 5'-ppp检测 | ✓ 正确绕过 | Schlee 2009 | ✓ |
| dsRNA backbone检测 | ✓ inverted repeat | Zhang 2016 | ✓ |
| GU-rich motif | ✓ 内容分析 | Schlee 2009 | ✓ |
| 权重(40% dsRNA, 30% motif) | △ 启发式 | 无直接依据 | △ |

**评价:** 机制正确，权重是估计值。

### TLR7/8机制: 4/5 ✓

我检查了TLR评分代码：

```python
# TLR7: GU-rich motifs
TLR7_MOTIFS = ["GUUG", "GUGU", "UGUU", "GUCU", "GUUU"]

# TLR8: AU-rich motifs  
TLR8_MOTIFS = ["AUUA", "UUAU", "UAUU", "AUUU", "UAAU"]
```

**这是正确的！** 文献(Forsbach 2008)确实表明：
- TLR7偏好GU-rich序列
- TLR8偏好AU-rich序列

**创新点:** 你们将TLR7和TLR8分开评分，这是比现有工具更精细的处理。

**但我有一个问题:** circRNA闭环校正因子0.70从哪来？

```python
# Apply circRNA closed-loop correction
score *= 0.70  # Reduced ssRNA exposure
```

这个0.70是合理的定性估计，但没有文献依据。

### PKR机制: 4.5/5 ✓

```python
PKR_MIN_DSRNA = 30  # or 33 in structure_prediction.py
```

文献依据正确：Nallagatla et al., RNA 2007确实表明PKR激活需要>33bp dsRNA。

**小问题:** 为什么两个模块定义不一致(30 vs 33)？

### m6A机制: 4/5 ✓

我检查了`rna_modifications.py`，发现你们正确处理了m6A的**双向效应**：

```python
# m6A can have TWO opposing effects on immunogenicity:
# 1. IMMUNE EVASION (dominant effect):
#    - YTHDF2-mediated degradation reduces RNA availability
#    - RIG-I evasion through reduced dsRNA sensing
#
# 2. TRANSLATION ENHANCEMENT:
#    - YTHDF1/YTHDF3 enhance translation
#    - May increase antigen presentation and immune visibility

evasion_weight = prob * 0.40  # RIG-I evasion
evasion_weight += prob * 0.25  # dsRNA destabilization
enhancement_weight = prob * 0.15  # Translation enhancement
```

**这是非常好的处理！** 很多工具只考虑m6A"降低免疫原性"，但你们正确识别了context-dependent效应。

**小问题:** 权重(0.40, 0.25, 0.15)从哪来？

---

## 我发现的问题

### 必须修改

| 问题 | 位置 | 要求 |
|------|------|------|
| Methods说"blunt-end detection" | 第34行 | 改为"dsRNA backbone" |
| PKR阈值不一致 | 两个模块 | 统一为33bp |

### 建议修改

| 问题 | 说明 |
|------|------|
| 权重缺乏验证 | TLR7/8分离、m6A双向等创新需数据支撑 |
| m6A预测仅为DRACH | 说明与SRAMP/WHISTLE精度差距 |
| 细胞类型特异性 | TLR7(pDC) vs TLR8(单核)表达差异未讨论 |

---

## 我认可的部分

### ✓ 文献引用准确

| 通路 | 引用 | 适用性 |
|------|------|--------|
| RIG-I 5'-ppp | Schlee 2009 | ✓ 线性RNA机制，你们正确绕过 |
| circRNA RIG-I | Zhang 2016 | ✓ circRNA特异性，引用正确 |
| TLR7/8 motif | Forsbach 2008 | ✓ GU/AU偏好，正确应用 |
| PKR dsRNA | Nallagatla 2007 | ✓ 33bp阈值，引用正确 |
| m6A修饰 | Liu 2022 | ✓ DRACH定义，引用正确 |

### ✓ circRNA生物学处理正确

| 特性 | 处理 | 评价 |
|------|------|------|
| 无5'/3'端 | RIG-I绕过blunt-end | ✓ 正确 |
| Back-splice junction | 操作时保护 | ✓ 正确 |
| 无传统UTR | 改为"IRES/flanking" | ✓ 正确 |
| 共价闭环 | TLR校正因子0.70 | △ 合理估计 |

### ✓ ViennaRNA集成

你们用真实的RNAfold结构预测替代启发式估计，这是正确的选择。

---

## 我的具体要求

### 必须修改 (Minor Revision接受条件)

1. **修正论文RIG-I描述** — 一句话修改，改为"dsRNA backbone"

2. **统一PKR阈值** — 删除重复定义，统一为33bp

### 建议修改

3. 补充权重文献依据或标注"启发式估计"
4. 说明m6A预测为初步筛查，精度低于专业工具
5. 讨论TLR细胞类型特异性

---

## 我的结论

> "作为RNA生物学专家，我对你们的生物学理解给予**高度评价**。circRNA特性处理正确，免疫机制理解深入，文献引用准确。主要问题是**论文描述与代码不一致**——代码是正确的，论文需要修正一句话。这是一个简单的Minor Revision即可解决的问题。"

---

# 软件工程专家独立阐述

## 我的专业视角

我是软件工程专家，专注于评估软件架构、代码质量、API设计、测试覆盖、以及多平台支持。我的评审标准是：**架构是否清晰？代码是否规范？测试是否充分？是否易于使用和扩展？**

---

## 我的评分与建议

- **软件质量评分: 4.2/5**
- **建议: Minor Revision**

---

## 我为什么给4.2分

### 加分项 (共+4.4分)

| 方面 | 加分 | 原因 |
|------|------|------|
| 模块化设计 | +0.8 | 共享库模式，11核心模块职责清晰 |
| API规范 | +0.7 | 131个显式导出，命名一致 |
| 多平台支持 | +0.8 | Python/R/VS Code/CLI/Docker六平台 |
| 插件系统 | +0.6 | 支持用户扩展 |
| 文档完整 | +0.8 | README详尽，性能指标清晰 |
| 配置规范 | +0.7 | pyproject.toml标准配置 |

### 扣分项 (共-0.2分)

| 问题 | 扣分 | 原因 |
|------|------|------|
| Drug模块测试缺失 | -0.15 | tests目录不存在或为空 |
| 无CI自动化 | -0.05 | .github存在但无测试workflow |

---

## 我的核心理由

### 为什么软件评价高：架构成熟，设计规范

我检查了项目结构，发现了良好的软件工程实践：

```
Confluencia/
├── confluencia_shared/        # 共享库 — 避免重复
│   ├── models.py              # ModelFactory
│   ├── metrics.py             # 统一指标
│   ├── moe.py                 # 集成学习
│   └── utils/                 # 工具函数
│
├── confluencia_circrna/       # circRNA模块
│   └── core/                  # 11个核心模块
│
├── confluencia-2.0-drug/      # Drug模块
├── confluencia-2.0-epitope/   # Epitope模块
│
├── confluencia-rpkg/          # R包
├── confluencia-vscode/        # VS Code扩展
├── confluencia_cli/           # CLI工具
│
└── tests/                     # 测试目录
```

### 共享库模式: 优秀设计

```python
# confluencia_shared/models.py

class ModelFactory:
    """统一的模型工厂，被Drug/Epitope/circRNA模块共享"""
    
    @staticmethod
    def create(model_name: ModelName, **kwargs) -> PredictableRegressor:
        ...
```

这种设计避免了代码重复，保证了各模块行为一致。

### API设计: 规范良好

```python
# confluencia_circrna/core/__init__.py

__all__ = [
    # Core functions
    "predict_circrna_immunogenicity",
    "predict_modifications", 
    "predict_clinical_outcome",
    "run_cirrna_evolution",
    ...
    # Classes
    "ImmunogenicityScorer",
    "StructurePredictor",
    ...
]
```

131个API显式导出，边界清晰。

### 多平台支持: 业界领先

| 平台 | 实现方式 | 覆盖度 |
|------|----------|--------|
| Python API | 原生 | 100% |
| CLI | confluencia_cli | 60+子命令 |
| R Package | reticulate桥接 | 27函数 |
| VS Code | TypeScript JSON-RPC | 11命令 |
| Web UI | Streamlit | 多前端 |
| Docker | 多阶段构建 | 完整 |

**这在中文学术软件中非常少见！**

---

## 我发现的问题

### Major问题: Drug模块测试缺失

我在`pyproject.toml`中看到：

```toml
[tool.pytest.ini_options]
testpaths = [
    "tests",
    "confluencia-2.0-drug/tests",  # 这个目录不存在！
    "confluencia-2.0-epitope/tests",
]
```

但`confluencia-2.0-drug/tests/`目录不存在或为空。

**这是Major问题：** Drug模块是核心功能之一，没有测试意味着：
1. 无法保证代码质量
2. 重构风险高
3. 不符合Bioinformatics期刊对软件质量的要求

### Minor问题: 无CI自动化

```
.github/
├── workflows/
│   └── (空)  # 没有测试workflow
```

**建议:** 添加GitHub Actions自动运行测试。

### Minor问题: 依赖版本过严

```toml
dependencies = [
    "numpy>=1.24,<1.25",  # 为什么限制<1.25？
    "pandas>=2.0,<2.1",
]
```

这种严格限制可能导致兼容性问题。科学计算场景通常允许更宽松的版本。

---

## 测试覆盖评估

| 测试位置 | 测试数 | 覆盖内容 | 评价 |
|----------|--------|----------|------|
| `tests/test_shared_modules.py` | 28 | 共享库 | ✓ 充分 |
| `tests/test_core_modules.py` | 50+ | Features/MOE/Mamba3/CTM | ✓ 充分 |
| `confluencia_circrna/tests/` | 6 files | Pipeline全流程 | ✓ 充分 |
| `confluencia-rpkg/tests/` | testthat | R桥接 | ✓ 存在 |
| `confluencia-2.0-drug/tests/` | **0** | Drug模块 | **✗ 缺失** |

**覆盖率估算:** ~60-70% (Drug模块为0)

---

## 我认可的部分

### ✓ 模块化设计清晰

| 设计特点 | 实现 | 评价 |
|----------|------|------|
| 共享库 | confluencia_shared | ✓ 避免重复 |
| 模块分离 | 11核心模块 | ✓ 高内聚低耦合 |
| 插件系统 | register_model/dimension | ✓ 可扩展 |
| Bridge模式 | ConfluenciaBridge | ✓ 统一访问 |

### ✓ API文档化良好

README详细列出：
- 27个R函数
- 11个VS Code命令
- 60+个CLI子命令
- Python API使用示例

### ✓ Docker配置完备

```dockerfile
# 多阶段构建
FROM python:3.10-slim as builder
...

FROM python:3.10-slim as runtime
...
```

生产就绪的多阶段构建。

---

## 我的具体要求

### Major (必须完成)

1. **补充Drug模块测试**
   - 新建`confluencia-2.0-drug/tests/`目录
   - 至少添加smoke tests

### Minor (建议完成)

2. 添加GitHub Actions CI workflow
3. 放宽依赖版本约束(`numpy>=1.24,<2.0`)
4. 统一测试目录结构
5. VS Code扩展添加单元测试
6. 清理冗余目录(`confluencia_2_0_drug` vs `confluencia-2.0-drug`)

---

## 我的结论

> "作为软件工程专家，我对你们的软件架构给予**高度评价**。共享库模式、多平台支持、插件系统都体现了成熟的工程实践。主要问题是**Drug模块测试缺失**，这是一个需要补充但工作量不大的问题。建议Minor Revision。"

---

# 临床转化专家独立阐述

## 我的专业视角

我是临床转化专家，专注于评估生物医学工具的临床相关性、预测模型的适用性、以及向临床应用的转化潜力。我的评审标准是：**解决什么临床问题？预测结果如何指导决策？验证是否充分？安全性考量是否完整？**

---

## 我的评分与建议

- **临床相关性评分: 3/5**
- **建议: Minor Revision**

---

## 我为什么给3分

### 加分项 (共+3分)

| 方面 | 加分 | 原因 |
|------|------|------|
| 解决真实临床需求 | +1 | circRNA疫苗免疫原性预测是有价值的临床问题 |
| PK模型设计合理 | +0.5 | 六室模型反映真实生物学过程 |
| 安全性考量已纳入 | +0.5 | irAE预测模块存在 |
| 文献基础扎实 | +1 | 免疫机制有充分文献支撑 |

### 扣分项 (共-2分)

| 问题 | 扣分 | 原因 |
|------|------|------|
| 临床验证薄弱 | -0.75 | 无真实临床数据支撑预测结果 |
| 决策阈值不完整 | -0.5 | 免疫原性评分临床解释模糊 |
| AE风险无依据 | -0.5 | 概率值未经校准 |
| 适用人群不清 | -0.25 | 预防性vs治疗性疫苗边界模糊 |

---

## 我的核心理由

### 平台定位: 有价值的临床工具

circRNA疫苗是新兴治疗手段，免疫原性控制是关键挑战：

| 场景 | 免疫原性要求 | 平台价值 |
|------|--------------|----------|
| 疫苗设计 | 高免疫原性=强疫苗效应 | ✓ 筛选高免疫原性候选 |
| 药物递送 | 低免疫原性=减少副作用 | ✓ 筛选低免疫原性候选 |

**这是真正的临床需求。**

### 问题1: 临床验证薄弱

我检查了验证数据：

| 验证类型 | 数据来源 | 临床相关性 |
|----------|----------|------------|
| circBase N=10 | 公共数据库 | ✗ 无临床样本 |
| IFN相关性 N=7 | 文献案例 | ✗ 样本太小 |
| 生存预测 | 无真实数据 | ✗ 算法框架存在但未验证 |

**关键缺失:**

1. 无TCGA等临床数据集验证生存预测
2. 无已知免疫原性的临床circRNA验证
3. IPS/TIDE在circRNA场景的适用性未建立

### 问题2: 决策阈值体系不完整

论文提到：
> "高免疫原性序列(0.88)适合疫苗，低免疫原性(0.35)适合药物递送"

**我的问题:**

| 问题 | 现状 |
|------|------|
| 阈值0.88和0.35的选择依据？ | 不明 |
| 中间区域(0.35-0.88)如何决策？ | 未定义 |
| 不同疫苗类型的阈值差异？ | 未讨论 |

**临床医生需要明确的决策边界。**

### 问题3: AE风险预测缺乏依据

我检查了AE预测代码：

```python
# clinical_prediction.py

colitis_prob = 0.05 + immunogenicity * 0.1   # 范围5-15%
dermatitis_prob = 0.03 + immunogenicity * 0.05  # 范围3-8%
hepatitis_prob = 0.03 + immunogenicity * 0.05   # 范围3-8%
```

**这些数字从哪来？**

- 5-15%的colitis风险范围有文献依据吗？
- 为什么是线性关系(immunogenicity × 0.1)？
- 这些概率值经过临床数据校准吗？

**如果没有依据，应该标注为"估计值"或移除具体数值。**

### 问题4: 适用人群边界不清

IPS/TIDE评分的使用场景：

| Score | 原设计目的 | 验证人群 | circRNA适用性 |
|-------|------------|----------|---------------|
| IPS | ICI响应预测 | 肿瘤患者 | 治疗性疫苗：可能适用 |
| TIDE | 免疫逃逸评估 | 肿瘤患者 | 预防性疫苗：不适用 |

**关键问题:** 预防性疫苗用于健康人群，基因表达谱与肿瘤患者差异显著。IPS/TIDE是否适用？

---

## 预测模型临床适用性评估

| 模块 | 临床用途 | 验证程度 | 可信度 |
|------|----------|----------|--------|
| 免疫原性评分 | 疫苗候选筛选 | 文献权重，无临床验证 | △ 理论可靠，验证弱 |
| 生存预测 | 患者预后评估 | 算法完整，无真实数据 | △ 框架可用，数据缺 |
| PK模型 | 剂量优化 | 六室模型，文献参数 | △ 理论合理，需实验 |
| AE风险 | 安全性评估 | 概率模型，无临床校准 | ✗ 缺乏依据 |

---

## 我发现的问题

### 必须修改

| 问题 | 要求 |
|------|------|
| 临床阈值定义 | 建立免疫原性评分的决策边界 |
| AE风险依据 | 补充文献引用或标注"估计值" |
| TCGA验证 | 展示生存预测C-index结果 |

### 建议修改

| 问题 | 要求 |
|------|------|
| 适用人群明确 | 区分预防性/治疗性疫苗 |
| IPS/TIDE限制讨论 | 说明从mRNA/ICI场景迁移的假设 |
| 剂量-毒性关系 | PK模型整合剂量与AE风险 |

---

## 我认可的部分

### ✓ 解决真实临床问题

circRNA疫苗免疫原性预测是新兴领域的技术痛点，平台定位有价值。

### ✓ PK模型设计科学

六室模型(注射→LNP→内体→胞质→翻译→清除)反映真实生物学过程。

### ✓ 安全性考量已纳入

irAE预测模块存在，提供了AE类型和管理策略。

### ✓ 代码透明度高

参数来源有注释，限制条件有声明。

---

## 我的具体要求

### 必须修改 (Minor Revision接受条件)

1. **明确临床阈值**
   - 定义免疫原性评分的决策边界
   - 区分不同应用场景(疫苗vs药物递送)

2. **补充TCGA验证**
   - 至少展示一个数据集的生存预测结果
   - 报告C-index和校准曲线

3. **AE风险依据**
   - 补充概率值的文献引用
   - 或标注为"启发式估计"

### 建议修改

4. 讨论IPS/TIDE在circRNA场景的适用性限制
5. 区分预防性/治疗性疫苗的应用边界
6. 整合剂量-毒性预测

---

## 临床转化建议

### 短期(论文发表前)

1. 补充TCGA-BRCA生存预测验证
2. 建立免疫原性评分阈值体系
3. 讨论IPS/TIDE适用性限制

### 中期(工具开发)

1. 剂量-毒性预测模块
2. 特殊人群风险模型
3. 临床试验数据校准

### 长期(临床转化)

1. circRNA疫苗临床试验合作
2. FDA/EMA预测报告格式
3. 监管决策支持模块

---

## 我的结论

> "作为临床转化专家，我认为平台有明确的临床价值和转化潜力。主要短板是**临床验证薄弱**和**决策支持体系不完整**。这些都是可以通过补充数据和澄清说明解决的问题。建议Minor Revision，要求补充验证数据和明确临床决策边界。"

---

# 四专家共识汇总

## 评分对比

| 专家 | 评分 | 建议 | 核心关注 |
|------|------|------|----------|
| 方法论 | 3.0/5 | Major | 权重校准、验证充分性 |
| RNA生物学 | 4.0/5 | Minor | 论文描述修正 |
| 软件工程 | 4.2/5 | Minor | 测试补充 |
| 临床转化 | 3.0/5 | Minor | 临床验证、决策边界 |

## 共识问题

| 问题 | 方法论 | RNA生物学 | 软件工程 | 临床转化 | 优先级 |
|------|--------|-----------|----------|----------|--------|
| 验证不足(N=10) | ⚠️ Critical | — | — | ⚠️ 提及 | **Critical** |
| 论文RIG-I描述 | — | ⚠️ 必须 | — | — | **High** |
| Drug测试缺失 | — | — | ⚠️ Major | — | **High** |
| 临床阈值不清 | — | — | — | ⚠️ 必须 | **High** |
| 权重无校准 | ⚠️ Critical | △ 提及 | — | △ 提及 | **High** |

## 分歧点

| 问题 | 方法论 | 其他专家 | 编辑判断 |
|------|--------|----------|----------|
| 修订类型 | Major | Minor | **Minor** (验证代码已存在，可快速补充) |

---

*四位专家独立阐述完成*

*评审日期: 2026-06-01*