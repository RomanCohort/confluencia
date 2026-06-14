# Bioinformatics Application Note - 四位审稿人独立评审报告

## 稿件信息

**标题:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**类型:** Application Note

**评审日期:** 2026-06-01

**评审模式:** 四位审稿人并发独立评审

---

# Reviewer #1: Methodology & Technical Accuracy

## 我的专业背景

我是专注于计算生物学方法论和技术准确性的审稿人。我的评审重点包括：算法描述的完整性、参数设置的合理性、论文与代码的一致性、以及方法论的可复现性。

---

## 总体评价

- **建议:** Minor Revision
- **质量评分:** 3.5/5

我认可该平台的技术价值，代码实现质量较高，但论文描述与实际实现存在若干不一致之处需要修正。

---

## 核心发现

### 发现1: 权重参数论文-代码严重不一致 ⚠️ Critical

我仔细检查了论文Methods部分和核心代码`immune_sensing.py`，发现权重参数存在明显矛盾：

**论文声明 (Methods, 第34行):**
> RIG-I recognition...weight=0.35; TLR7/8 activation...weights=0.25/0.20; PKR activation...weight=0.20

**代码实际 (`immune_sensing.py`):**
```python
WEIGHTS = {
    "rig_i": 0.35,
    "tlr7": 0.20,   # 论文说0.25
    "tlr8": 0.15,   # 论文说0.20
    "pkr": 0.30,    # 论文说0.20
}
```

更关键的是，代码注释明确声明：
> "weights are author-informed heuristics, NOT empirically calibrated"

这意味着：
1. 论文声称的"literature-backed"具有误导性
2. 论文应明确区分"文献支持的参数"和"启发式估计的参数"
3. 读者有权知道哪些参数经过验证，哪些是猜测

**我的建议:** 论文Methods部分应添加声明：
> "Pathway weights are heuristic estimates informed by literature mechanisms; quantitative values have not been empirically calibrated against experimental data."

---

### 发现2: PKR阈值模块间不一致

论文声称PKR激活基于">33bp threshold (Nallagatla et al., 2007)"，但代码存在两处定义：

- `immune_sensing.py`: `PKR_MIN_DSRNA = 30`
- `structure_prediction.py`: `PKR_MIN_DSRNA_LENGTH = 33`

文献Nallagatla et al., 2007确实支持33bp阈值，代码中30bp的定义没有文献依据。

**我的建议:** 删除`immune_sensing.py`中的重复定义，统一使用`structure_prediction.py`的33bp阈值。

---

### 发现3: RIG-I机制描述需要澄清

论文Methods描述：
> "RIG-I recognition is predicted using blunt-end detection and GU-rich content analysis"

但代码注释明确指出：
```python
# RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing
# (the canonical linear RNA pathway; Schlee et al., 2009)
# RIG-I may be INDIRECTLY activated by circRNA through:
# * dsRNA structures (backbone-forming inverted repeats)
```

这是一个重要的生物学区分：
- **线性RNA**: RIG-I通过识别5'-三磷酸末端激活
- **circRNA**: 共价闭环，无5'端，只能通过dsRNA backbone结构间接激活

代码实现是**正确的**，但论文描述是**误导的**。"blunt-end detection"对circRNA不适用。

**我的建议:** Methods应修改为：
> "RIG-I recognition is predicted via dsRNA backbone structure detection (inverted repeat analysis), as circRNAs lack 5' termini required for canonical blunt-end sensing."

---

### 发现4: Cox回归命名具有误导性

论文声称：
> "Survival analysis uses Cox regression approximation with IPS and TIDE integration"

但`clinical_prediction.py`的实际实现是：
```python
# Simplified exponential
os_1yr = np.exp(-12 / os_median)
os_5yr = np.exp(-60 / os_median)
```

这不是Cox回归，而是基于中位生存时间的简化指数估算。Cox回归需要拟合比例风险模型，包含协变量和偏似然估计。

**我的建议:** 改为更准确的描述：
> "Survival estimation based on IPS and TIDE scores with exponential approximation of median survival time."

---

### 发现5: m6A预测范围限制未披露

代码注释明确：
```python
# NOTE: This is motif-level screening, NOT site-level prediction.
# TODO: crosscheck with m6A-Atlas/RMBase when database is available
```

论文声称"m6A site prediction"，但实际上只是DRACH motif扫描。与SRAMP (AUC 0.90+)、WHISTLE (AUC 0.94)等专业工具相比，精度差距显著。

**我的建议:** 论文应明确说明：
> "m6A prediction is limited to DRACH motif screening; cross-validation with m6A-Atlas is planned for future versions."

---

## 论文-代码一致性检查表

| 功能 | 论文描述 | 代码实现 | 一致性 |
|------|----------|----------|--------|
| RIG-I权重 | 0.35 | 0.35 | ✓ |
| TLR7权重 | 0.25 | 0.20 | ✗ |
| TLR8权重 | 0.20 | 0.15 | ✗ |
| PKR权重 | 0.20 | 0.30 | ✗ |
| PKR阈值 | >33bp | 30/33不一致 | △ |
| m6A方法 | site prediction | motif screening | △ |
| DRACH定义 | 正确 | 正确实现 | ✓ |
| IRES预测 | 提及 | 完整+增强 | ✓ |
| ViennaRNA | 集成 | ✓ + fallback | ✓ |

---

## 我发现的优点

1. **circRNA特异性生物学正确** — 代码正确处理了circRNA无5'/3'端的特性，back-splice junction保护机制也考虑到了

2. **模块化设计良好** — 10个核心模块职责清晰，API设计合理

3. **代码注释详尽** — 包含文献引用和生物学解释，透明度高

4. **Fallback机制完善** — ViennaRNA不可用时有备用方案

5. **IRES预测超越描述** — 实现了G-quadruplex检测、ITAF binding、Kozak context评分，比论文描述更完整

---

## 我认为必须修改的问题

### 必须修改 (Acceptance Criteria)

1. **Methods权重参数** — 与代码一致或明确标注"heuristic estimates"
2. **PKR阈值统一** — 所有模块统一为33bp
3. **RIG-I机制描述** — 明确circRNA特异性通路

### 建议修改

4. 添加Methods Limitations段落
5. 修正"Cox regression"命名
6. 披露m6A预测范围限制
7. 补充BSJ feature extraction模块描述

---

## 我的结论

作为一个技术审稿人，我关注的是**可复现性**和**描述准确性**。该稿件的代码实现质量高，生物学理解正确，但论文描述存在多处与代码不符的情况。这些问题可以通过Minor Revision解决。

我建议作者：
1. 逐一核对论文中所有参数声明与代码是否一致
2. 添加方法局限性章节
3. 使用更精确的术语描述启发式方法

---

# Reviewer #2: Novelty & Application Value

## 我的专业背景

我是专注于评估工具新颖性和应用价值的审稿人。我的评审重点包括：与现有工具的对比、创新点的真实价值、目标用户群体的需求匹配度、以及领域空白填补情况。

---

## 总体评价

- **建议:** Minor Revision
- **新颖性评分:** 3/5
- **实用性评分:** 4/5

我认为Confluencia circRNA的主要价值在于**集成创新**而非算法突破。平台填补了circRNA疫苗设计工具的空白，但单一模块与专业工具相比存在差距。

---

## 核心发现

### 发现1: 平台定位 — 集成创新为主

我系统分析了各创新点的性质：

| 创新点 | 类型 | 新颖程度 |
|--------|------|----------|
| circRNA免疫原性多通路评分 | 方法整合 | **部分新颖** — 通路机制已知，整合为平台首创 |
| circRNA闭环RIG-I机制 | 概念创新 | **新颖** — 正确区分线性RNA通路 |
| TLR7/TLR8分开评分 | 方法改进 | **部分新颖** — motif偏好已知 |
| ViennaRNA + fallback | 工程创新 | **实用** — 非算法创新 |
| Pareto多目标优化 | 方法整合 | **不新颖** — 标准方法 |
| m6A DRACH扫描 | 实现层面 | **不新颖** — SRAMP/WHISTLE更先进 |
| RNACTM六室PK模型 | 架构创新 | **新颖** — circRNA特异性设计 |

**我的判断:** 平台的核心价值是"一站式集成"，而非任何单一模块的算法突破。

---

### 发现2: 与现有工具的详细对比

我研究了相关领域的主要工具：

#### circRNA数据库/注释工具

| 工具 | 功能 | Confluencia对比 |
|------|------|-----------------|
| circInteractome | circRNA-miRNA/RBP互作数据库 | Confluencia提供预测而非查询 |
| CIRCexplorer | circRNA鉴定与注释 | 聚焦identification，无疫苗设计 |
| circBase | circRNA数据库 | 数据存储而非预测工具 |
| NetCircRNA | circRNA功能网络 | 功能网络层面，无免疫原性评分 |

**空白填补:** 确实没有工具整合免疫原性+结构+修饰+临床预测。

#### m6A预测工具

| 工具 | 方法 | AUC | Confluencia |
|------|------|-----|-------------|
| SRAMP | 序列+结构特征 | 0.90+ | 仅DRACH motif |
| WHISTLE | 序列+基因组特征 | 0.94 | 未整合数据库 |
| m6A-Deep | Deep learning | 0.95 | 精度差距显著 |

**差距明显:** Confluencia的m6A预测仅为motif扫描，精度无法与专业工具相比。

#### mRNA疫苗设计工具

| 工具 | 功能 | Confluencia对比 |
|------|------|-----------------|
| LinearDesign (Baidu) | mRNA稳定性+密码子优化 | **算法级创新**，动态规划最优解 |
| RiboTree | 树状mRNA优化 | mRNA专用 |
| OpenVaccine/Eterna | 众包mRNA设计 | 社区驱动，大规模实验验证 |

**关键差异:** LinearDesign使用动态规划找到最优解，Confluencia使用启发式进化搜索，不保证全局最优。

#### 免疫原性预测工具

| 工具 | 领域 | AUC | Confluencia |
|------|------|-----|-------------|
| NetMHCpan | MHC-peptide binding | 0.92-0.96 | 0.80 (差距0.12-0.16) |
| MHCflurry | MHC binding | 0.85-0.90 | 0.83 (接近) |

**性能差距:** 表位预测模块AUC=0.80，与SOTA工具差距明显。

---

### 发现3: 权重校准是关键方法论缺陷

代码注释明确承认：
> "weights are author-informed heuristics, NOT empirically calibrated"

这意味着：
1. 免疫原性评分的定量精度未经验证
2. 论文声称"literature-backed"暗示有文献支持，但实际权重数值无直接文献依据
3. 用户无法判断评分的可靠性

**我的建议:** 平台应定位为"定性筛查工具"而非"定量预测工具"。论文应明确说明：
> "Immunogenicity scores provide qualitative ranking of candidate sequences; quantitative accuracy has not been empirically validated."

---

### 发现4: circBase验证样本量严重不足

论文声称用circBase验证，但仅N=10个序列：
- 无法建立统计显著性
- 无法验证各模块性能
- 仅能作为案例展示

**我的建议:** 扩大验证至至少N=100，或使用独立实验数据集。

---

### 发现5: Comparison Table不完整

论文Table 1存在以下问题：
1. "Linear RNA Tools"应具体指明(LinearDesign等)
2. 未包含m6A专用工具(SRAMP, WHISTLE)
3. 未包含IRES预测工具(IRESite, IRESPred)
4. 仅有"✓"符号，无定量性能指标

**我的建议:** 添加定量指标列(AUC/精度/速度)，并与专业工具对比。

---

## 我对应用价值的评估

### 目标用户群体分析

| 用户类型 | 需求 | 满足程度 |
|----------|------|----------|
| circRNA疫苗研发团队 | 快速筛选候选序列 | **高度满足** — 集成流水线 |
| 学术研究者 | circRNA免疫原性机制探索 | **中度满足** — 定性评分，需实验验证 |
| 生物信息学开发者 | circRNA分析工具整合 | **高度满足** — Python API + 插件系统 |
| 临床转化团队 | PK曲线+剂量优化 | **中度满足** — RNACTM模型新颖但未临床验证 |

### 实际应用场景匹配

| 场景 | 适用性 | 说明 |
|------|--------|------|
| 高通量筛选阶段 | ✓适合 | 定性评分+快速计算 |
| 候选序列优化 | △部分适合 | 进化优化为启发式，非DP最优 |
| 实验前预测 | △需谨慎 | 精度未验证 |
| 临床决策支持 | ✗不适合 | 临床模块为定性整合 |

---

## 我发现的优点

1. **领域定位准确** — 明确针对circRNA疫苗设计，填补空白

2. **生物学机制理解深入** — 正确识别circRNA闭环结构对RIG-I激活的特殊性

3. **集成度高** — 免疫原性+结构+修饰+临床+进化的全流水线

4. **开源完整** — Python API + Streamlit + R + VS Code + CLI + Docker

5. **文档透明** — 明确声明权重为启发式，诚实标注TODO项

6. **可扩展设计** — 插件系统支持自定义

---

## 我的具体建议

### Major Revision要求

1. **完善工具对比表** — 添加定量指标，包含专业m6A/IRES工具
2. **权重校准或声明** — 提供实验验证或明确定位为定性筛查
3. **扩大验证规模** — circBase N>=100

### Minor Revision建议

4. 定位表述为"定性筛查+集成平台"
5. 补充LinearDesign对比
6. m6A精度差距说明
7. IRES预测权重来源解释

---

## 我的结论

Confluencia circRNA是一个**高集成度、低门槛**的circRNA疫苗设计平台。它的主要价值在于：

1. **填补空白** — 首个circRNA疫苗设计一站式平台
2. **实用性强** — 多种接口，开箱即用
3. **可扩展** — 插件系统支持社区迭代

但需要认识到：
1. **单一模块与专业工具有差距**
2. **定量精度未经验证**
3. **定位应是定性筛查**

我建议Minor Revision，要求作者完善工具对比和验证数据。

---

# Reviewer #3: Biological Validity & Immunology

## 我的专业背景

我是专注于免疫学机制验证和生物学准确性的审稿人。我的评审重点包括：免疫传感器激活机制的准确性、circRNA特异性生物学、临床预测的适用性、以及文献引用的正确性。

---

## 总体评价

- **建议:** Minor Revision
- **生物学准确性评分:** 3.5/5

代码实现体现了正确的circRNA生物学理解，但论文描述存在误导性表述。RIG-I机制需要澄清，m6A-免疫原性关系过度简化。

---

## 核心发现

### 发现1: RIG-I机制论文描述与代码实现矛盾 ⚠️ Critical

这是我发现的最重要的生物学问题。

**论文Methods描述 (第34行):**
> "RIG-I recognition is predicted using blunt-end detection and GU-rich content analysis"

**这个描述对circRNA是错误的。**

#### 生物学背景

RIG-I (Retinoic acid-Inducible Gene I) 是一种胞质RNA传感器，其激活机制对RNA拓扑结构有严格要求：

| RNA类型 | RIG-I激活机制 | 文献依据 |
|---------|--------------|----------|
| 线性RNA (5'-ppp) | 识别5'-三磷酸末端 + blunt-end dsRNA | Schlee et al., Nature 2009 |
| 线性RNA (5'-OH) | 弱激活或无激活 | — |
| **circRNA** | **无5'端，不能通过canonical pathway激活** | Zhang et al., Nat Immunol 2016 |

**关键点:** circRNA是共价闭环结构，缺乏5'端和3'端。

#### 代码是正确的

我检查了`immune_sensing.py`和`rig_i_improved.py`，代码注释非常清楚：

```python
# circRNA is a covalently closed loop with NO 5' or 3' ends.
# Therefore:
# - RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing
#   (the canonical linear RNA pathway; Schlee et al., 2009)
#
# - RIG-I may be INDIRECTLY activated by circRNA through:
#   * dsRNA structures (backbone-forming inverted repeats)
#   * Reference: Zhang et al., Nature Immunology 2016
```

代码实现了`_detect_dsRNA_structure()`函数，检测：
- Inverted repeats (回文序列)
- GC含量相关稳定性
- Stem-loop density

**这是正确的circRNA RIG-I激活机制模型。**

#### 问题在于论文描述

论文的"blunt-end detection"描述会让读者误以为circRNA通过canonical 5'-ppp pathway激活RIG-I，这是生物学上不正确的。

**我的建议:** Methods应修改为：

> "RIG-I recognition is predicted via dsRNA backbone structure detection, as circRNAs lack 5' termini required for canonical blunt-end sensing. The algorithm identifies inverted repeats and stem-loop structures that may indirectly activate RIG-I through dsRNA-mediated pathways (Zhang et al., 2016)."

---

### 发现2: PKR阈值代码间不一致

论文声称：
> "PKR activation is predicted from dsRNA length (>33bp threshold)"

文献依据正确：Nallagatla et al., RNA 2007确实表明PKR激活需要>33bp dsRNA。

但代码存在两处定义：

| 文件 | 变量 | 值 |
|------|------|-----|
| `immune_sensing.py` | PKR_MIN_DSRNA | **30** |
| `structure_prediction.py` | PKR_MIN_DSRNA_LENGTH | **33** |

这种不一致会导致：
1. 不同模块计算结果可能不同
2. 论文声称的33bp在某些代码路径中未被执行

**我的建议:** 删除重复定义，统一使用33bp。

---

### 发现3: TLR7/8激活模型不完整

论文描述：
> "TLR7/8 activation scores are computed from U-rich and GU-rich motifs"

这个描述有几个生物学问题：

#### 问题1: 内体定位未被考虑

TLR7和TLR8是**内体受体**(endosomal receptors)，它们的激活需要：
1. RNA进入内体 compartment
2. 酸性环境下TLR构象变化
3. RNA与TLR的LRR结构域结合

**circRNA递送方式影响TLR激活：**

| 递送方式 | 内体进入 | TLR7/8激活预期 |
|----------|----------|----------------|
| LNP包裹 | ✓ 高效 | **强激活** |
| 电转 | ✗ 直接胞质 | **弱激活** |
| 裸RNA注射 | 部分 | 中等 |

论文未讨论此递送依赖性。

#### 问题2: RNA修饰影响未纳入

文献明确表明RNA修饰影响TLR激活：

| 修饰 | TLR激活影响 | 文献 |
|------|-------------|------|
| 假尿苷(Ψ) | 显著降低TLR7/8激活 | Karikó et al., Immunity 2005 |
| m1Ψ | 降低TLR激活 | Anderson et al., NAR 2018 |
| m6A | 可能影响免疫原性 | Liu et al., Nature 2022 |

**代码中TLR评分未考虑修饰状态。**

#### 问题3: 二级结构可及性

TLR7/8识别单链RNA motif，但：
- 二级结构会影响motif可及性
- 高GC含量序列形成稳定结构，掩盖U-rich/GU-rich motif

**我的建议:** Discussion应增加：
> "TLR7/8 activation scores assume endosomal localization typical of LNP delivery. For electroporation or direct injection, TLR pathway activation may be reduced. Future versions will incorporate RNA modification effects (pseudouridine, m6A) on TLR signaling."

---

### 发现4: m6A-免疫原性关系过度简化

论文声称：
> "m6A site prediction with immunogenicity modulation effect"

和Results部分：
> "potentially reducing immunogenicity through modification-mediated immune evasion"

**这个关系比论文描述的复杂得多。**

#### m6A的双重作用

| 效应 | 机制 | 文献 |
|------|------|------|
| **免疫抑制** | YTHDF2介导降解，减少RNA存在时间 | Liu et al., Nature 2022 |
| **免疫激活** | YTHDF3促进翻译，增加抗原表达 | Wang et al., MCB 2019 |
| **翻译增强** | YTHDF1促进ribosome loading | Wang et al., Nature 2015 |

**关键:** m6A效应是**context-dependent**的：
- 细胞类型(YTHDF表达谱不同)
- circRNA翻译效率
- m6A位点位置(IRES区 vs CDS区)

代码`rna_modifications.py`仅考虑了"immune evasion"方向，论文未反映此复杂性。

**我的建议:** 论文应说明：
> "m6A modification has context-dependent effects on immunogenicity. While m6A can promote immune evasion through YTHDF2-mediated degradation, it may also enhance translation through YTHDF1/3 pathways. The current model provides a simplified estimate; detailed modeling of reader protein effects is planned for future versions."

---

### 发现5: 临床预测模块适用性限制

论文声称：
> "Survival analysis uses Cox regression approximation with IPS and TIDE integration"

**适用性存在问题：**

#### IPS/TIDE设计目的

| Score | 设计目的 | 适用人群 |
|-------|----------|----------|
| IPS (Immunotherapy Potential Score) | 预测ICI反应 | **肿瘤患者** |
| TIDE (Tumor Immune Dysfunction) | 预测免疫逃逸 | **肿瘤患者** |

#### circRNA疫苗应用场景

| 场景 | 目标人群 | IPS/TIDE适用性 |
|------|----------|----------------|
| 治疗性疫苗 | 肿瘤患者 | **可能适用** |
| 预防性疫苗 | 健康人群 | **不适用** |

**关键问题:** TCGA训练数据来自肿瘤患者，健康人群的基因表达谱差异显著。

**我的建议:** 论文应明确：
> "Clinical prediction modules are designed for therapeutic vaccine applications in cancer patients. For prophylactic vaccines in healthy populations, these scores may not be directly applicable and require separate validation."

---

## 机制验证总结表

| 通路 | 论文描述 | 代码实现 | 生物学正确性 | 问题 |
|------|----------|----------|--------------|------|
| RIG-I (论文) | blunt-end detection | — | **错误** | circRNA无5'端 |
| RIG-I (代码) | dsRNA backbone | ✓正确 | **正确** | 论文描述需修正 |
| TLR7 | GU-rich motifs | ✓实现 | **部分正确** | 内体定位未考虑 |
| TLR8 | AU-rich motifs | ✓实现 | **部分正确** | 修饰影响未纳入 |
| PKR | dsRNA >33bp | 阈值不一致 | **概念正确** | 代码阈值需统一 |
| m6A | DRACH motif | ✓实现 | **正确** | 免疫原性关系简化 |

---

## 我发现的优点

1. **circRNA特异性RIG-I机制正确实现** — 代码明确区分circRNA与线性RNA的激活通路

2. **TLR7/TLR8分离评分** — 符合文献(TLR7偏好GU-rich, TLR8偏好AU-rich)

3. **ViennaRNA集成可靠** — 使用RNAfold进行真实结构预测

4. **PKR阈值有文献依据** — 33bp正确引用Nallagatla et al., 2007

5. **权重透明声明** — 代码注释明确承认是heuristics

6. **DRACH定义正确** — D=A/G/U, R=A/G, A, C, H=A/C/U

---

## 我认为必须解决的生物学问题

### Critical

1. **修正RIG-I描述** — Methods明确说明circRNA特异性机制

### High

2. **统一PKR阈值** — 所有模块统一为33bp
3. **递送方式影响讨论** — TLR激活依赖内体定位
4. **m6A复杂性承认** — 说明context-dependent

### Medium

5. **临床预测适用性声明** — 区分治疗性/预防性疫苗
6. **修饰影响标注** — Ψ/m6A对TLR的影响

---

## 我的结论

作为免疫学审稿人，我关注的是**机制正确性**和**适用性声明**。

该稿件的代码实现质量高，体现了一流的circRNA生物学理解。特别是RIG-I模块，代码正确区分了circRNA特异性激活机制。问题在于论文描述未能反映这种理解。

我的建议聚焦于：
1. 修正误导性描述
2. 补充机制复杂性讨论
3. 明确适用性边界

这些都是可以通过Minor Revision解决的问题。

---

# Reviewer #4: Statistical Rigor & Validation

## 我的专业背景

我是专注于统计严谨性和验证充分性的审稿人。我的评审重点包括：样本量合理性、统计报告完整性、置信区间和假设检验、模型验证方法、以及性能声称的可信度。

---

## 总体评价

- **建议:** Major Revision
- **统计严谨性评分:** 2/5

该稿件的统计验证严重不足，不符合Bioinformatics期刊Application Note的标准。核心问题包括：样本量过小、缺失置信区间、无统计检验、选择性报告。

---

## 核心发现

### 发现1: circBase验证样本量严重不足 ⚠️ Critical

论文声称：
> "To demonstrate practical utility, we analyzed 10 circRNA sequences from circBase database"

**N=10完全不足以支持任何统计结论。**

#### 为什么N=10不够？

| 分析类型 | 最小推荐样本量 | 原因 |
|----------|----------------|------|
| 相关性分析 | N≥30 | 相关系数需要足够自由度 |
| 均值比较 | N≥15每组 | t检验功效要求 |
| 机器学习验证 | N≥100 | 性能估计稳定性 |

#### N=10时相关系数的不稳定性

我计算了r=0.85 (N=10)的95%置信区间：

```
95% CI: [0.47, 0.96]
```

**置信区间宽度为0.49！** 这意味着真实相关性可能在"中等相关"到"极强相关"之间任何位置。

更严重的是，N=10时单个异常值可以剧烈改变结果：

| 移除一个数据点后 | 可能的r值变化 |
|------------------|----------------|
| 最大值 | r可能降至0.65 |
| 最小值 | r可能升至0.92 |

**我的建议:** 扩大验证至N≥100。我注意到`benchmarks/circbase_large_scale_validation.py`已有5000样本验证代码，应使用这些数据。

---

### 发现2: 相关系数报告不完整 ⚠️ Critical

论文报告：
> "Strong correlation between GC content and overall immunogenicity (r=0.85)"

**缺失的关键信息：**

| 应报告项 | 论文是否报告 |
|----------|--------------|
| 95%置信区间 | ✗ |
| p值 | ✗ |
| Pearson还是Spearman | ✗ |
| 样本量 | ✓ (N=10) |
| 效应量解读 | △ (声称"strong"但无CI支撑) |

#### 规范的报告格式

按照APA和Bioinformatics期刊标准，应报告为：

```
GC content showed a positive correlation with immunogenicity score
(r = 0.85, 95% CI [0.47, 0.96], p = 0.002, N = 10, Pearson).
```

#### 我检查了代码

`benchmarks/circbase_validation.py`中：

```python
correlation = np.corrcoef(gc_contents, immunogenicity_scores)[0, 1]
print(f"Correlation: {correlation:.3f}")
```

仅计算了点估计，**无置信区间，无p值，无统计检验。**

**我的建议:** 使用`scipy.stats.pearsonr`或`scipy.stats.spearmanr`，报告完整统计。

---

### 发现3: 均值比较无统计检验 ⚠️ Critical

论文声称：
> "GC-rich sequences (GC>0.6) showed higher immunogenicity scores (mean=0.76 vs mean=0.40 for moderate GC)"

**完全缺失统计分析：**

| 应报告项 | 论文是否报告 |
|----------|--------------|
| 两组样本量 | ✗ (N_high=?, N_low=?) |
| 标准差/标准误 | ✗ |
| 均值差置信区间 | ✗ |
| t检验或Mann-Whitney | ✗ |
| 效应量(Cohen's d) | ✗ |

#### N=10时可能的分组问题

如果GC>0.6的序列只有2-3个，均值比较完全不可靠。

**正确的分析应该是：**

```python
from scipy.stats import mannwhitneyu, ttest_ind

# 分组
high_gc = scores[gc > 0.6]
low_gc = scores[gc <= 0.6]

# 统计检验
stat, p = mannwhitneyu(high_gc, low_gc)

# 效应量
cohens_d = (np.mean(high_gc) - np.mean(low_gc)) / np.sqrt(
    (np.std(high_gc)**2 + np.std(low_gc)**2) / 2
)

# 报告
print(f"High GC (n={len(high_gc)}): {np.mean(high_gc):.2f} ± {np.std(high_gc):.2f}")
print(f"Low GC (n={len(low_gc)}): {np.mean(low_gc):.2f} ± {np.std(low_gc):.2f}")
print(f"Mann-Whitney U = {stat:.1f}, p = {p:.4f}")
print(f"Cohen's d = {cohens_d:.2f}")
```

**我的建议:** 补充完整统计检验，或删除此比较声明。

---

### 发现4: 文献案例验证相关性极弱但未如实报告

我检查了`benchmarks/consolidated_paper_metrics.json`：

```json
{
  "literature_cases": {
    "n": 17,
    "pearson_r_with_ifn": -0.056,
    "pearson_p_with_ifn": 0.83,
    "direction_accuracy": 0.59
  }
}
```

**关键发现：**
- pearson_r = **-0.056** (负相关！)
- p = **0.83** (完全不显著)
- direction_accuracy = **0.59** (接近随机猜测的0.50)

**论文未报告这些负面结果。**

这意味着免疫原性评分与实验IFN数据之间**无统计显著相关性**。

**我的建议:** 论文必须如实报告：
> "In 17 literature cases with reported IFN induction, predicted immunogenicity scores showed weak correlation with experimental values (Pearson r = -0.06, p = 0.83), indicating limited quantitative accuracy. Direction prediction accuracy was 59%, close to random expectation."

---

### 发现5: Cox回归模型缺乏验证

论文声称：
> "Survival analysis uses Cox regression approximation with IPS and TIDE integration"

**我检查了`clinical_prediction.py`：**

```python
# Simplified exponential (NOT Cox regression!)
os_1yr = np.exp(-12 / os_median)
os_5yr = np.exp(-60 / os_median)
```

这不是Cox回归。这是基于中位生存时间的简化指数估算。

#### 真正的Cox模型需要

| 要素 | 是否存在 |
|------|----------|
| 训练数据集 | ✗ |
| 协变量 | ✗ |
| 偏似然估计 | ✗ |
| HR及其95% CI | ✗ |
| 模型验证(C-index) | ✗ |

**我的建议:**
1. 删除"Cox regression"命名
2. 或提供真正的Cox模型验证数据

---

### 发现6: 缺乏独立测试集验证

所有验证来自内部数据：

| 验证类型 | 数据来源 | 独立性 |
|----------|----------|--------|
| circBase N=10 | 同一数据库 | 不独立 |
| 10-fold CV | 内部交叉验证 | 不独立 |
| 文献案例 | 多个来源 | 部分独立 |

#### 外部验证问题

我检查了NetMHCpan外部验证：

```json
{
  "binding_61_vs_netmhcpan": {
    "r2": -1.60,  // 负值意味着模型比均值预测还差！
    "auc": 0.653  // 仅略高于随机0.5
  }
}
```

**R² = -1.60意味着模型失效。** 这说明表位预测模块在独立数据上表现很差。

**我的建议:** 使用TCGA或其他公开数据集进行独立验证。

---

### 发现7: AUC报告不完整

论文声称：
> "288K IEDB AUC (allele-aware) = 0.80"

但未报告：

| 应报告项 | 是否报告 |
|----------|----------|
| 95% CI | ✗ |
| 阳性/阴性样本比例 | △ (仅报告binder率40.6%) |
| AUPRC | ✗ |
| F1/MCC | △ (部分模块有) |

**我的建议:** 使用DeLong方法计算AUC置信区间：

```python
from scipy.stats import bootstrap

def auc_ci(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    # DeLong method or bootstrap
    ci = bootstrap_auc(y_true, y_pred, n_bootstraps=1000)
    return auc, ci
```

---

## 验证充分性评估表

| 验证项目 | 论文报告 | 充分性 | 问题 |
|----------|----------|--------|------|
| circBase样本量 | N=10 | **不充分** | 需N≥100 |
| 相关系数CI | 无 | **不充分** | 必须报告95% CI |
| 相关系数p值 | 无 | **不充分** | 必须报告 |
| 均值比较检验 | 无 | **不充分** | 需t-test/Mann-Whitney |
| Cox模型验证 | 无 | **不充分** | 需独立生存数据 |
| 性能测试方差 | 无 | **不充分** | 需多次运行 |
| 文献验证相关性 | r=-0.06 | **选择性报告** | 论文未披露 |
| NetMHCpan验证 | R²=-1.60 | **模型失效** | 论文未披露 |
| 大规模验证 | 代码存在 | **未报告** | 应使用N=5000数据 |

---

## 我发现的优点

1. **已有完整统计模块** — `stat_tests.py`包含paired t-test, Wilcoxon, Cohen's d, bootstrap CI

2. **Ablation实验有CV** — 10-fold CV结果稳定(MAE std=0.048)

3. **大规模验证代码已准备好** — `circbase_large_scale_validation.py`可运行N=5000

4. **IEDB验证样本量大** — N=1955有统计意义

5. **Sequence-aware split** — 避免数据泄露

---

## 我认为必须补充的统计分析

### 必须项 (Acceptance Criteria)

1. **扩大circBase验证至N≥100**
   - 使用已有5000样本代码
   - 或从circBase随机采样

2. **报告所有相关系数的95% CI和p值**
   ```python
   from scipy.stats import pearsonr
   r, p = pearsonr(x, y)
   ci = fisher_z_ci(r, n)
   ```

3. **均值比较添加统计检验**
   - t-test或Mann-Whitney U
   - 报告Cohen's d效应量
   - 报告均值差CI

4. **如实报告文献验证结果**
   - pearson_r=-0.056, p=0.83
   - direction_accuracy=0.59

5. **补充独立数据集验证**
   - TCGA验证生存模型
   - 或使用其他公开circRNA数据

### 建议项

6. 报告AUC的95% CI (DeLong方法)
7. 报告ROC曲线图
8. Confusion matrix + CI
9. 多重比较校正(Bonferroni)
10. 功效分析

---

## 我的结论

作为统计学审稿人，我关注的是**可重复性**和**统计严谨性**。

该稿件的算法框架有创新价值，但验证部分的统计报告**严重不足**：

1. **N=10无法支持任何统计结论** — 这是最大的问题
2. **缺失置信区间** — 读者无法评估不确定性
3. **选择性报告** — 负面结果未披露
4. **Cox命名误导** — 实际是简化估算

Bioinformatics期刊对Application Note有明确的统计标准。N=10的验证和缺失CI不符合这些标准。

我建议**Major Revision**，要求作者：
1. 扩大验证样本量
2. 补充完整统计报告
3. 如实披露负面结果

---

## 四位审稿人共识总结

| 问题 | R1 | R2 | R3 | R4 | 共识 |
|------|----|----|----|----|----|
| 论文-代码权重不一致 | ✓ | ✓ | — | — | **Critical** |
| RIG-I描述需修正 | ✓ | — | ✓ | — | **Critical** |
| N=10样本量不足 | — | ✓ | — | ✓ | **Critical** |
| 缺失CI和统计检验 | — | — | — | ✓ | **Critical** |
| PKR阈值不一致 | ✓ | — | ✓ | — | **High** |
| 权重缺乏校准 | ✓ | ✓ | ✓ | — | **High** |
| 临床预测验证缺失 | — | ✓ | ✓ | ✓ | **High** |

---

## 编辑决定

**综合建议: Minor Revision with Statistical Enhancement Required**

四位审稿人一致认可：
- 平台的**集成价值**和**领域定位**
- 代码实现的**技术质量**
- circRNA生物学理解的**准确性**

主要分歧在于统计验证是否充分。建议作者优先解决Critical和High优先级问题。

---

*评审完成时间: 2026-06-01*

*评审模式: 四位审稿人并发独立评审*