# Bioinformatics Application Note - 四位审稿人并发评审报告

## 稿件信息

**标题:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**评审日期:** 2026-06-01

**评审模式:** 四位审稿人并发独立评审

---

## 审稿人评分汇总

| 审稿人 | 角色 | 评分 | 建议 |
|--------|------|------|------|
| **Reviewer #1** | Methodology & Technical Accuracy | 3.0/5 | Major Revision |
| **Reviewer #2** | Novelty & Application Value | 新颖性 3.0/5, 实用性 4.0/5 | Major Revision |
| **Reviewer #3** | Biological Validity & Immunology | 2.5/5 | Major Revision |
| **Reviewer #4** | Statistical Rigor & Validation | 2.5/5 | Major Revision |

**综合建议: Reject with Invitation to Resubmit**

### Reject 理由

1. **RIG-I 机制对 circRNA 不适用 (Critical)** — 论文核心免疫原性预测方法存在根本性生物学错误
2. **论文-代码不一致** — 权重参数描述与实际实现不同，属学术诚信问题
3. **N=10 零统计效力** — 95% CI [0.47, 0.96] 宽度过大，结论不可靠
4. **临床预测模块无验证** — 声称生存预测但无任何验证数据

### Resubmission 条件

若作者能在 **3-6 个月内** 解决以下问题，欢迎重新投稿：

1. 修正或移除 RIG-I 评分模块 (circRNA 无 5' 端)
2. 统一论文描述与代码实现
3. 扩大验证至 n≥50 并提供完整统计报告
4. 移除或明确标注临床预测为未验证功能

---

## Reviewer #1: Methodology & Technical Accuracy (评分 3.0/5)

### 关键发现

**1. 论文与代码权重参数严重不一致** ⚠️ Critical
- 论文声明: RIG-I=0.35, TLR7=0.25, TLR8=0.20, PKR=0.20
- 代码实际: RIG-I=0.35, TLR7=0.20, TLR8=0.15, PKR=0.30
- 必须统一论文与代码

**2. RIG-I "blunt-end detection" 描述错误**
- circRNA 无 5'/3' 端，经典 blunt-end pathway 不适用
- 代码正确实现了 dsRNA backbone 检测，但论文描述错误

**3. 所有权重参数缺乏定量依据**
- 代码注释承认 "author-informed heuristics, NOT empirically calibrated"
- 需敏感性分析或文献支持

**4. ViennaRNA fallback 准确性未量化**
- fallback 声明 "unknown accuracy"
- 论文未告知读者此局限性

**5. 多个评分公式未完整给出**
- U-rich/GU-rich motif 评分公式缺失
- DRACH 概率估计公式缺失
- 折叠动力学参数未说明

### Major Comments

1. **Method Description Insufficiency:** Methods section (~300 words) is too brief. Algorithmic details for RIG-I/TLR/PKR scoring are not adequately described.

2. **Parameter Justification Gap:** Weights cited as "literature-backed" but no quantitative justification provided. Sensitivity analysis required.

3. **Code Reproducibility Concerns:** Fallback accuracy trade-off not specified.

### Minor Comments

- Line 36: Specify ViennaRNA version tested
- Kinetics prediction heuristics need validation against experimental folding rates
- Pareto front selection algorithm details missing

### 优点

- circRNA 特异性实现准确
- 模块化设计良好
- 文献引用全面
- 透明度高（诚实标注启发式参数）

---

## Reviewer #2: Novelty & Application Value (新颖性 3.0/5, 实用性 4.0/5)

### 关键发现

**1. "First comprehensive platform" 声明过度** ⚠️
- 未引用 circInteractome, CIRCexplorer, NetCircRNA
- 未与 LinearDesign (mRNA疫苗设计) 对比

**2. 验证数据严重不足** ⚠️ Critical
- N=10 circBase 序列验证无法支撑方法论可信度
- 无真实免疫实验数据验证
- 缺乏阴性/阳性对照组

**3. 免疫原性评分权重来源不明确**
- 不同文献使用的实验体系不同
- 权重未经验证或优化
- 缺乏权重敏感性分析

**4. 临床预测模块缺乏验证**
- 无 TCGA 数据验证结果
- 无与现有生存预测工具对比

### 创新类型评估：集成创新为主

| 创新点 | 类型 | 评价 |
|--------|------|------|
| circRNA免疫原性多通路评分 | 方法整合 | 已有单独通路评分文献，整合为平台首创 |
| RIG-I/TLR/PKR联合预测 | 应用创新 | 权重来源于文献，原创算法有限 |
| 进化优化设计 | 算法整合 | Pareto优化为标准方法，REINFORCE应用新颖 |
| circRNA疫苗工作流整合 | 系统创新 | 主要创新点，填补空白 |

### 优点

- 填补 circRNA 疫苗设计工具空白
- 文献支撑充分（Nature等高影响力期刊）
- 工作流完整，从序列到优化
- 开源 MIT 协议，Python API + GUI 双界面
- 性能良好：毫秒级免疫原性评分

---

## Reviewer #3: Biological Validity & Immunology (评分 2.5/5)

### 关键发现

**1. RIG-I 评分机制存在根本性缺陷** ⚠️ Critical
- circRNA 是共价闭环结构，缺乏 5' 端
- RIG-I 核心识别特征是 5'-三磷酸末端
- Schlee et al., 2009 研究针对线性 RNA，直接外推至 circRNA 缺乏机制基础
- 论文未解释 circRNA 如何产生被 RIG-I 识别的 "blunt-end"

**2. TLR7/8 激活模型不完整**
- 内体定位未被考虑（TLR7/8 是内体受体）
- RNA 修饰（假尿苷、m1Ψ 等）对 TLR 激活的影响未纳入
- 二级结构可及性未说明

**3. m6A-免疫原性关联过度简化**
- 声称 "potentially reducing immunogenicity" 过于绝对
- m6A 可增强或抑制免疫反应（YTHDF2 降解 vs YTHDF3 激活）

**4. 临床预测模块缺乏验证**
- Cox 回归模型训练数据来源不明
- IPS/TIDE 用于癌症免疫治疗，是否适用于 circRNA 疫苗未验证
- circRNA 疫苗临床数据匮乏，无法支撑生存预测模型

### Major Comments

1. **RIG-I Scoring Oversimplification:** CircRNAs are covalently closed and lack 5' ends. How does the platform account for circular topology?

2. **TLR7/8 Activation Mechanism:** TLR7/8 recognize ssRNA in endosomes. Scoring does not account for endosomal accessibility, RNA modifications, or secondary structure accessibility.

3. **m6A-Immunogenicity Link:** Claim is not well-supported. m6A can either enhance or suppress immune responses depending on context.

4. **Clinical Prediction Validation:** Cox model training data unspecified. IPS/TIDE applicability to circRNA vaccine recipients (potentially healthy individuals) is questionable.

### 优点

- 模块化 Python API 设计清晰
- ViennaRNA 集成可靠
- Pareto 多目标优化框架合理
- MIT 开源 + Streamlit 界面便于使用

---

## Reviewer #4: Statistical Rigor & Validation (评分 2.5/5)

### 关键发现

**1. 验证样本量严重不足** ⚠️ Critical
- n=10 无统计学意义
- 无法计算置信区间或进行假设检验
- 建议 n≥100

**2. 相关系数无统计检验** ⚠️ Critical
- r=0.85 但无 p 值、无 CI
- n=10 时 95% CI 极宽 [0.45, 0.96]
- 单个异常值可显著改变结果

**3. 均值比较无检验**
- "mean=0.76 vs 0.40" 无 t-test
- 未报告 SD/SE

**4. 无与现有方法的 benchmarking**
- 功能对比表仅有 "✓"，无定量性能指标
- 缺 AUROC/AUPRC/F1 等指标

**5. 无独立测试集验证**
- 无 train/test split
- 无 cross-validation

**6. Cox 回归命名误导**
- 实际是简单线性调整，非 Cox 模型
- 未与临床数据验证

**7. 测试代码过时**
- 引用不存在的 `_detect_blunt_end` 函数
- 测试与代码不同步

### Major Comments

1. **Sample Size Inadequacy:** N=10 is insufficient to establish generalizable performance metrics or support "strong correlation" claims.

2. **Missing Confidence Intervals:** All correlation coefficients and mean comparisons must include 95% CIs and appropriate statistical tests.

3. **Clinical Model Validation Unspecified:** No validation data shown for survival prediction model.

4. **Over-claiming in Results:** r=0.85 with N=10 has CI ~0.5-0.96 - too wide for strong conclusions.

---

## 跨审稿人共识问题

| 问题 | 提及审稿人 | 优先级 |
|------|-----------|--------|
| **N=10 验证样本量不足** | R2, R3, R4 | Critical |
| **RIG-I 评分对 circRNA 环状拓扑的适用性** | R1, R3 | Critical |
| **论文与代码权重参数不一致** | R1 | Critical |
| **缺失统计检验和置信区间** | R4 | Critical |
| **临床预测模型验证缺失** | R2, R3, R4 | High |
| **权重参数缺乏敏感性分析** | R1, R2 | High |
| **"First comprehensive platform" 声明过度** | R2 | High |
| **ViennaRNA fallback 准确性未量化** | R1 | Medium |
| **m6A-免疫原性关联过度简化** | R3 | Medium |

---

## 编辑决定：Major Revision Required

### 必须修改 (Acceptance Criteria)

1. **扩大验证样本至 n≥50** (建议 n≥100)
   - 包含已知高/低免疫原性的 circRNA 作为对照
   - 分层抽样覆盖不同 GC 含量、长度

2. **修正论文-代码不一致**
   - 统一权重参数
   - 论文描述与实际实现匹配

3. **添加统计检验**
   - 所有相关性报告 95% CI 和 p 值
   - 均值比较使用 t-test 或 Mann-Whitney U
   - 报告 SD/SE

4. **解决 RIG-I 机制问题**
   - 明确 circRNA 环状拓扑的处理方式
   - 或承认该模块为探索性功能

5. **修正过度声明**
   - 将 "first comprehensive platform" 改为谨慎表述
   - 明确临床预测模块为未验证功能

### 建议修改

1. 补充工具对比表：添加 LinearDesign, circInteractome, CIRCexplorer
2. 权重敏感性分析
3. 与现有 m6A 预测工具 (SRAMP, m6A-Atlas) 精度对比
4. 展示进化优化收敛曲线
5. 添加完整案例研究

---

## 审稿流程说明

**评审模式:** 四位审稿人并发独立评审

**模型:** Claude Sonnet 4.6

**评审时间:** 2026-06-01

**项目路径:** D:/IGEM集成方案

---

*本报告由 Bioinformatics 四审稿人并发评审系统生成*
