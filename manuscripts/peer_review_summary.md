# Confluencia 3.0 同行评审综合报告

**审稿日期**: 2026-06-27  
**目标期刊**: Bioinformatics  
**论文路径**: `D:\IGEM集成方案\manuscripts\confluencia_3.0_research_paper.md`

---

## 一、审稿人组成（5位并行审稿）

| # | 审稿人 | 专业领域 | 关注点 | 状态 |
|---|--------|----------|--------|------|
| 1 | Reviewer1-ComputationalBiologist | RNA结构预测、计算药代动力学 | 方法验证、Benchmark对比 | ⏳ 等待提交 |
| 2 | Reviewer2-CancerResearcher | TNBC分子亚型、免疫治疗 | 生物相关性、临床应用 | ⏳ 等待提交 |
| 3 | Reviewer3-SoftwareArchitect | 平台架构、API设计 | EventBus扩展性、代码质量 | ⏳ 等待提交 |
| 4 | Reviewer4-Immunologist | 先天免疫、circRNA免疫原性 | MDA5/TLR通路、m6A建模 | ✅ 已提交 |
| 5 | Reviewer5-Statistician | 生物统计学 | 样本量、置信区间、多重检验 | ✅ 已提交 |

---

## 二、已收到的审稿意见

### ✅ Reviewer5-Statistician (生物统计学家)

**总体评估**: **Major Revision Required**

**核心观点**: 论文方法学框架合理，但统计验证严重不足且不完整。多个关键验证样本量过小，无法得出可靠结论，置信区间大量缺失。

#### 关键问题（按严重性排序）

**Critical Issues (必须解决)**:

1. **免疫原性验证样本量严重不足 (N=7)**
   - Spearman r=0.91, N=7 → 95% CI ≈ [0.41, 0.99]（极宽区间）
   - 统计检验力仅~40% (α=0.05, 检测r=0.7)
   - 单个异常值可大幅改变相关系数
   - **要求**: 报告精确95% CI，承认为初步验证需独立重复

2. **PK验证样本量不足以区分模型 (N=4)**
   - 12%相对误差 [CI 3-21%] 区间过宽
   - 无法区分好模型与系统偏差
   - **要求**: 重新表述为"模型一致性检查"而非"验证"

3. **置信区间大量缺失 (TBD)**
   - Spearman correlation CI: TBD
   - 结构预测所有指标(RMSD, TM-score, Pair F1): TBD
   - 消融研究所有对比: TBD
   - **要求**: 使用Fisher's z变换或bootstrap补全所有95% CI

4. **无多重检验校正**
   - 结构预测: 4指标×4方法=16个比较 → FWER ~56%
   - 消融研究: 4配置×3指标=12个比较
   - **要求**: 应用Bonferroni (α=0.05/16=0.003) 或 Holm-Bonferroni校正

5. **基线对比未执行**
   - IsRNAcirc对比: TBD
   - AlphaFold3+circularize对比: TBD
   - **要求**: 完成对比并报告效应量（paired tests）

**Major Issues (应该解决)**:

6. **无统计检验力分析** - 未报告每个验证的可检测效应量
7. **无效应量报告** - 仅报告p值，缺少Cohen's d等
8. **无外部验证集特征描述** - 测试集来源/分布未说明
9. **5折交叉验证计划未执行** - 仅提及无结果
10. **旋转增强验证不完整** - 有无增强的对比为TBD

**Minor Issues**:

11. 无敏感性分析
12. 无统计假设检验（正态性、方差齐性）
13. 无缺失数据处理说明

---

### ✅ Reviewer4-Immunologist (免疫学家)

**总体评估**: **Conditional Acceptance pending major revisions**

**核心观点**: 概念创新平台，circRNA特异性免疫建模有真正进步，但定量参数未经实证验证，验证实验统计功效严重受限。

#### 关键问题（按严重性排序）

**Critical Issues**:

1. **m6A抑制值未验证**
   - MDA5 ~90%, TLR ~30%, PKR ~20% 标记为"估计值，未在circRNA系统直接测量"
   - 敏感性分析(±50%)显示2/15序列排名改变 → 中等稳健但不验证绝对值
   - **MDA5 ~90%抑制机制理由错误**: 声称"m6A破坏dsRNA结构"但引用文献(Chen 2019)比较的是生产方法而非m6A修饰
   - **双向m6A建模混淆机制**: 将免疫感应影响与翻译效率影响合并为单一评分
   - **要求**: 
     - 完全移除双向增强权重组件直到有实验数据
     - m6A抑制值报告为范围而非点估计
     - 摘要明确声明为需验证的计算预测

2. **统计功效不足以支撑相关性声称**
   - 主基准 r=0.91 (N=7) → 功效≈0.35
   - Leave-one-out分析证实不稳定 (range 0.79-0.94)
   - 模型比较 ΔAIC=4.5 (N=4) → 功效~0.20
   - **要求**: 将所有相关性声明重新表述为"假设生成观察"而非验证基准

**Major Issues**:

3. **MDA5/TLR通路权重缺乏实证校准**
   - 权重(MDA5 0.35, PKR 0.30, TLR7 0.20, TLR8 0.15)声称"literature-derived"但文献未提供数值权重
   - MDA5评分过度简化：未考虑intron identity（Chen 2019显示这是关键决定因素）
   - **要求**: 使用IFN-β数据集多变量回归校准，或将权重表述为可调超参数进行敏感性分析

4. **TLR引用错误**
   - Line 150引用Gilleron 2013描述TLR生物学 → Gilleron研究LNP递送而非TLR motif
   - **正确引用**: Hemmi et al. J Exp Med 2003 (TLR7 GU-rich motifs); Marquis et al. Eur J Immunol 2014 (TLR8 uridine preference)

5. **生产方法异质性未建模**
   - 模型假设"高度纯化、内含子-free circRNA"但大多数生产方法残留10-30%内含子序列
   - Chen 2019显示杂质而非circRNA本身驱动大部分免疫原性
   - **要求**: 添加"生产质量"参数（内含子污染水平0%/5%/20%）

6. **环状校正因子任意**
   - 0.70校正因子标记为"启发式"且无实验验证
   - 混淆配方效应(LNP包封)与内在RNA序列性质
   - **要求**: 提供敏感性分析全范围结果

**Moderate Issues**:

7. **PKR阈值过度简化**
   - 引用Nallagatla 2007的>33bp阈值，但未考虑PKR激活还需特定结构特征
   - **要求**: 引用Pfaller et al. NAR 2021 (>60bp而非>33bp)

**Minor Issues**:

8. 缺失Bamford et al. Cell 2018引用 (RNase L/RIG-I替代通路)
9. 缺失Abe et al. Nature 2020引用 (MDA5感应circRNA直接证据)

---

### ✅ Reviewer2-CancerResearcher (TNBC癌症生物学专家)

**总体评估**: **Major Revision**

**整体评分**: 2.5-3/5

**核心观点**: 概念方向正确，但验证深度严重不足。几乎所有关键生物学声明都依赖小样本或完全缺乏实验验证。

#### 关键问题（按严重性排序）

**Major Issues (必须解决)**:

1. **Jiang 2019引用过度解读**
   - 声称"consistent with Jiang 2019 clinical observations"
   - **问题**: Jiang 2019是基因组研究，非治疗性临床试验，此声明循环论证
   - **要求**: 改为"consistent with immune microenvironment characterization reported by Jiang 2019"

2. **LAR亚型化疗敏感性矛盾**
   - 模拟显示LAR对多柔比星反应最好（0.36 mm³）
   - **问题**: 与临床经验矛盾（LAR通常化疗耐药）
   - **要求**: 重新审视参数设定并讨论差异原因

3. **统计效力不足未正面处理**
   - N=7免疫原性验证 (power≈0.35)
   - **要求**: 增加显式统计power表格，列出每个数据集的N、检测效果量、power

4. **四基因签名C-index调门过高**
   - C-index 0.52声称"validated"
   - **问题**: 仅略优于随机，不具备临床预测价值
   - **要求**: 如实报告为"marginally better than random"

5. **缺少TNBC亚型分类演变说明**
   - 未提及Lehmann 2016从6种修订为4种
   - **要求**: 在Introduction中解释分类演变，说明为何采用Jiang分类

**Minor Issues**:

6. N=7 IFN-β数据序列特征说明不足
7. GC含量混淆的增量价值讨论不够深入
8. TorusFold >3Å RMSD未置于结构生物学标准上下文
9. 未注明Jiang 2019的中国人群特异性
10. 缺少BLIS的BRCA1突变率/PARP抑制剂敏感性等临床知识

#### 评分明细

| 维度 | 评分 | 备注 |
|------|------|------|
| 生物学相关性 | 3/5 | 概念正确，参数需更严格验证 |
| 临床应用潜力 | 2/5 | 早期假设生成可行，距临床很远 |
| Jiang 2019应用准确性 | 3/5 | 引用正确但过度解读"临床观察" |
| 亚型预测合理性 | 2-3/5 | 定性合理，定量结果存在矛盾 |
| 透明度（限制陈述） | 4/5 | **明显优点** - 论文非常诚实 |
| **整体** | **2.5-3/5** | 概念站得住，需显著更多验证 |

#### 临床应用潜力评估

**最可行场景**:
- 早期筛选工具（预筛选→IsRNAcirc/实验验证）
- 免疫原性相对排名（即使绝对值不准）
- 教学与假设生成

**限制性障碍**:
- 无任何临床验证（无患者数据、PDX数据）
- 免疫原性模型仅N=7，不可靠
- GC含量混淆未解决（r=0.85 vs 通路模型r=0.85，增量信息有限）

---

## 三、跨审稿人共识问题

### 🔴 Critical共识（多位审稿人共同指出）

1. **统计功效严重不足**
   - Reviewer5: N=7/N=4样本量无法得出可靠结论
   - Reviewer4: 统计功效不足以支撑相关性声称
   - **共识**: 将免疫原性和PK验证降级为"假设生成"而非"验证"

2. **参数缺乏实证验证**
   - Reviewer4: m6A抑制值、通路权重均为未验证估计值
   - Reviewer5: 所有TBD值需补充置信区间
   - **共识**: 明确声明哪些参数需实验验证，哪些已验证

### 🟡 Major共识

3. **置信区间缺失**
   - 两份报告均指出大量TBD需补全

4. **基线对比不完整**
   - IsRNAcirc/AlphaFold3对比未执行

---

## 四、修改优先级建议

### 🔴 必须修改（Critical - 不修改则拒稿）

1. **补充所有置信区间**
   - Spearman r 95% CI (使用Fisher's z变换)
   - 所有结构预测指标 (bootstrap 1000次)
   - 消融研究paired differences with CI

2. **应用多重检验校正**
   - 结构预测16个比较 → Bonferroni α=0.003
   - 消融研究12个比较 → Holm-Bonferroni

3. **降级验证声称**
   - 免疫原性r=0.91 (N=7) → "假设生成观察"
   - PK 12%误差 (N=4) → "模型一致性检查"
   - 明确声明需独立重复验证

4. **修正m6A建模**
   - 移除未验证的双向增强权重
   - m6A抑制值报告为范围
   - 明确标注为计算预测需实验验证

### 🟡 应该修改（Major - 修改后可接受）

5. **校准通路权重**
   - 使用IFN-β数据集回归
   - 或报告为可调超参数+敏感性分析

6. **补充正确文献引用**
   - TLR: Hemmi 2003, Marquis 2014
   - PKR: Pfaller 2021
   - MDA5: Abe 2020
   - RNase L: Bamford 2018

7. **添加生产质量参数**
   - 内含子污染水平 (0%/5%/20%)
   - 模型据此权重免疫原性预测

8. **完成基线对比**
   - IsRNAcirc, AlphaFold3+circularize
   - 报告效应量和paired tests

9. **执行交叉验证**
   - 5折CV报告mean±SD
   - 报告跨fold方差评估稳定性

### 🟢 建议修改（Minor - 可选）

10. 敏感性分析
11. 统计假设检验
12. 缺失数据处理说明

---

## 五、审稿裁决预测

基于已收到的2份报告：

- **Reviewer5-Statistician**: Major Revision Required
- **Reviewer4-Immunologist**: Conditional Acceptance (pending major revisions)

**综合预测**: **Major Revision Required**

**理由**: 
- 两位审稿人均指出Critical级别的统计功效和参数验证问题
- 这些问题可通过补充数据分析和重新表述验证声称来解决
- 平台架构创新得到认可，但定量声称需降级

---

## 六、下一步行动

等待收到：
- ⏳ Reviewer1-ComputationalBiologist 报告
- ⏳ Reviewer2-CancerResearcher 报告
- ⏳ Reviewer3-SoftwareArchitect 报告

完成后将生成最终综合评审报告。

---

**报告生成时间**: 2026-06-27 11:30  
**状态**: 部分完成 (2/5审稿人已提交)
