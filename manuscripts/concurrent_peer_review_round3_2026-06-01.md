# Confluencia circRNA: 并发审稿汇总报告
## Bioinformatics Application Note - 四位审稿人意见

**审稿日期:** 2026-06-01  
**论文标题:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction  
**审稿模式:** 四位专家并发审稿

---

## 审稿结果概览

| 审稿人 | 角色 | 推荐意见 | 核心关注点 |
|--------|------|----------|------------|
| **Reviewer 1** | 计算方法论 | **Major Revision** | 权重参数缺乏依据、ViennaRNA回退未验证、折叠动力学简化 |
| **Reviewer 2** | 创新性评估 | **Major Revision** | "首个平台"声明过度、验证不足、临床预测缺少circRNA特异性证据 |
| **Reviewer 3** | 生物学验证 | **Major Revision** | 免疫机制需实验验证、circBase样本偏差、m6A预测过于简化 |
| **Reviewer 4** | 统计与实验设计 | **Major Revision** | 样本量严重不足、统计检验完全缺失、性能评估不完整 |

**综合意见: Major Revision (大修)**

---

## Reviewer 1: Methodology Assessment

### Summary (摘要)
The paper presents Confluencia circRNA, a comprehensive computational platform integrating immunogenicity scoring, structure prediction, and evolutionary optimization for circRNA vaccine design. The platform combines multiple existing computational methods into a unified workflow with clinical translation capabilities.

### Major Strengths (主要优点)
- **Comprehensive Integration**: Successfully integrates multiple aspects of circRNA vaccine design (immunogenicity, structure, modifications, clinical prediction) into a single platform, addressing a real gap in the field
- **Literature-Backed Foundation**: Core scoring algorithms reference established literature (Schlee et al., 2009; Nallagatla et al., 2007; Forsbach et al., 2008) for immunogenicity mechanisms
- **Practical Validation**: circBase validation on 10 sequences demonstrates practical utility with biologically interpretable results (GC-immunogenicity correlation r=0.85)

### Major Weaknesses (主要缺陷)
- **Weight Parameters Lack Justification**: The weights (RIG-I: 0.35, TLR7: 0.25, TLR8: 0.20, PKR: 0.20) are presented without empirical or theoretical derivation
- **Folding Kinetics Oversimplified**: The heuristic approach (k = exp(-barrier/RT)) based solely on GC-content and sequence complexity lacks validation against experimental kinetics data
- **ViennaRNA Fallback Accuracy Unquantified**: No benchmark comparing fallback estimation accuracy against actual ViennaRNA predictions

### Specific Comments (具体意见)

**1. 算法理论基础**
The literature citations for immunogenicity pathways are appropriate and foundational. However:
- RIG-I blunt-end detection and GU-rich content analysis are well-supported by Schlee et al. (2009), but the specific scoring formula is not provided in the manuscript
- TLR7/8 U-rich and GU-rich motif scoring references Forsbach et al. (2008), yet the threshold values and scoring functions are not explicitly defined
- PKR dsRNA >33bp threshold correctly cites Nallagatla et al. (2007), but the fractional analysis methodology needs elaboration

**Recommendation:** Provide explicit mathematical formulations for each scoring function in supplementary materials.

**2. 权重参数合理性**
The weight distribution (RIG-I: 0.35, TLR7: 0.25, TLR8: 0.20, PKR: 0.20) raises concerns:
- **No justification provided** for the relative importance assigned to each pathway
- RIG-I receives highest weight (0.35), yet the biological rationale for this prioritization is not discussed
- The weights sum to 1.0, suggesting a linear combination model, but no sensitivity analysis is presented
- Alternative immune pathways (MDA5, OAS, etc.) are not discussed

**Recommendation:** 
- Cite experimental evidence supporting relative pathway contributions
- Perform and report sensitivity analysis for weight variations
- Consider making weights user-configurable with documented defaults

**3. 折叠动力学**
The kinetics prediction has significant limitations:
- Formula k = exp(-barrier/RT) is thermodynamically sound but relies on **estimated** energy barriers from heuristics
- No comparison to kinetic folding simulations (e.g., Kinefold, CoFold)
- Metastable state counting based on "sequence complexity heuristics" is vague
- The relationship between predicted kinetics and experimental circRNA stability is not validated

**Recommendation:**
- Validate kinetics predictions against experimental half-life data
- Provide explicit formulas for barrier estimation from GC-content
- Consider integrating established kinetics tools (CoFold) as optional dependencies

**4. 进化优化算法**
The evolutionary optimization module has both strengths and weaknesses:
- **Strengths:** Pareto multi-objective selection is appropriate for balancing competing objectives (stability, translation, immune evasion)
- **Weaknesses:** 
  - REINFORCE policy learning for operator selection is not well-justified; simpler methods (e.g., adaptive operator selection) may suffice
  - No convergence criteria or iteration limits are specified
  - Mutation operators lack rate parameters
  - No comparison to baseline optimization methods (random search, Bayesian optimization)

**Recommendation:**
- Provide algorithmic details: population size, generations, mutation rates
- Benchmark against simpler optimization approaches
- Document typical convergence behavior

**5. ViennaRNA回退策略**
The fallback estimation approach lacks validation:
- No accuracy metrics comparing fallback to ViennaRNA predictions
- "Simple estimation" methodology is not described
- Performance claim (<50ms fallback vs <1s ViennaRNA) suggests significant approximation
- Risk of misleading users in environments without ViennaRNA

**Recommendation:**
- Quantify fallback accuracy: correlation coefficient, RMSD, structure similarity metrics
- Implement warning system when fallback is used
- Consider requiring ViennaRNA installation for published research use

### Recommendations (具体建议)

**Essential (必须修改):**
1. Provide explicit mathematical formulations for all scoring functions
2. Justify weight parameters with literature evidence or empirical optimization
3. Quantify ViennaRNA fallback accuracy against ground truth predictions
4. Add sensitivity analysis for immunogenicity weights

**Recommended (建议修改):**
1. Validate kinetics predictions against experimental data
2. Compare evolutionary optimization to baseline methods
3. Make weights configurable with documented biological rationale
4. Add uncertainty quantification for all predictions

**Optional (可选修改):**
1. Include additional immune pathways (MDA5, OAS)
2. Add cross-validation with independent circRNA datasets
3. Implement confidence intervals for clinical predictions

### Verdict (推荐意见)
**[x] Major Revision (大修)**

**理由:**
The platform represents a valuable contribution to circRNA vaccine design, addressing an important gap in the field. The integration of multiple computational approaches and the circBase validation demonstrate practical utility. However, **critical methodological details are missing or underspecified**, particularly:
1. Explicit scoring formulas that would allow reproducibility
2. Justification for weight parameters that are central to the immunogenicity prediction
3. Validation of approximation methods (kinetics heuristics, ViennaRNA fallback)

These issues are addressable through revised manuscript text and supplementary materials, but require substantial revision to meet scientific reproducibility standards for *Bioinformatics*. The methodological gaps prevent independent verification of the reported results, which is essential for a computational methods paper.

---

## Reviewer 2: Novelty and Relevance Assessment

### Summary (摘要)
The manuscript presents Confluencia circRNA, a platform integrating immunogenicity prediction, structure analysis, modification mapping, and clinical outcome prediction for circRNA vaccine design—claiming to be the first comprehensive platform in this space.

### Major Strengths (主要优点)
- **Targeted gap identification**: The paper correctly identifies that existing tools (ViennaRNA, LinearDesign) focus on linear mRNA, while circRNA databases (circBase, CircInteractome) are annotation-focused, leaving a genuine tool gap for circRNA vaccine design
- **Multi-pathway immunogenicity scoring**: Literature-backed scoring for RIG-I, TLR7/8, and PKR with weighted contributions addresses a real need—these pathways are indeed critical for vaccine immunogenicity
- **Clinical translation module**: Integration of IPS, TIDE, and survival prediction represents a novel attempt to bridge computational prediction with clinical outcomes for circRNA therapeutics

### Major Weaknesses (主要缺陷)
- **"First comprehensive platform" claim is overstated**: circRNA vaccine design tools exist—e.g., circVaccine (bioRxiv 2023), CIRCulator, and the recent CircularDesign framework. The claim should be refined to specify "first to integrate X+Y+Z" rather than "first comprehensive platform"
- **Validation is extremely weak**: Only 10 sequences from circBase with no experimental validation, no comparison to known immunogenic/non-immunogenic circRNA controls, and no benchmark against existing vaccine design methods. This undermines the utility claims
- **Clinical prediction lacks circRNA-specific validation**: IPS and TIDE were developed for mRNA/mAb applications. Applying these to circRNA without circRNA-specific validation data or mechanistic justification is speculative

### Specific Comments (具体意见)

1. **创新性声明**: 
   - The claim "No comprehensive platform exists that integrates circRNA-specific immunogenicity prediction" is too absolute. Recent publications describe similar integrations (e.g., Liu et al., NAR 2023 on circRNA design). Recommend revising to: "We present one of the first platforms to systematically integrate..."
   - The novelty lies more in the **combination** than individual components. This should be emphasized differently—position as a workflow integration contribution rather than novel algorithm development.

2. **与现有工具比较**: 
   - Table comparison is **incomplete and potentially misleading**:
     - "Linear RNA Tools" is vague—specify which tools (LinearDesign, mRNA vaccine tools?)
     - circRNA databases like circInteractome, starBase DO provide miRNA/RBP binding annotation—marking "✓ (annotation)" understates their capabilities
     - ViennaRNA DOES provide some kinetics analysis (barrier trees via RNAkinetics, RNAsubopt)
     - Missing comparison to: CIRCexplorer2, CIRIquant, CircPro (circRNA translation prediction)
   - Recommend adding specific tool names and a more balanced comparison

3. **集成创新价值**: 
   - The integration is valuable but the paper doesn't quantify the benefit. Does integration improve prediction accuracy? Does the workflow save time compared to using tools separately?
   - The REINFORCE policy learning for sequence evolution is interesting but lacks detail on training data and performance
   - Pareto optimization is standard in multi-objective design—what's the specific innovation here?

4. **临床预测新颖性**: 
   - This is the most novel component but also the **weakest validated**. Cox regression approximation without circRNA-specific training data raises concerns about reliability
   - The citation "Cristescu et al., 2018; Jiang et al., 2018" references mRNA/mAb immunotherapy studies—no evidence these apply to circRNA
   - Recommend: Either remove clinical prediction claims to core scope, or provide circRNA-specific validation data

5. **表格准确性**: 
   - Missing critical competitors: circRNA prediction tools like CIRCexplorer, find_circ, CIRI
   - "Immunogenicity scoring" row is misleading—ViennaRNA doesn't claim this, but tools like mRNA-LNP immunogenicity predictors (e.g., Immuno-mRNA) do exist and could be adapted
   - "Clinical prediction" row: no validation provided that this module works; claiming "✓" while competitors show "-" is unfair without evidence

### Recommendations (具体建议)

1. **Revise novelty claims**: Change "No comprehensive platform exists" to "We present an integrated platform addressing the gap in circRNA vaccine design tools"

2. **Strengthen validation**:
   - Add comparison to known immunogenic circRNAs (e.g., those from vaccine studies)
   - Include correlation analysis with experimental immunogenicity data if available
   - Benchmark structure prediction against ViennaRNA directly to show added value

3. **Expand comparison table**:
   - Add specific tool names: LinearDesign, mRNA vaccine tools (Moderna/pfizer design frameworks)
   - Add circRNA-specific tools: CIRCexplorer, CircPro, CIRIquant
   - Be more honest about what competitors do provide

4. **Tone down clinical prediction claims**: Either validate on circRNA clinical data or reframe as "exploratory clinical translation module"

5. **Quantify integration benefits**: Compare end-to-end workflow time/accuracy vs. using tools separately

### Verdict (推荐意见)
**[x] Major Revision (大修)**

**理由:** The platform addresses a real gap and the integration is valuable, but novelty claims are overstated, validation is insufficient, and the clinical prediction module lacks circRNA-specific evidence. A major revision should: (1) temper claims appropriately, (2) add proper competitive analysis, (3) strengthen or remove clinical prediction claims, and (4) provide quantitative validation of the integrated workflow's benefits over standalone tool usage.

---

## Reviewer 3: Biological Validation Assessment

### Summary (摘要)
本文介绍了Confluencia circRNA平台，一个整合circRNA免疫原性预测、结构分析和序列优化的开源计算工具，为circRNA疫苗设计提供了有用的计算框架。

### Major Strengths (主要优点)
- **多通路免疫原性评分整合**：基于文献权重整合RIG-I、TLR7/8、PKR三通路，权重设置有生物学依据（RIG-I 0.35, TLR7 0.25, TLR8 0.20, PKR 0.20），符合目前对circRNA免疫原性机制的理解
- **结构-功能关联设计**：将dsRNA区域检测与PKR激活关联、将GC含量与免疫原性关联，体现了合理的生物学逻辑
- **开放源代码与多接口设计**：Python API + Streamlit + Electron多种接口便于不同用户群体使用

### Major Weaknesses (主要缺陷)
- **免疫原性评分缺乏实验验证**：所有评分权重来源于文献推断，缺乏circRNA特异性实验数据验证。RIG-I识别5'三磷酸末端主要针对病毒RNA，而circRNA因环化无游离末端，直接应用此机制需更多证据支持
- **circBase验证样本选择偏差**：仅使用10个序列验证，且GC含量范围(0.50-1.00)不包含低GC序列，无法代表生理circRNA多样性。参考文献中Du et al. 2016、Hansen et al. 2013、Zheng et al. 2016并非circRNA免疫原性研究，而是circRNA功能和miRNA海绵研究
- **m6A预测方法过于简化**：仅使用DRACH motif检测，未考虑组织特异性m6A writer（METTL3/14, WTAP）分布和circRNA特有的反向剪接位点m6A富集模式

### Specific Comments (具体意见)

1. **免疫原性机制**
   - **RIG-I通路**：RIG-I主要识别5'三磷酸和双链RNA blunt ends。circRNA因环化结构理论上无游离5'端，其RIG-I激活机制应更侧重于dsRNA茎环结构。建议：明确讨论circRNA环化对RIG-I识别的影响，或引用circRNA特异性RIG-I研究（如Li et al., 2017 Cell Host Microbe）
   - **TLR7/8通路**：TLR7/8位于内体，识别ssRNA。circRNA作为疫苗载体时主要通过脂质纳米颗粒递送，是否进入内体通路需讨论。建议：增加递送系统对免疫原性的影响讨论
   - **PKR通路**：33bp dsRNA阈值合理，但circRNA茎环结构可能形成更短的双链区域，建议讨论hairpin stem长度对PKR激活的影响

2. **circBase验证**
   - 样本量(n=10)不足以得出统计结论，r=0.85的相关性可能受样本量影响
   - GC含量范围(0.50-1.00)偏高，实际circRNA GC含量多为0.40-0.60，高GC序列代表性不足
   - 建议：(1) 增加验证序列至30+，(2) 包含不同来源circRNA（外显子circRNA、内含子circRNA、融合circRNA），(3) 与实验验证的circRNA免疫原性数据对比

3. **修饰位点预测**
   - **m6A**：DRACH motif是必要非充分条件。建议增加：(1) 剪接位点附近的m6A富集特征，(2) RNA二级结构对m6A可及性的影响（单链DRACH更易被甲基化），(3) circRNA特有反向剪接位点的m6A模式
   - **IRES**：IRES活性预测仅用polypyrimidine tract，过于简化。建议整合已知IRES数据库和结构特征
   - **miRNA结合位点**：15+种子序列列表未明确说明选择标准，且未区分种子匹配类型（6mer, 7mer-m8, 7mer-A1, 8mer）

4. **临床预测可靠性**
   - Cox回归使用IPS和TIDE参数，但这些参数来源于mRNA癌症免疫治疗研究（Cristescu 2018, Jiang 2018），直接应用于circRNA需谨慎
   - circRNA作为疫苗载体与癌症免疫治疗的临床情境不同，建议区分：(1) 预防性疫苗（传染病），(2) 治疗性癌症疫苗，(3) 递送载体应用
   - 缺乏实际临床circRNA疫苗数据验证（目前无FDA批准的circRNA疫苗）

5. **疫苗设计应用**
   - 实际价值：平台对初步筛选候选序列有参考价值，但距离临床应用差距较大
   - 建议：(1) 增加抗原表位预测模块（B细胞表位、T细胞表位），(2) 整合递送系统参数（LNP成分对免疫原性影响），(3) 增加蛋白质表达预测（翻译效率评估）

### Recommendations (具体建议)

1. **增加免疫原性实验验证**：至少对少量代表序列进行体外验证（HEK293T IFN-β reporter assay, THP-1细胞因子释放检测）
2. **扩展circBase验证**：增加至30+序列，覆盖低GC(<0.45)、中GC(0.45-0.55)、高GC(>0.55)范围，并区分circRNA类型
3. **完善m6A预测**：整合m6A-Atlas数据库的circRNA m6A数据，增加结构可及性评估
4. **明确临床应用边界**：区分疫苗开发（需高免疫原性）和递送载体（需低免疫原性）两种场景，分别提供优化策略
5. **增加局限性讨论**：诚实讨论当前预测模型的局限性，包括：(1) 缺乏实验验证，(2) 权重参数需更多数据校准，(3) 递送系统影响未纳入

### Verdict (推荐意见)
**[x] Major Revision (大修)**

**理由:** 本文提出的平台填补了circRNA疫苗设计工具的空白，方法学框架合理，代码实现完整。但作为Bioinformatics应用笔记，关键缺陷在于：**(1) 免疫原性评分缺乏实验验证**，所有参数来源于文献推断而非circRNA特异性数据；**(2) circBase验证样本不足且选择性偏差**，无法支撑结论的可靠性；**(3) 修饰位点预测方法过于简化**，未考虑circRNA特有的结构特征。建议大修后重新审稿，重点补充实验验证数据和扩展验证样本量。

---

## Reviewer 4: Statistical and Experimental Design Assessment

### Summary (摘要)
本文介绍了Confluencia circRNA平台，用于circRNA疫苗设计及免疫原性预测，整合了文献支持的评分系统、结构预测和序列优化功能。

### Major Strengths (主要优点)
- 提供了模块化的Python API和Streamlit前端，可用性强
- 整合了文献支持的权重系统（RIG-I, TLR7/8, PKR），方法论相对透明
- 与现有工具进行了较为详细的比较分析

### Major Weaknesses (主要缺陷)
- **样本量严重不足**：仅使用10个circRNA序列进行验证，无法支持广泛的结论推广
- **统计检验完全缺失**：相关性分析和均值比较均未提供p值或置信区间
- **性能评估不完整**：时间测量缺乏样本量、标准差和置信区间

### Specific Comments (具体意见)

**1. 样本量充分性**

当前验证仅使用10个circRNA序列，这在统计学上存在严重问题：
- **功效分析缺失**：未提供任何样本量计算依据
- **代表性不足**：10个样本难以代表circRNA序列的多样性
- **子组分析不可靠**：分组后各组样本量更小，统计功效降低
- **建议最小样本量**：相关性分析至少n≥30；临床预测验证建议n≥50-100

**2. 相关性分析**

r=0.85的GC含量与免疫原性相关性存在以下问题：
- **p值缺失**：未报告统计显著性检验
- **置信区间缺失**：n=10时95% CI很宽（约0.49-0.96），估计精度极低
- **小样本偏倚**：单个异常值可能严重影响相关系数
- **建议**：补充回归分析，报告R²、回归系数β及标准误

**3. 性能评估**

时间测量问题：

| 测量项 | 报告值 | 问题 |
|--------|--------|------|
| 免疫原性评分 | <100ms | 无样本量、无SD、无CI |
| 结构预测 | <1s / <50ms | 两种方法未进行统计比较 |
| 完整流程 | 2-3s | 仅报告范围，缺乏分布信息 |

建议：报告mean±SD或median(IQR)、95% CI、明确测试样本量

**4. 统计检验缺失**

以下比较均缺乏统计检验：
- "mean=0.76 vs mean=0.40" - 需t检验或Mann-Whitney U检验
- "mean=18.5 vs 0" - 需统计检验确认差异显著性
- 所有均值比较缺少：SD/SE、p值、效应量、置信区间

**5. 结论支持强度**

| 结论 | 统计支持强度 |
|------|--------------|
| GC含量与免疫原性正相关 | 弱 (n=10, 无CI) |
| 优化序列可作为疫苗候选 | 极弱 (无对照) |
| 临床预测模块有效性 | 无 (未验证) |

### Recommendations (具体建议)

1. **扩大验证集**：至少增加至n≥50个circRNA序列
2. **完整统计报告**：r值+p值+95% CI；mean±SD+检验统计量+p值+效应量
3. **性能测试标准化**：n≥20次重复，报告mean±SD，提供检验比较
4. **补充验证图表**：散点图+回归线、ROC曲线等
5. **方法学补充**：功效分析、多重比较校正

### Verdict (推荐意见)
**[x] Major Revision (大修)**

**理由:** 论文提出了有价值的综合平台，但存在严重统计学缺陷：(1)验证样本量不足(n=10)；(2)所有统计比较缺少检验、p值和置信区间；(3)性能评估缺乏完整统计描述。建议扩大验证集至≥50样本，添加完整统计分析后重新提交。

---

## 编辑综合意见

### 共识问题（四位审稿人一致指出）

1. **验证样本量严重不足** (n=10)
   - R1: "circBase validation on 10 sequences"
   - R2: "Only 10 sequences from circBase"
   - R3: "仅使用10个序列验证"
   - R4: "样本量严重不足"

2. **权重参数缺乏依据**
   - R1: "Weight Parameters Lack Justification"
   - R3: "所有评分权重来源于文献推断，缺乏circRNA特异性实验数据验证"

3. **创新性声明过度**
   - R2: "First comprehensive platform claim is overstated"

4. **临床预测模块验证不足**
   - R2: "IPS and TIDE were developed for mRNA/mAb applications"
   - R3: "直接应用于circRNA需谨慎"
   - R4: "临床预测模块有效性：无（未验证）"

### 必须修改事项（Major Revision必需）

1. **扩展验证数据集**
   - 最小要求：n≥30个序列（R4建议n≥50）
   - 覆盖不同GC含量范围（R3）
   - 包含不同circRNA类型（R3）

2. **补充统计检验**
   - 相关性分析：r值 + p值 + 95% CI
   - 均值比较：mean ± SD + 检验统计量 + p值 + 效应量
   - 性能测试：n≥20次重复，报告mean±SD

3. **明确算法细节**
   - 提供显式评分公式（R1）
   - 权重参数敏感性分析（R1）
   - ViennaRNA回退精度验证（R1）

4. **修正创新性声明**
   - 改为"one of the first platforms"（R2）
   - 扩展比较表格，添加具体工具名称（R2）

5. **增加局限性讨论**
   - 缺乏实验验证（R3）
   - 临床预测的circRNA特异性问题（R2, R3）
   - 递送系统影响未纳入（R3）

### 可选改进建议

- 增加体外实验验证（R3）
- 整合m6A-Atlas数据库（R3）
- 区分疫苗开发与递送载体场景（R3）
- 添加抗原表位预测模块（R3）

---

## 最终决定

**编辑决定: Major Revision (大修)**

四位审稿人一致建议大修。论文提出了有价值的综合平台，方法学框架合理，代码实现完整。但存在以下关键缺陷需要重大修订：

1. **方法论透明度不足**：评分公式、权重依据、回退精度缺失
2. **验证严重不足**：样本量n=10、无统计检验、无实验验证
3. **创新性声明过度**：需修正"首个综合平台"的表述
4. **临床预测模块验证不足**：IPS/TIDE来源于mRNA研究，缺少circRNA特异性证据

建议作者逐一回应四位审稿人的意见，补充必要的数据和分析后重新提交。

---

**审稿完成时间:** 2026-06-01  
**审稿人:** Reviewer 1 (Methodology), Reviewer 2 (Novelty), Reviewer 3 (Biology), Reviewer 4 (Statistics)
