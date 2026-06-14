# Bioinformatics Journal - Editorial Decision Letter

## Manuscript Information

**Manuscript ID:** BIOINF-2026-XXXX

**Title:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Type:** Application Note

**Submission Date:** 2026-05-XX

**Decision Date:** 2026-06-01

---

## Editorial Decision

**Decision: MINOR REVISION WITH STATISTICAL ENHANCEMENT REQUIRED**

After careful consideration of four independent peer review reports, I have decided to invite the authors to submit a revised version of their manuscript addressing the concerns raised by the reviewers.

---

## Dear Authors,

Thank you for submitting your Application Note to *Bioinformatics*. I have received detailed reviews from four independent experts in the following areas:

- **Reviewer #1:** Methodology & Technical Accuracy
- **Reviewer #2:** Novelty & Application Value
- **Reviewer #3:** Biological Validity & Immunology
- **Reviewer #4:** Statistical Rigor & Validation

All four reviewers recognize the potential value of Confluencia circRNA as an integrated platform for circRNA vaccine design. However, they have identified several issues that must be addressed before the manuscript can be accepted for publication.

---

## Editorial Summary

### What the Reviewers Appreciate

I want to first highlight what the reviewers found valuable in your work:

**1. Integration Value (Consensus from R2, R3)**

> "Confluencia circRNA填补了circRNA疫苗设计一站式平台的空白。现有工具均为单一功能，您的平台整合了免疫原性+结构+修饰+临床预测+序列优化，这是真正的领域贡献。"

**2. Technical Implementation Quality (Consensus from R1, R3)**

> "代码实现质量高，模块化设计清晰，API设计合理。特别是circRNA特异性生物学理解正确——RIG-I模块正确区分了circRNA无5'端的特性，通过dsRNA backbone机制而非canonical blunt-end pathway。"

**3. Open-Source Contribution (Consensus from R1, R2)**

> "MIT开源协议，Python API + Streamlit + R + VS Code + CLI + Docker多平台覆盖，降低了使用门槛。插件系统支持社区扩展，这是值得称赞的设计。"

**4. Transparent Documentation (R1, R2)**

> "代码注释详尽，明确承认权重为'heuristics而非calibrated values'，诚实标注TODO项，这种透明度在学术软件中是良好实践。"

---

### What Needs to Be Addressed

The reviewers have raised concerns in four categories. I have organized them by priority:

---

## Category A: Critical Issues (Must Be Fixed)

### A1. Manuscript-Code Parameter Inconsistency

**Raised by:** R1 (Methodology), R2 (Novelty)

**Severity:** Critical

**The Problem:**

Your Methods section states:
> "RIG-I recognition...weight=0.35; TLR7/8 activation...weights=0.25/0.20; PKR activation...weight=0.20"

But your code (`immune_sensing.py`) implements:
> RIG-I=0.35, TLR7=0.20, TLR8=0.15, PKR=0.30

**Reviewer #1's exact words:**
> "这是一个关键矛盾。论文声称的'literature-backed'具有误导性。代码注释明确声明权重是'author-informed heuristics, NOT empirically calibrated'。"

**What You Must Do:**

1. **Either:** Update the Methods section to match the code exactly
2. **Or:** Add an explicit statement:
   > "Pathway weights (RIG-I=0.35, TLR7=0.20, TLR8=0.15, PKR=0.30) are heuristic estimates informed by literature mechanisms. Quantitative values have not been empirically calibrated against experimental data."

**My Editorial Note:** This is a matter of scientific integrity. Parameters must be accurately reported, and their derivation must be transparently stated.

---

### A2. RIG-I Mechanism Description Misleading

**Raised by:** R1 (Methodology), R3 (Immunology)

**Severity:** Critical

**The Problem:**

Your Methods section states:
> "RIG-I recognition is predicted using blunt-end detection and GU-rich content analysis"

**Reviewer #3's exact words:**
> "这个描述对circRNA是错误的。RIG-I识别5'-三磷酸末端+blunt-end dsRNA。circRNA是共价闭环，缺乏5'端。论文的'blunt-end detection'会让读者误以为circRNA通过canonical 5'-ppp pathway激活RIG-I。"

**However, your code is correct:**

Reviewer #3 examined your `immune_sensing.py` and found:
```python
# circRNA is a covalently closed loop with NO 5' or 3' ends.
# RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing.
# RIG-I may be INDIRECTLY activated by circRNA through dsRNA structures.
```

Reviewer #3 noted: "代码实现是正确的，但论文描述是误导的。"

**What You Must Do:**

Rewrite the Methods section to accurately describe your circRNA-specific implementation:

> "RIG-I recognition is predicted via dsRNA backbone structure detection, as circRNAs lack 5' termini required for canonical blunt-end sensing. The algorithm identifies inverted repeats and stem-loop structures that may indirectly activate RIG-I through dsRNA-mediated pathways (Zhang et al., Nat Immunol 2016)."

---

### A3. Validation Sample Size Insufficient

**Raised by:** R2 (Novelty), R4 (Statistics)

**Severity:** Critical

**The Problem:**

You report validation on N=10 circRNA sequences from circBase.

**Reviewer #4's exact words:**
> "N=10完全不足以支持任何统计结论。对于相关性分析，最小推荐样本量N≥30。N=10时r=0.85的95% CI为[0.47, 0.96]，置信区间宽度为0.49——真实相关性可能在'中等相关'到'极强相关'之间任何位置。单个异常值可剧烈改变结果。"

Reviewer #4 also noted that you have `circbase_large_scale_validation.py` with N=5000 capability, but did not use or report it.

**What You Must Do:**

1. Expand circBase validation to at least **N≥100 sequences**
   - Use stratified sampling across GC content and length
   - Include sequences with known high/low immunogenicity as controls

2. For each reported metric, provide:
   - Sample size (N)
   - 95% confidence interval
   - p-value (where applicable)

3. If N=100 validation is not feasible within revision timeline:
   - Acknowledge the limitation: "Current validation (N=10) provides preliminary case studies; larger-scale validation (N≥100) is planned for future versions."
   - BUT: This will weaken your claims significantly. I strongly recommend expanding the validation.

---

### A4. Missing Confidence Intervals and Statistical Tests

**Raised by:** R4 (Statistics)

**Severity:** Critical

**The Problem:**

Reviewer #4 identified multiple instances of incomplete statistical reporting:

| Claim in Manuscript | What's Missing |
|---------------------|----------------|
| "r=0.85" (GC-immunogenicity) | 95% CI, p-value, Pearson vs Spearman |
| "mean=0.76 vs mean=0.40" | t-test/Mann-Whitney, CI, Cohen's d, group sizes |
| "AUC=0.80" (IEDB) | 95% CI via DeLong method |
| Performance timings | Variance (SD), hardware specs |

**Reviewer #4's exact words:**
> "所有相关系数和均值比较必须包含95%置信区间和适当的统计检验。Bioinformatics期刊对Application Note有明确的统计标准。缺失CI不符合这些标准。"

**What You Must Do:**

Provide complete statistical reporting using this format:

```
Correlation: r = X.XX (95% CI [X.XX, X.XX], p = X.XXX, N = XX, Pearson/Spearman)

Group comparison: Group A (n=X): mean ± SD; Group B (n=Y): mean ± SD;
                  Statistical test: t/U = X.XX, p = X.XX, Cohen's d = X.XX

AUC: AUC = X.XX (95% CI [X.XX, X.XX], N = XX)
```

**My Editorial Note:** Your manuscript already cites that you use bootstrap CI in other contexts (ablation experiments). This suggests you have the statistical infrastructure. Please apply it consistently throughout.

---

## Category B: High Priority Issues (Should Be Fixed)

### B1. PKR Threshold Inconsistency Between Modules

**Raised by:** R1, R3

**The Problem:**

Your manuscript cites ">33bp threshold (Nallagatla et al., 2007)" for PKR activation.

But your code has two definitions:
- `immune_sensing.py`: `PKR_MIN_DSRNA = 30`
- `structure_prediction.py`: `PKR_MIN_DSRNA_LENGTH = 33`

**What You Must Do:**

Unify to 33bp (the literature-supported value) across all modules, and remove duplicate definitions.

---

### B2. Weight Parameters Lack Empirical Calibration

**Raised by:** R1, R2, R3

**The Problem:**

Your manuscript implies "literature-backed" weights, but all three reviewers noted the code explicitly states these are "heuristics, NOT empirically calibrated."

**Reviewer #2's exact words:**
> "论文声称'literature-backed'暗示有文献支持，但实际权重数值无直接文献依据。用户无法判断评分的可靠性。"

**What You Must Do:**

Add a Methods section clarification:
> "Immunogenicity pathway weights were determined based on relative importance inferred from literature mechanisms, but have not been empirically calibrated. Users should interpret scores as qualitative rankings rather than precise quantitative predictions."

Optionally, provide a sensitivity analysis showing how varying weights affects predictions.

---

### B3. Clinical Prediction Module Validation Missing

**Raised by:** R2, R3, R4

**The Problem:**

You claim "survival analysis uses Cox regression approximation" but:
- No validation data is shown
- Reviewer #4 found it's actually simplified exponential estimation, not Cox regression
- Reviewer #3 noted IPS/TIDE scores were developed for cancer immunotherapy, not circRNA vaccines

**Reviewer #4's exact words:**
> "这不是Cox回归。这是简化指数估算。Cox回归需要拟合比例风险模型，包含协变量和偏似然估计。"

**What You Must Do:**

Either:
1. Remove or downgrade the clinical prediction claims to "exploratory feature"
2. Provide validation data (training cohort, C-index, HR with CI)
3. Change naming from "Cox regression" to "survival estimation based on IPS/TIDE"

---

### B4. Negative Validation Results Not Disclosed

**Raised by:** R4

**Severity:** This is a transparency issue

**The Problem:**

Reviewer #4 examined your benchmark data files and found:

```json
"literature_cases": {
  "pearson_r_with_ifn": -0.056,  // Negative correlation!
  "pearson_p_with_ifn": 0.83,    // Not significant
  "direction_accuracy": 0.59     // Near random (0.50)
}

"binding_61_vs_netmhcpan": {
  "r2": -1.60,  // Model worse than mean prediction!
  "auc": 0.653
}
```

These results indicate:
- Immunogenicity scores show **no correlation** with experimental IFN data (r=-0.06, p=0.83)
- Direction prediction is **near random** (59% vs 50% baseline)
- External validation against NetMHCpan shows **model failure** (R²=-1.60)

**Reviewer #4's exact words:**
> "论文未如实报告这些负面结果。这意味着免疫原性评分与实验IFN数据之间无统计显著相关性。"

**What You Must Do:**

You must disclose these results, even if unfavorable. Scientific integrity requires transparent reporting of all validation outcomes.

Suggested wording:
> "In 17 literature cases with reported IFN induction, predicted immunogenicity scores showed weak correlation with experimental values (Pearson r = -0.06, p = 0.83). Direction prediction accuracy was 59%, indicating limited quantitative accuracy for the current heuristic model. Future versions will incorporate empirical weight calibration."

---

## Category C: Medium Priority Issues (Recommended)

### C1. Tool Comparison Table Incomplete

**Raised by:** R2

Add specific tool names and quantitative comparisons:
- m6A tools: SRAMP (AUC 0.90), WHISTLE (AUC 0.94)
- mRNA tools: LinearDesign (dynamic programming optimization)
- IRES tools: IRESite, IRESPred

---

### C2. m6A-Immunogenicity Relationship Oversimplified

**Raised by:** R2, R3

Acknowledge that m6A effects are context-dependent (YTHDF2 degradation vs YTHDF1/3 translation enhancement).

---

### C3. TLR7/8 Delivery Method Dependency

**Raised by:** R3

Note that TLR activation depends on delivery method (LNP → endosomal vs electroporation → cytosolic).

---

### C4. Cox Regression Naming

**Raised by:** R1, R4

Change to "survival estimation" if no true Cox model is implemented.

---

## Editorial Timeline

**Revision Deadline:** 90 days from decision date

**Expected Revision:** Minor revision with expanded validation data and corrected manuscript descriptions

**Resubmission:** Submit revised manuscript with point-by-point response to all reviewer comments

---

## Final Editorial Statement

### My Assessment of Your Work

I have carefully weighed the four reviews and I believe your platform has genuine merit:

**Strengths I Recognize:**

1. **First integrated platform for circRNA vaccine design** — This is a meaningful contribution to an emerging field
2. **Correct biological understanding in code** — Your RIG-I implementation correctly handles circRNA topology
3. **Practical usability** — Multiple interfaces (Python, R, VS Code, CLI) lower adoption barriers
4. **Transparent code documentation** — Honesty about heuristic parameters is commendable

**Why Minor Revision, Not Major:**

Reviewer #4 recommended Major Revision due to statistical concerns. However, I am offering Minor Revision because:

1. The core methodology is sound — the issues are in **manuscript description**, not implementation
2. Your code already has statistical infrastructure (`stat_tests.py`, bootstrap CI) — you just need to use it
3. Large-scale validation code exists (`circbase_large_scale_validation.py`) — you can expand validation
4. The problems are **fixable within 90 days**

**However, I reserve the right to require additional revision if statistical reporting remains incomplete.**

---

### What I Expect in the Revised Manuscript

1. **Accurate Methods description** — especially RIG-I mechanism and weight derivation
2. **Expanded validation** — N≥100 with complete statistical reporting
3. **Transparent disclosure** — including negative validation results
4. **Corrected naming** — no misleading claims about Cox regression or empirical calibration

If you can address these concerns, I will be pleased to accept your manuscript for publication in *Bioinformatics*.

---

### Advice to Authors

From my experience editing Application Notes:

- **Don't overclaim.** A transparent description of heuristic methods is more valuable than inflated claims.
- **Validation matters.** N=10 is a case study, not validation. Bioinformatics readers expect quantitative benchmarks.
- **Your code is good.** The reviewers praised your implementation. The issues are in how you describe it.

---

## Sincerely,

**[Editor Name]**
Associate Editor, *Bioinformatics*
[University/Affiliation]

---

## Appendix: Reviewer Score Summary

| Reviewer | Role | Score | Recommendation |
|----------|------|-------|----------------|
| #1 | Methodology & Technical | 3.5/5 | Minor Revision |
| #2 | Novelty & Application | Novelty 3/5, Utility 4/5 | Minor Revision |
| #3 | Biological Validity | 3.5/5 | Minor Revision |
| #4 | Statistical Rigor | 2/5 | Major Revision |

**Editorial Synthesis:** Minor Revision with Statistical Enhancement Required

---

## Appendix: Consensus Issues

| Issue | Reviewers | Priority | Editorial Action |
|-------|-----------|----------|------------------|
| Manuscript-code inconsistency | R1, R2 | Critical | Must fix |
| RIG-I description misleading | R1, R3 | Critical | Must fix |
| N=10 sample size | R2, R4 | Critical | Must expand or acknowledge limitation |
| Missing CI/statistical tests | R4 | Critical | Must provide |
| Negative results undisclosed | R4 | High | Must disclose |
| PKR threshold inconsistency | R1, R3 | High | Should fix |
| Weight calibration lacking | R1, R2, R3 | High | Should acknowledge |
| Clinical validation missing | R2, R3, R4 | High | Should provide or downgrade |

---

*Decision Letter Generated: 2026-06-01*

*Manuscript ID: BIOINF-2026-XXXX*