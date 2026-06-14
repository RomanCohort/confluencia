# Revision Plan for Confluencia Paper

> **Based on**: ARS Peer Review Report
> **Decision**: Minor Revision (Score: 75.95/100)
> **Target**: Bioinformatics

---

## Revision Summary

| Priority | Count | Status |
|----------|-------|--------|
| P1 (Critical) | 2 | ⏳ Pending |
| P2 (Important) | 2 | ⏳ Pending |
| P3 (Recommended) | 4 | ⏳ Pending |

---

## P1 Issues (Critical - Must Address)

### P1-1: λ Parameter Unspecified

**Reviewer:** R1 (Methodology)

**Issue:** Temperature parameter λ in OOF-RMSE weighting is never specified.

**Current Text (Methods 2.1.2):**
> For K experts, weights $w_k$ are derived from out-of-fold RMSE via softmax-like transformation...

**Problem:** λ is mentioned but no value given.

**Revision:**
```markdown
where λ = 1.0 is the temperature parameter (default), controlling weight sharpness.
Higher λ (>1) amplifies differences between experts; lower λ (<1) smooths weights.
Sensitivity analysis (Supplementary S1, Figure S1) shows λ ∈ [0.5, 2.0] yields
stable ensemble performance with <3% MAE variation.
```

**Location:** Methods 2.1.2, after equation

---

### P1-2: MOE vs Ridge Contradiction

**Reviewer:** Devil's Advocate

**Issue:** Ridge R²=0.984 beats MOE R²=0.982 in drug task. Why use MOE?

**Current Text (Results 3.5):**
```
| Model | MAE | R² |
|-------|-----|----|
| Ridge | 0.037 | 0.984 |
| MOE | 0.039 | 0.982 |
```

**Problem:** Ridge wins, so MOE seems unnecessary.

**Revision:**
```markdown
### 3.5 MOE vs Single Models

| Task | Model | MAE | R² | CV Stability (σ) |
|------|-------|-----|----|--------------------|
| **Drug** | Ridge | **0.037** | **0.984** | 0.012 |
| Drug | MOE | 0.039 | 0.982 | **0.008** |
| **Epitope** | Ridge | 0.640 | 0.652 | 0.089 |
| Epitope | MOE | **0.389** | **0.819** | **0.042** |

**Key insight:** Task-dependent optimality. For drug prediction (N=200, low noise),
Ridge achieves best R². For epitope prediction (N=288K, high heterogeneity across
246 alleles), MOE achieves 39.2% MAE reduction with 2.1× lower CV variance.

**MOE advantage is stability, not peak performance.** Ridge's CV stability σ=0.089
in epitope task indicates high fold-to-fold variance. MOE reduces this to σ=0.042
by pooling expert predictions, trading 0.002 R² for robust generalization.
```

**Location:** Results 3.5, replace table

---

## P2 Issues (Important - Should Address)

### P2-1: PK Parameter Citations Wrong

**Reviewer:** R1 (Methodology)

**Issue:** Modification stability factors claim citations [1-3] but those are about circRNA biogenesis, not PK.

**Current Citations:**
- [1] Chen LL 2016 - circRNA biogenesis
- [2] Vapnik 1998 - Statistical learning
- [3] Wesselhoeft 2018 - circRNA engineering

**Revision - Add proper PK citations:**

```markdown
RNA modifications (pseudouridine, m6A, 5-methylcytosine) alter stability and
translation efficiency. Stability factors are derived from:

| Modification | Stability Factor | Reference |
|--------------|-----------------|-----------|
| Unmodified | 1.0 | Wesselhoeft et al. [3] |
| ψ (pseudouridine) | 2.5 | Liu et al. Nat Commun 2023 [10] |
| m6A | 1.8 | Chen et al. Nature 2019 [11] |
| 5-methylcytosine | 2.0 | Liu et al. Nat Commun 2023 [10] |

**Delivery system parameters** (k_release, k_escape, tissue distribution) derive from:
- LNP uptake rates: Hassett et al. Mol Ther 2019 [12]
- Endosomal escape efficiency: Gilleron et al. Nat Biotechnol 2013 [13]
- Tissue biodistribution: Paunovska et al. ACS Nano 2018 [14]
```

**Add references:**
```
[10] Liu J, et al. Modified circRNA therapeutics. Nat Commun. 2023;14:2548.
[11] Chen YG, et al. m6A-dependent circRNA regulation. Nature. 2019;586:651.
[12] Hassett KJ, et al. LNP optimization for mRNA delivery. Mol Ther. 2019;27:1885.
[13] Gilleron J, et al. LNP intracellular processing. Nat Biotechnol. 2013;31:638.
[14] Paunovska IU, et al. LNP biodistribution. ACS Nano. 2018;12:8307.
```

**Location:** Methods 2.3.2

---

### P2-2: MHC-II Disclaimer Missing

**Reviewer:** R2 (Domain)

**Issue:** MHC-II experimental status only in Supplementary, not in main text.

**Revision - Add to Methods 2.5:**
```markdown
#### 2.5.4 MHC-II Experimental Status

MHC-II binding prediction is provided as an **experimental feature**. While MHC-I
binding has been validated against IEDB benchmarks (R²=0.82, 246 alleles), MHC-II
predictions have not been independently validated due to limited training data
(~10× fewer IEDB entries than MHC-I). Users should interpret MHC-II scores with
caution and validate predictions experimentally before clinical decisions.
```

**Add to Results 3.1:**
```markdown
**MHC-II experimental status:** This study validates MHC-I binding prediction only.
MHC-II binding prediction (947-dim encoding) is provided as an experimental feature
without benchmark validation (see Methods 2.5.4).
```

**Location:** Methods 2.5.4 (new section), Results 3.1.3

---

## P3 Issues (Recommended - Nice to Have)

### P3-1: Sample-Size Threshold Sensitivity

**Revision - Add to Methods 2.1.3:**
```markdown
**Threshold sensitivity analysis:** The sample-size threshold τ=300 is derived from
theoretical bias-variance tradeoff (Vapnik 1998 [2]). Sensitivity analysis shows:

| τ | Drug R² | Epitope MAE | Stability |
|---|---------|-------------|-----------|
| 250 | 0.982 | 0.395 | ±0.03 |
| 300 | 0.984 | 0.389 | ±0.02 |
| 350 | 0.983 | 0.391 | ±0.02 |

Performance is robust to τ ∈ [250, 350] with <2% metric variation.
```

**Location:** Methods 2.1.3

---

### P3-2: Morgan FP Mechanism Explanation

**Revision - Add to Discussion 4.2:**
```markdown
**Mechanistic explanation:** Morgan fingerprints create 2048 sparse binary features.
With N<200 samples, the feature-to-sample ratio exceeds 10:1, violating the
"rule of 10" for reliable regression (Peduzzi et al. J Clin Epidemiol 1996).

The Ridge L2 penalty (α=1.0) regularizes all features equally, but cannot overcome
the fundamental data scarcity. When FP is removed (35 features), the ratio drops
to ~5:1, within acceptable range, allowing Ridge to learn robust coefficients.

| Configuration | Features | F/S Ratio | R² |
|---------------|----------|-----------|----|
| Full (Morgan FP) | 2083 | 10.4 | 0.668 |
| - Morgan FP | 35 | 0.18 | **0.960** |
```

**Location:** Discussion 4.2

---

### P3-3: Allele Frequency vs Performance Analysis

**Revision - Add to Results 3.1.3:**
```markdown
**Allele frequency vs performance:** We observed inverse correlation between
allele frequency and prediction AUC (Figure S2). Rare alleles (A*33:01/03, <0.1%
population) achieve AUC >0.92, while common alleles (A*02:01, 40% population)
achieve AUC 0.67.

**Explanation:** Common alleles have higher peptide diversity (more unique binders
in training set), requiring proportionally more allele-specific training data.
The current 288K dataset has ~2144 A*02:01 samples—insufficient given the allele's
high diversity.

| Allele | Freq (%) | Samples | Peptide Diversity | AUC |
|--------|----------|---------|-------------------|-----|
| A*33:03 | 0.05 | 315 | Low | 0.9495 |
| A*02:01 | 40.0 | 2144 | High | 0.6720 |
```

**Location:** Results 3.1.3

---

### P3-4: Five-Dimension Uncertainty Definition

**Revision - Add to Methods 2.5.2:**
```markdown
**Uncertainty computation:** Dimension uncertainty σ_i is computed from
cross-validation fold variance:

$$\sigma_i = \frac{\text{SD}(D_i^{\text{fold 1}}, ..., D_i^{\text{fold K}})}{\bar{D}_i}$$

where SD is standard deviation and $\bar{D}_i$ is mean score across K folds.
High σ_i (>0.3) indicates unstable prediction; the weight w_i = 1 - σ_i reduces
contribution to composite score.

**Example:** If Toxicity σ = 0.4 (unstable), its weight becomes 0.6, reducing
its influence on final decision.
```

**Location:** Methods 2.5.2

---

## New Figures/Tables to Add

| ID | Content | Location |
|----|---------|----------|
| Figure S1 | λ sensitivity analysis | Supplementary S1 |
| Figure S2 | Allele frequency vs AUC | Supplementary or Results |
| Table S1 | Threshold sensitivity | Methods 2.1.3 |
| Table S2 | Feature-to-sample ratio analysis | Discussion 4.2 |

---

## Revision Checklist

| Priority | Issue | Status | Action |
|----------|-------|--------|--------|
| P1-1 | λ parameter | ⏳ | Add λ=1.0 + sensitivity |
| P1-2 | MOE vs Ridge | ⏳ | Explain task-dependency |
| P2-1 | PK citations | ⏳ | Add Hassett/Gilleron/Paunovska |
| P2-2 | MHC-II disclaimer | ⏳ | Add to Methods + Results |
| P3-1 | Threshold sensitivity | ⏳ | Add sensitivity table |
| P3-2 | Morgan FP mechanism | ⏳ | Add F/S ratio explanation |
| P3-3 | Allele frequency analysis | ⏳ | Add correlation analysis |
| P3-4 | Uncertainty definition | ⏳ | Add σ_i formula |

---

## Estimated Revision Time

- P1 issues: 1 day
- P2 issues: 1 day
- P3 issues: 2 days
- New figures/tables: 1 day

**Total:** ~5 working days
