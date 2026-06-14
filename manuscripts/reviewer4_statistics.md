# Peer Review Report - Reviewer #4 (Statistics & Pharmacokinetics)

## Manuscript Information
**Title:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Review Focus:** Statistical Rigor, Sample Size, Confidence Intervals, Model Validation

---

## Overall Assessment

**Recommendation:** Major Revisions Required

The manuscript presents an interesting computational platform for circRNA vaccine design with comprehensive feature integration. However, the statistical validation is severely underpowered and multiple claims lack appropriate statistical support. The N=10 sample size for validation is insufficient for robust conclusions, and the absence of confidence intervals, variance measures, and formal statistical tests undermines the reliability of reported results.

---

## Statistical Rigor Score: 2/5

**Rationale:** The manuscript lacks fundamental statistical reporting (CIs, p-values, variance measures) and draws conclusions from an inadequately small sample. While the methodology appears sound, the validation component does not meet standards for reproducible computational biology.

---

## Major Comments

### 1. Critical Sample Size Deficiency (N=10)

The circBase validation uses only N=10 sequences. This sample size is fundamentally inadequate for the claims made.

**Confidence Interval Calculation for r=0.85 with N=10:**

Using Fisher's z-transformation:
- z = 0.5 × ln((1+r)/(1-r)) = 0.5 × ln((1+0.85)/(1-0.15)) = 1.256
- SE(z) = 1/√(n-3) = 1/√7 = 0.378
- 95% CI for z: (0.515, 1.997)
- **95% CI for r: (0.47, 0.96)**

**Interpretation:** The 95% confidence interval spans from moderate (r=0.47) to very strong (r=0.96) correlation. This extremely wide interval (width = 0.49) means the correlation estimate is highly imprecise. The claim of "strong correlation" is not statistically justified.

**Minimum Sample Size Recommendations:**

| Desired CI Width | Required N (for r=0.85) |
|-----------------|------------------------|
| ±0.10           | ~50 sequences          |
| ±0.08           | ~80 sequences          |
| ±0.05           | ~200 sequences         |

**Required Action:** Expand validation to at least N=50 sequences with stratified sampling across GC content ranges, sequence lengths, and biological sources. Report exact p-values and 95% CIs for all correlation coefficients.

### 2. Missing Confidence Intervals and Statistical Tests

**Issue 2a: Correlation Coefficient (r=0.85)**
- No confidence interval reported
- No p-value reported
- t-statistic calculation: t = r × √((n-2)/(1-r²)) = 0.85 × √(8/0.2775) = 4.56
- df = 8, p ≈ 0.0018 (two-tailed)
- **Required:** Report as "r=0.85, 95% CI [0.47, 0.96], p=0.0018"

**Issue 2b: Mean Comparison (0.76 vs 0.40)**
The statement "GC-rich sequences showed higher immunogenicity scores (mean=0.76 vs mean=0.40)" lacks:
- Standard deviations or standard errors
- Sample sizes per group (how many GC-rich vs moderate GC?)
- Statistical test (t-test, Mann-Whitney, or permutation test)
- Effect size (Cohen's d)
- Confidence interval for the difference

**Required:** Report as:
```
GC-rich (n=X): mean=0.76, SD=__, 95% CI [__,__]
Moderate GC (n=Y): mean=0.40, SD=__, 95% CI [__,__]
Difference: __, 95% CI [__,__], p=__ (test used)
Effect size (Cohen's d): __
```

**Issue 2c: Performance Metrics**
Claims of "<100ms per sequence" and "<1s" provide no:
- Number of sequences benchmarked
- Hardware specifications (CPU, RAM, OS)
- Variance (SD, IQR, or min-max range)
- Distribution across sequence lengths

**Required:** Report benchmark methodology:
```
N sequences tested: ___
Hardware: [CPU model, RAM, OS]
Mean time: ___ ms (SD: ___)
95% CI: [__ , __] ms
Range: [__ , __] ms
```

### 3. Clinical Model Validation Deficiencies

**Claim:** "Survival analysis uses Cox regression approximation with IPS and TIDE integration"

**Critical Issues:**

1. **No Training Cohort Specified**
   - What dataset was used to derive model parameters?
   - Sample size of training data?
   - Feature selection methodology?

2. **No Validation Metrics**
   - Concordance index (C-index) for survival prediction?
   - Calibration plots?
   - Time-dependent ROC/AUC?

3. **No Cross-Validation**
   - Was k-fold cross-validation performed?
   - Bootstrap validation?
   - External validation on independent cohort?

4. **Hazard Ratio Interpretation**
   - Are hazard ratios reported with 95% CIs?
   - Proportional hazards assumption tested?

**Required:** Provide complete validation framework:
```
Training cohort: [dataset name, N=___, source]
Validation: [internal k-fold / bootstrap / external]
C-index: ___ (95% CI: [__,__])
Calibration: slope=__, intercept=__
Time-dependent AUC at 1/3/5 years: __/__/__
```

### 4. Over-Claiming Without Adequate Evidence

**Problematic Claims:**

| Claim | Issue |
|-------|-------|
| "Strong correlation" (r=0.85, N=10) | CI (0.47-0.96) includes moderate correlation; "strong" is not statistically justified |
| "Consistent with PKR activation" | Causal inference from correlation with insufficient sample size |
| "Higher immunogenicity scores" (0.76 vs 0.40) | No statistical test, no variance, no CI for difference |
| "Potentially reducing immunogenicity through modification-mediated immune evasion" | Speculative without experimental validation |

**Required Revisions:**
- Change "strong correlation" to "positive correlation (r=0.85, 95% CI [0.47, 0.96], p=0.002)"
- Remove or qualify causal claims ("consistent with" → "may be associated with")
- Add explicit limitations section acknowledging sample size constraints

### 5. Performance Benchmarking Standards

The performance claims lack essential methodological details:

**Missing Information:**
1. Benchmark dataset characteristics (sequence lengths, complexity)
2. Hardware specifications
3. Software versions (ViennaRNA version, Python version)
4. Statistical distribution of timing results
5. Comparison to baseline methods

**Suggested Benchmark Protocol:**
```python
# Report:
- N sequences: 100+ spanning 100-2000 nt
- Hardware: CPU model, cores, RAM, OS, Python version
- ViennaRNA version: X.XX
- Timing: mean ± SD, median, IQR
- Stratify by sequence length: short (<500 nt), medium (500-1000 nt), long (>1000 nt)
```

---

## Minor Comments

1. **Line 62-63:** "Sequences ranged from 200-1000 nt with GC content 0.50-1.00" - Report median and IQR for GC content, not just range.

2. **Line 65:** "consistent with PKR activation by GC-rich dsRNA structures" - This is an interpretation that requires experimental validation or stronger statistical support.

3. **Line 67:** "mean=18.5 vs 0 for GC-rich" - Report N per group and variance measures.

4. **Line 68:** "Optimized vaccine candidates: high-immunogenicity sequence (0.88)" - Is this a single sequence? How was optimization validated?

5. **Table 1:** Consider adding statistical comparison columns (p-values, effect sizes) where applicable.

6. **Supplementary Materials:** The "Clinical validation on TCGA datasets (if available)" suggests validation may not have been performed - this is a critical gap for clinical prediction claims.

---

## Validation Assessment

### Sample Size Adequacy: INSUFFICIENT

| Analysis | Current N | Recommended Minimum N | Status |
|----------|-----------|----------------------|--------|
| Correlation (r=0.85) | 10 | 50-80 | FAIL |
| Mean comparison | ~10 | 30+ per group | FAIL |
| Performance benchmark | Not reported | 100+ sequences | FAIL |
| Clinical validation | Not reported | 200+ with external validation | FAIL |

### Confidence Interval Reporting: NOT COMPLIANT

- 0/3 correlation coefficients have CIs
- 0/2 mean comparisons have CIs
- 0/3 performance metrics have variance measures
- 0 clinical predictions have CIs

### Statistical Test Reporting: NOT COMPLIANT

- No p-values reported
- No test statistics reported
- No multiple comparison corrections discussed
- No assumptions testing reported

---

## Model Reliability Assessment

### Clinical Prediction Models: UNRELIABLE WITHOUT VALIDATION

| Requirement | Status |
|-------------|--------|
| Training cohort specified | NOT REPORTED |
| Internal validation (CV/bootstrap) | NOT REPORTED |
| External validation | NOT REPORTED |
| Calibration assessment | NOT REPORTED |
| Discrimination (C-index) | NOT REPORTED |
| Feature selection validation | NOT REPORTED |
| Model comparison to baselines | NOT REPORTED |

**Risk:** Deploying unvalidated clinical prediction models could lead to inappropriate patient stratification or treatment decisions.

**Recommendation:** Either (a) remove clinical prediction claims until proper validation is completed, or (b) clearly label these as "exploratory" and "not clinically validated" with prominent disclaimers.

---

## Strengths

1. **Comprehensive Feature Integration:** The platform addresses multiple relevant aspects of circRNA immunogenicity (RIG-I, TLR, PKR, structure, modifications) in a unified framework.

2. **Literature-Based Scoring:** The immunogenicity scoring system is grounded in published literature with cited weights, providing transparent rationale for parameter choices.

3. **Open Source Availability:** MIT license and Python implementation enhance reproducibility and community adoption.

4. **Modular Architecture:** Clean API design allows users to access individual components independently.

5. **Practical Utility:** The evolutionary optimization feature addresses a real need in vaccine design workflows.

6. **Comparison Table:** Clear positioning against existing tools helps users understand the platform's niche.

---

## Specific Recommendations

### Priority 1 (Must Address Before Acceptance)

1. Expand circBase validation to minimum N=50 sequences with proper stratification
2. Add 95% confidence intervals for all correlation coefficients and mean differences
3. Perform and report statistical tests with exact p-values
4. Add hardware specifications and variance measures for all performance benchmarks
5. Either validate clinical prediction models or clearly label them as "unvalidated exploratory features"

### Priority 2 (Should Address)

1. Add a Limitations section explicitly discussing sample size constraints
2. Provide sensitivity analysis for immunogenicity scoring weights
3. Add cross-validation for any machine learning components
4. Report sequence characteristics (length, GC%) distribution for benchmark datasets

### Priority 3 (Consider)

1. Add external validation cohort from independent database
2. Compare predictions against experimental data where available
3. Add reproducibility section with random seed specifications
4. Consider pre-registration of validation protocol

---

## Statistical Summary Table

| Metric | Reported | Required | Gap |
|--------|----------|----------|-----|
| Sample size (correlation) | N=10 | N≥50 | -40 |
| 95% CI for r=0.85 | None | [0.47, 0.96] | Missing |
| p-value for r=0.85 | None | p=0.0018 | Missing |
| Mean comparison test | None | t-test/M-W | Missing |
| Effect size (Cohen's d) | None | Required | Missing |
| Benchmark variance | None | SD/IQR | Missing |
| Clinical validation C-index | None | Required | Missing |
| Calibration metrics | None | Required | Missing |

---

## Conclusion

The Confluencia circRNA platform presents valuable methodology for circRNA vaccine design, but the validation component is statistically underpowered and incomplete. The N=10 sample size for validation is insufficient to support the claims made, and the absence of confidence intervals, statistical tests, and variance measures does not meet the reporting standards expected in Bioinformatics.

The manuscript requires major revisions focusing on:
1. Substantially expanded validation with appropriate sample sizes
2. Complete statistical reporting (CIs, p-values, effect sizes)
3. Rigorous clinical model validation or clear disclaimers

I recommend major revisions with particular attention to statistical rigor.

---

**Reviewer:** #4 (Pharmacokinetics & Biostatistics)
**Date:** 2026-06-01
**Recommendation:** Major Revisions Required