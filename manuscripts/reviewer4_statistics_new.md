# Reviewer #4: Statistical Rigor & Validation

## Overall Assessment
- Recommendation: **Major Revisions**
- Statistical Rigor Score: 2/5

The manuscript presents an interesting computational platform for circRNA vaccine design, but the statistical validation is severely underpowered and lacks the rigor expected for a bioinformatics publication. The primary concerns are: (1) inadequate sample size for validation claims, (2) missing confidence intervals and p-values for all statistical assertions, (3) group comparisons without hypothesis testing, and (4) performance metrics reported without variance estimates.

---

## Major Comments

### 1. Critically Underpowered Validation (N=10)

The circBase validation uses only N=10 sequences, which is insufficient to support the manuscript's conclusions:

- **Correlation claim (r=0.85)**: With N=10, the 95% confidence interval for r=0.85 spans approximately 0.50 to 0.96. This extremely wide interval means the true correlation could be anywhere from moderate to near-perfect. The point estimate alone is misleading without acknowledging this uncertainty.

- **Power analysis missing**: No justification for sample size is provided. For correlation analysis, achieving 80% power to detect r=0.85 at α=0.05 requires approximately N=8-10, but this assumes the true effect size is indeed 0.85. If the true correlation is lower (e.g., r=0.6), N=10 provides only ~40% power.

- **Recommendation**: Either substantially increase the validation sample size (minimum N=30-50 for correlation studies; ideally 100+ for a platform paper) or reframe the validation as a pilot/demonstration study with appropriately cautious language.

### 2. Missing Confidence Intervals and Statistical Tests

**Correlation analysis**: The GC-immunogenicity correlation (r=0.85) is reported without:
- 95% confidence interval
- p-value (though likely significant given r=0.85)
- Test of assumptions (linearity, normality of residuals)

**Group comparisons**: The manuscript reports:
- GC-rich vs moderate GC immunogenicity: mean=0.76 vs mean=0.40
- AU-rich vs GC-rich m6A sites: mean=18.5 vs 0

Neither comparison includes:
- Sample sizes per group
- Standard deviations or standard errors
- Confidence intervals for mean differences
- Statistical tests (t-test, Mann-Whitney, etc.)
- p-values

Without these, readers cannot assess whether observed differences are statistically meaningful or due to chance variation in a tiny sample.

### 3. Undefined Group Boundaries

The classification "GC-rich sequences (GC>0.6)" vs "moderate GC" and "AU-rich" is introduced without:
- Clear definition of all group boundaries
- How many sequences fall into each group
- Justification for these thresholds

With N=10 total, each group likely contains only 3-4 sequences, making any group comparison unreliable.

### 4. Performance Claims Without Variance

The performance metrics are reported as single values:
- "Immunogenicity scoring <100ms per sequence"
- "Structure prediction <1s (ViennaRNA) or <50ms (fallback)"
- "Full pipeline ~2-3s per sequence"

Missing:
- Number of timing measurements
- Mean ± standard deviation (or median with IQR)
- Hardware specifications
- Confidence intervals

"~2-3s" suggests multiple measurements were taken, but no quantitative summary is provided.

### 5. Clinical Outcome Validation Not Demonstrated

The manuscript claims "clinical outcome prediction including survival analysis" but provides:
- No validation dataset description
- No sample size for clinical validation
- No metrics (C-index, calibration, time-dependent AUC)
- No comparison to established baselines

The mention of "Cox regression approximation" raises questions about model specification and validation strategy that are not addressed.

---

## Validation Assessment

### circBase Validation: Inadequate

| Criterion | Assessment |
|-----------|------------|
| Sample size | N=10; critically underpowered |
| Statistical tests | None reported |
| Confidence intervals | None reported |
| Effect size precision | Extremely imprecise (r CI: ~0.50-0.96) |
| Generalizability | Unknown; no power analysis |
| Reproducibility | Sequences not fully specified |

### Specific Statistical Gaps

1. **No hypothesis testing framework**: What null hypotheses are being tested? What significance level?

2. **No effect size interpretation**: Cohen's guidelines suggest r=0.5 is "large," but the wide CI means we cannot be confident about even the direction of moderate vs strong effects.

3. **No correction for multiple comparisons**: Multiple outcomes (immunogenicity, m6A sites, structure metrics) were analyzed but no adjustment for multiple testing.

4. **No validation split**: The same 10 sequences appear to be used for both demonstration and validation, introducing circularity.

---

## Model Reliability

### Immunogenicity Scoring Model

The model uses "literature-backed" weights:
- RIG-I: 0.35
- TLR7: 0.25
- TLR8: 0.20
- PKR: 0.20

**Concerns**:
- No statistical validation of weight assignments
- No sensitivity analysis showing how predictions change with weight perturbations
- No cross-validation or independent test set
- Agreement with experimental data not quantified

### Clinical Prediction Model

**Critical concerns**:
- Cox regression "approximation" method not described
- No concordance index (C-index) reported
- No calibration plots
- No comparison to existing clinical models
- TCGA validation mentioned in supplementary but not shown

### Recommendations for Model Validation

1. **Internal validation**: k-fold cross-validation (k≥5) with held-out test set
2. **External validation**: Independent circRNA dataset
3. **Performance metrics**: C-index for survival, AUC for binary outcomes, calibration slope
4. **Comparison to baselines**: Random sequences, known immunogenic/non-immunogenic circRNAs
5. **Sensitivity analysis**: Vary model weights and assess prediction stability

---

## Strengths

1. **Comprehensive framework**: The platform integrates multiple relevant biological signals (RIG-I, TLR, PKR, structure, modifications) into a unified scoring system.

2. **Literature-backed design**: Weight assignments are based on published studies, providing biological plausibility.

3. **Open-source availability**: MIT license and Python implementation promote reproducibility and community validation.

4. **Practical utility**: The modular API and Streamlit interface lower barriers to adoption.

5. **Honest performance claims**: The authors do not overstate precision; "<100ms" appropriately indicates approximate timing.

---

## Required Revisions

1. **Expand validation sample size**: Minimum N=50-100 circRNAs for correlation studies, ideally with known immunogenicity labels from experimental studies.

2. **Report all statistics with uncertainty**:
   - Correlations: r, 95% CI, p-value, N
   - Group comparisons: mean ± SD, N per group, test statistic, p-value, effect size (Cohen's d)
   - Performance metrics: mean ± SD or median (IQR), N measurements

3. **Define group boundaries clearly**: Report exact group assignments and sample sizes.

4. **Validate clinical prediction**: Provide C-index, calibration metrics, and comparison to baseline models.

5. **Add sensitivity analysis**: Demonstrate robustness of predictions to weight perturbations.

6. **Include power analysis**: Justify sample sizes prospectively or acknowledge limitations.

---

## Statistical Reporting Checklist

| Item | Reported? | Action Needed |
|------|-----------|---------------|
| Sample size for each analysis | Partially (N=10) | Expand sample; report N per group |
| Confidence intervals for correlations | No | Add 95% CI |
| p-values for group comparisons | No | Add appropriate tests |
| Effect sizes with CIs | No | Add Cohen's d with 95% CI |
| Performance variance (SD/IQR) | No | Add mean ± SD or median (IQR) |
| Multiple testing correction | No | Apply Bonferroni or FDR if applicable |
| Validation/test split | No | Use independent test set |
| Model performance metrics | No | Add C-index, calibration for clinical model |

---

## Conclusion

The Confluencia circRNA platform addresses an important need in circRNA vaccine design, but the current validation is statistically inadequate for publication in a rigorous bioinformatics journal. The N=10 validation sample, absence of confidence intervals and hypothesis testing, and lack of model performance quantification are critical weaknesses that must be addressed.

The manuscript would benefit from either: (a) substantial additional validation with proper statistical reporting, or (b) reframing as a methods/tool announcement with explicit acknowledgment that validation is preliminary and ongoing.