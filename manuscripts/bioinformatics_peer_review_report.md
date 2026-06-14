# Bioinformatics Application Note - Peer Review Report

## Manuscript: Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Review Date:** 2026-06-01
**Review Type:** Concurrent Four-Reviewer Panel Review

---

## Reviewer #1: Methodology & Technical Accuracy

### Overall Assessment
- **Recommendation:** Minor Revisions
- **Quality Score:** 3.5/5

### Major Comments

1. **Method Description Insufficiency:** The Methods section (~300 words) is too brief for an Application Note. The algorithmic details for RIG-I/TLR/PKR scoring are not adequately described. For instance:
   - How exactly is "blunt-end detection" implemented algorithmically?
   - What is the precise formula for the U-rich and GU-rich motif scoring?
   - The DRACH motif is mentioned but the "probability estimation based on local GC context" needs explicit equations.

2. **Parameter Justification Gap:** The weights (RIG-I=0.35, TLR7=0.25, TLR8=0.20, PKR=0.20) are cited as "literature-backed" but no quantitative justification is provided. Were these derived from specific experimental data? A sensitivity analysis should be included.

3. **Code Reproducibility Concerns:** The paper mentions "fallback estimation for environments without ViennaRNA" but does not specify the accuracy trade-off. Users need to know when fallback results are unreliable.

### Minor Comments

- Line 36: "RNAfold with fallback estimation" - specify which version of ViennaRNA was tested
- The kinetics prediction (k = exp(-barrier/RT)) uses heuristics - clarify the validation against experimental folding rates
- Pareto front selection algorithm details are missing - what optimization solver is used?

### Specific Technical Issues

1. **Algorithm Accuracy:** The PKR activation threshold (>33bp dsRNA) is a binary criterion. How does the platform handle partial dsRNA regions or bulged duplexes?

2. **Parameter Justification:** The m6A "probability estimation" lacks citation for the GC-context model. Liu et al. 2022 describes m6A sites but not necessarily this specific probability formula.

3. **Method Gaps:** Folding kinetics uses "GC-content and sequence complexity heuristics" - this is too vague. Define "sequence complexity" operationally.

### Strengths

- Comprehensive feature integration is impressive
- Clear API design with modular architecture
- ViennaRNA integration is appropriate for structure prediction
- Performance metrics (<100ms for immunogenicity) are well-reported

---

## Reviewer #2: Novelty & Application Value

### Overall Assessment
- **Recommendation:** Minor Revisions
- **Novelty Score:** 3.5/5
- **Utility Score:** 4/5

### Major Comments

1. **Validation Sample Size Insufficient:** The circBase validation uses only N=10 sequences. For a platform claiming comprehensive capabilities, this is inadequate. A more systematic benchmark against existing tools (even for individual features) would strengthen the manuscript.

2. **Novelty Claim Assessment:** The platform integrates existing methods (ViennaRNA, DRACH motif, literature-based scoring). The novelty lies primarily in the **integration** rather than new algorithms. This should be stated more clearly to manage expectations.

3. **Comparison Table Incomplete:** Table 1 shows "Linear RNA Tools" as a single column, but should specify which tools (e.g., LinearDesign, mRNA optimizer tools). Some mRNA vaccine design tools do include immunogenicity prediction.

### Minor Comments

- The evolutionary optimization module is mentioned but results are not shown in detail
- "10 core modules" mentioned but not enumerated - a supplementary table would help
- The claim "first comprehensive platform" needs careful justification against tools like circInteractome or CIRCexplorer

### Novelty Assessment

1. **Unique Features:**
   - Multi-pathway immunogenicity scoring specifically for circRNA is novel
   - Clinical outcome prediction (survival, biomarkers) integrated with circRNA design is innovative
   - Pareto optimization for circRNA vaccine design is a novel application

2. **Comparison Gaps:**
   - No comparison with NetCircRNA or other circRNA-specific tools
   - Comparison with mRNA vaccine tools (e.g., LinearDesign) would contextualize circRNA-specific advantages

3. **Validation Sufficiency:** N=10 sequences cannot robustly validate all platform capabilities. At minimum, each module should have separate validation benchmarks.

### Application Value

1. **Target Users:** circRNA vaccine researchers, synthetic biology community - clearly beneficial
2. **Practical Utility:** The Streamlit frontend and Python API make it accessible; the modular design allows extension

### Strengths

- First platform to integrate immunogenicity + structure + clinical prediction for circRNA
- Clear practical value for vaccine design workflow
- Open-source with MIT license enables community adoption
- Multi-objective optimization addresses real design trade-offs

---

## Reviewer #3: Biological Validity & Immunology

### Overall Assessment
- **Recommendation:** Major Revisions
- **Biological Accuracy Score:** 3/5

### Major Comments

1. **RIG-I Scoring Oversimplification:** The manuscript states RIG-I recognition uses "blunt-end detection and GU-rich content." However, RIG-I specifically recognizes 5'-triphosphate ends and blunt-end dsRNA termini. CircRNAs are covalently closed and lack 5' ends. **How does the platform account for the circular topology?** This is a fundamental issue that needs clarification.

2. **TLR7/8 Activation Mechanism:** TLR7/8 recognize single-stranded RNA, particularly GU-rich and U-rich regions, but their activation depends on endosomal localization. The manuscript should clarify whether the scoring accounts for:
   - Endosomal accessibility
   - RNA modifications (pseudouridine reduces TLR activation)
   - Secondary structure accessibility

3. **m6A-Immunogenicity Link:** The claim that m6A sites "potentially reducing immunogenicity through modification-mediated immune evasion" (Results) is not well-supported. m6A can either enhance or suppress immune responses depending on context. The manuscript should nuance this claim.

4. **Clinical Prediction Validation:** The survival prediction uses "Cox regression approximation with IPS/TIDE integration." However, these scores were developed for checkpoint inhibitor response in cancer immunotherapy. Their applicability to circRNA vaccine recipients (potentially healthy individuals) is questionable.

### Minor Comments

- Schlee et al. 2009 citation for RIG-I is correct but focuses on 5'-triphosphate, not applicable to circRNA
- The PKR activation threshold (>33bp) is appropriate but should note that PKR also responds to dsRNA structure, not just length
- IRES prediction should mention dependence on cellular IRES trans-acting factors (ITAFs)

### Mechanism Assessment

1. **RIG-I Scoring Validity:** Concerns about circular topology - circRNAs lack 5' ends, so the RIG-I scoring needs explicit justification for circRNA-specific contexts

2. **TLR7/8 Prediction:** The motif-based approach is reasonable but ignores endosomal trafficking and modification effects

3. **PKR Activation Logic:** The dsRNA length threshold is literature-accurate (Nallagatla et al.), but structure complexity effects are under-addressed

4. **m6A Immunogenicity Claims:** Over-simplified; needs balanced discussion of YTHDF2-mediated degradation vs. immune activation contexts

### Clinical Relevance

1. **Survival Prediction:** Unclear whether the Cox model was trained on circRNA-treated patients or general cancer cohorts
2. **Vaccine Design Guidance:** The high/low immunogenicity binary (0.88 vs 0.35) is useful, but should discuss optimal ranges for specific vaccine types (prophylactic vs therapeutic)

### Strengths

- Appropriate literature foundation for immune sensor mechanisms
- Multi-pathway approach reflects biological complexity
- Integration of structure-immunity relationships is conceptually sound
- m6A prediction based on DRACH motif is standard

---

## Reviewer #4: Statistical Rigor & Validation

### Overall Assessment
- **Recommendation:** Major Revisions
- **Statistical Rigor Score:** 2.5/5

### Major Comments

1. **Sample Size Inadequacy:** The circBase validation uses N=10 sequences. This is insufficient to:
   - Establish generalizable performance metrics
   - Support the claimed "strong correlation" (r=0.85)
   - Validate the comparison between GC-rich (unspecified n) vs moderate GC groups

   **Recommendation:** Expand to at least N=50 sequences with stratified sampling across GC content, length, and known immunogenicity profiles.

2. **Missing Confidence Intervals:** The manuscript reports:
   - r=0.85 (no CI, no p-value)
   - mean=0.76 vs 0.40 (no CI, no statistical test)
   - Performance metrics (<100ms, <1s) without variance estimates

   All correlation coefficients and mean comparisons must include 95% confidence intervals and appropriate statistical tests.

3. **Clinical Model Validation Unspecified:** The "survival analysis uses Cox regression approximation" - was this validated? What dataset? What performance metrics? The manuscript claims "clinical outcome prediction" but provides no validation statistics.

4. **Over-claiming in Results:** Statements like:
   - "Strong correlation between GC content and overall immunogenicity (r=0.85)"
   - "Consistent with PKR activation by GC-rich dsRNA structures"

   These infer causation from correlation with insufficient sample size. The r=0.85 with N=10 has CI from ~0.5 to 0.96 - too wide for strong conclusions.

### Minor Comments

- The "mean=18.5 vs 0 for GC-rich" comparison suggests zero variance in one group - this needs clarification
- Performance timing (<100ms) should report mean ± SD across multiple sequences
- The evolutionary optimization "Pareto front tracking" is mentioned but no quantitative optimization results shown

### Validation Assessment

1. **Sample Size (N=10):** Grossly inadequate for an Application Note. Bioinformatics typically requires benchmarks with hundreds of sequences.

2. **Statistical Reporting:** Missing:
   - 95% confidence intervals for all correlations
   - P-values for group comparisons
   - Effect sizes with uncertainty bounds
   - Multiple testing corrections (if applicable)

3. **Correlation Analysis Limitations:** With N=10, the correlation is highly unstable. A single outlier can dramatically change r.

4. **Performance Claims:** The timing metrics lack:
   - Sample size for benchmarking
   - Hardware specifications
   - Variance across different sequence lengths/complexities

### Model Reliability

1. **Clinical Prediction:** No validation data shown. Cox model requires:
   - Training cohort specification
   - Cross-validation or holdout test results
   - Calibration metrics (e.g., Hosmer-Lemeshow test)

2. **Performance Claims:** Needs systematic benchmarking against ground truth datasets

### Strengths

- Clear statement of performance timing
- Recognition of limitation regarding ViennaRNA dependency
- Modular validation approach (each capability tested)

---

## Editorial Summary

### Overall Recommendation: Major Revisions Required

### Consensus Issues (Raised by Multiple Reviewers)

| Issue | Reviewers | Priority |
|-------|-----------|----------|
| Validation sample size (N=10) insufficient | R2, R4 | **Critical** |
| Missing confidence intervals and statistical tests | R4 | **Critical** |
| Method description too brief | R1 | High |
| RIG-I scoring for circular topology unclear | R3 | High |
| Clinical prediction model validation missing | R3, R4 | High |

### Decision Rationale

The manuscript presents a potentially valuable platform for circRNA vaccine design with genuine integration of multiple prediction capabilities. However, the validation is insufficient for an Application Note in Bioinformatics:

1. **Statistical rigor** is below journal standards (N=10, no CIs, no statistical tests)
2. **Biological validity** concerns about RIG-I scoring for circular RNAs need clarification
3. **Method reproducibility** requires more detailed algorithm descriptions

### Required Revisions for Acceptance

1. **Expand validation to at least N=50 sequences** with diverse characteristics
2. **Add confidence intervals** for all correlation coefficients and group comparisons
3. **Clarify RIG-I scoring mechanism** for circRNA-specific topology
4. **Validate or remove clinical prediction claims** if no validation data available
5. **Expand Methods section** to include algorithmic details and equations
6. **Add sensitivity analysis** for weight parameters

### Timeline

- Major revisions requested
- Resubmission deadline: [To be specified by editor]

---

*Review generated by concurrent four-reviewer panel simulation*
*Review model: Claude Sonnet 4.6*
