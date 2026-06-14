# Peer Review Report: Confluencia circRNA Platform

**Manuscript:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Reviewer:** #1 (Methodology Review)

**Date:** 2026-06-01

---

## Overall Assessment

**Recommendation:** Minor Revisions

**Quality Score:** 4/5

The manuscript presents a well-conceived computational platform for circRNA vaccine design. The scope and integration of multiple analysis modules is impressive and addresses a genuine gap in the field. The Methods section is appropriately sized (~300 words) for an Application Note, but several algorithmic details require clarification for full reproducibility.

---

## Major Comments

### 1. Incomplete Algorithm Descriptions

The Methods section mentions several key algorithms without providing sufficient detail for implementation:

- **Blunt-end detection algorithm**: The RIG-I scoring mentions "blunt-end detection" but provides no algorithmic description. How are blunt ends identified? Is this based on secondary structure prediction from ViennaRNA? What constitutes a "blunt end" algorithmically - exact base-pair termination, length threshold, energy criteria?

- **U-rich scoring formula**: TLR7/8 activation uses "U-rich and GU-rich motifs" but the exact scoring formula is not provided. Forsbach et al. (2008) describe multiple motif patterns - which specific motifs are used? What is the scoring function (linear, weighted, position-dependent)?

- **Pareto selection algorithm**: The evolutionary optimization mentions "Pareto front selection" but does not specify the algorithm (NSGA-II, SPEA2, or custom implementation?). The number of objectives (4 mentioned) and how dominance is computed should be clarified.

**Recommendation:** Add a supplementary methods section or GitHub wiki with algorithmic pseudocode for these core algorithms.

### 2. Weight Justification and Aggregation

The weights (0.35/0.25/0.20/0.20) for RIG-I/TLR7/TLR8/PKR pathways are provided with literature citations, but:

- The manuscript does not explain how these specific weights were derived from the cited literature. Schlee et al. (2009), Forsbach et al. (2008), and Nallagatla et al. (2007) do not directly provide these numerical weights.

- The aggregation formula (weighted sum?) for computing the "overall score (0-1)" is not specified. Is it a simple weighted average? Are there normalization steps?

- Cross-reactivity between pathways is not addressed. For instance, dsRNA regions contribute to both PKR and potentially RIG-I activation - is there overlap handling?

**Recommendation:** Provide explicit derivation of weights from source literature or acknowledge as empirical calibration; specify the aggregation formula.

### 3. Parameter Thresholds Without Justification

Several thresholds are used without adequate justification:

- **PKR threshold >33bp**: This is attributed to Nallagatla et al. (2007). While this is a reasonable reference, the paper should note that Nallagatla's work uses in vitro transcribed dsRNA, and the threshold may differ for circRNA contexts.

- **GC content correlation (r=0.85)**: Strong correlation reported, but no confidence intervals or p-values provided.

- **m6A probability estimation**: Mentions "probability estimation based on local GC context" without describing the probability model.

### 4. Fallback Mechanism Documentation

The manuscript notes "fallback estimation for environments without ViennaRNA installation" but provides no details:

- What approximation method is used?
- What is the accuracy trade-off (the <50ms vs <1s timing suggests significant differences)?
- Users need to know if fallback results are publication-quality.

**Recommendation:** Document the fallback algorithm and its limitations; consider adding a performance benchmark comparing ViennaRNA vs. fallback accuracy.

### 5. Kinetics Prediction Heuristics

The kinetics prediction "using GC-content and sequence complexity heuristics" is underspecified:

- What is the formula for `k = exp(-barrier/RT)`?
- How is `barrier` estimated from GC-content?
- What is `R` (gas constant) and `T` (temperature, presumably 310K?)?
- "Metastable state count" - how is this defined and detected?

---

## Minor Comments

### 1. Code Availability Statement

The GitHub URL is provided, but:
- No specific version/commit hash or DOI for reproducibility
- No mention of test coverage or CI/CD
- Dependencies are listed but version constraints are not specified (numpy>=?, pandas>=?)

**Recommendation:** Add Zenodo DOI for version archival; specify minimum dependency versions.

### 2. Validation Dataset Description

The circBase validation uses "10 circRNA sequences" but:
- circBase IDs or accession numbers are not provided
- "Literature (Du et al., 2016; Hansen et al., 2013; Zheng et al., 2016)" - which specific sequences from these papers?

**Recommendation:** Provide a supplementary table with sequence IDs, lengths, and sources.

### 3. Performance Metrics

The performance metrics (<100ms, <1s, 2-3s) lack:
- Hardware specifications
- Single sequence vs. batch processing timing
- Memory usage

---

## Specific Technical Issues

| Line | Issue | Required Action |
|------|-------|-----------------|
| 34 | "GU-rich content analysis" - GU content ratio formula not specified | Define formula |
| 34 | "weight=0.35" notation is informal | Use proper subscript or equation format |
| 40 | DRACH motif defined as D=A/G/U, but D typically includes all nucleotides except C in IUPAC | Clarify or correct |
| 44 | "REINFORCE policy learning" - no algorithmic detail or hyperparameters | Expand or reference supplementary |
| 62 | "GC content 0.50-1.00" - GC=1.00 is impossible for RNA (no T) | Clarify or correct range |
| 66 | "mean=0.76 vs mean=0.40" - no statistical test or n values | Add statistical comparison |

---

## Strengths

1. **Comprehensive Integration**: The platform uniquely integrates immunogenicity prediction, structure analysis, modification mapping, and evolutionary optimization in a circRNA-specific context.

2. **Literature-Backed Approach**: The use of established literature for parameter values (Schlee, Nallagatla, Forsbach) provides biological grounding.

3. **Practical Tooling**: Python API with Streamlit frontend makes the tool accessible to both computational and experimental researchers.

4. **Fallback Mechanism**: Including a ViennaRNA-free fallback improves accessibility, though documentation needs improvement.

5. **Clinical Translation Focus**: Integration of survival analysis and biomarker assessment addresses real-world research needs.

6. **Open Source**: MIT license and GitHub availability support reproducibility and community contribution.

---

## Summary of Required Revisions

**Must Address (for acceptance):**
1. Provide algorithmic details for blunt-end detection, U-rich scoring, and Pareto selection
2. Specify weight derivation and aggregation formula
3. Correct GC content range (GC=1.00 impossible for RNA)
4. Add statistical tests for validation comparisons

**Should Address (recommended):**
1. Document fallback mechanism with accuracy comparison
2. Specify kinetics prediction formulas with parameter values
3. Add version DOI and dependency constraints
4. Provide validation sequence accessions in supplementary material

---

## Verdict

The Confluencia circRNA platform is a valuable contribution that fills an important gap in circRNA vaccine design tooling. The manuscript is well-structured for an Application Note, but methodological gaps prevent full reproducibility. With minor revisions addressing the algorithmic details and parameter justifications noted above, the manuscript should be suitable for publication.

**Quality Score Breakdown:**
- Originality: 5/5
- Methodology: 3/5 (needs details)
- Validation: 4/5
- Presentation: 4/5
- Reproducibility: 3/5 (needs details)
- **Overall: 4/5**