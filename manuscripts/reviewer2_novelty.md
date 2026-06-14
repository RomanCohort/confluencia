# Peer Review Report: Reviewer #2

## Manuscript: Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Reviewer Expertise:** circRNA biology and RNA Therapeutics

**Date:** 2026-06-01

---

## Overall Assessment

**Recommendation:** Major Revisions Required

This manuscript presents Confluencia circRNA, an integrated platform for circRNA vaccine design. While the platform demonstrates utility as a workflow integration tool, several claims require substantiation, and the comparison with existing tools is incomplete. The "first comprehensive platform" claim requires more rigorous justification.

---

## Novelty Score: 3/5

**Rationale:** The platform represents a meaningful integration of existing methods rather than novel algorithmic contributions. The novelty lies primarily in:
- Aggregation of circRNA-specific workflows into a single platform
- Pareto optimization for multi-objective sequence design
- Clinical prediction module integration (IPS, TIDE)

However, core components are derived from established tools (ViennaRNA) and literature-based scoring heuristics rather than novel computational methods.

---

## Utility Score: 4/5

**Rationale:** The platform addresses a practical need for circRNA researchers. The modular API design and Streamlit interface make it accessible. The integration of multiple analysis steps into a single workflow has clear value for the field.

---

## Major Comments

### 1. "First Comprehensive Platform" Claim is Overstated

The manuscript claims "No comprehensive platform exists that integrates circRNA-specific immunogenicity prediction with sequence optimization capabilities." This requires verification. The authors should:

- Specifically address how Confluencia differs from established circRNA tools such as:
  - **LinearDesign** (Zhang et al., 2020): While mRNA-focused, it addresses RNA vaccine design optimization
  - **NetCircRNA** (if applicable): circRNA-specific prediction tools exist
  - **CIRCexplorer** (Zhang et al., 2016): circRNA annotation and analysis pipeline
  - **circInteractome** (Dudekula et al., 2016): circRNA interaction prediction

The current Table 1 comparison is incomplete and uses vague column headers ("Linear RNA Tools", "circRNA databases"). A more rigorous comparison table should include specific tool names with version numbers and citations.

### 2. Validation Dataset Insufficiency

N=10 circBase sequences is inadequate for a platform claiming comprehensive utility. Major concerns:

- **No experimental validation:** All results are computational predictions without wet-lab confirmation
- **Limited sequence diversity:** 200-1000 nt range may not represent therapeutic circRNA sizes (typically 500-3000 nt for vaccine applications)
- **No comparison with known immunogenic/non-immunogenic standards:** Authors should include experimentally validated circRNAs with known immunogenicity profiles
- **Statistical analysis missing:** The reported correlation (r=0.85) lacks p-value, confidence intervals, or proper statistical framework

**Recommended benchmarks:**
- Minimum 50-100 sequences spanning diverse circRNA types
- Include validated vaccine candidates from literature (e.g., mRNA vaccine sequences adapted to circRNA format)
- Cross-validation with published immunogenicity datasets
- Performance comparison against existing prediction tools on same benchmark

### 3. Scoring Algorithm Validation

The immunogenicity scoring weights (RIG-I=0.35, TLR7=0.25, TLR8=0.20, PKR=0.20) are stated as "literature-backed" but the specific derivation is unclear:

- How were these weights calibrated?
- Is there any experimental data supporting the weighting scheme?
- Were alternative weighting schemes tested?
- What is the false positive/negative rate against known immunogenic sequences?

### 4. Clinical Prediction Claims Require Caveats

The clinical prediction module (survival analysis, biomarker interpretation) makes strong claims without validation:

- Cox regression "approximation" is not explained
- IPS and TIDE integration methodology unclear
- No validation against clinical trial data
- Risk of over-interpreting computational predictions as clinical guidance

The manuscript should clearly state these are research tools, not clinical decision aids.

---

## Minor Comments

### 1. The "10 core modules" mentioned in Results are not enumerated. Please provide a complete list with brief descriptions.

### 2. Folding kinetics prediction uses "GC-content and sequence complexity heuristics" - these should be explicitly detailed or referenced.

### 3. m6A DRACH motif prediction is well-established but the "probability estimation based on local GC context" requires validation or citation.

### 4. The claim "ViennaRNA Package 2.0" in references should be updated (current version is 2.6+).

### 5. Performance metrics (<100ms, <1s) lack system specifications. Please specify hardware/environment used for benchmarking.

### 6. The miRNA binding site database (15+ oncogenic/regulatory miRNA seeds) should be listed in supplementary materials.

### 7. REINFORCE policy learning for operator selection is mentioned but not explained - this is a potentially novel contribution that deserves more detail.

### 8. Line 88: "circRNA-specific" in the table is unclear - ViennaRNA can predict structure for any RNA including circRNA.

---

## Novelty Assessment

### Unique Features (Genuinely Novel):
1. **Integrated circRNA-specific workflow** - Combining immunogenicity, structure, modifications, and clinical prediction
2. **Pareto multi-objective optimization** for circRNA sequence design
3. **Literature-derived weight integration** for immune pathway scoring
4. **REINFORCE-based operator adaptation** in evolutionary design

### Features That Are Not Novel:
1. ViennaRNA structure prediction (wrapper around existing tool)
2. DRACH motif m6A detection (standard approach)
3. miRNA binding site prediction (complementarity scoring is standard)
4. Cox regression survival modeling (standard statistical approach)

### Comparison Gaps in Table 1:

The current comparison table should include:

| Tool | Type | Immunogenicity | Structure | circRNA-specific | Optimization |
|------|------|----------------|-----------|------------------|--------------|
| LinearDesign | mRNA vaccine design | Partial | Yes | No | Yes (codon) |
| CIRCexplorer | circRNA annotation | No | No | Yes | No |
| circInteractome | circRNA interactions | No | No | Yes | No |
| NetCircRNA | circRNA prediction | No | No | Yes | No |
| RNAfold | RNA structure | No | Yes | No | No |
| **Confluencia** | circRNA vaccine | Yes | Yes | Yes | Yes |

---

## Application Value

**Target Audience:**
- circRNA researchers designing vaccine candidates
- Computational biologists needing integrated workflows
- Biotech companies developing circRNA therapeutics

**Accessibility Assessment:**
- Python API: Accessible to computational users
- Streamlit frontend: Accessible to non-programmers
- MIT license: Good for open-source adoption

**Concerns:**
- ViennaRNA dependency may limit accessibility (fallback mode less accurate)
- No cloud/deployment option mentioned
- Documentation completeness unknown (only GitHub referenced)

---

## Strengths

1. **Practical integration:** Addresses real workflow needs for circRNA researchers
2. **Modular design:** API allows component reuse
3. **Open source:** MIT license enables community contribution
4. **Clinical translation consideration:** Includes survival and biomarker modules
5. **Multi-objective optimization:** Pareto approach is appropriate for vaccine design tradeoffs
6. **Performance:** Fast computational times suitable for screening

---

## Recommended Revisions

1. Revise "first comprehensive platform" claim to "first integrated platform combining X, Y, Z" with proper tool comparison
2. Expand Table 1 with specific tools (LinearDesign, CIRCexplorer, circInteractome, NetCircRNA)
3. Substantially expand validation (minimum 50 sequences, statistical framework)
4. Enumerate the 10 core modules explicitly
5. Clarify clinical prediction limitations
6. Provide supplementary materials with benchmark datasets and validation code
7. Consider adding experimental validation or collaboration data

---

## Summary

Confluencia circRNA is a valuable integration tool that fills a practical gap in the circRNA research workflow. However, the novelty is primarily in integration rather than methodology, and the validation is insufficient to support comprehensive claims. With major revisions addressing the tool comparison, expanded validation, and appropriate caveats, this could be a useful Application Note.

**Decision:** Major revisions required before publication.

---

*Review completed: 2026-06-01*