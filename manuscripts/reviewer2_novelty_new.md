# Reviewer #2: Novelty & Application Value

## Overall Assessment
- Recommendation: **Major Revisions**
- Novelty Score: 2.5/5
- Utility Score: 3.5/5

---

## Major Comments

### 1. Validation Sample Size is Inadequate (Critical)

The claim of practical utility rests on analysis of **only 10 circRNA sequences** from circBase. This sample size is woefully inadequate for several reasons:

- **Statistical Power**: No power analysis is provided. With N=10, the study lacks statistical power to detect meaningful differences or validate predictive accuracy. The reported correlation (r=0.85 for GC-immunogenicity) has extremely wide confidence intervals.

- **Selection Bias**: The sequences (200-1000 nt, GC 0.50-1.00) appear hand-picked. Were these randomly selected? Why only these 10? The manuscript notes they come from "circBase database and literature" but provides no systematic selection criteria.

- **External Validation Missing**: There is no independent test set, cross-validation, or comparison against gold-standard immunogenicity assays. The platform's predictions remain unvalidated against experimental data.

- **Minimum Expectation**: For an Application Note claiming clinical utility, validation on at least 50-100 sequences with experimental immunogenicity data should be required.

### 2. Novelty Claims Are Overstated

The manuscript positions itself as filling a "gap" with "no comprehensive platform existing." However, closer examination reveals:

- **Immunogenicity Scoring**: The algorithms are direct implementations of published literature (Schlee 2009, Forsbach 2008, Nallagatla 2007). The weights (0.35, 0.25, 0.20, 0.20) appear arbitrary—no justification or optimization is provided. This is **integration, not innovation**.

- **Structure Prediction**: Uses ViennaRNA (Lorenz 2011) as a dependency. The "fallback estimation" mentioned is a heuristic substitute, not a novel algorithm.

- **Folding Kinetics**: Described as "heuristics" based on GC content. This is a simplification, not a validated kinetics model.

- **m6A Prediction**: DRACH motif scanning is standard practice, implemented identically in existing tools (SRAMP, m6A-Atlas).

- **Clinical Prediction**: Cox regression approximation with IPS/TIDE integration—these scores already exist and are imported, not developed.

**The true novelty is the integration itself**, not the individual components. This is valuable but should be framed as an "integration platform" rather than novel methodology.

### 3. Comparison Table Incomplete and Misleading

The comparison table (Table 1) omits several relevant tools:

- **circRNA-specific tools not mentioned**:
  - *circBank* (provides functional annotation including immune relevance)
  - *CircInteractome* (miRNA/RBP binding for circRNAs)
  - *circRNAfinder* (structure prediction for circular RNAs)

- **Immunogenicity prediction tools**:
  - *IMMUNO-CIRC* (if exists—authors should verify)
  - General RNA immunogenicity predictors (RNAtherm, RNAStructure with immune modules)

- **m6A tools**: SRAMP, m6A-Atlas, Gene2Func all provide m6A prediction. The table marks "linear only" but DRACH motifs are identical in circRNA.

The comparison should include a comprehensive literature search for circRNA-specific analysis tools.

### 4. Practical Utility Questions

**Who would use this tool?**

The manuscript claims clinical relevance but lacks clarity on:
- Target users: Bioinformaticians? Clinical researchers? Pharma R&D?
- Use case workflow: How does this fit into an actual vaccine development pipeline?
- Decision thresholds: What immunogenicity score indicates "good vaccine candidate" vs "therapeutic cargo"?

**Clinical outcome prediction is premature:**
- Cox regression "approximation" on what training data?
- IPS/TIDE scores require tumor data that may not be available for prophylactic vaccines
- No validation against actual clinical outcomes

### 5. Performance Claims Unverified

- "Immunogenicity scoring <100ms"—on what hardware? What sequence lengths?
- "Full pipeline ~2-3s"—this is acceptable but should be benchmarked systematically
- ViennaRNA fallback accuracy: How does it compare to native ViennaRNA? What is the error margin?

---

## Novelty Assessment

### What is Truly Novel:
1. **Integration architecture**: Bringing together immunogenicity, structure, modifications, and clinical prediction in one platform
2. **Pareto multi-objective optimization**: The evolutionary sequence design module with REINFORCE policy learning is a genuine contribution
3. **circRNA-specific immunogenicity weighting**: While weights are not validated, the framework for circRNA-specific scoring is novel

### What is Integration of Existing Methods:
1. Immunogenicity scoring algorithms (directly from literature)
2. Structure prediction (ViennaRNA dependency)
3. m6A site detection (DRACH motif—standard)
4. IRES prediction (polypyrimidine tract analysis—standard)
5. miRNA binding (seed matching—standard)
6. Clinical scores (IPS/TIDE—imported)

**Recommendation**: The manuscript should be reframed as "Confluencia circRNA: An Integrated Platform..." rather than claiming novel algorithms. The novelty lies in integration and accessibility, not methodological innovation.

---

## Application Value

### Strengths for Practical Use:
1. **Accessibility**: Open-source Python package with Streamlit GUI lowers barriers for non-programmers
2. **Comprehensive workflow**: Covers multiple analysis stages in one platform
3. **Sequence optimization**: Evolutionary design addresses a real need in vaccine development
4. **Literature-backed defaults**: Weights derived from literature provide reasonable starting points

### Limitations for Practical Use:
1. **Unvalidated predictions**: No experimental immunogenicity data confirms accuracy
2. **Arbitrary thresholds**: No guidance on what scores indicate vaccine suitability
3. **Clinical claims premature**: Survival prediction lacks validation
4. **Small sample validation**: N=10 is insufficient for confidence in recommendations

### Target User Assessment:
- **Bioinformaticians**: May use Python API but will verify against literature
- **Clinical researchers**: GUI is helpful but will need validation before trusting predictions
- **Pharma R&D**: Would require extensive validation before adoption

---

## Strengths

1. **Comprehensive scope**: Addresses multiple aspects of circRNA vaccine design
2. **Open-source commitment**: MIT license, available on GitHub
3. **Multiple interfaces**: Python API, Streamlit, Electron app—flexible access
4. **Modular architecture**: Core modules can be used independently
5. **Evolutionary optimization**: Novel contribution for sequence design
6. **Literature foundation**: Algorithms traceable to published sources
7. **Active development**: Version 2.6.0 suggests ongoing maintenance
8. **Clear documentation**: Code examples and usage instructions provided

---

## Specific Revision Requests

1. **Expand validation**: Minimum 50-100 circRNA sequences with documented selection criteria; ideally include sequences with experimental immunogenicity data

2. **Validate predictions**: Compare platform outputs against:
   - Published immunogenicity assays
   - Clinical outcomes where available
   - Other prediction tools (benchmark)

3. **Revise novelty framing**: Acknowledge that components are integrations; claim novelty for:
   - Integration architecture
   - Evolutionary optimization module
   - circRNA-specific workflow

4. **Expand comparison table**: Include all relevant circRNA analysis tools with fair feature comparison

5. **Provide decision guidance**: Establish threshold values for:
   - "High immunogenicity" vs "low immunogenicity"
   - Suitable for vaccine vs therapeutic cargo
   - Clinical risk levels

6. **Validate or remove clinical predictions**: Either validate survival/biomarker predictions against outcomes, or remove/relabel as "hypothetical"

7. **Benchmark performance**: Systematic benchmarking with hardware specs, sequence length variations, accuracy comparisons

---

## Summary

Confluencia circRNA represents a useful integration of existing tools with some novel contributions (Pareto optimization). However, the validation is critically insufficient (N=10, no experimental comparison), novelty claims are overstated (integration ≠ innovation), and clinical predictions are premature. The platform has potential utility for researchers, but requires substantial validation before claims of practical utility can be supported.

**Recommendation**: Major revisions with focus on expanded validation and reframed novelty claims.
