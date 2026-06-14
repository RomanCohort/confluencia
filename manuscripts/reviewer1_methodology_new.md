# Reviewer #1: Methodology & Technical Accuracy

**Manuscript:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Review Date:** 2026-06-01

**Reviewer Focus:** Algorithm accuracy, parameter justification, code reproducibility, technical implementation

---

## Overall Assessment

- **Recommendation:** Minor Revisions
- **Quality Score:** 3.5/5

The manuscript presents a comprehensive computational platform for circRNA vaccine design with substantial technical merit. The integration of multiple analysis modules (immunogenicity, structure, modifications, clinical prediction, evolution) addresses a genuine gap in circRNA research tooling. However, several critical methodological gaps prevent full reproducibility and require clarification before publication.

---

## Major Comments

### 1. Weight Derivation is Not Literature-Backed

**Critical Issue:** The manuscript claims weights are "literature-backed" but the actual implementation reveals these are "author-informed heuristics, NOT empirically calibrated" (see `immune_sensing.py` lines 19-40).

The manuscript states:
- RIG-I weight = 0.35
- TLR7 weight = 0.25 (manuscript) vs 0.20 (implementation)
- TLR8 weight = 0.20 (manuscript) vs 0.15 (implementation)
- PKR weight = 0.20 (manuscript) vs 0.30 (implementation)

**Discrepancy found:** The manuscript reports different weights than the actual code implementation:
- Manuscript: RIG-I(0.35), TLR7(0.25), TLR8(0.20), PKR(0.20)
- Implementation (line 448): RIG-I(0.35), TLR7(0.20), TLR8(0.15), PKR(0.30)

The cited papers (Schlee et al. 2009; Forsbach et al. 2008; Nallagatla et al. 2007) do NOT provide numerical weights for pathway contributions. The weights are explicitly labeled as "heuristic" in the code, contradicting the manuscript's "literature-backed" claim.

**Recommendation:** Either:
1. Revise manuscript to acknowledge weights as heuristic estimates with literature-informed direction (not magnitude)
2. Provide empirical derivation from validation data (correlation with IFN response data)
3. Update weight values in manuscript to match implementation (TLR7=0.20, TLR8=0.15, PKR=0.30)

### 2. RIG-I Algorithm Description Incorrect for circRNA

**Critical Issue:** The manuscript describes RIG-I scoring as using "blunt-end detection" (line 34), but circRNA is a covalently closed loop with NO 5' or 3' ends. The canonical RIG-I pathway (5'-triphosphate blunt-end recognition) does NOT apply to circRNA.

The implementation correctly addresses this (lines 11-17 in `immune_sensing.py`):
```
IMPORTANT BIOLOGICAL NOTE:
  circRNA is a covalently closed loop with NO 5' or 3' ends. Therefore:
  - RIG-I CANNOT recognize circRNA via 5'-triphosphate blunt-end sensing
  - RIG-I may be INDIRECTLY activated by circRNA through:
    * dsRNA structures (backbone-forming inverted repeats)
```

The manuscript's "blunt-end detection" terminology is misleading and contradicts the actual implementation which uses dsRNA structure potential scoring.

**Recommendation:** Revise Methods to clarify:
- RIG-I activation for circRNA occurs through dsRNA backbone structures (NOT blunt ends)
- Replace "blunt-end detection" with "dsRNA structure potential estimation"
- Cite Zhang et al., Nat Immunol 2016 for circRNA-specific RIG-I activation

### 3. Aggregation Formula Not Specified

**Issue:** The manuscript states outputs include "weighted overall score (0-1)" but does not specify:
- The aggregation formula (simple weighted sum? normalized?)
- Normalization method for individual pathway scores
- Handling of score correlations/overlap between pathways

The implementation shows a simple weighted sum (line 448):
```python
overall = (0.35 * rig_i_score +
           0.20 * tlr7_score +
           0.15 * tlr8_score +
           0.30 * pkr_score)
```

But individual scores have different normalization methods (some clipped to [0,1], some with circRNA corrections), making the final scale ambiguous.

**Recommendation:** Add explicit equation:
```
Overall = w_RIG-I × S_RIG-I + w_TLR7 × S_TLR7 + w_TLR8 × S_TLR8 + w_PKR × S_PKR
where S_i ∈ [0,1] are pathway-specific normalized scores
```

### 4. Kinetics Formula Parameters Not Specified

**Issue:** The manuscript mentions `k = exp(-barrier/RT)` but does not specify:
- R value (gas constant = 1.987 cal/(mol·K) or 8.314 J/(mol·K)?)
- T value (temperature - is it 310K/37°C?)
- How barrier is estimated from sequence

The implementation provides these (lines 30-31):
```python
RT_37C = 0.616  # kcal/mol at 37°C (310K)
K0_FOLDING = 1e6  # Base folding rate constant (s^-1)
```

**Recommendation:** Add to manuscript:
- R = 1.987 cal/(mol·K) ≈ 0.001987 kcal/(mol·K)
- T = 310K (37°C physiological temperature)
- RT = 0.616 kcal/mol
- k₀ = 10⁶ s⁻¹ (base folding rate)

### 5. Pareto Selection Algorithm Underspecified

**Issue:** The manuscript mentions "Pareto front selection" but does not specify:
- Algorithm used (NSGA-II, SPEA2, custom?)
- Dominance criterion (maximization of all objectives?)
- How solutions are selected from Pareto front

The implementation uses a custom dominance filter (lines 97-110):
```python
def _pareto_front_mask(X: np.ndarray) -> np.ndarray:
    """Identify Pareto-optimal points (maximization)."""
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        ...
```

This is a simple dominance filter, not a standard Pareto algorithm like NSGA-II.

**Recommendation:** Clarify in manuscript:
- "Custom Pareto dominance filter for maximization"
- Reference to standard Pareto methods (NSGA-II) for context

---

## Minor Comments

### 1. DRACH Motif Definition Inconsistency

Line 40 defines DRACH as "D=A/G/U, R=A/G, A, C, H=A/C/U" but IUPAC code D typically means "A/G/U" (not C). The implementation correctly uses:
```python
M6A_MOTIF_PATTERN = re.compile(r"[AGU][AG]AC[ACU]", re.IGNORECASE)
```

This matches the manuscript definition. No correction needed, but consider adding IUPAC reference for clarity.

### 2. GC Content Range Error

Line 62 states "GC content 0.50-1.00" but GC=1.00 is impossible for RNA (RNA has A, U, G, C; if GC=100%, then AU=0%, which means the sequence has no A or U - possible but extremely unlikely for circRNA). Consider revising to "0.50-0.85" or explaining that GC=1.00 represents theoretical maximum.

### 3. Statistical Tests Missing

Lines 66-68 report mean comparisons ("mean=0.76 vs mean=0.40") without:
- Sample sizes (n values)
- Statistical tests (t-test, Mann-Whitney)
- Confidence intervals
- P-values

**Recommendation:** Add statistical rigor:
- Report n values for each comparison
- Include p-values or effect sizes
- Consider non-parametric tests for small samples

### 4. REINFORCE Policy Underspecified

Line 44 mentions "REINFORCE policy learning" without:
- Learning rate (implementation uses lr=0.06)
- Exploration strategy (epsilon-greedy with ε=0.15)
- Convergence criteria
- Expected sample complexity

The implementation notes convergence limitations (lines 455-461):
```
CONVERGENCE NOTE: With only 4 actions and 5 evolution rounds, REINFORCE
may not fully converge... Full convergence would require 10-20 rounds.
```

**Recommendation:** Add convergence discussion and hyperparameters to manuscript or supplement.

### 5. Version/DOI Missing

No specific version, commit hash, or Zenodo DOI for reproducibility. Dependencies lack version constraints (numpy>=?, pandas>=?).

---

## Specific Technical Issues

| Line | Issue | Severity | Required Action |
|------|-------|----------|-----------------|
| 34 | "blunt-end detection" incorrect for circRNA | **Major** | Replace with "dsRNA structure potential" |
| 34 | Weights differ from implementation | **Major** | Update manuscript weights or explain discrepancy |
| 35-36 | TLR weights 0.25/0.20 vs impl 0.20/0.15 | **Major** | Correct to match implementation |
| 40 | DRACH definition needs IUPAC reference | Minor | Add IUPAC nucleotide code reference |
| 44 | REINFORCE hyperparameters not specified | Minor | Add lr=0.06, ε=0.15 to supplement |
| 62 | GC=1.00 improbable for RNA | Minor | Revise range to 0.50-0.85 |
| 66 | No statistical tests for mean comparison | Minor | Add n values and p-values |
| 70 | Performance lacks hardware specs | Minor | Add CPU/RAM specifications |
| 126 | m6A pattern regex not explained | Minor | Show regex pattern in supplement |

---

## Strengths

1. **Comprehensive circRNA-Specific Design**: The platform correctly identifies that circRNA has no 5'/3' ends and implements dsRNA-based RIG-I scoring instead of blunt-end detection. This biological awareness is commendable.

2. **Well-Documented Implementation**: The code contains extensive literature citations and biological notes (e.g., lines 11-17 explaining circRNA RIG-I mechanism), exceeding typical bioinformatics documentation.

3. **Fallback Mechanism**: Including ViennaRNA-free fallback improves accessibility for users without bioinformatics infrastructure, with appropriate warnings about accuracy limitations.

4. **Separate TLR7/TLR8 Scoring**: Recognizing distinct motif preferences (TLR7=GU-rich, TLR8=AU-rich) reflects current immunology understanding.

5. **Modification Directionality**: The m6A immunogenicity effect considers both immune evasion and enhancement directions (lines 816-864), showing sophisticated understanding of modification biology.

6. **Open Source with MIT License**: Full code availability supports reproducibility and community contribution.

7. **Multi-Objective Optimization**: Pareto-based evolution enables trade-off analysis between stability, translation, immune safety, and delivery.

---

## Code Verification Summary

I verified the manuscript claims against the implementation (`immune_sensing.py`, `rna_modifications.py`, `folding_kinetics.py`, `cirrna_evolution.py`):

| Claim | Verification | Status |
|-------|--------------|--------|
| RIG-I blunt-end detection | Implementation uses dsRNA structure | **Contradicted** |
| Weights 0.35/0.25/0.20/0.20 | Implementation uses 0.35/0.20/0.15/0.30 | **Contradicted** |
| Literature-backed weights | Code says "heuristic, NOT empirically calibrated" | **Contradicted** |
| DRACH motif | Implementation matches | **Verified** |
| ViennaRNA fallback | Implementation exists with warnings | **Verified** |
| Pareto selection | Custom dominance filter implemented | **Verified** |
| REINFORCE policy | Implemented with convergence notes | **Verified** |
| Separate TLR7/TLR8 | Implemented with distinct motifs | **Verified** |

---

## Required Revisions Summary

**Must Address for Acceptance:**
1. Correct manuscript weights to match implementation (TLR7=0.20, TLR8=0.15, PKR=0.30)
2. Replace "blunt-end detection" with "dsRNA structure potential estimation"
3. Acknowledge weights are heuristic, not empirically calibrated
4. Specify aggregation formula for overall immunogenicity score
5. Add statistical tests for validation comparisons

**Should Address (Recommended):**
1. Add kinetics formula parameters (RT=0.616 kcal/mol, k₀=10⁶ s⁻¹)
2. Clarify Pareto algorithm (custom dominance filter)
3. Add REINFORCE hyperparameters to supplement
4. Revise GC content range (0.50-1.00 → 0.50-0.85)
5. Add Zenodo DOI and dependency version constraints
6. Include hardware specifications for performance metrics

---

## Verdict

The Confluencia circRNA platform represents a valuable contribution to circRNA vaccine design, with sophisticated biological understanding and comprehensive analysis modules. However, the manuscript contains critical discrepancies between described methods and actual implementation, particularly regarding RIG-I algorithm and pathway weights. These must be corrected for scientific accuracy and reproducibility.

With minor revisions addressing the weight discrepancies, RIG-I algorithm description, and statistical rigor, the manuscript should be suitable for publication.

**Quality Score Breakdown:**
- Originality: 4.5/5
- Methodology: 3/5 (discrepancies reduce score)
- Technical Accuracy: 3/5 (implementation contradicts manuscript)
- Validation: 3.5/5
- Presentation: 4/5
- Reproducibility: 3/5 (missing parameters, version info)
- **Overall: 3.5/5**