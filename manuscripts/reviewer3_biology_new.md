# Reviewer #3: Biological Validity & Immunology

## Overall Assessment
- Recommendation: **Major Revisions**
- Biological Accuracy Score: 3/5

The manuscript presents an ambitious platform for circRNA vaccine design, but several critical biological considerations regarding circRNA-specific immunology are inadequately addressed or potentially misrepresented.

---

## Major Comments

### 1. RIG-I Scoring Does Not Account for circRNA Circular Topology

**Critical Issue:** The manuscript claims RIG-I recognition prediction using "blunt-end detection and GU-rich content analysis" (line 34), but this fundamentally contradicts circRNA biology.

**Biological Problem:** RIG-I is a cytosolic pattern recognition receptor that specifically recognizes 5'-triphosphate ends (Schlee et al., 2009). By definition, circRNAs lack free 5' ends due to their covalently closed circular structure. The authors cite Schlee et al. (2009) for RIG-I weight=0.35, but this paper describes linear RNA recognition mechanisms that are **not applicable** to circRNAs.

**Required Revision:**
- Explicitly acknowledge that native, intact circRNAs should NOT trigger RIG-I due to circular topology
- If scoring RIG-I, clarify this is for **potential circRNA degradation products** or **incomplete circularization products**, not the circRNA itself
- Re-evaluate the 0.35 weight - this seems inappropriately high given circRNA design aims to avoid 5' ends
- Consider adding a "circularization integrity" metric instead

### 2. TLR7/8 Mechanism Lacks Endosomal Localization Context

**Issue:** TLR7/8 activation scoring (weights=0.25/0.20) based on "U-rich and GU-rich motifs" (line 34) oversimplifies TLR biology.

**Biological Problem:**
- TLR7/8 are **endosomal receptors**, not cytosolic - they require RNA internalization via endocytosis
- circRNA cellular localization (nuclear vs. cytoplasmic) affects TLR7/8 access
- Endosomal modification (e.g., endosomal maturation state, pH) affects TLR activation kinetics
- U-rich motifs alone are insufficient - RNA secondary structure influences TLR7/8 recognition (Forsbach et al., 2008 actually shows this)

**Required Revision:**
- Discuss how delivery method (lipid nanoparticle, electroporation, exosome) affects endosomal TLR7/8 access
- Incorporate RNA secondary structure into TLR7/8 scoring, not just sequence motifs
- Discuss cell-type specificity (pDCs express high TLR7/8; most somatic cells do not)

### 3. m6A-Immunogenicity Link Oversimplified

**Issue:** The manuscript states "m6A site prediction with immunogenicity modulation effect" (line 56) without adequate nuance.

**Biological Complexity Missing:**
- m6A can **either enhance or suppress** immunogenicity depending on context:
  - m6A can promote immune evasion by reducing RIG-I recognition (in linear RNAs) - NOT applicable to circRNA
  - m6A can recruit YTHDC1/YTHDF proteins which have immune-modulatory functions
  - m6A can affect circRNA translation efficiency, indirectly affecting antigen presentation
- The claim that "AU-rich sequences contained more predicted m6A sites... potentially reducing immunogenicity" (lines 67-68) is speculative without mechanistic support
- DRACH motif alone is **insufficient** for functional m6A prediction - epitranscriptomic context (writer/eraser/reader presence) matters

**Required Revision:**
- Provide evidence that m6A modification reduces circRNA immunogenicity (this is debated in literature)
- Clarify mechanism: is this through YTH domain protein recruitment? Translation modulation? Stability effects?
- Acknowledge that m6A's immune effects may be context-dependent

### 4. Clinical Prediction: IPS/TIDE Not Validated for circRNA Vaccine Recipients

**Critical Issue:** The manuscript claims clinical prediction using "IPS (Immunotherapy Potential Score) and TIDE integration" (line 42) for vaccine outcome prediction.

**Biological Problem:**
- IPS (Cristescu et al., 2018) was developed for **cancer immunotherapy patients receiving checkpoint inhibitors**, not vaccine recipients
- TIDE (Jiang et al., 2018) predicts response to anti-PD1/CTLA4 therapy using tumor-infiltrating lymphocyte profiles
- Neither tool has been validated for circRNA vaccine applications
- Applying these scores to circRNA vaccine design is a **category error** - these predict patient response to therapy, not vaccine immunogenicity

**Required Revision:**
- Either remove IPS/TIDE claims or explicitly state this is a **hypothesis-generating extrapolation**, not validated prediction
- Consider more relevant clinical endpoints: local reactogenicity, systemic adverse events, neutralizing antibody titers
- If keeping IPS/TIDE, validate against actual circRNA vaccine clinical data (if available)

### 5. GC-Immunogenicity Correlation Lacks Mechanistic Basis

**Issue:** "Strong correlation between GC content and overall immunogenicity (r=0.85)" (line 65) is presented without proper mechanistic explanation.

**Problems:**
- The correlation may reflect dsRNA formation potential, but this should be tested directly via structure prediction
- High GC does not automatically mean dsRNA - single-stranded GC-rich regions exist
- GC content affects circRNA biogenesis efficiency, which confounds immunogenicity interpretation

---

## Mechanism Assessment

### RIG-I Pathway (Score: 2/5 - Major Concerns)
| Aspect | Manuscript Claim | Biological Reality | Issue |
|--------|------------------|---------------------|-------|
| 5' end recognition | "Blunt-end detection" | circRNAs have no 5' end | Fundamental contradiction |
| GU-rich content | "GU-rich content analysis" | GU-content affects other PRRs | Misattributed mechanism |
| Weight | 0.35 | Should be ~0 for intact circRNA | Inappropriate emphasis |

### TLR7/8 Pathway (Score: 3/5 - Moderate Concerns)
| Aspect | Manuscript Claim | Biological Reality | Issue |
|--------|------------------|---------------------|-------|
| Localization | Not addressed | Endosomal only | Missing context |
| Structure role | Sequence only | Structure affects binding | Oversimplified |
| Cell specificity | Not addressed | pDCs, B cells express TLR7/8 | Missing context |

### PKR Pathway (Score: 4/5 - Adequate)
| Aspect | Manuscript Claim | Biological Reality | Issue |
|--------|------------------|---------------------|-------|
| dsRNA detection | >33bp threshold | Correct (Nallagatla, 2007) | Appropriate |
| CircRNA relevance | dsRNA regions in circRNA | Valid - circRNA can form dsRNA | Correct |

### m6A Modification (Score: 2.5/5 - Incomplete)
| Aspect | Manuscript Claim | Biological Reality | Issue |
|--------|------------------|---------------------|-------|
| DRACH motif | Primary prediction method | Necessary but not sufficient | Missing context |
| Immune effect | "Immunogenicity modulation" | Context-dependent, bidirectional | Oversimplified |
| circRNA-specific | Not distinguished | circRNA m6A may differ from mRNA | Not addressed |

---

## Clinical Relevance

### Validity of Clinical Predictions (Score: 2/5)

**Concerns:**
1. **IPS/TIDE Misapplication:** These are cancer immunotherapy response predictors, not vaccine immunogenicity predictors. Using them for circRNA vaccine design is methodologically inappropriate.

2. **Missing Relevant Endpoints:**
   - Local reactogenicity (injection site pain, erythema)
   - Systemic adverse events (fever, fatigue, myalgia)
   - Humoral response (neutralizing antibody titers)
   - Cellular response (T cell activation markers)
   - Durability of response

3. **Population Mismatch:** Cristescu et al. (2018) and Jiang et al. (2018) datasets are from cancer patients. circRNA vaccine recipients may be healthy individuals (prophylactic) or have different baseline immune status.

4. **No circRNA-Specific Validation:** The manuscript provides no validation of clinical predictions against actual circRNA vaccine clinical data.

**Recommended Addition:**
- Include a "Clinical Validation Gap" section acknowledging the extrapolation from cancer immunotherapy to circRNA vaccines
- Partner with experimental groups to validate predictions in vitro/in vivo

---

## Strengths

1. **Comprehensive Integration:** The platform commendably integrates multiple analysis modules (structure, modifications, immunogenicity) into a unified framework.

2. **Open-Source Availability:** MIT license and Python package availability enhance reproducibility and community adoption.

3. **PKR Modeling:** The dsRNA length threshold (>33bp) and secondary structure linkage to PKR activation is well-grounded in literature (Nallagatla et al., 2007).

4. **Pareto Optimization:** Multi-objective optimization for stability, translation, and immune evasion is methodologically sound.

5. **ViennaRNA Integration:** Leveraging established structure prediction tools is appropriate.

6. **Performance Benchmarks:** Quantitative timing information (<100ms for immunogenicity, ~2-3s full pipeline) is helpful for users.

---

## Minor Comments

1. **Line 19:** "inherent stability" - consider adding reference to circRNA half-life comparisons with linear mRNA

2. **Line 39:** "Kinetics prediction estimates folding rate (k = exp(-barrier/RT))" - specify temperature (presumably 37°C)

3. **Line 40:** DRACH motif definition uses IUPAC codes but should specify circRNA-specific validation

4. **Line 65:** "r=0.85" correlation - report p-value and confidence interval

5. **Table 1:** "circRNA-specific" row should note that RIG-I scoring is NOT circRNA-specific (this is a contradiction)

---

## Summary of Required Revisions

### Must Address (Major)
1. Revise RIG-I scoring to acknowledge circRNAs lack 5' ends - current approach is biologically invalid
2. Add endosomal localization context for TLR7/8 predictions
3. Provide mechanistic basis for m6A-immunogenicity relationship or remove the claim
4. Remove or substantially revise IPS/TIDE clinical predictions with appropriate caveats

### Should Address (Moderate)
5. Add cell-type specificity to immune scoring (pDCs, macrophages, T cells)
6. Discuss circRNA delivery method effects on immune pathway activation
7. Include direct validation of GC-dsRNA-immunogenicity relationship

### Nice to Have (Minor)
8. Add circRNA-specific m6A reader/writer discussion
9. Include incomplete circularization detection methods
10. Add benchmark against experimental immunogenicity data

---

## Recommendation Summary

**Major Revisions Required** - The platform architecture and computational methods are sound, but several key biological claims are either inaccurate (RIG-I scoring for circRNA) or inadequately validated (IPS/TIDE for vaccine prediction, m6A-immunogenicity link). Addressing these issues is essential before the biological community can trust this tool for vaccine design applications.

**Revised Score Distribution:**
- Computational methods: 4/5
- RIG-I biology: 2/5
- TLR7/8 biology: 3/5
- m6A biology: 2.5/5
- Clinical prediction validity: 2/5
- Overall: 3/5

---

*Review completed: 2026-06-01*
*Reviewer #3 - Biological Validity & Immunology*
