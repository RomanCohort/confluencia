# Peer Review Report: Confluencia circRNA Platform

## Reviewer #3: Immunology and Innate Immune Sensing

**Manuscript:** Confluencia circRNA: A Comprehensive Platform for circRNA Vaccine Design and Immunogenicity Prediction

**Date:** 2026-06-01

---

## Overall Assessment

**Recommendation:** Major Revisions Required

This manuscript presents an ambitious computational platform for circRNA vaccine design. While the integration of multiple analysis modules is commendable, the biological foundation for key immunogenicity predictions contains significant inaccuracies that undermine the platform's validity for circRNA-specific applications. The central issue is that several mechanisms described were characterized for linear RNA and may not translate directly to circular topologies.

---

## Biological Accuracy Score: 2.5/5

The platform demonstrates broad scope but contains fundamental mechanistic oversimplifications that require correction before publication.

---

## Major Comments

### 1. RIG-I Mechanism is Fundamentally Misapplied to circRNA

**Critical Issue:** The manuscript states that RIG-I recognition is predicted using "blunt-end detection and GU-rich content analysis" (line 34). This reveals a fundamental misunderstanding of circRNA biology in the context of innate immunity.

RIG-I recognizes:
- 5'-triphosphate or 5'-diphosphate ends (Schlee et al., 2009)
- Blunt-end dsRNA with 5'-triphosphate
- Short dsRNA regions with free 5' ends

**CircRNAs are covalently closed loops lacking any 5' or 3' termini.** By definition, circRNAs cannot present the molecular features RIG-I requires for activation.

**Questions for authors:**
- Does the platform actually evaluate circRNA sequences, or is it applying linear RNA scoring to circular sequences?
- If circRNA-specific, how does the algorithm detect "blunt ends" on a molecule with no ends?
- Has the platform been validated against experimental data showing circRNA activation (or lack thereof) of RIG-I?

**Required revision:** Either:
(a) Remove RIG-I scoring from circRNA-specific analysis and clarify it applies only to linear RNA contaminants or degradation products, OR
(b) Provide experimental evidence that circRNAs can activate RIG-I through alternative mechanisms, with corresponding algorithm adjustments.

**Suggested approach:** Consider that RIG-I activation in circRNA preparations typically reflects linear RNA contamination or incomplete circularization, not intrinsic circRNA immunogenicity. The platform could score "contamination risk" rather than direct RIG-I activation.

---

### 2. TLR7/8 Activation Scoring Lacks Critical Contextual Factors

The TLR7/8 scoring based on "U-rich and GU-rich motifs" (line 34) is incomplete for accurate prediction.

**Missing considerations:**

**(a) Endosomal Localization:** TLR7/8 are endosomal receptors. CircRNA must reach endolysosomal compartments for activation. The scoring does not account for:
- Cellular uptake mechanisms
- Endosomal escape efficiency
- Subcellular localization signals

**(b) RNA Modifications:** The manuscript mentions m6A but not other modifications critical for TLR7/8:
- Pseudouridine strongly reduces TLR activation (Anderson et al., 2011; Karikó et al., 2005)
- 2'-O-methylation abrogates TLR7/8 signaling
- N1-methylpseudouridine (used in mRNA vaccines) nearly eliminates TLR activation

**Do the TLR7/8 scores account for:**
- Presence of modified nucleosides in the sequence?
- Secondary structure accessibility (TLRs require single-stranded regions)?
- GU-rich motif accessibility within folded structures?

**Required revision:** Include discussion of RNA modification effects on TLR7/8 activation and clarify whether scoring applies to unmodified RNA only.

---

### 3. PKR Activation Threshold Oversimplified

The >33bp threshold (Nallagatla et al., 2007) is correctly cited but the implementation appears simplistic.

**Complexities not addressed:**
- **Partial dsRNA:** PKR can be activated by imperfect duplexes; bulges and internal loops affect binding affinity
- **PKR dimerization requirements:** dsRNA length affects dimerization efficiency; 33bp is minimum but activation strength increases with length
- **Structural context:** Hairpin stems, not just linear dsRNA, can activate PKR
- **PKR inhibitors:** Cellular PKR inhibitors (PACT, TRBP regulation) modulate response

**Questions for authors:**
- How does the algorithm handle bulged duplexes and internal loops?
- Does dsRNA region detection account for hairpin structures?
- Is PKR activation scored as binary (>33bp threshold) or continuous (activation strength)?

**Required revision:** Clarify PKR scoring methodology and address structural complexity in dsRNA detection.

---

### 4. m6A-Immunogenicity Relationship Oversimplified

**Problematic statement (line 65-67):**
"m6A sites...potentially reducing immunogenicity through modification-mediated immune evasion"

This is an oversimplification that ignores context-dependent effects:

**m6A can ENHANCE immune responses:**
- YTHDF2-mediated degradation produces RNA fragments that activate RIG-I (but this requires linear RNA degradation products)
- m6A can increase RNA stability in certain contexts, prolonging immune stimulation
- m6A-modified RNAs can still activate PKR

**m6A can SUPPRESS immune responses:**
- Reduced RIG-I recognition when m6A near 5' end (not applicable to circRNA)
- YTHDF-mediated sequestration may reduce availability to sensors
- "Don't eat me" signal hypothesis for m6A

**Required revision:**
- Provide nuanced discussion of m6A-immunogenicity relationships
- Cite contradictory findings (e.g., recent work showing m6A can enhance immunogenicity in certain contexts)
- Clarify that the direction of m6A effect is context-dependent

---

### 5. Clinical Prediction Tools Misapplied

**Critical concern:** The manuscript uses IPS (Immunotherapy Potential Score) and TIDE (Tumor Immune Dysfunction and Exclusion) for clinical prediction (line 41-42). These tools were developed for and validated on:
- Cancer patients receiving immune checkpoint inhibitors
- Tumor-infiltrating lymphocyte characterization
- Prediction of anti-PD-1/anti-CTLA-4 response

**Application to circRNA vaccine recipients is problematic:**
- Healthy vaccine recipients have fundamentally different immune landscapes than cancer patients
- IPS/TIDE were not designed for prophylactic vaccine contexts
- No validation data exists for applying these scores to vaccine design

**Questions for authors:**
- What validation supports using IPS/TIDE for circRNA vaccine prediction?
- Have these scores been tested in any vaccine context?
- How should clinicians interpret "survival prediction" for healthy vaccine recipients?

**Required revision:**
- Either remove clinical prediction for non-therapeutic vaccine contexts, or
- Provide clear disclaimers about applicability limitations
- Consider whether these tools are appropriate for the stated use case

---

## Minor Comments

1. **Line 19:** "capacity to evade innate immune detection when properly designed" - This statement implies circRNAs are inherently less immunogenic, which contradicts the platform's purpose of predicting immunogenicity. Clarify the baseline immunogenicity of circRNAs.

2. **Line 65-66:** "Strong correlation between GC content and overall immunogenicity (r=0.85)" - This correlation may be driven primarily by PKR activation through dsRNA formation. Consider whether this reflects true immunogenicity or an artifact of the scoring algorithm design.

3. **Line 68-69:** "high-immunogenicity sequence (0.88) suitable for vaccine delivery" - High immunogenicity is not always desirable for vaccines. Some vaccines (e.g., protein subunit) may benefit from low innate immune activation. Clarify the intended vaccine modality.

4. **Table 1:** The comparison table positions this platform as uniquely providing "circRNA-specific" analysis, but the mechanistic issues identified above question whether the immunogenicity scoring is truly circRNA-specific.

5. **References:** Consider adding:
   - Chen et al. (2019) on circRNA immune sensing
   - Wesselhoeft et al. (2018) methodology papers on circRNA translation
   - Recent circRNA vaccine literature (post-2020)

---

## Mechanism Assessment Summary

| Mechanism | Valid for circRNA? | Issues |
|-----------|-------------------|--------|
| **RIG-I** | **No** | circRNAs lack 5' ends; cannot present required molecular features |
| **TLR7/8** | **Partially** | Missing modification effects, localization, accessibility |
| **PKR** | **Yes** | Threshold oversimplified; structural complexity not addressed |
| **m6A effects** | **Context-dependent** | Relationship oversimplified; bidirectional effects ignored |

---

## Clinical Relevance Assessment

**Low to Moderate**

The clinical prediction modules (IPS/TIDE integration) are not validated for vaccine applications. The survival analysis framework is designed for cancer immunotherapy contexts and its applicability to prophylactic vaccines is untested. The authors should:

1. Clearly state limitations of clinical predictions
2. Validate predictions against experimental circRNA vaccine data if available
3. Consider removing or repositioning clinical modules as "exploratory" rather than validated predictions

---

## Strengths

1. **Comprehensive scope:** Integration of multiple analysis modules (structure, modifications, immunogenicity, clinical) is valuable

2. **Open-source availability:** MIT license and Python package facilitate community adoption and validation

3. **Performance metrics:** Reported computation times (<100ms for immunogenicity) are practical for interactive use

4. **Evolutionary optimization:** Pareto multi-objective optimization is methodologically sound for sequence design

5. **Structure-kinetics integration:** Linking dsRNA regions to PKR activation potential is conceptually appropriate

6. **Validation attempt:** circBase analysis demonstrates practical application, though validation against experimental immunogenicity data would strengthen the manuscript

---

## Recommendation Summary

The manuscript requires major revisions to address fundamental mechanistic issues, particularly the RIG-I scoring inapplicability to circRNA and the oversimplified m6A-immunogenicity relationship. The clinical prediction module requires either validation or removal/repositioning.

**Specific action items:**
1. Revise or remove RIG-I scoring for circRNA applications
2. Expand TLR7/8 scoring to include modifications and accessibility
3. Clarify PKR scoring methodology for complex structures
4. Provide nuanced discussion of m6A effects
5. Address clinical prediction tool applicability limitations

After revisions addressing these concerns, the manuscript could make a valuable contribution to circRNA vaccine design tools.

---

## Confidential Notes to Editor

The platform is ambitious and well-intentioned, but the immunological foundations need correction before publication. The RIG-I issue is most concerning - it suggests the authors may have applied linear RNA knowledge to circular RNAs without considering topological differences. This could mislead users into thinking their circRNAs will activate RIG-I when this is unlikely. I encourage resubmission after addressing these mechanistic concerns.

---

*Review completed: 2026-06-01*