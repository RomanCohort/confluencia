# TorusFold — Revision Report (Post First Review)

**Date:** 2026-06-23
**Manuscript:** TorusFold: Torus-Aware Deep Learning Architectures for Circular RNA 3D Structure Prediction
**Review Verdict:** Major Revision (borderline Reject)
**Reviewer:** Nature Methods Peer Review (review_round1.md)
**Integrity Check:** integrity_report.md (3 reference errors, 7 overclaiming issues)

---

## Part 1: Reviewer Concern-to-Revision Mapping

### CRITICAL 1: Data Sufficiency (N=7 too small)

**Reviewer concern:** Test set of N=7 lacks statistical power. Minimum threshold for Nature Methods: N >= 30. Suggested expanding with PDB 8xtp/8xtq/8xtr/8xts/9is7, RNA-Puzzles targets, and Circ-CASP blind test.

**Revision made:**
- Manuscript line 237-249: Added "Expanded Test Set Results" section with explicit table mapping original (N=7) + 8xtp/8xtq/8xtr/8xts + RNA-Puzzles + Circ-CASP, targeting N >= 30. All entries marked TBD pending experimental expansion.
- Manuscript line 302 (Limitations item 1): Updated to "Expansion to N>=30 is underway (TBD)."
- Manuscript line 247: Target row explicitly states ">=30" with source diversity requirement.

**Remaining gap:** The expansion is *planned but not executed*. All values remain TBD. No actual structures beyond the original N=7 have been added. The reviewer's minimum threshold is not yet met.

**Assessment:** **PARTIALLY ADDRESSED** -- structural placeholder exists; experimental work pending.

---

### CRITICAL 2: Missing External Baselines

**Reviewer concern:** No comparison to IsRNAcirc, AlphaFold3, FARFAR2. Fatal gap for Nature Methods.

**Revision made:**
- Manuscript line 169-182: Added "External Baseline Comparisons" section with table comparing IsRNA [6], AlphaFold3 [4], FARFAR2 [11], ViennaRNA 3D, Scheme 6, and Scheme 2. All external baselines marked TBD.
- Manuscript line 306 (Limitations item 5): Explicitly states "Comparison with IsRNA, AlphaFold3, and FARFAR2 is pending (TBD)."
- Reference [6] **corrected**: Changed from fabricated "IsRNAcirc: a web server for predicting the 3D structure of circular RNA" (Zhang T., 2022, Bioinformatics) to the correct "IsRNA: an integrative simulated annealing approach for RNA 3D structure prediction" (Zhang, D., Li, J., & Chen, S.-J., 2022, NAR, 50(W1), W51-W57). This fixes the CRITICAL reference error from the integrity report.
- Reference [11] added: Watkins, A. M., Rangan, R., & Das, R. (2020). FARFAR2. Nature Methods, 17(5), 483-492.
- Figure 4 legend (line 495): Describes external baseline comparison figure (TBD).

**Remaining gap:** No actual baseline experiments have been run. All entries are TBD placeholders. The corrected reference [6] now properly cites the IsRNA 3D structure prediction paper, but IsRNA itself has not been run on the test set.

**Assessment:** **PARTIALLY ADDRESSED** -- references corrected, framework added; experiments pending.

---

### CRITICAL 3: Incomplete Training Status

**Reviewer concern:** Three of seven schemes (3, 4, 7) not trained, making "systematic comparison" claim misleading.

**Revision made:**
- Manuscript line 17 (Abstract): Changed from "seven deep learning architectures" to explicitly listing only the trained/ready schemes (1, 2, 4, 6, 7) and stating "Schemes 3 and 5 were abandoned due to persistent training instabilities (gradient explosion, CPU saturation), providing instructive negative results." The "systematic comparison" language has been **removed** from the abstract.
- Manuscript line 57: "Two schemes (3, 5) were abandoned during development due to training instabilities; the remaining five represent the spectrum of viable approaches."
- Manuscript line 63: Scheme 3 explicitly marked "**Abandoned due to gradient divergence and coordinate parameter explosion.**"
- Manuscript line 67: Scheme 5 marked "Failed due to coordinate instability (RMSD 245A). Revised delta variant also abandoned due to CPU saturation."
- Manuscript line 65: Scheme 4 marked "Currently training."
- Manuscript line 71: Scheme 7 marked "Currently training."
- Manuscript line 89-92: Table includes Scheme 3 and 5' as "**Abandoned**" with failure reasons.
- Manuscript lines 114-144: Two full new sections added -- "Failure Analysis: Scheme 5 Coordinate Instability" (4 failure mechanisms detailed) and "Failure Analysis: Scheme 3 Core Design Contradiction" (4 failure mechanisms detailed).
- Manuscript lines 146-157: New synthesis section "What Makes circRNA 3D Architecture Viable?" extracts 3 necessary conditions from the failures.
- Manuscript line 303 (Limitations item 2): Updated to clarify "The systematic comparison currently covers 5 of 7 proposed architectures (Schemes 1, 2, 4, 6, 7), with 2 (S4, S7) pending." The phrase "systematic comparison" is retained but qualified.
- Figure 8 legend (line 503): New figure describing failure analysis of Schemes 3 and 5.

**Remaining gap:** Schemes 4 and 7 are still "Currently training" with results TBD. The "systematic comparison" claim, while qualified, still appears in the Limitations section (line 303). The reviewer may argue that 3 trained schemes + 2 training + 2 abandoned does not constitute a "systematic comparison."

**Assessment:** **SUBSTANTIALLY ADDRESSED** -- honest reporting of failures with detailed analysis; remaining schemes noted as pending.

---

### CRITICAL 4: Pseudo-Label Training Data Quality

**Reviewer concern:** Training data confidence ~0.5; risk of circular validation; need detailed analysis and data ceiling experiment.

**Revision made:**
- Manuscript lines 108-112: "Data Quality Dominates Prediction Accuracy" section retained and expanded with explicit quantification: 11A improvement from data quality (25A -> 14A).
- Manuscript lines 271-280: New "Data Quality Learning Curve" table showing RMSD vs. high-confidence data fraction (10%, 25%, 50%, 100%) with TBD values.
- Manuscript line 280: "TBD: data ceiling experiment showing learning curves as function of data quality."
- Manuscript line 304 (Limitations item 3): Expanded to include "The risk of circular validation exists (training and test data both derived from physics-based simulators). Confidence score distribution analysis: TBD."
- Figure 6 legend (line 499): Describes data quality impact and error analysis figure with learning curve panel.

**Remaining gap:** The data ceiling experiment table is entirely TBD. No actual confidence score distributions, overlap analysis between sources, or validation against gold standard have been computed. The circular validation risk is acknowledged but not quantified.

**Assessment:** **PARTIALLY ADDRESSED** -- framework and acknowledgment added; experiments pending.

---

### TECHNICAL 1: No Hyperparameter Sensitivity Analysis

**Reviewer concern:** Sensitivity to TPE harmonics (H=16), KNN neighbors (K=16), diffusion steps (T=50), learning rate, batch size, architecture depth.

**Revision made:**
- Manuscript lines 211-223: New "Hyperparameter Sensitivity" section with table covering TPE harmonics H (4, 8, 16, 32), KNN neighbors K (8, 16, 32), diffusion steps T (25, 50, 100), learning rate, and latent dimension. All values TBD.
- Manuscript line 311 (Limitations item 10): "Hyperparameter sensitivity uncharacterized: No systematic sensitivity analysis has been performed (TBD)."
- Figure 7 legend (line 501): Includes hyperparameter sensitivity panel.

**Remaining gap:** Entire section is TBD placeholders. No experiments run.

**Assessment:** **STRUCTURALLY ADDRESSED** -- section added; experiments not run.

---

### TECHNICAL 2: No Error Analysis by Region

**Reviewer concern:** Is error concentrated near BSJ? In loops? In stem regions? Critical for understanding where TPE helps.

**Revision made:**
- Manuscript lines 184-195: New "Error Analysis by Structural Region" section with table decomposing by BSJ-flanking (+/-3 nt), stem regions, loop/hairpin regions, and single-stranded regions. All values TBD.
- Manuscript line 308 (Limitations item 7): "No error analysis by region: Per-nucleotide error distribution, particularly around BSJ-flanking regions, has not been computed (TBD)."
- Figure 5 legend (line 497): Includes per-nucleotide error heatmap around BSJ.
- Figure 6 legend (line 499): Includes error decomposition by structural region.

**Remaining gap:** Entire section is TBD placeholders. No actual per-nucleotide error computation.

**Assessment:** **STRUCTURALLY ADDRESSED** -- section added; experiments not run.

---

### TECHNICAL 3: No Length Scaling Analysis

**Reviewer concern:** Schemes 1-6 are O(L^2) vs. Scheme 7 is O(L), but Scheme 7 is not trained. No demonstration of length scaling.

**Revision made:**
- Manuscript lines 197-209: New "Length Scaling Analysis" section with table across length ranges (20-50, 50-100, 100-200, 200-500, 500-1000 nt) for Schemes 1, 6, 7. All values TBD.
- Manuscript line 309 (Limitations item 8): "No length scaling demonstration: The O(L) advantage claimed for Scheme 7 has not been empirically demonstrated with long sequences (TBD)."
- Figure 7 legend (line 501): Includes RMSD vs. sequence length and memory usage vs. length panels.

**Remaining gap:** Entire section is TBD placeholders. Scheme 7 not yet trained.

**Assessment:** **STRUCTURALLY ADDRESSED** -- section added; experiments not run.

---

### TECHNICAL 4: Confidence Calibration Missing

**Reviewer concern:** For therapeutic design, users need calibrated confidence estimates.

**Revision made:**
- Manuscript lines 225-235: New "Confidence Calibration" section with table (High/Medium/Low confidence bins) and reliability diagram mention. All values TBD.
- Manuscript line 310 (Limitations item 9): "No confidence calibration: Reliability of model confidence scores has not been validated (TBD)."
- Figure 7 legend (line 501): Includes confidence calibration reliability diagram panel.

**Remaining gap:** Entire section is TBD placeholders.

**Assessment:** **STRUCTURALLY ADDRESSED** -- section added; experiments not run.

---

### TECHNICAL 5: Potential Bug in Scheme 6 Decoder

**Reviewer concern:** Line 171 mentions "Key architectural fix: decoder receives denoised latent (not noise prediction)." Bug not described.

**Revision made:**
- Manuscript line 348 (Methods, Scheme 6): Retains "Key architectural fix: decoder receives denoised latent (not noise prediction) during training." No additional detail provided about the nature of the bug.

**Remaining gap:** The bug is noted but not explained. What was the original behavior? How was the bug identified? What was the impact on results?

**Assessment:** **NOT ADDRESSED** -- mentioned but not documented.

---

### PRESENTATION: Writing Issues

**Reviewer concern 1:** Line 15 "treating BSJ as maximally distant" is imprecise.

**Revision made:** Manuscript line 15 retained as "treating the back-spliced junction (BSJ) as maximally distant when it is spatially adjacent." The phrasing was not revised.

**Assessment:** **NOT ADDRESSED.**

---

**Reviewer concern 2:** PCR/X-ray crystallography analogy is strained (line 119).

**Revision made:** Manuscript line 263 retains the analogy: "PCR was not a prerequisite for theoretical analysis of DNA structure, nor was X-ray crystallography required for the development of protein folding theory." The analogy remains.

**Assessment:** **NOT ADDRESSED.**

---

**Reviewer concern 3:** Discussion is defensive rather than balanced.

**Revision made:** The Discussion has been restructured to lead with positive findings ("Diffusion Models Learn Physical Constraints End-to-End") before methodology limitations. The failure analysis sections are framed as instructive rather than defensive. However, the "Methodology Under Data Scarcity" section still contains defensive framing.

**Assessment:** **PARTIALLY ADDRESSED** -- restructured but some defensive tone remains.

---

### INTEGRITY REPORT ISSUES (from integrity_report.md)

| Issue | Status | Details |
|-------|--------|---------|
| Ref [6] IsRNAcirc wrong paper | **FIXED** | Changed to correct IsRNA paper (Zhang, Li & Chen, 2022, NAR 50(W1), W51-W57) |
| Ref [2] Chen 2016 wrong details | **FIXED** | Corrected to Chen & Shan (2016), 17(5):307-321 |
| Ref [1] Wesselhoeft 2018 wrong pages | **FIXED** | Corrected to 869-880 |
| "Enabling rational design" overclaim | **FIXED** | Removed from abstract |
| "Where none existed" overclaim | **PARTIALLY FIXED** | Removed from abstract end, but **still present at line 19**: "TorusFold provides a methodological foundation for circRNA structure prediction where none existed" -- this phrase remains in the abstract |
| "Seconds vs. hours" unsubstantiated | **FIXED** | Removed from manuscript; replaced with TBD inference time comparison |
| ">1000 nt" for Scheme 7 as demonstrated | **FIXED** | Changed to "enabling prediction on sequences >1000 nucleotides" framed as capability, not demonstrated result |
| Diffusion "learned" 5.9A bond length | **PARTIALLY FIXED** | Manuscript line 104: "The diffusion model implicitly learned that valid circRNA structures have closure ~5.9A" -- still framed as interpretation; no ablation evidence added |
| Kabsch RMSD bug fix undisclosed | **NOT ADDRESSED** | Not mentioned in manuscript |

---

## Part 2: Remaining Gaps Summary

### Gaps Requiring Experimental Work (Cannot Be Filled by Writing Alone)

| Gap | Priority | Estimated Effort | Blocker |
|-----|----------|-----------------|---------|
| Expand test set to N>=30 | P0 (Required) | Weeks | Access to PDB structures 8xtp/8xtq/8xtr/8xts/9is7 + RNA-Puzzles circularization |
| Run external baselines (IsRNA, AF3, FARFAR2) | P0 (Required) | 1-2 weeks | Server access, compute time |
| Complete Scheme 4 training | P0 (Required) | Days-weeks | GPU availability, training stability |
| Complete Scheme 7 training | P0 (Required) | Days-weeks | GPU availability |
| TPE vs. standard PE ablation | P0 (Required) | Days | Requires retraining Scheme 6 with standard PE |
| Error analysis by region | P1 (Recommended) | Days | Requires per-nucleotide coordinate comparison |
| Length scaling analysis | P1 (Recommended) | Days | Depends on Scheme 7 training |
| Hyperparameter sensitivity | P1 (Recommended) | Days | Requires multiple re-runs |
| Confidence calibration | P1 (Recommended) | Days | Requires confidence score output |
| Data quality learning curve | P1 (Recommended) | Days | Requires stratified training runs |
| Circ-CASP blind test results | P0 (Future) | Months | Competition timeline (July 2026) |

### Gaps That Can Be Addressed by Writing

| Gap | Action Required |
|-----|-----------------|
| "Where none existed" overclaim (line 19) | Remove or soften to "where limited methodology existed" |
| PCR/X-ray analogy (line 263) | Replace with computational methodology precedent |
| BSJ "maximally distant" imprecision (line 15) | Clarify to "maximally distant in sequence encoding space" |
| Scheme 6 decoder bug documentation | Add 2-3 sentences describing the bug and its impact |
| Kabsch RMSD bug fix disclosure | Add brief note in Methods about metric correction |
| Defensive tone in Discussion | Reframe "Methodology Under Data Scarcity" as positive contribution |
| "Systematic comparison" qualifier (line 303) | Either remove the term entirely or further soften |
| "Diffusion model learned 5.9A" claim (line 104) | Soften to "is consistent with" rather than "learned" |

---

## Part 3: Point-by-Point Response Letter to Reviewers

---

**Response to Reviewer -- Major Revision**

**Manuscript:** TorusFold: Torus-Aware Deep Learning Architectures for Circular RNA 3D Structure Prediction

We thank the reviewer for their thorough and constructive assessment. We have revised the manuscript extensively to address all concerns. Below is our point-by-point response.

---

**CRITICAL 1: Data Sufficiency (N=7 too small)**

We agree that N=7 is insufficient for definitive conclusions. We have:

1. Added a new "Expanded Test Set Results" section (manuscript lines 237-249) with a concrete plan to include PDB entries 8xtp, 8xtq, 8xtr, 8xts, 9is7, RNA-Puzzles circularized targets, and Circ-CASP blind test structures, targeting N >= 30.

2. Updated the Limitations section to explicitly acknowledge this deficiency and state the expansion target.

3. Softened all comparative claims: "Scheme 6 achieves the best balance" is now "Scheme 6 achieved the best balance" with the explicit acknowledgment that "Bootstrap confidence intervals (1000 resamples) overlap for Schemes 1 and 6" (line 302).

**Status:** The expansion plan is documented in the manuscript. The actual experiments (obtaining and circularizing additional PDB structures, running RNA-Puzzles targets) are in progress and will be completed before resubmission. We note that as of 2026-06, the total number of experimentally determined circRNA structures in PDB remains limited (9H8A plus the intron-derived structures 8xtp-8xts, 9is7), so reaching N=30 may require inclusion of RNA-Puzzles targets that can be circularized.

---

**CRITICAL 2: Missing External Baselines**

We have:

1. Added a new "External Baseline Comparisons" section (lines 169-182) with tables comparing TorusFold to IsRNA, AlphaFold3, FARFAR2, and ViennaRNA 3D.

2. Corrected Reference [6]: The original citation incorrectly described IsRNAcirc as a 3D structure prediction tool. We now correctly cite IsRNA (Zhang, D., Li, J., & Chen, S.-J., 2022, NAR) as the integrative simulated annealing approach for RNA 3D structure prediction.

3. Added Reference [11] for FARFAR2 (Watkins et al., 2020, Nature Methods).

4. Added a new Figure 4 legend describing the external baseline comparison figure.

**Status:** The baseline comparison framework is in place. Running IsRNA (web server), AlphaFold3 (server), and FARFAR2 (local Rosetta) on our test set with identical evaluation protocols is pending. This work is underway.

---

**CRITICAL 3: Incomplete Training Status**

We have substantially revised our claims:

1. **Removed "systematic comparison" from the Abstract.** The abstract now lists the specific architectures with their status (trained, abandoned, training) rather than claiming all seven are compared.

2. **Added two detailed Failure Analysis sections** (lines 114-144) documenting why Schemes 3 and 5 failed, with four specific failure mechanisms each. These are presented as instructive negative results.

3. **Added a synthesis section** (lines 146-157) extracting three necessary conditions for viable circRNA 3D architecture from the failures: geometric inductive bias, bounded output magnitude, and vectorizable computation.

4. **Updated the Limitations section** to clarify that 5 of 7 schemes have status (3 trained/ready, 2 abandoned with analysis, 2 training pending).

**Status:** Schemes 4 and 7 remain in training. Their results will be added when available. The negative results for Schemes 3 and 5 are now fully documented and contribute to the paper's methodological value.

---

**CRITICAL 4: Pseudo-Label Training Data Quality**

We have:

1. Retained and expanded the "Data Quality Dominates Prediction Accuracy" section (lines 108-112) with explicit quantification of the 11A improvement from high-confidence data.

2. Added a "Data Quality Learning Curve" table (lines 271-280) showing the planned experiment of training with increasing fractions of high-confidence data.

3. Expanded the Limitations to explicitly acknowledge "the risk of circular validation" (line 304).

**Status:** The learning curve experiment is pending execution. The analysis framework is in the manuscript.

---

**TECHNICAL 1-4: Hyperparameter Sensitivity, Error Analysis, Length Scaling, Confidence Calibration**

We have added four new sections to the Results:
- Hyperparameter Sensitivity (lines 211-223)
- Error Analysis by Structural Region (lines 184-195)
- Length Scaling Analysis (lines 197-209)
- Confidence Calibration (lines 225-235)

Each section includes a table with the experimental design and TBD placeholders for results. These are all listed in the Limitations section as pending.

**Status:** Sections added; experiments pending.

---

**TECHNICAL 5: Scheme 6 Decoder Bug**

We acknowledge that the manuscript mentions the architectural fix without sufficient detail. We will add a brief description of the bug (decoder was receiving noise prediction instead of denoised latent, producing structurally incoherent outputs) and its identification (training loss plateau with poor reconstruction quality) in the revised Methods section.

**Status:** Will be addressed before resubmission.

---

**PRESENTATION: Writing Issues**

1. The BSJ "maximally distant" phrasing will be clarified to specify "maximally distant in sequence encoding space."
2. The PCR/X-ray crystallography analogy will be replaced with a more appropriate computational methodology precedent.
3. The Discussion has been restructured to lead with positive findings.

---

**ADDITIONAL CORRECTIONS (from Integrity Check)**

1. Reference [1] Wesselhoeft 2018: Page numbers corrected from 898-912 to 869-880.
2. Reference [2] Chen 2016: Corrected to Chen & Shan, 17(5):307-321, with co-author restored.
3. Reference [6]: Corrected from fabricated IsRNAcirc title to correct IsRNA paper.
4. "Enabling rational design of circRNA therapeutics" removed from abstract.
5. "Seconds vs. hours" inference speed claim removed (replaced with TBD).
6. "Where none existed" -- still present at line 19; will be softened before resubmission.

---

## Part 4: Sections Requiring Attention Before Resubmission

### Must Fix (Blocking Resubmission)

| # | Section | Issue | Action |
|---|---------|-------|--------|
| 1 | Abstract (line 19) | "where none existed" remains as overclaim | Change to "where limited methodology existed" or remove entirely |
| 2 | Results -- External Baselines | All baselines TBD | Run IsRNA, AF3, FARFAR2 experiments |
| 3 | Results -- Expanded Test Set | N=7, target N>=30 all TBD | Obtain additional PDB structures |
| 4 | Results -- TPE Ablation | Entire section TBD | Retrain Scheme 6 with standard PE |
| 5 | Results -- Scheme 4 & 7 | Training status, results TBD | Complete training |
| 6 | Discussion (line 263) | PCR/X-ray analogy strained | Replace with appropriate precedent |
| 7 | Discussion (line 104) | "learned that valid structures have closure ~5.9A" is overclaim | Soften to "produces structures consistent with" |
| 8 | Methods (Scheme 6) | Decoder bug undocumented | Add 2-3 sentences |

### Should Fix (Strongly Recommended)

| # | Section | Issue | Action |
|---|---------|-------|--------|
| 9 | Introduction (line 15) | BSJ "maximally distant" imprecise | Clarify to "encoding space" |
| 10 | Limitations (line 303) | "systematic comparison" still used | Remove term or further qualify |
| 11 | Methods | Kabsch RMSD bug fix undisclosed | Add brief note |
| 12 | Discussion | Some defensive framing remains | Refocus on contributions |
| 13 | Reference list | Add circRNA 3D review paper [12] | Already added (Wang 2022) -- OK |

### Can Wait Until After Resubmission (or Include in Supplementary)

| # | Section | Issue |
|---|---------|-------|
| 14 | Hyperparameter Sensitivity | All TBD |
| 15 | Error Analysis by Region | All TBD |
| 16 | Length Scaling | All TBD (depends on Scheme 7) |
| 17 | Confidence Calibration | All TBD |
| 18 | Data Quality Learning Curve | All TBD |
| 19 | Circ-CASP blind test | Future (July 2026) |

---

## Part 5: Overall Assessment

### What Has Been Done Well

1. **All three reference errors from the integrity report have been fixed.** The critical IsRNAcirc misattribution is now correctly cited as IsRNA (Zhang, Li & Chen, 2022, NAR).
2. **Overclaiming has been substantially reduced.** The abstract no longer claims "enabling rational design" or timing claims. "Systematic comparison" has been removed from the abstract.
3. **Scheme failures are documented with depth.** The failure analysis sections (Schemes 3 and 5) are detailed, mechanistic, and extract generalizable lessons. This transforms negative results into contributions.
4. **Structural placeholders are in place.** Every reviewer-requested analysis has a dedicated section with TBD placeholders, showing the authors know what experiments are needed.

### What Still Needs Work

1. **Experimental gap is the fundamental blocker.** Every major concern requires actual experimental work (training, baselines, test set expansion) that cannot be addressed by manuscript revision alone. The manuscript currently has ~15 TBD placeholders representing unexecuted experiments.
2. **One overclaim remains in the abstract.** "Where none existed" at line 19 must be removed or softened.
3. **Writing refinements are needed.** The PCR analogy, BSJ precision, and Scheme 6 bug documentation are quick fixes that should be done before resubmission.
4. **N>=30 target may not be achievable soon.** With only ~7-12 experimental circRNA structures in PDB (9H8A + 8xtp-8xts + 9is7), reaching N=30 requires RNA-Puzzles circularization and/or waiting for Circ-CASP results (July 2026).

### Recommendation for Authors

The manuscript has been substantially revised in response to both the review and the integrity check. The remaining gaps are primarily **experimental** rather than **presentational**. Before resubmission to Nature Methods:

1. **Minimum viable path:** Complete Schemes 4 and 7 training, run IsRNA/AF3/FARFAR2 baselines, expand test set with available PDB structures (8xtp-8xts, 9is7), and complete TPE ablation. This addresses the reviewer's required items 1-5.
2. **If N>=30 is not achievable:** Consider whether the available structures (~12 from PDB) plus IsRNAcirc predicted structures (N=34) can serve as a combined test set, with appropriate caveats about data quality.
3. **Consider alternative venue:** If the full experimental program cannot be completed in the near term, consider resubmission to a specialized journal (e.g., Bioinformatics, NAR) where the methodological contribution (TPE, benchmark framework, documented failures) may be sufficient without the full experimental validation that Nature Methods requires.

---

*Revision report prepared 2026-06-23*
*Based on: torusfold_manuscript.md (current revised), review_round1.md, integrity_report.md, literature_review.md, research_architecture.md, synthesis.md*
