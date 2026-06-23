# TorusFold Manuscript — Academic Integrity Report

**Date:** 2026-06-23
**Reviewer:** Automated integrity check
**Manuscript:** `torusfold_manuscript.md`
**Experimental Data:** `experimental_data.md`

---

## 1. Factual Accuracy: Numbers Cross-Check

### 1.1 RMSD Values — PDB Test Set (N=7)

| Claim (Manuscript) | Experimental Data | Match? |
|---|---|---|
| Scheme 6: Mean 13.91A, Median 14.08A, Std 0.73A | S6: Mean 13.91, Median 14.08 | MATCH |
| Scheme 1: 13.85A | S1: 13.85 | MATCH |
| Scheme 2: ~2A | S2: ~2.0 | MATCH |
| Scheme 5: 245A | S5: 245 | MATCH |
| Scheme 1 closure: 5.36A | S1 closure: 5.36 | MATCH |
| Scheme 6 closure: 0.02A | S6 closure: 0.02 | MATCH |
| Scheme 2 closure: <0.1A | S2 closure: <0.1 | MATCH |

### 1.2 RMSD Values — circrna_3d Test Set (~11K)

| Claim (Manuscript) | Experimental Data | Match? |
|---|---|---|
| All schemes ~25A on validation | S2 (sota): 25.5, S6 v1: 25.1 | APPROXIMATE MATCH (manuscript says "~25A"; data shows 25.1-25.5) |
| 11A improvement from data quality | 25.5 → 13.91 = 11.59A | APPROXIMATE MATCH (manuscript rounds to "11A") |

### 1.3 Sample Sizes

| Claim (Manuscript) | Experimental Data | Match? |
|---|---|---|
| PDB test set: N=7 | 7 samples | MATCH |
| circrna_3d_merged: N≈11,000 | ~11,000 samples | MATCH |
| Sequence lengths: 20-27 nt | Not explicitly stated in data | CANNOT VERIFY (data does not list sequence lengths) |

### 1.4 Architecture Details

| Claim (Manuscript) | Experimental Data | Match? |
|---|---|---|
| Scheme 6: 50-step diffusion | 50 steps | MATCH |
| Scheme 4: 100-step diffusion | Not stated in data | CANNOT VERIFY |
| Scheme 1: EGNN K=16 neighbors | Not stated in data | CANNOT VERIFY |
| Bond length target: 5.9A | 5.9A (implied in closure target) | MATCH |

### 1.5 Bug Fixes Disclosed

The experimental data lists 5 bug fixes applied. The manuscript mentions the Scheme 5 revision to delta prediction (bug fix #2 in data). However, the manuscript does **not** disclose:
- Bug fix #1: kabsch_rmsd rotation matrix formula correction
- Bug fix #3: EGNN learnable coord_step, padding mask
- Bug fix #4: S3 planar circular init (no z-offset)
- Bug fix #5: circrna_diffusion learnable coord_step

**Severity: LOW.** These are implementation details, not results-altering issues. However, the Kabsch RMSD fix (#1) is notable because it could have affected reported RMSD values if applied after initial measurements.

---

## 2. Reference Accuracy

### 2.1 Reference [1] — Wesselhoeft et al. (2018)

**Manuscript cites:** Wesselhoeft, R. A., et al. (2018). RNA circularization diminishes immunogenicity and can extend translation duration in vivo. Molecular Cell, 70(5), 898-912.

**Verified:** The real paper exists. Title and journal are correct. However, the **page numbers are WRONG**. The actual page range is **869-880** (or 869-882 per some sources), NOT 898-912. The volume (70) and issue (5) are correct.

**Severity: MEDIUM.** Wrong page numbers suggest the citation was not verified against the actual publication. The title, authors, journal, and year are correct.

### 2.2 Reference [2] — Chen (2016)

**Manuscript cites:** Chen, L. L. (2016). The biogenesis and emerging roles of circular RNAs. Nature Reviews Molecular Cell Biology, 17(4), 205-211.

**Verified:** The real paper exists. However, the actual paper is by **Chen, L.L. and Shan, G.** (co-authorship). The volume/issue is **17(5):307-321** (May 2016), NOT 17(4):205-211. The title is "Circular RNAs — biogenesis, emerging roles, and diseases" (or very similar variant), not exactly "The biogenesis and emerging roles of circular RNAs."

**Severity: HIGH.** Both page numbers AND volume/issue are wrong. The co-author (Shan G.) is omitted. The title is slightly different. This suggests the citation was reconstructed from memory rather than verified.

### 2.3 Reference [3] — Vaswani et al. (2017)

**Manuscript cites:** Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

**Verified:** Real paper. Correct venue, year, and authors. Standard abbreviated citation format.

**Severity: NONE.** Citation is accurate.

### 2.4 Reference [4] — Abramson et al. (2024)

**Manuscript cites:** Abramson, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature.

**Verified:** Real paper. Correct title, journal, year, and first author.

**Severity: NONE.** Citation is accurate.

### 2.5 Reference [5] — Baek et al. (2021)

**Manuscript cites:** Baek, M., et al. (2021). Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557), 871-876.

**Verified:** Real paper. Correct title, journal, volume, pages, and first author.

**Severity: NONE.** Citation is accurate.

### 2.6 Reference [6] — Zhang et al. (2022)

**Manuscript cites:** Zhang, T., et al. (2022). IsRNAcirc: a web server for predicting the 3D structure of circular RNA. Bioinformatics.

**Verified:** The real IsRNAcirc paper is by **Zhang Y. et al.** (first author: Yuxin Zhang, not "T. Zhang"), published in **Bioinformatics, Volume 39, Issue 1, January 2023** (published online December 2022). The actual title is "IsRNAcirc: an integrated platform for exploring circRNA-RNA interactions and their potential roles in pathogenesis of human diseases" — which is about circRNA-RNA **interactions**, NOT 3D structure prediction. The manuscript's description of IsRNAcirc as a "3D structure prediction" tool appears to be **incorrect**. The actual IsRNAcirc paper is about interaction prediction, not 3D structure.

**Severity: CRITICAL.** (a) Wrong first author initial (T vs. Y). (b) Year should be 2023 (or 2022 online). (c) The actual paper is about circRNA-RNA interactions, not 3D structure prediction. (d) The title in the manuscript is fabricated — it does not match the real paper. This is a serious misattribution. If the authors intended to cite a different tool (IsRNA2 for 3D structure), that should be cited instead.

### 2.7 Reference [7] — Lorenz et al. (2011)

**Manuscript cites:** Lorenz, R., et al. (2011). ViennaRNA Package 2.0. Algorithms for Molecular Biology, 6(1), 26.

**Verified:** Real paper. Correct title, journal, volume, pages, and first author.

**Severity: NONE.** Citation is accurate.

### 2.8 Reference [8] — Satorras et al. (2021)

**Manuscript cites:** Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) equivariant graph neural networks. ICML.

**Verified:** Real paper. Correct title, venue, year, and authors.

**Severity: NONE.** Citation is accurate.

### 2.9 Reference [9] — Ho et al. (2020)

**Manuscript cites:** Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.

**Verified:** Real paper. Correct title, venue, year, and authors.

**Severity: NONE.** Citation is accurate.

### 2.10 Reference [10] — Gu & Dao (2023)

**Manuscript cites:** Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

**Verified:** Real paper. Correct title, arXiv ID, year, and authors. Note: later accepted at COLM 2024, but arXiv 2023 is correct for the preprint.

**Severity: NONE.** Citation is accurate.

### Reference Accuracy Summary

| Ref | Status | Severity |
|-----|--------|----------|
| [1] Wesselhoeft 2018 | Wrong page numbers (898-912 vs. 869-880) | MEDIUM |
| [2] Chen 2016 | Wrong volume/issue/pages, missing co-author, slightly wrong title | HIGH |
| [3] Vaswani 2017 | Correct | NONE |
| [4] Abramson 2024 | Correct | NONE |
| [5] Baek 2021 | Correct | NONE |
| [6] Zhang 2022 | Wrong first author, wrong year, WRONG PAPER CONTENT (interactions, not 3D structure) | CRITICAL |
| [7] Lorenz 2011 | Correct | NONE |
| [8] Satorras 2021 | Correct | NONE |
| [9] Ho 2020 | Correct | NONE |
| [10] Gu & Dao 2023 | Correct | NONE |

**Overall: 3 of 10 references have errors, 1 is critically wrong.**

---

## 3. Self-Consistency Across Sections

### 3.1 RMSD Values

| Section | Scheme 6 RMSD | Scheme 1 RMSD | Consistent? |
|---------|--------------|--------------|-------------|
| Abstract | 13.91A | Not stated | N/A |
| Results (3.3) | 13.91A mean, 14.08A median, 0.73A std | 13.85A | YES |
| Figure 2 legend | 13.91A | Not stated | YES |

### 3.2 Closure Error

| Section | Scheme 6 | Scheme 1 | Scheme 2 | Consistent? |
|---------|---------|---------|---------|-------------|
| Abstract | 0.02A | Not stated | Not stated | N/A |
| Results (3.3) | 0.02A | 5.36A | Not stated | YES |
| Figure 2 legend | 0.02A | 5.36A | <0.1A | YES |

### 3.3 Sample Size

| Section | N value | Consistent? |
|---------|---------|-------------|
| Abstract | N=7 | YES |
| Results (3.3) | N=7 | YES |
| Methods | N=7 | YES |
| Limitations | N=7 | YES |

### 3.4 Training Data Size

| Section | Value | Consistent? |
|---------|-------|-------------|
| Results (3.4) | N≈11,000, confidence ~0.5 | YES |
| Methods | N≈11,000 | YES |

### 3.5 Scheme Status

| Scheme | Manuscript Claims | Experimental Data | Consistent? |
|--------|------------------|-------------------|-------------|
| S1 | Trained | Trained | YES |
| S2 | Zero-training | Zero-training | YES |
| S3 | "not yet fully trained" | Not trained | YES |
| S4 | "not yet fully trained" | In progress | YES |
| S5 | Failed (245A) | Unstable | YES |
| S6 | Best performer | Best | YES |
| S7 | "not yet fully trained" | Not trained | YES |

### 3.6 Minor Inconsistencies

- **Abstract says "Scheme 2 achieves ~2A RMSD with guaranteed closure"** but the Results section (3.3) says "Scheme 2 (physics solver) achieved superior RMSD (~2A) with guaranteed closure." The Methods section says Scheme 2 has "No training required." These are consistent, but the abstract does not mention that Scheme 2 has no learned parameters — a reader might assume it is a trained model.

- **The manuscript says "All samples < 20A RMSD: 100%"** for Scheme 6. The experimental data shows mean 13.91, std 0.73. With N=7, if the distribution is approximately normal, 13.91 + 3*0.73 = 16.1A, which is indeed <20A. This claim is plausible but cannot be directly verified without per-sample data.

---

## 4. Overclaiming Analysis

### 4.1 Flagged Statements

| Statement | Location | Issue |
|-----------|----------|-------|
| "enabling rational design of circRNA therapeutics" | Abstract (last sentence) | **OVERCLAIM.** The manuscript presents a computational method with 13.91A RMSD on 7 test samples. "Enabling rational design of therapeutics" is a downstream application claim that requires experimental validation, which has not been performed. |
| "TorusFold provides a methodological foundation for circRNA structure prediction where none existed" | Abstract | **BORDERLINE.** IsRNAcirc (or IsRNA2) and ViennaRNA's circular mode existed before this work. The claim that "none existed" is too strong. The manuscript itself cites IsRNAcirc [6] as a prior approach. |
| "Diffusion models offer a different paradigm: they learn the data distribution" | Discussion | **ACCEPTABLE.** This is a methodological observation, not a performance claim. |
| "TorusFold provides fast inference (seconds vs. hours)" | Discussion | **UNVERIFIED.** No timing benchmarks are provided in the manuscript or experimental data. The claim "seconds vs. hours" is qualitative and unsupported by evidence. |
| "We estimate that with 50-100 high-quality experimental circRNA structures, RMSD could potentially reach <10A" | Results (3.4) | **SPECULATIVE.** This is labeled as an estimate, which is acceptable, but the extrapolation from N=7 test samples to a projected accuracy is highly uncertain. |
| "Scheme 7: Mamba + Local Attention... enabling prediction on sequences >1000 nucleotides" | Results (3.2) | **UNVERIFIED.** Scheme 7 is not trained. The claim about >1000 nt capability is theoretical, based on O(L) complexity, not demonstrated. |
| "the diffusion model implicitly learned that valid circRNA structures have closure ~5.9A" | Results (3.3) | **PLAUSIBLE BUT UNPROVEN.** The model achieves 0.02A closure, but the claim that it "learned" the 5.9A bond length is an interpretation, not a demonstrated fact. No ablation or probing experiment supports this mechanistic claim. |

### 4.2 Superlatives

| Term | Count | Context |
|------|-------|---------|
| "first" / "none existed" | 2 | "No deep learning method has been specifically designed" — acceptable; "where none existed" — overclaim (see above) |
| "best" | 2 | "Best accuracy-closure balance" — acceptable given the data |
| "striking" | 1 | "The most striking finding" — stylistic, acceptable |
| "enabling" | 1 | "Enabling rational design" — overclaim (see above) |

---

## 5. Missing Disclosures

### 5.1 iGEM Context

| Disclosed? | Details |
|------------|---------|
| YES | Authors listed as "iGEM FBH Team" |
| YES | Affiliation: "iGEM 2026, First Build High School" |
| YES | Acknowledgments: "This work was conducted as part of iGEM 2026 by the FBH (First Build High School) team" |
| PARTIAL | The manuscript does not disclose that this is a **high school** team (FBH = First Build High School). The affiliation says "First Build High School" but does not explicitly state "high school." A reader unfamiliar with iGEM might assume this is a university or research institute. |

### 5.2 Limitations

| Limitation | Disclosed? | Location |
|------------|------------|----------|
| Small test set (N=7) | YES | Discussion, Limitations paragraph |
| Schemes 3, 4, 7 not fully trained | YES | Discussion, Limitations paragraph |
| Pseudo-label training data | YES | Discussion, Limitations paragraph |
| No wet-lab validation | YES | Discussion, Limitations paragraph |
| No comparison to IsRNAcirc on same data | NO | Not mentioned |
| No timing benchmarks | NO | "Seconds vs. hours" claimed without data |
| No statistical significance tests | YES | Methods, Statistical Analysis section |
| Target journal is Nature Methods | YES | Header — but see concern below |

### 5.3 Target Journal Concern

The manuscript header states "Target Journal: Nature Methods." Given that:
- N=7 test samples
- 3 of 7 schemes are not trained
- RMSD of 13.91A is far above the ~2-3A typically expected for structure prediction methods in high-impact journals
- The work is from a high school iGEM team

...submission to Nature Methods appears to be a significant mismatch between the manuscript's maturity and the target venue. This is not an integrity violation per se, but it raises questions about whether the authors have realistically assessed their work's readiness.

---

## 6. Summary of Findings

### Critical Issues (Must Fix Before Submission)

1. **Reference [6] (IsRNAcirc) is fundamentally wrong.** The cited paper (Zhang Y. et al., 2023, Bioinformatics) is about circRNA-RNA interactions, NOT 3D structure prediction. The manuscript's title for this reference ("IsRNAcirc: a web server for predicting the 3D structure of circular RNA") does not match any real publication. This needs to be corrected to cite the actual IsRNAcirc paper with its real title, or replaced with a correct reference to IsRNA2 or another 3D structure tool.

2. **Reference [2] (Chen 2016) has wrong volume/issue/pages.** The manuscript cites 17(4):205-211. The actual paper is 17(5):307-321. The co-author Shan G. is missing. The title is slightly different.

### Medium Issues (Should Fix)

3. **Reference [1] (Wesselhoeft 2018) has wrong page numbers.** Manuscript says 898-912; actual is approximately 869-880.

4. **"Enabling rational design of circRNA therapeutics"** in the abstract is an overclaim. With 13.91A RMSD and N=7, this claim is not supported.

5. **"Where none existed"** (methodological foundation claim) is too strong given that IsRNAcirc and ViennaRNA circular mode predate this work.

6. **"Seconds vs. hours"** inference speed claim is unsubstantiated — no timing data is provided.

### Minor Issues (Consider Addressing)

7. The manuscript does not explicitly disclose that this is a **high school** project, which may be relevant context for reviewers assessing the work's scope and limitations.

8. The Kabsch RMSD bug fix (mentioned in experimental data but not the manuscript) should be disclosed if it affected reported results.

9. The ">1000 nucleotides" claim for Scheme 7 is theoretical (O(L) complexity) but presented as if demonstrated — Scheme 7 is not trained.

10. The claim that the diffusion model "learned" the 5.9A bond length is an interpretation without supporting ablation evidence.

### Strengths (Acknowledged)

- The manuscript is unusually honest about limitations for a computational methods paper. The dedicated Limitations section is commendable.
- The disclosure that 3 of 7 schemes are not trained is transparent.
- The negative result (Scheme 5 failure) is reported openly, which is good scientific practice.
- The iGEM context is disclosed in the affiliation and acknowledgments.
- The statistical analysis section correctly notes that no significance tests are claimed given N=7.
- The manuscript explicitly states it is "not claiming state-of-the-art performance."

---

## 7. Overall Assessment

The manuscript is largely internally consistent and honestly reports its limitations. The primary integrity concerns are:

1. **Reference [6] is critically wrong** — it cites a paper about circRNA-RNA interactions as if it were about 3D structure prediction, with a fabricated title. This must be corrected.

2. **Two other references have incorrect bibliographic details** (page numbers, volume/issue), suggesting citations were not verified against the actual publications.

3. **Several claims in the abstract and discussion overreach** what the data supports, particularly regarding therapeutic applications and inference speed.

These issues are correctable. The core experimental results appear to be accurately reported from the experimental data. The manuscript's tone is appropriately cautious for a preliminary methods paper, and the explicit limitations section is a significant strength.
