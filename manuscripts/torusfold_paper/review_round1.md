# Nature Methods Peer Review: TorusFold Manuscript

**Review Date:** 2026-06-23

**Manuscript:** TorusFold: Torus-Aware Deep Learning Architectures for Circular RNA 3D Structure Prediction

**Authors:** iGEM FBH Team

---

## Summary

This manuscript presents TorusFold, a systematic exploration of seven deep learning architectures for circular RNA (circRNA) 3D structure prediction. The key innovation is Torus Positional Encoding (TPE), which enforces circular periodicity. The work addresses an important gap---no experimental circRNA structures exist in PDB, and standard architectures violate circular topology. The best-performing scheme (GNN latent diffusion) achieves 13.91A RMSD with 0.02A closure error on a test set of N=7.

---

## Major Concerns

### 1. Data Sufficiency (Critical)

**The test set (N=7) is far too small for Nature Methods standards.**

With N=7 sequences, the manuscript lacks statistical power to make any meaningful claims about performance differences between architectures. The authors acknowledge this limitation (line 134), but this does not make it acceptable for publication. Specifically:

- No statistical significance can be established. Bootstrap confidence intervals on N=7 are not a substitute for adequate sample size.
- The reported RMSD values (13.91A, 13.85A) have overlapping confidence intervals. Claims that "Scheme 6 outperforms Scheme 1" are not statistically supported.
- The test set was constructed by circularizing linear RNA structures from PDB. This is a synthetic proxy, not experimental circRNA data. The "ground truth" is itself a model output (from GeometricConstraintSolver), creating circular validation.

**Recommendation:** Expand the test set to include:
- All available PDB structures with circular RNA (the authors mention only 9H8A, but the research architecture document mentions 8xtp, 8xtq, etc.---these should be included in the manuscript)
- RNA-Puzzles targets that can be circularized
- Blind test set from the proposed Circ-CASP competition before publication

**Minimum threshold for Nature Methods:** N >= 30 test structures with diverse lengths, topologies, and biological contexts.

### 2. Missing External Baselines (Critical)

**The manuscript compares only internal architectures (Schemes 1-7) against each other, with no comparison to existing published methods.**

This is a fatal gap for Nature Methods. The research architecture document explicitly identifies GAP-1 through GAP-5 as critical missing comparisons:

- **No comparison to IsRNAcirc:** The only published circRNA 3D predictor (Zhang et al., 2022) is not mentioned in the Results section. Without this comparison, the field cannot assess whether TorusFold advances beyond the current state-of-the-art.

- **No comparison to AlphaFold3 or RoseTTAFold2:** AF3 now handles RNA. A key question is whether torus-aware encoding provides value over generic RNA prediction. The expected result (AF3 fails at BSJ closure) would strengthen the paper, but it must be demonstrated, not assumed.

- **No comparison to Rosetta FARFAR2:** The standard for RNA 3D de novo prediction is absent.

- **No ablation study:** The core claim is that TPE helps, but the only evidence is a "proxy experiment" mentioned in the research architecture document, not in the manuscript itself. There is no comparison of identical architecture with TPE vs. standard positional encoding on 3D structure prediction.

**Recommendation:** Run IsRNAcirc on the same test set and report RMSD/closure error. Run AF3 on circRNA sequences. Include at minimum a TPE vs. standard PE ablation in the manuscript.

### 3. Incomplete Training Status (Critical)

**Three of seven architectures (Schemes 3, 4, 7) are explicitly stated as "not trained" or "in progress" (lines 91-92, research architecture document lines 437-440).**

This makes the "systematic comparison" claim in the Abstract and Introduction misleading. A systematic comparison requires all methods to be fully trained and evaluated under identical conditions. The current manuscript presents a partial comparison (4/7 architectures trained).

Additionally, the research architecture document reveals that Scheme 5 "failed due to coordinate instability" with RMSD 245A. While this negative result is instructive (and the authors should be commended for reporting it), it raises questions about whether the architecture was properly designed and tuned before being declared a failure.

**Recommendation:** Complete training of all seven schemes before publication, or reduce the scope to architectures that have been properly trained and evaluated.

### 4. Pseudo-Label Training Data Quality

**The training data consists of N~11,000 pseudo-labels with confidence ~0.5.**

The manuscript acknowledges that "data quality dominates prediction accuracy" (lines 92-96), with an 11A improvement when switching to high-confidence data. However, this raises a fundamental question: what are the models actually learning?

Pseudo-labels generated by IsRNAcirc, ViennaRNA, and other computational methods inherit biases and errors from those methods. If TorusFold is trained on IsRNAcirc pseudo-labels and compared to IsRNAcirc predictions, any improvement may simply reflect architectural overfitting to the specific characteristics of IsRNAcirc's outputs rather than genuine structural understanding.

**Recommendation:**
- Provide detailed analysis of pseudo-label quality: distribution of confidence scores, overlap between sources, validation against the N=7 "gold standard."
- Include a "data ceiling" experiment: train on increasing fractions of high-confidence data and show learning curves.
- Discuss the risk of circular validation (training and test data both derived from physics-based simulators).

---

## Technical Quality

### Strengths

1. **TPE is mathematically sound.** The derivation of periodic positional encoding (lines 46-51) is correct and elegant. The guarantee TPE(i) = TPE(i+L) is a legitimate innovation for circular topology.

2. **Closure error as a metric is appropriate.** The introduction of BSJ closure error (line 189-193) as an evaluation criterion is a valuable contribution that should become standard in circRNA structure prediction.

3. **Honest reporting of failures.** The authors transparently report that Scheme 5 failed with RMSD 245A and explain why (coordinate instability without equivariance constraints). This negative result is valuable for the field.

4. **Analysis of diffusion learning physical constraints.** The observation that Scheme 6 learned closure (0.02A) without explicit penalty (lines 86-88, 110-114) is interesting and suggests that diffusion models can internalize physical constraints from data.

### Weaknesses

1. **No hyperparameter sensitivity analysis.** How sensitive are results to:
   - Number of TPE harmonics (default H=16)?
   - K in KNN graph construction (default K=16)?
   - Diffusion steps (default 50)?
   - Learning rate, batch size, architecture depth?

2. **No error analysis by region.** The research architecture document identifies GAP-11: "No error analysis by region." Is the error concentrated near the BSJ? In loops? In stem regions? This is critical for understanding where TPE helps.

3. **No length scaling analysis.** The research architecture document notes that Schemes 1-6 are O(L^2) while Scheme 7 is O(L). But Scheme 7 is not trained. There is no demonstration of length scaling in the manuscript.

4. **Confidence calibration is missing.** If these models are to be used for therapeutic design, users need calibrated confidence estimates. The current manuscript does not address whether the confidence scores are meaningful.

5. **Potential bug in Scheme 6 decoder.** Line 171 mentions: "Key architectural fix: decoder receives denoised latent (not noise prediction) during training." This suggests a bug was discovered and fixed, but the manuscript does not describe what the bug was or how it was identified. Technical documentation of such bugs is important for reproducibility.

---

## Novelty Assessment

### What is genuinely novel:

1. **Torus Positional Encoding (TPE)** is a legitimate innovation. While Fourier features on a circle are not new (they are standard in harmonic analysis), applying them as positional encoding for circRNA sequences is novel and mathematically appropriate.

2. **Systematic comparison of architectures for circRNA 3D** is a useful contribution, even if incomplete. The field had no prior benchmark.

### What is not novel or insufficiently justified:

1. **Seven architectures:** This is a design space exploration, but the choices are not well motivated. Why EGNN? Why diffusion? Why Mamba? The manuscript states these choices but does not justify them theoretically or empirically.

2. **GNN latent diffusion (Scheme 6)** achieved the best results, but this architecture is a standard design pattern in the diffusion literature. The novelty claim is reduced to "we applied existing diffusion techniques to circRNA."

3. **The comparison of 7 architectures is incomplete.** With 3 architectures untrained and no external baselines, the "systematic" claim is weakened.

---

## Significance

The significance of this work depends on whether it provides a foundation for circRNA structure prediction that will be useful when experimental data becomes available. The answer is mixed:

**Positive:**
- Introduces proper evaluation metrics (BSJ closure, circular distance)
- Provides a benchmark framework that the community can build on
- Identifies that diffusion models can learn physical constraints implicitly

**Negative:**
- Current accuracy (13.91A RMSD) is far from useful for therapeutic design. For context, protein structure prediction below 2A is considered high accuracy.
- No wet-lab validation. All results are computational.
- The gap between pseudo-label accuracy and what might be achievable with experimental data is unknown.

**Assessment:** This is a methods paper that describes a framework for future work, rather than a breakthrough in circRNA structure prediction. Whether this meets the threshold for Nature Methods depends on editorial judgment about preliminary methodology.

---

## Presentation Quality

### Writing

The writing is generally clear and well-structured. The Abstract, Introduction, Results, Discussion, and Methods sections follow standard journal format. The honest acknowledgment of limitations (lines 133-136) is commendable.

**Issues:**
- Line 15: "treating the back-spliced junction (BSJ) as maximally distant when it is spatially adjacent" is imprecise. Standard positional encoding treats positions 0 and L-1 as maximally distant in sequence space, not necessarily in 3D space after folding.
- Line 119: The PCR/X-ray crystallography analogy is strained. PCR and X-ray crystallography are experimental techniques; methodology development here is computational.
- The Discussion section (lines 107-141) is defensive rather than balanced. More space should be given to what the results actually show, rather than defending against perceived criticisms.

### Figures

The figure legends (lines 245-256) describe appropriate visualizations. However, the actual figures are not included in the manuscript file, so I cannot assess their quality.

**Requested figures for revision:**
- Figure showing TPE vs. standard PE with actual 3D structures colored by error
- Figure showing per-nucleotide error heatmap around BSJ region
- Figure showing confidence calibration (reliability diagram)
- Table showing comparison to IsRNAcirc, AF3, FARFAR2

---

## Reproducibility

### Positive factors:
- Code availability statement (github.com/RomanCohort/confluencia)
- Training data mentioned as "provided in supplementary materials"
- Architecture details provided in Methods section

### Concerns:
- Pseudo-label generation pipeline is not fully described. What exact settings were used for IsRNAcirc? ViennaRNA?
- The training data file is described as "provided in supplementary materials" but is not actually included in the review materials.
- "Circ-CASP benchmark (30 blind targets) will be released" is future work, not reproducible now.
- Three architectures are not trained, making the systematic comparison non-reproducible in its current form.

---

## Research Architecture Document Concerns

The research architecture document reveals numerous gaps that are not adequately addressed in the manuscript:

1. **GAP-1 through GAP-5 (missing comparisons):** None of these external baselines appear in the manuscript.

2. **GAP-10 (no ablation study):** No TPE vs. standard PE ablation in the manuscript.

3. **GAP-12 (no length dependency):** No analysis of how performance scales with sequence length.

4. **Timeline status:** The research architecture document shows Phase 1 (training Schemes 3, 4, 7) as incomplete. This confirms that the manuscript is submitted prematurely.

The research architecture document is an excellent planning tool but raises the question: why was this manuscript submitted before the planned experiments were completed?

---

## Additional Issues

### Statistical Analysis (Lines 203-205)

"Due to small test set (N=7), we report mean, median, and bootstrap 95% confidence intervals (1000 resamples). No statistical tests are claimed to be significant given the limited sample size."

This acknowledgment of limited statistical power is honest, but it also means the manuscript cannot make strong comparative claims. Statements like "Scheme 6 achieves the best balance" should be softened to "Scheme 6 numerically achieves the best balance, though statistical significance cannot be established with N=7."

### Data Provenance

The PDB circularized set is described as "linear RNA structures from PDB were circularized using GeometricConstraintSolver." This means the "ground truth" structures are not experimental circRNA structures but computational models. The validation is circular: the same or similar geometric constraints may have been used in training.

### Wet-Lab Validation

The manuscript states: "Longer-term goals include wet-lab validation of predictions" (line 140). For Nature Methods, validation against experimental data is typically required. The absence of any wet-lab validation, combined with the absence of experimental circRNA structures, makes the claims difficult to evaluate.

---

## Recommendation

**Major Revision** (borderline Reject)

This manuscript addresses an important problem (circRNA 3D structure prediction) with a legitimate methodological innovation (TPE). However, the current submission is incomplete in ways that prevent fair evaluation:

### Required revisions for consideration:

1. **Expand test set to N >= 30.** Include PDB 8xtp, 8xtq, 8xtr, 8xts, 9is7 (mentioned in research architecture but not in manuscript), RNA-Puzzles circRNA targets, and the Circ-CASP blind test set.

2. **Include external baselines.** Run IsRNAcirc, AlphaFold3, and FARFAR2 on the same test set. Report RMSD, closure error, and computational cost for each.

3. **Complete training of all seven schemes.** The "systematic comparison" claim requires all architectures to be properly trained and evaluated.

4. **Include TPE ablation study.** Compare identical architecture with TPE vs. standard positional encoding on 3D structure prediction (not just pairing probability MSE).

5. **Add error analysis by region.** Show whether TPE specifically helps BSJ-flanking regions as claimed.

6. **Add length scaling analysis.** Demonstrate whether O(L) architectures (Scheme 7) offer advantages for long circRNAs.

7. **Address pseudo-label quality.** Provide confidence score distributions, validation statistics, and discussion of circular validation risk.

### Optional but recommended:

8. Wet-lab validation of at least one predicted structure.
9. Confidence calibration analysis.
10. Hyperparameter sensitivity analysis.

---

## Summary Verdict

| Criterion | Assessment |
|-----------|------------|
| **Novelty** | Moderate. TPE is novel; architecture comparison is useful but incomplete. |
| **Technical Quality** | Mixed. Sound mathematical derivations; incomplete experiments; missing baselines. |
| **Data Sufficiency** | Insufficient. N=7 test set; pseudo-label training; no wet-lab validation. |
| **Significance** | Preliminary. Foundation for future work, not a breakthrough. |
| **Presentation** | Good. Clear writing; honest limitations; missing figures. |
| **Reproducibility** | Mixed. Code available; data not fully specified; incomplete training. |

**Decision: Major Revision**

The work has potential, but the manuscript was submitted before the research plan was completed. With expanded experiments, external baselines, and complete training, this could be a solid Nature Methods paper. In its current form, it reads as a preliminary report or arXiv preprint rather than a publication-ready manuscript.

---

## Questions for Authors

1. What is the exact provenance of the N=7 test structures? Were they all derived from the same PDB source using the same circularization procedure?

2. What were the exact settings for IsRNAcirc/ViennaRNA pseudo-label generation?

3. Why was the manuscript submitted before Schemes 3, 4, and 7 were trained?

4. Have any predictions been validated experimentally, even indirectly (e.g., comparing predicted secondary structure to SHAPE data)?

5. What is the expected timeline for Circ-CASP, and will the authors wait for those results before resubmission?

---

*Review prepared following Nature Methods evaluation criteria*