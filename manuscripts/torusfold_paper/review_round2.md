# Nature Methods Peer Review — Round 2 (Re-review after Major Revision)

**Review Date:** 2026-06-23

**Manuscript:** TorusFold: Torus-Aware Deep Learning Architectures for Circular RNA 3D Structure Prediction

**Authors:** iGEM FBH Team

**Revision Type:** Major Revision (Round 1 verdict)

---

## Summary of Revision

The authors have responded to all seven Round 1 concerns by adding structural sections, tables, and explicit acknowledgments. However, the vast majority of these additions contain "TBD" (To Be Determined) placeholders rather than actual experimental results. I estimate approximately 50-55 TBD placeholders remain in the current manuscript, spanning:

- External baseline comparison tables (IsRNA, AF3, FARFAR2, ViennaRNA 3D): all values TBD
- TPE ablation study: all values TBD
- Error analysis by structural region: all values TBD
- Length scaling analysis: all values TBD
- Hyperparameter sensitivity: all values TBD
- Confidence calibration: all values TBD
- Expanded test set (N >= 30): all values TBD
- Data quality learning curves: most values TBD
- Figure legends: 10+ TBD references

Two of seven architectures (Schemes 3 and 5) have been formally abandoned with detailed failure analysis. Two more (Schemes 4 and 7) remain in training. Only Schemes 1, 2, and 6 have completed results.

The positive changes in this revision are real and significant: the failure analysis is genuinely excellent, overclaiming has been substantially reduced, and reference errors have been corrected. But the proliferation of TBD placeholders transforms this from a paper with gaps into a paper that is, in essence, a detailed protocol for experiments that have not yet been performed.

---

## 1. Round 1 Concern Assessment

### Concern 1: N=7 Test Set Too Small

**Round 1:** "The test set (N=7) is far too small for Nature Methods standards. Minimum threshold: N >= 30."

**Revision:** Added "Expanded Test Set Results" section (lines 237-249) identifying future additions: PDB structures 8xtp/8xtq/8xtr/8xts, RNA-Puzzles circularized targets, and Circ-CASP blind targets. Target: N >= 30. All results are TBD.

**Assessment: Partially Addressed (Structural)**

The authors have correctly identified where additional structures can be found and established a target. The framework for expansion is in place. However, the actual expanded test set does not exist in the manuscript — it is a plan, not a result. The structural addition is appropriate; the experimental content is absent.

**What remains to be done:** Actually circularize the identified PDB structures, evaluate on them, and report results. Without this, the N=7 limitation is as severe as in Round 1.

---

### Concern 2: Missing External Baselines (IsRNA, AF3, FARFAR2)

**Round 1:** "A fatal gap for Nature Methods. No comparison to any published method."

**Revision:** Added "External Baseline Comparisons" section (lines 169-182) with a table including IsRNA, AlphaFold3, FARFAR2, and ViennaRNA 3D. Also added a direct IsRNA comparison table in the Discussion (lines 286-295). All values are TBD.

**Assessment: Partially Addressed (Structural)**

The comparison tables are correctly structured and include the right methods. Reference [11] for FARFAR2 has been properly added. The discussion of complementarity with IsRNA (lines 283-297) is improved. However, not a single baseline has been run. The reader learns nothing about how TorusFold compares to existing methods — which was the entire point of the concern.

**What remains to be done:** Run IsRNA, AF3, and FARFAR2 on the identical test set with the same evaluation protocol. The manuscript explicitly acknowledges this (line 182: "run all external baselines on identical test set").

---

### Concern 3: Incomplete Training (3 of 7 Schemes Untrained)

**Round 1:** "Three of seven architectures are explicitly stated as 'not trained' or 'in progress'. The 'systematic comparison' claim is misleading."

**Revision:** Schemes 3 and 5 formally abandoned with four pages of detailed failure analysis (lines 114-144). Scheme 3 renamed from "not trained" to "abandoned due to gradient divergence." Scheme 5' (delta variant) abandoned due to CPU saturation. Schemes 4 and 7 remain "Training" (TBD results). The abstract now states "Five of seven proposed architectures... with 2 pending" (line 303). Overclaiming reduced: "Rather than claiming state-of-the-art performance" (line 31).

**Assessment: Partially Addressed (Improved Honesty)**

This is the most substantively addressed concern. The authors have made a responsible decision to formally abandon two failed architectures rather than leave them in limbo. The failure analysis is genuinely excellent (see Section 3 below). However, Schemes 4 and 7 are still not complete, so the comparison remains 3/7 fully evaluated (Schemes 1, 2, 6), 2/7 abandoned with analysis (Schemes 3, 5), and 2/7 in progress (Schemes 4, 7).

The shift from "not trained" to "abandoned with failure analysis" is intellectually honest. The remaining gap is Schemes 4 and 7.

---

### Concern 4: No TPE Ablation Study

**Round 1:** "The core claim is that TPE helps, but there is no comparison of identical architecture with TPE vs. standard positional encoding."

**Revision:** Added "TPE Ablation Study" section (lines 158-167) with a comparison table (Standard PE vs. TPE on Scheme 6 backbone). Includes a hypothesis: "We hypothesize that TPE will show the largest improvement in BSJ-flanking regions." All values TBD.

**Assessment: Partially Addressed (Structural)**

The ablation is correctly designed: same backbone (Scheme 6), same test set, only encoding differs. The hypothesis about BSJ-flanking regions is testable and specific. But the experiment has not been run. The mathematical proof of periodicity (TPE(i) = TPE(i+L)) remains the only evidence that TPE provides benefit over standard PE.

**What remains to be done:** Train Scheme 6 with standard PE, evaluate on the same test set, report paired comparison with statistical test.

---

### Concern 5: No Error Analysis by Region

**Round 1:** "Is the error concentrated near the BSJ? In loops? In stem regions? This is critical for understanding where TPE helps."

**Revision:** Added "Error Analysis by Structural Region" section (lines 184-195) with a table decomposing error into BSJ-flanking, stem, loop/hairpin, and single-stranded regions. All values TBD.

**Assessment: Partially Addressed (Structural)**

The decomposition into four structural regions is appropriate. The per-nucleotide error heatmap plan (line 195) is the right visualization. But no analysis has been performed.

---

### Concern 6: No Length Scaling Analysis

**Round 1:** "No demonstration of length scaling in the manuscript."

**Revision:** Added "Length Scaling Analysis" section (lines 197-209) with a table across five length ranges (20-50, 50-100, 100-200, 200-500, 500-1000 nt). All values TBD.

**Assessment: Partially Addressed (Structural)**

The length range table is well-structured. The O(L) claim for Scheme 7 is appropriately framed as a design goal rather than a demonstrated result. But the analysis has not been performed.

---

### Concern 7: Pseudo-Label Quality Concerns

**Round 1:** "Provide detailed analysis of pseudo-label quality... data ceiling experiment... discussion of circular validation risk."

**Revision:** Added "Data Quality Dominates Prediction Accuracy" section (lines 108-112) reporting an 11 RMSD improvement from low-confidence to high-confidence data. Added confidence score distribution discussion (lines 271-280) with a learning curve table (most values TBD). Added explicit acknowledgment of "circular validation" risk (line 304: "training and test data both derived from physics-based simulators").

**Assessment: Mostly Addressed (Substantive)**

This is the most substantively addressed concern alongside the failure analysis. The observation that data quality accounts for an 11 RMSD improvement (from ~25A to ~14A) is a concrete, data-driven finding. The explicit acknowledgment of circular validation risk (line 304) directly addresses the Round 1 concern. The learning curve framework is in place, though most values remain TBD.

---

## 2. Assessment of TBD Placeholders

### Scope and Distribution

I counted approximately 50-55 distinct TBD placeholders in the manuscript, distributed as follows:

| Section | TBD Count |
|---------|-----------|
| Abstract | 4 |
| Table 1 (Scheme results) | 2 |
| TPE Ablation | 3 |
| External Baselines | 12 |
| Error Analysis | 8 |
| Length Scaling | 8 |
| Hyperparameter Sensitivity | 5 |
| Confidence Calibration | 3 |
| Expanded Test Set | 6 |
| Learning Curves | 3 |
| IsRNA Comparison | 3 |
| Limitations (TBD references) | 8 |
| TBD Checklist | 35 items |
| Figure Legends | 10+ |

**Some TBDs overlap across these counts; the total is approximately 50-55.**

### Are TBD Placeholders Appropriate?

For a few specific items, TBD is defensible:
- **Circ-CASP blind targets** (line 246): These are future community targets that do not yet exist. TBD is appropriate.
- **Wet-lab validation** (line 305): Acknowledged as future work. TBD is appropriate.
- **Confidence calibration** (line 235): Secondary analysis. TBD is understandable.

However, the majority of TBDs represent experiments that should have been completed before submission:
- **External baselines** (12 TBD): These are the most critical missing results. Running IsRNA and AF3 on 7 sequences is a matter of days, not months.
- **TPE ablation** (3 TBD): A single controlled experiment. Same architecture, different encoding.
- **Error analysis by region** (8 TBD): Post-hoc computation on existing predictions. No new training required.
- **Length scaling** (8 TBD): Post-hoc analysis of existing predictions, plus Scheme 7 training.

### Impact on Publishability

A manuscript with 50+ TBD placeholders is, by definition, incomplete. It is a **research protocol**, not a **research report**. For Nature Methods, this makes the manuscript unpublishable in its current form. Even for lower-tier journals, the majority of TBDs would need to be resolved.

The authors frame TBD as "pending" — implying that experiments are ongoing and results will follow. But the purpose of a manuscript is to report completed research, not to pre-register planned research. TBD placeholders should appear in supplementary materials or a "Future Work" section, not embedded throughout the Results section.

---

## 3. Failure Analysis Assessment (Schemes 3 and 5)

The failure analysis for Schemes 3 and 5 is the strongest component of this revision. I assess it as **excellent** and recommend that it be retained and potentially expanded in future versions.

### Scheme 5 (Physics-Biased Attention) — Failure Analysis Quality: 4.5/5

The identification of four specific failure mechanisms is precise and actionable:

1. **Unbounded coordinate output space** (line 118): The transformer can produce coordinates of arbitrary magnitude, leading to MSE > 250,000 and immediate gradient explosion. This is a fundamental architectural flaw that the analysis correctly identifies.

2. **Unstable learnable scaling** (line 120): The `output_scale=50.0` parameter creates a feedback loop where the scaling factor itself becomes unstable. The analysis correctly traces this through the backpropagation chain.

3. **Ineffective closure correction** (line 122): The "physics-biased" label was misleading — it was a single scalar post-hoc correction, not genuine physical constraint integration. The NaN generation from `safe_dist` approaching zero is a concrete bug that the analysis identifies.

4. **Length-limited positional encoding** (line 124): `nn.Embedding(512)` hard-caps sequence length, incompatible with circRNA applications.

The core insight (line 126) is particularly valuable: "A transformer can learn arbitrary mappings from sequence tokens to 3D coordinates, but the optimization landscape contains no structure that steers predictions toward physically valid RNA conformations." This is a general principle that applies beyond circRNA.

### Scheme 3 (Dual-Engine Iterative) — Failure Analysis Quality: 4.5/5

The identification of the fundamental design contradiction is incisive: "the 'dual-engine' was effectively a single-engine transformer operating from scratch" (line 132). The initialization was planar circular geometry, not physics solver output, making the "dual-engine" claim misleading.

The four failure mechanisms are well-analyzed:

1. **Loss balance instability** (line 136): Three loss terms with static weights (1.0, 0.1, 0.1) that create dominance-shifting during training.
2. **Closure loss magnitude mismatch** (line 138): Clamping to [-5, 5] in normalized space creates discontinuities.
3. **Unvectorized per-sample computation** (line 140): CPU bottleneck from sequential loops, saturating utilization >100%.
4. **Residual prediction on inadequate reference** (line 142): Planar circular geometry deviates ~60A RMSD, requiring large deltas that reintroduce unbounded output.

The counterfactual insight (line 144) is excellent: "If Scheme 3 had been initialized from Scheme 2 outputs rather than planar geometry, the delta prediction space would be small (2-5A corrections) and the architecture might have succeeded."

### Synthesis (Three Necessary Conditions) — Quality: 5/5

The synthesis (lines 146-156) identifies three necessary conditions that emerge from the failures:
1. Geometric inductive bias (equivariance or latent compression)
2. Bounded output magnitude (diffusion naturally provides this)
3. Vectorizable computation (batch-level GPU parallelism)

These conditions are general, testable, and correctly distinguish successful from failed architectures. This synthesis elevates the failure analysis from post-hoc rationalization to a genuine methodological contribution.

---

## 4. NEW Concerns Introduced by Revisions

### 4a. The "TBD Skeleton" Problem (Major)

The revision has transformed the manuscript from a paper with missing experiments into a paper that is a skeleton of planned experiments. Every table has rows, but the data is absent. This creates a new readability problem: the manuscript reads as a promise of science rather than science itself.

Specifically, the proliferation of TBD sections dilutes the impact of the results that *are* present. The 13.91A RMSD for Scheme 6 and the failure analysis for Schemes 3/5 are buried among dozens of empty tables.

### 4b. Shrinking Scope Without Scope Declaration (Moderate)

The manuscript was submitted as "a systematic comparison of seven architectures." After this revision, it is:
- 3 architectures fully evaluated (Schemes 1, 2, 6)
- 2 architectures abandoned (Schemes 3, 5)
- 2 architectures in progress (Schemes 4, 7)

This means the core contribution is now primarily about Schemes 1, 2, and 6, with extensive documentation of two failures. The title and abstract still claim seven architectures, which is misleading when only three have results.

### 4c. Scheme 2 as a "Zero-Training" Baseline (Moderate)

Scheme 2 (Physics Solver, ~2A RMSD) is described as having "no learned parameters" (line 61). Yet it is presented alongside learned methods in Table 1. If Scheme 2 achieves ~2A RMSD while all learned methods achieve ~14A RMSD, this undermines the case for deep learning approaches. The manuscript acknowledges this (line 106) but does not fully grapple with the implication: a physics solver with no training outperforms all deep learning methods by a factor of 7x.

This is not a new concern per se, but the revision's addition of "ready" status for Scheme 2 makes it more prominent and raises the question of whether the learned methods provide any value over the physics baseline beyond speed.

### 4d. Circular Validation Risk Insufficiently Mitigated (Minor)

The manuscript acknowledges circular validation risk (line 304) but does not mitigate it. The test set (N=7) is derived from the same GeometricConstraintSolver that generates the training pseudo-labels. This means the "ground truth" and training data share the same generative model. Any method that learns the properties of this solver will appear to perform well, regardless of whether it captures real circRNA physics.

The revision adds "Validation against known RNA motifs" (line 433) as a TBD item, which is the right direction. But it remains unexecuted.

### 4e. Inconsistent Treatment of Scheme 4 and 7 (Minor)

Schemes 4 and 7 are listed as "Training" with TBD results throughout, yet they also appear in the synthesis (line 156) as "surviving architectures." This is premature — they have not yet demonstrated viability. They should be excluded from the synthesis until training is complete.

---

## 5. Readiness for Journal Submission

### Nature Methods: No

The manuscript is not ready for Nature Methods. The journal requires:
- External baseline comparisons (all TBD)
- Adequate test set size (N=7, expansion TBD)
- Ablation studies (all TBD)
- Complete method evaluation (4/7 incomplete)

The failure analysis and TPE derivation are strong, but the experimental results are far too incomplete.

### Nature Communications / NAR: No

Same concerns, though with a lower bar for completeness. The TBD placeholders alone would likely result in desk rejection at any Nature-family journal.

### PLOS Computational Biology / Bioinformatics: No (but closer)

These journals are more tolerant of preliminary work, but 50+ TBD placeholders in the Results section is unusual for any peer-reviewed journal. The failure analysis alone might be publishable as a "negative results" paper, but the full manuscript needs completed experiments.

### iGEM Wiki / arXiv Preprint: Yes (with revision)

The manuscript in its current form is suitable for:
1. **arXiv preprint** — with a clear statement that results are preliminary and TBD items are planned
2. **iGEM competition wiki** — the level of completeness is appropriate for a competition submission
3. **Conference workshop** — some workshops accept work-in-progress papers

### Recommendation for Path to Publication

The authors should:

1. **First priority:** Run external baselines (IsRNA, AF3, FARFAR2) on the N=7 test set. This is the highest-impact missing result and requires no new training — just running existing tools.

2. **Second priority:** Run the TPE ablation study. Same backbone, different encoding. One experiment.

3. **Third priority:** Perform error analysis by region. This is post-hoc computation on existing predictions — no new training.

4. **Fourth priority:** Expand test set to N >= 15 (halfway to 30) with available PDB circRNA structures.

5. **Fifth priority:** Complete training of Schemes 4 and 7, or formally abandon them.

With priorities 1-3 completed, the manuscript would be a candidate for submission to PLOS Computational Biology or Bioinformatics. With priorities 1-5 completed, it could target Nature Communications or NAR. Nature Methods would require the full N >= 30 expansion plus wet-lab validation.

---

## 6. Updated Verdict

### Major Revision (unchanged from Round 1)

The revision demonstrates genuine intellectual engagement with Round 1 concerns. The failure analysis for Schemes 3 and 5 is a significant improvement and represents a legitimate methodological contribution. The reduction in overclaiming is appropriate and welcomed.

However, the proliferation of TBD placeholders means that the fundamental gap identified in Round 1 — incomplete experimental results — has not been closed. It has been **documented** rather than **resolved**.

The manuscript is now more honest about what it does not know, but it does not know substantially more than it did in Round 1.

| Criterion | Round 1 | Round 2 | Change |
|-----------|---------|---------|--------|
| Novelty | Moderate | Moderate | — |
| Technical Quality | Mixed | Mixed | Improved honesty; same data gaps |
| Data Sufficiency | Insufficient | Insufficient | — |
| Failure Documentation | Good | Excellent | Significant improvement |
| Experimental Completeness | Incomplete | Incomplete | — |
| Presentation | Good | Good | TBD proliferation hurts readability |

**Decision: Major Revision (borderline Reject)**

The border between Major Revision and Reject is defined by whether the authors have demonstrated a credible path to addressing the concerns. The authors have: they have the frameworks in place, the failure analysis shows deep understanding, and the remaining experiments are clearly defined. But the manuscript cannot progress to publication until the TBDs become numbers.

---

## Questions for Authors

1. **What is the expected timeline for completing the TBD experiments?** The manuscript reads as if experiments are ongoing, but there is no timeline. Are external baselines expected to be completed in weeks or months?

2. **Can the failure analysis be published independently?** The four pages on Schemes 3 and 5 failures (with the three-condition synthesis) are the strongest part of the manuscript and could stand alone as a "lessons learned" paper while the remaining experiments are completed.

3. **Is Scheme 2 (~2A RMSD) the main result?** The physics solver outperforms all learned methods by 7x. Is the paper's message actually "diffusion models learn closure but physics solvers are more accurate"? This reframing might strengthen the manuscript.

4. **Can error analysis be done immediately?** The per-nucleotide error heatmap and regional decomposition (lines 184-195) require no new training — they are computations on existing predictions. Why are these TBD rather than completed?

5. **Will the authors submit to arXiv first?** Given the TBD-heavy state, an arXiv preprint with a "work in progress" designation would establish priority for TPE and the failure analysis while experiments continue.

---

*Review prepared following Nature Methods evaluation criteria*
