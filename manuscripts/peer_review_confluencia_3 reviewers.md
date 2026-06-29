# Peer Review Report: Confluencia 3.0
## Submitted to *Bioinformatics*

---

## Reviewer #1: Computational Biology Expert

### Summary

This manuscript presents Confluencia 3.0, an integrated computational platform for circRNA vaccine design coupled with TNBC molecular subtype simulation. The work introduces an EventBus-first architecture with six subsystems (Tumor, TME, Treatment, CircRNA, Biomarker, Clinical), novel pharmacokinetic modeling (CirculaPK), pathway-resolved immunogenicity scoring, and TNBC subtype-specific simulation capabilities.

### Major Strengths

1. **Comprehensive platform architecture**: The EventBus-first design with lazy-loading backend integration and five user interfaces addresses the critical accessibility gap in computational biology tools. This architectural innovation could serve as a template for other domain-specific platforms.

2. **Novel circRNA-specific pharmacokinetics**: The six-compartment CirculaPK model explicitly captures circRNA bottlenecks (endosomal escape 1-4%, IRES-dependent translation) that are absent in existing PK tools. The 4.1% error validation against Wesselhoeft 2018 literature is impressive.

3. **Pathway-resolved immunogenicity**: The differential m6A suppression modeling (MDA5=0.90, TLR7/8=0.30, PKR=0.20) corrects the oversimplified "m6A reduces immunogenicity" assumption. The statistically significant improvement over GC-only baseline (ΔAIC=-8.2, p=0.004) demonstrates non-trivial contribution.

4. **TNBC subtype integration**: Coupling four TNBC subtypes (BLIS, IM, M, LAR) with circRNA design via EventBus enables subtype-adaptive vaccine optimization. The 2.6x IM vs. BLIS response difference is clinically meaningful.

5. **Federated model sharing**: Confluencia Hub with ethics-gated uploads and dual-use screening addresses the small-sample problem endemic to circRNA research while maintaining privacy and ethical standards.

### Major Weaknesses

1. **Extremely small validation sample sizes**: Immunogenicity correlation N=7, PK validation N=4, subtype comparison N=4. These sample sizes severely limit statistical power. With N=7, SE(r)≈0.18 and 95% CI for r=0.91 is [0.47, 0.99]—too wide for definitive conclusions.

2. **Circular validation concern**: The parameter-swap experiment validates internal consistency, not predictive utility. The IM vs. BLIS 2.6x response difference reflects input parameterization from Jiang 2019 rather than novel prediction. This is hypothesis confirmation, not hypothesis generation.

3. **Incomplete immunogenicity pathway accuracy**: MDA5/dsRNA and PKR pathways show 0% classification accuracy, while only TLR7/8 achieves 100%. This suggests the pathway-resolved scoring is incomplete—four pathways but only one validated.

4. **No wet-lab validation**: All claims are computational. The ongoing collaborations with medical school are promising but results won't be available for 6 months. Publishing without experimental confirmation risks overstating predictive utility.

5. **Training data quality bottleneck**: Structure prediction backend achieves ~2Å RMSD on physics solver but ~14Å on deep learning. The manuscript acknowledges training data quality dominates accuracy, but this undermines the claim that deep learning models contribute meaningful improvement.

### Specific Comments

**Introduction:**
- Gap 3 is well-articulated, but the claim "no platform links circRNA design to TNBC subtype simulation" should cite specific attempts or explain why this integration gap exists.
- The contribution statement lists four innovations but mixes scientific claims (PK, immunogenicity) with architectural claims (EventBus). Consider separating these categories.

**Methods - Module 2 (CirculaPK):**
- The endosomal escape fraction shows 158% error (simulated 5.16% vs literature 2%). The manuscript claims this derives from stochastic efficiency, but the large discrepancy undermines the validation argument.
- The six-compartment model comparison with two-compartment (ΔAIC=4.5) is underpowered at N=4. A minimum N=12 is needed for significance—acknowledge this limitation explicitly.

**Methods - Module 3 (Immunogenicity):**
- Equation 1 shows pathway weights and m6A suppression coefficients, but the derivation of these values is unclear. How were MDA5=0.35, TLR7/8=0.30 determined? Are these empirical or theoretical?
- The secondary HEK293 validation shows r=0.68 with CI [0.26-0.88]. The CI width 0.62 is insufficient to distinguish from moderate or strong correlation—this should be reported as inconclusive rather than validating.

**Results - Pharmacokinetics:**
- The 100% pass rate against seven literature parameters is misleading. The endosomal escape 158% error is included in this "pass"—what tolerance threshold defines "pass"?

**Results - Immunogenicity:**
- The pathway classification accuracy table should be presented: Overall 43.5% accuracy is mediocre, and the 0% accuracy for MDA5/dsRNA (the primary pathway claimed) contradicts the pathway resolution contribution.
- Literature case studies n=17 shows direction agreement 58.8% and r=-0.056 (negative correlation). This negative result should be discussed—why does the model fail on case studies?

**Discussion:**
- Claim 4 "EventBus architecture enables longevity" is labeled as "architectural claim, not scientific claim." This is fair, but the manuscript should avoid presenting architectural features as scientific contributions in the abstract/introduction.

**Power Analysis section is excellent**: The manuscript is unusually transparent about statistical limitations. This should be moved from Discussion to Results to contextualize all N-dependent claims earlier.

### Questions for the Authors

1. How were the pathway weights (MDA5=0.35, TLR7/8=0.30, PKR=0.20, JAK-STAT=0.15) and m6A suppression coefficients derived? Provide empirical justification or theoretical derivation.

2. Why does the pathway-resolved scoring achieve 0% accuracy for MDA5/dsRNA when this is claimed as the "primary pathway for circRNA immunogenicity"? What sequence features are missing for dsRNA structure prediction?

3. The immunogenicity validation shows contradictory results: Chen 2019 r=0.91 (strong), HEK293 r=0.68 (moderate, wide CI), case studies r=-0.056 (negative). How do you reconcile these discrepancies?

4. What tolerance threshold defines the "100% pass rate" for PK validation? The 158% error on endosomal escape should not qualify as "pass" without justification.

5. How many circRNA sequences are in the Circ-CASP competition training set? The manuscript mentions "multi-source training data" but lacks quantitative details for reproducibility.

6. When will wet-lab validation results be available? If within 6 months, would you consider revising the manuscript after experimental confirmation rather than publishing hypothesis-generating claims?

### Recommendation

**Major Revision**

The platform architecture and circRNA-specific innovations are novel and valuable. However, the validation relies on extremely small sample sizes (N=4-7) with contradictory results across datasets. The pathway-resolved immunogenicity achieves 0% accuracy for MDA5 (the primary pathway), and case studies show negative correlation. Before acceptance, the authors should:

1. Expand validation sample sizes if possible, or explicitly acknowledge that N=4-7 provides preliminary evidence only
2. Address the contradictory immunogenicity validation results (Chen 2019 vs. HEK293 vs. case studies)
3. Explain why MDA5 pathway accuracy is 0% and whether this undermines the pathway-resolved contribution
4. Reclassify architectural claims (EventBus, federated sharing) separately from scientific claims (PK, immunogenicity validation)
5. Consider waiting for wet-lab validation results before final publication

---

## Reviewer #2: Software Architecture & Implementation Expert

### Summary

This work presents Confluencia 3.0 as an extensible computational platform for circRNA vaccine design. The technical innovation centers on EventBus-first architecture with lazy-loading backend integration, enabling algorithm replacement without platform reimplementation. The platform provides five interfaces targeting diverse user communities and federated model sharing via Confluencia Hub.

### Major Strengths

1. **EventBus architecture design**: The pub/sub decoupling with 34+ event types and six SubsystemManagers is technically sound. The lazy-loading with three-tier fallback (GPU→CPU→heuristic) enables offline-first operation—a critical feature for resource-limited settings.

2. **Multi-interface accessibility**: Python API, Streamlit UI, CLI, R package, and PyQt6 desktop IDE cover the spectrum from computational biologists to experimental researchers. The 87% test coverage via pytest indicates professional software engineering standards.

3. **Bridge architecture**: Confluencia 2.0 backward compatibility via lazy-loading bridges (DrugPredictionBridge, PKModelBridge, EpitopePredictionBridge, JointEvaluationBridge) demonstrates thoughtful migration design.

4. **Federated model sharing**: Confluencia Hub with SHA256 hash verification and ethics-gating addresses both security and ethical concerns. The "upload model bundles not raw data" approach is privacy-preserving.

5. **Event-driven treatment dispatch**: Three circRNA therapy mechanisms (miRNA sponge, protein coding, immune stimulation) with RL-ABM closed-loop optimization demonstrates end-to-end integration capability.

### Major Weaknesses

1. **EventBus complexity overhead**: 34+ event types and 37+ sub-modules across six SubsystemManagers introduce substantial complexity. The manuscript lacks performance benchmarks—latency, throughput, memory footprint under concurrent event dispatch.

2. **Lazy-loading edge cases**: The three-tier fallback is described conceptually, but edge cases are unaddressed: What happens when ViennaRNA fails during StructurePredictEvent? How are timeout and retry handled? The heuristic fallback quality is not quantified.

3. **RL-ABM optimization details**: PPO with 1000 episodes is mentioned, but training stability, convergence criteria, reward shaping, and episode duration are unspecified. RL for sequence optimization is notoriously unstable—provide implementation details.

4. **Confluencia Hub security**: SHA256 verification mitigates code execution risks, but model weight injection attacks are not discussed. PyTorch model loading can execute arbitrary code—what sandboxing mechanisms are implemented?

5. **Scalability limits**: Streamlit frontend with 10 tabs and 15 experiment modules is mentioned, but performance under large-scale simulations (e.g., 1000 tumor cells over 180 days) is not benchmarked. Memory requirements are unspecified.

### Specific Comments

**Methods - Software Architecture:**
- Line 108: "State schema: ~180 state keys with prefix namespacing" is vague. Provide the actual state schema or a representative subset for reproducibility.
- Event types list should include full signatures: event name, payload schema, publisher/subscriber pattern. Current list is incomplete.

**Methods - TNBC Simulacrum:**
- Shannon diversity 0.4→1.2 increase is reported, but the simulation timestep, spatial resolution, and boundary conditions are unspecified. Is this an ODE-based agent model or a true spatial PDE/ABM hybrid?
- "3-5 resistant subclones emerging" is stochastic—provide confidence intervals across simulation runs.

**Methods - CirculaPK:**
- The differential equations for six-compartment model are not provided. For reproducibility, include the PK rate equations:
  ```
  dC1/dt = -k_admin * C1
  dC2/dt = k_admin * C1 - k_distribution * C2
  ...
  ```
- Endosomal escape rate k_escape=0.025/h needs derivation from 2-4% efficiency. Explain the conversion.

**Methods - RL-ABM:**
- Reward function "Simulated immune response × stability × expression" needs mathematical formulation. How are these three factors measured and combined?
- PPO hyperparameters: clip_ratio, value_coef, entropy_coef are standard but unspecified.

**Results - Implementation:**
- No performance benchmarks: inference time, memory usage, concurrent event handling capacity. For a platform claiming scalability, these metrics are essential.
- 87% test coverage is good, but which modules are uncovered? Critical paths like EventBus dispatch should have 100% coverage.

**Discussion - Integration Ecosystem:**
- External tool licensing: ViennaRNA is GPL, ESM2 may have restrictions, NetMHCpan requires registration. How does Confluencia handle licensing compatibility? Users may face legal issues.

### Questions for the Authors

1. Provide EventBus performance benchmarks: latency per event dispatch, throughput under concurrent subscribers, memory footprint with 34+ event types registered.

2. Detail the lazy-loading fallback edge cases: timeout handling, retry logic, heuristic quality quantification. What happens if all three tiers fail?

3. Provide RL-ABM implementation details: reward shaping, convergence criteria, episode definition, training stability over 1000 episodes. Include hyperparameters.

4. What sandboxing mechanisms prevent model weight injection attacks in Confluencia Hub? Can uploaded model weights execute arbitrary code during torch.load()?

5. Provide the six-compartment PK differential equations for reproducibility. Derive k_escape=0.025/h from literature 2-4% efficiency.

6. Benchmark scalability: memory and runtime for large-scale simulations (1000 cells, 180 days, concurrent event dispatch). What are the limits?

### Recommendation

**Minor Revision**

The technical architecture is sound and novel for computational biology platforms. The EventBus-first design is well-conceptualized, but implementation details and performance benchmarks are insufficient. Before acceptance:

1. Provide EventBus performance benchmarks (latency, throughput, memory)
2. Detail lazy-loading edge cases (timeout, retry, heuristic quality)
3. Include PK differential equations and RL-ABM implementation details
4. Address Confluencia Hub security (model weight injection)
5. Benchmark scalability for large-scale simulations
6. Clarify external tool licensing compatibility

---

## Reviewer #3: Statistical Validation & Data Quality Expert

### Summary

This manuscript claims four scientific innovations validated against limited experimental data: (1) circRNA-specific PK (N=4), (2) pathway-resolved immunogenicity (N=7 primary, N=50 GC comparison, N=3000 pathway classification), (3) TNBC subtype simulation (N=4), and (4) structure prediction backend. The validation relies on heterogeneous pseudo-labeled training data due to the fundamental absence of experimental circRNA structures in PDB.

### Major Strengths

1. **Transparent power analysis**: The manuscript explicitly acknowledges statistical limitations (N=7 SE(r)≈0.18, N=4 underpowered model comparison, power≈0.35 for distinguishing r=0.91 from r=0.50). This transparency is rare and commendable.

2. **Multi-source training data documentation**: The five-source pipeline (PDB circularized, ViennaRNA circ-mode, IsRNAcirc, icSHAPE-constrained, Rfam consensus) with confidence scoring (0.3-1.0) addresses data scarcity systematically.

3. **Circ-CASP community benchmark**: Establishing a community benchmark with standardized metrics (RMSD, BSJ closure, bond consistency, pair F1) provides infrastructure for future method comparison.

4. **Parameter-swap validation**: The BLIS↔IM parameter swap validates internal consistency, confirming simulation outcomes are determined by input parameters rather than hardcoded dynamics. This eliminates circular validation concerns.

5. **Differential m6A modeling**: The pathway-specific suppression coefficients (MDA5=0.90, TLR7/8=0.30) demonstrate mechanistic refinement beyond the GC-only baseline. The ΔAIC=-8.2 with p=0.004 is statistically significant.

### Major Weaknesses

1. **Contradictory validation across datasets**: Immunogenicity shows r=0.91 (Chen 2019), r=0.68 (HEK293, CI 0.26-0.88), and r=-0.056 (case studies). The negative correlation on case studies undermines the claimed predictive utility.

2. **Pathway classification accuracy paradox**: Overall accuracy 43.5%, with MDA5 0%, TLR7/8 100%, PKR 0%, JAK-STAT 0%. The primary pathway for circRNA immunogenicity (MDA5/dsRNA) achieves 0% accuracy—this contradicts the pathway-resolved contribution.

3. **PK validation tolerance undefined**: "100% pass rate against seven literature parameters" is misleading when endosomal escape shows 158% error. What threshold defines "pass"? This should be explicit.

4. **Training data circularity concern**: ViennaRNA circ-mode predictions are used as training data for structure prediction, then ViennaRNA is used as baseline (Scheme 2). This circular use of the same tool for training and validation inflates performance.

5. **Sample size requirements unmet**: The manuscript correctly calculates N=12 needed for PK model comparison at power=0.80, but provides N=4. The correct approach is to acknowledge underpowered evidence rather than present ΔAIC=4.5 as meaningful.

### Specific Comments

**Introduction:**
- The four innovations mix scientific claims (validated against literature) with architectural claims (EventBus). The abstract should separate these: "We present a computational platform with architectural innovations (EventBus, federated sharing) and preliminary scientific evidence for circRNA-specific PK and immunogenicity."

**Results - Immunogenicity:**
- The Chen 2019 r=0.91 correlation is the primary validation, but N=7 limits statistical power. The manuscript correctly calculates SE≈0.18 and CI [0.47, 0.99], but should report this as "preliminary evidence" rather than "validation."
- The pathway classification accuracy table is critical but buried in text. Present it explicitly:
  ```
  Pathway         Accuracy    Interpretation
  MDA5/dsRNA      0%          Primary pathway fails
  TLR7/8          100%        Well-characterized in vitro
  PKR             0%          Sequence features missing
  JAK-STAT        0%          Secondary pathway fails
  Overall         43.5%       Mediocre
  ```
- The case studies showing r=-0.056 (p=0.83, negative correlation) contradict the Chen 2019 validation. This discrepancy requires explanation—different dataset characteristics? Model overfitting to Chen 2019?

**Results - Pharmacokinetics:**
- The endosomal escape 158% error (simulated 5.16% vs literature 2%) is large. The manuscript attributes this to stochastic efficiency derivation, but this explanation is insufficient. What stochastic model produces 158% error?
- The six-compartment vs. two-compartment comparison (ΔAIC=4.5) is underpowered at N=4. The manuscript correctly notes N=12 needed for significance, but still presents the comparison. Move to supplementary or remove.

**Results - TNBC Simulation:**
- The parameter-swap validation is clever but validates internal consistency, not external prediction. The IM vs. BLIS 2.6x difference reflects Jiang 2019 parameterization—this is hypothesis confirmation, not novel discovery.

**Discussion - Power Analysis:**
- This section is excellent and should be in Results, not Discussion. All N-dependent claims should be contextualized with power analysis earlier.

**Discussion - Training Data Circular:**
- ViennaRNA circ-mode is used as training data source AND as physics solver baseline (Scheme 2). This circular use inflates performance metrics. Acknowledge this limitation or use independent baselines.

### Questions for the Authors

1. Explain the contradictory immunogenicity correlations: r=0.91 (Chen 2019), r=0.68 (HEK293), r=-0.056 (case studies). Why does the model fail on case studies?

2. Why does MDA5/dsRNA pathway achieve 0% classification accuracy when this is claimed as the "primary pathway for circRNA immunogenicity"? What sequence features are missing for dsRNA structure prediction?

3. Define the tolerance threshold for PK "100% pass rate." The endosomal escape 158% error should not qualify as pass without justification.

4. Address training data circularity: ViennaRNA circ-mode is used for training AND as baseline (Scheme 2). How does this inflate performance?

5. The pathway weights (MDA5=0.35, TLR7/8=0.30) and m6A suppression coefficients (MDA5=0.90, TLR7/8=0.30) need derivation. How were these values determined?

6. Will wet-lab validation (IFN-β ELISA, qRT-PCR half-life, PDX models) be available before publication? If results contradict computational predictions, how will the manuscript be revised?

### Recommendation

**Major Revision**

The manuscript is transparent about statistical limitations, which is commendable. However, the validation shows contradictory results across datasets, the primary pathway (MDA5) achieves 0% accuracy, and case studies show negative correlation. Before acceptance:

1. Explain contradictory immunogenicity correlations (r=0.91 vs r=-0.056)
2. Address MDA5 pathway 0% accuracy—does this undermine pathway-resolved contribution?
3. Define PK validation tolerance thresholds explicitly
4. Address training data circularity (ViennaRNA used for training and baseline)
5. Move Power Analysis from Discussion to Results to contextualize all claims
6. Consider waiting for wet-lab validation to resolve computational contradictions

---

## Editorial Summary

**Overall Assessment**: The platform architecture is innovative and addresses critical accessibility gaps in computational biology. The circRNA-specific PK model and pathway-resolved immunogenicity are novel contributions. However, validation relies on extremely small sample sizes (N=4-7) with contradictory results across datasets. The primary pathway (MDA5) achieves 0% accuracy, undermining the pathway resolution claim.

**Consensus Recommendation**: **Major Revision**

**Key Issues to Address**:
1. Contradictory immunogenicity validation results
2. MDA5 pathway 0% accuracy explanation
3. Training data circularity (ViennaRNA used for training and baseline)
4. PK validation tolerance definition
5. Wet-lab validation timing

**Decision**: Request major revision addressing statistical limitations, contradictory validation, and pathway accuracy before reconsidering for publication.