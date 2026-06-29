# Peer Review Report: TorusFold
## Submitted to *Nature Methods*

---

## Reviewer #1: Deep Learning for Molecular Structure Expert

### Summary

This manuscript presents TorusFold, a systematic exploration of eight deep learning architectures for circular RNA 3D structure prediction under extreme data scarcity (0 experimental circRNA structures in PDB). The work introduces Torus Positional Encoding (TPE) to guarantee circular periodicity and evaluates architectures spanning equivariance (EGNN), diffusion models, attention mechanisms, and physics-based solvers. Scheme 6 (GNN latent diffusion) achieves 13.91Å RMSD with 0.02Å closure error on a PDB-derived circularized test set (N=7).

### Major Strengths

1. **Torus Positional Encoding innovation**: The periodic encoding TPE(i)=TPE(i+L) elegantly solves the circular topology violation in standard positional encodings. The mathematical derivation is clear, and empirical verification (max deviation <10⁻⁶) demonstrates implementation correctness. This is a fundamental contribution applicable to any circular sequence modeling.

2. **Comprehensive architectural exploration**: Eight schemes spanning four paradigm groups (physics-first, equivariance-based, generative diffusion, sequence-first) provide systematic comparison. The failure analysis (Scheme 5 coordinate explosion, Scheme 3 gradient divergence) identifies necessary conditions (geometric inductive bias, bounded output magnitude) for stable circRNA prediction.

3. **Diffusion models learn physical constraints**: Scheme 6 achieves 0.02Å closure without explicit penalty—the diffusion model implicitly learns closure from data distribution. The ablation experiment (training on linear RNA produces 2.3Å closure) validates this emergent property. This finding extends to other diffusion-based structure prediction.

4. **Data quality dominates accuracy**: The 11Å improvement from high-confidence PDB data (N=7) vs. heterogeneous pseudo-labels (N≈14,000) demonstrates that architectural ceiling hasn't been reached—training data quality is the bottleneck. This insight is critical for the circRNA field.

5. **Evaluation protocols for circRNA**: BSJ closure error and circular distance metrics are standardized for the first time. The Circ-CASP benchmark provides community infrastructure for future method comparison.

### Major Weaknesses

1. **Extremely small test set (N=7)**: The PDB circularized test set contains only 7 sequences (lengths 20-27 nt). This severely limits statistical power—no significance testing can distinguish Scheme 1 (13.85Å) from Scheme 6 (13.91Å) with N=7. The manuscript correctly acknowledges this, but N=7 is insufficient for Nature Methods standards.

2. **Missing external baseline comparisons**: The manuscript claims comparison with IsRNA, AlphaFold3, and FARFAR2 is "pending" but includes Figure 4 showing comparison. This contradiction is confusing. Either complete the comparison or remove Figure 4.

3. **Scheme 6 closure learning mechanism unclear**: The 0.02Å closure is impressive, but "learned end-to-end without explicit penalty" lacks mechanistic explanation. Does the latent diffusion enforce periodicity in latent space? Or does the decoder learn to output closed coordinates? The black-box nature limits interpretability.

4. **Scheme 2 circular validation concern**: ViennaRNA circ-mode pair probabilities are used as constraints for Scheme 2, but ViennaRNA predictions also appear in training data. This circular use of the same tool inflates baseline performance—Scheme 2 RMSD 25.47Å may be lower than true physics-only baseline.

5. **Pseudo-label training data quality**: 7024 synthetic pseudo-labels from IsRNA, ViennaRNA, icSHAPE are used as training data. These computational predictions contain systematic errors (IsRNA coarse-grained, ViennaRNA 2D-only). The manuscript acknowledges this but doesn't quantify label noise impact on RMSD ceiling.

### Specific Comments

**Introduction:**
- "Protein Data Bank contains no circRNA crystal structures" is correct, but the field may have NMR or cryo-EM circRNA structures not deposited in PDB. Verify this claim comprehensively.
- The gap statement "no deep learning method has been designed for circRNA 3D structure" is strong but should acknowledge recent works like RoseTTAFold2NA, AlphaFold3 RNA capabilities that could potentially adapt to circRNA.

**Results - TPE Periodicity:**
- The TPE derivation is elegant. For reproducibility, provide the code for TPE implementation or pseudocode.
- The empirical verification max deviation <10⁻⁶ is good, but provide the test script—this is easy to verify but should be reproducible.

**Results - Eight Architectures:**
- Table 1 shows Scheme 5 RMSD 245Å (failed). Include the per-sample RMSD distribution—did all samples fail catastrophically or only some?
- Scheme 2' (no pairs) RMSD 85.39Å vs. Scheme 2 25.47Å demonstrates secondary structure importance. This is a strong result—highlight it more prominently.

**Results - Scheme 6 Performance:**
- RMSD 13.91Å with closure 0.02Å is impressive, but N=7 limits significance. Provide bootstrap confidence intervals for mean RMSD.
- The ablation (training on linear RNA produces 2.3Å closure) validates emergent learning, but the mechanism is unclear. How does latent diffusion enforce periodicity?

**Results - Data Quality:**
- Figure 6 shows 11Å improvement from high-confidence data. This is the most important finding. Quantify the label noise in pseudo-labels—what percentage of IsRNA/ViennaRNA predictions are wrong?

**Discussion - Data Ceiling:**
- "1,624-sample ceiling" from PDB circularization is a fundamental limit. Acknowledge that even perfect architecture cannot overcome this—Nature Methods readers expect realistic assessments.

**Discussion - Sparse Attention:**
- The FlashAttention discussion is insightful. Scheme 8's theoretical O(L·K) vs. actual O(L²) GPU memory is a practical limitation. Propose concrete implementation: FlashAttention-2 with pair bias.

### Questions for the Authors

1. Complete the external baseline comparison (IsRNA, AlphaFold3, FARFAR2) or remove Figure 4. Which comparison is correct: "pending" or Figure 4 showing results?

2. Provide bootstrap confidence intervals for Scheme 6 RMSD (N=7). What is the 95% CI for mean 13.91Å?

3. Explain the mechanism for Scheme 6 closure learning: Does latent space have implicit periodicity? Or decoder outputs closed coordinates? Is this interpretable or black-box?

4. Quantify label noise in pseudo-label training data: What percentage of IsRNA/ViennaRNA predictions have large errors? How does noise impact RMSD ceiling?

5. Address ViennaRNA circularity: ViennaRNA circ-mode is used for Scheme 2 constraints AND appears in training data. Does this inflate Scheme 2 baseline performance?

6. Are there NMR or cryo-EM circRNA structures not in PDB? Verify "0 experimental circRNA structures" comprehensively.

7. Scheme 7 and 8 are "training in progress." Will results be available for final publication? Nature Methods expects complete experiments.

### Recommendation

**Major Revision**

The TPE innovation and systematic architectural comparison are excellent contributions to the circRNA structure field. The finding that diffusion models learn closure implicitly is novel and extends beyond circRNA. However, the test set N=7 is insufficient for Nature Methods standards, and external baseline comparison is incomplete/contradictory. Before acceptance:

1. Expand test set if possible, or acknowledge N=7 provides preliminary validation only (not for Nature Methods primary conclusion)
2. Complete IsRNA/AlphaFold3/FARFAR2 comparison or remove Figure 4
3. Explain Scheme 6 closure learning mechanism
4. Quantify pseudo-label noise and its impact
5. Address ViennaRNA circularity concern
6. Complete Scheme 7/8 training or acknowledge incomplete experiments

Consider publishing TPE as a Methods paper separate from the full architectural comparison, given the N=7 limitation.

---

## Reviewer #2: RNA Structure Prediction & Physics-Based Methods Expert

### Summary

This manuscript addresses circRNA 3D structure prediction through eight architectures, with a physics-based solver (Scheme 2) and deep learning approaches (Schemes 1, 6). The physics solver uses ViennaRNA circ-mode pair constraints with simulated annealing, achieving 25.47Å RMSD and 2.75Å closure. Scheme 6 (GNN latent diffusion) achieves better accuracy (13.91Å) with near-perfect closure (0.02Å).

### Major Strengths

1. **Physics solver as zero-training baseline**: Scheme 2 demonstrates that ViennaRNA pair constraints improve RMSD from 85.39Å (no pairs) to 25.47Å. This validates secondary structure as a strong prior—a critical insight for RNA structure methods.

2. **BSJ closure as structural definition**: The manuscript correctly treats closure <2Å as a structural requirement, not a prediction target. The physics solver hard constraint approach (E_closure penalty) guarantees closure, unlike learned methods that may violate it.

3. **Simulated annealing protocol well-designed**: T₀=1000K → T_f=10K with 1000 iterations and cooling schedule T_{n+1}=0.95×T_n is standard but appropriate. The energy function E=E_bond+E_pair+E_clash+E_closure captures essential RNA physics.

4. **Scheme 6 curriculum learning**: Three-phase curriculum (high-quality→mixed→long sequences) addresses data scarcity. Confidence-weighted loss prevents low-quality pseudo-labels from dominating training—sound ML practice.

5. **Failure analysis for Scheme 5**: The coordinate explosion to |x|>100Å within 50 epochs is a valuable negative result. Transformers lack geometric inductive bias for coordinate prediction—this identifies necessary conditions for RNA structure architectures.

### Major Weaknesses

1. **Physics solver performance mediocre**: 25.47Å RMSD is substantially worse than Scheme 6 (13.91Å). For a zero-training baseline claiming "guarantees closure," the accuracy gap suggests physics alone is insufficient. Is the simulated annealing under-optimized?

2. **ViennaRNA pair constraint quality**: Pair probability P_ij>0.3 threshold and K=5 pairs per position are arbitrary. How sensitive is RMSD to these parameters? Provide ablation: P_ij>0.5, K=10, etc.

3. **OpenMM refinement underutilized**: Scheme 1 uses OpenMM physics refinement with bond length constraints (P-O 1.6Å, C-C' 1.5Å) and BSJ restraint (5.9±0.5Å). Scheme 6 could benefit from similar refinement—why not apply physics post-processing to Scheme 6?

4. **EGNN vs. diffusion comparison unfair**: Scheme 1 (EGNN+Physics) achieves 13.85Å RMSD, comparable to Scheme 6 (13.91Å). But Scheme 1 has 5.36Å closure vs. Scheme 6's 0.02Å. Is closure the only difference? If physics refinement improves closure, would Scheme 1 match Scheme 6?

5. **Bond length variance unrealistic**: The manuscript claims "variances under 0.02Å in experimental structures" but OpenMM tolerance is 0.02Å—this means 0.02Å standard deviation, not variance. Correct the terminology.

### Specific Comments

**Methods - Scheme 2:**
- The simulated annealing protocol should include convergence criteria: How many iterations until energy stabilizes? What is the acceptance rate at final temperature?
- The pair selection P_ij>0.3 threshold is arbitrary. Provide sensitivity analysis: RMSD vs. threshold (0.2, 0.3, 0.4, 0.5).

**Methods - Scheme 1:**
- OpenMM L-BFGS 500 steps may be insufficient for full energy minimization. Typical RNA relaxation requires 1000-5000 steps. Benchmark convergence.
- BSJ restraint 5.9±0.5Å is weak (1Å range). Tighten to 5.9±0.1Å and measure RMSD impact.

**Methods - Scheme 6:**
- Cosine noise schedule (β_start=10⁻⁴, β_end=0.02) is standard for DDPM. For RNA structure, linear schedule may work better—provide comparison.
- Latent dimension d=256 is arbitrary. Ablation: d=128, d=512, measure RMSD and closure.

**Discussion - Physics Prior:**
- "Four categories of physics priors substitute for missing data" is insightful. Expand: Can hard bond constraints alone achieve <10Å RMSD with perfect secondary structure? Estimate theoretical lower bound.

**Discussion - Data Ceiling:**
- The 1624 PDB circularized samples ceiling is fundamental. But this ceiling applies to physics methods too—ViennaRNA training data comes from known RNA structures. Acknowledge physics methods also face data limits.

### Questions for the Authors

1. Why is physics solver RMSD 25.47Å substantially worse than learned methods? Is simulated annealing under-optimized? Provide convergence analysis.

2. Provide sensitivity analysis for ViennaRNA pair threshold (P_ij>0.2, 0.3, 0.4, 0.5) and K pairs per position (K=3, 5, 10). How robust is Scheme 2?

3. Apply OpenMM refinement to Scheme 6. Would physics post-processing improve RMSD or closure further? Why not combine learned+physics for best performance?

4. Scheme 1 (13.85Å) and Scheme 6 (13.91Å) have similar RMSD but different closure (5.36Å vs. 0.02Å). Is this the only difference? Can physics refinement close Scheme 1?

5. Correct bond length terminology: "variance under 0.02Å" means standard deviation 0.02Å, not variance. Confirm the correct statistical term.

6. Provide convergence criteria for simulated annealing: acceptance rate at final T, energy plateau detection, iteration count to convergence.

7. Can hard bond constraints + perfect secondary structure achieve theoretical lower bound for RMSD? Estimate this bound.

### Recommendation

**Minor Revision**

The physics-based approach and comparison with learned methods are valuable. The finding that secondary structure priors improve RMSD by 60Å (85.39→25.47) is important. However, physics solver optimization is incomplete, and sensitivity analysis for ViennaRNA parameters is missing. Before acceptance:

1. Optimize simulated annealing or explain why physics alone achieves 25.47Å
2. Provide ViennaRNA threshold sensitivity analysis
3. Apply physics refinement to Scheme 6 and benchmark improvement
4. Correct bond length terminology (variance vs. standard deviation)
5. Provide convergence criteria for annealing protocol
6. Estimate theoretical RMSD lower bound with perfect physics constraints

---

## Reviewer #3: Statistical Methods & Benchmarking Expert

### Summary

This manuscript presents eight architectures for circRNA 3D structure prediction, evaluated on a PDB-derived circularized test set (N=7, lengths 20-27 nt). Scheme 6 achieves RMSD 13.91Å±0.73Å with closure 0.02Å. The work introduces Torus Positional Encoding and demonstrates that data quality dominates accuracy (11Å improvement from high-confidence vs. pseudo-label data).

### Major Strengths

1. **Transparent acknowledgment of N=7 limitation**: The manuscript explicitly states "limited statistical power precludes definitive conclusions" in Limitations section. This honesty is commendable—many papers would present N=7 results as definitive.

2. **Curriculum learning design**: Three-phase curriculum (high-quality foundation→generalization expansion→long-sequence extension) with confidence-weighted loss is methodologically sound for data scarcity scenarios.

3. **Bootstrap approach suggested**: The manuscript could use bootstrap resampling for confidence intervals (though not currently implemented). N=7 allows bootstrap-based inference.

4. **Multi-source training data quality scoring**: Confidence scoring (PDB circularized=1.0, SHAPE=0.9, Rfam=0.8, IsRNAcirc=0.7, synthetic=0.3) provides transparency on training data reliability.

5. **Circ-CASP benchmark design**: Standardized metrics (RMSD, closure, bond consistency, pair F1) with hidden test set and baseline methods provides infrastructure for future rigorous comparison.

### Major Weaknesses

1. **No statistical significance testing**: Scheme 1 (13.85Å) vs. Scheme 6 (13.91Å) with N=7 cannot be distinguished statistically. Provide bootstrap confidence intervals or paired t-test (if same 7 samples).

2. **Bootstrap CI missing**: The manuscript provides mean±std for Scheme 6 (13.91±0.73Å) but no confidence interval. Bootstrap resampling from N=7 can estimate CI.

3. **Scheme 2 median vs. mean discrepancy**: RMSD 25.47Å (mean) vs. 23.35Å (median) suggests outlier influence. Provide per-sample RMSD distribution—is one sample catastrophically wrong?

4. **Test set length distribution narrow**: Lengths 20-27 nt are short. This may not generalize to therapeutic circRNA lengths (typically 500-2000 nt). Acknowledge this limitation explicitly.

5. **Scheme 7/8 incomplete**: "Training in progress" for Schemes 4, 7, 8 is unacceptable for Nature Methods. Either complete experiments or remove incomplete schemes from Table 1.

### Specific Comments

**Results - Scheme 6 Performance:**
- "Mean 13.91Å, Median 14.08Å, Std 0.73Å" needs confidence interval. Bootstrap 1000 samples from N=7:
  - Generate bootstrap sample (sample 7 with replacement)
  - Compute mean RMSD
  - Repeat 1000 times
  - Report 95% CI from bootstrap distribution
- Without CI, the comparison with Scheme 1 (13.85Å) is meaningless.

**Results - Scheme 2:**
- Mean 25.47Å vs. median 23.35Å suggests right-skewed distribution. Provide:
  - Per-sample RMSD table
  - Histogram/distribution
  - Identify outlier samples
- Is one sample catastrophically wrong (e.g., 60Å while others 23Å)?

**Results - Length Scaling:**
- Figure 7 shows memory vs. length, but RMSD vs. length is more critical. For Scheme 6, does RMSD increase with length? This is essential for therapeutic circRNA prediction (500-2000 nt).

**Table 1:**
- Scheme 4, 7, 8 listed as "Training†" with footnote. This is preliminary—Nature Methods expects complete experiments. Either complete or remove.

**Methods - Evaluation Metrics:**
- RMSD formula provided, but Kabsch alignment details missing. Provide:
  - Superimposition method (Kabsch algorithm reference)
  - Atom selection (all atoms vs. backbone only)
  - Treatment of BSJ in alignment

**Discussion - Limitations:**
- "Small test set (N=7)" is correctly acknowledged. Add: "Test set lengths (20-27 nt) may not generalize to therapeutic circRNA (500-2000 nt)."

### Questions for the Authors

1. Provide bootstrap 95% CI for Scheme 6 mean RMSD (N=7). How does CI overlap with Scheme 1 (13.85Å)? Can they be statistically distinguished?

2. Explain Scheme 2 mean-median discrepancy (25.47 vs. 23.35). Provide per-sample RMSD distribution and identify outliers.

3. Does Scheme 6 RMSD increase with sequence length? Test on longer sequences (>50 nt) if available, or acknowledge length limitation.

4. Complete Schemes 4, 7, 8 training or remove from Table 1. Nature Methods does not publish incomplete experiments with "training in progress."

5. Provide Kabsch alignment details: atom selection (all vs. backbone), BSJ treatment in superimposition.

6. How many bootstrap samples are needed for stable CI estimation with N=7? Provide technical details.

7. Will Circ-CASP competition include longer sequences (>100 nt) for therapeutic relevance? The current test set (20-27 nt) is unrealistic for vaccines.

### Recommendation

**Major Revision**

The TPE innovation and systematic architectural comparison are valuable, but the manuscript lacks statistical rigor for Nature Methods standards. N=7 test set cannot distinguish Scheme 1 from Scheme 6, Scheme 2 mean-median discrepancy suggests outliers, and Schemes 4/7/8 are incomplete. Before acceptance:

1. Provide bootstrap 95% CI for all RMSD means
2. Explain Scheme 2 mean-median discrepancy with per-sample distribution
3. Test RMSD vs. length or acknowledge length limitation
4. Complete Schemes 4/7/8 or remove incomplete experiments
5. Provide Kabsch alignment details
6. Address therapeutic circRNA length relevance (500-2000 nt vs. test set 20-27 nt)

Consider increasing test set size if possible, or acknowledge that N=7 provides preliminary validation insufficient for definitive conclusions in Nature Methods.

---

## Editorial Summary

**Overall Assessment**: Torus Positional Encoding is a fundamental innovation applicable beyond circRNA to any circular sequence modeling. The systematic architectural comparison provides valuable negative results (Scheme 5 failure conditions) and demonstrates diffusion models learn closure implicitly. However, the N=7 test set severely limits statistical significance, and external baseline comparison (IsRNA, AlphaFold3, FARFAR2) is incomplete/contradictory.

**Consensus Recommendation**: **Major Revision**

**Key Issues to Address**:
1. Bootstrap confidence intervals for RMSD means (N=7 cannot distinguish schemes)
2. Complete external baseline comparison or remove Figure 4
3. Scheme 2 mean-median discrepancy explanation
4. Complete Schemes 4/7/8 training or remove
5. Address therapeutic circRNA length relevance (test set 20-27 nt vs. therapeutic 500-2000 nt)
6. ViennaRNA circularity concern (used for training and baseline)

**Decision**: The TPE contribution is strong enough for Nature Methods, but the architectural comparison needs statistical rigor and complete experiments. Request major revision addressing statistical significance, baseline comparison completeness, and length generalization before reconsidering. Consider splitting into two papers: TPE as a Methods paper, and architectural comparison as a separate comprehensive study with larger test set.