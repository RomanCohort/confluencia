# TorusFold Research Architecture

## Systematic Benchmarking of 7 Architectures for circRNA 3D Structure Prediction

**Target venue:** Nature Methods
**Date:** 2026-06-23
**Status:** Research design phase

---

## 1. Key Research Questions

### RQ1: Torus Topology Advantage
Does explicit torus topology encoding (TPE + circular relative bias) improve circRNA 3D structure prediction over linear positional encoding, and if so, where (BSJ-flanking region, global fold, long-range contacts)?

### RQ2: Architecture Comparison
Among 7 deep learning architectures for circRNA 3D prediction, which design principles (equivariance, diffusion, iterative refinement, physics bias, SSM-attention hybrid) produce the most accurate and physically valid structures?

### RQ3: Length Scaling
How do the architectures scale with sequence length (50-2000 nt), and does the O(L) Mamba-based architecture (S7) outperform O(L^2) transformer-based architectures on long circRNAs (>500 nt)?

### RQ4: Data Quality Ceiling
What is the ceiling on prediction accuracy imposed by current training data (pseudo-labels vs. experimental structures), and how many experimental circRNA structures are needed to break through?

### RQ5: Physics-DL Trade-off
What is the optimal balance between physics-based constraints (guaranteed closure, bond geometry) and learned representations (flexible, data-driven), and can a hybrid approach capture the best of both?

### RQ6: Generalization to Unseen Topologies
Can models trained primarily on synthetic data generalize to experimentally determined circRNA structures (PDB 9H8A and future entries), and does torus topology encoding improve this generalization?

---

## 2. Benchmark Methodology

### 2.1 Datasets

| Dataset | N | Quality | Source | Role |
|---------|---|---------|--------|------|
| **PDB Experimental** | 7-40 | conf ~0.95 | 9H8A + 8xtp/8xtq/8xtr/8xts/9is7 | Primary test set (gold standard) |
| **IsRNAcirc Predicted** | 34 | conf ~0.7 | IsRNAcirc web server | Secondary test set (physics-validated) |
| **circrna_3d_merged** | ~11,000 | conf ~0.5 | BGSU + synthetic pseudo-labels | Training + tertiary test set |
| **Circ-CASP Competition** | 30 | conf ~0.9 | Physical high-fidelity simulation | Blind test set (available July 2026) |
| **Synthetic Benchmark** | 200 | conf ~1.0 | ViennaRNA circ + 3dRNA | Controlled ablation (known ground truth) |
| **Length-Stratified** | 500 | varies | Generated: 50/100/200/500/1000/2000 nt | Length-scaling analysis |

**Dataset construction pipeline (existing):**
- `fetch_rna_3d_data.py` -- download from BGSU + PDB
- `generate_circrna_pseudo_labels.py` -- ViennaRNA circ-mode pseudo-labels for 140K circBase sequences
- `validate_against_gold_standard.py` -- PDB 9H8A + IsRNAcirc validation

### 2.2 Metrics

**Primary (structure accuracy):**

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| RMSD | sqrt(mean(sum((pred - true)^2))) | Global structural deviation |
| TM-score | Length-normalized RMSD | Size-independent fold similarity |
| GDT-TS | % residues within distance thresholds | Tolerant global accuracy |
| LDDT | Local distance difference test | Local accuracy (no superposition) |

**Secondary (circRNA-specific):**

| Metric | Formula | Why it matters |
|--------|---------|---------------|
| BSJ Closure Error | abs(norm(p_0 - p_{N-1}) - 5.9 A) | Defining property of circRNA |
| Circular Distance d_circ(i,j) | min(abs(i-j), L-abs(i-j)) | Topology-aware distance metric |
| BSJ-Flanking RMSD | RMSD over positions within 20nt of BSJ | Region where topology matters most |
| Base-Pair F1 | 2TP/(2TP+FP+FN) on WC/GU pairs | Secondary structure recovery |

**Tertiary (physics validity):**

| Metric | Target | Why |
|--------|--------|-----|
| Bond length error | < 0.5 A (target: 5.9 A) | Backbone geometry |
| Clash count | 0 | Steric validity |
| Stacking energy | Near-native | Base stacking quality |
| Ramachandran-like | - | Torsion angle validity |

**Quaternary (computational):**

| Metric | Why |
|--------|-----|
| Inference time per target | Practical usability |
| Peak GPU memory | Hardware requirements |
| Parameter count | Model complexity |
| FLOPs | Algorithmic efficiency |

### 2.3 The 7 Architectures (Schemes)

| Scheme | Architecture | Key Innovation | Complexity | Status |
|--------|-------------|----------------|------------|--------|
| S1 | EGNN + Physics Refinement | Equivariant GNN with KNN graph | O(N^2) | Trained |
| S2 | Pure Physics Solver | Geometric constraint solver (zero-training) | O(N) | Complete |
| S3 | Dual-Engine Iterative | DL + physics iterative refinement | O(N^2 * K) | Not trained |
| S4 | DDPM + EGNN Guided Diffusion | Conditional diffusion with physics conditioning | O(N^2 * T) | In progress |
| S5 | CircPairformer + Physics Bias | AF3-style pairformer with physics constraints | O(N^2) | Unstable |
| S6 | GNN + Latent Diffusion | Latent space diffusion with GNN encoder/decoder | O(N^2 * T) | Best performer |
| S7 | Mamba + Attention Hybrid | O(L) SSM backbone with local attention | O(N) | Not trained |

### 2.4 External Baselines (currently MISSING)

| Baseline | Type | Why needed | Availability |
|----------|------|-----------|-------------|
| **IsRNAcirc** | Physics-based (Rosetta) | Published circRNA-specific predictor, standard comparison | Web server, free |
| **Rosetta FARFAR2** | Fragment assembly | State-of-the-art for linear RNA 3D, needed to show circRNA gap | Open source |
| **AlphaFold3** | DL (protein/RNA) | General biomolecular predictor, shows what generic DL achieves | Server access |
| **RoseTTAFold2** | DL (3-track) | Alternative DL baseline, faster than AF3 | Open source |
| **RNAstructure** | Thermo + 3D | Classical RNA folding, established baseline | Open source |
| **SimRNA** | Coarse-grained MC | Established RNA 3D predictor | Open source |
| **Linear baseline** | Same arch, standard PE | Ablation: isolate TPE contribution | Implement internally |
| **Random + Helical** | Physics heuristic | Lower bound (Circ-CASP M1) | Implement internally |

---

## 3. Research Gaps

### 3.1 Missing Comparisons

**GAP-1: No comparison to published circRNA 3D predictors.**
The IsRNAcirc method (Zhang et al., 2022) is the only published tool specifically designed for circRNA 3D structure. Without comparing to it, we cannot claim superiority for circRNA-specific DL. This is a fatal gap for Nature Methods.

**GAP-2: No comparison to AlphaFold3 on circRNA.**
AF3 now handles RNA. Running AF3 on circRNA sequences (without telling it they are circular) and comparing to TorusFold quantifies the value of torus-aware encoding. Expected: AF3 fails at BSJ closure and BSJ-flanking contacts.

**GAP-3: No comparison to Rosetta FARFAR2.**
FARFAR2 is the standard for RNA 3D de novo prediction. Running it on circRNA sequences (with and without circularity constraints) establishes a physics-based upper bound.

**GAP-4: No linear-PE ablation.**
The core claim is that TPE helps, but the only existing comparison is the proxy experiment (MSE on pairing probabilities). We need full 3D structure prediction with TPE vs. standard PE on identical architectures.

**GAP-5: No comparison across backbone encoders.**
Current experiments use ESM2 (frozen). Need to test: RNA-FM, RiNALMo, EVO-2, and random embeddings to separate backbone contribution from architecture contribution.

### 3.2 Missing Datasets

**GAP-6: No RNA-Puzzles circRNA subset.**
RNA-Puzzles (Cruz et al.) provides blind test cases for RNA 3D. We need to identify which RNA-Puzzles targets are circular or can be circularized, and benchmark on those.

**GAP-7: No experimentally determined circRNA structure dataset beyond 9H8A.**
As of 2026-06, PDB 9H8A is the only experimentally solved circRNA structure. We need:
- All available RNA structures with covalently closed backbones
- Structures from cryo-EM with circular topology
- NMR structures of circular ribozymes

**GAP-8: No synthetic benchmark with controlled difficulty.**
We need a synthetic benchmark where we systematically vary:
- Sequence length (50, 100, 200, 500, 1000, 2000)
- GC content (30%, 50%, 70%)
- Number of stem-loops (1, 2, 3, 5, 10)
- BSJ proximity to paired regions (near, far)
- Presence of IRES / inverted repeats

This is partially addressed by `generate_circrna_pseudo_labels.py` but needs systematic construction.

**GAP-9: No cross-organism validation.**
circRNA structures may differ across species due to different splicing machinery. Need sequences from human, mouse, and model organisms in the test set.

### 3.3 Missing Analyses

**GAP-10: No ablation study.**
Critical for Nature Methods. Required ablations:
- TPE vs. standard PE (full 3D prediction, not just pairing MSE)
- TPE 1D vs. TPE 2D vs. TPE 3D
- Learnable vs. fixed harmonic weights in TPE
- Number of harmonics: 4, 8, 16, 32, 64
- CircularRelativeBias vs. standard relative bias
- CircPairformer depth: 1, 2, 4, 8 blocks
- With vs. without circular distance features in pair initialization
- With vs. without BSJ rotational consistency (the 0.9/0.1 mixing)
- Diffusion vs. simple structure head vs. physics head

**GAP-11: No error analysis by region.**
Need to decompose prediction error by:
- BSJ-flanking region (within 20nt of junction)
- Stem regions (paired)
- Loop regions (unpaired)
- Long-range contact regions (d_circ > L/4)
- Per-nucleotide error heatmap

**GAP-12: No length dependency analysis.**
Current experiments test 50-500 nt. Nature Methods requires showing how performance scales:
- RMSD vs. sequence length curve for each architecture
- Closure error vs. length
- Inference time vs. length
- Memory vs. length
- Identify the "crossover point" where S7 (O(L)) surpasses S1-S6 (O(L^2))

**GAP-13: No confidence calibration.**
For each prediction, we output a confidence score. Is it calibrated?
- Reliability diagram: predicted confidence vs. actual RMSD
- Expected vs. observed error stratified by confidence bins
- This is critical for practical deployment

**GAP-14: No ensemble / multi-sample analysis.**
Diffusion-based methods (S4, S6) can generate multiple conformations. Need to analyze:
- Does the best-of-K strategy improve over single-sample?
- What is the diversity of sampled conformations?
- Is the ensemble capturing conformational heterogeneity?

**GAP-15: No secondary structure recovery analysis.**
3D structure implicitly contains secondary structure. Need to:
- Extract base pairs from predicted 3D (distance < 10.6 A for WC pairs)
- Compare to ViennaRNA circular-mode secondary structure
- Measure base-pair F1, PPV, sensitivity

**GAP-16: No topology preservation analysis.**
The defining property of circRNA is circular topology. Need:
- Writhe and linking number of predicted vs. true structure
- Gaussian knot analysis
- Whether BSJ-crossing contacts are preserved

---

## 4. Experimental Design

### 4.1 Experiment Matrix

| Experiment | Purpose | Schemes | Dataset | Metrics | Priority |
|-----------|---------|---------|---------|---------|----------|
| E1 | Main comparison | All 7 + baselines | PDB + IsRNAcirc | RMSD, TM-score, closure, GDT-TS, LDDT | P0 |
| E2 | TPE ablation | S5-TPE vs. S5-stdPE | PDB + synthetic | RMSD, BSJ-RMSD, pair F1 | P0 |
| E3 | Length scaling | All 7 | Length-stratified | RMSD, time, memory vs. L | P0 |
| E4 | Physics validity | All 7 | PDB | Bond error, clashes, energy | P0 |
| E5 | External baseline comparison | Best TorusFold vs. IsRNAcirc/AF3/FARFAR2 | PDB + IsRNAcirc | RMSD, closure, pair F1 | P0 |
| E6 | Data quality ceiling | S6 (best) | circrna_3d (conf=0.5) vs. PDB (conf=0.95) | RMSD gap | P1 |
| E7 | Ablation: TPE components | S5 variants | synthetic | All metrics | P1 |
| E8 | Ablation: architecture depth | S5-S7 variants | synthetic | All metrics | P1 |
| E9 | Confidence calibration | S4, S6 | PDB | Calibration curves | P1 |
| E10 | Ensemble analysis | S4, S6 | PDB | Best-of-K, diversity | P2 |
| E11 | Secondary structure recovery | All 7 | PDB + ViennaRNA | Pair F1, PPV, sensitivity | P1 |
| E12 | Topology preservation | All 7 | PDB | Writhe, crossing contacts | P2 |
| E13 | Cross-backbone comparison | S5 + RNA-FM/RiNALMo/ESM2 | PDB | RMSD, classification | P2 |
| E14 | Circ-CASP blind test | All 7 | Competition (30 targets) | Competition score | P0 (July) |

### 4.2 Experiment E1: Main Comparison Protocol

**Step 1: Prepare test sets**
- Download PDB files: 9H8A, 8xtp, 8xtq, 8xtr, 8xts, 9is7
- Run IsRNAcirc web server on all test sequences to get baseline predictions
- Extract C3' (or P) atom coordinates as 1-bead representation
- For PDB structures: use mmCIF parser to extract backbone coordinates

**Step 2: Run all 7 schemes**
- For each test sequence, run each scheme 5 times with different seeds
- Record: coords, confidence, closure distance, inference time, peak memory
- Use Kabsch alignment before computing RMSD

**Step 3: Run external baselines**
- IsRNAcirc: submit to web server, download predictions
- FARFAR2: `rosetta_scripts.default.linuxgccrelease` with FARFAR2 flags
- AF3: submit to AlphaFold Server (or run locally if weights available)
- For AF3: run both with and without "this is circular" hint

**Step 4: Compute all metrics**
- Primary: RMSD, TM-score, GDT-TS, LDDT
- Secondary: BSJ closure error, BSJ-flanking RMSD, pair F1
- Tertiary: bond error, clash count, stacking energy

**Step 5: Statistical comparison**
- Paired Wilcoxon signed-rank test for each pair of methods (small N)
- Report effect sizes (Cohen's d)
- Bootstrap confidence intervals for all metrics (1000 resamples)

### 4.3 Experiment E2: TPE Ablation Protocol

**Conditions:**
| Condition | Positional Encoding | Relative Bias | Distance Feature |
|-----------|--------------------|--------------|-----------------|
| Full TPE | TPE (learnable, 16 harmonics) | CircularRelativeBias | circular_distance |
| No TPE | Standard sinusoidal PE | Standard relative bias | linear distance |
| TPE only | TPE | Standard relative bias | linear distance |
| Circular bias only | Standard PE | CircularRelativeBias | circular_distance |
| Fixed TPE | TPE (fixed weights, 16 harmonics) | CircularRelativeBias | circular_distance |

**Architecture:** Use S5 (CircPairformer) as the base, swap PE components.

**Dataset:** PDB test set + 200 synthetic sequences (controlled length/GC/pairing).

**Metrics:** RMSD, BSJ-RMSD, pair F1, closure error -- with stratified analysis by region.

### 4.4 Experiment E3: Length Scaling Protocol

**Lengths tested:** 50, 100, 200, 500, 1000, 2000 nt

**Per length:** 10 random sequences (same seeds across all schemes)

**Measured:**
- RMSD, closure error, pair F1
- Wall-clock inference time (single GPU)
- Peak GPU memory (tracemalloc + torch.cuda.max_memory_allocated)
- Number of parameters

**Analysis:**
- Fit power-law: metric = a * L^b for each scheme
- Identify crossover point where S7 becomes competitive
- Plot all metrics on log-log scale

---

## 5. Statistical Analyses

### 5.1 Required Tests

| Test | Application | Conditions |
|------|------------|-----------|
| **Paired Wilcoxon signed-rank** | Compare two methods on same targets (small N) | Non-parametric, no normality assumption |
| **Friedman test** | Compare 7+ methods across multiple targets | Non-parametric repeated-measures ANOVA |
| **Nemenyi post-hoc** | Pairwise comparisons after Friedman | Controls family-wise error |
| **Bootstrap CI** | Confidence intervals for all metrics | 1000 resamples, BCa method |
| **Cohen's d** | Effect size between TPE vs. standard PE | Report alongside p-values |
| **Pearson correlation** | Length vs. RMSD, confidence vs. accuracy | Check linearity first |
| **Spearman rank correlation** | Monotonic but non-linear relationships | More robust than Pearson |
| **Kruskal-Wallis** | Compare 3+ groups (e.g., TPE variants) | Non-parametric one-way ANOVA |
| **McNemar's test** | Compare binary predictions (paired/not-paired) | For secondary structure |
| **ICC (intraclass correlation)** | Inter-rater reliability across scheme rankings | Consistency of method ranking |

### 5.2 Multiple Comparison Correction

- Bonferroni correction for planned pairwise comparisons
- Benjamini-Hochberg FDR for exploratory analyses
- Report both raw and adjusted p-values

### 5.3 Effect Size Requirements for Nature Methods

- Minimum: report Cohen's d for all pairwise comparisons
- Preferred: also report common language effect size (probability that a randomly selected prediction from method A is better than from method B)
- For non-significant results: report Bayes factor to quantify evidence for null

### 5.4 Sample Size Justification

- PDB test set: N=7-40 (limited by available experimental structures)
  - With N=7, minimum detectable effect at 80% power: d >= 1.4 (large)
  - With N=40, minimum detectable effect: d >= 0.5 (medium)
- Synthetic benchmark: N=200, powered to detect d >= 0.2 (small)
- circrna_3d test: N=11,000, powered for very small effects (d >= 0.02)

---

## 6. Figures and Tables Plan

| Figure | Content | Type |
|--------|---------|------|
| Fig 1 | TorusFold architecture overview (S7 with all components) | Schematic |
| Fig 2 | TPE visualization: periodic encoding vs. standard PE | Heatmap |
| Fig 3 | Main comparison: RMSD bar chart across 7 schemes + baselines | Bar + scatter |
| Fig 4 | Length scaling: RMSD, time, memory vs. L (log-log) | Multi-panel line |
| Fig 5 | TPE ablation: BSJ-RMSD by condition | Box plot |
| Fig 6 | Error analysis heatmap: per-nucleotide error on representative structure | Heatmap |
| Fig 7 | Confidence calibration: reliability diagram | Line plot |

| Table | Content |
|-------|---------|
| Table 1 | Dataset composition and statistics |
| Table 2 | Main results: all metrics for all schemes on PDB test set |
| Table 3 | External baseline comparison (TorusFold vs. IsRNAcirc vs. AF3 vs. FARFAR2) |
| Table 4 | TPE ablation results |
| Table 5 | Computational cost comparison |
| Table 6 | Length-stratified results |

---

## 7. Risk Assessment and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Only 7 PDB structures available | Low statistical power | High | Augment with IsRNAcirc + synthetic + Circ-CASP |
| S7 not trained yet | Missing data point | High | Prioritize S7 training -- it is the novel contribution |
| AF3 refuses to close BSJ | Trivial comparison | Medium | Frame as "demonstrating the gap" not "beating AF3" |
| TPE shows no improvement over standard PE | Core claim weakened | Low-Medium | Already shown 5-15% improvement in proxy experiment; if 3D shows nothing, analyze why |
| Reviewers demand blind test | No held-out data | Medium | Circ-CASP provides 30 blind targets (July 2026) |
| Pseudo-label data quality too low | Unfair to all methods | High | Stratify analysis by data confidence; show ceiling effect |
| Schemes 3, 4, 7 not trained | Incomplete comparison | High | Must train all 7 before submission |

---

## 8. Timeline and Priorities

### Phase 1: Foundation (Weeks 1-2)
- [ ] Complete training of S3, S4, S7 (currently untrained)
- [ ] Download and prepare PDB test set (9H8A + intron structures)
- [ ] Run IsRNAcirc web server on all test sequences
- [ ] Implement Kabsch RMSD, TM-score, GDT-TS, LDDT metrics

### Phase 2: Core Experiments (Weeks 3-5)
- [ ] E1: Run all 7 schemes on PDB + IsRNAcirc test sets
- [ ] E2: TPE ablation (5 conditions)
- [ ] E3: Length scaling (6 lengths x 10 sequences x 7 schemes)
- [ ] E4: Physics validity metrics for all schemes
- [ ] E5: External baseline comparison (IsRNAcirc, FARFAR2, AF3)

### Phase 3: Analysis (Weeks 6-7)
- [ ] E6: Data quality ceiling analysis
- [ ] E7: TPE component ablation
- [ ] E11: Secondary structure recovery
- [ ] Statistical analysis: Friedman, Nemenyi, bootstrap CIs
- [ ] Generate all figures and tables

### Phase 4: Circ-CASP (Weeks 8-9, July 2026)
- [ ] E14: Blind test on 30 competition targets
- [ ] Integrate Circ-CASP results into paper

### Phase 5: Writing (Weeks 10-12)
- [ ] Draft manuscript
- [ ] Internal review
- [ ] Submit to Nature Methods

---

## 9. Novelty Claims (What Makes This Nature Methods)

1. **First systematic benchmark** of DL architectures for circRNA 3D structure prediction (7 architectures, 4+ baselines, 6 datasets)
2. **Torus Positional Encoding** -- first periodic positional encoding that enforces PE[i] = PE[i+L] for circular topology
3. **Circular Relative Bias** -- first attention mechanism that uses circular distance d_circ(i,j) = min(|i-j|, L-|i-j|)
4. **BSJ-closure as evaluation criterion** -- introducing BSJ closure error as a standard metric for circRNA structure prediction
5. **Circ-CASP competition** -- first community benchmark for circRNA structure prediction (30 blind targets)
6. **Length-scaling analysis** -- first demonstration that O(L) architectures (Mamba) outperform O(L^2) architectures on long circRNAs
7. **Data quality ceiling** -- first quantitative analysis of how pseudo-label quality limits circRNA 3D prediction

---

## 10. Current Implementation Status

| Component | File | Status |
|-----------|------|--------|
| TPE (1D, 2D, CircularRelativeBias) | `core/tpe.py` | Complete |
| CircEquivariantBackbone | `core/equivariant_backbone.py` | Complete |
| CircPairformerStack | `core/triangle_update.py` | Complete |
| Diffusion structure head | `core/diffusion_structure.py` | Complete |
| Physics structure head | `core/physics_structure_head.py` | Complete |
| BSJ pair analyzer | `core/irs_pair.py` | Complete |
| TorusFold v2 (integrated) | `core/torusfold.py` | Complete |
| Benchmark script | `scripts/benchmark_torusfold.py` | Complete (classification only) |
| Scheme benchmark | `manuscripts/scripts/benchmark_schemes.py` | Complete (6 schemes, no gold standard) |
| Gold standard validation | `manuscripts/scripts/validate_against_gold_standard.py` | Partial (PDB loader, no IsRNAcirc runner) |
| Proxy experiment | `manuscripts/scripts/torusfold_proxy_experiment.py` | Complete (pairing MSE only) |
| Pretraining (enhanced) | `scripts/pretrain_torusfold_enhanced.py` | Complete |
| S3 (Dual-Engine) training | Not implemented | BLOCKING |
| S4 (DDPM) training | In progress | BLOCKING |
| S7 (Mamba hybrid) training | Not implemented | BLOCKING |

**Three schemes untrained = incomplete paper. This is the critical path.**
