# TorusFold 实验结果模板
## Critical Experiments Results (To Be Filled)

**Status:** Template ready for data filling
**Last Updated:** 2026-06-27

---

## 📊 Experiment 1: External Baseline Comparisons

### Results Summary Table

| Method | RMSD Mean (Å) | RMSD Std (Å) | RMSD Median (Å) | Closure Mean (Å) | Closure Std (Å) | Inference Time (s) | N Samples | Status |
|--------|---------------|--------------|-----------------|------------------|-----------------|--------------------|-----------|--------|
| **Scheme 6 (TorusFold)** | 13.91 | 0.73 | 14.08 | 0.02 | 0.01 | 45 | 7 | ✅ Trained |
| **Scheme 2 (Physics)** | 25.47 | 1.20 | 23.35 | 2.75 | 0.30 | 60 | 7 | ✅ Ready |
| **IsRNA** | TBD | TBD | TBD | TBD | TBD | TBD | 7 | ⏳ Pending |
| **FARFAR2** | TBD | TBD | TBD | TBD | TBD | TBD | 7 | ⏳ Pending |
| **AlphaFold3** | TBD | TBD | TBD | TBD | TBD | TBD | 7 | ⏳ Pending |
| **Random Baseline** | TBD | TBD | TBD | TBD | TBD | TBD | 7 | ✅ Baseline |

**Note:** All external baseline metrics will be populated after experiments complete.

### Statistical Comparison (Paired t-tests)

| Comparison | t-statistic | p-value | Significant? |
|------------|-------------|---------|--------------|
| Scheme 6 vs IsRNA | TBD | TBD | TBD |
| Scheme 6 vs FARFAR2 | TBD | TBD | TBD |
| Scheme 6 vs AF3 | TBD | TBD | TBD |
| Scheme 6 vs Scheme 2 | TBD | TBD | TBD |

**Note:** Statistical tests will be computed after baseline runs complete.

### Key Observations

**Note:** Observations will be populated after external baseline experiments complete.

1. **Closure Performance:**
   - TorusFold: 0.02Å (trained)
   - Physics Solver: 2.75Å (ready)
   - External baselines: TBD (pending experiments)

2. **RMSD Trade-off:**
   - TorusFold: TBD (pending comparison)
   - Key advantage: closure + RMSD balance

3. **Computational Efficiency:**
   - TorusFold: 45s inference
   - External methods: TBD (pending timing experiments)

---

## 📊 Experiment 2: TPE Ablation Study

### Global Performance Comparison

| Metric | TPE | Standard PE | Difference | p-value | Significant? |
|--------|-----|-------------|------------|---------|--------------|
| **Global RMSD (Å)** | TBD | TBD | TBD | TBD | TBD |
| **RMSD Std (Å)** | TBD | TBD | TBD | - | - |
| **RMSD 95% CI Lower** | TBD | TBD | TBD | - | - |
| **RMSD 95% CI Upper** | TBD | TBD | TBD | - | - |
| **Closure Error (Å)** | TBD | TBD | TBD | TBD | TBD |

### BSJ-Flanking Region Performance

**Definition:** Positions within 10 nt of junction (0-10, L-10 to L-1)

| Metric | TPE | Standard PE | Improvement | p-value |
|--------|-----|-------------|-------------|---------|
| **BSJ-flanking RMSD (Å)** | TBD | TBD | TBD | TBD |
| **BSJ-flanking Std (Å)** | TBD | TBD | TBD | - |
| **% of Total Error** | TBD% | TBD% | TBD% | - |

### Per-Nucleotide Error Analysis

| Position Range | TPE RMSD (Å) | Std PE RMSD (Å) | Δ Improvement |
|----------------|--------------|-----------------|---------------|
| **0-5 (near BSJ)** | TBD | TBD | TBD |
| **5-15** | TBD | TBD | TBD |
| **15-25** | TBD | TBD | TBD |
| **25-35** | TBD | TBD | TBD |
| **35-45** | TBD | TBD | TBD |
| **L-10 to L-1 (near BSJ)** | TBD | TBD | TBD |

### Hypotheses to Test

**H1: TPE reduces BSJ-flanking error**
- Hypothesis: TPE may show improvement at boundaries
- Mechanism: Periodic encoding TPE(i)=TPE(i+L) may reduce boundary discontinuity
- Validation: TBD (pending ablation experiment)

**H2: TPE global effect**
- Hypothesis: Global RMSD effect TBD
- Validation: TBD (pending ablation experiment)

**H3: Closure error with different PE**
- Hypothesis: Closure error TBD
- Validation: TBD (pending ablation experiment)

---

## 📊 Experiment 3: Error Analysis by Structural Region

### RMSD by Region (Scheme 6, TPE)

| Structural Region | Mean RMSD (Å) | Std Dev (Å) | % of Positions | % of Total Error |
|--------------------|---------------|-------------|----------------|------------------|
| **BSJ-flanking** | TBD | TBD | TBD | TBD |
| **Stems (base-paired)** | TBD | TBD | TBD | TBD |
| **Loops/Hairpins** | TBD | TBD | TBD | TBD |
| **Internal Loops** | TBD | TBD | TBD | TBD |
| **Single-stranded** | TBD | TBD | TBD | TBD |

**Note:** Regional analysis will be computed after post-hoc analysis.

### TPE vs Standard PE: Regional Error Differences

| Region | TPE RMSD (Å) | Std PE RMSD (Å) | Δ Improvement (Å) | % Improvement |
|--------|--------------|-----------------|-------------------|---------------|
| **BSJ-flanking** | TBD | TBD | TBD | TBD% |
| **Stems** | TBD | TBD | TBD | TBD% |
| **Loops/Hairpins** | TBD | TBD | TBD | TBD% |
| **Internal Loops** | TBD | TBD | TBD | TBD% |
| **Single-stranded** | TBD | TBD | TBD | TBD% |

### Error Distribution Summary

**Total Error Decomposition:**
- BSJ-flanking: TBD%
- Stems: TBD%
- Loops: TBD%
- Internal loops: TBD%
- Single-stranded: TBD%

**Key Finding:** Error distribution TBD (pending analysis)

---

## 📊 Experiment 4: Hyperparameter Sensitivity

### Harmonic Count H (TPE)

| H | Global RMSD (Å) | BSJ RMSD (Å) | Closure (Å) | Training Stability |
|---|-----------------|--------------|-------------|-------------------|
| **4** | TBD | TBD | TBD | TBD |
| **8** | TBD | TBD | TBD | TBD |
| **16 (default)** | 13.91 | TBD | 0.02 | Stable |
| **32** | TBD | TBD | TBD | TBD |

**Note:** Hyperparameter sensitivity will be characterized after primary experiments complete.

**Expected finding:** Optimal hyperparameters TBD.

### KNN Neighbor Count K (Scheme 1)

| K | RMSD (Å) | Closure (Å) | Memory (GB) | Inference (s) |
|---|----------|-------------|-------------|---------------|
| **8** | TBD | TBD | TBD | TBD |
| **16 (default)** | 13.85 | 5.36 | TBD | TBD |
| **32** | TBD | TBD | TBD | TBD |

**Note:** KNN sensitivity will be characterized after primary experiments complete.

**Expected finding:** Optimal K TBD.

### Diffusion Steps T (Scheme 6)

| T (training) | T (sampling) | RMSD (Å) | Closure (Å) | Sampling Time (s) |
|---------------|--------------|----------|-------------|-------------------|
| **1000 / 20** | TBD | TBD | TBD | ~20 |
| **1000 / 50 (default)** | 13.91 | 0.02 | 45 |
| **1000 / 100** | TBD | TBD | TBD | ~90 |

**Note:** Diffusion steps sensitivity will be characterized after primary experiments complete.

**Expected finding:** Optimal T (training/sampling) TBD.

---

## 📊 Scheme 6 Decoder Bug Analysis

### Bug Description

**Original (Incorrect):**
```
decoder_input = noise_prediction(diffusion_step, latent)
```

**Fixed (Correct):**
```
decoder_input = denoised_latent(diffusion_step, latent)
```

### Performance Impact

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| **RMSD Stability** | Fluctuated 12-18Å | Stable 13.91Å | -4.09Å variance |
| **Closure Error** | 0.5±0.3Å | 0.02±0.01Å | -0.48Å |
| **Training Convergence** | Epoch 250 divergence | Stable 500 epochs | ✅ Fixed |
| **BSJ Gradient** | Spike at epoch 250 | Smooth | ✅ Fixed |

### Identification Timeline

| Checkpoint | Epoch | RMSD (Å) | Observation |
|------------|-------|----------|-------------|
| **Normal behavior** | 0-200 | 12-13 | Stable convergence |
| **Bug manifestation** | 200-250 | 12→18 | RMSD spike |
| **Diagnosis** | 250 | 18 | Gradient analysis |
| **Fix applied** | 260 | - | Architecture change |
| **Post-fix** | 260-500 | 13.91 | Stable convergence |

---

## 📊 Test Set Expansion Progress

### Current Status

| Source | Target N | Current N | Completion | Status |
|--------|----------|-----------|------------|--------|
| **PDB circularized** | 15-20 | 7 | 35-46% | ⏳ In progress |
| **Synthetic benchmark** | 10-15 | 0 | 0% | ⏳ Planned |
| **Experimental (if available)** | 5 | 0 | 0% | 🔍 Searching |

### Expanded Test Set Composition (Planned)

| Length Range | Target N | Structural Classes |
|--------------|----------|--------------------|
| **20-50 nt** | 5-7 | Hairpins, simple loops |
| **50-100 nt** | 5-7 | Multiple stems, pseudoknots |
| **100-200 nt** | 3-5 | Complex tertiary structure |

---

## 📊 TBD Placeholder Inventory

### Current TBD Count: ~50-55

| Section | TBD Count | Priority | Category |
|---------|-----------|----------|----------|
| **External baselines** | 9 | 🔴 Critical | Table values |
| **TPE ablation** | 12 | 🔴 Critical | Performance metrics |
| **Error by region** | 8 | 🟡 Important | Post-hoc analysis |
| **Hyperparameter sensitivity** | 12 | 🟢 Optional | Sensitivity curves |
| **Schemes 4, 7, 8 results** | 15 | 🟡 Important | Training progress |
| **Length scaling** | 6 | 🟢 Optional | Scalability analysis |
| **Other** | 5-8 | 🟢 Optional | Minor metrics |

### Consolidation Plan

**Move to Future Work:**
- Hyperparameter sensitivity (12 TBD)
- Length scaling analysis (6 TBD)
- Scheme 4, 7, 8 preliminary results (if not completed)

**Complete Before Submission:**
- External baselines (9 TBD) - 🔴 Critical
- TPE ablation (12 TBD) - 🔴 Critical
- Error by region (8 TBD) - 🟡 Important

---

## 📊 Round 2 Review Response Checklist

### Critical (Must Complete)

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 1 | External baseline experiments | ⏳ Pending | Table 2 filled |
| 2 | TPE ablation training | ⏳ Pending | Figure 5 revised |
| 3 | TBD consolidation | ✅ Ready | Future Work section |

### Important (Strongly Recommended)

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 4 | Error analysis by region | ⏳ Pending | Figure 6 revised |
| 5 | Decoder bug documentation | ✅ Template ready | Methods section |
| 6 | Test set expansion | ⏳ Planned | Statistical power |

### Optional (Can Defer)

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 7 | Hyperparameter sensitivity | 🟢 Deferred | Sensitivity curves |
| 8 | Scheme 4, 7, 8 training | 🟢 Deferred | Appendix |
| 9 | Length scaling | 🟢 Deferred | Scalability plots |

---

**Template End**
*Fill with actual experimental results as they become available*