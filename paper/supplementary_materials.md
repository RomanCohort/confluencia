# Supplementary Materials for Confluencia: An Integrated Computational Platform for circRNA Drug Discovery

## Table of Contents

1. [Hyperparameter Configurations](#s1-hyperparameter-configurations)
2. [Complete Ablation Results](#s2-complete-ablation-results)
3. [Model Architecture Details](#s3-model-architecture-details)
4. [Additional Validation Results](#s4-additional-validation-results)
   - S4.1 Cross-Validation Stability
   - S4.2 Confidence Interval Estimation
   - S4.3 Stratified Cross-Validation
   - S4.4 Per-Target Drug Prediction
5. [Feature Engineering Details](#s5-feature-engineering-details)
6. [Dataset Statistics](#s6-dataset-statistics)

---

## S1. Hyperparameter Configurations

### S1.1 Epitope Prediction Models

| Model | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| **Ridge** | alpha | 1.0 | L2 regularization |
| | fit_intercept | True | |
| **HGB** | max_depth | 6 | Tree depth limit |
| | learning_rate | 0.05 | Shrinkage factor |
| | max_iter | 100 | Boosting iterations |
| | random_state | 42 | Reproducibility |
| **RF** | n_estimators | 300 | Number of trees |
| | max_depth | 12 | Tree depth limit |
| | min_samples_split | 5 | Minimum samples for split |
| | random_state | 42 | Reproducibility |
| **MLP** | hidden_layers | (128, 64) | Two hidden layers |
| | activation | relu | Activation function |
| | max_iter | 500 | Maximum iterations |
| | early_stopping | True | Prevent overfitting |
| | random_state | 42 | Reproducibility |

### S1.2 MOE Ensemble Parameters

| Parameter | Low Profile (N<80) | Medium (80≤N<300) | High (N≥300) |
|-----------|-------------------|-------------------|--------------|
| cv_folds | 3 | 5 | 5 |
| experts | Ridge, HGB | Ridge, HGB, RF | Ridge, HGB, RF, MLP |
| weighting | OOF-RMSE inverse | OOF-RMSE inverse | OOF-RMSE inverse |

### S1.3 Drug Prediction Models

| Model | Parameter | Value |
|-------|-----------|-------|
| **Ridge** | alpha | 1.0 |
| **HGB** | max_depth | 6 |
| | learning_rate | 0.05 |
| **RF** | n_estimators | 300 |
| | max_depth | 12 |

### S1.4 Optional Hyperparameter Tuning Search Spaces

When hyperparameter tuning is enabled, the following parameter grids are searched
using RandomizedSearchCV (default: 20 iterations, 3-fold CV) or GridSearchCV:

| Model | Parameter | Search Space |
|-------|-----------|--------------|
| **Ridge** | alpha | [0.01, 0.1, 1.0, 10.0, 100.0] |
| **HGB** | max_depth | [4, 6, 8, 10] |
| | learning_rate | [0.05, 0.1, 0.15, 0.2] |
| | l2_regularization | [0.0, 0.1, 0.3, 0.5] |
| **RF** | n_estimators | [100, 200, 300] |
| | max_depth | [6, 10, 14] |
| | min_samples_split | [2, 5, 10] |
| **MLP** | hidden_layer_sizes | [(64,), (128,), (64, 32)] |
| | alpha | [0.0001, 0.001, 0.01] |

For MOE ensembles, each expert is tuned independently, and the optimized
parameters are passed to `ExpertConfig` for ensemble construction.

---

## S2. Complete Ablation Results

### S2.1 Epitope Ablation (N=300, HGB backbone)

| Configuration | Feature Dim | MAE | R² | Δ MAE |
|--------------|-------------|-----|-----|-------|
| Full (all components) | 317 | 0.308 | 0.853 | — |
| - Mamba summary | 221 | 0.343 | 0.819 | +11.4% |
| - Mamba local pool | 293 | 0.311 | 0.843 | +1.0% |
| - Mamba meso pool | 293 | 0.308 | 0.853 | +0.0% |
| - Mamba global pool | 293 | 0.308 | 0.853 | +0.0% |
| - k-mer (2-mer) | 253 | 0.305 | 0.858 | -1.0% |
| - k-mer (3-mer) | 253 | 0.307 | 0.855 | -0.3% |
| - Biochem stats | 301 | 0.511 | 0.542 | +65.7% |
| - Environment | 312 | 0.557 | 0.515 | +81.0% |
| Only Mamba+env | 173 | 0.546 | 0.520 | +77.3% |
| Only kmer+bio+env | 149 | 0.333 | 0.811 | +8.1% |
| Only env (baseline) | 5 | 0.799 | -0.016 | +159.5% |

### S2.2 Drug Ablation (N=200)

| Configuration | Feature Dim | MAE | R² | Δ R² |
|--------------|-------------|-----|-----|------|
| Full (Morgan + descriptors) | 2083 | 0.082 | 0.67 | — |
| - Morgan FP | 35 | 0.076 | 0.96 | +0.29 |
| - Descriptors | 2048 | 0.089 | 0.61 | -0.06 |
| Only context | 3 | 0.156 | 0.12 | -0.55 |

---

## S3. Model Architecture Details

### S3.1 Mamba3Lite Encoder

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Config** | d_model | 24 (default), 64 (large) |
| | n_layer | 3 |
| | dt_rank | 4 |
| | d_state | 8 |
| **Pooling scales** | residue | 1 (no pooling) |
| | local | 3 (±1 neighbors) |
| | meso | 11 (±5 neighbors) |
| | global | full sequence |
| **Output** | summary | 4×d_model (mean, max, last, mix) |
| | neighborhood | 3×d_model (local, meso, global) |

### S3.2 RNACTM Six-Compartment Model

| Compartment | Symbol | Rate Constant | Description |
|-------------|--------|---------------|-------------|
| Injection | Inj | k_release | Dose depot release |
| LNP | LNP | k_release | LNP encapsulation |
| Endosome | Endo | k_escape | Endosomal escape |
| Cytoplasm | Cyto | k_degrade, k_translate | RNA degradation, translation |
| Translation | Trans | k_protein_degrade | Protein production |
| Clearance | Clear | — | Cumulative elimination |

**Default rate constants (unmodified circRNA, IV delivery):**
- k_release: 0.12 /h
- k_escape: 0.02 /h
- k_degrade: 0.12 /h (t½ ≈ 6h)
- k_translate: 0.02 /h
- k_protein_degrade: ln(2)/24 ≈ 0.029 /h

### S3.3 Feature Engineering

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| Mamba3Lite summary | 96 | 4 pooling types × d_model |
| Mamba3Lite neighborhood | 72 | 3 scales × d_model |
| k-mer hashing (2-mer) | 64 | Hashed 2-mer counts |
| k-mer hashing (3-mer) | 64 | Hashed 3-mer counts |
| Biochemical stats | 16 | Physicochemical properties |
| Environment | 5 | dose, freq, treatment_time, circ_expr, ifn_score |

---

## S4. Additional Validation Results

### S4.1 Cross-Validation Stability (5-fold, N=300)

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|--------|--------|--------|--------|--------|--------|------------|
| MOE MAE | 0.382 | 0.395 | 0.378 | 0.401 | 0.389 | 0.389 ± 0.009 |
| MOE R² | 0.825 | 0.812 | 0.831 | 0.808 | 0.819 | 0.819 ± 0.009 |

### S4.2 Confidence Interval Estimation

We use an adaptive strategy for 95% CI estimation that accounts for small
cross-validation sample sizes:

| Method | Condition | Formula |
|--------|-----------|---------|
| t-distribution | n < 10 | $CI = \bar{x} \pm t_{0.025, n-1} \cdot s / \sqrt{n}$ |
| Bootstrap percentile | n ≥ 10 | $CI = [P_{2.5}(\hat{\mu}^*), P_{97.5}(\hat{\mu}^*)]$ |

Where $\hat{\mu}^*$ are bootstrap resampled means (B=1,000 iterations, seed=42).

**Rationale:** For typical 5-fold CV (n=5), the t-distribution critical value
$t_{0.025,4}=2.776$ produces a 42% wider CI than the z-based approach (z=1.96),
providing more honest uncertainty estimates. The bootstrap percentile method is
used for n ≥ 10, where the central limit theorem provides adequate coverage.

| Metric | Mean | 95% CI (t-distribution) |
|--------|------|-------------------------|
| MOE MAE | 0.389 | [0.380, 0.398] |
| MOE R² | 0.819 | [0.803, 0.835] |
| vs Ridge improvement | 39.2% | [32.5%, 45.9%] |

### S4.3 Stratified Cross-Validation for Regression

To ensure balanced target distributions across folds, we implement quantile-based
stratified CV:

1. Bin continuous target y into q quantile groups (q = min(2×n_splits, 10))
2. Apply `StratifiedKFold` using bin labels for stratification
3. Fall back to standard `KFold(shuffle=True)` when N < 3q

This prevents pathological splits where one fold contains predominantly high-efficacy
samples while another contains only low-efficacy samples.

### S4.4 Per-Target Drug Prediction (91k)

| Target | Ridge R² | HGB R² | RF R² | MOE R² | Best Model |
|--------|----------|--------|-------|--------|------------|
| efficacy | 0.586 | 0.586 | 0.563 | 0.603 | MOE |
| target_binding | 0.965 | 0.948 | 0.921 | 0.951 | Ridge |
| immune_activation | 0.712 | 0.737 | 0.698 | 0.720 | HGB |
| immune_cell_activation | 0.689 | 0.725 | 0.702 | 0.698 | HGB |
| inflammation_risk | 0.621 | 0.685 | 0.698 | 0.633 | RF |
| toxicity_risk | 0.598 | 0.652 | 0.670 | 0.624 | RF |

---

## S5. Feature Engineering Details

### S5.1 Biochemical Statistics (16 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | bio_length | Peptide length |
| 1 | bio_hydrophobic_frac | Fraction of hydrophobic residues (A, V, L, I, M, F, W, P) |
| 2 | bio_polar_frac | Fraction of polar residues (S, T, N, Q, Y, C) |
| 3 | bio_acidic_frac | Fraction of acidic residues (D, E) |
| 4 | bio_basic_frac | Fraction of basic residues (K, R, H) |
| 5 | bio_entropy | Sequence entropy (-Σ p_i log p_i) |
| 6 | bio_n_hydrophobic | Count of N-terminal hydrophobic (first 3) |
| 7 | bio_c_hydrophobic | Count of C-terminal hydrophobic (last 3) |
| 8 | bio_proline_frac | Proline fraction |
| 9 | bio_glycine_frac | Glycine fraction |
| 10 | bio_aromatic_frac | Aromatic fraction (F, W, Y) |
| 11 | bio_basic2_frac | Basic fraction excluding H (K, R) |
| 12 | bio_acidic2_frac | Acidic fraction (D, E) |
| 13 | bio_amide_frac | Amide fraction (N, Q) |
| 14 | bio_unique_residue_ratio | Unique residues / total length |
| 15 | bio_unknown_ratio | Unknown/X residues ratio |

### S5.2 k-mer Hashing Algorithm

```python
def _hash_kmer(seq: str, k: int = 2, dim: int = 64) -> np.ndarray:
    """
    Hash k-mers to fixed-dimensional vector.
    
    Uses stable 64-bit hash of each k-mer, modulo dim for binning.
    Output is L2-normalized.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        h = stable_hash_u64(kmer) % dim
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
```

---

## S6. Dataset Statistics

### S6.1 Epitope Training Data

| Dataset | N | Unique Sequences | Source |
|---------|---|------------------|--------|
| example_epitope.csv | 300 | 300 | Curated training set |
| epitope_training_full.csv | 10,461 | 10,461 | IEDB MHC-I |
| IEDB 288k | 288,135 | 288,135 | Full IEDB MHC-I |

### S6.2 Drug Training Data

| Dataset | N | Unique SMILES | Unique Epitopes |
|---------|---|---------------|-----------------|
| example_drug.csv | 200 | 200 | 1 |
| drug_91k.csv | 91,150 | 905 | 95 |

### S6.3 External Validation Data

| Dataset | N | Purpose | Source |
|---------|---|---------|--------|
| IEDB held-out | 1,955 / 2,166 | External epitope validation | IEDB |
| NetMHCpan benchmark | 61 | Binding concordance | Jurtz et al. 2017 |
| ChEMBL | 300 | Drug bioactivity | ChEMBL |
| TCCIA | 75 | circRNA immunotherapy | TCCIA Atlas |
| GDSC | 50 | Drug sensitivity | GDSC |
| Literature cases | 17 | IFN response | Published studies |

---

## Data Availability

All benchmark datasets, trained models, and analysis scripts are available at:
- **Code:** https://github.com/IGEM-FBH/confluencia
- **Docker:** ghcr.io/igem-fbh/confluencia:latest

## License

MIT License
