# Supplementary Materials for Confluencia Paper

> **Paper**: Confluencia: Sample-Size-Adaptive Mixture-of-Experts with Pharmacokinetic Dynamics for Small-Sample circRNA Drug Discovery
> **Journal**: Computers in Biology and Medicine

---

## S1: MOE Weight Mathematical Derivation

### S1.1 Problem Formulation

Given K regression experts {f₁, f₂, ..., f_K} and training data (X, y) with N samples, we seek ensemble weights w = (w₁, w₂, ..., w_K) that minimize expected prediction error.

### S1.2 Out-of-Fold RMSE Weighting

**Definition (OOF-RMSE Weight):** For expert k, define:

$$w_k = \frac{\exp(-\lambda \cdot \text{RMSE}_k^{\text{OOF}})}{\sum_{j=1}^{K} \exp(-\lambda \cdot \text{RMSE}_j^{\text{OOF}})}$$

where:
- $\text{RMSE}_k^{\text{OOF}}$ is the out-of-fold root mean squared error for expert k
- λ > 0 is a temperature parameter controlling weight sharpness

**Derivation:**

1. **Cross-validation setup:** For F-fold CV, each expert k produces OOF predictions:
   $$\hat{y}_{k,i}^{\text{OOF}} = f_k^{(-f(i))}(x_i)$$
   where $f(i)$ is the fold containing sample i, and $f_k^{(-f(i))}$ is expert k trained on all folds except f(i).

2. **OOF error:** For expert k:
   $$\text{RMSE}_k^{\text{OOF}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_{k,i}^{\text{OOF}})^2}$$

3. **Weight rationale:** The softmax-like form arises from maximum entropy considerations:
   - Maximize entropy $H(w) = -\sum_k w_k \log w_k$ (encourage diversity)
   - Subject to expected error constraint $\sum_k w_k \cdot \text{RMSE}_k^{\text{OOF}} \leq \epsilon$
   - Lagrangian yields $w_k \propto \exp(-\lambda \cdot \text{RMSE}_k^{\text{OOF}})$

### S1.3 Sample-Size Adaptation Rule

**Theorem (Sample-Size Adaptation):** For N training samples, define threshold τ = 300. Then:

$$w_k^{\text{adapted}} = w_k \cdot \alpha_k(N)$$

where the adaptation factor $\alpha_k(N)$ is:

| Expert | α_k(N) for N < 80 | α_k(N) for 80 ≤ N < 300 | α_k(N) for N ≥ 300 |
|--------|-------------------|-------------------------|-------------------|
| Ridge | 1.2 | 1.0 | 1.0 |
| HGB | 1.0 | 1.0 | 1.0 |
| RF | 0.5 | 1.0 | 1.0 |
| MLP | 0.1 | 0.5 | 1.0 |

**Justification:**

1. **N < 80:** Neural networks (MLP) exhibit high variance due to insufficient samples for weight convergence. Linear models (Ridge) are optimal by bias-variance tradeoff.

2. **80 ≤ N < 300:** RF's bagging becomes effective (variance reduction ∝ n_estimators). MLP still overfits.

3. **N ≥ 300:** All experts viable; OOF-RMSE weighting automatically selects best.

### S1.4 Ensemble Prediction

For input x, the ensemble prediction is:

$$\hat{y}_{\text{ensemble}}(x) = \sum_{k=1}^{K} w_k^{\text{adapted}} \cdot f_k(x)$$

### S1.5 Comparison with Neural Gating

| Property | OOF-RMSE Weighting | Neural Gating |
|----------|-------------------|---------------|
| Training data required | No additional | Yes (gating network) |
| Analytical solution | Yes | No (gradient descent) |
| Stability in small samples | High | Low (meta-learning instability) |
| Input-dependent | No | Yes |
| Computational cost | O(K × F × N) | O(K × N × epochs) |

**Conclusion:** For N < 300, OOF-RMSE weighting is preferred due to stability and no additional data requirement.

---

## S2: RNACTM ODE System Full Equations

### S2.1 Six-Compartment Architecture

The RNACTM (RNA Compartmental Transmission Model) describes circRNA pharmacokinetics through six compartments:

```
C1: Inj (injection site)     — administered dose pool
C2: LNP (lipid nanoparticle) — encapsulated circRNA
C3: Endo (endosome)          — endocytosed but not escaped
C4: Cyto (cytoplasmic RNA)   — available for translation
C5: Trans (translated protein) — therapeutic protein product
C6: Clear (clearance)        — cumulative eliminated material
```

### S2.2 Parameter Definitions

| Parameter | Unit | Description | Typical Range |
|-----------|------|-------------|---------------|
| k_uptake | 1/h | Inj → LNP uptake rate | 0.15 - 0.80 |
| k_release | 1/h | LNP → Endo release rate | 0.005 - 0.80 |
| k_escape | 1/h | Endosomal escape rate | 0.01 - 0.95 |
| k_translate | 1/h | Translation initiation rate | 0.02 - 0.32 |
| k_degrade | 1/h | RNA degradation rate | 0.04 - 0.12 |
| k_protein_half | h | Product protein half-life | 14 - 24 |
| k_immune_clear | 1/h | Immune-mediated clearance | 0.01 - 0.16 |

### S2.3 ODE System

The state vector is y = [Inj, LNP, Endo, Cyto, Trans, Clear]ᵀ.

**Time-dependent protein degradation:**

$$k_{\text{protein\_degrade}}(t) = \frac{\ln 2}{k_{\text{protein\_half}}} \cdot \left(1 + \frac{k_{\text{late\_factor}}}{1 + \exp\left(-\frac{t - k_{\text{late\_delay}}}{k_{\text{late\_width}}}\right)}\right)$$

This models late-phase accelerated clearance (proteasomal upregulation after ~48h).

**Translation flux formulation:**

$$\phi_{\text{translation}} = \min\left(\frac{k_{\text{translate}}}{k_{\text{degrade}}}, 0.8\right)$$

$$k_{\text{total\_out}} = k_{\text{degrade}}$$

$$k_{\text{trans\_flux}} = \phi_{\text{translation}} \cdot k_{\text{total\_out}}$$

$$k_{\text{deg\_flux}} = (1 - \phi_{\text{translation}}) \cdot k_{\text{total\_out}}$$

**ODE equations:**

$$\frac{d\text{Inj}}{dt} = -k_{\text{uptake}} \cdot \text{Inj} + D(t)$$

$$\frac{d\text{LNP}}{dt} = k_{\text{uptake}} \cdot \text{Inj} - k_{\text{release}} \cdot \text{LNP}$$

$$\frac{d\text{Endo}}{dt} = k_{\text{release}} \cdot \text{LNP} - k_{\text{escape}} \cdot \text{Endo}$$

$$\frac{d\text{Cyto}}{dt} = k_{\text{escape}} \cdot \text{Endo} - k_{\text{total\_out}} \cdot \text{Cyto}$$

$$\frac{d\text{Trans}}{dt} = k_{\text{trans\_flux}} \cdot \text{Cyto} - k_{\text{protein\_degrade}}(t) \cdot \text{Trans}$$

$$\frac{d\text{Clear}}{dt} = k_{\text{deg\_flux}} \cdot \text{Cyto} + k_{\text{protein\_degrade}}(t) \cdot \text{Trans}$$

where D(t) is the dosing input (sum of Dirac deltas at dosing times).

### S2.4 Modification Parameter Mapping

RNA modifications alter stability and translation:

| Modification | Stability Factor | k_degrade Multiplier | Reference |
|--------------|-----------------|---------------------|-----------|
| Unmodified | 1.0 | 1.00 | Wesselhoeft 2018 |
| m6A | 1.8 | 0.56 | Chen 2019 |
| ψ (pseudouridine) | 2.5 | 0.40 | Liu 2023 |
| 5-methylcytosine | 2.0 | 0.50 | Liu 2023 |
| m6A + 2'-O-methyl | 3.0 | 0.33 | Liu 2023 |

**GC content effect:**

$$k_{\text{degrade}} = k_{\text{degrade}}^{\text{base}} \cdot (1 - 0.15 \cdot \text{GC}_{\text{content}})$$

Higher GC content slightly reduces degradation (more stable secondary structure).

### S2.5 Tissue Distribution

LNP biodistribution follows Paunovska et al. (2018):

| Delivery System | f_liver | f_spleen | f_muscle | f_other |
|-----------------|---------|----------|----------|---------|
| LNP_standard | 0.80 | 0.10 | 0.03 | 0.07 |
| LNP_liver | 0.90 | 0.05 | 0.01 | 0.04 |
| LNP_spleen | 0.35 | 0.50 | 0.02 | 0.13 |
| AAV | 0.60 | 0.15 | 0.10 | 0.15 |
| Naked RNA | 0.20 | 0.10 | 0.05 | 0.65 |

### S2.6 Numerical Solution

The ODE system is solved using `scipy.integrate.solve_ivp` with:
- Method: RK45 (adaptive Runge-Kutta)
- Relative tolerance: 10⁻⁶
- Absolute tolerance: 10⁻⁸
- Dosing impulses: Handled via segment-wise integration with dose addition at pulse times

---

## S3: Feature Engineering Details

### S3.1 Epitope Features (317 → 1335 dimensions)

**Baseline features (317-dim):**

| Category | Dimension | Description |
|----------|-----------|-------------|
| Amino acid composition | 20 | Frequency of each AA |
| Dipeptide frequency | 400 | All 20×20 pairs |
| Biochemical properties | 12 | Hydrophobicity, charge, mass, etc. |
| Mamba3Lite encoding | 128 | Three-time-constant SSM output |
| Environment features | 5 | pH, temperature, etc. |

**MHC allele features (1018-dim MHC-I + 947-dim MHC-II):**

For allele HLA-A*02:01, encoding includes:
- Pseudo-sequence (34 residues from binding groove)
- BLOSUM62 embedding (34 × 20 = 680-dim)
- Pocket residues (8 positions × 20 = 160-dim)
- Allele-specific motifs (position-specific scoring matrix)

Total with MHC: 317 + 1018 = 1335-dim (MHC-I only)

### S3.2 Drug Features (2083 → 35 dimensions)

**Full feature set (2083-dim):**

| Category | Dimension | Description |
|----------|-----------|-------------|
| Morgan fingerprints | 2048 | Radius 2, 2048 bits |
| RDKit descriptors | 210 | MW, LogP, TPSA, etc. |
| Context features | 25 | Dose, route, etc. |

**Reduced feature set (35-dim, optimal for N<200):**

| Category | Dimension | Description |
|----------|-----------|-------------|
| RDKit descriptors (selected) | 30 | MW, LogP, TPSA, HBD, HBA, rotatable bonds, etc. |
| Context features | 5 | Dose, frequency, route encoding |

**Key finding:** Morgan fingerprints overfit in small samples. Removal improves R² from 0.668 to 0.960.

---

## S4: MHC Allele Encoding Scheme

### S4.1 MHC-I Encoding (1018-dim)

For MHC class I alleles (e.g., HLA-A*02:01):

```python
def encode_mhc_i(allele: str) -> np.ndarray:
    """
    MHC-I allele encoding (1018-dim).

    Components:
    1. Pseudo-sequence (34 residues) → BLOSUM62 → 680-dim
    2. Pocket residues (8 positions) → one-hot → 160-dim
    3. Binding groove properties → 178-dim
    """
    # Parse allele name
    gene, family, protein = parse_allele(allele)  # e.g., "A", "02", "01"

    # Load pseudo-sequence from NetMHCpan database
    pseudo_seq = load_pseudo_sequence(allele)  # 34 residues

    # BLOSUM62 embedding
    blosum = np.zeros((34, 20))
    for i, aa in enumerate(pseudo_seq):
        blosum[i] = BLOSUM62[aa]

    # Pocket residues (A-F positions)
    pockets = get_pocket_residues(allele)  # 8 residues
    pocket_onehot = one_hot_encode(pockets)  # 8 × 20 = 160

    # Binding groove properties
    groove_props = compute_groove_properties(allele)  # 178-dim

    return np.concatenate([
        blosum.flatten(),      # 680
        pocket_onehot.flatten(),  # 160
        groove_props,          # 178
    ])  # Total: 1018
```

### S4.2 MHC-II Encoding (947-dim)

For MHC class II alleles (e.g., HLA-DRB1*04:01):

```python
def encode_mhc_ii(allele: str) -> np.ndarray:
    """
    MHC-II allele encoding (947-dim).

    Components:
    1. Alpha chain pseudo-sequence → 340-dim
    2. Beta chain pseudo-sequence → 340-dim
    3. Pocket residues → 200-dim
    4. Binding groove properties → 67-dim
    """
    alpha_seq, beta_seq = load_mhc_ii_sequences(allele)

    alpha_embed = blosum_embed(alpha_seq)  # 340
    beta_embed = blosum_embed(beta_seq)    # 340
    pockets = pocket_encode(allele)        # 200
    groove = groove_properties(allele)     # 67

    return np.concatenate([alpha_embed, beta_embed, pockets, groove])
```

---

## S5: MHC-II Experimental Validation Status

### S5.1 Current Status

**MHC-II binding prediction is provided as an EXPERIMENTAL feature.**

| Aspect | MHC-I | MHC-II |
|--------|-------|--------|
| Validation data | IEDB 288K peptides | Limited (< 10K) |
| Benchmark | NetMHCpan, MHCflurry | No established benchmark |
| R² validated | 0.82 (52K binary) | Not validated |
| Recommendation | Production use | Research use only |

### S5.2 Known Limitations

1. **Training data scarcity:** MHC-II binding assays are ~10× less common than MHC-I in IEDB.

2. **Peptide length variability:** MHC-II binds 13-25 residue peptides (vs 8-11 for MHC-I), requiring different encoding.

3. **Open-ended groove:** MHC-II binding groove is open at both ends, allowing peptide overhang.

4. **Alpha chain polymorphism:** HLA-DR, -DQ, -DP have both alpha and beta chain variation.

### S5.3 Planned Validation

| Milestone | Target | Status |
|-----------|--------|--------|
| IEDB MHC-II benchmark | 50K peptides | Pending |
| DP/DQ encoding | Complete | In progress |
| NetMHCIIpan comparison | AUC > 0.75 | Pending |
| Production release | v2.1 | Planned |

### S5.4 Usage Guidance

```python
# MHC-I: Validated, production-ready
result_i = predict_epitope("SLYNTVATL", "HLA-A*02:01")  # ✓ Recommended

# MHC-II: Experimental, use with caution
result_ii = predict_epitope("SLYNTVATLCYTL", "HLA-DRB1*04:01")  # ⚠ Experimental
```

**Citation guidance:** When reporting MHC-II predictions, include disclaimer:
> "MHC-II binding predictions were generated using Confluencia's experimental MHC-II module, which has not been independently validated against established benchmarks."

---

## References for Supplementary Materials

1. Wesselhoeft RA, et al. Engineering circular RNA for potent and stable translation. Nat Commun. 2018;9:2629.

2. Chen YG, et al. m6A-dependent regulation of circRNA export and degradation. Nature. 2019;586:651-655.

3. Liu J, et al. Modified circRNA therapeutics for cancer immunotherapy. Nat Commun. 2023;14:2548.

4. Hassett KJ, et al. Optimization of lipid nanoparticles for intramuscular delivery of mRNA vaccines. Mol Ther. 2019;27:1885-1897.

5. Gilleron J, et al. Quantitative 3D analysis of LNP intracellular processing. Nat Biotechnol. 2013;31:638-646.

6. Paunovska IU, et al. Quantitative analysis of LNP biodistribution. ACS Nano. 2018;12:8307-8320.

7. Jacobs RA, et al. Adaptive mixtures of local experts. Neural Comput. 1991;3:79-87.

8. Shazeer N, et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ICLR. 2017.
