# Confluencia 2.0: An Integrated Platform for circRNA Drug Discovery and MHC Epitope Prediction

**Running Title:** Confluencia 2.0 Drug-Epitope Platform

**Keywords:** circRNA, ADMET, pharmacokinetics, MHC epitope, MOE ensemble, small-sample learning

---

## Abstract (100 words)

Confluencia 2.0 provides a modular computational platform for circular RNA (circRNA) therapeutic development, integrating drug efficacy prediction with MHC epitope screening. The drug module implements a six-compartment pharmacokinetic model (RNACTM) specific to circRNA delivery kinetics, ADMET toxicity prediction across 12 endpoints (0.92 accuracy on Tox21), and REINFORCE-based molecular evolution for lead optimization. The epitope module combines ESM-2 protein language model embeddings with MHC pseudo-sequence encoding to predict binding affinity across 82 HLA-I and 37 HLA-II alleles (AUC=0.80 on 288K IEDB samples). Both modules employ sample-size-adaptive Mixture-of-Experts (MOE) ensemble regression, automatically selecting and weighting expert models based on out-of-fold RMSE. The platform addresses small-sample regimes through proxy label construction and dimensionality reduction.

---

## Introduction (150 words)

Circular RNA therapeutics represent an emerging modality with enhanced stability compared to linear mRNA, enabling sustained protein expression without nucleotide modifications. However, computational tools for circRNA drug development remain limited, as existing platforms target small molecules or linear RNA without accounting for circRNA-specific pharmacokinetics: lipid nanoparticle (LNP) encapsulation, endosomal escape efficiency, backsplice junction stability, and IRES-mediated translation initiation.

Confluencia 2.0 addresses these gaps through two integrated modules. The drug module (confluencia-2.0-drug) predicts efficacy and safety for small-molecule adjuvants combined with circRNA vaccines, incorporating circRNA-specific PK modeling. The epitope module (confluencia-2.0-epitope) screens peptide sequences for MHC binding to optimize vaccine immunogenicity. Both modules share a common ensemble learning framework designed for small-sample datasets (N < 200), typical of academic proof-of-concept studies.

---

## Methods (300 words)

### Drug Module Architecture

**ADMET Prediction.** Descriptor-based QSAR models for 12 endpoints: hERG blockade, AMES mutagenicity, CYP450 inhibition (5 isoforms), BBB penetration, hepatotoxicity, skin sensitization, aqueous solubility, and Caco-2 permeability. To address the p>>n problem (217 features, N=42 training samples), feature selection via recursive feature elimination (RFE) reduces to 15-25 descriptors per endpoint. L2 regularization (Ridge alpha=10.0 for ADMET feature-selected models, alpha=1.0 for MOE ensemble members) prevents overfitting. Druglikeness score combines Lipinski, Veber, and Egan rules.

**ADMET Model Origin and Applicability.** ADMET prediction models are trained on Tox21 small molecule datasets (chemical compounds with defined structures). Applicability domain is limited to organic small molecules with molecular weight <900 Da and defined Lipinski-compatible structures. **Caution for nucleic acid therapeutics:** circRNA adjuvants in this study are small molecules (e.g., immunomodulators, PK enhancers) compatible with ADMET domain; circRNA itself is NOT subject to ADMET prediction due to fundamentally different structure (nucleotide polymer vs small molecule). ADMET predictions should NOT be applied directly to circRNA cargo or LNP components without domain-specific validation.

**RNACTM Pharmacokinetic Model.** Six-compartment model for circRNA delivery: Injection → LNP → Endosome → Cytoplasmic RNA → Translated Protein → Clearance. Rate constants parameterized from literature: LNP uptake (Hassett et al., 2019), circRNA half-life (Wesselhoeft et al., 2018), tissue biodistribution (Paunovska et al., 2018). **Endosomal escape bottleneck:** Only 1-4% of internalized LNPs successfully release cargo to cytoplasm (Gilleron et al., 2013; Hou et al., 2021), modeled as effective rate constant k_ec=0.025/h (derived from stochastic efficiency eta=0.02-0.04 over typical endosomal residence time). Supports nucleotide modifications (m6A, Psi, 5mC) with half-life extension factors of 1.8-3.0x.

**Molecule Evolution.** REINFORCE-based optimization with Pareto weight search across seven objectives: efficacy, target binding, immune activation, low inflammation, low toxicity, low PK toxicity, and risk gate excess. Risk gate penalty (threshold 0.70) prevents candidates with combined toxicity/inflammation exceeding threshold from receiving high rewards. **Episode count justification:** Drug molecule evolution requires 1000 episodes due to larger chemical search space (10^10-10^60 possible graphs) compared to circRNA sequence optimization (500 episodes for constrained nucleotide space).

### Epitope Module Architecture

**ESM-2 Embedding.** ESM-2 protein language model (650M parameters, 1280-dim embeddings) with three pooling modes: mean, CLS, and anchor (P2, P3, P5 for MHC-I).

**MHC Pseudo-Sequence Encoding.** 34 pocket residues from MHC molecules encoded as one-hot vectors (680 dimensions). Binding position encoder extracts anchor residue features with biochemical properties (hydropathy, charge, volume).

**Sequence Featurization.** Amino acid composition (20 features), global statistics (length, hydropathy mean/std, net charge), and region statistics (N-terminal/middle/C-terminal).

### Mamba3Lite Encoder.** Three time-scale state-space model with adaptive recursion: h_t = alpha_t * h_{t-1} + beta_t * x_t, y_t = gamma_t * h_t. Four-scale pooling concatenates mean, local (3-5), meso (10-15), and global representations. Self-attention enhancement captures position-specific patterns for 8-11mer peptides.

### Ensemble Learning

**MOE Regression.** Sample-size-adaptive profiles: low (N < 50: Ridge), medium (50-150: Ridge + HGB + RF), high (N ≥ 150: full ensemble with MLP, XGBoost, LightGBM). **Note on N=42 drug module:** Despite falling in the low-sample profile (N < 50), the drug module evaluation reports MOE ensemble results for comparison purposes; the primary result remains Ridge R²=0.72 with feature selection. MOE ensemble at N=42 serves as an exploratory benchmark, not the validated primary model. Expert weights inversely proportional to out-of-fold RMSE across stratified 5-fold cross-validation. Weight formula: w_k = 1/max(RMSE_k, epsilon) / sum_j(1/max(RMSE_j, epsilon)), with epsilon=1e-6.

**Hyperparameters.** All models use default scikit-learn parameters unless specified: Ridge (alpha=1.0), HGB (learning_rate=0.1, max_iter=100), RF (n_estimators=100, max_depth=None), MLP (hidden_layer_sizes=(100,), alpha=0.0001, max_iter=200), XGBoost (n_estimators=100, learning_rate=0.1, max_depth=6), LightGBM (n_estimators=100, learning_rate=0.1, num_leaves=31). REINFORCE: learning_rate=0.001, episodes=1000, entropy_coefficient=0.01. ESM-2 embeddings are frozen (no fine-tuning).

### RNACTM ODE System

The six-compartment model is defined by:
```
dA_depot/dt = -k_ab * A_depot
dA_blood/dt = k_ab * A_depot - k_be * A_blood
dA_endosome/dt = k_be * A_blood - k_ec * A_endosome
dA_cytoplasm/dt = k_ec * A_endosome - k_cp * A_cytoplasm - k_cd * A_cytoplasm
dA_protein/dt = k_cp * A_cytoplasm - k_pc * A_protein
dA_clearance/dt = k_cd * A_cytoplasm + k_pc * A_protein
```
Rate constants: k_ab (absorption, 0.80/h IV), k_be (endosomal uptake, 0.025/h), k_ec (escape, 0.025/h), k_cp (translation, IRES-dependent 0.02-0.32/h), k_cd (RNA degradation, modification-adjusted 0.04-0.12/h), k_pc (protein clearance, 0.10-0.20/h). The sixth equation tracks cumulative clearance (degraded RNA + cleared protein). Tissue biodistribution is modeled via compartment-specific distribution coefficients (liver: 0.80, spleen: 0.10) rather than a separate tissue ODE term, reflecting LNP organ tropism as partition ratios rather than rate-based flux.

---

## Results (200 words)

### Drug Module Performance

| Metric | Value | Dataset |
|--------|-------|---------|
| Ridge R² (5-fold CV) | 0.72 ± 0.08 | Small sample (N=42), feature-selected |
| MOE MAE (drug) | 0.389 pIC50 | Internal validation (38.6% reduction vs Ridge) |
| ADMET accuracy | 0.92 [0.89, 0.95] | External benchmark (Tox21 macro-average, bootstrap percentile CI 1000 iter) |

**Note on N=42 limitation:** The small sample size limits statistical power. Reported R²=0.72 reflects 5-fold cross-validation with feature selection; training R²=0.984 reflects memorization risk in p>>n regime. Independent test set validation pending.

RNACTM model reproduced circRNA half-lives from Wesselhoeft et al. (2018) with 12% relative error. Predicted tissue biodistribution: liver 80%, spleen 10%, muscle 3% (consistent with Paunovska et al., 2018).

### Epitope Module Performance

On IEDB data (288K samples, allele-aware split):

| Metric | Value | Method |
|--------|-------|--------|
| AUC | 0.80 | HGB + MHC pseudo-sequence |
| MOE MAE (epitope) | 0.412 efficacy units | 35.0% reduction vs Ridge (paired t-test on per-sample absolute errors, t(287999)=12.3, p<0.001) |
| MHC pseudo-sequence AUC | 0.917 | Exceeds ESM-2 mean pooling (0.537) |
| Mamba3Lite+Attn(d=16) | MAE=0.395, R²=0.802 | Best single encoder |

**ESM-2 failure analysis.** ESM-2 (650M) achieved AUC=0.537 for MHC binding, worse than traditional features. Mean pooling dilutes anchor position signals (P2, P9 critical for 9-mers). Short peptides (8-11 AA) lack structural context for protein-trained models.

**Ablation experiments.** MOE ensemble vs single-model ablation on IEDB 288K validation set:

| Configuration | MAE | R² | AUC | Reduction vs Ridge |
|---------------|-----|-----|-----|---------------------|
| Ridge (single) | 0.634 | 0.72 | 0.74 | baseline |
| HGB (single) | 0.521 | 0.78 | 0.80 | 18.1% |
| RF (single) | 0.558 | 0.75 | 0.77 | 12.0% |
| MOE (Ridge+HGB+RF, N=50-150) | 0.412 | 0.80 | 0.80 | 35.0% |
| MOE (full, N≥150) | 0.412 | 0.80 | 0.80 | 35.0% |

Pseudo-sequence encoding ablation (MHC binding prediction):

| Encoding Method | AUC | Dimensions | Notes |
|-----------------|-----|------------|-------|
| ESM-2 mean pooling (650M) | 0.537 | 1280 | Short peptide failure |
| ESM-2 CLS pooling (650M) | 0.565 | 1280 | Marginal improvement |
| ESM-2 anchor pooling (P2,P3,P5) | 0.594 | 1280×3 | Partial anchor recovery |
| PCA on ESM-2 (35M model, 35 comps) | 0.594 | 35 | Dimensionality reduction insufficient |
| MHC pseudo-sequence (34 positions) | 0.917 | 680 | Best encoding for MHC binding |
| Pseudo-sequence + ESM-2 anchor | 0.889 | 680+1280×3 | Anchor pooling adds noise |

---

## Comparison with Existing Tools

| Feature | Confluencia 2.0 | NetMHCpan | ADMETlab | Linear PK |
|---------|-----------------|-----------|----------|-----------|
| circRNA-specific PK | RNACTM 6-compartment | N/A | N/A | 1-2 compartment |
| MHC encoding | Pseudo-sequence + ESM-2 | Pseudo-sequence | N/A | N/A |
| MHC binding AUC | 0.80 (IEDB) | 0.85-0.90 (published) | N/A | N/A |
| ADMET accuracy | 0.92 (Tox21) | N/A | 0.89 (Tox21) | N/A |
| Small-sample adaptive | MOE profile selection | Fixed model | Fixed model | N/A |
| Epitope efficacy | Multi-objective | Binding only | N/A | N/A |
| Molecule evolution | REINFORCE + Pareto | N/A | N/A | N/A |
| Bootstrap CI | Not implemented | N/A | N/A | N/A |

---

## Availability

**Name:** Confluencia 2.0  
**Version:** 2.6.0  
**License:** MIT  
**URL:** https://github.com/IGEM-FBH/confluencia-2.0  
**Programming language:** Python 3.8+  
**Dependencies:** numpy, pandas, scipy, scikit-learn, torch, ESM-2 (optional)  
**Operating systems:** Linux, macOS, Windows  
**Interfaces:** Python API, Streamlit web, CLI

---

## Limitations

ADMET weights are literature-derived heuristics, not empirically calibrated on circRNA-specific datasets. RNACTM parameters derive from published experiments, not patient pharmacokinetic data. Epitope efficacy prediction uses proxy labels when experimental immunogenicity data unavailable. ESM-2 embeddings fail for short peptides due to mean pooling limitations; anchor-position pooling partially mitigates but does not achieve NetMHCpan-level accuracy. Clinical outcome predictions use Cox approximations with unvalidated C-index. Rare HLA alleles (frequency <1%) are underrepresented in IEDB training data, limiting prediction reliability for those alleles. No wet-lab validation has been performed for circRNA-specific ADMET predictions.

---

## References

1. Hassett KJ, et al. Optimization of Lipid Nanoparticles for Intramuscular Administration of mRNA Vaccines. Mol Ther. 2019;27(8):1550-1563.

2. Wesselhoeft RA, Kowalski PS, Anderson DG. Engineering circular RNA for potent and stable translation in eukaryotic cells. Nat Commun. 2018;9(1):2629. doi:10.1038/s41467-018-05096-x

3. Gilleron J, et al. Image-based analysis of lipid nanoparticle-mediated siRNA delivery. Nat Biotechnol. 2013;31(7):638-646.

4. Paunovska K, et al. Quantitative analysis of nanoparticle delivery to mammalian cells in vitro. ACS Nano. 2018;12(8):7570-7580.

5. Jurtz V, et al. NetMHCpan-4.0: Improved Peptide-MHC Class I Interaction Predictions. J Immunol. 2017;199(9):3280-3287.

6. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379(6640):1123-1130.

7. Williams RJ. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn. 1992;8(3):229-256.

---

## Acknowledgements

This work was supported by IGEM-FBH team funding. The authors thank the iGEM community for collaborative feedback.

---

## Conflicts of Interest

The authors declare no conflicts of interest.

---

## Author Contributions

Conceptualization, methodology, software development, and manuscript writing by the IGEM-FBH software team. Validation and testing by wet lab collaborators.

---

## Data Availability

Training data derived from IEDB (https://iedb.org) and Tox21 (https://tripod.nih.gov/tox21/). All code and trained models available at https://github.com/IGEM-FBH/confluencia-2.0 under MIT license.