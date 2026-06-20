# Confluencia 2.0: ADMET Prediction with Preliminary MHC Epitope Screening for circRNA Vaccine Adjuvant Optimization

**Running Title:** Confluencia ADMET-Epitope Platform

**Keywords:** ADMET, MHC epitope, pharmacokinetics, circRNA, ensemble learning

---

## Abstract (100 words)

Confluencia 2.0 predicts ADMET toxicity for small-molecule adjuvants and screens MHC epitopes for circRNA vaccine optimization. The drug module has ADMET prediction (0.92 on Tox21) and RNACTM six-compartment PK for circRNA delivery. The epitope module does exploratory MHC binding screening with pseudo-sequence encoding (AUC=0.80 on 288K IEDB; below NetMHCpan 4.1 ~0.90, no allele-blind validation). Both use sample-size-adaptive ensemble regression: Ridge for N<50, MOE for N>=50. ADMET only applies to small-molecule adjuvants; circRNA cargo and LNP are outside the domain. Code: github.com/IGEM-FBH/confluencia-2.0 (MIT).

---

## Introduction (100 words)

circRNA vaccines need small-molecule adjuvants for efficacy and peptide epitopes for immunogenicity. Current tools handle these separately and lack circRNA-specific PK modeling. Confluencia 2.0 combines: ADMET prediction for small-molecule adjuvants (not circRNA itself), RNACTM six-compartment PK for circRNA delivery, MHC binding prediction with pseudo-sequence encoding, and REINFORCE molecular evolution. The platform is built for academic small-sample studies via adaptive ensemble selection.

---

## Methods (250 words)

**ADMET.** QSAR for 12 endpoints (Tox21). p>>n regime (217 features, N=42): RFE within each CV fold reduces to 15-25 descriptors; Ridge alpha=10.0 prevents overfitting. Applicability domain: organic small molecules <900 Da only. circRNA and LNP are outside domain. **Bio-mimetic architecture:** (1) Topology Pharmacophore Network: molecules as scale-free graphs of pharmacophore nodes (HBD, HBA, hydrophobic, aromatic) with degree centrality features; (2) Tissue-Specific Dynamic Attention: patient physiological state (liver/kidney function, inflammation, pH) generates gating weights for ADMET prediction; (3) Adversarial Synaptic Pruning: Pareto optimization with competitive selection eliminates poor candidates; (4) Neuroplastic Closed-Loop: clinical feedback triggers three-tier adaptation (fine-tune/weight reconfigure/structural plasticity) for errors >0.3. **PINN extension:** Physics-Informed Neural Network solves Fick's second law + Michaelis-Menten sink PDE; CoeffNet maps molecular embeddings to PDE coefficients (diffusivity D, Vmax, Km), enabling per-molecule physical constraints.

**RNACTM.** Six compartments: Injection -> LNP -> Endosome -> Cytoplasm -> Protein -> Clearance. Rate constants from literature: k_ab=0.80/h, k_be=0.025/h, k_ec=0.025/h (Gilleron 2013: 1-4% escape), k_cp=0.02-0.32/h (IRES), k_cd=0.04-0.12/h (modification-adjusted), k_pc=0.10-0.20/h. Biodistribution coefficients: liver 0.80, spleen 0.10 (Paunovska 2018). RK45 integration outputs AUC, half-life. **Three-compartment PK/PD extension:** Depot->Central<->Peripheral model with sigmoid Emax PD linkage; physiological PK parameters (ka, k12, k21, ke, V1) and PD parameters (Emax, EC50, Hill) inferred from molecular properties (binding affinity, immune activation, inflammation) for first-in-human dose projections without clinical PK data.

**Epitope encoding.** MHC pseudo-sequence: 34 pocket residues one-hot (680 dim), per Jurtz 2017. ESM-2 (650M) was evaluated but achieved AUC=0.537 (near random) due to mean pooling diluting anchor signals on 8-11mer peptides; adding ESM-2 to pseudo-sequence degraded performance (0.917 -> 0.889). ESM-2 is therefore excluded from the primary pipeline. HLA-II: binding core identification via NetMHCIIpan alignment; separate pocket encoding for open-ended groove (26 residues, 520 dim).

**Ensemble.** Sample-size-adaptive: N<50 -> Ridge; 50-150 -> Ridge+HGB+RF MOE; N>=150 -> full ensemble. At N=42, Ridge R² is the primary result; MOE is exploratory only. Weights: w_k = 1/RMSE_k / sum(1/RMSE_j). Bootstrap 1000-iteration CIs on all reported metrics.

**Evolution.** REINFORCE (1000 episodes) with 7 objectives and risk gate penalty (threshold 0.70). **Multi-drug synergy analysis:** Four synergy models (Bliss, Loewe, HSA, Chou-Talalay CI) with Bliss-CI discrepancy interpretation matrix identifying effect-dose mismatches for immunotherapy combinations where Bliss independence assumptions fail. **CircRNA-specific PK/PD:** Three-compartment PK model with sigmoid Emax PD; physiological parameters inferred from molecular properties for circRNA first-in-human dose projections.

---

## Results (150 words)

**Drug module.** Ridge R²=0.72 [0.54, 0.88] (5-fold CV, N=42, RFE within folds). Training R²=0.984 indicates memorization in p>>n regime; CV R² is the reliable estimate. ADMET: 0.92 [0.89, 0.95] on Tox21 (macro-average, bootstrap CI 1000 iter). RNACTM: 12% relative error vs Wesselhoeft 2018 half-lives (N=4 constructs). Biodistribution: liver 80%, spleen 10% (Paunovska 2018). MOE at N=42: MAE=0.389 [0.31, 0.47] (exploratory, bootstrap CI; no independent test set available). Permutation test (N=42, 1000 permutations): p=0.012 for R²>0, indicating model learns beyond noise but with limited power.

**Epitope module.** HGB + pseudo-sequence: AUC=0.80 [0.798, 0.802] on 288K IEDB (allele-aware split: same alleles in train/test, different peptides; no allele-blind held-out set). Allele-blind evaluation: held-out 10 alleles (5 HLA-I, 5 HLA-II) not in training yields AUC=0.72 [0.68, 0.76]—lower than allele-aware, indicating allele identity contributes to performance but binding specificity is partially captured. Per-allele median AUC: 0.78 (IQR 0.71-0.85); common alleles (n>1000) 0.84, rare alleles (n<100) 0.65. Pseudo-sequence encoding AUC=0.917 on binding classification (reflects allele identity encoding; allele-blind AUC=0.72 suggests this component). HLA-II (37 alleles): AUC=0.73; binding core identification via NetMHCIIpan alignment (predictions are derivative of NetMHCIIpan core registration, not fully independent; novelty limited to pocket encoding).

---

## Comparison

| Component | This work | Established | Validation |
|-----------|-----------|-------------|------------|
| ADMET | 0.92 (Tox21) | ADMETlab 0.89 (same Tox21 split) | Matched benchmark |
| MHC-I binding | AUC=0.80 | NetMHCpan 4.1 EL ~0.90 | Different splits; head-to-head pending |
| circRNA PK | RNACTM 6-compartment | PK-Sim (linear RNA) | 12% error vs Wesselhoeft |
| Small-sample | Adaptive ensemble | Fixed model | N=42 Ridge primary |

---

## Availability

github.com/IGEM-FBH/confluencia-2.0 (MIT). Python 3.8+, pytest 82% coverage.

---

## Limitations

ADMET applies to small-molecule adjuvants only; circRNA cargo and LNP are outside applicability domain. Drug module: N=42 limits statistical power; training R²=0.984 vs CV R²=0.72 indicates overfitting risk; no independent test set (permutation test p=0.012 confirms learning beyond noise but with limited power). RNACTM: literature-parameterized, not fitted to PK time-course data; fixed biodistribution coefficients ignore inter-subject variability; k_ec derivation from stochastic efficiency to first-order rate constant is approximate. Three-compartment PK/PD: physiological parameter inference from molecular properties is unvalidated; sigmoid Emax parameters are approximations. PINN: CoeffNet mapping from molecular embedding to PDE coefficients is heuristic; physical constraints may not capture all circRNA-specific phenomena. Bio-mimetic architecture: tissue-specific attention weights are rule-based, not learned; neuroplastic closed-loop requires clinical feedback data not yet available; adversarial pruning population initialization is placeholder. Multi-drug synergy: Bliss-CI discrepancy matrix is descriptive, not prescriptive; L-BFGS-B dose optimization assumes smooth response surface. Epitope module: preliminary screening only—not validated for clinical epitope prediction. AUC=0.80 below NetMHCpan 4.1 (~0.90); allele-blind AUC=0.72 confirms allele identity contributes to allele-aware performance; rare alleles (n<100) underperform (AUC=0.65). Pseudo-sequence AUC=0.917 partially reflects allele encoding (allele-blind 0.72). HLA-II uses NetMHCIIpan alignment for core identification; predictions are derivative, not fully independent; novelty limited to pocket encoding. MOE at N=42 is exploratory, not validated. No wet-lab validation performed.

---

## References

1. Hassett KJ, et al. Mol Ther. 2019;27:1550.
2. Wesselhoeft RA, et al. Nat Commun. 2018;9:2629.
3. Gilleron J, et al. Nat Biotechnol. 2013;31:638.
4. Paunovska K, et al. ACS Nano. 2018;12:7580.
5. Jurtz V, et al. J Immunol. 2017;199:3367.