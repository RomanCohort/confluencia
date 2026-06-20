# Confluencia 3.0: circRNA Vaccine Design with TNBC Subtype Simulation

**Running Title:** Confluencia circRNA-TNBC Platform

**Keywords:** circRNA, TNBC, simulation, immunogenicity, pharmacokinetics

---

## Abstract (100 words)

Confluencia 3.0 links circRNA design to TNBC subtype simulation through four modules: TNBC Simulacrum (Jiang 2019 subtypes with ODE tumor dynamics), CirculaPK (six-compartment PK modeling), heuristic immunogenicity scoring for innate pathways, and REINFORCE sequence evolution. An EventBus runs iterative in silico experiments using simulated outcomes to refine sequences. External benchmarks: immunogenicity scores vs Chen 2019 IFN-β (r=0.91, N=7; HEK293 N=15 r=0.68 [0.26-0.88]), PK vs Wesselhoeft 2018 (12% error, N=4). Code: github.com/IGEM-FBH/confluencia-3.0 (MIT).

---

## Introduction (100 words)

TNBC has four molecular subtypes (Jiang 2019: BLIS, BLIA, IM, LAR) with different immune microenvironments, and circRNA is more stable than linear mRNA for vaccine cargo (Wesselhoeft 2018). No existing tool combines circRNA design with subtype-specific tumor simulation. Confluencia 3.0 fills this gap: TNBC Simulacrum with heterogeneity tracking, CirculaPK pharmacokinetics, heuristic immunogenicity scoring with circRNA pathway logic, and Pareto sequence evolution driven by simulated outcomes. The framework generates in silico hypotheses for circRNA vaccine design; wet-lab validation is still needed.

---

## Methods (250 words)

**TNBC Simulacrum.** Jiang 2019 four-subtype parameters (derived from n=360 TNBC tumors with RNA-seq and immune profiling, Supplementary Table S2): BLIS (TIL 0.08-0.15 [mean 0.12], worst prognosis, BRCA1-associated, n=108 tumors); BLIA (TIL 0.25-0.40 [mean 0.33], immune gene signatures, n=72); IM (PD-L1 0.40-0.60 [mean 0.50], TIL 0.50-0.70 [mean 0.60], checkpoint inhibitor responsive, n=85); LAR (AR expression 0.70-0.85, anti-androgen sensitivity, n=95). Shannon diversity tracks subclone heterogeneity with drug-induced instability (mutation rate scales with treatment pressure). TME simulation: nine immune cell populations, six cytokines (IFN-gamma, TNF-alpha, IL-2, IL-6, IL-10, TGF-beta), three spatial compartments (hypoxic core, immune-rich margin, stromal barrier), TME classified as hot/cold/excluded/mixed. Treatment arms: chemotherapy, immunotherapy, circRNA.

**CirculaPK.** Six compartments: Injection → LNP → Endosome → Cytoplasm → Protein → Clearance. Rate constants from literature: uptake 0.80/h, escape 0.025/h (Gilleron 2013), translation 0.02-0.32/h, degradation 0.04-0.12/h. Six-compartment vs simplified 2-compartment: the additional structure captures circRNA-specific bottlenecks (LNP encapsulation, endosomal escape at 1-4% efficiency) that linear mRNA PK models omit; however, with limited validation data (N=4), parameter tuning is not feasible and values are fixed from literature rather than fitted. RK45 integration outputs AUC, half-life.

**Immunogenicity heuristic.** Four pathway weights (RIG-I 0.35, TLR7 0.20, TLR8 0.15, PKR 0.30) come from literature, not empirical calibration. circRNAs lack 5' termini; RIG-I is scored via dsRNA backbone structures (not the canonical 5'-ppp pathway inapplicable to covalently closed circRNA), as proxy for MDA5 co-sensing activation (Chen 2019). TLR7/8 are scored separately with distinct GU-rich and AU-rich motif preferences, modeling endosomal sensing dominant for LNP formulations (>96% endosomal residence). PKR uses the >33 bp dsRNA threshold (Nallagatla 2007) without circularity adjustment. m6A has differential suppression across pathways (RIG-I 90%, TLR 30%, PKR 20%) reflecting pathway-specific modification sensitivity; m6A immunomodulation is modeled bidirectionally (evasion via RIG-I suppression vs. potential enhancement via translation upregulation). Sensitivity analysis: ±50% weight variation preserves rank order for 12/15 sequences. Ablation: removing PKR (redistributing 0.30) changes 2/15 ranks; removing RIG-I changes 3/15, so RIG-I contributes most to discrimination.

**Sequence evolution.** REINFORCE (500 episodes; reward plateaus by ~400 episodes) with BSJ-protected operators: mutation, IRES insertion, modification selection. Reward = weighted sum of stability, translation efficiency, and immunogenicity score (heuristic serves as reward signal—optimizer converges on heuristic's landscape, not validated biological optima). Multi-objective formulation: 0.35 efficacy + 0.30 immune + 0.20 safety + 0.15 synergy. Pareto front balances stability, translation, immune evasion. EventBus coordinates modules via deterministic event ordering with 18+ event types across eight subsystems (tumor, TME, treatment, biomarker, clinical, circRNA, evolution, joint evaluation).

---

## Results (150 words)

**TNBC simulation.** IM subtype sustains immunoediting equilibrium (TIL >0.50); BLIS escapes by cycle 12 (TIL <0.05). Shannon diversity: 0.4→1.2 under chemotherapy (30 cycles).

**PK validation.** Simulated half-lives match Wesselhoeft 2018 experimental values (12% relative error, N=4 sequences). m6A extends half-life to ~15-22 h; Psi to ~20-30 h.

**Immunogenicity.** Chen 2019 correlation: r=0.91 (Spearman, N=7 circRNA sequences with published IFN-β); N=7 is small and single outlier can dominate, but rank order preserved across sequences supports heuristic validity; leave-one-out analysis: removing highest-scoring sequence reduces r to 0.79, removing lowest increases to 0.94—direction consistent but magnitude sensitive to individual points. HEK293 validation: N=15 (independent from Chen 2019 set), r=0.68 [CI 0.26-0.88]—wide CI reflects limited sample; this supports heuristic framework as exploratory. GC-immunogenicity correlation r=0.85 (N=50 circBase, Spearman); partial correlation controlling for GC content: pathway scores retain r=0.42 with IFN-β (p=0.03), indicating pathway decomposition captures signal beyond GC-mediated dsRNA propensity alone.

**Iterative experiment (computational proof-of-concept).** Three-round refinement: simulated tumor response 0.32→0.48→0.71. This demonstrates optimizer convergence on simulator reward surface, not biological model validation. IM-optimized shows 2.3x higher simulated response than BLIS-optimized—expected given IM's higher TIL parameterization (0.50-0.70 vs BLIS 0.08-0.15), confirming simulator recaptures input assumptions.

---

## Comparison

| Component | This work | Established | Validation |
|-----------|-----------|-------------|------------|
| TNBC subtype | Jiang 2019 | PhysiCell (generic) | Literature-parameterized |
| circRNA PK | 6-compartment | PK-Sim (linear) | 12% error vs Wesselhoeft |
| Immunogenicity | 4 pathways | Chen 2019 correlation | r=0.91 (N=7), exploratory |

---

## Availability

github.com/IGEM-FBH/confluencia-3.0 (MIT). Python 3.10+, pytest 87% coverage, CI/CD via GitHub Actions.

---

## Limitations

TNBC: ODE dynamics without stochastic effects; no TCGA/METABRIC validation performed; subtype parameters derive from Jiang 2019 cohort (n=360) but simulated dynamics are simplified abstractions. Immunogenicity: pathway weights are uncalibrated heuristics; 20% of sequences show rank inversion under weight variation; removing PKR redistributes 0.30 weight and changes 2/15 sequence ranks; N=7 Chen 2019 correlation is sensitive to individual points (LOO range 0.79-0.94); N=15 HEK293 validation has wide CI. GC content partially confounds pathway scores (r=0.85); partial correlation controlling for GC retains r=0.42, indicating but not fully resolving the confound. PK: literature priors only, not fitted to time-course data; six-compartment structure captures circRNA bottlenecks but with N=4 validation cannot distinguish from simpler models; no patient-specific PK. Sequence evolution: REINFORCE optimizes on heuristic reward surface, not validated biological optima; convergence assessed by reward plateau (~400 episodes) but no out-of-distribution validation of evolved sequences. EventBus: single-threaded, deterministic event ordering; does not affect reward computation. Simulated outcomes validate internal consistency, not external predictions. All claims are hypothesis-generation, not validated predictions.

---

## References

1. Wesselhoeft RA, et al. Nat Commun. 2018;9:2629.
2. Jiang YZ, et al. Cancer Cell. 2019;35:428.
3. Chen YG, et al. Mol Cell. 2019;73:422.
4. Gilleron J, et al. Nat Biotechnol. 2013;31:638.
5. Nallagatla SR, et al. Science. 2007;318:1455.