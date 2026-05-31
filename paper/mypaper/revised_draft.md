# Confluencia: An Integrated circRNA Drug Discovery Platform with Six-Compartment Pharmacokinetic Modeling

**IGEM-FBH Team**

**Affiliations:** IGEM-FBH Team, The First Bethune Hospital of Jilin University, College of Computer Science and Technology, Jilin University, ChangChun, China

**Contact:** 18806370529@163.com

---

## Abstract

**Summary:** Confluencia is a fully integrated computational platform for circRNA drug discovery featuring RNACTM—a six-compartment pharmacokinetic model simulating circRNA delivery from injection through LNP encapsulation, endosomal escape, cytoplasmic translation and clearance over 72 h. The platform includes ADMET multi-endpoint toxicity prediction with toxicophore alerts, dose-dependent toxicity modeling, and a five-dimension hybrid evaluation system. Confluencia employs a sample-size-adaptive Mixture of Experts ensemble and Mamba3Lite multi-scale sequence encoding optimized for small-sample circRNA studies.

**Availability:** https://github.com/RomanCohort/confluencia (MIT license). Python 3.8+ on Linux/MacOS/Windows. Docker and Confluencia Studio IDE available.

**Contact:** 18806370529@163.com

---

## Introduction

Therapeutic circular RNA (circRNA) offers advantages over linear mRNA in stability, expression duration, and immunogenicity control (Wesselhoeft et al. 2018; Chen et al. 2019), yet computational tools remain fragmented. Researchers switch between specialized predictors (NetMHCpan for binding, QSAR packages for toxicity) while lacking circRNA-specific pharmacokinetic modeling. Confluencia addresses this gap as the first integrated platform for circRNA drug discovery, featuring RNACTM—a six-compartment PK model with literature-derived parameters for five nucleotide modifications (unmodified, Ψ, m6A, 5mC, ms2m6A), a sample-size-adaptive MOE ensemble, Mamba3Lite sequence encoder, ADMET toxicity prediction, and 5D joint evaluation.

---

## System Overview

### RNACTM Pharmacokinetic Model

RNACTM implements a six-compartment ODE system: Injection Site → LNP Circulation → Endosomal → Cytoplasm → Translation → Clearance. Rate constants derive from published studies: absorption k_abs=0.12/h (Liu et al. 2023), endocytosis k_endo=0.48/h (Wesselhoeft et al. 2018), endosomal escape k_escape=0.02/h (4.43% escape fraction; Hassett et al. 2019), translation k_trans=0.10/h, degradation k_deg=0.111/h (half-life 6.24 h; Wesselhoeft et al. 2018). Ψ modification extends half-life to ~15 h (Liu et al. 2023), m6A extends to ~10.8 h.

### MOE Ensemble and Mamba3Lite

The MOE ensemble selects experts adaptively: N<80 uses Ridge+HGB; 80≤N<300 adds RF; N≥300 adds MLP. Weights derive from inverse OOF-RMSE from 5-fold stratified CV. Mamba3Lite encodes 8-11 AA peptides via three parallel state-space recurrences (fast/medium/slow decay) and four-scale pooling (residue/local/meso/global), producing 96-dim embeddings. Feature importance (40.3% contribution) was assessed via ablation experiments on 288K IEDB data by removing Mamba components and measuring MAE increase.

### ADMET and Toxicophore Detection

QSAR models cover hERG, AMES, five CYP450 isoforms, BBB, and hepatotoxicity using literature-derived weights (Wessel et al. 2015; Hansen et al. 2009; Veith et al. 2009). ~90 SMARTS patterns from PAINS (Baell & Holloway 2010), Brenk filters (Brenk et al. 2008), and circRNA-specific alerts detect structural liabilities. Dose-dependent modeling estimates LD50, TD50, and therapeutic index.

---

## Results

**Table 1. Confluencia performance across modules**

| Module | Dataset | Method | Key Metric | Value |
|--------|---------|--------|------------|-------|
| **Epitope Prediction** | N=300 | MOE Ensemble | MAE, R² | 0.389, 0.819 |
| **vs Baselines** | | Ridge | R² | 0.533 |
| | | HGB | R² | 0.794 |
| **Drug Efficacy** | N=200 | Ridge | MAE, R² | 0.037, 0.984 |
| **Drug Ablation** | N=200 | No Morgan FP | R² | 0.960 (+0.29 vs full) |
| **Binding (288K IEDB)** | 288,135 | HGB | AUC | 0.734 |
| **vs NetMHCpan** | 61 peptides | Confluencia | AUC | 0.678 (gap 0.12-0.17) |
| **RNACTM Validation** | Literature | Half-life error | Unmodified: 4.1% | Ψ: 4.1% |
| **Toxicophore** | 25 PAINS | Recall | 100% | |

RNACTM half-life predictions match literature within 4.1% error (unmodified: 6.24h vs 6.0h; Ψ: 15.61h vs 15.0h). MOE ensemble achieves 39.2% MAE reduction versus Ridge baseline for epitope prediction. Drug prediction reveals that removing 2048-bit Morgan fingerprints improves R² from 0.67 to 0.96—high-dimensional sparse features overfit in small samples. Binding prediction (AUC 0.73) trails NetMHCpan-4.1 (AUC 0.92-0.96); we recommend NetMHCpan for binding-only tasks. Toxicophore detection achieves 100% recall on PAINS patterns.

---

## Discussion

Confluencia integrates circRNA-specific PK simulation (RNACTM), small-sample ensemble learning (MOE), and ADMET evaluation unavailable in specialized binding predictors. Honest limitations: RNACTM parameters derive from literature rather than fitted PK data; ADMET weights are literature-derived; MHC-I only (no class II); clinical trial simulation is parameterized. The 5D evaluation shows weak correlation (Spearman r=0.135) with literature IFN responses, indicating current weighting may not fully capture immune response determinants—future versions should incorporate additional immunogenicity features.

---

## Acknowledgements

IGEM-FBH Team, Jilin University. Funding: First Bethune Hospital and College of Computer Science and Technology. We thank IEDB, ChEMBL, and open-source communities.

---

## References

1. Wesselhoeft RA et al. (2018) Nature Communications 9:2629.
2. Chen YG et al. (2019) Nature 586:651-655.
3. Liu X et al. (2023) Nature Communications 14:2548.
4. Hassett KJ et al. (2019) Molecular Therapy 27:824-836.
5. Reynolds DM et al. (2020) Bioinformatics 36:4138-4145.
6. Jurtz VI et al. (2017) Journal of Immunology 199:3360-3368.
7. O'Donnell TJ et al. (2018) Cell Systems 7:129-132.
8. Baell JB, Holloway GA (2010) J Med Chem 53:2719-2740.
9. Brenk R et al. (2008) ChemMedChem 3:435-444.
10. Wessel MD et al. (2015) J Chem Inf Model 55:2243-2255.
11. Hansen NT et al. (2009) Bioorg Med Chem 17:4110-4117.
12. Veith H et al. (2009) Chem Res Toxicol 22:237-247.
13. Pedregosa F et al. (2011) JMLR 12:2825-2830.
14. Gu A, Dao T (2023) arXiv:2312.00752.
15. Vita R et al. (2019) Nucleic Acids Res 47:D339-D343.
16. Yang W et al. (2013) Nucleic Acids Res 41:D955-D961.
17. Lin Z et al. (2022) Science 379:1123-1130.
18. Wang Z et al. (2023) Nat Rev Drug Discov 22:345-362.
19. Xiong C et al. (2024) Acta Pharmacol Sin 45:1-15.
20. Ramsundar B et al. (2019) Deep Learning for Life Sciences. O'Reilly.
21. Egan WJ et al. (2000) Pharm Res 17:147-153.
22. Delaney JS (2004) J Chem Inf Comput Sci 44:1000-1005.
23. Sushko I et al. (2012) Chem Res Toxicol 25:1479-1492.
24. Yang D et al. (2025) Int J Biol Macromol 322:146767.
25. Egan WJ et al. (2000) Pharm Res 17:147-153.

---

**Figure 1 Caption:** Confluencia architecture: unified inputs (SMILES, epitope sequences, MHC alleles, dosing parameters) processed through Drug Pipeline (MOE ensemble, ADMET), Epitope Pipeline (Mamba3Lite, MHC features), RNACTM PK, and Five-Gene Module → 5D evaluation → Go/Conditional/No-Go output.

---

**Word Count**: ~1,600 words (condensed for 2-4 page Application Note format)

**Page Estimate**: ~2.5 pages with Table 1 and references