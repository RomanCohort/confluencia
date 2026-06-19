---
title: "Confluencia 3.0: an integrated platform for circRNA vaccine design with TNBC subtype-specific simulation"
author:
  - Author One
  - Author Two
affiliation:
  - Institution
keywords: circRNA, TNBC, vaccine design, pharmacokinetics, immunogenicity, simulation, application note
---

# Confluencia 3.0: an integrated platform for circRNA vaccine design with TNBC subtype-specific simulation

## 1 Introduction

Circular RNA (circRNA) offers compelling advantages over linear mRNA as vaccine cargo: covalently closed back-splice junctions (BSJ) confer exonuclease resistance, yielding half-lives of 8--24 h versus 2--4 h for linear mRNA (Wesselhoeft et al., 2018). Triple-negative breast cancer (TNBC) presents a particularly suitable target, as Jiang et al. (2019) identified four molecular subtypes---BLIS, BLIA, IM, and LAR---with distinct immune microenvironments demanding subtype-adaptive vaccine strategies.

However, no existing platform links circRNA design to TNBC subtype-specific simulation. Current tools address components in isolation: ViennaRNA (Lorenz et al., 2011) predicts secondary structure, PK-Sim models pharmacokinetics, and PhysiCell simulates tumor dynamics. Three critical gaps remain. First, circRNA pharmacokinetics differ fundamentally from linear mRNA: LNP encapsulation creates tissue-specific biodistribution (liver 80%, spleen 10%), endosomal escape is a bottleneck at 1--4% efficiency (Gilleron et al., 2013), and circRNA degradation follows exonuclease-resistant pathways. Standard PK models omit these features. Second, circRNA innate immune sensing differs: circRNAs lack 5' termini, invalidating RIG-I 5'-ppp sensing (Hornung et al., 2006); immunogenicity instead arises from dsRNA backbone structures sensed by MDA5 (Chen et al., 2019) and modulated by intron identity. Existing tools assume linear RNA sensing. Third, no platform provides closed-loop optimization coupling circRNA sequence design with patient-specific simulation.

Confluencia 3.0 addresses these gaps through an event-driven architecture coupling four modules: (1) TNBC Simulacrum with spatial tumor microenvironment (TME) simulation, (2) CirculaPK six-compartment pharmacokinetics, (3) circRNA-specific immunogenicity scoring, and (4) RL-ABM closed-loop sequence optimization. The platform provides five access interfaces (Python API, Streamlit web UI, CLI, R package, PyQt6 desktop IDE) and supports federated model sharing via Confluencia Hub.

## 2 System overview

Confluencia 3.0 employs an EventBus architecture that decouples modules via event-driven communication (Fig. 1). Each component subscribes to events and emits results independently, enabling modular extension: new algorithms can replace existing implementations by subscribing to the same events without modifying other subsystems. The architecture supports 18+ event types across eight subsystems, with deterministic event ordering ensuring reproducibility.

![Figure 1: System architecture. (A) EventBus coupling four modules: TNBC Simulacrum, CirculaPK, Immunogenicity Scoring, and RL-ABM. (B) Spatial TME simulation showing three compartments. (C) Subclonal evolution under treatment pressure.](fig3_tnbc_simulation.png)

The platform defaults to established external tools via lazy-loading bridges: ViennaRNA for structure prediction, PK-Sim or physiologically-based PK models as alternatives to CirculaPK, and NetMHCpan (AUC ~0.90) for standalone MHC binding screening. Confluencia's custom modules serve integration needs rather than competing with established benchmarks.

## 3 Methods

### 3.1 TNBC Simulacrum

The TNBC simulation module parameterizes four molecular subtypes from Jiang et al. (2019) (360 tumors, RNA-seq and immune profiling):

- **BLIS** (n=108): TIL 0.08--0.15, worst prognosis, early immune escape by cycle 12
- **BLIA** (n=72): TIL 0.25--0.40, immune-activated gene signatures
- **IM** (n=85): TIL 0.50--0.70, PD-L1 0.40--0.60, checkpoint inhibitor responsive
- **LAR** (n=95): AR expression 0.70--0.85, anti-androgen sensitivity

Tumor-immune dynamics follow an ODE system:

$$\frac{dT}{dt} = r_T \cdot T \cdot \left(1 - \frac{T}{K}\right) - d_T \cdot TIL \cdot T$$

$$\frac{dTIL}{dt} = r_{TIL} \cdot \left(\frac{T}{K}\right) - d_{TIL} \cdot T$$

$$\frac{dP}{dt} = k_{cp} \cdot circRNA - d_P \cdot P$$

Three immunoediting phases (elimination, equilibrium, escape) are modeled with treatment arms for chemotherapy, immunotherapy (PD-1 blockade), and circRNA vaccine. Subclonal evolution tracks tumor heterogeneity via Shannon diversity $H = -\sum p_i \log(p_i)$ with drug-induced mutation rate increase (1%→50%/step under treatment), capturing resistance emergence (Ding et al., 2012).

The spatial TME simulation models three compartments (hypoxic core, immune-rich margin, stromal barrier) with nine immune cell populations (CD8+ T, CD4+ T, Treg, B, NK, M1/M2 macrophages, MDSCs, CAFs) and six cytokines (IFN-γ, TNF-α, IL-2, IL-6, IL-10, TGF-β). TME classification (hot, cold, excluded, mixed) directly informs immunotherapy response prediction.

A four-gene signature encoder (TROP2, NECTIN4, LIV-1, B7-H4) computes 19-dimensional feature vectors for therapeutic response prediction with literature-derived correlations.

### 3.2 CirculaPK: circRNA-specific pharmacokinetics

CirculaPK implements a six-compartment model: Injection → LNP → Endosome → Cytoplasm → Protein → Clearance, capturing three circRNA-specific bottlenecks:

1. **LNP encapsulation** with tissue-specific biodistribution (liver 0.80, spleen 0.10; Paunovska et al., 2018)
2. **Endosomal escape** at 1--4% efficiency ($k_{ec} = 0.025$/h; Gilleron et al., 2013; Hou et al., 2021)
3. **IRES-dependent translation** at 0.02--0.32/h depending on IRES sequence (Martinez-Salas et al., 2018)

Rate constants are literature-derived (not fitted): $k_{ab}=0.80$/h, $k_{be}=0.025$/h, $k_{ec}=0.025$/h, $k_{cp}=0.02$--$0.32$/h (IRES-dependent), $k_{cd}=0.04$--$0.12$/h (modification-adjusted), $k_{pc}=0.10$--$0.20$/h. RK45 integration outputs AUC, $C_{max}$, and half-life. Nucleotide modifications alter $k_{cd}$: m6A reduces to 0.06--0.08/h; Ψ reduces to 0.04--0.06/h.

### 3.3 circRNA-specific immunogenicity scoring

This module implements the first scoring system that explicitly distinguishes circRNA from linear RNA innate immune activation. Since circRNAs lack 5' termini, RIG-I 5'-ppp sensing is inapplicable (Hornung et al., 2006). We model four sensing pathways with literature-derived weights:

| Pathway | Weight | Sensor | Mechanism | Confidence |
|---------|--------|--------|-----------|------------|
| MDA5/dsRNA | 0.35 | MDA5 | Long dsRNA structures (>16 bp) | Medium |
| PKR | 0.30 | PKR | dsRNA length >33 bp | High |
| TLR7 | 0.20 | TLR7 | GU-rich ssRNA motifs | Medium |
| TLR8 | 0.15 | TLR8 | AU-rich ssRNA motifs | Medium |

![Figure 2: Immunogenicity model validation. (A) Correlation with Chen et al. (2019) IFN-β measurements (Spearman r=0.91, N=7). (B) HEK293 validation (r=0.68 [CI 0.26--0.88], N=15). (C) Pathway contribution analysis showing MDA5/dsRNA dominates discrimination. (D) GC confound analysis: pathway decomposition vs. GC-only baseline (ΔAIC = −8.2, p=0.004).](fig5_immunogenicity_correlation.png)

Differential m6A suppression is modeled with pathway-specific intensity: MDA5 ~90%, TLR7/8 ~30%, PKR ~20%. These estimated values correct the oversimplified "m6A reduces immunogenicity" assumption; uniform suppression reduces partial correlation from $r=0.42$ to $r=0.31$ ($p=0.08$), demonstrating non-redundant information. Bidirectional m6A modeling balances evasion (dsRNA destabilization) against enhancement (IRES translation upregulation; Yang et al., 2018).

### 3.4 RL-ABM closed-loop sequence optimization

circRNA sequence optimization is formulated as a Gym-style reinforcement learning environment with four BSJ-protected operators: point mutation, IRES insertion, nucleotide modification selection (m6A, Ψ, 5mC), and combination therapy adjustment. The multi-objective reward function integrates:

$$R = 0.35 \cdot \text{efficacy} + 0.30 \cdot \text{immune\_score} + 0.20 \cdot \text{safety} + 0.15 \cdot \text{synergy}$$

where efficacy = CirculaPK-predicted protein expression (AUC-normalized), immune_score = Module 3 immunogenicity, safety = ADMET toxicity with risk gate penalty (threshold 0.70), and synergy = multi-drug Bliss-CI score. REINFORCE converges by ~400 episodes (5 seeds, $\sigma=0.03$). Four synergy models are implemented: Bliss independence, Loewe additivity, HSA, and Chou-Talalay CI, with L-BFGS-B dose optimization.

### 3.5 Bio-mimetic drug ADMET prediction

Four brain-inspired components enable patient-specific ADMET prediction: (1) Topology Pharmacophore Network (scale-free graph of pharmacophore nodes), (2) Tissue-Specific Dynamic Attention (patient-specific gating weights), (3) Adversarial Synaptic Pruning (Pareto optimization with competitive selection), and (4) Neuroplastic Closed-Loop (three-tier adaptation when prediction error >0.3).

## 4 Results

### 4.1 TNBC simulation

IM subtype sustains immunoediting equilibrium (TIL >0.50) across 30 cycles; BLIS escapes by cycle 12 (TIL <0.05). Shannon diversity increases from 0.4 to 1.2 under chemotherapy, with dominant clone frequency decreasing from 0.85 to 0.42. Hot TME (IM) shows 2.3× higher simulated response than cold TME (BLIS). Parameter-swap validation confirms internal consistency: BLIS with IM parameters sustains equilibrium, while IM with BLIS parameters escapes.

### 4.2 Pharmacokinetics

Simulated half-lives match Wesselhoeft et al. (2018) with 12% relative error [CI 3--21%] (N=4). m6A extends half-life to ~15--22 h; Ψ to ~20--30 h. The model captures the endosomal escape bottleneck: only 2--4% of injected circRNA reaches the cytoplasm. Six-compartment AIC = 18.2 vs. two-compartment AIC = 22.7 (ΔAIC = 4.5), though N=4 is insufficient for statistical model discrimination.

### 4.3 Immunogenicity

Primary benchmark: Spearman $r=0.91$ with Chen et al. (2019) IFN-β (N=7); LOO median $r=0.87$ [IQR 0.82--0.91]. Secondary validation: HEK293 $r=0.68$ [CI 0.26--0.88] (N=15). Pathway decomposition provides statistically significant improvement over GC-only baseline ($r=0.85$ vs. $r=0.79$, ΔAIC = −8.2, $p=0.004$, N=50). MDA5/dsRNA contributes most to discrimination (ablation changes 3/15 ranks vs. 2/15 for PKR).

## 5 Software features

### 5.1 Access interfaces

Confluencia 3.0 provides five access modes targeting diverse user communities:

- **Python API:** `import confluencia_3_0; confluencia_3_0.simulate(config)`
- **Streamlit web UI:** 6-page interface (CircRNA Analysis, Drug Prediction, Epitope Screening, TNBC Simulator, Joint Analysis, Report Export)
- **CLI:** `confluencia simulate --subtype IM --steps 100`
- **R package:** Functions `cf_drug_predict()`, `cf_hub_push_model()`, `cf_hub_pull_model()`
- **PyQt6 desktop IDE:** Editor, notebook, variable explorer, git integration, natural-language query

### 5.2 Confluencia Hub: federated model sharing

Confluencia Hub enables federated model sharing where labs upload trained weights (not raw data), addressing the small-sample problem endemic to circRNA work (N=42 drug, N=7 immunogenicity). Privacy is preserved: no SMILES or nucleotide sequences are logged; `strip_env_medians` removes statistical traces before upload. Ethics-gated uploads require data source declaration (DOI or IRB number) and dual-use screening. SHA256 hash verification mitigates code execution risks.

### 5.3 Installation

```bash
pip install confluencia          # core dependencies
pip install confluencia[all]     # including deep learning backends
```

Python 3.10+, MIT License, pytest 87% coverage, CI/CD via GitHub Actions.

## 6 Discussion

Confluencia 3.0 contributes an integrated platform that links circRNA vaccine design to TNBC subtype-specific simulation through four key innovations: (1) circRNA-specific immune sensing via MDA5/dsRNA pathway with pathway decomposition providing statistically significant improvement over GC-only baseline (ΔAIC = −8.2, $p=0.004$); (2) six-compartment circRNA PK capturing the endosomal escape bottleneck (1--4% efficiency); (3) differential m6A suppression modeling with pathway-specific values; and (4) RL-ABM closed-loop sequence optimization with multi-objective reward.

Important limitations should be noted. TNBC simulation outcomes are determined by input parameters (confirmed by parameter-swap experiment), validating internal consistency but not external predictions. Immunogenicity weights are literature-derived, not empirically calibrated; N=7 primary benchmark provides statistical power ≈0.35. PK validation with N=4 cannot statistically distinguish six-compartment from simpler models. m6A suppression values (90%/30%/20%) are estimated, not measured. All claims are hypothesis-generating; wet-lab validation is ongoing with collaborating medical school researchers (IFN-β ELISA, qRT-PCR half-life measurement, PDX models).

The EventBus architecture ensures algorithmic extensibility: as new methods emerge (e.g., improved circRNA structure predictors, validated PK parameters), they can replace existing implementations by subscribing to the same events without modifying other subsystems. Confluencia Hub addresses the small-sample problem through federated aggregation while preserving privacy and requiring ethical data source declaration.

## Acknowledgements

We thank [collaborating medical school researchers] for ongoing wet-lab validation support.

## Funding

This work was supported by [funding information to be added].

## References

Chen,Y.G. et al. (2019) Sensing Self and Foreign Circular RNAs by Intron Identity. *Mol. Cell*, 73, 422.

Ding,L. et al. (2012) Clonal evolution in relapsed acute myeloid leukaemia revealed by whole-genome sequencing. *Nature*, 481, 506.

Gilleron,J. et al. (2013) Image-based analysis of lipid nanoparticle-mediated siRNA delivery, intracellular trafficking and endosomal escape. *Nat. Biotechnol.*, 31, 638.

Hornung,V. et al. (2006) 5'-Triphosphate RNA is the ligand for RIG-I. *Science*, 314, 994.

Hou,X. et al. (2021) Lipid nanoparticles for mRNA delivery. *Nat. Rev. Mater.*, 6, 1078.

Jiang,Y.Z. et al. (2019) Genomic and Transcriptomic Landscape of Triple-Negative Breast Cancer. *Cancer Cell*, 35, 428.

Lorenz,R. et al. (2011) ViennaRNA Package 2.0. *Algorithms Mol. Biol.*, 6, 26.

Martinez-Salas,E. et al. (2018) IRES mechanisms: connecting structure and function. *Trends Microbiol.*, 26, 651.

Paunovska,K. et al. (2018) Quantification of nanoprotein distribution at the single-cell level. *ACS Nano*, 12, 7580.

Wesselhoeft,R.A. et al. (2018) RNA circularization diminishes immunogenicity and can extend translation duration in vivo. *Nat. Commun.*, 9, 2629.

Yang,Y. et al. (2018) Extensive translation of circular RNAs driven by N6-methyladenosine. *Cell Res.*, 28, 743.