# Confluencia 3.0: A Circular Topology-Aware Integrated Platform for circRNA Vaccine Design

**Running Title:** Confluencia circRNA Platform with TorusFold

**Keywords:** circRNA, circular topology, TorusFold, pharmacokinetics, immunogenicity, S¹ positional encoding, structure prediction, federated learning

---

## Abstract

Circular RNA (circRNA) presents unique computational challenges arising from its covalently closed topology: position *i* and *i+L* are identical, invalidating standard positional encodings. We present **Confluencia 3.0**, the first integrated platform for circRNA vaccine design that natively accounts for S¹ circular topology through **TorusFold**—a neural architecture with Torus Positional Encoding (TPE) guaranteeing periodicity $|TPE(i) - TPE(i+L)| < 10^{-6}$, circular distance metrics, and rotation-equivariant pair representations.

---

## Introduction

Circular RNA offers compelling advantages for therapeutic applications: covalently closed back-splice junctions confer exonuclease resistance, yielding half-lives of 8-24 hours versus 2-4 hours for linear mRNA (Wesselhoeft et al., 2018). However, circRNA's closed topology creates three computational gaps that existing tools do not address.

**Figure 1: System Architecture Overview**

![System Architecture](D:/IGEM集成方案/confluencia_3_0/docs/fig1_system_architecture.png)

The Confluencia 3.0 platform integrates four core modules: TorusFold (structure prediction), CirculaPK (pharmacokinetics), Immunogenicity Scoring, and RL-ABM optimization. The EventBus architecture enables modular communication between components.

---

## Methods

### TorusFold: Circular Topology-Aware Architecture

**Problem Formulation.** Standard positional encoding breaks periodicity for circRNA: given sequence length L, position i and position i+L are identical, but standard PE assigns different values.

**Figure 2: TorusFold Architecture**

![TorusFold Flow](D:/IGEM集成方案/confluencia_3_0/docs/fig2_torusfold_flow.png)

TorusFold replaces standard positional encoding with Torus Positional Encoding (TPE), using Fourier series on S¹ to guarantee PE(i) = PE(i+L) by construction. The circular distance metric d_circ correctly identifies BSJ-flanking positions as neighbors.

**Torus Positional Encoding (TPE).** We encode positions on a torus S¹×S¹:

$$TPE(i) = \sum_{h=1}^{H} \left[\sin\left(\frac{2\pi h \cdot i}{L}\right), \cos\left(\frac{2\pi h \cdot i}{L}\right)\right]$$

**Figure 6: TorusFold Detailed Architecture**

![TorusFold Architecture](D:/IGEM集成方案/manuscripts/figures/fig6_torusfold_architecture.png)

The architecture includes: (a) TPE layer with guaranteed periodicity, (b) Circular distance bias in attention, (c) CircPairformer with rotation-equivariant triangle operations, and (d) Pair prediction head (reserved for future activation).

### TNBC Simulacrum

**Figure 3: RNActm Model**

![RNActm Model](D:/IGEM集成方案/confluencia_3_0/docs/fig3_rnactm_model.png)

The RNActm tumor-immune interaction model captures: tumor growth (T), TIL dynamics, and protein expression from circRNA translation. ODE system with subclonal evolution tracks Shannon diversity.

**Figure 4: TNBC Subtypes**

![TNBC Subtypes](D:/IGEM集成方案/confluencia_3_0/docs/fig4_tnbc_subtypes.png)

Four TNBC molecular subtypes parameterized from Jiang et al. (2019): BLIS (worst prognosis, early immune escape), BLIA (immune gene signatures), IM (checkpoint inhibitor responsive), LAR (anti-androgen sensitivity).

### CirculaPK: circRNA-Specific Pharmacokinetics

Six-compartment model capturing three circRNA-specific bottlenecks:
1. LNP biodistribution: liver 0.80, spleen 0.10
2. Endosomal escape: 1-4% efficiency
3. IRES-dependent translation: 0.02-0.32/h

### Immunogenicity Scoring

**Figure 5: Immunogenicity Correlation**

![Immunogenicity Correlation](D:/IGEM集成方案/manuscripts/figures/fig5_immunogenicity_correlation.png)

Pathway-resolved immunogenicity scoring: Spearman r=0.91 with Chen et al. (2019) IFN-β (N=7). Four sensing pathways (MDA5/dsRNA, PKR, TLR7, TLR8) with differential m6A suppression.

### RL-ABM Closed-Loop Optimization

**Figure 5: Sequence Evolution**

![Sequence Evolution](D:/IGEM集成方案/confluencia_3_0/docs/fig5_sequence_evolution.png)

REINFORCE optimization with 500 episodes. Reward function: 0.35 efficacy + 0.30 immune_score + 0.20 safety + 0.15 synergy. Convergence by ~400 episodes.

---

## Results

### TorusFold Proxy Experiment

**Figure 6: Validation Results**

![Validation](D:/IGEM集成方案/confluencia_3_0/docs/fig6_validation.png)

TPE vs Standard PE comparison on 50 circBase sequences. Mathematical verification: $|TPE(i) - TPE(i+L)| < 10^{-6}$ for all positions.

### TNBC Simulation

**Figure 3: TNBC Simulation Results**

![TNBC Simulation](D:/IGEM集成方案/manuscripts/figures/fig3_tnbc_simulation.png)

IM subtype sustains immunoediting equilibrium across 30 cycles; BLIS escapes by cycle 12. Shannon diversity increases from 0.4 to 1.2 under chemotherapy.

### Platform Dashboard

**Figure 7: Dashboard Interface**

![Dashboard](D:/IGEM集成方案/confluencia_3_0/docs/fig7_dashboard.png)

Streamlit web interface with six pages: circRNA design, pharmacokinetics, immunogenicity, epitope screening, drug response, report export.

### Workflow Integration

**Figure 8: Complete Workflow**

![Workflow](D:/IGEM集成方案/confluencia_3_0/docs/fig8_workflow.png)

End-to-end workflow: sequence input → TorusFold structure → CirculaPK simulation → immunogenicity scoring → epitope prediction → drug response → optimization → report generation.

---

## Benchmark Results

### Drug Response Prediction

**Figure 1: Architecture and Benchmarks**

![Architecture](D:/IGEM集成方案/figures/fig1_architecture.png)

MOE ensemble architecture for drug response prediction.

**Figure 2: MOE Mechanism**

![MOE Mechanism](D:/IGEM集成方案/figures/fig2_moe_mechanism.png)

Seven regression experts weighted by validation performance: Ridge, HGB, RF, MLP, XGB, LGB, ET.

**Figure 3: Learning Curves**

![Learning Curves](D:/IGEM集成方案/figures/fig3_learning_curves.png)

Training convergence across 500 episodes.

**Figure 4: Ablation Study**

![Ablation](D:/IGEM集成方案/figures/fig4_ablation.png)

Feature importance analysis showing MHC pseudo-sequence encoding contribution.

**Figure 5: Baseline Comparison**

![Baselines](D:/IGEM集成方案/figures/fig5_baselines.png)

Comparison against ESM-2, NetMHCpan, and other baselines.

**Figure 6: Validation Scatter**

![Validation](D:/IGEM集成方案/figures/fig6_validation.png)

Prediction vs actual for drug response and epitope efficacy.

---

## Discussion

We present the first integrated platform that natively accounts for circRNA's S¹ circular topology. TorusFold demonstrates that topology-aware neural design improves performance even in proxy tasks. We propose circRNA-CASP as a community validation mechanism.

---

## Confluencia Hub: Federated Learning

**Figure: HD Architecture**

![HD Architecture](D:/IGEM集成方案/docs/ARCHITECTURE_3.0_HD.png)

Hub architecture for federated model aggregation: trained model bundles with metadata, ethics gating, privacy preservation.

---

## Code Availability

**Repository:** github.com/RomanCohort/confluencia (MIT License)

**Interfaces:** Python API, Streamlit (6 pages), CLI, R package, PyQt6 desktop IDE

---

## References

1. Wesselhoeft RA, et al. RNA circularization diminishes immunogenicity. Nat Commun. 2018;9:2629.

2. Jiang YZ, et al. Genomic and Transcriptomic Landscape of TNBC. Cancer Cell. 2019;35:428.

3. Chen YG, et al. Sensing Self and Foreign Circular RNAs by Intron Identity. Mol Cell. 2019;73:422.

4. Gilleron J, et al. Image-based analysis of LNP-mediated siRNA delivery. Nat Biotechnol. 2013;31:638.

5. Lorenz R, et al. ViennaRNA Package 2.0. Algorithms Mol Biol. 2011;6:26.

---

**Figure Summary:**

| Figure | Content | Source |
|--------|---------|--------|
| Fig 1 | System Architecture | confluencia_3_0/docs/fig1_system_architecture.png |
| Fig 2 | TorusFold Flow | confluencia_3_0/docs/fig2_torusfold_flow.png |
| Fig 3 | RNActm Model | confluencia_3_0/docs/fig3_rnactm_model.png |
| Fig 4 | TNBC Subtypes | confluencia_3_0/docs/fig4_tnbc_subtypes.png |
| Fig 5 | Sequence Evolution | confluencia_3_0/docs/fig5_sequence_evolution.png |
| Fig 6 | Validation | confluencia_3_0/docs/fig6_validation.png |
| Fig 7 | Dashboard | confluencia_3_0/docs/fig7_dashboard.png |
| Fig 8 | Workflow | confluencia_3_0/docs/fig8_workflow.png |
| Fig A | Architecture/Benchmarks | figures/fig1_architecture.png - fig6_validation.png |
| Fig HD | Hub Architecture | docs/ARCHITECTURE_3.0_HD.png |