# Confluencia 3.0: Integrated circRNA Vaccine Design with TNBC Subtype Simulation

**Running Title:** Confluencia circRNA-TNBC Platform

**Keywords:** circRNA, TNBC, simulation, immunogenicity, pharmacokinetics, deep learning, structure prediction, federated learning, data sharing

---

## Abstract

Confluencia 3.0 integrates computational circRNA vaccine design with TNBC molecular subtype simulation through an event-driven architecture coupling four modules: (1) TNBC Simulacrum with spatial TME simulation (nine immune cell populations, six cytokines, three spatial compartments, subclonal evolution), (2) CirculaPK six-compartment pharmacokinetics capturing circRNA-specific bottlenecks (1-4% endosomal escape), (3) circRNA-specific innate immune sensing via MDA5/dsRNA pathway, TLR7/8, and PKR with differential m6A suppression modeling, and (4) RL-ABM closed-loop sequence optimization. Preliminary benchmarks: immunogenicity scores correlate with Chen 2019 IFN-β (Spearman r=0.91, N=7; HEK293 N=15 r=0.68 [0.26-0.88]); PK matches Wesselhoeft 2018 half-lives (12% error [CI 3-21%], N=4). Sample sizes are insufficient for definitive validation; all results are hypothesis-generating. We further propose TorusFold, a theoretical architecture for circRNA 3D structure prediction accounting for S¹ topology through Torus Positional Encoding with guaranteed periodicity, circular distance metric, and rotation-equivariant CircPairformer; the pair prediction head is currently non-functional. Wet-lab validation ongoing. Code: github.com/RomanCohort/confluencia (MIT). Federated model sharing via Confluencia Hub.

---

## Introduction

### The circRNA Vaccine Opportunity

Circular RNA (circRNA) is more stable than linear mRNA for vaccine cargo—think of it as RNA that learned to hold its own ends together rather than fraying like a cheap rope. Wesselhoeft et al. (2018) demonstrated circRNA half-lives of 8-24 hours versus linear mRNA's 2-4 hours, with sustained protein expression over multiple days. The back-splice junction (BSJ) covalently links the 3' and 5' ends, eliminating exonuclease degradation pathways that limit linear RNA. This stability advantage translates to reduced dosing frequency and potentially lower manufacturing costs for therapeutic applications.

Triple-negative breast cancer (TNBC) presents a compelling vaccine target. Jiang et al. (2019) identified four molecular subtypes (BLIS, BLIA, IM, LAR) with distinct immune microenvironments. IM (Immunomodulatory) tumors exhibit high TIL density (0.50-0.70), PD-L1 expression (0.40-0.60), and checkpoint inhibitor responsiveness. BLIS (Basal-like Immune Suppressed) shows the worst prognosis with TIL <0.15 and early immune escape. This subtype heterogeneity suggests that vaccine design should be subtype-adaptive rather than uniform.

### The Computational Gap

No existing platform links circRNA design to TNBC subtype-specific simulation. Current tools address components independently: ViennaRNA predicts secondary structure, PK-Sim models pharmacokinetics, PhysiCell simulates tumor dynamics. Integration is manual, and circRNA-specific features are not captured in existing frameworks.

Three gaps matter. circRNA pharmacokinetics differ from linear mRNA: LNP encapsulation creates tissue-specific biodistribution (liver 80%, spleen 10%), endosomal escape is a bottleneck at 1-4% efficiency (Gilleron et al., 2013)—meaning over 96% of your expensive therapeutic never reaches its destination—and circRNA degradation follows exonuclease-resistant pathways. Standard PK models omit these. circRNA innate immune sensing also differs: circRNAs lack 5' termini, so RIG-I 5'-ppp sensing does not apply (Hornung et al., 2006); instead, immunogenicity arises from dsRNA backbone structures sensed by MDA5 (Chen et al., 2019; Peisley and Hur, 2013) and modulated by intron identity. Existing tools assume linear RNA sensing. Finally, no deep learning architecture handles circRNA's S¹ topology, where position i and i+L are the same location.

Confluencia 3.0 fills these gaps through an EventBus architecture that couples: (1) TNBC Simulacrum with spatial TME simulation and subclonal evolution, (2) CirculaPK pharmacokinetics with circRNA-specific compartments, (3) circRNA-specific immunogenicity scoring with pathway-resolved sensing, and (4) RL-ABM closed-loop sequence evolution. We further propose TorusFold as a theoretical structure architecture.

### The Structure Prediction Challenge

Current circRNA structure prediction relies on thermodynamic models (ViennaRNA circ mode) that correctly handle circular topology through dynamic programming. However, thermodynamic models do not directly support structure-informed downstream tasks such as predicting BSJ-flanking region stability, estimating IRES accessibility in 3D context, or modeling ribosome binding site exposure.

Deep learning approaches (AlphaFold, ESM) revolutionized protein structure prediction but are not designed for circRNA's S¹ topology. Standard positional encoding PE(i) ≠ PE(i+L) for circRNA lengths, breaking periodicity at the BSJ—a problem that would not exist if circRNA could politely inform the transformer that it is, in fact, circular. We propose TorusFold as a theoretical architecture that natively models circular topology. Its validation awaits circRNA 3D structure data, which we propose as circRNA-CASP, analogous to CASP's role in validating protein structure predictors.

### The Accessibility, Extensibility, and Data Sharing Problem

Most computational biology tools require programming expertise, limiting adoption by experimental biologists who generate the data these tools need. circRNA researchers are often molecular biologists, not software engineers. Confluencia 3.0 addresses this through multiple interfaces: Python API, Streamlit web UI, CLI, R package, and PyQt6 desktop IDE with natural-language query capability for non-programming users.

Beyond accessibility, a critical challenge is longevity: algorithms become outdated as methods emerge. Confluencia 3.0 addresses this through an EventBus architecture that decouples modules via event-driven communication. Each component subscribes to events and emits results independently; new algorithms can replace existing implementations without modifying other modules. This ensures the platform remains current as methods evolve rather than becoming obsolete when individual algorithms are superseded.

The small-sample problem is endemic to circRNA computational work. We introduce Confluencia Hub, a federated model and data sharing system where users upload trained model bundles, not raw data. Privacy is preserved: no SMILES or nucleotide sequences are logged; data contributors can strip statistical traces (env_medians) before upload. Ethics-gated uploads require data source declaration (DOI or IRB number) and dual-use screening, directly addressing the small-sample problem through collaborative aggregation.

---

## Methods

### Module 1: TNBC Simulacrum

**Parameterization.** Jiang et al. (2019) classified 360 TNBC tumors into four subtypes via RNA-seq and immune profiling (Supplementary Table S2). Subtype-specific parameters:

- **BLIS** (n=108): TIL 0.08-0.15 [mean 0.12], worst prognosis, BRCA1-associated, early immune escape by cycle 12
- **BLIA** (n=72): TIL 0.25-0.40 [mean 0.33], immune gene signatures (STAT1, CXCL10), immune-activated profile
- **IM** (n=85): TIL 0.50-0.70 [mean 0.60], PD-L1 0.40-0.60 [mean 0.50], checkpoint inhibitor responsive, sustains immunoediting equilibrium
- **LAR** (n=95): AR expression 0.70-0.85, anti-androgen sensitivity (enzalutamide trial responder subset)

**Tumor Dynamics.** ODE system modeling tumor-immune interactions:

$$\frac{dT}{dt} = r_T \cdot T \cdot \left(1 - \frac{T}{K}\right) - d_T \cdot TIL \cdot T$$

$$\frac{dTIL}{dt} = r_{TIL} \cdot \left(\frac{T}{K}\right) - d_{TIL} \cdot T$$

$$\frac{dP}{dt} = k_{cp} \cdot circRNA - d_P \cdot P$$

where the first equation describes tumor growth with immune killing, the second describes TIL recruitment and exhaustion, and the third describes circRNA protein expression.

**Important caveat.** These dynamics recapitulate input assumptions. IM is parameterized with higher TIL (0.50-0.70) than BLIS (0.08-0.15); the finding that IM responds better to immunotherapy is a direct consequence of this parameterization, not a novel prediction. The simulation validates internal consistency, not external predictions.

Three immunoediting phases: elimination (immune surveillance dominates), equilibrium (tumor-immune balance), escape (immune suppression). Treatment arms: chemotherapy (30 cycles), immunotherapy (PD-1 blockade), circRNA vaccine (IRES-mediated antigen expression).

![Figure 1: TNBC simulation results across four molecular subtypes. (A) Tumor-immune dynamics showing immunoediting phases. (B) Shannon diversity evolution under chemotherapy pressure. (C) TME classification and spatial compartment effects. (D) Subclonal evolution with drug-induced genomic instability.](fig3_tnbc_simulation.png)

**Subclonal Evolution with Drug-Induced Instability.** Tumor heterogeneity is tracked via Shannon diversity $H = -\sum p_i \log(p_i)$ across subclones, acknowledging that this discrete approximation simplifies the continuous reality of tumor heterogeneity. Drug pressure induces genomic instability: mutation rate increases from baseline 1%/step to 50%/step under treatment, modeling the clinical reality of accelerated mutagenesis under chemotherapy (Ding et al., 2012). Resistant clones acquire selection advantage proportional to drug concentration, capturing the emergence of drug resistance that static tumor models cannot predict. Epigenetic adaptation occurs at 2%/step, modeling non-genetic resistance mechanisms. The EventBus emits DRUG_RESISTANCE_EMERGED events when Shannon diversity exceeds threshold, triggering treatment adaptation.

**Spatial TME Simulation.** The tumor microenvironment is modeled across three spatial compartments with distinct properties:

| Compartment | Oxygen | Drug Penetration | Immune Cell Density | Key Features |
|-------------|--------|-----------------|--------------------|--------------|
| Hypoxic core | <2% | Low (20-40%) | Sparse | HIF-1α active, immunosuppressive |
| Immune-rich margin | 5-10% | Moderate (60-80%) | High | Active immune surveillance |
| Stromal barrier | 2-5% | Variable (40-70%) | Moderate | CAF-mediated exclusion |

Nine immune cell populations are modeled with compartment-specific dynamics: CD8+ T cells, CD4+ T cells, Treg, B cells, NK cells, M1 macrophages, M2 macrophages, MDSCs, and CAFs. Six cytokines mediate intercellular communication: IFN-gamma (anti-tumor), TNF-alpha (pro-inflammatory), IL-2 (T cell proliferation), IL-6 (pro-tumor inflammatory), IL-10 (immunosuppressive), TGF-beta (immune exclusion). PD-1/PD-L1 checkpoint dynamics are modeled with binding kinetics and exhaustion markers.

TME classification follows four categories: hot (high CD8+ and IFN-gamma), cold (low immune infiltration), excluded (immune cells at margin, blocked by stroma), mixed (spatially heterogeneous). This classification directly informs immunotherapy response prediction: hot TME responds to checkpoint blockade, excluded TME requires stroma disruption, cold TME needs immune activation strategies.

**Four-Gene Signature Scoring.** TNBC therapeutic response prediction via four target protein expressions: TROP2 (TACSTD2, ADC target), NECTIN4 (PVRL4, metastasis marker), LIV-1 (SLC39A8, EMT marker), B7-H4 (VTCN1, immunotherapy target). The encoder computes 19-dimensional feature vectors: raw/normalized expressions, high/low binary flags, combined signature, proliferation score (TROP2+NECTIN4), immune score (B7-H4+TROP2), metastasis score (NECTIN4+LIV-1), efficacy score, and expression heterogeneity/balance metrics. Literature-derived correlations: TROP2-proliferation 0.72, NECTIN4-metastasis 0.65, LIV1-EMT 0.68, B7H4-immune 0.62. Therapeutic sensitivity: TROP2-high → ADC response 0.85, circRNA efficacy 0.78; B7H4-high → immuno_response 0.82.

**Bio-Mimetic Drug Architecture.** The drug ADMET prediction incorporates four brain-inspired components: (1) Topology Pharmacophore Network, representing molecules as scale-free graphs of pharmacophore nodes (HBD, HBA, hydrophobic, aromatic) with degree centrality features; (2) Tissue-Specific Dynamic Attention, generating patient-specific gating weights based on physiological state (liver function, kidney function, inflammation, pH); (3) Adversarial Synaptic Pruning, combining Pareto optimization with competitive selection to eliminate poor molecule candidates; (4) Neuroplastic Closed-Loop, adjusting model structure when clinical feedback indicates prediction error >0.3, implementing three-tier adaptation: fine-tuning (small errors), weight reconfiguration (moderate), structural plasticity (large errors). This architecture enables tissue-specific ADMET modulation and clinical feedback integration without retraining.

### Module 2: CirculaPK Pharmacokinetics

**Compartment Structure.** Six compartments: Injection → LNP → Endosome → Cytoplasm → Protein → Clearance. This structure captures three circRNA-specific bottlenecks that linear mRNA PK models omit:

1. **LNP encapsulation** with tissue-specific biodistribution (liver 0.80, spleen 0.10 per Paunovska et al., 2018).
2. **Endosomal escape** at 1-4% efficiency (Gilleron et al., 2013; Hou et al., 2021). The escape rate $k_{ec} = 0.025$/h is derived from stochastic efficiency $\eta = 0.02-0.04$ (Gilleron 2013) converted to first-order kinetics.
3. **IRES-dependent translation** at 0.02-0.32/h depending on IRES sequence (e.g., EMCV IRES ~0.25/h, HIV-1 IRES ~0.08/h; Martinez-Salas 2018), creating sequence-dependent translation efficiency variability spanning an order of magnitude.

**Rate Constants.** Literature-derived, not fitted: $k_{ab}=0.80$/h (absorption), $k_{be}=0.025$/h (endosomal uptake), $k_{ec}=0.025$/h (escape), $k_{cp}=0.02-0.32$/h (translation, IRES-dependent), $k_{cd}=0.04-0.12$/h (degradation, modification-adjusted), $k_{pc}=0.10-0.20$/h (protein clearance). RK45 integration outputs AUC, $C_{max}$, half-life.

**Modification Effects.** Nucleotide modifications alter degradation rate $k_{cd}$: unmodified circRNA $k_{cd} = 0.12$/h; m6A reduces to 0.06-0.08/h; Psi reduces to 0.04-0.06/h. These adjustments are derived from in vitro stability data and applied as multipliers to the base degradation rate.

### Module 3: circRNA-Specific Innate Immune Sensing

This module implements the first immunogenicity scoring system that explicitly distinguishes circRNA from linear RNA innate immune activation mechanisms. The key biological insight is that circRNAs lack 5' termini, invalidating the canonical RIG-I 5'-ppp sensing pathway (Hornung et al., 2006).

**Important caveat on circRNA production method.** Chen et al. (2019) demonstrated that circRNA immunogenicity critically depends on production method: in vitro transcribed circRNAs with residual intron sequences are immunogenic, whereas highly purified circRNAs from certain production methods are not. Our model scores immunogenicity based on sequence features alone and assumes highly purified, intron-free circRNA. Predictions for differently-produced circRNAs (e.g., those with residual intron sequences) should be interpreted with caution.

**Pathway Overview.** We model four sensing pathways with literature-derived weights. The pathway labels, mechanisms, and confidence levels are:

| Pathway | Sensor | circRNA Sensing Mechanism | Literature Basis | Confidence |
|---------|--------|---------------------------|------------------|------------|
| MDA5/dsRNA | MDA5 | Long dsRNA structures (>16 bp), inverted repeats | Chen 2019, Peisley and Hur 2013 | Medium |
| TLR7 | TLR7 | GU-rich ssRNA motifs in endosome | Gilleron 2013 | Medium |
| TLR8 | TLR8 | AU-rich ssRNA motifs with uridine preference | Gilleron 2013 | Medium |
| PKR | PKR | dsRNA length >33 bp | Nallagatla 2007 | High |

![Figure 2: Immunogenicity model validation. (A) Correlation with Chen 2019 IFN-β measurements (Spearman r=0.91, N=7). (B) HEK293 validation (r=0.68 [CI 0.26-0.88], N=15). (C) Pathway contribution analysis showing MDA5/dsRNA dominates discrimination. (D) GC confound analysis: pathway model vs. GC-only baseline (ΔAIC = -8.2, p=0.004).](fig5_immunogenicity_correlation.png)

**MDA5/dsRNA Pathway (weight 0.35).** circRNAs lack 5' termini, making RIG-I 5'-ppp sensing inapplicable. Instead, circRNA immunogenicity arises primarily from dsRNA backbone structures sensed by MDA5 (Peisley and Hur, 2013). Chen et al. (2019) demonstrated that circRNA immunogenicity correlates with dsRNA content and intron identity, not terminal features. The scoring identifies inverted repeat Alu elements and extended stem structures (>16 bp) that form dsRNA backbones; this threshold identifies potential MDA5 ligands, though activation strength scales cooperatively with dsRNA length (Peisley and Hur, 2013). Signaling proceeds through MAVS → IRF3/7 → IFN-β.

**TLR7/TLR8 Pathways (0.20/0.15).** These endosomal sensors dominate for LNP formulations (>96% endosomal residence per Gilleron et al., 2013). We score TLR7 and TLR8 separately with distinct motif preferences:
- TLR7: GU-rich motifs (5'-GUGU-3', 5'-GUCC-3') in single-stranded regions
- TLR8: AU-rich motifs (5'-AU-3', 5'-UUAU-3') with uridine preference

A circRNA-specific circularity correction factor (0.70, estimated) adjusts TLR scores downward, reflecting the reduced accessibility of circRNA sequences within LNP formulations. This parameter is heuristic and requires experimental validation.

**PKR Pathway (0.30).** PKR activation requires dsRNA length >33 bp (Nallagatla et al., 2007). circRNA circularity does not affect PKR activation (no termini requirement). The scoring counts dsRNA regions exceeding the 33 bp threshold.

**Differential m6A Suppression (Estimated Parameters).** m6A modification suppresses immune activation with pathway-specific intensity. These values are estimated from mechanistic reasoning, not directly measured in circRNA systems:

| Pathway | Estimated m6A Suppression | Mechanistic Rationale | Evidence Level |
|---------|--------------------------|----------------------|----------------|
| MDA5/dsRNA | ~90% | m6A destabilizes dsRNA structures, reducing MDA5 ligand availability | Indirect (Chen 2019) |
| TLR7/8 | ~30% | Endosomal sensors less affected by internal modifications | Estimated, no direct data |
| PKR | ~20% | PKR responds primarily to dsRNA length; m6A may modestly reduce effective duplex length | Estimated, no direct data |

These pathway-specific suppression values correct the oversimplified "m6A reduces immunogenicity" assumption but require experimental validation. Sensitivity analysis (±50% variation on suppression values) changes immunogenicity rank order for 2/15 test sequences, indicating moderate robustness to these parameter estimates.

**Bidirectional m6A Modeling.** m6A immunomodulation is modeled as a balance between evasion_weight (immune suppression via dsRNA destabilization) and enhancement_weight (potential immune potentiation via translation upregulation and increased antigen expression). The enhancement_weight component is hypothetical, motivated by m6A's known role in enhancing IRES-dependent translation (Yang et al., 2018), but lacks direct experimental validation in circRNA immune contexts. In highly structured regions, evasion dominates; in IRES-proximal regions, enhancement may prevail.

**Sensitivity Analysis.** ±50% weight variation preserves rank order for 12/15 sequences. Ablation: removing PKR (redistributing 0.30) changes 2/15 ranks; removing MDA5/dsRNA changes 3/15, so MDA5/dsRNA contributes most to discrimination.

**GC Confound Analysis.** GC-immunogenicity correlation r=0.85 (N=50 circBase, Spearman), reflecting GC's role in promoting dsRNA structure. Partial correlation controlling for GC: pathway scores retain r=0.42 with IFN-β (p=0.03, computed on N=50 circBase sequences with matched IFN-β measurements from literature). A simple GC-only baseline model achieves Spearman r=0.79 (N=50) with IFN-β; the pathway decomposition model achieves r=0.85 (ΔAIC = -8.2 relative to GC-only model), indicating pathway scoring provides statistically significant improvement over GC content alone (likelihood ratio test, p=0.004). The differential m6A suppression model contributes to this improvement: uniform m6A suppression reduces partial correlation to r=0.31 (p=0.08), suggesting pathway-resolved m6A modeling provides non-redundant information.

### Module 4: RL-ABM Closed-Loop Sequence Evolution

**Problem Formulation.** We formulate circRNA sequence optimization as a Gym-style reinforcement learning environment where the state space encompasses sequence features (GC content, IRES context, modification status) and patient profile (TNBC subtype, TME classification, gene signature scores). The action space includes four BSJ-protected operators: (1) point mutation (preserving BSJ sequence integrity), (2) IRES insertion, (3) nucleotide modification selection (m6A, Psi, 5mC), and (4) combination therapy adjustment.

**Multi-Objective Reward Function.** The reward integrates four biological objectives:

$$R = 0.35 \cdot \text{efficacy} + 0.30 \cdot \text{immune\_score} + 0.20 \cdot \text{safety} + 0.15 \cdot \text{synergy}$$

where efficacy = CirculaPK-predicted protein expression normalized by AUC, immune_score = Module 3 immunogenicity, safety = ADMET toxicity prediction with risk gate penalty (threshold 0.70), and synergy = multi-drug combination Bliss-CI score. These weights are heuristic and application-dependent; they prioritize efficacy and immune response for vaccine design but can be adjusted for protein replacement therapy (where immune evasion is desired). Sensitivity analysis: varying weights by ±0.10 changes the Pareto front composition but preserves 8/10 top-ranked sequences across weight configurations, indicating moderate robustness.

**ABM Reward vs. TME-Enhanced Reward.** Two reward functions are available. The ABM reward uses the agent-based immune simulation (9 cell populations, 6 cytokines) to compute treatment response. The TME-enhanced reward incorporates spatial compartment effects. The TME-enhanced reward is used by default for TNBC applications.

**Convergence.** REINFORCE with 500 episodes; reward plateaus by ~400 episodes in pilot runs (defined as <2% reward change over 50 episodes). Across 5 random seeds, convergence episode ranges from 350-450, with final reward variance σ=0.03. Pareto front balances stability, translation efficiency, and immune properties. The EventBus coordinates modules via deterministic event ordering with 18+ event types across eight subsystems.

**Multi-Drug Synergy Analysis.** Four synergy models are implemented: Bliss independence, Loewe additivity, Highest Single Agent (HSA), and Chou-Talalay Combination Index (CI). A Bliss-CI discrepancy interpretation matrix identifies effect-dose mismatches: when Bliss>0 but CI>2, the combination shows effect synergy but dose mismatch requiring optimization. This is particularly relevant for immunotherapy combinations where Bliss independence assumptions fail. L-BFGS-B dose optimization searches for optimal concentration ratios.

---

## Results

### TNBC Simulation

IM subtype sustains immunoediting equilibrium (TIL >0.50) across 30 simulation cycles. BLIS escapes by cycle 12 (TIL <0.05). Shannon diversity increases from 0.4 to 1.2 under chemotherapy (30 cycles). TME classification correlates with treatment response: hot TME (IM subtype) shows 2.3x higher simulated response than cold TME (BLIS). Stromal barrier compartment reduces drug penetration to 40-70%, partially explaining TME-excluded tumors' resistance.

**Parameter-swap validation.** To test whether simulation results are circular, we swapped BLIS and IM parameters: BLIS initialized with IM parameters (TIL 0.50-0.70) now sustains immunoediting equilibrium, while IM initialized with BLIS parameters (TIL 0.08-0.15) escapes by cycle 12. This confirms that simulation outcomes are determined by input parameters rather than model-specific dynamics, validating internal consistency but not external predictive utility.

### Pharmacokinetics

Simulated half-lives match Wesselhoeft et al. (2018) experimental values with 12% relative error [CI 3-21%] (N=4 constructs). m6A modification extends half-life to ~15-22 h; Psi to ~20-30 h. The six-compartment model captures the endosomal escape bottleneck: only 2-4% of injected circRNA reaches the cytoplasm.

**Model comparison.** With N=4, statistical model comparison is underpowered. Six-compartment AIC = 18.2 vs. two-compartment AIC = 22.7 (ΔAIC = 4.5 favoring six-compartment), but this difference is not significant at N=4. A minimum of ~12 constructs would be needed to distinguish models at α=0.05, power=0.80.

### Immunogenicity

**Primary benchmark.** Chen et al. (2019) preliminary correlation: r=0.91 (Spearman, N=7 circRNA sequences with published IFN-β). Leave-one-out analysis: median LOO r=0.87 [IQR 0.82-0.91, range 0.79-0.94]. Direction consistent but magnitude sensitive to individual points. With N=7, standard error of r ≈ 0.18, and statistical power to distinguish r=0.91 from r=0.50 at α=0.05 is approximately 0.35.

**Secondary validation.** HEK293 experimental data (N=15, independent from Chen 2019): r=0.68 [CI 0.26-0.88]. The CI width (0.62) is insufficient to distinguish from moderate or strong correlation.

**GC baseline comparison.** Simple GC-only model: r=0.79 (N=50 circBase); pathway decomposition model: r=0.85 (ΔAIC = -8.2, p=0.004). Pathway scoring provides statistically significant but modest improvement over GC content alone.

### Subclonal Evolution

Under chemotherapy (30 cycles), Shannon diversity increased from 0.4 to 1.2, with dominant clone frequency decreasing from 0.85 to 0.42. Drug-induced mutation rate increase (1% → 50%) produced 3-5 resistant subclones per simulation run, compared to 0-1 without treatment pressure.

### Wet-Lab Validation (Pending)

Experimental validation is underway with collaborating medical school researchers. Planned experiments: (1) IFN-β ELISA for 15 evolved circRNA sequences in HEK293 cells, (2) circRNA half-life measurement via qRT-PCR in primary TNBC cell lines (MDA-MB-231, HCC1937), (3) subtype-specific tumor response in BLIS and IM PDX models (n=6 per group). Results will be reported in a follow-up publication.

---

## Discussion

### What Confluencia 3.0 Contributes

We organize contributions by evidence level to distinguish implemented features from validated capabilities:

**(A) Implemented features with preliminary correlation support:**

1. **circRNA-specific immune sensing via MDA5/dsRNA pathway.** The immunogenicity model correctly identifies that circRNA lacks 5'-ppp termini, making RIG-I sensing inapplicable, and instead models dsRNA backbone activation via MDA5. TLR7/TLR8 are scored separately with distinct motif preferences. Pathway decomposition provides statistically significant improvement over GC-only baseline (ΔAIC = -8.2, p=0.004, N=50).

2. **Six-compartment circRNA PK.** Captures endosomal escape bottleneck (1-4% efficiency) that linear mRNA models omit. Preliminary accuracy: 12% error vs Wesselhoeft 2018 (N=4), but N=4 cannot statistically distinguish from simpler models.

3. **Differential m6A suppression modeling.** Pathway-specific m6A suppression values (MDA5 ~90%, TLR ~30%, PKR ~20%) are estimated parameters that correct the oversimplified "m6A reduces immunogenicity" assumption. Uniform m6A suppression reduces partial correlation from r=0.42 to r=0.31, demonstrating non-redundant information. However, these values require experimental validation.

**(B) Implemented features requiring validation:**

4. **Spatial TME simulation.** Nine immune cell populations, six cytokines, PD-1/PD-L1 checkpoint dynamics, and three spatial compartments. This is an implementation of Jiang 2019 subtype stratification with spatial extension, not a validated predictive system.

5. **Subclonal evolution with drug-induced instability.** Tumor heterogeneity tracked via Shannon diversity with drug-induced mutation rate increase. Models the clinical reality of resistance emergence rather than assuming static tumor populations. No TCGA/METABRIC validation performed.

6. **RL-ABM closed-loop optimization.** circRNA sequence optimization with ABM-computed reward. Reward weights are heuristic; Pareto front moderately robust to weight variation. Optimizer converges on simulator reward surface, not validated biological optima.

7. **Bio-mimetic drug architecture.** Four brain-inspired components enable patient-specific ADMET modulation and clinical feedback integration. Tissue-specific attention weights are rule-based, not learned; neuroplastic closed-loop requires clinical feedback data not yet available.

8. **Bidirectional m6A modeling.** Models m6A's dual role (evasion via dsRNA destabilization vs. enhancement via translation upregulation), reflecting context-dependent function. The enhancement component is hypothetical and requires experimental validation.

**(C) Verified mathematical properties (biological utility unverified):**

9. **TorusFold architecture.** Mathematical formulation for circRNA's S¹ topology: TPE periodicity (verified: |TPE(i) - TPE(i+L)| < 10⁻⁶), circular distance metric, and rotation equivariance (verified). These are design properties of the architecture, not contributions requiring biological validation. The pair prediction head is non-functional (~0% predictions). Possible causes include insufficient training (1 epoch), additive pair initialization (vs AlphaFold2's outer product), or fundamental architectural limitation. Without circRNA 3D structure training data, we cannot distinguish implementation bug from architectural flaw. A physics-based structure head provides zero-training 3D prediction via constraint solving as a fallback when no training data is available.

![Figure 3: TorusFold architecture for circRNA 3D structure prediction. (A) Torus Positional Encoding (TPE) on S¹ topology with guaranteed periodicity. (B) Circular distance metric accounting for BSJ continuity. (C) CircPairformer rotation-equivariant pair representation. (D) Verified mathematical properties: TPE periodicity |TPE(i) - TPE(i+L)| < 10⁻⁶, rotation equivariance confirmed.](fig6_torusfold_architecture.png)

**(D) Software and community infrastructure:**

10. **Integrated event-driven framework with algorithmic extensibility.** EventBus with 18+ event types couples eight subsystems, enabling modular extension. New algorithms can replace existing implementations by subscribing to the same events, ensuring the platform remains current as methods evolve rather than becoming obsolete when individual algorithms are superseded.

11. **Accessibility design.** Five interfaces (Python API, Streamlit web, CLI, R package, PyQt6 desktop IDE with natural-language query) target diverse user communities from molecular biologists to software developers.

12. **Federated data sharing.** Confluencia Hub with ethics-gated uploads, data source declaration, dual-use screening, and SHA256 hash verification.

### Integration Ecosystem

The EventBus architecture enables integration beyond the core four modules. The platform defaults to established external tools via lazy-loading bridges: (1) structure prediction uses ViennaRNA (thermodynamic) with TorusFold as theoretical extension; (2) PK simulation uses literature-derived compartment models but can swap to PK-Sim or physiologically-based PK via EventBus subscription; (3) MHC binding uses NetMHCpan (AUC ~0.90) for standalone screening, with Confluencia 2.0's epitope module (AUC=0.80) available when integration with circRNA PK simulation is required. The integrated epitope module supports joint vaccine efficacy prediction via environment variables (dose, frequency, circRNA expression, IFN score) and gradient-based sensitivity analysis for wet-lab optimization, capabilities not available in standalone tools. Confluencia's custom modules serve integration needs rather than competing with established benchmarks. New methods (e.g., future circRNA-specific structure predictors) can replace existing implementations by subscribing to the same events without modifying other subsystems.

### What Remains Unvalidated

1. **TNBC simulation.** No TCGA/METABRIC validation performed. Parameter-swap experiment confirms circular dependency: outcomes determined by input parameters.

2. **Immunogenicity weights.** Literature-derived, not empirically calibrated. 20% rank inversion under weight perturbation. N=7 primary benchmark provides statistical power ≈0.35.

3. **TorusFold.** Non-functional pair head. Validation awaits circRNA 3D structure data.

4. **RL-ABM reward.** Optimizer converges on simulator reward surface, not validated biological optima.

### The circRNA Data Challenge

No circRNA crystal structures or cryo-EM reconstructions exist in PDB. Fewer than two dozen circRNA structural annotations are available from literature. This is the fundamental barrier for TorusFold validation: one cannot validate a structure predictor without structures, a circular problem that even our circular architecture cannot solve. We propose circRNA-CASP as a community mechanism, analogous to CASP's role in protein structure prediction. Until circRNA structure data becomes available, TorusFold remains a theoretical proposal with verified mathematical properties but unverified biological utility.

### Power Analysis

Current sample sizes are insufficient for definitive validation. Required sample sizes for key benchmarks:

| Module | Current N | Required N (α=0.05, power=0.80) | Current Power | Effect Size Assumption |
|--------|-----------|-------------------------------|---------------|----------------------|
| Immunogenicity (r=0.91 vs r=0.50) | 7 | ~25 | ~0.35 | Large (Cohen q=0.85) |
| PK (6-comp vs 2-comp) | 4 | ~12 | ~0.20 | ΔAIC=4.5 (medium) |

### The Federated Sharing Response

The small-sample problem (N=42 drug, N=7 immunogenicity) is endemic to circRNA work. Confluencia Hub enables federated model sharing: labs upload trained weights (not raw data), enabling community aggregation. Ethics-gated uploads require data source declaration and dual-use screening; `strip_env_medians` removes statistical traces; SHA256 verification mitigates code execution risks.

---

## Limitations

**TNBC.** ODE dynamics without stochastic effects; no TCGA/METABRIC validation; parameter-swap experiment confirms circular dependency; spatial TME model assumes fixed compartment boundaries; Shannon diversity discretizes continuous heterogeneity.

**Immunogenicity.** Pathway weights uncalibrated; 20% rank inversion under weight variation; N=7 primary benchmark fragile (power ≈0.35); N=15 HEK293 CI too wide to claim validation (width 0.62); GC confound partially resolved (partial r=0.42); m6A suppression values (90%/30%/20%) are estimated, not measured—sensitivity analysis shows 2/15 rank changes under ±50% variation; TLR circularity correction factor (0.70) is heuristic; bidirectional m6A enhancement_weight is hypothetical; model assumes highly purified circRNA and does not account for production method effects (intron retention, purification method) that Chen 2019 showed are critical.

**PK.** Literature priors only, not fitted to time-course data; N=4 validation cannot distinguish six-compartment from simpler models (ΔAIC=4.5 not significant at N=4); no patient-specific PK; $k_{ec}$ derivation from stochastic efficiency to first-order kinetics is approximate; modification effect on $k_{cd}$ is a linear multiplier.

**TorusFold.** Non-functional pair head (~0% predictions); no circRNA 3D structure training data (the field awaits someone to first solve this chicken-or-egg problem); cannot distinguish bug from architectural flaw; additive pair initialization may contribute to failure; H=16 harmonics choice is unvalidated.

**Evolution.** REINFORCE optimizes heuristic landscape, not validated biological optima; 500 episodes with convergence at 350-450 across seeds; reward weights (0.35/0.30/0.20/0.15) are heuristic; Bliss-CI discrepancy matrix is descriptive, not prescriptive.

**All claims are hypothesis-generation.** Simulated outcomes validate internal consistency, not external predictions. Wet-lab validation ongoing with collaborating medical school.

---

## Data Availability Statement

TNBC subtype parameters derived from Jiang et al. (2019) Supplementary Table S2 (publicly available). Pharmacokinetic validation uses Wesselhoeft et al. (2018) published half-life data. Immunogenicity validation uses Chen et al. (2019) published IFN-β measurements. circRNA 3D structure data: not available (no public database exists).

**Wet-lab validation (in progress).** We are collaborating with medical school researchers to generate experimental data: (1) IFN-β ELISA for 15 evolved circRNA sequences in HEK293 cells, (2) half-life quantification via qRT-PCR in MDA-MB-231 and HCC1937 TNBC cell lines, (3) subtype-specific response in BLIS and IM PDX models (n=6 per group). All protocols approved by institutional IRB. Results will be reported in follow-up publication within 6 months of this submission.

---

## Code Availability

**Repository.** github.com/RomanCohort/confluencia (MIT License). Python 3.10+, pytest 87% coverage, CI/CD via GitHub Actions. Documentation and installation instructions available in repository README. A DOI-linked archive is available at [Zenodo DOI to be added upon acceptance].

**Interfaces.** Five access modes for different user profiles:
- **Python API:** `import confluencia_3_0; confluencia_3_0.simulate(config)`
- **Streamlit web UI:** `streamlit run confluencia-studio/streamlit_app/Home.py` (6 pages: CircRNA Analysis, Drug Prediction, Epitope Screening, TNBC Simulator, Joint Analysis, Report Export)
- **CLI:** `confluencia simulate --subtype IM --steps 100`
- **R package:** Available from github.com/RomanCohort/confluencia-rpkg; functions `cf_drug_predict()`, `cf_hub_push_model()`
- **Desktop IDE:** `confluencia-studio` PyQt6 application with editor, notebook, variable explorer, and git integration

**Hub.** Federated model sharing API. Upload: `hub.push_model("bundle.joblib", strip_env_medians=True)`. Download: `hub.pull_model("hub:drug:user:v1")`. Ethics-gated uploads require `data_source_type`, `data_source_reference`, and `dual_use_declaration`. R bindings: `cf_hub_push_model()`, `cf_hub_pull_model()`, `cf_hub_list_models()`.

**One-command installation.** `pip install confluencia` installs all dependencies except optional deep learning backends (ESM, PyTorch GPU). Full setup: `pip install confluencia[all]`.

---

## Acknowledgments

We thank [collaborating medical school researchers] for ongoing wet-lab validation support. We thank reviewers for constructive feedback on manuscript revisions.

---

## References

1. Wesselhoeft RA, et al. RNA circularization diminishes immunogenicity and can extend translation duration in vivo. Nat Commun. 2018;9:2629.

2. Jiang YZ, et al. Genomic and Transcriptomic Landscape of Triple-Negative Breast Cancer. Cancer Cell. 2019;35:428.

3. Chen YG, et al. Sensing Self and Foreign Circular RNAs by Intron Identity. Mol Cell. 2019;73:422.

4. Gilleron J, et al. Image-based analysis of lipid nanoparticle-mediated siRNA delivery, intracellular trafficking and endosomal escape. Nat Biotechnol. 2013;31:638.

5. Hou X, et al. Lipid nanoparticles for mRNA delivery. Nat Rev Mater. 2021;6:1078.

6. Paunovska K, et al. Quantification of nanoprotein distribution at the single-cell level. ACS Nano. 2018;12:7580.

7. Nallagatla SR, et al. 5'-terminal oligonucleotide determines RNA duplex structure and PKR activation efficiency. Science. 2007;318:1455.

8. Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596:583.

9. Vaswani A, et al. Attention is all you need. NeurIPS. 2017.

10. Lorenz R, et al. ViennaRNA Package 2.0. Algorithms Mol Biol. 2011;6:26.

11. Cohen J, Welling M. Group equivariant CNNs. ICML. 2016.

12. Romero DW, et al. Equivariant Transformers. ICLR. 2022.

13. Hornung V, et al. 5'-Triphosphate RNA is the ligand for RIG-I. Science. 2006;314:994.

14. Ding L, et al. Clonal evolution in relapsed acute myeloid leukaemia revealed by whole-genome sequencing. Nature. 2012;481:506.

15. Peisley A, Hur S. Multi-level regulation of cellular recognition of viral dsRNA. Cell Mol Life Sci. 2013;70:1949.

16. Martinez-Salas E, et al. IRES mechanisms: connecting structure and function. Trends Microbiol. 2018;26:651.

17. Yang Y, et al. Extensive translation of circular RNAs driven by N6-methyladenosine. Cell Res. 2018;28:743.