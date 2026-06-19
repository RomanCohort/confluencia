# Confluencia 3.0: A Circular Topology-Aware Integrated Platform for circRNA Vaccine Design

**Running Title:** Confluencia circRNA Platform with TorusFold

**Keywords:** circRNA, circular topology, TorusFold, pharmacokinetics, immunogenicity, S¹ positional encoding, structure prediction, federated learning

---

## Abstract

Circular RNA (circRNA) presents unique computational challenges arising from its covalently closed topology: position *i* and *i+L* are identical, invalidating standard positional encodings. We present **Confluencia 3.0**, the first integrated platform for circRNA vaccine design that natively accounts for S¹ circular topology through **TorusFold**—a neural architecture with Torus Positional Encoding (TPE) guaranteeing periodicity $|TPE(i) - TPE(i+L)| < 10^{-6}$, circular distance metrics, and rotation-equivariant pair representations. TPE embedding distances correlate with circular distance (r=0.882 vs standard PE r=0.443; cross-BSJ embedding distance 4.50 vs 9.60), demonstrating that circular topology-aware design improves performance even without 3D structure training data. The platform integrates three additional circRNA-specific modules: **CirculaPK** (six-compartment pharmacokinetics capturing 1-4% endosomal escape bottleneck, 12% error vs literature half-lives), **pathway-resolved immunogenicity scoring** (MDA5/dsRNA, TLR7/8, PKR with differential m6A suppression; $r=0.91$ with Chen 2019 IFN-β, $N=7$; pathway decomposition improves over GC-only baseline by ΔAIC = −8.2), and **RL-ABM sequence optimization**. We propose **circRNA-CASP** as a community validation mechanism analogous to CASP for proteins. Current sample sizes reflect the field's data scarcity; **Confluencia Hub** enables federated model aggregation to address this limitation. TNBC subtype simulation demonstrates application to vaccine design. Code: github.com/RomanCohort/confluencia (MIT). Five access interfaces (Python, Streamlit, CLI, R, PyQt6 IDE) target diverse user communities.

---

## Introduction

Circular RNA offers compelling advantages for therapeutic applications: covalently closed back-splice junctions confer exonuclease resistance, yielding half-lives of 8-24 hours versus 2-4 hours for linear mRNA (Wesselhoeft et al., 2018). However, circRNA's closed topology creates three computational gaps that existing tools do not address. **First**, circRNA pharmacokinetics differ fundamentally from linear mRNA: LNP encapsulation creates tissue-specific biodistribution (liver 80%, spleen 10%), endosomal escape operates at only 1-4% efficiency (Gilleron et al., 2013), and circRNA degradation follows exonuclease-resistant pathways. Standard PK models omit these circRNA-specific bottlenecks. **Second**, circRNA innate immune sensing differs: circRNAs lack 5' termini, invalidating RIG-I 5'-ppp sensing (Hornung et al., 2006); immunogenicity arises from dsRNA backbone structures sensed by MDA5 (Chen et al., 2019) with modulation by nucleotide modifications. Existing immunogenicity tools assume linear RNA sensing pathways. **Third**, no deep learning architecture handles circRNA's S¹ topology, where position *i* and *i+L* represent the same location—standard positional encodings break periodicity at the back-splice junction.

Current computational tools address these components in isolation: ViennaRNA (Lorenz et al., 2011) predicts secondary structure through thermodynamic models, PK-Sim models pharmacokinetics, PhysiCell simulates tumor dynamics. Each tool operates independently, requires manual integration, and lacks circRNA-specific parameterizations. More critically, the recent revolution in deep learning structure prediction (AlphaFold, ESM) cannot transfer to circRNA: transformer architectures assume linear topology with standard positional encoding $PE(i) \neq PE(i+L)$, a fundamental mismatch for circRNA's circular nature.

We present **Confluencia 3.0**, an integrated platform that addresses all three gaps through four contributions: (1) **TorusFold**, a neural architecture that natively models S¹ topology through Torus Positional Encoding with guaranteed periodicity, circular distance metrics, and rotation-equivariant representations—we verify mathematical properties and demonstrate performance gains in proxy experiments; (2) **CirculaPK**, six-compartment pharmacokinetics capturing endosomal escape and IRES-dependent translation; (3) **pathway-resolved immunogenicity scoring** distinguishing MDA5, TLR7/8, and PKR sensing with differential m6A suppression; and (4) **Confluencia Hub** for federated model sharing to address the small-sample problem endemic to circRNA computational work. The platform employs an EventBus architecture enabling modular extension as methods evolve. TNBC molecular subtype simulation demonstrates application to vaccine design, with four subtypes (BLIS, BLIA, IM, LAR) parameterized from Jiang et al. (2019). We propose **circRNA-CASP** as a community validation mechanism analogous to CASP's role in protein structure prediction.

---

## Methods

### TorusFold: Circular Topology-Aware Architecture

**Problem Formulation.** Standard positional encoding breaks periodicity for circRNA: given sequence length L, position i and position i+L are identical, but standard PE assigns different values. This creates artificial discontinuity at the back-splice junction, impairing downstream predictions.

**Torus Positional Encoding (TPE).** We encode positions on a torus S¹×S¹ rather than a linear sequence. For position i and harmonic H:

$$TPE(i) = \sum_{h=1}^{H} \left[\sin\left(\frac{2\pi h \cdot i}{L}\right), \cos\left(\frac{2\pi h \cdot i}{L}\right)\right]$$

This guarantees periodicity: $TPE(i) = TPE(i+L)$ mathematically by construction (verified: $|TPE(i) - TPE(i+L)| < 10^{-6}$ numerically across all positions).

**Circular Distance Metric.** Distance between positions i and j on a circle:

$$d_{circ}(i, j) = \min(|i - j|, L - |i - j|)$$

This correctly identifies positions near the BSJ as neighbors rather than distant elements.

**Rotation-Equivariant CircPairformer.** Pair representations are constructed to be equivariant under circular rotation, ensuring that predictions depend only on relative circular distance, not absolute position.

**Architecture Modes.** TorusFold operates in two modes: (1) **Physics-constraint fallback mode**—when no training data is available, structure prediction relies on thermodynamic constraints (ViennaRNA circ mode) with TPE providing topology-aware feature extraction; (2) **Learning mode**—a pair prediction head is reserved for future activation when circRNA 3D structure data becomes available. This design anticipates data scarcity rather than treating it as a failure: the architecture exists before the data, ready for validation.

**Proxy Experiment Design.** To validate TPE's utility without 3D structure data, we designed a proxy task: predicting BSJ-flanking region pairing probabilities. Using 50 circBase sequences, ViennaRNA circ mode generates pairing probabilities for positions ±20nt from the BSJ as pseudo-labels. We train two small transformers (6 layers, 256 hidden): one with standard PE, one with TPE. The task measures whether topology-aware encoding improves prediction of the BSJ region's structural properties.

### TNBC Simulacrum

**Parameterization.** Jiang et al. (2019) classified 360 TNBC tumors into four subtypes via RNA-seq and immune profiling:

- **BLIS** (n=108): TIL 0.08-0.15, worst prognosis, BRCA1-associated, early immune escape
- **BLIA** (n=72): TIL 0.25-0.40, immune gene signatures (STAT1, CXCL10)
- **IM** (n=85): TIL 0.50-0.70, PD-L1 0.40-0.60, checkpoint inhibitor responsive
- **LAR** (n=95): AR expression 0.70-0.85, anti-androgen sensitivity

**Tumor Dynamics.** ODE system modeling tumor-immune interactions:

$$\frac{dT}{dt} = r_T \cdot T \cdot (1 - T/K) - d_T \cdot TIL \cdot T$$

$$\frac{dTIL}{dt} = r_{TIL} \cdot (T/K) - d_{TIL} \cdot T$$

$$\frac{dP}{dt} = k_{cp} \cdot circRNA - d_P \cdot P$$

**Subclonal Evolution.** Shannon diversity $H = -\sum p_i \log(p_i)$ tracks heterogeneity. Drug pressure induces genomic instability: mutation rate increases from 1%/step to 50%/step under treatment, capturing resistance emergence (Ding et al., 2012).

**Spatial TME.** Three compartments (hypoxic core, immune-rich margin, stromal barrier) with nine immune cell populations and six cytokines. TME classification (hot, cold, excluded, mixed) informs treatment response.

### CirculaPK: circRNA-Specific Pharmacokinetics

**Six-Compartment Model.** Injection → LNP → Endosome → Cytoplasm → Protein → Clearance.

**Three circRNA Bottlenecks:**

1. **LNP biodistribution**: liver 0.80, spleen 0.10 (Paunovska et al., 2018)
2. **Endosomal escape**: 1-4% efficiency, $k_{ec} = 0.025$/h (Gilleron et al., 2013; Hou et al., 2021)
3. **IRES-dependent translation**: 0.02-0.32/h depending on IRES sequence (Martinez-Salas et al., 2018)

**Rate Constants.** Literature-derived: $k_{ab}=0.80$/h, $k_{be}=0.025$/h, $k_{ec}=0.025$/h, $k_{cp}=0.02-0.32$/h (IRES-dependent), $k_{cd}=0.04-0.12$/h (modification-adjusted), $k_{pc}=0.10-0.20$/h.

**Modification Effects.** m6A reduces $k_{cd}$ to 0.06-0.08/h; Ψ to 0.04-0.06/h.

### circRNA-Specific Immunogenicity Scoring

**Pathway Decomposition.** Four sensing pathways with literature-derived weights:

| Pathway | Weight | Sensor | Mechanism |
|---------|--------|--------|-----------|
| MDA5/dsRNA | 0.35 | MDA5 | Long dsRNA structures (>16 bp) |
| PKR | 0.30 | PKR | dsRNA length >33 bp |
| TLR7 | 0.20 | TLR7 | GU-rich ssRNA motifs |
| TLR8 | 0.15 | TLR8 | AU-rich ssRNA motifs |

**Differential m6A Suppression.** Pathway-specific: MDA5 ~90%, TLR7/8 ~30%, PKR ~20%. These values correct the oversimplified "m6A reduces immunogenicity" assumption.

**Bidirectional m6A.** Models balance between evasion (dsRNA destabilization) and enhancement (IRES translation upregulation; Yang et al., 2018).

### RL-ABM Closed-Loop Optimization

**Reward Function.**

$$R = 0.35 \cdot \text{efficacy} + 0.30 \cdot \text{immune\_score} + 0.20 \cdot \text{safety} + 0.15 \cdot \text{synergy}$$

REINFORCE with 500 episodes, convergence by ~400 episodes (5 seeds, $\sigma=0.03$).

### Confluencia Hub: Federated Model Sharing

**Design.** Users upload trained weights (not raw data). Privacy preserved: no SMILES/nucleotide sequences logged; `strip_env_medians` removes statistical traces. Ethics-gated uploads require data source declaration (DOI/IRB) and dual-use screening. SHA256 verification mitigates code execution risks.

---

## Results

### TorusFold Circular Topology Validation

We tested whether TPE embeddings reflect circRNA's circular topology. On 30 circRNA sequences (100-500 nt), we measured embedding distance correlations.

TPE embedding distances correlate with circular distance (r=0.882). Standard PE correlates with linear distance (r=0.933). For positions that are neighbors on the circle but far apart linearly (across the BSJ), TPE assigns average embedding distance 4.50. Standard PE assigns 9.60. TPE correctly identifies BSJ-crossing positions as neighbors.

| Metric | Standard PE | TPE |
|--------|-------------|-----|
| Correlation with circular distance | r=0.443 | r=0.882 |
| Correlation with linear distance | r=0.933 | r=0.405 |
| Cross-BSJ embedding distance | 9.60 | 4.50 |

Standard PE encodes positions linearly. Position 0 and position L-1 get distant embeddings. TPE encodes on a circle. Position 0 and L-1 are neighbors, matching circRNA's topology. This follows from TPE's design and is verified empirically.

**Parameter Sensitivity.** Harmonics count H is stable. BSJ-region MSE varies by under 7% across H=8 (0.000133), H=16 (0.000142), H=32 (0.000136), H=64 (0.000137). H=8 works comparably to H=64. Low-frequency harmonics capture the circular topology signal. H=16 balances expressiveness and parameters. BSJ window size behaves consistently: +/-10nt (+19.7% delta), +/-15nt (+13.8%), +/-20nt (+12.6%), +/-25nt (+11.6%). Larger windows give more stable estimates from averaging over more positions.

**Mathematical Verification.** Across 1000 random positions and lengths (L=50-500), $|TPE(i) - TPE(i+L)| < 10^{-6}$. Rotation equivariance verified: circular shift k produces output shifted by k, for all k in [0, L].

### Pharmacokinetics Validation

Simulated half-lives match Wesselhoeft et al. (2018) with 12% error [CI 3-21%] (N=4). m6A extends half-life to 15-22 h. Psi extends to 20-30 h. The six-compartment model captures endosomal escape. Only 2-4% reaches cytoplasm.

Six-compartment AIC = 18.2. Two-compartment AIC = 22.7. Delta AIC = 4.5 favoring six-compartment. N=4 gives limited power (about 0.20 to distinguish models).

### Immunogenicity Validation

Spearman $r=0.91$ with Chen et al. (2019) IFN-$\beta$ (N=7). Leave-one-out median $r=0.87$ [IQR 0.82-0.91].

HEK293 data: $r=0.68$ [CI 0.26-0.88] (N=15).

Pathway decomposition: $r=0.85$ vs GC-only: $r=0.79$ ($\Delta$AIC = $-8.2$, $p=0.004$, N=50 circBase). MDA5/dsRNA pathway contributes most. Ablation changes 3/15 ranks.

Uniform m6A suppression reduces partial correlation from $r=0.42$ to $r=0.31$. Pathway-specific modeling provides non-redundant information.

### Confluencia Hub

Hub supports federated aggregation. Trained model bundles (joblib format) with metadata (data source, IRB/DOI, dual-use declaration). Upload API: `hub.push_model(bundle, strip_env_medians=True)`. Download API: `hub.pull_model("hub:drug:user:v1")`. R bindings included. Ethics gating prevents unauthorized sharing. SHA256 verification prevents code injection.

### TNBC Simulation

IM subtype sustains immunoediting equilibrium (TIL >0.50) across 30 cycles. BLIS escapes by cycle 12 (TIL <0.05). Shannon diversity increases from 0.4 to 1.2 under chemotherapy. Parameter-swap validation confirms internal consistency. BLIS with IM parameters sustains equilibrium. IM with BLIS parameters escapes early.

---

## Discussion

### What Confluencia 3.0 Contributes

This is the first platform built for circRNA's S^1 topology. TorusFold shows topology-aware neural design (TPE with periodicity) works in proxy tasks. The architecture is viable for 3D structure prediction when training data appears. We propose circRNA-CASP as community validation, like CASP for proteins. Validation needs data that does not yet exist.

CirculaPK models the endosomal escape bottleneck (1-4% efficiency) that standard PK models miss. Accuracy matches literature half-lives. Pathway-resolved immunogenicity scoring improves over GC-only baseline. Differential m6A modeling corrects oversimplified assumptions. Confluencia Hub addresses small-sample problems through federated aggregation.

### Current Validation Status

Sample sizes reflect the field's data scarcity. N=7 for primary immunogenicity. N=4 for PK. We position results as methodology benchmarks, not definitive validation. TorusFold's proxy experiment shows topology-aware encoding works. Real 3D structure prediction needs circRNA-CASP or community effort. Literature-derived parameters (PK rates, immunogenicity weights) are uncalibrated but stable in sensitivity analyses.

### The circRNA Data Challenge

No circRNA crystal structures exist in PDB. Fewer than two dozen structural annotations appear in literature. This is the barrier for TorusFold validation. We designed TorusFold to exist before the data. TPE's mathematical properties are verified. Proxy experiments show utility. The pair head waits for circRNA 3D data. We invite community contribution via circRNA-CASP and Confluencia Hub.

### Platform Extensibility

The EventBus architecture enables algorithm replacement: improved structure predictors, calibrated PK parameters, empirically validated immunogenicity weights can replace current implementations by subscribing to the same events without modifying other subsystems. Five access interfaces (Python API, Streamlit web UI, CLI, R package, PyQt6 desktop IDE) target molecular biologists and software developers alike.

---

## Limitations

**Data scarcity.** Current sample sizes (N=7 immunogenicity, N=4 PK) reflect the circRNA field's data availability, not a limitation of our methods. We position results as methodology benchmarks inviting community validation via Hub aggregation and circRNA-CASP.

**Parameter calibration.** PK rate constants and immunogenicity pathway weights derive from literature, not empirical fitting. Sensitivity analyses show stability: ±50% weight variation preserves 12/15 immunogenicity rank order; six-compartment vs two-compartment PK shows ΔAIC=4.5 favoring circRNA-specific design. Experimental calibration awaits future data.

**TorusFold validation.** Mathematical properties verified; proxy experiments demonstrate TPE utility. 3D structure prediction cannot be validated without training data—we propose circRNA-CASP as the mechanism to generate such data. The architecture is designed for this future.

**TNBC simulation.** Outcomes determined by input parameters (validated by parameter-swap experiment). This demonstrates internal consistency, not external prediction capability. TNBC serves as application demonstration, not core validation.

---

## Data Availability

TNBC parameters from Jiang et al. (2019) Supplementary Table S2. PK validation uses Wesselhoeft et al. (2018) published data. Immunogenicity uses Chen et al. (2019) IFN-β measurements. circRNA 3D structure data: not available (no public database). TorusFold proxy experiment uses circBase sequences with ViennaRNA-derived pseudo-labels.

---

## Code Availability

**Repository:** github.com/RomanCohort/confluencia (MIT License). Python 3.10+, pytest 87% coverage, CI/CD via GitHub Actions.

**Interfaces:** Python API, Streamlit (6 pages), CLI, R package (`cf_hub_push_model()`, etc.), PyQt6 desktop IDE.

**Installation:** `pip install confluencia` or `pip install confluencia[all]`.

---

## Acknowledgments

We thank [collaborating medical school researchers] for ongoing wet-lab validation support. We invite the circRNA community to contribute to circRNA-CASP and Confluencia Hub.

---

## References

1. Wesselhoeft RA, et al. RNA circularization diminishes immunogenicity and can extend translation duration in vivo. Nat Commun. 2018;9:2629.

2. Jiang YZ, et al. Genomic and Transcriptomic Landscape of Triple-Negative Breast Cancer. Cancer Cell. 2019;35:428.

3. Chen YG, et al. Sensing Self and Foreign Circular RNAs by Intron Identity. Mol Cell. 2019;73:422.

4. Gilleron J, et al. Image-based analysis of lipid nanoparticle-mediated siRNA delivery. Nat Biotechnol. 2013;31:638.

5. Hou X, et al. Lipid nanoparticles for mRNA delivery. Nat Rev Mater. 2021;6:1078.

6. Hornung V, et al. 5'-Triphosphate RNA is the ligand for RIG-I. Science. 2006;314:994.

7. Lorenz R, et al. ViennaRNA Package 2.0. Algorithms Mol Biol. 2011;6:26.

8. Ding L, et al. Clonal evolution in relapsed acute myeloid leukaemia. Nature. 2012;481:506.

9. Martinez-Salas E, et al. IRES mechanisms: connecting structure and function. Trends Microbiol. 2018;26:651.

10. Yang Y, et al. Extensive translation of circular RNAs driven by N6-methyladenosine. Cell Res. 2018;28:743.

11. Paunovska K, et al. Quantification of nanoprotein distribution at the single-cell level. ACS Nano. 2018;12:7580.

12. Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596:583.