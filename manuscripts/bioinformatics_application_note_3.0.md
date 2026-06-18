# Confluencia 3.0: Integrated circRNA Design and Triple-Negative Breast Cancer Simulation

**Running Title:** Confluencia 3.0 circRNA-TNBC Platform

**Keywords:** circRNA, TorusFold, pharmacokinetics, TNBC, molecular subtype, simulation

---

## Abstract (100 words)

Confluencia 3.0 integrates circular RNA (circRNA) therapeutic design with triple-negative breast cancer (TNBC) simulation. TorusFold predicts circRNA 3D structure using torus positional encoding for circular topology, validated with periodicity error <1e-6 and closure distance <0.5 Å. CirculaPK models six-compartment pharmacokinetics from injection to protein translation. Four-pathway immunogenicity scoring (RIG-I, TLR7, TLR8, PKR) guides sequence evolution via Pareto multi-objective optimization. The TNBC Simulacrum simulates four Jiang 2019 molecular subtypes (BLIS, BLIA, IM, LAR) with tumor heterogeneity, microenvironment dynamics, and treatment response. An EventBus architecture coordinates modules for circRNA vaccine design experiments. Available at github.com/IGEM-FBH/confluencia-3.0 under MIT license.

---

## Introduction (150 words)

Circular RNAs offer superior stability for therapeutic cargo delivery (Wesselhoeft et al., 2018), yet existing structure prediction tools assume linear topology. circRNA's covalently closed loop requires positional encoding invariant to sequence origin. Triple-negative breast cancer (TNBC) exhibits heterogeneous molecular subtypes with distinct immunotherapy responses. Lehmann et al. (2011) identified six subtypes (BL1, BL2, M, IM, MSL, LAR); Jiang et al. (2019) proposed an independent four-subtype classification (BLIS, BLIA, IM, LAR) based on immune microenvironment characteristics, while Lehmann's mesenchymal (M) and mesenchymal stem-like (MSL) subtypes remain relevant for stromal-enriched tumors.

Confluencia 3.0 addresses three gaps: (1) circRNA-specific structure prediction accounting for circular topology; (2) pharmacokinetic modeling from injection through LNP delivery to protein translation; and (3) patient-stratified TNBC simulation for circRNA therapeutic assessment. The platform integrates TorusFold (torus positional encoding + AlphaFold3-inspired architecture), CirculaPK (six-compartment PK model), immunogenicity scoring, Pareto sequence evolution, and a TNBC Simulacrum implementing the Jiang 2019 four-subtype classification (BLIS/BLIA/IM/LAR) with optional Lehmann M/MSL extension for mesenchymal tumors. Modules coordinate via an EventBus, enabling in silico circRNA vaccine experiments with biomarker-stratified patient cohorts.

---

## Methods (350 words)

**TorusFold structure prediction.** Torus positional encoding (TPE) enforces periodicity: PE[i] = PE[i+L] for sequence length L, ensuring invariance to circRNA origin position. Angles theta_i = 2pi * i / L generate harmonic embeddings: PE[2k] = sum_h w_{h,k} * sin(h*theta_i), PE[2k+1] = sum_h w_{h,k} * cos(h*theta_i), where k indexes the embedding dimension (k = 0, 1, ..., K-1) and h denotes harmonic index within each dimension, with h enumerated as 1, 2, 4, ..., 2^{floor(K/2)} per dimension pair. Learnable harmonic weights w_{h,k} are initialized as w_{h,k} = 1/(h+1) and updated during training. Embedding dimension d = 2K typically ranges 128-256 for circRNA sequences (K = 64-128), with higher harmonics capturing fine-grained positional distinctions near the back-splice junction and lower harmonics encoding global circular topology. Higher harmonics capture fine-grained positional distinctions near back-splice junction, while lower harmonics encode global circular topology. The periodicity constraint ensures PE[0] - PE[L] norm < 1e-6, critical for learning BSJ-proximal base pairing. An AlphaFold3-inspired CircPairformer refines pair representations via triangle multiplication updates with circular distance bias d(i,j) = min(|i-j|, L-|i-j|). A configurable structure module (simple MDS, diffusion, physics_b constraint solver, or physics_ba with OpenMM refinement) outputs 3D coordinates with closure constraint. Multi-task heads predict immunotherapy scores, translation efficiency, and back-splice junction (BSJ) confidence.

**CirculaPK pharmacokinetics.** Six compartments model circRNA fate: Injection -> LNP -> Endosome -> Cytoplasm -> Protein -> Clearance. Rate constants derive from literature: uptake (IV: 0.80/h), endosomal escape (LNP: 0.025/h), translation (IRES-dependent: 0.02-0.32/h), degradation (modification-adjusted: 0.04-0.12/h). Tissue distribution coefficients capture LNP organ tropism (liver: 0.80, spleen: 0.10). PK curves are solved via RK45 integration, outputting AUC, peak protein, expression window, and half-life estimates.

**Immunogenicity scoring.** Four innate pathways are scored with circRNA-specific logic. RIG-I activation (weight 0.35) uses dsRNA backbone detection via inverted repeat analysis, not blunt-end sensing (circRNA lacks 5'/3' termini). TLR7 (0.20) scores GU-rich motifs; TLR8 (0.15) scores AU-rich motifs with separate models. PKR (0.30) applies the >33 bp dsRNA threshold. m6A modification marks circRNA as "self," preventing RIG-I activation; Chen et al. (2019) showed that approximately 90% of endogenous circRNAs carry m6A modifications that prevent immune recognition, whereas foreign circRNAs lacking intron identity marks may activate RIG-I through dsRNA regions (mechanism distinct from linear RNA blunt-end sensing). Overall immunogenicity is weighted sum normalized [0,1].

**Sequence evolution.** REINFORCE-based policy (learning_rate=0.001, episodes=500, entropy_coefficient=0.01) optimizes circRNA over 5 rounds (convergence criterion: reward change <0.01 for 3 consecutive rounds) with four operators: backbone mutation (BSJ-protected, 10% mutation rate), IRES motif insertion, flanking shuffling, modification selection (m6A, Psi, 5mC, 2OMe). Pareto front selection balances four circRNA-specific objectives: stability, translation efficiency, immune evasion, and delivery compatibility. **Objective count justification:** circRNA optimization requires fewer objectives (4) than drug molecule evolution (7 in Confluencia 2.0) due to constrained sequence space and circRNA-specific constraints (BSJ topology, IRES requirement). Dirichlet weight sampling (alpha=[1.0, 1.0, 1.0, 1.0], 100 samples) explores objective trade-offs.

**TNBC Simulacrum.** The Jiang 2019 four-subtype classification is implemented: BLIS (basal-like immune-suppressed, worst prognosis, BRCA1-associated, low TIL density ~10%); BLIA (basal-like immune-activated, better prognosis, immune signaling, moderate TIL ~30%); IM (immunomodulatory, high PD-L1 expression, abundant TIL ~60%, checkpoint inhibitor responsive); LAR (luminal androgen receptor, AR-driven, responds to anti-androgen therapy, distinct from basal subtypes). An optional Lehmann extension adds M/MSL subtypes for mesenchymal tumors with EMT signature and stromal enrichment. Each subtype modulates growth rate, immune evasion, and drug sensitivity with literature-derived parameter ranges (Jiang et al., 2019; Lehmann et al., 2011). Tumor heterogeneity tracks subclone fitness, resistance evolution, and Shannon diversity. TME modules simulate immune dynamics, CAF activation, and immunoediting phases. Treatment arms include chemotherapy, immunotherapy, and circRNA therapy. EventBus coordinates 40+ event types for module communication.

---

## Results (200 words)

**TorusFold validation.** No public circRNA 3D benchmark exists (Circ-CASP not established). Periodicity verification confirms PE[i] - PE[i+L] norm < 1e-6. Pair representation (c_z=128) captures BSJ proximity via circular distance bias. Closure distance < 0.5 Å (physics_ba mode, geometric constraint enforcement). Multi-task head correlation with experimental immune activation: **Leave-One-Out Cross-Validation (LOO-CV)** for N=15 validation samples yields Pearson r=0.68 [95% CI: 0.26, 0.88] computed from 15 LOO-CV aggregated predictions vs. actual values. Wide CI reflects small sample size; independent validation on larger datasets required. Internal HEK293 IFN-β ELISA validation (15 sequences spanning immunogenic to safe profiles) supports immunogenicity prediction.

**CirculaPK predictions.** Simulated 7-day PK curves for LNP-delivered circRNA show peak cytoplasmic RNA at 4-8 h, protein expression window 24-96 h. Modification comparison: m6A extends IVT circRNA half-life from baseline 8-12 h (unmodified, consistent with Wesselhoeft et al. 2018) to approximately 15-22 h; Psi extends to approximately 20-30 h. Note: endogenous circRNAs exhibit longer half-lives (18-24+ h) due to cellular stabilization mechanisms; IVT circRNAs have reduced stability absent these factors. LNP liver tropism: 80% hepatic distribution matches Paunovska et al. (2018) biodistribution data.

**TNBC subtype simulation.** BLIS subtype shows low TIL density (0.10 [0.08-0.15] from Jiang 2019), worst prognosis, BRCA1 mutation association, high basal markers (CK5/6=0.80 [0.65-0.90]); BLIA shows moderate TIL (0.30 [0.25-0.40]), immune gene signatures, better prognosis than BLIS; IM shows high PD-L1 (0.50 [0.40-0.60]), TIL density (0.60 [0.50-0.70]), checkpoint inhibitor responsiveness; LAR shows AR expression (0.70 [0.60-0.85]), responds to enzalutamide/bicalutamide, luminal gene signature distinct from basal subtypes. Subclone heterogeneity evolution: Shannon diversity increases under chemotherapy pressure from 0.4 to 1.2 over 30 cycles. Resistance clone fraction: 18 cycles to 20% resistant population under gemcitabine monotherapy.

**Sequence evolution convergence.** REINFORCE policy shifts action logits toward IRES optimization (+0.15) and m6A selection (+0.08) over 5 rounds. Pareto front size: 3-5 candidates per round spanning stability-immune trade-off.

---

## Comparison with Existing Tools

| Feature | Confluencia 3.0 | ViennaRNA [6] | AlphaFold [7] | PK-Sim/ADAPT |
|---------|-----------------|-----------|-----------|-----------------|
| circRNA topology | Torus PE (circular) | Linear assumption | Linear chain | Linear RNA |
| Structure prediction | CircPairformer + diffusion | MFE folding | Protein-centric | N/A |
| Pharmacokinetics | 6-compartment circRNA | N/A | N/A | 1-2 compartment |
| Immunogenicity | 4 pathways (circRNA) | N/A | N/A | N/A |
| TNBC subtypes | 4 molecular | N/A | N/A | N/A |
| Sequence evolution | Pareto + REINFORCE | N/A | N/A | Codon optimization |
| EventBus coordination | 40+ event types | N/A | N/A | N/A |

---

## Availability

**Name:** Confluencia 3.0  
**Version:** 3.0.0  
**License:** MIT  
**URL:** https://github.com/IGEM-FBH/confluencia-3.0  
**Programming language:** Python 3.10+  
**Dependencies:** numpy, pandas, scipy, torch, ViennaRNA (optional)  
**Operating systems:** Linux, macOS, Windows  
**Interfaces:** Python API, CLI, experiment sandbox

---

## Limitations

TorusFold lacks experimental circRNA 3D structure validation (Circ-CASP does not exist). Immunogenicity weights are author-informed heuristics, not empirically calibrated. PK parameters derive from literature priors, not patient-specific data. TNBC Simulacrum uses simplified ODE dynamics; clinical outcome predictions (PFS/OS) use Cox approximations with unvalidated C-index. EventBus is single-threaded synchronous; parallel execution requires external orchestration. TorusFold requires GPU with >=8GB VRAM for sequences >500 nt. Full pipeline (5 evolution rounds, 7-day PK simulation) completes in approximately 15 minutes on NVIDIA RTX 3080.

---

## References

1. Wesselhoeft RA, Kowalski PS, Anderson DG. Engineering circular RNA for potent and stable translation in eukaryotic cells. Nat Commun. 2018;9(1):2629. doi:10.1038/s41467-018-05096-x

2. Jiang YZ, Ma D, Suo C, et al. Genomic and Transcriptomic Landscape of Triple-Negative Breast Cancers: Subtypes and Treatment Strategies. Cancer Cell. 2019;35(3):428-440.e5. doi:10.1016/j.ccell.2019.02.001

3. Chen YG, Chen R, Ahmad S, et al. Sensing Self and Foreign Circular RNAs by Intron Identity. Mol Cell. 2019;73(3):422-434. doi:10.1016/j.molcel.2018.12.018

4. Paunovska K, Dahlman JE, et al. Quantitative Analysis of Particle Biodistribution. ACS Nano. 2018;12(8):7580-7593. doi:10.1021/acsnano.8b02167

5. Lehmann BD, Bauer JA, Chen X, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. J Clin Invest. 2011;121(7):2750-2767. doi:10.1172/JCI45014

6. Lorenz R, et al. ViennaRNA Package 2.0. Algorithms Mol Biol. 2011;6:26. doi:10.1186/1748-7188-6-26

7. Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596(7873):583-589. doi:10.1038/s41586-021-03819-2

---

## Acknowledgements

This work was supported by IGEM-FBH team funding. The authors thank collaborators for HEK293 ELISA validation support.

---

## Conflicts of Interest

The authors declare no conflicts of interest.

---

## Author Contributions

Conceptualization, methodology, software development, and manuscript writing by the IGEM-FBH software team. Wet lab validation protocol design and execution by collaborating laboratory.

---

## Data Availability

Source code available at https://github.com/IGEM-FBH/confluencia-3.0 under MIT license. Internal validation data (HEK293 IFN-β ELISA) available upon reasonable request pending publication of full wet lab results.
