# Confluencia 3.0

## Why this project exists: The problem we solve

**TNBC patients have no targeted therapy options. Existing AI drug discovery approaches are built on fundamentally wrong data. Confluencia is the first computational platform that integrates circRNA immunogenicity dynamics with TNBC molecular subtype heterogeneity to enable rational circRNA therapeutic design.**

Here is what most people do not know about computational drug discovery for TNBC:

1. **TNBC lacks druggable targets**: Other breast cancers have HER2 (trastuzumab target) or hormone receptors (endocrine therapy targets). TNBC has none. Small molecule drug screens that look for "activity against breast cancer" find compounds that work through estrogen pathways, which is useless for TNBC because TNBC does not have estrogen receptors.

2. **Public screening data is contaminated**: NCI-60, ChEMBL, and other public databases contain measurements from mixed breast cancer cell lines, not TNBC-specific models. A compound that kills MCF-7 cells might appear "active" in the dataset, but its mechanism is irrelevant for TNBC patients. We learned this the hard way after six months of failed machine learning experiments.

3. **circRNA is a different paradigm**: Instead of searching for existing small molecules with activity, circRNA therapeutics let you design sequences that produce specific proteins. This changes the computational problem from "pattern recognition on noisy screening data" to "mechanistic prediction of sequence behavior in a biological context."

4. **No existing tool connects circRNA to tumors**: RNA structure predictors treat sequences as isolated molecules. Tumor simulators model drug effects without circRNA-specific pharmacology. Neither can answer the question: "If I design this circRNA to express this protein, what happens inside a TNBC tumor with its specific immune microenvironment?"

Confluencia was built to fill this gap. It is not just another RNA structure predictor. It is a simulation platform that evaluates circRNA sequences in the context of TNBC biology, including subtype-specific heterogeneity (BLIS, IM, M, LAR), immune dynamics, and circRNA-specific pharmacokinetics.

---

## Project development history

### How we started, and why it went wrong

We began with what seemed like a reasonable plan: use machine learning to find small molecule drugs for TNBC. Pharma companies have decades of screening data. There are public databases full of compounds tested against cancer cell lines. The standard approach is to train a model on that data, then ask it to predict which new molecules might work.

We spent about 3 months on this. NCI-60 has maybe 50,000 compounds tested across 60 cell lines. ChEMBL has hundreds of thousands of bioactivity measurements. PubChem has structures and assay results. We built neural networks(you mat see them in developer's github) to predict TNBC-specific activity. Graph neural networks, transformers, random forests. We tried most of the usual architectures.

The best model we got had about [准确率待填充] accuracy. [对比分析待填充: 是否比随机猜测好? 与文献baseline对比?]

The problem was not the code. The problem was the data. Most public screening data comes from generic cancer cell lines, not TNBC. A compound that kills MCF-7 cells might work through an estrogen pathway, which is irrelevant for TNBC because TNBC does not have estrogen receptors. The datasets were full of noise from mechanisms that do not matter for the disease we were targeting.

There is also a deeper issue with TNBC itself. HER2-positive breast cancer has a target: HER2. Hormone-positive cancers have targets: ER and PR. TNBC has none of these. There is no obvious molecular vulnerability to exploit with a small molecule. The disease is heterogeneous and aggressive, and the drug development track record reflects that.

We had assumed that statistical patterns in screening data would point toward something useful. That assumption turned out to be wrong for TNBC.

### What we switched to

Around the time we realized the small molecule work was not going to work, circRNA was starting to look interesting as a therapeutic format. Several groups had shown that circRNA could express proteins in vivo with better stability than linear mRNA. It is a different paradigm: instead of searching for an existing compound with activity, you design a sequence that produces a protein you want.

But circRNA has its own problems. RNA triggers immune sensors. If you inject unmodified circRNA, you get inflammation. circRNA folds into circles, not linear strands. The structure prediction problem is different. The pharmacokinetics are different. And there was no tool that combined circRNA design with tumor simulation. You could predict circRNA structure in isolation, or simulate tumor growth in isolation, but not both together.

We needed something that could answer: "If I design this circRNA to express this protein, what happens in the tumor?"

### Why the architecture looks like this

The EventBus setup came from a specific frustration. The small molecule models failed because they had no biological context. They predicted activity against "cancer" without distinguishing between mechanisms that matter and mechanisms that do not. We wanted the circRNA system to be different. If a circRNA triggers RIG-I, that changes immune cell recruitment. If immune cells change, that changes tumor growth. These things are connected.

The EventBus lets modules talk to each other without hard wiring the connections. The tumor module does not need to know how circRNA PK works. It just needs to know when concentration changes. The circRNA module handles its own math and publishes events. Everyone else responds.

The physics-based structure prediction was also a compromise we did not originally plan for. We wanted to train deep learning models, like AlphaFold. Then we looked for circRNA structural data and found there is almost none. Maybe twelve published structures, most of them fragments. So we wrote a constraint solver instead. It is not as good as a trained neural net would be, but it runs without any training data.

The deep learning modules are still in the code. They are just dormant. If someone publishes a large circRNA structural dataset, or if synthesis gets cheap enough that we can generate our own, the diffusion model can train overnight and wake up.

### Where we are now

The software runs. Wet lab validation is in progress.

**Current validation status:**

| Validation type | Method | Status | Results |
|-----------------|--------|--------|---------|
| Immunogenicity | IFN-β ELISA in HEK293 cells | [序列数量] sequences being tested | [相关性结果待填充] |
| Structure | Collaborator crystallography | Samples submitted | Awaiting results, expected before Giant Jamboree |
| PK model | Comparison to published data | Literature benchmark in progress | [对比结果待填充] |

**Planned wet lab experiments:**

- ICC staining for immune markers (RIG-I, TLR7) in transfected cells
- ELISA for IFN-β and IL-6 induction levels
- Flow cytometry for cell viability and protein expression

**Expected timeline:**

All validation data will be updated on this Wiki before the Giant Jamboree. Preliminary results are expected to show correlation between predicted immunogenicity scores and measured cytokine induction, but quantitative analysis is pending completion of wet lab experiments.

The small molecule prediction work has been discontinued. The shift from pattern recognition on noisy datasets to mechanistic simulation was driven by empirical failure, not theoretical insight. We learned from what did not work, and that learning shaped everything that followed.

---

### Software components

The platform comprises four primary modules:

- TorusFold: circRNA three-dimensional structure prediction using deep learning methods informed by AlphaFold3 architecture
- CirculaPK: six-compartment pharmacokinetic model from injection through protein expression
- REINFORCE Evolution: multi-objective sequence optimization
- TNBC Simulacrum: tumor growth, immune dynamics, and treatment response simulation

---

## Description

### Module functions

| Module | Function | Technical approach |
|--------|----------|-------------------|
| TorusFold | circRNA 3D structure prediction | Torus Positional Encoding enforces mathematical equivalence of positions 0 and L, preserving circular topology |
| Immune Sensing | Four-pathway immunogenicity assessment | Separate prediction scores for RIG-I, TLR7, TLR8, and PKR pathways |
| CirculaPK | Pharmacokinetic simulation | Six compartments track circRNA from injection depot through endosomal escape to protein output |
| Sequence Evolution | Multi-objective optimization | Pareto front generation enables trade-off analysis among stability, translation, immune safety, and delivery parameters |
| TNBC Simulacrum | Tumor simulation | Four molecular subtypes (BLIS, IM, M, LAR) with distinct growth rates and immune profiles |

### TNBC molecular subtypes

The simulator implements the four TNBC subtypes described by Lehmann et al. (2011):

- BLIS: basal-like immune-suppressed, high proliferation rate, low immune cell infiltration
- IM: immunomodulatory, abundant tumor-infiltrating lymphocytes
- M: mesenchymal, epithelial-mesenchymal transition signature, stromal enrichment
- LAR: luminal androgen receptor driven, amenable to AR-targeted therapy

### Backend tier system

To ensure functionality across diverse computational environments, Confluencia implements a three-tier backend architecture:

```
ESM2 (Tier 0) → ViennaRNA (Tier 1) → Heuristic rules (Tier 2)
    GPU required    CPU, local         pure Python, no dependencies
```

The tiered system enables operation on hospital intranet systems without external API access, with progressively reduced prediction accuracy at lower tiers.

### Plug-and-play model integration

Training custom neural networks for most prediction tasks is not necessarily required. Instead, it integrates existing mature models as interchangeable backend modules(And it also leave a way for future developers to custom algorithms as their needs).

**Why plug-and-play matters:**

| Problem with custom training | Plug-and-play solution |
|------------------------------|------------------------|
| Training requires large datasets | Mature models (ESM2, ViennaRNA) already trained on millions of sequences |
| Training is computationally expensive | Pre-trained weights loaded directly, no GPU training needed |
| Custom models may underperform | Mature models are state-of-art, validated across benchmarks |
| Model maintenance is ongoing work | We delegate maintenance to original model developers |

**Current backend modules:**

| Module | Mature model used | What it provides | Why we chose it |
|--------|------------------|------------------|-----------------|
| Sequence embeddings | ESM2 (Meta AI) | 1280-dimensional embeddings for RNA sequences | Largest public RNA language model, trained on 10M+ sequences |
| Secondary structure | ViennaRNA | Minimum free energy folding | 30+ years of validation, standard in field |
| Molecular dynamics | OpenMM | Physics-based structure refinement | GPU-accelerated, flexible, open source |
| Immunogenicity scoring | our model | Pathway-specific immune activation predictions | Custom scoring based on published motif libraries |

**How the system works:**

When Confluencia needs a sequence embedding, it calls ESM2. ESM2 was trained by Meta AI on millions of RNA sequences. We do not train anything. We just load their pre-trained weights (frozen) and use the outputs as inputs to our downstream modules.

When Confluencia needs secondary structure prediction, it calls ViennaRNA. ViennaRNA implements thermodynamic folding algorithms developed over 30+ years. It does not need training. It computes minimum free energy structures directly from physics-based rules.

When hardware fails, Confluencia automatically drops to a lower tier with a clear alarm. If GPU is unavailable, ESM2 cannot run. The system falls back to ViennaRNA (CPU-only). If ViennaRNA is not installed, the system falls back to heuristic rules (pure Python). The user gets predictions at whatever accuracy tier their hardware supports.

**Future model upgrades:**

If a better RNA language model is released (e.g., ESM3), we can swap ESM2 for the new model by changing one configuration line. No code rewrite needed. The plug-and-play architecture means Confluencia improves as the field improves, without us rebuilding anything.

**What we do train:**

The only module that might require custom training is the immunogenicity scorer, which combines motif-based rules with sequence context. Current implementation uses rule-based scoring. If sufficient wet lab validation data accumulates through Hub contributions, a neural network could be trained to refine predictions. But this is optional, not required for the system to function.

---

## Design

### Architecture

The computational core is an EventBus that coordinates six subsystem managers. Each manager maintains its state partition and publishes events upon state changes. The circRNA manager communicates with tumor and TME managers through a bridge layer rather than direct function calls.

```
         EventBus (40 event types)
            │
    ┌───────┼───────┬───────┬───────┐
    │       │       │       │       │
 Tumor   TME   Treatment CircRNA Clinical
    │       │       │       │       │
    └───────┼───────┼───────┼───────┘
            │
      Confluencia Bridge
```

Event categories include:

- Tumor biology: growth, heterogeneity, angiogenesis, metastasis, cancer stem cell dynamics
- Microenvironment: immune cell dynamics, fibroblast activation, endothelial changes, immune evasion
- Treatment: drug administration, pharmacokinetic updates, pharmacodynamic effects, resistance emergence
- circRNA: immune evaluation, structure prediction, sequence evolution, pharmacokinetic simulation
- Clinical: RECIST evaluation, survival updates, toxicity grading, subtype reclassification

### TorusFold

TorusFold accepts a circRNA sequence as input and outputs three-dimensional structure coordinates with associated confidence metrics. The primary distinction from conventional RNA structure predictors is the treatment of sequence topology as circular rather than linear.

Standard positional encoding treats nucleotide positions 0 and L as distant positions. Torus Positional Encoding (TPE) establishes mathematical equivalence between these positions, enabling the neural network to recognize that nucleotides proximal to the back-splice junction can form base pairs despite their separation in linear sequence representation.

Architecture flow:

```
Sequence → TPE (periodic encoding) → ESM2 backbone (frozen weights)
         → CircPairformer (triangle updates with circular distance bias)
         → Structure head (four modes: simple MDS, diffusion, geometry solver, OpenMM refinement)
```

### Rationale for physics-based structure prediction

AlphaFold achieves high accuracy because the Protein Data Bank contains thousands of experimentally determined protein structures that serve as training data. For circRNA, this resource does not exist.It cannot be simply said that there is a lack of data; rather, there is no qualified data at all, with most of those limited data represent partial sequences rather than complete circular molecules.

The physics-based structure head (physics_b and physics_ba modes) was developed to operate without training data. The module accepts predicted pair probabilities from the neural network, converts them into geometric constraints (bond lengths, pairing distances, clash avoidance parameters), and solves for atomic coordinates through constraint propagation. This approach requires no experimental training data, relying solely on physical principles.

The deep learning modules (diffusion head, complete CircPairformer architecture) are implemented but remain inactive pending data availability. The neural architecture is complete. The limitation is the absence of training datasets.

Current circRNA synthesis costs present a practical barrier. Commercial providers charge hundreds of US dollars per 500-nucleotide circRNA molecule. Large-scale synthesis of 10,000 variants with subsequent crystallization for training data generation is economically infeasible for most laboratories. However, if circRNA manufacturing costs decrease substantially, or if a large structural dataset becomes publicly available, the dormant diffusion module could be trained and activated within approximately 24 hours, enabling AlphaFold-equivalent accuracy for circRNA predictions.

In the interim, physics_ba mode (constraint solver with OpenMM molecular dynamics refinement) provides structure prediction capability without experimental data requirements. Prediction accuracy is lower than a trained model would achieve, but the method functions for any circRNA sequence.

We designed TorusFold to be CASP-ready — fold-level accuracy, domain segmentation, confidence estimation — even though there is no CASP category for circular RNA. Yet. (AlphaFold debuted in Nature 2021. If Jilin University ever funds a Circ-CASP category... well, we have ambitions too.)

### CirculaPK pharmacokinetic model

The six-compartment model structure:

```
Depot → Blood → Tissue
              ↓
         Endosome → Cytoplasm → Protein
```

Each transition is characterized by a rate constant. Model outputs include area under the concentration curve (AUC), peak concentration, elimination half-life, and predicted protein expression magnitude.

### Additional modules

**RNA modifications module** (`rna_modifications.py`):

| Modification | Effect on immunogenicity | Effect on translation |
|--------------|--------------------------|----------------------|
| m6A (N6-methyladenosine) | Reduces RIG-I activation by ~50% | Minor impact, may enhance IRES activity |
| Ψ (pseudouridine) | Reduces TLR7/8 activation | Enhances stability |
| 2'-O-methyl | Reduces all pathway activation | Strong stability enhancement |

The module predicts which modifications should be applied at which positions to optimize the trade-off between immune evasion and translation efficiency.

**Folding kinetics module** (`folding_kinetics.py`, `folding_pathways.py`):

circRNA does not fold instantly. The folding process takes milliseconds to seconds and proceeds through intermediate states. The kinetics module predicts:
- Folding time scale (how long until stable structure)
- Intermediate states (potential kinetic traps)
- Final structure accessibility (is the IRES exposed?)

This matters because a circRNA that folds into a "kinetic trap" may never reach its functional conformation. Predicting folding pathways helps design sequences that reliably reach useful structures.

**Co-transcriptional folding** (`cotrans_folding.py`):

circRNA is synthesized by ribosomes rolling around the circular template. The sequence folds as it is being produced, not after synthesis is complete. This module simulates folding during synthesis, predicting which structures form early and whether they persist.

**BSJ feature extraction** (`bsj_features.py`):

The back-splice junction is the defining feature of circRNA. This module extracts structural features specific to the junction:
- Junction stability (will the circle stay closed?)
- Junction accessibility (can enzymes cut it?)
- Junction pairing potential (does it base-pair with distant regions?)

**RNA docking module** (`rna_docking.py`):

If a circRNA expresses a miRNA sponge, it needs to bind target miRNAs. If it expresses a protein-binding scaffold, it needs to dock with that protein. This module predicts circRNA-target interactions using docking algorithms.

**Multi-drug combination** (`multi_drug_combination.py`):

circRNA therapy is unlikely to be used alone. Most TNBC patients receive chemotherapy or immunotherapy. This module predicts how circRNA affects response to:
- Chemotherapy (does circRNA enhance chemo sensitivity?)
- Immunotherapy (does circRNA modulate checkpoint inhibitor efficacy?)
- Combination therapy (optimal dosing schedules)

**Adaptive dosing** (`adaptive_dosing.py`):

PK/PD models predict concentration curves, but patient variability is high. This module implements adaptive dosing strategies:
- Start with predicted dose
- Monitor simulated biomarkers
- Adjust dose based on simulated response
- Iterate toward optimal

**Patient stratification** (`patient_stratification.py`):

Not all TNBC patients are the same. This module stratifies patients based on:
- Molecular subtype (BLIS, IM, M, LAR)
- Immune profile (hot vs cold tumor)
- Prior treatment history
- Predicted response likelihood

**Clinical prediction outputs** (`clinical_prediction.py`, `clinical/`):

The simulator generates clinical-relevant outputs:

| Output | Module | What it predicts |
|--------|--------|------------------|
| RECIST response | `clinical/recist.py` | Complete/partial response, stable disease, progression |
| Survival estimate | `clinical/survival.py` | Median survival, progression-free survival |
| Toxicity grade | `clinical/toxicity.py` | CTCAE grading for immune-related adverse events |

**Biomarker tracking** (`biomarker/`):

| Biomarker | Module | Clinical relevance |
|-----------|--------|-------------------|
| Resistance markers | `resistance_detector.py` | Early detection of treatment failure |
| Subtype reclassification | `subtype_classifier.py` | Tumor subtype may shift during treatment |
| Longitudinal tracking | `tracker.py` | Biomarker trajectories over time |

**Bridge modules** (`confluencia/`):

| Bridge | What it connects |
|--------|------------------|
| `drug_bridge.py` | circRNA PK → tumor drug concentration |
| `epitope_bridge.py` | circRNA protein output → antigen presentation |
| `pk_bridge.py` | circRNA pharmacokinetics → treatment efficacy |
| `joint_bridge.py` | circRNA + chemotherapy combined effects |

These bridges are the key integration layer. They translate circRNA module outputs into events that tumor, TME, and clinical modules can consume.

---

## Epitope 2.0: MHC-I epitope efficacy prediction

### What it does

After a circRNA is translated, the resulting peptide must be presented by MHC class I to trigger T cell responses. Epitope 2.0 predicts which sequences will work. It connects to Confluencia 3.0 through `EpitopeBridge`, which sends protein output to the tumor immune subsystem as antigen presentation events.

### Three encoding strategies

| Approach | How it works | Result |
|----------|--------------|--------|
| Mamba3Lite encoder | Three time-scale state-space model with pooling at four scales (mean, local 3-5, meso 10-15, global full peptide) plus attention | MAE=0.395, R²=0.802 |
| MHC pseudo-sequence | 34-position encoding of each HLA allele's binding groove (from NetMHCpan-4.1) | AUC=0.917 |
| MOE ensemble | Seven regression models weighted by inverse out-of-fold RMSE, adapting to dataset size | MAE=0.389 |

### ESM-2 failed. Here is why.

ESM-2 (650M parameters, Meta AI) got AUC=0.537 on MHC binding prediction. Worse than the pseudo-sequence method. We tried three variants, and all failed.

| Variant | AUC | Problem |
|---------|-----|---------|
| ESM-2 PCA 64D replacing traditional features | 0.508 | PCA threw out the directions that matter |
| Traditional + ESM-2 PCA 35M supplement | 0.594 | Extra embedding dimensions added noise |
| Traditional + ESM-2 PCA 650M supplement | 0.537 | Mean pooling averaged away the signal |

The core issue: MHC binding depends on anchor positions (P2 and P9 for 9-mers). ESM-2 mean pooling averages all positions together, which washes out the anchor-specific signal. ESM-2 was trained on full proteins, not 8-11mer peptides. There is not enough sequence for the model to learn anything useful about structure. The embedding directions that work for protein-level tasks do not align with what determines MHC binding.

We spent about 48 GPU hours on this. The takeaway: large protein language models do not transfer to short peptide immunogenicity. Pseudo-sequence encoding from NetMHCpan remains the best approach.

### Algorithm details

**Mamba3Lite state recursion:**

```
h_t = alpha_t * h_{t-1} + beta_t * x_t
y_t = gamma_t * h_t
```

alpha, beta, gamma are learned per position. Pooling operates at four scales and concatenates outputs.

**MOE weighting:**

```
w_k = 1/max(RMSE_k, epsilon) / sum_j(1/max(RMSE_j, epsilon))
y_pred = sum_k w_k * y_k
```

Models with lower validation RMSE get higher weight.

**MHC pseudo-sequence:** 34 amino acid positions from the MHC binding groove. One-hot encoding gives a 680-dimensional vector per allele-peptide pair.

### Validation numbers

| Dataset | Metric | Value | Method |
|---------|--------|-------|--------|
| IEDB 288K peptides | AUC (allele-aware) | 0.80 | HGB + MHC features |
| Internal validation | MAE | 0.389 | MOE ensemble |
| MHC binding subset | AUC | 0.917 | Pseudo-sequence |
| Same subset | AUC | 0.537 | ESM-2 baseline |

### Input and output

Input CSV columns:

```
epitope_seq,dose,freq,treatment_time,circ_expr,ifn_score,efficacy
```

epitope_seq is the 8-11mer peptide. Other columns are experimental context. If efficacy labels are missing, the model builds a proxy target from the numeric features.

Output: predicted efficacy score, sensitivity analysis (top features and importance), and neighborhood contribution at three scales.

### Windows note

Mamba-ssm (CUDA) does not run on Windows. The module falls back to pure PyTorch on CPU. Same architecture, slower but functional.

### Code structure

```
confluencia-2.0-epitope/
├── core/
│   ├── training.py          # Train/predict API
│   ├── mamba3.py            # Mamba3Lite encoder
│   ├── moe.py               # MOE ensemble
│   ├── mhc_features.py      # Pseudo-sequence encoding
│   ├── features.py          # Sequence features
│   └── sensitivity.py       # Multi-scale sensitivity
├── epitope_frontend.py      # Streamlit interface
└── tools/epitope_cli.py     # CLI
```

---

## Implementation

### Technology stack

Python 3.10 or higher, PyTorch for neural network implementation, ESM2 sequence embeddings from Meta AI Research (frozen weights). ViennaRNA provides thermodynamic folding capability when GPU resources are unavailable. OpenMM enables molecular dynamics refinement in physics_ba mode as an optional component. System configuration through Python dataclasses and YAML files.

### Core algorithms

**Torus Positional Encoding**

```python
omega = 2 * pi / length
# Harmonic components encode positional information
pe[2*i] = sin(omega * (2**i) * position)
pe[2*i+1] = cos(omega * (2**i) * position)
# Positions 0 and L generate identical vector representations
```

**Circular Distance Matrix**

```python
d_circ(i, j) = min(|i - j|, L - |i - j|)
# Linear distance metric for linear RNA sequences
# Circular distance metric for circRNA where terminal positions are connected
```

**Multi-objective reward function**

```python
reward = 0.35 * stability + 0.30 * translation + 0.25 * immune_evasion + 0.10 * delivery
# Weight parameters adjustable based on optimization objectives
```

### Directory structure

```
confluencia_3_0/
  main.py
  core/
    agent.py                 # Main simulation loop
    event_bus.py             # Pub/sub core
    state_schema.py          # State keys (~200)
    events.py                # 40 event types
    config.py                # Configuration management
    
    tumor/                   # Growth, CSC, angiogenesis, metastasis
    tme/                     # Immune cells (CD4, CD8, Treg, NK), cytokines, fibroblasts
    
    treatment/               # Chemo, immunotherapy, targeted, radio
    
    circrna/
      torusfold/             # Structure prediction modules
        torusfold.py         # Main entry point
        tpe.py               # Torus positional encoding
        triangle_update.py   # CircPairformer triangle updates
        diffusion_structure.py  # Diffusion head (dormant)
        physics_structure_head.py  # Physics-based solver
        constraint_solver.py # Geometric constraint propagation
        cgmd_refiner.py      # Coarse-grained MD refinement
        equivariant_backbone.py  # Rotation-equivariant network
        structure_validator.py  # Physical sanity checks
        tertiary_interaction.py  # 3D interaction prediction
      
      immune_sensing.py      # Four pathway scoring
      torusfold_scorer.py    # DL output to objectives
      rna_modifications.py   # m6A, Ψ modification effects
      folding_kinetics.py    # Folding time scale prediction
      folding_pathways.py    # Intermediate state prediction
      cotrans_folding.py     # Synthesis-time folding
      bsj_features.py        # Back-splice junction analysis
      rna_docking.py         # RNA-target docking
      cirrna_evolution.py    # Multi-objective sequence optimization
      circrna_rl_abm.py      # Agent-based model for circRNA dynamics
      
      multi_drug_combination.py  # circRNA + chemo/immunotherapy
      adaptive_dosing.py     # PK/PD feedback dosing
      patient_stratification.py  # Patient classification
      drug_response.py       # Treatment response prediction
      clinical_prediction.py # RECIST, survival outputs
      tme_simulation.py      # TME immune dynamics
    
    confluencia/             # Bridge modules (integration layer)
      drug_bridge.py         # PK → tumor concentration
      epitope_bridge.py      # Protein → antigen presentation
      pk_bridge.py           # PK → treatment efficacy
      joint_bridge.py        # circRNA + drug combined
    
    pk/
      rnactm.py              # Six-compartment ODE solver
    
    biomarker/
      resistance_detector.py # Treatment failure markers
      subtype_classifier.py  # Tumor subtype tracking
      tracker.py             # Longitudinal biomarker monitoring
    
    clinical/
      recist.py              # Response evaluation criteria
      survival.py            # Survival prediction
      toxicity.py            # Adverse event grading
    
    encoder/
      adapter.py             # Model interface layer
      config.py              # Encoder configuration
      model.py               # ESM2/ViennaRNA wrapper
```

---

## Usage

### Installation

```bash
git clone https://github.com/your-team/confluencia-3.0.git
cd confluencia-3.0
pip install -r requirements.txt

# Optional dependencies
conda install -c bioconda viennarna    # Thermodynamic folding
conda install -c conda-forge openmm    # MD refinement (physics_ba mode)
```

### Command-line execution

```bash
python -m confluencia_3_0 --subtype BLIS --steps 365
python -m confluencia_3_0 --circrna-backend vienna
python -m confluencia_3_0 --structure-mode diffusion
```

### Python API examples

Sequence evaluation:

```python
from confluencia_3_0.core.circrna.torusfold_scorer import quick_score

result = quick_score("AUGCGC...", modification="m6A")
print(result['stability'], result['immune_evasion'])
```

Sequence evolution optimization:

```python
from confluencia_3_0.core.evolution.cirrna_evolution import evolve_cirrna

df, artifacts = evolve_cirrna(seed_seq="...", generations=50)
print(artifacts.best_sequence)
```

Pharmacokinetic simulation:

```python
from confluencia_3_0.core.pk.rnactm import simulate_rna_ctm

curve = simulate_rna_ctm(dose=1.0, params=infer_params("m6A", "LNP"))
print(f"Half-life: {curve['rna_half_life']} hours")
```

---

## Demonstration

### Case 1: Immunogenicity assessment

**Input sequence:** [示例序列待填充] (contains GGGG and UUGU motifs)

**Initial scores (predicted):**
| Pathway | Score | Risk level |
|---------|-------|------------|
| RIG-I | [分数待填充] | [风险等级] |
| TLR7 | [分数待填充] | [风险等级] |
| TLR8 | [分数待填充] | [风险等级] |
| PKR | [分数待填充] | [风险等级] |
| **Overall** | [分数待填充] | [优化建议] |

**Recommended modification:** [修改位置和类型待填充]

**After optimization (predicted):**
| Pathway | Score | Change |
|---------|-------|--------|
| RIG-I | [分数待填充] | [变化百分比] |
| TLR7 | [分数待填充] | [变化百分比] |
| TLR8 | [分数待填充] | [变化百分比] |
| PKR | [分数待填充] | [变化百分比] |

**Wet lab validation:** [实验结果待填充]

---

### Case 2: Subtype-specific treatment response

**Question:** Which TNBC subtype responds best to circRNA therapy?

| Subtype | Chemo response (literature) | Checkpoint inhibitor (literature) | circRNA therapy (predicted) |
|---------|----------------------------|----------------------------------|-----------------------------|
| BLIS | [文献数据] | [文献数据] | [预测分数待填充] |
| IM | [文献数据] | [文献数据] | [预测分数待填充] |
| M | [文献数据] | [文献数据] | [预测分数待填充] |
| LAR | [文献数据] | [文献数据] | [预测分数待填充] |

**Biological interpretation:** [分析待填充]

**Note:** Chemo and checkpoint inhibitor responses are based on published literature (cite specific studies). circRNA therapy predictions are computational outputs pending wet lab validation.

---

### Case 3: Sequence optimization evolution

**Input:** [起始序列描述待填充]

**Process:** [代数] generations of multi-objective optimization

**Results (predicted):**
| Generation | Stability | Translation | Immune evasion | Delivery |
|------------|-----------|-------------|----------------|----------|
| 0 | [分数] | [分数] | [分数] | [分数] |
| [中间代] | [分数] | [分数] | [分数] | [分数] |
| [最终代] | [分数] | [分数] | [分数] | [分数] |

**Sequence changes:**
- GC content: [变化待填充]
- IRES motifs: [数量变化待填充]
- dsRNA fraction: [变化待填充]
- BSJ stability: [变化待填充]

**Validation status:** [实验验证结果待填充]

---

### Accessibility and open science

Confluencia is open source under MIT license. We deliberately chose MIT rather than a more restrictive license because we want other iGEM teams to use, modify, and redistribute the code without legal barriers.

**What is available now:**
- Full source code on GitHub
- Installation documentation
- API documentation with examples
- Three-tier backend system (works on any hardware)

**What will be released after validation:**
- Neural network model weights for TorusFold
- Pre-trained immunogenicity scoring models
- Benchmark datasets for reproducibility testing

We are not releasing model weights until wet lab validation confirms prediction accuracy. This is a scientific integrity decision, not a proprietary one. Once validation is complete, all weights will be public.

---

## Usability

### Event-driven architecture rationale

Conventional bioinformatics tools execute as single-run scripts producing static outputs. Confluencia implements an EventBus architecture enabling real-time simulation responsiveness.

EventBus provides two operational advantages. New modules can be integrated without modification of existing code; a toxicity model can subscribe to relevant events independently. Second, simulation replay is enabled through the EventBus timestamp log, permitting retrospective analysis of event sequences.

The event vocabulary spans the simulation lifecycle: tumor biology events (growth, heterogeneity, angiogenesis, metastasis), microenvironment events (immune dynamics, fibroblast activation, immune evasion), treatment events (drug administration, pharmacokinetic updates, resistance emergence), circRNA subsystem events (immune evaluation, structure prediction, sequence evolution), and clinical outcome events (RECIST evaluation, survival updates, toxicity grading). Forty event types are defined.

### Claude Code skill integration

Confluencia is available as an installable skill for Claude Code (Anthropic's command-line interface).Just by installing our release code and load skill,Users can easily execute simulations, evaluate circRNA sequences, and retrieve immunogenicity scores through natural language commands without writing Python code.

Example interaction:

```
[TBD]
```

The skill encapsulates the Python API, managing configuration, logging, and output formatting.

### Confluencia Studio web interface

A web interface (Confluencia Studio) is under development for users without Python programming experience. The interface Drow on R,which is widely used by most of Bioinformatics analyst​,operates locally without cloud dependencies and presents workflows through form-based input.

Studio provides three primary functions:

1. Sequence evaluation: circRNA sequence input generates immunogenicity scores and structure predictions
2. Pharmacokinetic simulation: dose and modification parameters produce concentration-time curve visualizations
3. Tumor simulation: subtype and treatment selection enables 365-day simulation with tumor volume and immune cell count tracking

Outputs are exportable as CSV or JSON formats.

### Confluencia Hub: A shared resource for the iGEM community

Confluencia Hub is not just a local database. It is designed as a community infrastructure for circRNA therapeutic development.

**What Hub stores:**

- circRNA sequence data (FASTA format, synthesis-ready)
- Immunogenicity pathway scores (RIG-I, TLR7, TLR8, PKR)
- Structure prediction results (when TorusFold executed)
- Pharmacokinetic parameters (when CirculaPK executed)
- Target annotations (TNBC subtype, pathway, drug combination)
- Wet lab validation results (when available)
- Version history tracking sequence evolution

**How iGEM teams can contribute:**

Any iGEM team working on circRNA projects can upload their designed sequences to Hub. Even if a team does not have wet lab validation data, their computational predictions contribute to the collective knowledge base. Over time, Hub accumulates sequences from multiple teams, each with different targets and experimental contexts.

**Why this matters for data scarcity:**

The fundamental limitation we faced in development was the lack of circRNA structural data. There are maybe 12 published circRNA 3D structures. This is insufficient to train deep learning models. But if 100 iGEM teams each upload 5-10 validated sequences, we would have 500-1000 data points. This is enough to train a diffusion model.

**Federated learning for data privacy:**

Some teams may not want to share their proprietary sequences publicly. Hub supports federated learning: a team can contribute model gradients (not raw sequences) to a shared model update. The central model improves from distributed training, but individual team data remains private. This approach is used in medical AI where patient data cannot be shared, and we are adapting it for circRNA design.

**From "adapting to data scarcity" to "solving data scarcity":**

Our physics-based structure prediction was a workaround for the absence of training data. It is not ideal. With Hub and federated learning, the community can collectively generate the data needed to train better models. This transforms the problem from "we lack data, so we use physics" to "the community generates data, so we can use deep learning."

**Available for all iGEM teams:**

We explicitly invite all iGEM teams working on RNA therapeutics to use Confluencia and contribute to Hub. The software is free, open source, and designed for diverse hardware environments. Teams can run predictions locally, validate in their own wet labs, and upload results to Hub. The collective knowledge base benefits everyone.

Hub operates on a local SQLite database by default. For team collaboration, it can sync to a shared server. For federated learning, gradient aggregation servers are planned for the next release cycle.

### Graphical dashboard

The simulation dashboard provides real-time visualization across four display panels:

1. Tumor panel: volume trajectory, clonal composition, metastatic site distribution
2. Microenvironment panel: CD8, Treg, NK cell counts, cytokine concentrations, fibroblast activation status
3. Treatment panel: drug concentration, response biomarkers, resistance indicators
4. circRNA panel: expression magnitude, immune activation scores, predicted protein output

Display updates occur in response to event triggers. The interface permits simulation pause, manual event injection (e.g., simulating unexpected dosing), and observation of system response dynamics.

---

## Available for iGEM: How your team can use Confluencia

**We built this for you.**

Confluencia is explicitly designed for iGEM teams working on RNA therapeutics, synthetic biology, and cancer-related projects. Here is what your team can do:

| What you need | What Confluencia provides |
|---------------|---------------------------|
| A circRNA sequence to test | Immunogenicity scores across 4 pathways, structure prediction, PK simulation |
| No Python expertise | Confluencia Studio web interface (local, no cloud) |
| Limited compute resources | Three-tier backend: works on hospital intranet, laptop, or GPU cluster |
| Wet lab validation data | Upload to Hub, contribute to collective knowledge base |
| Proprietary sequences | Federated learning: contribute gradients, keep sequences private |

**Getting started:**

```bash
# 1. Clone the repository
git clone https://github.com/confluencia-3-0/confluencia.git

# 2. Install (Tier 2 works with pip alone, no conda needed)
pip install -r requirements.txt

# 3. Run your first prediction
python -m confluencia --sequence "AUGCGCUAUAGC..." --output scores.csv
```

**What we ask from iGEM teams:**

1. Use the tool for your circRNA projects
2. If you validate sequences in wet lab, upload results to Hub
3. If you find bugs or have feature requests, open a GitHub issue
4. If you want to collaborate, contact us via the iGEM team directory

**What we will do for iGEM teams:**

1. Maintain the software throughout the competition cycle
2. Release model weights after our validation completes
3. Provide documentation and support for common use cases
4. Acknowledge contributing teams in our Giant Jamboree presentation

The goal is for Confluencia to become a shared infrastructure for the iGEM RNA therapeutics community. Your team's contributions make the tool better for everyone.

---

## Future development

Confluencia Studio web interface development is ongoing. Neural network model weights will be released upon completion of laboratory validation. Confluencia Hub currently operates as a local prototype, with cloud synchronization planned for subsequent release.

Planned extensions include: application to additional solid tumor types beyond TNBC, clinical trial simulation module development, and validation against published clinical datasets.

---

<details>
<summary><b>Behind the names</b> (click to expand)</summary>

**Confluencia** — Spanish for "confluence." We chose this name because the platform integrates three biological scales (molecular, cellular, tissue) and three computational approaches (physics-based structure, deep learning embeddings, heuristic rules) into one coherent system. Different data streams flow together.

**TorusFold** — A torus is the mathematical shape of a circle. Standard RNA structure predictors treat sequences as linear strings, where position 0 and position L are far apart. circRNA is a closed loop, so these positions are adjacent. Our Torus Positional Encoding makes them mathematically equivalent, preserving circular topology throughout structure prediction.

**CirculaPK** — The name combines "Circular RNA" with "PK" (pharmacokinetics). The model tracks circRNA through six physiological compartments: injection depot, bloodstream, tumor tissue, endosome, cytoplasm, and protein expression. Each compartment has its own rate constants.

</details>

---

## References

1. Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596(7873):583-589.

2. Abramson J, Adler J, Dunger J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature. 2024.

3. Lehmann BD, Bauer JA, Chen X, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. J Clin Invest. 2011;121(7):2750-2767.

4. Liu CX, Chen LL. Circular RNAs: Characterization, cellular roles, and applications. Cell. 2022;185(23):4231-4250.

5. Wesselhoeft RA, Kowalski PS, Anderson DG. Engineering circular RNA for potent and stable translation in eukaryotic cells. Nat Commun. 2018;9(1):2629.

---

## Team

[Team member names and roles to be completed]

---

## Repository

GitHub repository URL and MIT license information.

---

## Contact

Correspondence email and iGEM team wiki page.