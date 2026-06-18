# Confluencia 3.0

## Overview

Confluencia 3.0 links two problems that usually get studied separately: Triple-Negative Breast Cancer (TNBC) tumor modeling and circular RNA (circRNA) therapeutic design. The idea is that if you are designing a circRNA therapy for cancer, you should be able to simulate how it behaves in an actual tumor microenvironment, not just predict its structure in isolation.

TNBC makes up about 15-20% of breast cancers. It lacks ER, PR, and HER2 receptors, so hormone therapies do not work. circRNA is a relatively new therapeutic format that has some advantages over linear RNA, mainly better stability and the potential for longer expression windows.

The gap we noticed: existing tools either predict circRNA structure or simulate tumor behavior, but nothing connects the two. If your circRNA triggers an immune response that changes how the tumor responds to treatment, you would not know that from running the tools separately.

### What the software does

Confluencia runs TNBC simulation and circRNA design in a coupled loop:

- TorusFold predicts circRNA 3D structure using deep learning (inspired by AlphaFold3)
- CirculaPK models pharmacokinetics from injection through protein expression (six compartments)
- REINFORCE Evolution optimizes sequences against multiple objectives at once
- TNBC Simulacrum simulates tumor growth, immune dynamics, and treatment response

---

## Description

### Module breakdown

| Module | What it does | The main trick |
|--------|--------------|----------------|
| TorusFold | 3D structure for circRNA | Torus Positional Encoding makes position 0 and L mathematically adjacent, matching circular topology |
| Immune Sensing | Scores immunogenicity across four pathways | RIG-I, TLR7, TLR8, PKR each get separate predictions rather than one aggregate score |
| CirculaPK | PK simulation | Six compartments track circRNA from injection depot through endosomal escape to protein output |
| Sequence Evolution | Multi-objective optimization | Pareto front generation lets you trade off stability vs translation vs immune safety vs delivery |
| TNBC Simulacrum | Tumor simulation | Four molecular subtypes (BLIS, IM, M, LAR) each have different growth rates and immune profiles |

### TNBC subtypes

The simulator runs four TNBC subtypes based on Lehmann et al. 2011 classification:

- BLIS: Basal-like, immune suppressed, fast growth
- IM: Immunomodulatory, lots of T cells in the tumor
- M: Mesenchymal, EMT signature, stromal
- LAR: Luminal androgen receptor driven, responds to AR targeting

### Backend fallback chain

Not every lab has GPU access or reliable internet. Confluencia drops through three backend tiers when hardware or connectivity fails:

```
ESM2 (Tier 0) → ViennaRNA (Tier 1) → Heuristic rules (Tier 2)
    GPU needed      CPU, local         pure Python, no deps
```

If you run it on a hospital intranet machine with no external API access, it still works. It just gives you less accurate predictions.

---

## Design

### Architecture sketch

The core is an EventBus that coordinates six subsystem managers. Each manager owns its slice of state and publishes events when things change. The circRNA manager talks to the tumor and TME managers through the bridge layer, not directly.

```
         EventBus (40 event types)
            │
    ┌───────┼───────┬───────┬───────┐
    │       │       │       │       │
 Tumor   TME   Treatment CircRNA Clinical
    │       │       │       │       │
    └───────┼───────┼───────┼───────┘
            │
      Confluencia Bridge (circRNA ↔ tumor coupling)
```

Event categories:

- Tumor biology: growth, heterogeneity, angiogenesis, metastasis, CSC updates
- Microenvironment: immune dynamics, fibroblast activation, endothelial changes, immune evasion
- Treatment: drug administration, PK updates, PD effects, resistance emergence
- circRNA: immune evaluation, structure prediction, sequence evolution, PK simulation
- Clinical: RECIST evaluation, survival updates, toxicity grading, subtype reclassification

### TorusFold

TorusFold takes a circRNA sequence and outputs structure coordinates plus confidence scores. The key difference from standard RNA structure predictors: it treats the sequence as circular from the start.

Standard positional encoding treats position 0 and position L as far apart. Torus Positional Encoding (TPE) makes them identical mathematically. That way the network learns that nucleotides near the back-splice junction can pair even though they are at opposite ends of the linear representation.

Architecture flow:

```
Sequence → TPE (periodic encoding) → ESM2 backbone (frozen) 
         → CircPairformer (triangle updates with circular distance bias) 
         → Structure head (four modes: simple MDS, diffusion, geometry solver, OpenMM refinement)
```

### Why physics-based, not deep learning?

AlphaFold works because there are thousands of protein structures in the PDB database. The model learned from real data. circRNA does not have that. There are maybe a dozen published circRNA 3D structures, and most of those are fragments, not full circles.

So we wrote a physics-based structure head (physics_b and physics_ba modes) that does not need training data. It takes the predicted pair probabilities from the neural net, converts them into geometric constraints (bond lengths, pairing distances, clash avoidance), and solves for coordinates using constraint propagation. No training required, just physics.

The deep learning modules (diffusion head, full CircPairformer stack) are in the code, but they are essentially dormant until we get data. We have the architecture ready. What we lack is the training set.

The reality: circRNA synthesis is still expensive. A single 500nt circRNA costs hundreds of dollars from commercial providers. No lab is going to synthesize 10,000 variants and crystallize them for a training dataset. But once circRNA manufacturing costs drop, or once someone publishes a large structural dataset, the hidden code flips on. The diffusion model wakes up, trains overnight, and suddenly you get AlphaFold-quality circRNA predictions.

Until then, physics_ba mode (constraint solver plus OpenMM molecular dynamics refinement) is the best we can do. It is not as accurate as a trained model would be, but it runs on any circRNA sequence without needing experimental data.

Side note: we wrote the architecture to be CASP-ready. The torus encoding and circular distance matrix are genuine innovations that do not exist in standard structure predictors. AlphaFold debuted in Nature 2021. If Jilin University ever funds a Circ-CASP category... well, we have ambitions too.

### CirculaPK PK model

Six compartments:

```
Depot → Blood → Tissue
              ↓
         Endosome → Cytoplasm → Protein
```

Each arrow has a rate constant. The model outputs AUC, peak concentration, half-life, and predicted protein expression level.

---

## Implementation

### Tech stack

Python 3.10+, PyTorch for the neural nets, ESM2 from Meta for sequence embeddings (frozen weights). ViennaRNA for thermodynamic folding when GPU is unavailable. OpenMM for molecular dynamics refinement in the physics_ba mode, but that is optional. Configuration through dataclasses and YAML.

### Three algorithms worth explaining

**Torus Positional Encoding**

```python
omega = 2 * pi / length
# Higher harmonics capture finer positional distinctions
pe[2*i] = sin(omega * (2**i) * position)
pe[2*i+1] = cos(omega * (2**i) * position)
# Result: position 0 and position L produce identical vectors
```

**Circular Distance Matrix**

```python
d_circ(i, j) = min(|i - j|, L - |i - j|)
# Linear distance works for linear RNA
# Circular distance works for circRNA where ends are connected
```

**Multi-objective reward**

```python
reward = 0.35 * stability + 0.30 * translation + 0.25 * immune_evasion + 0.10 * delivery
# Weighting can change depending on what you are optimizing for
```

### File layout

```
confluencia_3_0/
  main.py
  core/
    agent.py            # Main simulation loop
    event_bus.py        # Pub/sub core
    state_schema.py     # State keys (~200)
    events.py           # 40 event types
    tumor/              # Growth, CSC, angiogenesis, metastasis
    tme/                # Immune cells, fibroblasts, evasion
    treatment/          # Chemo, immunotherapy, targeted, radio
    circrna/
      torusfold/        # Neural net modules
      immune_sensing.py # Four pathway scoring
      torusfold_scorer.py # DL output → objectives
    pk/
      rnactm.py         # Six-compartment ODE solver
    evolution/          # Sequence optimization
    confluencia/        # Bridge modules (circRNA ↔ tumor)
```

---

## Usage

### Install

```bash
git clone https://github.com/your-team/confluencia-3.0.git
cd confluencia-3.0
pip install -r requirements.txt

# Optional but recommended
conda install -c bioconda viennarna    # Thermodynamic folding
conda install -c conda-forge openmm    # MD refinement (physics_ba mode)
```

### Run

```bash
python -m confluencia_3_0 --subtype BLIS --steps 365
python -m confluencia_3_0 --circrna-backend vienna
python -m confluencia_3_0 --structure-mode diffusion
```

### Python examples

Evaluate a sequence:

```python
from confluencia_3_0.core.circrna.torusfold_scorer import quick_score

result = quick_score("AUGCGC...", modification="m6A")
print(result['stability'], result['immune_evasion'])
```

Evolve a better sequence:

```python
from confluencia_3_0.core.evolution.cirrna_evolution import evolve_cirrna

df, artifacts = evolve_cirrna(seed_seq="...", generations=50)
print(artifacts.best_sequence)
```

Run PK simulation:

```python
from confluencia_3_0.core.pk.rnactm import simulate_rna_ctm

curve = simulate_rna_ctm(dose=1.0, params=infer_params("m6A", "LNP"))
print(f"Half-life: {curve['rna_half_life']} hours")
```

---

## Demonstration

### Case 1: Immunogenicity check

Input sequence (500 nt) had GGGG and UUGU motifs.

Output: RIG-I score 0.72 (high risk), PKR 0.58, overall 0.48.

Recommendation: Add m6A at positions 45, 78, 156. That dropped RIG-I to 0.31 without hurting translation.

### Case 2: Subtype comparison

| Subtype | Chemo response | checkpoint inhibitor response | circRNA predicted effect |
|--------|----------------|------------------------------|-------------------------|
| BLIS | 0.78 | 0.32 | 0.65 |
| IM | 0.45 | 0.82 | 0.71 |
| M | 0.38 | 0.55 | 0.58 |
| LAR | 0.52 | 0.48 | 0.73 |

IM subtype responds better to checkpoint inhibitors. BLIS needs chemotherapy. LAR gets decent circRNA effect scores, probably because the AR pathway is easier to target with miRNA sponges.

### Case 3: Sequence evolution

Random 800nt sequence optimized over 50 generations:

```
Gen 0:   stability 0.42, translation 0.35, immune 0.28
Gen 50:  stability 0.79, translation 0.72, immune 0.68

Changes: GC went to 52%, added 3 IRES motifs, dsRNA fraction dropped from 45% to 28%
```

---

## Epitope 2.0: Predicting which peptides get presented

### What it does

After your circRNA gets translated into protein, the protein gets chopped into peptides. Some of those peptides end up on MHC class I molecules, where they get shown to T cells. That is how the immune system knows to attack. Epitope 2.0 predicts which peptide sequences will make effective vaccines.

This is useful for circRNA vaccine design. If you are engineering a circRNA to express a tumor antigen, you want the resulting peptides to be presented well. Epitope 2.0 tells you which sequences work before you spend money synthesizing them.

### How it works

The module takes an 8-11mer peptide sequence and some experimental context (dose, frequency, treatment time) and predicts the immunogenic efficacy score. It uses three main tricks:

| Trick | What it does | Why it matters |
|-------|--------------|----------------|
| Mamba3Lite encoder | State-space model that reads the peptide sequence | Captures position-specific patterns better than RNNs |
| MHC pseudo-sequence | 34-position encoding of the MHC binding groove | Different HLA alleles have different preferences |
| MOE ensemble | Seven models vote, weighted by how good each is | Reduces variance, handles small datasets better |

### Why we do not use ESM-2

ESM-2 is a huge protein language model from Meta. We tried it. It got AUC=0.537 for MHC binding prediction, which is worse than a coin flip. The problem: ESM-2 was trained on full proteins. Short peptides (8-11 amino acids) are too short for the model to learn anything useful. Mean pooling (averaging across positions) destroys the anchor position signals that determine MHC binding.

So we went back to the NetMHCpan approach: encode the MHC binding groove as a 34-position pseudo-sequence, then use traditional features plus allele-specific encoding. That got AUC=0.917. Lesson learned: large language models do not transfer to short peptide immunogenicity.

### Experimental results

| Test | Score | Method |
|------|-------|--------|
| 288K IEDB peptides | AUC 0.80 | HGB + MHC features |
| MOE ensemble | MAE 0.389 | 39% better than single model |
| MHC binding prediction | AUC 0.917 | Pseudo-sequence encoding |
| ESM-2 baseline | AUC 0.537 | Failed approach |

### How it connects to the main platform

There is a bridge module called `EpitopeBridge` that takes the protein output from your circRNA design and feeds it to Epitope 2.0. The workflow:

1. Design circRNA sequence with Confluencia
2. CircRNA gets translated in silico
3. Resulting peptides get scored by Epitope 2.0
4. High-scoring peptides are vaccine candidates

You can also use Epitope 2.0 standalone. Feed it a CSV of peptide sequences with some context columns, and it returns predictions with sensitivity analysis showing which features mattered.

### Windows note

Mamba-ssm (the fast CUDA version) does not work on Windows. The module falls back to a pure PyTorch implementation that runs on CPU. Same architecture, just slower. If you are on Linux with a GPU, you get the fast version automatically.

---

## Usability

### Why event-driven matters

Most bioinformatics tools are scripts you run once and get a static output. Confluencia is built around an EventBus, which means the simulation responds to events as they happen.

When a drug gets administered, that triggers `DRUG_ADMINISTERED`. The PK module hears that event and starts computing concentration curves. The tumor module hears it and updates growth inhibition. The TME module hears it and adjusts immune cell recruitment. Everything happens in response to the same event, without any module directly calling another.

This has two practical benefits. First, you can plug in new modules without touching existing code. A new toxicity model just subscribes to the same events. Second, you can replay the simulation. The EventBus logs every event with timestamps, so you can trace back exactly what happened and when.

The event vocabulary covers the whole simulation lifecycle: tumor biology (growth, heterogeneity, angiogenesis, metastasis), microenvironment (immune dynamics, fibroblast activation, immune evasion), treatment (drug administration, PK updates, resistance emergence), circRNA subsystem (immune evaluation, structure prediction, sequence evolution), and clinical outcomes (RECIST evaluation, survival updates, toxicity grading). About 40 event types total.

### Claude Code skill

If you use Claude Code (Anthropic's CLI), Confluencia comes as an installable skill. That means you can ask Claude to run simulations, evaluate circRNA sequences, or check immunogenicity scores directly from your terminal without writing Python code.

Example:

```
User: Evaluate this circRNA sequence for immune safety
Claude: [calls confluencia skill]
Result: RIG-I 0.31, TLR7 0.22, PKR 0.41, overall safe profile
```

The skill wraps the Python API and handles configuration, logging, and output formatting. You get the same results as running the code, but through natural language commands.

### Confluencia Studio

For wet lab teams that do not want to touch Python, we are building a web interface called Confluencia Studio. It runs locally (no cloud dependency) and exposes the main workflows through forms and buttons.

Studio handles three main tasks:

1. Sequence evaluation: paste a circRNA sequence, get immunogenicity scores and structure predictions
2. PK simulation: input dose and modification type, see concentration-time curves rendered as plots
3. Tumor simulation: pick subtype and treatment, run a 365-day simulation, watch tumor volume and immune cell counts update

The output is downloadable as CSV or JSON. No coding required.

### Confluencia Hub

Hub is where you store and share circRNA designs. Think of it like a GitHub for circRNA sequences. Each entry has:

- The sequence itself
- Immunogenicity scores (four pathways)
- Structure prediction results (if TorusFold ran)
- PK parameters (if CirculaPK ran)
- Tags for what it targets (TNBC subtype, specific pathway, drug combination)
- A version history showing how the sequence evolved

You can clone an existing design, modify it, and push your own version. Other teams can pull and test it in their own simulations. The goal is to build a shared library of validated circRNA therapeutics that any iGEM team can use as a starting point.

Hub runs as a local SQLite database by default. If your team wants to share across multiple machines, it can sync to a shared server.

### GUI dashboard

The simulation dashboard shows live updates while the model runs. You get four panels:

1. Tumor panel: volume curve, clone composition pie chart, metastatic sites
2. TME panel: CD8/Treg/NK cell counts over time, cytokine levels, fibroblast activation
3. Treatment panel: drug concentration, response markers, resistance flags
4. circRNA panel: expression level, immune activation scores, predicted protein output

Everything updates in real time as events fire. You can pause the simulation, inject an event manually (like simulating an unexpected dose), and watch how the system responds.

---

## What is next

The web interface (Studio) is in active development. Neural net weights will be released once we finish validation against our lab data. Hub is currently a local prototype, with cloud sync planned for later this year.

Longer term: expand beyond TNBC to other solid tumors, add a clinical trial simulation module, and validate against published datasets.

---

## References

1. Jumper et al. 2021. AlphaFold. Nature 596.
2. Abramson et al. 2024. AlphaFold 3. Nature.
3. Lehmann et al. 2011. TNBC subtypes. JCI 121.
4. Liu & Chen 2022. Circular RNAs review. Cell 185.
5. Wesselhoeft et al. 2018. circRNA engineering. Nat Commun 9.

---

## Team

Fill in actual names and roles.

---

## Repository

GitHub link and MIT license.

---

## Contact

Email and iGEM team page.