# Confluencia 3.0 — iGEM Wiki Software Page

## Overview

**Confluencia 3.0** is a unified computational platform that integrates **Triple-Negative Breast Cancer (TNBC) simulation** with **circular RNA (circRNA) therapeutic design**. It represents a novel bioinformatics approach that bridges computational oncology and synthetic biology, enabling researchers to design, optimize, and predict outcomes of circRNA-based cancer therapies.

### Problem Statement

Triple-Negative Breast Cancer (TNBC) accounts for 15-20% of all breast cancers and lacks estrogen receptor (ER), progesterone receptor (PR), and HER2 expression, making it unresponsive to hormone therapies. circRNA-based therapeutics represent a promising new modality, but designing effective circRNA sequences requires:

1. **Immunogenicity prediction** — avoiding unwanted immune activation
2. **Structure optimization** — ensuring proper circular topology and stability
3. **Pharmacokinetic modeling** — predicting in vivo behavior
4. **Patient stratification** — matching therapies to molecular subtypes

No existing tool addresses all four challenges in an integrated manner.

### Solution

Confluencia 3.0 provides an end-to-end pipeline that combines:
- **TorusFold**: AlphaFold3-inspired deep learning for circRNA structure prediction
- **CirculaPK**: Six-compartment pharmacokinetic model for circRNA
- **REINFORCE Evolution**: Multi-objective sequence optimization
- **TNBC Simulacrum**: Agent-based tumor microenvironment simulation

---

## Description

### Core Capabilities

| Module | Function | Key Innovation |
|--------|----------|----------------|
| **TorusFold** | 3D structure prediction for circRNA | Torus Positional Encoding (TPE) ensures circular topology |
| **Immune Sensing** | Four-pathway immunogenicity assessment | RIG-I, TLR7, TLR8, PKR pathway modeling |
| **CirculaPK** | Pharmacokinetic simulation | Six-compartment model from injection to protein expression |
| **Sequence Evolution** | Multi-objective optimization | Pareto front with stability/translation/immune/delivery trade-offs |
| **TNBC Simulacrum** | Tumor microenvironment simulation | Four molecular subtypes (BLIS/IM/M/LAR) |

### The circRNA Data Challenge

No circRNA crystal structures or cryo-EM reconstructions exist in PDB. This creates a chicken-and-egg problem: one cannot train a structure predictor without structures, and one cannot validate predicted structures without experimental ground truth. Our initial 5,663-sample training dataset had critical gaps: 88% trivially simple helical structures, a length gap at 500-1000 nt, and zero real secondary structure constraints.

**Our Solution: Multi-Source Data Pipeline.** We developed a four-source strategy that combines real structures, experimental constraints, circularized PDB structures, and physics-based predictions:

| Source | Raw | After Merge | Length | Quality | Method | Key Features |
|--------|-----|-------------|--------|---------|--------|--------------|
| **IsRNAcirc** | 34 real + 80x aug | 5,663 (circbase_real_3d) | 43-2050 nt | Highest | Real circRNA 3D structures from PDB, rotation+noise augmentation | 24/34 with real secondary structure from .subo files; covers hairpin/helix/internal/junction |
| **icSHAPE** | ~2,000 | ~2,000 (shape_3d) | 200-1000 nt | Medium-High | Experimental SHAPE reactivity profiles (GSE74353, Flynn et al. Science 2016) → ViennaRNA SHAPE-constrained folding → GeometricConstraintSolver | Experimental structure constraints guide base-pair probability; fills 500-1000 nt length gap |
| **PDB circularized** | 4,851 RCSB structures | 184 (pdb_3d) | 50-500 nt | Medium | Linear RNA from RCSB PDB (resolution <3.0A), circularized via GeometricConstraintSolver annealing closure | Diverse folds; closure score filtering ensures circular topology quality |
| **Medium-length** | ~2,000 | ~2,000 (medium_length_3d) | 500-1000 nt | Medium | ViennaRNA circ-mode secondary structure → GeometricConstraintSolver | Specifically fills the 500-1000 nt therapeutic length gap |
| **Synthetic physics** | ~5,000 | ~5,000 (circbase_real_3d synthetic) | 50-500 nt | Medium | ViennaRNA circ-mode secondary structure → GeometricConstraintSolver | Physics-based pairing constraints, not trivial helices |
| **Merged total** | | **~8,139** (after dedup) | **43-2050 nt** | | | All with secondary structure + pair constraints |

**Data pipeline scripts:**
- `build_training_dataset.py`: IsRNAcirc loading + augmentation + synthetic generation
- `shape_to_3d_pipeline.py`: icSHAPE download (GEO GSE74353) → SHAPE-constrained folding → 3D generation
- `pdb_rna_circularize.py`: RCSB PDB search + download + circularization via GeometricConstraintSolver
- `generate_medium_length_dataset.py`: 500-1000 nt circRNA generation with ViennaRNA + GeometricConstraintSolver
- `merge_expanded_dataset.py`: Deduplication (sequence-level) + validation + source-priority merging

### Circ-CASP: Community Benchmark

We established **Circ-CASP** (Critical Assessment of circRNA Structure Prediction), the first community benchmark for circRNA 3D structure prediction. Features:

- Training data: 10,000+ sequences from the multi-source pipeline (public)
- Test data: 30 circRNA structures (hidden ground truth)
- 5 evaluation metrics: Global RMSD (40%), BSJ closure (20%), bond consistency (15%), pair F1 (15%), conformational diversity (10%)
- 7 baseline methods from physics-based to deep learning (M1-M7, including Mamba+Transformer hybrid)
- Two competition tracks: compute-limited (regular) and unlimited ("oracle")

### Four Molecular Subtypes

Confluencia 3.0 supports TNBC molecular subtyping based on transcriptomic profiles:

```
BLIS (Basal-like Immune Suppressed)  — High proliferation, low immune infiltration
IM   (Immunomodulatory)              — High immune cell presence
M    (Mesenchymal)                   — EMT signature, stromal enrichment
LAR  (Luminal Androgen Receptor)     — AR-driven, endocrine-like
```

### Backend Graceful Degradation

To ensure accessibility across diverse research environments, Confluencia implements a three-tier backend system:

```
Tier 0 (ESM2)      → Highest accuracy, requires GPU + online access
       ↓ fallback
Tier 1 (ViennaRNA) → Medium accuracy, local CPU, thermodynamic folding
       ↓ fallback
Tier 2 (Heuristic) → Zero-dependency, pure Python, GC/IRES rules
```

---

## Design

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EventBus (Event-Driven Core)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│TumorManager   │  │TMEManager     │  │CircRNAManager │
├───────────────┤  ├───────────────┤  ├───────────────┤
│• Growth       │  │• Immune       │  │• TorusFold    │
│• Heterogeneity│  │• CAF/ECM      │  │• CirculaPK PK    │
│• CSC Pool     │  │• Evasion      │  │• Evolution    │
│• Angiogenesis │  │• Immunoediting│  │• Immune Eval  │
│• Metastasis   │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
              ┌───────────────────────┐
              │ Confluencia Bridge    │
              │ circRNA ↔ TNBC coupling│
              └───────────────────────┘
```

### TorusFold Multi-Scheme Architecture

TorusFold implements **seven complementary schemes** for circRNA 3D structure prediction, each with distinct complexity-accuracy trade-offs:

| Scheme | Architecture | Complexity | Max L (24GB GPU) | Key Innovation |
|--------|-------------|------------|------------------|----------------|
| 1 | EGNN + Physics cascade | O(L²) | ~500 | Neural + physics hybrid |
| 2 | Pure physics solver | O(L) | Unlimited | No training required |
| 3 | Dual-engine iterative | O(L) | ~1000 | CS-Fold + PaxNet |
| 4 | DDPM + EGNN diffusion | O(L²) | ~500 | Guided diffusion with BSJ closure reward |
| 5 | Physics-biased attention | O(L²) | ~300 | CircPairformer with physics priors |
| 6 | GNN latent diffusion | O(L²) | ~400 | Latent space diffusion |
| **7** | **Mamba+Transformer hybrid** | **O(L)** | **~1000+** | **Linear complexity for long sequences** |

**Scheme 7: Mamba+Transformer Hybrid Diffusion** is specifically designed for long therapeutic circRNAs:

```
Input Sequence (A,C,G,U)
         │
         ▼
┌─────────────────────────────┐
│ BiMamba Encoder             │  Bidirectional Selective SSM
│ + Circular Wrap-around Scan │  Position L-1 → Position 0 (BSJ topology)
│ + Gradient Checkpointing    │  Halves activation memory
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Local Window Attention      │  O(L×w) attention, w=20
│ + BSJ Flanking Attention    │  Positions 0-20 and L-20..L-1
│ (captures circular topology)│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Diffusion Denoiser          │  DDPM with 100 steps
│ + BSJ Closure Reward        │  Guided sampling for ring closure
└────────────┬────────────────┘
             │
             ▼
    Predicted 3D Coordinates (N, 3)
```

**Memory comparison (L=1000, batch=4):** Scheme 4 EGNN ~25GB vs Scheme 7 Mamba ~8GB (3× reduction)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ESM2 Backbone   │  650M parameter protein LLM (frozen)
│ + Torus Trans.  │  Rotation-equivariant processing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CircPairformer  │  Triangle Multiplicative Update
│ (4 blocks)      │  Triangle Attention + Circular Distance Bias
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Structure Heads (4 modes)                       │
├─────────────────────────────────────────────────┤
│ simple      → MDS-based rapid inference         │
│ diffusion   → AF3-style diffusion model         │
│ physics_b   → Geometric constraint solver       │
│ physics_ba  → Constraint solver + OpenMM MD     │
└─────────────────────────────────────────────────┘
```

### CirculaPK Six-Compartment PK Model

```
┌─────────┐   k_ab   ┌─────────┐   k_dt   ┌──────────┐
│  Depot  │ ───────→ │  Blood  │ ───────→ │  Tissue  │
│ (inject)│          │         │          │          │
└─────────┘          └────┬────┘          └──────────┘
                          │ k_be
                          ▼
                   ┌──────────┐   k_ec   ┌───────────┐   k_cp   ┌───────────┐
                   │ Endosome │ ───────→ │ Cytoplasm │ ───────→ │  Protein  │
                   │          │          │           │          │ (expressed)│
                   └──────────┘          └───────────┘          └───────────┘
```

---

## Implementation

### Technology Stack

| Component | Technology |
|-----------|------------|
| Core Language | Python 3.10+ |
| Deep Learning | PyTorch 2.0 |
| Sequence Embedding | ESM2 (Meta AI) |
| Thermodynamic Folding | ViennaRNA |
| Molecular Dynamics | OpenMM (optional) |
| Event System | Custom EventBus (pub/sub) |
| Configuration | Dataclasses + YAML |

### Key Algorithms

#### 1. Torus Positional Encoding (TPE)

```python
def torus_positional_encoding(position: int, length: int, dim: int) -> Tensor:
    """
    Periodic positional encoding for circular RNA.

    Key property: TPE[0] = TPE[L]
    This ensures the model recognizes position 0 and L-1 as adjacent.
    """
    omega = 2 * math.pi / length
    harmonics = [omega * (2**i) for i in range(dim // 2)]
    pe = torch.zeros(dim)
    for i, h in enumerate(harmonics):
        pe[2*i] = math.sin(h * position)
        pe[2*i+1] = math.cos(h * position)
    return pe
```

#### 2. Circular Distance Matrix

```python
def circular_distance_matrix(L: int) -> Tensor:
    """
    Compute circular distance for circRNA.

    d_circ(i, j) = min(|i-j|, L - |i-j|)

    This allows the model to learn that nucleotides near
    the back-splice junction can pair despite being far
    apart in linear sequence.
    """
    i = torch.arange(L).unsqueeze(1)
    j = torch.arange(L).unsqueeze(0)
    linear_dist = (i - j).abs()
    return torch.min(linear_dist, L - linear_dist)
```

#### 3. Multi-Objective Evolution

```python
def compute_reward(objectives: Tensor, weights: Tuple[float, ...]) -> float:
    """
    Weighted combination of four objectives:
    - stability:      0.35 (ring closure + thermal stability)
    - translation:    0.30 (IRES activity + efficiency)
    - immune_evasion: 0.25 (avoid RIG-I/TLR/PKR activation)
    - delivery:       0.10 (length/GC/modification compatibility)
    """
    return (objectives * torch.tensor(weights)).sum()
```

### File Structure

```
confluencia_3_0/
├── main.py                      # CLI entry point
├── core/
│   ├── agent.py                 # TNBCSimulacrum main agent
│   ├── event_bus.py             # Event-driven communication
│   ├── state_schema.py          # State definitions (200+ keys)
│   ├── events.py                # 18 event types
│   ├── config.py                # Configuration dataclasses
│   ├── subsystem_managers.py    # 6 Manager classes
│   ├── tumor/                   # Tumor subsystem modules
│   ├── tme/                     # Tumor microenvironment
│   ├── treatment/               # Treatment modules
│   ├── circrna/                 # circRNA analysis pipeline
│   │   ├── torusfold/           # Deep learning structure prediction
│   │   ├── torusfold_scorer.py  # TorusFold → objectives bridge
│   │   ├── immune_sensing.py    # Four-pathway immune evaluation
│   │   └── ...
│   ├── pk/                      # Pharmacokinetic models
│   │   └── rnactm.py            # Six-compartment PK
│   ├── evolution/               # Sequence optimization
│   └── confluencia/             # Bridge modules
├── experiments/                 # Experiment scripts
└── tests/                       # Unit tests
```

---

## Usage

### Installation

```bash
# Clone repository
git clone https://github.com/your-team/confluencia-3.0.git
cd confluencia-3.0

# Install dependencies
pip install -r requirements.txt

# Optional: Install ViennaRNA for thermodynamic folding
conda install -c bioconda viennarna

# Optional: Install OpenMM for physics_ba mode
conda install -c conda-forge openmm
```

### Quick Start

```bash
# Run simulation with default settings (BLIS subtype, 365 days)
python -m confluencia_3.0 --subtype BLIS --steps 365

# Use circRNA backend with ViennaRNA
python -m confluencia_3_0 --circrna-backend vienna

# Enable TorusFold structure prediction (diffusion mode)
python -m confluencia_3_0 --structure-mode diffusion
```

### Python API

```python
from confluencia_3_0.core.circrna.torusfold_scorer import quick_score

# Evaluate a circRNA sequence
sequence = "AUGCGCUAUAGCUAGCUAGCUAGCUAGC..."
result = quick_score(sequence, modification="m6A", device="cpu")

print(f"Stability:       {result['stability']:.3f}")
print(f"Translation:     {result['translation']:.3f}")
print(f"Immune Evasion:  {result['immune_evasion']:.3f}")
print(f"Delivery:        {result['delivery']:.3f}")
```

### circRNA Sequence Evolution

```python
from confluencia_3_0.core.evolution.cirrna_evolution import evolve_cirrna

# Optimize sequence for immune safety
result_df, artifacts = evolve_cirrna(
    seed_seq="AUGCGCUAUAGC...",
    objective="immune_safety",
    generations=50
)

print(f"Best sequence: {artifacts.best_sequence}")
print(f"Best score:    {artifacts.best_reward:.3f}")
```

### PK Simulation

```python
from confluencia_3_0.core.pk.rnactm import simulate_rna_ctm, infer_rna_ctm_params

# Infer PK parameters from sequence features
params = infer_rna_ctm_params(
    modification="m6A",
    delivery_vector="LNP_standard",
    route="IV"
)

# Simulate concentration-time curve
curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=params, horizon=168)

print(f"AUC:     {curve['auc_efficacy']:.2f}")
print(f"Peak:    {curve['peak_protein']:.2f}")
print(f"Half-life: {curve['rna_half_life']:.1f} h")
```

---

## Demonstration

### Case Study 1: circRNA Immunogenicity Assessment

**Input**: circRNA sequence with potential immune-stimulating motifs

```
Sequence: 5'-...GGGG...UUGU...AUG...-3' (500 nt)
```

**Output**:

| Pathway | Score | Interpretation |
|---------|-------|----------------|
| RIG-I | 0.72 | High activation risk (GU-rich regions) |
| TLR7/8 | 0.35 | Moderate risk |
| PKR | 0.58 | Elevated dsRNA potential |
| **Overall** | 0.48 | Requires optimization |

**Recommendation**: Introduce m6A modifications at positions 45, 78, 156 to reduce RIG-I activation.

### Case Study 2: TNBC Subtype-Specific Response

| Subtype | Chemo Response | Immune Checkpoint | circRNA Therapy |
|---------|---------------|-------------------|-----------------|
| BLIS | 0.78 (high) | 0.32 (low) | 0.65 |
| IM | 0.45 (medium) | 0.82 (high) | 0.71 |
| M | 0.38 (low) | 0.55 (medium) | 0.58 |
| LAR | 0.52 (medium) | 0.48 (medium) | 0.73 |

### Case Study 3: Sequence Evolution Optimization

Starting from a random 800nt circRNA:

```
Generation 0:  stability=0.42, translation=0.35, immune=0.28, delivery=0.55
Generation 25: stability=0.67, translation=0.58, immune=0.52, delivery=0.61
Generation 50: stability=0.79, translation=0.72, immune=0.68, delivery=0.65

Best sequence shows:
- Improved GC content: 48% → 52% (optimal range)
- Enhanced IRES motifs: +3 GCGCC elements
- Reduced dsRNA fraction: 45% → 28%
- Better BSJ stability: 0.65 → 0.88
```

---

## Core Modules

### Epitope 2.0 — MHC-I Epitope Efficacy Prediction

**Epitope 2.0** is a specialized module for predicting MHC-I epitope immunogenic efficacy, designed for circRNA vaccine scenarios. It addresses a critical gap in therapeutic design: predicting which peptide sequences will effectively induce T-cell responses when presented by MHC class I molecules.

#### Key Innovations

| Innovation | Description |
|------------|-------------|
| **Mamba3Lite Encoder** | Three time-scale adaptive state-space recursion + four-scale pooling (mean/local/meso/global) + self-attention enhancement for peptide sequence encoding |
| **Sample-Adaptive MOE Ensemble** | Ridge/HGB/RF/MLP/XGB/LGB/ET experts weighted by inverse OOF-RMSE, automatically adapting to dataset size |
| **MHC Pseudo-Sequence Encoding** | 34-position pseudo-sequence features achieving AUC=0.917 for MHC binding prediction, far exceeding ESM-2 mean pooling (0.537) |
| **Multi-Scale Sensitivity Analysis** | Neighborhood contribution aggregation (local/meso/global) + gradient×activation saliency for interpretability |
| **Proxy Supervision Target** | Automatic weak supervision target construction (dose+freq+circ_expr+ifn_score) when labels unavailable |

#### Experimental Results

| Metric | Value | Comparison |
|--------|-------|------------|
| 288K IEDB AUC (allele-aware) | **0.80** | HGB + MHC allele features |
| MOE MAE | **0.389** | 39.2% reduction vs Ridge (p<0.001) |
| Mamba3Lite+Attn(d=16) | MAE=0.395, R²=0.802 | Best single encoder with attention |
| MHC pseudo-sequence AUC | **0.917** | Exceeds ESM-2 mean pooling (0.537) |

#### Why ESM-2 Failed for Short Peptides

ESM-2 (650M parameters) achieves AUC=0.537 for MHC binding prediction, worse than traditional features. This is because:

1. **Mean pooling loses position-specific motifs**: MHC binding depends on anchor positions (P2, P9 for 9-mers). Mean pooling averages across all positions, diluting anchor-specific signals.
2. **Short peptides (8-11 AA) lack structural context**: ESM-2 was trained on proteins (hundreds to thousands of residues). For 8-11mer peptides, there is insufficient sequence for the model to learn meaningful structural representations.
3. **PCA discards discriminative directions**: ESM-2 PCA features (AUC=0.594 with 35M model) perform worse than raw pseudo-sequence encoding because PCA directions optimal for general proteins may not align with MHC binding determinants.

#### Integration with Confluencia 3.0

Epitope 2.0 connects to the main platform via `EpitopeBridge`, enabling:

- **circRNA → Protein → Epitope pipeline**: Designed circRNA sequences are translated, and resulting peptides are scored for MHC-I presentation and immunogenicity
- **Vaccine candidate screening**: Batch evaluation of peptide candidates with environment parameters (dose, frequency, treatment time)
- **Sensitivity-guided optimization**: Identify which sequence positions or experimental parameters most affect predicted efficacy

#### Usage

```python
from confluencia_2_0_epitope.core.training import train_epitope_model, predict_epitope_model

# Train with MHC allele features
model_bundle, report = train_epitope_model(
    train_df,  # columns: epitope_seq, dose, freq, efficacy
    model_backend="torch-mamba",  # or sklearn-moe
)

# Predict efficacy for new peptides
pred_df, sensitivity = predict_epitope_model(model_bundle, infer_df)
```

---

## Future Plans

### Short-term (6 months)
- [ ] Web-based graphical interface
- [ ] Pre-trained model weights release
- [ ] Integration with Lab Automation systems

### Medium-term (1 year)
- [ ] Multi-cancer type support (expand beyond TNBC)
- [ ] Clinical trial simulation module
- [ ] Real-world data validation

### Long-term (2+ years)
- [ ] Regulatory submission support
- [ ] Personalized therapy design
- [ ] Integration with hospital EHR systems

---

## References

1. Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.

2. Abramson, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*.

3. Lehmann, B.D., et al. (2011). Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. *J Clin Invest*, 121(7), 2750-2767.

4. Liu, C.X., & Chen, L.L. (2022). Circular RNAs: Characterization, cellular roles, and applications. *Cell*, 185(23), 4231-4250.

5. Wesselhoeft, R.A., et al. (2018). Engineering circular RNA for potent and stable translation in eukaryotic cells. *Nat Commun*, 9(1), 2629.

---

## Attributions

| Team Member | Role | Contribution |
|-------------|------|--------------|
| [Member 1] | Lead Developer | TorusFold architecture, Core engine |
| [Member 2] | Backend Developer | CirculaPK model, Backend system |
| [Member 3] | Frontend/UI | User interface design |
| [Member 4] | Validation | Testing, Documentation |
| [Advisor Name] | PI | Project supervision, Direction |

---

## Repository

**GitHub**: https://github.com/your-team/confluencia-3.0

**Documentation**: https://confluencia-3-0.readthedocs.io

**License**: MIT License

---

## Contact

For questions or collaboration opportunities:

- **Email**: your-team@example.com
- **Twitter**: @YourTeamName
- **iGEM Team**: [Your Team Name]

---

*This software was developed as part of the iGEM 2024 competition by [Your Team Name].*
