# Confluencia 3.0: circRNA vaccine design pipeline

## Overview

Confluencia 3.0 is a software pipeline for designing circRNA vaccines. You put in a sequence, it runs immunogenicity screening, pharmacokinetics simulation, and epitope prediction, and gives you optimized vaccine candidates.

```bash
pip install confluencia
```

```python
from confluencia import VaccinePipeline
result = VaccinePipeline().run("AUGCGC...", target="TNBC")
print(result.recommendation)
```

## Why this exists

circRNA vaccines last 8-24 hours vs 2-4 hours for linear mRNA. They resist exonucleases. They can express proteins for weeks instead of days. After the mRNA vaccine rollout in 2020, circRNA is the obvious next step.

But designing one means answering questions that no existing tool handles together:

1. Will it trigger an immune reaction? (Immunogenicity)
2. How long will it express protein? (Pharmacokinetics)
3. Will the protein get presented to T cells? (Epitope prediction)

We built Confluencia because the answer to all three should come from one tool.

## What works right now

| Module | Usage | Validation | Status |
|--------|-------|------------|--------|
| Epitope 2.0 | `epitope_predict("SIINFEKL")` | AUC 0.80 (288K IEDB) | Production |
| Immunogenicity | `immune_score("AUGCGC...")` | r=0.91 (Chen 2019) | Production |
| CirculaPK | `pk_simulate(dose=1.0)` | 12% error (N=4) | Production |
| Optimizer | `evolve(seed_seq, gens=50)` | Pareto fronts | Production |
| Hub | `hub.push()` / `hub.pull()` | 12 seed designs | Deployed |
| TorusFold | `structure_predict(seq)` | Transfer learning in progress | Research |

Four modules work. TorusFold uses transfer learning: we freeze the ESM2 backbone (pretrained on linear RNA) and train only the TPE layer and CircPairformer on ViennaRNA pseudo-labels. This lets the model learn circRNA topology without real 3D structure data.

## Quickstart

```bash
pip install confluencia
# Optional: pip install confluencia[full]
# Optional: docker pull confluencia/confluencia:latest
```

```python
from confluencia import quick_scan, quick_pk, quick_epitope

# Check immunogenicity
immune = quick_scan("AUGCGCUUGU...")
print(f"MDA5: {immune['MDA5']:.2f}")  # dsRNA sensor
print(f"Overall: {immune['overall']:.2f}")  # Safe if <0.5

# Predict pharmacokinetics
pk = quick_pk(dose=1.0, modification="m6A")
print(f"Half-life: {pk['half_life']:.1f} hours")

# Check epitope (for vaccines)
epitope = quick_epitope("SIINFEKL", allele="HLA-A*02:01")
print(f"Presentation probability: {epitope['presentation_prob']:.2f}")
```

Full pipeline:

```python
from confluencia import VaccinePipeline

pipeline = VaccinePipeline()
result = pipeline.run(
    sequence="AUGCGC...",
    target="TNBC",
    subtype="IM",
    modification="m6A",
    optimize=True
)

print(result.recommendation)
# "Add m6A at positions 45, 78. Swap codon 234 for better epitope. Predicted half-life: 18.2h"

result.pareto_front.to_csv("optimized_sequences.csv")
```

## Module details

### Epitope 2.0

Predicts which 8-11mer peptides from your protein get presented by MHC class I molecules. T cells only see peptides on MHC-I. If your vaccine expresses a protein but the peptides never make it to the surface, nothing happens. That is a failed vaccine, and you just spent $500 on synthesis to find out.

```python
from confluencia.epitope import EpitopePredictor

predictor = EpitopePredictor()
result = predictor.score_peptide(
    peptide="SIINFEKL",
    allele="HLA-A*02:01",
    context={"dose": 50, "frequency": 2}
)

print(result)
# {'binding_score': 0.87, 'presentation_prob': 0.72, 'rank': 15,
#  'sensitivity': {'pos_2_I': 0.23, 'pos_9_L': 0.19}}
```

Batch prediction for entire proteins:

```python
protein = "MGS..."  # full sequence
peptides = predictor.scan_protein(protein, allele="HLA-A*02:01")
print(peptides.top5)
```

Validation numbers:

| Metric | Value | Dataset |
|--------|-------|---------|
| AUC (binding) | 0.917 | IEDB MHC-I |
| AUC (presentation) | 0.80 | 288K IEDB |
| MAE (efficacy) | 0.389 | MOE validation |

We tried ESM-2 for this task. It got AUC 0.537. Worse than a coin flip. The problem: ESM-2 was trained on full proteins, and mean pooling averages across all positions. That destroys the anchor signals at P2 and P9 that determine MHC binding. Short peptides (8-11 amino acids) are too short for the model to learn anything useful.

| Approach | AUC | What went wrong |
|----------|-----|-----------------|
| ESM-2 650M | 0.537 | Mean pooling destroys anchor signals |
| ESM-2 + PCA | 0.594 | Still loses position info |
| MHC pseudo-sequence | 0.917 | Preserves anchor positions |

We went back to NetMHCpan's approach: encode the MHC binding groove as a 34-position pseudo-sequence. That works.

### Immunogenicity scanner

Scores your circRNA across four innate immune pathways: MDA5 (dsRNA structures), PKR (long dsRNA), TLR7/8 (ssRNA motifs). Any of these can cause injection-site reactions or systemic inflammation.

```python
from confluencia.circrna import ImmunogenicityScanner

scanner = ImmunogenicityScanner()
score = scanner.scan(sequence="AUGCGCUUGU...", modification="m6A")

print(score)
# {'MDA5': 0.31, 'PKR': 0.41, 'TLR7': 0.22, 'TLR8': 0.18,
#  'overall': 0.28, 'interpretation': 'safe_profile'}
```

One thing that surprised us: m6A suppresses MDA5 by about 90%, but TLR7/8 only by about 30% and PKR by about 20%. The common line that "m6A reduces immunogenicity" is oversimplified. It depends heavily on which pathway you are asking about.

Validated: Spearman r = 0.91 with Chen et al. (2019) IFN-beta measurements (N=7). Leave-one-out median r = 0.87 [IQR 0.82-0.91].

### CirculaPK

Simulates circRNA from injection through protein expression. Six compartments: Depot, Blood, Tissue, Endosome, Cytoplasm, Protein.

The bottleneck is endosomal escape. Only 1-4% of circRNA makes it from the endosome into the cytoplasm. Standard PK models skip this. We do not, because if you skip the biggest bottleneck you get the wrong answer.

```python
from confluencia.pk import CirculaPK

pk = CirculaPK()
curve = pk.simulate(dose=1.0, modification="m6A", delivery="LNP", duration_hours=48)

print(f"Half-life: {curve['half_life']:.1f} hours")
print(f"Peak protein: {curve['peak_protein']:.1f} ng/mL")
print(f"Endosomal escape: {curve['escape_fraction']*100:.1f}%")
```

| Metric | Value | Reference |
|--------|-------|-----------|
| Half-life error | 12% [CI 3-21%] | Wesselhoeft 2018 (N=4) |
| m6A extension | 15-22h | Matches literature |
| Psi extension | 20-30h | Matches literature |

### Sequence optimizer

Evolves your circRNA sequence across stability, translation efficiency, and immune safety at the same time. Single-objective optimization gives you one answer. Multi-objective gives you a Pareto front so you can pick the trade-off yourself.

```python
from confluencia.evolution import SequenceOptimizer

optimizer = SequenceOptimizer()
front = optimizer.evolve(
    seed_sequence="AUGCGC...",
    objectives=['stability', 'translation', 'immune_evasion'],
    generations=50
)

print(front.pareto_table)
```

## Confluencia Hub

A shared database of circRNA vaccine designs. Upload your validated sequences, download designs from other teams, track version history. circRNA synthesis costs $200-500 per construct, so sharing computational predictions saves everyone wet-lab cycles.

```python
from confluencia.hub import Hub

hub = Hub()
designs = hub.search(target="TNBC", immune_profile="low")
design = hub.pull("TNBC_IM_v1")
hub.push(name="my_design_v1", sequence="AUGCGC...",
         predictions={'MDA5': 0.31, 'half_life': 18.2})
```

Seed designs:

| ID | Target | Half-life | Immune Score |
|----|--------|-----------|--------------|
| TNBC_BLIS_v1 | TNBC | 16.2h | 0.32 |
| TNBC_IM_v1 | TNBC | 18.5h | 0.28 |
| Lung_Adeno_v1 | Lung | 15.9h | 0.35 |

## Technical details

Architecture: EventBus-driven. Six subsystems communicate through 40 event types. No module directly calls another.

Backend fallback for labs without GPU or internet:

```
Tier 0: ESM2-650M (GPU required)
Tier 1: ViennaRNA (CPU, local install)
Tier 2: Heuristic rules (pure Python, no dependencies)
```

Interfaces:

| Interface | Install | Best for |
|-----------|---------|----------|
| Python API | `pip install confluencia` | Full control |
| Streamlit | `confluencia-studio` | Wet lab teams |
| CLI | `confluencia --help` | Quick scripts |
| R | `install.packages("confluencia")` | Bioinformatics |
| Docker | `docker pull confluencia/confluencia` | Reproducibility |

Testing: 87% coverage, GitHub Actions CI/CD.

## Validation summary

| Module | Metric | Dataset | Result |
|--------|--------|---------|--------|
| Epitope 2.0 | AUC | 288K IEDB | 0.80 |
| MHC binding | AUC | IEDB subset | 0.917 |
| Immunogenicity | Spearman r | Chen 2019 (N=7) | 0.91 |
| CirculaPK | Relative error | Wesselhoeft 2018 (N=4) | 12% |

## Negative results we share

ESM-2 (650M parameters) got AUC=0.537 for MHC binding prediction. Worse than random. Trained on full proteins, 8-11mer peptides too short, mean pooling destroys anchor position signals.

Other iGEM teams will probably try ESM-2 for similar tasks. It will fail. Use MHC pseudo-sequence encoding instead.

## How we solve the circRNA data problem

No circRNA 3D structures exist in PDB. Instead of waiting, we use transfer learning:

1. **Freeze ESM2 backbone**: Pretrained on linear RNA, it already understands nucleotide interactions. We keep these weights frozen.

2. **Train TPE layer**: Only the Torus Positional Encoding layer learns circRNA's circular topology. This is ~1% of the total parameters.

3. **ViennaRNA pseudo-labels**: We use thermodynamic predictions as training targets. Not perfect, but good enough to learn the topology.

4. **BSJ-weighted loss**: Higher loss weight for back-splice junction flanking regions, where circular topology matters most.

```python
from confluencia.torusfold import train_transfer_learning

# Train on circBase sequences with ViennaRNA pseudo-labels
model = train_transfer_learning(
    sequences="circbase.fasta",
    epochs=50,
    bsj_weight=2.0  # Higher weight for BSJ regions
)
```

This approach has been used successfully for other domains with limited data (protein structure prediction before AlphaFold, drug discovery with few labeled compounds). We are applying the same principle to circRNA.

## What other teams used

| Team | Module | Result |
|------|--------|--------|
| [Your team here] | Epitope 2.0 | Identified high-scoring peptides |
| [Your team here] | Immunogenicity Scanner | Reduced dsRNA content |
| [Your team here] | Hub designs | Saved synthesis iterations |

## Reproducibility

```bash
docker pull confluencia/confluencia:3.0
docker run -it confluencia/confluencia:3.0 python -c \
  "from confluencia import quick_scan; print(quick_scan('AUGCGC'))"
```

Requirements: Python 3.10+, numpy, pandas, scipy, scikit-learn. Optional: torch, viennarna, openmm.

## References

1. Chen et al. 2019. Mol Cell 73:422.
2. Wesselhoeft et al. 2018. Nat Commun 9:2629.
3. Jumper et al. 2021. Nature 596:583.
4. IEDB. iedb.org
5. Lorenz et al. 2011. Algorithms Mol Biol 6:26.

## Team

IGEM FBH 2026, Jilin University First Hospital

Software Lead: [Name]
ML/DL Development: [Names]
Validation: [Names]
Wet Lab Integration: [Names]

## Repository

https://github.com/RomanCohort/confluencia

MIT License. Python 3.10+. 87% test coverage. GitHub Actions CI/CD.

## Contact

Email: [team email]
Hub: https://hub.confluencia.dev
Docs: https://docs.confluencia.dev