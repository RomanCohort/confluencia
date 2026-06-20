# Circ-CASP: Community Contribution

## Overview

Circ-CASP (Critical Assessment of circRNA Structure Prediction) is the first community benchmark for circular RNA 3D structure prediction, established as part of our iGEM project to advance the field beyond our own research scope.

## Motivation

While developing Confluencia 3.0, we recognized that circRNA structure prediction lacks standardized benchmarks. Unlike protein structure prediction (CASP) or linear RNA structure prediction (RNA-Puzzles), circRNA's unique circular topology creates distinct challenges that existing benchmarks don't address. We established Circ-CASP to fill this gap.

## Benchmark Components

### Training Data
- **10,000 sequences** with pseudo-labeled 3D structures
- Length range: 50-2000 nucleotides
- Sources: Synthetic helical structures + IsRNAcirc augmentation
- Format: JSON (sequences) + NPY (coordinates)

### Test Data
- **30 circRNA structures** from high-quality physics simulation
- Hidden ground truth during competition
- Public sequence only during prediction phase

### Evaluation Metrics
| Task | Description | Weight |
|------|-------------|--------|
| T1 | Global RMSD | 40% |
| T2 | BSJ closure distance | 20% |
| T3 | Backbone bond consistency | 15% |
| T4 | Secondary structure pairing | 15% |
| T5 | Conformational diversity | 10% |

### Baseline Methods
We provide 6 baseline methods representing different approaches:

| Method | Type | Expected RMSD |
|--------|------|---------------|
| M1: Helical Baseline | Physics | ~25 Å |
| M2: EGNN + Physics | DL + Physics | ~15 Å |
| M3: Dual-Engine | Hybrid | ~12 Å |
| M4: DDPM Diffusion | Deep Learning | ~10 Å |
| M5: Physics Transformer | DL + Physics | ~8 Å |
| M6: GNN Latent Diffusion | Deep Learning | ~10 Å |

## Competition Tracks

### Regular Track
- Compute limits: 10 min/target, 24GB GPU memory
- Banned methods: Rosetta, IsRNAcirc, MD simulation
- Goal: Fair comparison of efficient methods

### Unlimited Track ("神仙打架")
- No compute/data/method restrictions
- Only restriction: Carbon-based participants only
- Goal: Establish theoretical upper bound

### Random Oracle Track ("欧皇奖")
- Submit random seeds instead of predictions
- Goal: Fun engagement + statistical baseline

## Results

**TBD** - Competition runs from July 10 to August 2026.

## Impact

### For the Community
- First standardized benchmark for circRNA 3D structure prediction
- Public training data and evaluation code
- Baseline implementations for future comparisons

### For Our Project
- Validates our methods against external approaches
- Identifies strengths/weaknesses of different strategies
- Engages broader research community with circRNA structure prediction

## How to Participate

1. Email: 18806370529@163.com
2. Provide: Team name, members, contact
3. Receive: Training data access after July 10

## Repository

- Code: https://github.com/RomanCohort/confluencia
- Competition rules: `manuscripts/Circ-CASP_competition_rules.md`
- Evaluation script: `confluencia_3_0/core/circrna/torusfold/circ_casp_evaluate.py`

---

*This contribution exemplifies iGEM's spirit of open science and community engagement. By establishing a public benchmark, we enable future researchers to build upon our work and advance circRNA structure prediction beyond what any single team could achieve alone.*
