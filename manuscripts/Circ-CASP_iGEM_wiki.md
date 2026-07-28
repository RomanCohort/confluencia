# Circ-CASP: Community Contribution

## Overview

Circ-CASP (Critical Assessment of circRNA Structure Prediction) is the first community benchmark for circular RNA 3D structure prediction, established as part of our iGEM project to advance the field beyond our own research scope.

## Motivation

While developing Confluencia 3.0, we recognized that circRNA structure prediction lacks standardized benchmarks. Unlike protein structure prediction (CASP) or linear RNA structure prediction (RNA-Puzzles), circRNA's unique circular topology creates distinct challenges that existing benchmarks don't address. We established Circ-CASP to fill this gap.

## Benchmark Components

### Training Data
- **10,000+ sequences** with pseudo-labeled 3D structures
- Length range: 50-2050 nucleotides
- Four heterogeneous sources (quality hierarchy):

| Source | Samples | Length | Quality | Method |
|--------|---------|--------|---------|--------|
| IsRNAcirc (real + aug) | ~2,754 | 161-2050 | Highest | 34 real circRNA 3D structures from PDB, 80x rotation+noise augmentation; 24/34 with real secondary structure from .subo files |
| icSHAPE-constrained | ~2,000 | 200-1000 | Medium-High | Experimental SHAPE reactivity profiles (GSE74353, Flynn et al. Science 2016) → ViennaRNA SHAPE-constrained folding → GeometricConstraintSolver |
| PDB circularized | ~4,000 | 50-500 | Medium | Linear RNA structures from RCSB PDB (resolution <3.0A), circularized via GeometricConstraintSolver annealing closure |
| Synthetic physics | ~5,000 | 50-500 | Medium | Random sequences → ViennaRNA circ-mode secondary structure → GeometricConstraintSolver |

- Format: JSON (sequences + secondary structure + pair constraints) + NPY (3D coordinates)
- All samples include secondary structure and base-pair constraints (not just helical coordinates)

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
We provide 7 baseline methods representing different approaches:

| Method | Type | Expected RMSD | TorusFold Scheme |
|--------|------|---------------|------------------|
| M1: Helical Baseline | Physics | ~25 A | (geometry-only) |
| M2: EGNN + Physics | DL + Physics | ~15 A | Scheme 1 |
| M3: Dual-Engine | Hybrid | ~12 A | Scheme 3 (deferred) |
| M4: DDPM Diffusion | Deep Learning | ~10 A | Scheme 4 |
| M5: Physics Transformer | DL + Physics | ~8 A | (deprecated → Scheme 5) |
| M6: GNN Latent Diffusion | Deep Learning | ~10 A | Scheme 6 |
| M7: Mamba+Transformer Hybrid | Deep Learning | ~8 A (L≤500), ~12 A (L>500) | Scheme 7 |

Note: Expected RMSD values are estimates. With the expanded multi-source training data (including real IsRNAcirc structures and icSHAPE-constrained profiles), deep learning methods (M4-M7) may achieve significantly better performance than these initial estimates.

Beyond the baselines, TorusFold also implements Scheme 8: Sparse Pair-Guided Hybrid, an O(L·K) architecture not included as a competition baseline but available as an open-source reference implementation. Scheme 8 uses ViennaRNA Top-K candidate selection plus a geometric feedback loop to recover pairs the energy-based predictor misses. It targets the long-therapeutic-circRNA regime (L>1000) where O(L²) methods become memory-bound. See the Software Page for full architecture details.

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
