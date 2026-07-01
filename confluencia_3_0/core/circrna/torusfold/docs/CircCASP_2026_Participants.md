# Circ-CASP 2026 Official Participants

## Competition Overview

**Official Name:** Circ-CASP 2026 (Critical Assessment of circRNA Structure Prediction)
**Participants:** 13 teams (9 main + 3 expert + 1 random baseline)
**Organizer:** Jilin University, College of Computer Science and Technology

---

## Main Track (9 Teams)

| Team | Institution | Method | Scheme | Status |
|------|-------------|--------|--------|--------|
| **Team 1** | JLU CS | EGNN + Physics Refinement | Scheme 1 | ✅ Implemented |
| **Team 2** | JLU CS | Atomic Force Field Solver | Scheme 2 | ✅ Implemented |
| **Team 3** | JLU CS | Dual-Engine Iterative Distillation | Scheme 3 | ✅ Implemented |
| **Team 4** | JLU CS | Coordinate Diffusion + EGNN | Scheme 4 | ✅ Implemented |
| **Team 5** | JLU CS | Transformer Physics-Bias | Scheme 5 | ⚠️ Deprecated (NaN explosion) |
| **Team 6** | JLU CS | Latent Space Diffusion | Scheme 6 | ✅ Implemented |
| **Team 7** | JLU CS | Local Attention + Circular Mamba | Scheme 7 | ⭐ Recommended |
| **Team 8** | JLU CS | Sparse Pair-Guided Hybrid | Scheme 8 | ✅ Implemented |
| **Team 9** | JLU CS | **Linear RNA Cyclization** | **Scheme 0** | ✅ **Official Baseline** |

---

## Expert Track (3 Teams)

| Team | Institution | Method | Integration Status |
|------|-------------|--------|-------------------|
| **Team 10** | Zhejiang University (Life Science & QB) | isRNAcirc | External mature method |
| **Team 11** | Shandong University (Math Center) | trRosettaRNA2 (Cyclization) | ✅ Integrated in Stage 2 |
| **Team 12** | University of Vienna (Theoretical Chemistry) | ViennaRNA-Circ | ✅ Integrated in Stage 1 |

---

## Random Baseline Track

| Team | Method |
|------|--------|
| **Team 13** | Random baseline {114514, 67, 886} |

---

## Key Relationships

### Scheme 0 (Team 9) - The Foundation

**Team 9 = Scheme 0 = CircFold Baseline = 线性RNA环化法**

Roles:
1. **Official Baseline** - Benchmark for all teams
2. **Data Generator** - Provides 80k training structures
3. **Teacher Model** - Knowledge source for Team 3 (Scheme 3)
4. **Pipeline** - 5-stage physics-based prediction

---

### Team 3 - Dual-Engine Distillation

**Teacher:** Team 9 (Scheme 0 - CircFold Baseline)
**Student:** Trainable neural network (Scheme 1/6/7)

Training Flow:
```
Team 9 generates data → Team 3 learns from Team 9 → Faster inference
```

---

## Predicted Rankings

| Rank | Team | Reason |
|------|------|--------|
| 🥇 1st | **Team 7** | Mamba long-range dependency + circular optimization |
| 🥈 2nd | **Team 3** | Knowledge distillation from Team 9 (Teacher) |
| 🥉 3rd | **Team 9** | Official baseline, physics-based quality |
| 4th | Team 8 | Sparse pair guidance, high BSJ accuracy |
| 5th | Team 10 | Mature external method (isRNAcirc) |
| 6th | Team 4 | Diffusion + EGNN, good convergence |
| 7th | Team 1 | EGNN physics refinement |
| 8th | Team 6 | Latent diffusion, slower convergence |
| 9th | Team 11 | trRosettaRNA2 improvement |
| 10th | Team 12 | ViennaRNA coordinate mapping |
| 11th | Team 2 | Atomic force field, computationally expensive |
| 12th | Team 5 | Deprecated (NaN issues) |
| 13th | Team 13 | Random baseline |

---

## Technical Stack Summary

| Technical Route | Teams | Representative Methods |
|----------------|-------|----------------------|
| **Physics-Based** | Team 9, 11, 12 | Pipeline, trRosettaRNA2, ViennaRNA |
| **EGNN Family** | Team 1, 4 | EGNN + Physics, Diffusion + EGNN |
| **Diffusion Models** | Team 4, 6 | Coordinate Diffusion, Latent Diffusion |
| **Distillation** | Team 3 | Dual-Engine (Teacher: Team 9) |
| **Mamba Architecture** | Team 7 | Mamba + Transformer |
| **Sparse Attention** | Team 8 | Sparse Pair-Guided |

---

## Citation

```bibtex
@competition{CircCASP2026,
  title={Circ-CASP 2026: Critical Assessment of circRNA Structure Prediction},
  organizer={Jilin University, College of Computer Science and Technology},
  year={2026},
  participants={13 teams (9 main + 3 expert + 1 random baseline)}
}
```

---

**Circ-CASP 2026 - Advancing circRNA 3D Structure Prediction**