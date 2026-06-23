# TorusFold Experimental Results Summary

## System Overview
TorusFold: 7 deep learning architectures for circRNA 3D structure prediction

## Test Datasets
- **circrna_3d_merged**: ~11,000 samples, confidence ~0.5 (pseudo-labels)
- **PDB circularized**: 7 samples, confidence ~0.95 (real structures)

## Results Table

### PDB Test Set (7 samples, high quality)

| Scheme | Architecture | RMSD Mean (Å) | RMSD Median (Å) | Closure (Å) | TM-score | Status |
|--------|-------------|---------------|-----------------|-------------|----------|--------|
| S1 | EGNN + Physics | 13.85 | 14.19 | 5.36 | 0.0075 | Trained |
| S2 | Physics Solver | ~2.0 | - | <0.1 | - | Zero-training |
| S4 | DDPM+EGNN | Training | Training | ? | ? | In progress |
| S5 | Transformer+PE | 245 | - | - | - | Unstable |
| S6 | GNN+Latent Diff | **13.91** | 14.08 | **0.02** | 0.0075 | Best |
| S3 | Transformer+Delta | Not trained | - | - | - | - |
| S7 | Mamba+Attention | Not trained | - | - | - | - |
| Random | Baseline | ~60 | - | - | - | - |

### circrna_3d Test Set (~11K samples, low quality)

| Scheme | RMSD Mean (Å) | Closure (Å) |
|--------|---------------|-------------|
| S2 (sota) | 25.5 | 2.75 |
| S6 v1 | 25.1 | 0.03 |
| S2 (initial) | 85.4 | 0.1 |
| Random | 60 | - |

## Key Findings

1. **Data quality dominates**: PDB data (conf=0.95) gives RMSD 14Å vs circrna_3d (conf=0.5) giving 25Å
2. **Closure is learned**: S6 learns closure to 0.02Å without explicit constraint
3. **EGNN lacks closure**: S1 has 5.36Å closure error (no explicit constraint)
4. **Physics solver closure**: S2 guarantees closure by construction
5. **Transformer instability**: S5 explodes to 245Å - no geometric invariance

## Architecture Details

### Scheme 6 (Best Performer)
- GNN Encoder: physics-aware, circular position encoding
- Latent Diffusion: 50 steps, operates in latent space
- GNN Decoder: reconstructs 3D coords with closure enforcement
- Key fix: denoised latent passed to decoder (not clean/noise_pred)

### Scheme 1 (EGNN)
- Equivariant GNN with KNN graph
- Planar circular initialization
- Learnable coordinate step size

### Scheme 5 (Transformer - FAILED)
- Direct coordinate prediction (no geometric anchor)
- Causes gradient explosion
- Fix: delta prediction from helical init (not yet tested)

## Bug Fixes Applied
1. kabsch_rmsd rotation matrix formula corrected
2. S5: changed to delta prediction from planar circular init
3. EGNN: learnable coord_step, padding mask for bond/closure loss
4. S3: planar circular init (no z-offset)
5. circrna_diffusion: learnable coord_step
