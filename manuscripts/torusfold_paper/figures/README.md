# TorusFold Figure Placeholders

This directory should contain the following figures for the TorusFold manuscript:

## Required Figures

### Figure 1: Torus Positional Encoding
- **Panel A**: Standard PE vs TPE comparison for L=100
- **Panel B**: TPE harmonics visualization
- **Panel C**: CircRNA topology diagram showing BSJ

### Figure 2: Architecture Comparison
- **Panel A**: RMSD bar chart with bootstrap CI
- **Panel B**: Closure error comparison
- **Panel C**: Per-sample RMSD scatter plot
- **Panel D**: Complexity vs accuracy trade-off

### Figure 3: Scheme 6 Architecture
- **Panel A**: GNN encoder diagram
- **Panel B**: Latent diffusion process
- **Panel C**: GNN decoder with closure
- **Panel D**: Training curves

### Figure 4: External Baselines
- **Panel A**: RMSD comparison (TorusFold vs IsRNA vs AF3 vs FARFAR2)
- **Panel B**: Closure error across methods
- **Panel C**: Inference time comparison
- **Panel D**: Accuracy vs computational cost

### Figure 5: TPE Ablation
- **Panel A**: TPE vs Standard PE on Scheme 6 backbone
- **Panel B**: Per-nucleotide error heatmap around BSJ
- **Panel C**: BSJ-flanking region RMSD
- **Panel D**: Circular distance vs prediction error

### Figure 6: Data Quality Analysis
- **Panel A**: PDB circularized vs circrna_3d_merged comparison
- **Panel B**: RMSD by data source
- **Panel C**: Learning curve (RMSD vs high-confidence fraction)
- **Panel D**: Error decomposition by region

### Figure 7: Length Scaling
- **Panel A**: RMSD vs sequence length
- **Panel B**: Memory usage vs sequence length
- **Panel C**: Hyperparameter sensitivity
- **Panel D**: Confidence calibration

### Figure 8: Failure Analysis
- **Panel A**: Scheme 5 coordinate explosion
- **Panel B**: Scheme 3 loss imbalance
- **Panel C**: CPU saturation patterns
- **Panel D**: Viable architecture conditions

## Figure Generation Scripts

Figures can be generated from the training logs and evaluation results using:
- `scripts/plot_results.py` - Main figure generation
- `scripts/plot_ablation.py` - TPE ablation figures
- `scripts/plot_failure.py` - Failure analysis diagrams
