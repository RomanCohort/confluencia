"""
TorusFold — Positioning & Value Proposition
=============================================

## The Data Reality

circRNA lacks the data infrastructure that made AlphaFold successful:
- No CASP-like competition for circRNA structure
- No PDB-scale database of circRNA 3D structures
- No MSA depth (circRNA is non-coding, endogenous)
- Only ~20 experimental data points for immunogenicity

This is NOT a limitation of our architecture — it's a field-wide gap.

## What TorusFold IS

TorusFold is a **methodological contribution** that addresses a real,
unsolved problem: how to do deep learning on circular RNA topology.

### The core innovations are architectural, not predictive:

1. **Torus Positional Encoding (TPE)**
   - First positional encoding designed for S¹ topology
   - Eliminates boundary artifacts at back-splice junction
   - Provable periodicity: TPE(x, L) = TPE(x+L, L)
   - No existing method (ESM2, RNA-FM, etc.) handles circular topology

2. **CircPairformer (AF3-inspired triangle updates for circRNA)**
   - AlphaFold3's Pairformer adapted for circular distance
   - Triangle inequality on S¹: d_circ(i,j) ≤ d_circ(i,k) + d_circ(k,j)
   - BSJ-crossing pair detection via circular relative bias
   - No existing method applies AF3-style pair updates to RNA

3. **Circular closure constraint**
   - Diffusion endpoint: x[0] ≈ x[-1] (covalent bond)
   - Unique to circRNA, impossible in linear RNA methods
   - ViennaRNA circular mode handles this for 2D; we extend to 3D

4. **BSJ-crossing pair prediction**
   - Pairs (i,j) where |i-j| > L/2 cross the back-splice junction
   - These are UNIQUE to circRNA and functionally critical
   - No other deep learning method models this

## What TorusFold CAN do (with current data)

| Task | Data Source | Status |
|------|------------|--------|
| Pathway classification (7-class) | sequences_enhanced.csv (3K) | Feasible |
| Immunogenicity prediction (binary) | sequences_enhanced.csv (3K) | Feasible |
| Secondary structure hypothesis | ViennaRNA circular mode as proxy | Research tool |
| BSJ-crossing pair prediction | Synthetic + thermodynamic validation | Novel |
| 3D structure hypothesis | No ground truth → qualitative validation | Future |

## What TorusFold CANNOT do (without data)

| Task | Why Not |
|------|---------|
| Accurate 3D structure prediction | No circRNA 3D structures in PDB |
| Quantitative immunogenicity prediction | Only 20 experimental points |
| Multi-task composite scores | 8 targets are collinear in current data |
| Gene expression → immunogenicity | Zero correlation in data |

## Comparison with existing methods

| Method | Topology | 2D/3D | Deep Learning | BSJ-aware |
|--------|----------|-------|---------------|-----------|
| ViennaRNA (circular) | Circular | 2D | No | Partial |
| CircRNAstructure | Circular | 2D | No | Yes |
| ESM2/RNA-FM | Linear | Embedding | Yes | No |
| AlphaFold3 | Linear | 3D | Yes | No |
| RoseTTAFold2NA | Linear | 3D | Yes | No |
| **TorusFold** | **Circular** | **3D** | **Yes** | **Yes** |

TorusFold is the ONLY method that combines:
- Circular topology (like ViennaRNA/CircRNAstructure)
- Deep learning (like ESM2/AlphaFold3)
- 3D structure (like AlphaFold3)
- BSJ-awareness (like CircRNAstructure)

## Paper positioning

Title idea: "TorusFold: Torus-Aware Deep Learning for Circular RNA Structure and Function Prediction"

Key selling points:
1. First application of AF3-style pair updates to circRNA
2. First torus positional encoding for circular topology
3. First BSJ-crossing pair prediction via deep learning
4. Framework ready for future data (like AlphaFold was before CASP14)

The analogy: AlphaFold1 (2018) had moderate accuracy but introduced
the key architectural ideas. AlphaFold2 (2020) achieved breakthrough
accuracy when sufficient data existed. TorusFold is at the AlphaFold1
stage — introducing the right architecture for circRNA before the
data catches up.

## Practical next steps (iGEM competition)

For iGEM, the value proposition is different from a Nature paper:
- Demonstrate that circular topology matters (TPE vs sinusoidal PE)
- Show BSJ-crossing pairs are predicted (qualitative validation)
- Visualize circRNA 3D structure hypotheses
- Provide a tool for the community

The iGEM judges will care about:
1. Is the problem real? YES (circRNA therapeutics is a hot field)
2. Is the approach novel? YES (no one has done AF3 for circRNA)
3. Can you demonstrate it works? PARTIALLY (pathway classification + visualization)
4. Is it useful? YES (framework for future circRNA drug design)
"""
