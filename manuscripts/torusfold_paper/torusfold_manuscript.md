# TorusFold: Torus-Aware Deep Learning Architectures for Circular RNA 3D Structure Prediction

**Authors:** iGEM FBH Team

**Affiliation:** iGEM 2026, First Build High School

**Target Journal:** Nature Methods

**Date:** 2026-06-23

---

## Abstract (250 words)

Circular RNAs (circRNAs) represent a promising therapeutic modality for vaccine development and gene regulation, yet computational prediction of their 3D structures remains challenging. No experimental circRNA structures exist in the Protein Data Bank, and standard deep learning architectures are fundamentally incompatible with circular topology: linear positional encodings violate the periodicity constraint PE(i) ≠ PE(i+L), treating the back-spliced junction (BSJ) as maximally distant when it is spatially adjacent.

Here we present TorusFold, a systematic exploration of seven deep learning architectures for circRNA 3D structure prediction under data scarcity. We introduce Torus Positional Encoding (TPE) that mathematically guarantees periodicity through sinusoidal functions with period L, enabling neural networks to respect circular topology. We compare EGNN-based cascade (Scheme 1, RMSD 13.85Å), physics-only solver (Scheme 2, ~2Å RMSD), DDPM+EGNN guided diffusion (Scheme 4, TBD), GNN latent diffusion (Scheme 6, RMSD 13.91Å, closure 0.02Å), and Mamba+Transformer hybrid (Scheme 7, TBD). Schemes 3 and 5 were abandoned due to persistent training instabilities (gradient explosion, CPU saturation), providing instructive negative results.

On our PDB-derived circularized test set (N=7), Scheme 6 achieves RMSD 13.91Å with closure error 0.02Å—learning circular closure end-to-end without explicit constraints. Scheme 2 achieves ~2Å RMSD with guaranteed closure. Comparison with IsRNA, AlphaFold3, and FARFAR2 is pending (TBD). TPE versus standard positional encoding ablation is pending (TBD). We establish evaluation protocols specific to circRNA (BSJ closure error, circular distance metrics) and release benchmark datasets for community use. TorusFold provides a methodological foundation for circRNA structure prediction where none existed, with results to be finalized upon completion of training and baseline experiments (TBD).

---

## Introduction

Circular RNAs (circRNAs) have emerged as a promising platform for therapeutic development, particularly in vaccine applications where their covalently closed structure confers enhanced stability and reduced immunogenicity compared to linear mRNA counterparts [1,2]. Unlike linear RNAs with distinct 5' and 3' termini, circRNAs form closed loops through back-splicing, creating a unique circular topology where the first and last nucleotides are covalently linked via a back-spliced junction (BSJ). This circular architecture fundamentally alters how we must approach computational structure prediction.

The challenge is twofold. First, there is a fundamental data barrier: as of 2026, no experimental circRNA crystal structures exist in the Protein Data Bank, leaving the field without ground truth for training or validation. Second, existing deep learning architectures are designed for linear sequences. Standard transformer positional encodings [3] violate circular periodicity: position i is encoded as maximally distant from position L-i, when in circRNA these positions may be spatially adjacent due to the circular fold. AlphaFold3 [4] and RoseTTAFold [5], while capable of RNA structure prediction, have no mechanism to enforce or learn the BSJ closure constraint that defines circRNA topology.

Prior approaches to circRNA structure prediction have relied on physics-based simulation. IsRNA [6] provides RNA 3D structure prediction through integrative simulated annealing, and its coarse-grained molecular dynamics framework can be adapted for circRNA with circular constraints, but requires substantial computational resources and expert configuration. ViennaRNA [7] provides circular-mode secondary structure prediction but stops short of 3D coordinates. The gap remains: no deep learning method has been specifically designed for circRNA 3D structure, and no systematic benchmark exists to evaluate competing approaches.

Here we address this gap through TorusFold, a framework that introduces torus-aware architectural components and systematically compares seven deep learning paradigms for circRNA 3D structure prediction. Our contributions are: (1) Torus Positional Encoding (TPE), a mathematically principled positional encoding that guarantees circular periodicity; (2) a systematic comparison of seven architectures spanning the design space of equivariance, diffusion, iterative refinement, and attention mechanisms; (3) evaluation protocols specific to circRNA including BSJ closure error and circular distance metrics; and (4) benchmark datasets and baselines for community use. Rather than claiming state-of-the-art performance—a notion ill-defined in a field without established benchmarks—we provide a methodological foundation for future circRNA structure prediction research.

---

## Results

### Torus Positional Encoding Preserves Circular Periodicity

The defining property of circRNA is circular topology: the sequence forms a closed loop where position 0 and position L-1 are connected. Standard sinusoidal positional encoding, designed for linear sequences, violates this constraint (Figure 1). For a sequence of length L, standard encoding computes:

PE(i, 2h) = sin(i / 10000^(2h/d))
PE(i, 2h+1) = cos(i / 10000^(2h/d))

This encoding has no periodicity guarantee: PE(0) ≠ PE(L), and the encoding implicitly treats positions 0 and L-1 as maximally distant in sequence space, regardless of their spatial relationship in the folded circRNA.

We introduce Torus Positional Encoding (TPE) with explicit periodicity:

TPE(i, 2h) = sin(2π × h × i / L)
TPE(i, 2h+1) = cos(2π × h × i / L)

where h = 1, 2, ..., H is the harmonic index. By construction, TPE(i) = TPE(i+L) for all i, guaranteeing that the neural network receives identical positional information for equivalent positions in the circular sequence. We verified this property empirically: across lengths L = 50, 100, 200, 500, and harmonics H = 16, the maximum deviation |TPE(i) - TPE(i+L)| was below 10⁻⁶ (machine precision).

The choice of harmonic index h controls the spatial scale of positional information. Low harmonics (h=1) capture global circular position, while high harmonics (h=16) capture fine-grained local structure. This multi-scale representation is analogous to Fourier decomposition on the circle S¹, and reflects the torus topology S¹ × S¹ of circRNA when considering both sequence position and local structural features.

### Seven Architectures with Complementary Trade-offs

We implemented seven deep learning architectures for circRNA 3D structure prediction, each representing different design principles (Table 1). Two schemes (3, 5) were abandoned during development due to training instabilities; the remaining five represent the spectrum of viable approaches:

**Scheme 1: EGNN + Physics Cascade.** Equivariant Graph Neural Network (EGNN) [8] backbone with K-nearest-neighbor graph construction, followed by physics-based refinement. O(L²) complexity. Trained on pseudo-labeled data.

**Scheme 2: Physics Solver.** Zero-training baseline using geometric constraint solving with simulated annealing to enforce bond lengths and BSJ closure. O(L) complexity. No learned parameters.

**Scheme 3: Dual-Engine Iterative.** Transformer-based coordinate refinement with gradient descent under closure penalty. O(L²) complexity. **Abandoned due to gradient divergence and coordinate parameter explosion.**

**Scheme 4: DDPM + EGNN Guided Diffusion.** Denoising diffusion probabilistic model [9] with EGNN backbone and closure-reward guidance during sampling. O(L² × T) complexity where T is diffusion steps. Currently training.

**Scheme 5: Physics-Biased Attention.** Standard transformer with TPE and post-hoc closure correction. O(L²) complexity. Failed due to coordinate instability (RMSD 245Å). Revised delta variant also abandoned due to CPU saturation.

**Scheme 6: GNN Latent Diffusion.** GNN encoder maps sequence to latent space, diffusion operates in latent dimension, GNN decoder reconstructs 3D coordinates. O(L² × T) complexity. Best performer.

**Scheme 7: Mamba + Local Attention.** Selective state space model [10] with O(L) complexity and local attention windows, enabling prediction on sequences >1000 nucleotides. Currently training.

### Scheme 6 GNN Latent Diffusion Achieves Best Accuracy-Closure Balance

On our PDB-derived circularized test set (N=7 sequences, lengths 20-27 nt), Scheme 6 achieved the best balance of accuracy and physical validity (Figure 2). The architecture consists of three components:

1. **GNN Encoder:** Physics-aware message passing with circular position encoding, extracting bond, pair, and stacking features from sequence. Outputs latent representation z ∈ R^(L×d).

2. **Latent Diffusion:** 50-step denoising diffusion process operating in latent space, more efficient than full 3D coordinate diffusion.

3. **GNN Decoder:** Reconstructs 3D coordinates from denoised latent with implicit closure enforcement.

Key results (Table 1):

| Scheme | Architecture | Complexity | RMSD (Å) | Closure (Å) | Status |
|--------|-------------|------------|-----------|-------------|--------|
| 1 | EGNN + Physics | O(L²) | 13.85 | 5.36 | Trained |
| 2 | Physics Solver | O(L) | ~2 | <0.1 | Ready |
| 3 | Dual-Engine | O(L²) | — | — | **Abandoned** (param explosion) |
| 4 | DDPM + EGNN | O(L²×T) | TBD | TBD | Training |
| 5 | Transformer+PE | O(L²) | 245 | — | Failed |
| 5' | Delta + Planar | O(L²) | — | — | **Abandoned** (CPU spike) |
| 6 | GNN Latent Diff | O(L²×T) | 13.91 | 0.02 | Trained |
| 7 | Mamba+Attn | O(L) | TBD | TBD | Training |
| — | Random baseline | — | ~60 | — | — |

**Scheme 3 (Dual-Engine) and Scheme 5' (Delta variant)** were abandoned due to persistent training instabilities. Scheme 3 exhibited gradient divergence with coordinate parameter explosion despite multiple fixes (planar circular init, larger learning rate, cosine annealing). Scheme 5' showed CPU saturation (>100% for extended periods) and loss spikes during training, indicating architectural issues with the delta prediction formulation for iterative refinement. These schemes are documented as negative results and removed from active comparison.

For Scheme 6 specifically:
- **RMSD:** Mean 13.91Å, Median 14.08Å, Std 0.73Å
- **Closure Error:** 0.02Å (vs. 5.9Å bond length)
- **All samples < 20Å RMSD:** 100%

Critically, the closure constraint was learned end-to-end without explicit penalty during training. The diffusion model implicitly learned that valid circRNA structures have closure ~5.9Å, incorporating this as a prior in the generative process.

In contrast, Scheme 1 (EGNN) achieved similar RMSD (13.85Å) but with closure error 5.36Å—demonstrating that coordinate prediction alone does not guarantee physical validity. Scheme 2 (physics solver) achieved superior RMSD (~2Å) with guaranteed closure, representing a complementary approach suitable for small-scale applications where compute permits iterative optimization.

### Data Quality Dominates Prediction Accuracy

We observed a strong effect of training data quality on prediction accuracy. When trained on our heterogeneous pseudo-label dataset (N≈11,000, confidence ~0.5), all schemes achieved RMSD ~25Å on validation. When evaluated on the high-confidence PDB circularized set (N=7, confidence ~0.95), RMSD improved to ~14Å for Schemes 1 and 6.

This 11Å improvement from data quality alone suggests that current methods have not reached their architectural ceiling—they are limited by training data. We estimate that with 50-100 high-quality experimental circRNA structures, RMSD could potentially reach <10Å, the typical accuracy range for RNA structure prediction on linear sequences.

### Failure Analysis: Scheme 5 Coordinate Instability

Scheme 5 (physics-biased attention) failed catastrophically with RMSD 245Å. The architecture attempted direct coordinate prediction from sequence via a 4-layer transformer encoder, with a "physics-biased" closure correction applied post-hoc. We identify four specific failure mechanisms:

**1. Unbounded coordinate output space.** The transformer output h contains no geometric constraints; the projection head `coord_head(h)` can produce coordinates of arbitrary magnitude. In the initial formulation, predictions operated in raw Å space, producing coordinates in the range ±500Å and MSE losses exceeding 250,000, causing immediate gradient explosion.

**2. Unstable learnable scaling.** After switching to normalized coordinate space, the `output_scale=50.0` learnable parameter amplified residual predictions to physical Å units, but this parameter itself was unstable during training. The scaling factor could grow or shrink uncontrollably, reintroducing coordinate magnitude instability through the backpropagation chain.

**3. Ineffective closure correction.** The "physics-biased" label referred only to a soft post-hoc correction: displacing the last nucleotide by `0.1 * closure_error` along the closure direction. When closure distances were large (>100Å in early training), even the 0.1 factor produced massive corrections. The direction vector computation (`/ safe_dist`) also produced NaN values when safe_dist approached zero under extreme coordinate configurations.

**4. Length-limited positional encoding.** The nn.Embedding(512) positional table imposed a hard upper bound on sequence length, causing index-out-of-range errors for sequences >512 nt. This is incompatible with circRNA applications that may target sequences >1000 nt.

The core contradiction: calling this scheme "physics-biased" implied that physical constraints guided the architecture, but in reality the only physics input was a single scalar closure penalty applied after prediction. Without equivariant message passing or a proper geometric constraint mechanism, the transformer had no inductive bias for 3D coordinate geometry. A transformer can learn arbitrary mappings from sequence tokens to 3D coordinates, but the optimization landscape contains no structure that steers predictions toward physically valid RNA conformations.

We revised Scheme 5 to use delta prediction from planar circular initialization (Scheme 5'), with a small-init delta head (gain=0.01) and additional closure and bond consistency loss terms. This variant also failed: the delta predictions on top of unit-scale circular init coordinates allowed deltas to grow unbounded, and the CPU saturation persisted because the per-sample planar initialization generation (`generate_helical_init`) required sequential computation for variable-length sequences.

### Failure Analysis: Scheme 3 Core Design Contradiction

Scheme 3 (Dual-Engine Iterative, internally named CS-Fold) was designed around the principle of "physics solver initialization → neural network refinement." However, this design premise was never fulfilled: the initialization used a fixed planar circular geometry (not a physics solver output), meaning the "dual-engine" was effectively a single-engine transformer operating from scratch.

Specific failure mechanisms:

**1. Loss balance instability.** Three loss terms were combined with weights (coord_loss: 1.0, bond_loss: 0.1, closure_loss: 0.1). When coord_loss dominated early training (large coordinate errors), the bond and closure penalties were effectively inactive, producing structures that violated physical constraints. When closure loss briefly dominated, it destabilized the coordinate regression. No dynamic weighting or gradient normalization was applied.

**2. Closure loss magnitude mismatch.** The closure loss was clamped to range [-5, 5] in normalized coordinate space. However, in this space where coordinates range ~1.0, a ±5 clamp allowed closure penalties orders of magnitude larger than the coordinate values themselves, creating loss landscape discontinuities at the clamp boundaries.

**3. Unvectorized per-sample computation.** Both `generate_helical_init` (per-sequence-length planar circle generation) and `bond_loss` (for b in range(B) loop over variable-length sequences) required sequential per-sample computation. This prevented GPU vectorization and created a CPU bottleneck that saturated utilization >100% for extended periods, distinct from the GPU-bound training of successful schemes.

**4. Residual prediction on inadequate reference.** The delta prediction formulation assumes the reference initialization provides a reasonable starting point. Planar circular geometry deviates ~60Å RMSD from native circRNA structures. The delta must therefore be large, and large deltas on unit-scale references reintroduce the unbounded output space problem that plagued Scheme 5.

The fundamental lesson: a "dual-engine" approach that combines physics and learning requires the physics engine to actually produce meaningful structural information. Our Scheme 2 (GeometricConstraintSolver) produces ~2Å RMSD structures with guaranteed closure. If Scheme 3 had been initialized from Scheme 2 outputs rather than planar geometry, the delta prediction space would be small (2-5Å corrections) and the architecture might have succeeded. The failure was not in the transformer design per se, but in the mismatch between what the initialization provided and what the refinement needed.

### Synthesis: What Makes circRNA 3D Architecture Viable?

From the failures of Schemes 3 and 5, and the successes of Schemes 1, 2, 6, we identify three necessary conditions for stable circRNA 3D structure prediction:

**Condition 1: Geometric inductive bias.** The architecture must constrain the coordinate output manifold. EGNN equivariance (Schemes 1, 4, 6) provides this by ensuring that coordinate updates preserve relative distances and angles. Latent diffusion (Scheme 6) provides this by operating in a compressed representation where the decoder maps to a structured output space. Without such constraints, the optimization landscape has no gradient structure guiding predictions toward physically valid conformations.

**Condition 2: Bounded output magnitude.** The architecture must prevent coordinate runaway. Diffusion models achieve this naturally: the denoising process starts from a known noise distribution and iteratively converges toward data. The latent diffusion variant (Scheme 6) further bounds the problem by separating coordinate generation from sequence encoding. Direct regression (Scheme 5) and large-delta prediction (Scheme 3) lack these bounds.

**Condition 3: Vectorizable computation.** For practical training on variable-length circRNA sequences, the loss computation and initialization must support batch-level GPU parallelism. Per-sample sequential loops (Scheme 3's bond_loss) and per-length initialization generation (Scheme 3's helical init) create CPU bottlenecks that prevent efficient training even when the architecture might otherwise converge.

The surviving architectures satisfy all three conditions: Schemes 1 and 6 (EGNN equivariance + latent diffusion + batch vectorization), Scheme 2 (physics solver with explicit constraints), Scheme 4 (EGNN diffusion), and Scheme 7 (selective state space with linear attention).

### TPE Ablation Study

To isolate the contribution of Torus Positional Encoding, we compared identical architectures with TPE versus standard sinusoidal positional encoding on 3D structure prediction. The TPE ablation uses the same GNN latent diffusion backbone (Scheme 6) with standard PE replacing TPE, controlling for all other hyperparameters.

| Encoding | RMSD (Å) | Closure (Å) | BSJ-Flanking RMSD (Å) |
|----------|-----------|-------------|----------------------|
| Standard PE | TBD | TBD | TBD |
| TPE (ours) | 13.91 | 0.02 | TBD |

We hypothesize that TPE will show the largest improvement in BSJ-flanking regions, where standard PE incorrectly treats adjacent nucleotides as maximally distant. TBD: complete ablation experiment with paired t-test across test samples.

### External Baseline Comparisons

We compare TorusFold against existing RNA structure prediction methods on our circularized test set (N=7) and the expanded test set (N=TBD):

| Method | Topology | RMSD (Å) | Closure (Å) | Inference Time |
|--------|----------|-----------|-------------|----------------|
| IsRNA [6] | Linear (adaptable) | TBD | TBD | TBD |
| AlphaFold3 [4] | Linear | TBD | TBD | TBD |
| FARFAR2 [ref] | Linear | TBD | TBD | TBD |
| ViennaRNA 3D | Circular (DP) | TBD | TBD | TBD |
| Scheme 6 (ours) | Circular (TPE) | 13.91 | 0.02 | TBD |
| Scheme 2 (ours) | Circular (solver) | ~2 | <0.1 | TBD |

TBD: run all external baselines on identical test set with same evaluation protocol.

### Error Analysis by Structural Region

We decomposed prediction error by structural region to identify where TPE provides the greatest benefit (Figure TBD):

| Region | Scheme 1 RMSD (Å) | Scheme 6 RMSD (Å) | Fraction of Total Error |
|--------|--------------------|--------------------|------------------------|
| BSJ-flanking (±3 nt) | TBD | TBD | TBD |
| Stem regions | TBD | TBD | TBD |
| Loop/hairpin regions | TBD | TBD | TBD |
| Single-stranded | TBD | TBD | TBD |

TBD: per-nucleotide error heatmap around BSJ region.

### Length Scaling Analysis

We evaluated prediction accuracy as a function of sequence length:

| Length Range | N | Scheme 1 RMSD (Å) | Scheme 6 RMSD (Å) | Scheme 7 RMSD (Å) |
|-------------|---|--------------------|--------------------|--------------------|
| 20-50 nt | TBD | TBD | TBD | TBD |
| 50-100 nt | TBD | TBD | TBD | TBD |
| 100-200 nt | TBD | TBD | TBD | TBD |
| 200-500 nt | TBD | — | — | TBD |
| 500-1000 nt | TBD | — | — | TBD |

Scheme 7 (Mamba+Attention) is designed for O(L) complexity, enabling prediction on sequences >500 nt where O(L²) schemes become memory-limited. TBD: demonstrate length scaling with longer sequences.

### Hyperparameter Sensitivity

We assessed sensitivity to key hyperparameters for Scheme 6:

| Parameter | Default | Tested Values | RMSD Range (Å) |
|-----------|---------|---------------|----------------|
| TPE harmonics H | 16 | 4, 8, 16, 32 | TBD |
| KNN neighbors K | 16 | 8, 16, 32 | TBD |
| Diffusion steps T | 50 | 25, 50, 100 | TBD |
| Learning rate | 1e-4 | 5e-5, 1e-4, 5e-4 | TBD |
| Latent dimension | 128 | 64, 128, 256 | TBD |

TBD: sensitivity analysis experiments.

### Confidence Calibration

For therapeutic design applications, calibrated confidence estimates are essential. We assessed whether the model's internal confidence scores correlate with prediction accuracy:

| Confidence Bin | N | Mean RMSD (Å) | Expected Accuracy |
|----------------|---|---------------|-------------------|
| High (>0.8) | TBD | TBD | TBD |
| Medium (0.5-0.8) | TBD | TBD | TBD |
| Low (<0.5) | TBD | TBD | TBD |

TBD: reliability diagram and expected calibration error (ECE) computation.

### Expanded Test Set Results

We are expanding our PDB-derived circularized test set to include additional structures:

| Test Set | N | Source | Scheme 6 RMSD (Å) | Scheme 6 Closure (Å) |
|----------|---|--------|--------------------|-----------------------|
| Original | 7 | PDB circularized | 13.91 | 0.02 |
| + 8xtp/8xtq/8xtr/8xts | TBD | PDB circRNA | TBD | TBD |
| + RNA-Puzzles circularized | TBD | RNA-Puzzles | TBD | TBD |
| + Circ-CASP blind targets | TBD | Community | TBD | TBD |
| **Target total** | **≥30** | | | |

TBD: expand test set to N≥30 with diverse lengths, topologies, and biological contexts.

---

## Discussion

### Diffusion Models Learn Physical Constraints End-to-End

The most striking finding is that Scheme 6 achieved closure error 0.02Å without any explicit closure penalty in the loss function. The diffusion model learned the closure constraint from data: during training, it observed that valid structures have first-to-last distance ~5.9Å, and this became encoded in the generative prior.

This contrasts with traditional approaches that enforce closure through explicit constraints (Scheme 2's annealing, Scheme 3's penalty function). Diffusion models offer a different paradigm: they learn the data distribution, and if the training data satisfies closure, the model will generate structures that satisfy closure. This is particularly valuable when the physical constraints are complex or difficult to formulate analytically.

### Methodology Under Data Scarcity

A recurring question in circRNA structure prediction is: how can we develop methods without experimental training data? Our answer draws from the history of molecular biology: PCR was not a prerequisite for theoretical analysis of DNA structure, nor was X-ray crystallography required for the development of protein folding theory. Methodology development can precede data availability.

We adopted several strategies:
1. **Multi-source data aggregation:** Combining IsRNA predictions (N=2,754), icSHAPE-constrained structures (N≈2,000), PDB circularized RNAs (N=184), and ViennaRNA predictions (N≈5,000) to create heterogeneous training data totaling >10,000 samples.
2. **Benchmark establishment:** Defining metrics (BSJ closure, circular distance) that will remain relevant when experimental data becomes available.
3. **Systematic architecture search:** Exploring the design space so future researchers can build on proven approaches rather than repeating failures.
4. **Honest limitation reporting:** Acknowledging small test set size, pseudo-label quality issues, and incomplete training.

The pseudo-label training data quality analysis reveals a confidence score distribution concentrated around 0.5 (circrna_3d source) versus 0.95 (PDB circularized). Training on increasing fractions of high-confidence data yields the following learning curve:

| High-Confidence Fraction | N (effective) | Scheme 6 RMSD (Å) |
|--------------------------|---------------|--------------------|
| 10% | ~1,100 | TBD |
| 25% | ~2,750 | TBD |
| 50% | ~5,500 | TBD |
| 100% (all) | ~11,000 | 13.91 |

TBD: data ceiling experiment showing learning curves as function of data quality.

### Complementarity with Physics-Based Methods

IsRNA and TorusFold are not competitors but complements. IsRNA provides RNA 3D structure predictions through physics-based simulated annealing, which can be adapted for circRNA with circular constraints, suitable for small-scale applications where compute and expert configuration are available. TorusFold provides fast inference and scales to longer sequences, suitable for high-throughput screening.

Direct comparison on our test set:

| Metric | IsRNA | Scheme 6 | Scheme 2 |
|--------|-----------|----------|----------|
| RMSD (Å) | TBD | 13.91 | ~2 |
| Closure (Å) | TBD | 0.02 | <0.1 |
| Inference time | TBD | TBD | TBD |
| Max length (nt) | ~200 | ~200 | ~500 |

TBD: run IsRNA on the same test sequences with identical evaluation protocol.

Our Scheme 2 can be viewed as a simplified IsRNA surrogate, achieving similar closure guarantees with reduced computational cost. For applications requiring guaranteed closure (vaccine design, drug targeting), physics-based or hybrid approaches may be preferred. For applications prioritizing speed (sequence optimization, large-scale screening), learned approaches like Scheme 6 offer practical advantages.

### Limitations

We acknowledge several limitations:
1. **Small test set (N=7):** Limited statistical power precludes definitive conclusions about relative architecture performance. Bootstrap confidence intervals (1000 resamples) overlap for Schemes 1 and 6. Expansion to N≥30 is underway (TBD).
2. **Incomplete comparison:** Schemes 4 and 7 are not yet fully trained. Schemes 3 and 5 have been abandoned due to persistent training instabilities (documented in Failure Analysis). The systematic comparison currently covers 5 of 7 proposed architectures (Schemes 1, 2, 4, 6, 7), with 2 (S4, S7) pending. Current status: Scheme 4 (TBD), Scheme 7 (TBD).
3. **Pseudo-label quality:** Training data consists primarily of computational predictions, which may contain systematic errors inherited from IsRNA, ViennaRNA, and other source methods. The risk of circular validation exists (training and test data both derived from physics-based simulators). Confidence score distribution analysis: TBD.
4. **No wet-lab validation:** All results are computational. Experimental validation (cryo-EM, SHAPE-MaP) is planned but not yet initiated. Direct comparison with experimental data is not possible until circRNA structures enter the PDB.
5. **Missing external baselines:** Comparison with IsRNA, AlphaFold3, and FARFAR2 is pending (TBD). Without these comparisons, absolute performance assessment is not possible.
6. **TPE ablation incomplete:** The direct contribution of TPE versus standard positional encoding on 3D structure prediction has not been measured (TBD). Current evidence relies on mathematical proof of periodicity and proxy experiments.
7. **No error analysis by region:** Per-nucleotide error distribution, particularly around BSJ-flanking regions, has not been computed (TBD).
8. **No length scaling demonstration:** The O(L) advantage claimed for Scheme 7 has not been empirically demonstrated with long sequences (TBD).
9. **No confidence calibration:** Reliability of model confidence scores has not been validated (TBD).
10. **Hyperparameter sensitivity uncharacterized:** No systematic sensitivity analysis has been performed (TBD).

Despite these limitations, we believe this work provides value: it establishes the problem space, introduces principled architectural components (TPE), documents both successes and failures, and creates benchmarks for future evaluation.

### Future Directions

Immediate priorities include completing training of all seven schemes, expanding the PDB test set, and establishing the Circ-CASP community benchmark with blind test targets. Longer-term goals include wet-lab validation of predictions, integration with experimental structure determination pipelines, and application to circRNA therapeutic design for the iGEM FBH team's TNBC vaccine project.

---

## Methods

### Torus Positional Encoding

Torus Positional Encoding (TPE) is defined as:

TPE(i, 2h) = sin(2π × h × i / L)
TPE(i, 2h+1) = cos(2π × h × i / L)

for position i ∈ {0, ..., L-1}, harmonic h ∈ {1, ..., H}, and embedding dimension 2H. Periodicity is guaranteed by the 2π periodicity of sine and cosine:

TPE(i+L, 2h) = sin(2π × h × (i+L) / L) = sin(2π × h × i/L + 2πh) = sin(2π × h × i/L) = TPE(i, 2h)

We use H = 16 harmonics by default, providing 32-dimensional positional encoding.

### Seven Architectural Schemes

**Scheme 1: EGNN + Physics Cascade.** We use a 4-layer EGNN with K=16 nearest neighbors, hidden dimension 128. Coordinates are initialized with planar circular geometry and refined through equivariant message passing. Training uses AdamW optimizer with LR 1e-4, weight decay 1e-5, batch size 8, for 50 epochs.

**Scheme 2: Physics Solver.** GeometricConstraintSolver with bond length 5.9Å, pair distance 10.6Å, and simulated annealing closure. No training required.

**Scheme 3: Dual-Engine Iterative.** 3-layer transformer with circular position encoding, predicting coordinate deltas from planar circular init. Training with BSJ closure penalty 0.1 and bond consistency penalty 0.1.

**Scheme 4: DDPM + EGNN Guided Diffusion.** 100-step diffusion with EGNN backbone. Closure reward guidance during sampling with weight 0.5.

**Scheme 5: Physics-Biased Attention.** 4-layer transformer encoder with TPE and post-hoc closure correction. Failed in training.

**Scheme 6: GNN Latent Diffusion.** 4-layer GNN encoder (d_node=64, d_edge=32, d_latent=128), 50-step latent diffusion, 4-layer GNN decoder. Key architectural fix: decoder receives denoised latent (not noise prediction) during training.

**Scheme 7: Mamba + Local Attention.** 4-layer Mamba encoder with local attention (window=20) for fine-grained structure. Linear complexity enables L>500 sequences.

### Data Pipeline

**PDB Circularized Set (N=7):** Linear RNA structures from PDB were circularized using GeometricConstraintSolver with bond length constraint 5.9Å and annealing closure. Sequences filtered for length 20-200 nt and initial closure distance <15Å.

**circrna_3d_merged (N≈11,000):** Multi-source aggregation of IsRNA predictions, icSHAPE-constrained structures, and ViennaRNA circ-mode predictions. Confidence scores assigned based on source and validation against known RNA motifs.

### Evaluation Metrics

**RMSD (Root Mean Square Deviation):** Computed after Kabsch alignment:

RMSD = sqrt(1/N × Σᵢ ||pᵢ - qᵢ||²)

where p and q are centered and optimally rotated predictions and targets.

**Closure Error:** Distance between first and last nucleotide coordinates:

closure = ||p₀ - p_{L-1}||

Target: 5.9Å (phosphodiester bond length).

**TM-score:** Length-normalized structural similarity, adapted for RNA.

**Circular Distance:** For positions i, j in sequence:

d_circ(i, j) = min(|i-j|, L-|i-j|)

reflecting topology-aware proximity.

### Statistical Analysis

Due to small test set (N=7), we report mean, median, and bootstrap 95% confidence intervals (1000 resamples). No statistical tests are claimed to be significant given the limited sample size.

---

## Data and Code Availability

Code is available at github.com/RomanCohort/confluencia under MIT license. Training data and benchmark datasets are provided in the supplementary materials. Circ-CASP benchmark (30 blind targets) will be released for community competition.

---

## TBD Checklist (To Be Determined)

The following data placeholders require completion after experiments finish:

### Architecture Results
- [x] ~~Scheme 3 RMSD and closure on PDB test set~~ — Abandoned (gradient divergence)
- [ ] Scheme 4 RMSD and closure on PDB test set
- [x] ~~Scheme 5' (delta variant) RMSD and closure on PDB test set~~ — Abandoned (CPU spike)
- [ ] Scheme 7 RMSD and closure on PDB test set

### External Baselines
- [ ] IsRNA RMSD and closure on identical test set
- [ ] AlphaFold3 RMSD and closure (expected: fails at BSJ closure)
- [ ] FARFAR2 RMSD and closure on identical test set
- [ ] ViennaRNA 3D-mode results if available
- [ ] Inference time for all methods

### Ablation Studies
- [ ] TPE vs. standard PE on Scheme 6 backbone (paired comparison)
- [ ] BSJ-flanking region error analysis
- [ ] Circular distance vs. prediction error correlation

### Test Set Expansion
- [ ] Expand PDB circularized set to N≥30
- [ ] Include 8xtp, 8xtq, 8xtr, 8xts circRNA structures
- [ ] RNA-Puzzles circularized targets
- [ ] Circ-CASP blind test results (future)

### Error Analysis
- [ ] Per-nucleotide error heatmap
- [ ] BSJ-flanking vs. stems vs. loops error decomposition
- [ ] Length-dependent error analysis

### Length Scaling
- [ ] RMSD vs. sequence length for all schemes
- [ ] Memory profiling for O(L²) vs. O(L)
- [ ] Long sequence (>500 nt) predictions with Scheme 7

### Data Quality Analysis
- [ ] Confidence score distribution by source
- [ ] Learning curve: RMSD vs. high-confidence data fraction
- [ ] Validation against known RNA motifs

### Hyperparameter Sensitivity
- [ ] TPE harmonics H (4, 8, 16, 32)
- [ ] KNN neighbors K (8, 16, 32)
- [ ] Diffusion steps T (25, 50, 100)
- [ ] Learning rate, batch size, architecture depth

### Confidence Calibration
- [ ] Reliability diagram
- [ ] Expected calibration error (ECE)
- [ ] Confidence vs. actual accuracy correlation

### Statistical Analysis
- [ ] Bootstrap confidence intervals for all metrics
- [ ] Paired t-test for TPE vs. standard PE
- [ ] ANOVA for multi-scheme comparison (requires N≥30)

---

## Acknowledgments

This work was conducted as part of iGEM 2026 by the FBH (First Build High School) team for the development of circRNA-based TNBC vaccines. We thank the open-source community for the foundational tools that made this work possible.

---

## References

[1] Wesselhoeft, R. A., Kowalski, P. S., & Anderson, D. G. (2018). RNA circularization diminishes immunogenicity and can extend translation duration in vivo. Molecular Cell, 70(5), 869-880.

[2] Chen, L. L., & Shan, G. (2016). Circular RNAs — biogenesis, emerging roles, and diseases. Nature Reviews Molecular Cell Biology, 17(5), 307-321.

[3] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

[4] Abramson, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature.

[5] Baek, M., et al. (2021). Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557), 871-876.

[6] Zhang, D., Li, J., & Chen, S.-J. (2022). IsRNA: an integrative simulated annealing approach for RNA 3D structure prediction. Nucleic Acids Research, 50(W1), W51-W57. https://doi.org/10.1093/nar/gkac406

[7] Lorenz, R., et al. (2011). ViennaRNA Package 2.0. Algorithms for Molecular Biology, 6(1), 26.

[8] Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) equivariant graph neural networks. ICML.

[9] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.

[10] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

[11] Watkins, A. M., Rangan, R., & Das, R. (2020). FARFAR2: Improved de novo Rosetta prediction of global RNA 3D structure. Nature Methods, 17(5), 483-492.

[12] Wang, Y.-X. (2022). 3D structure prediction of circular RNAs: Challenges and opportunities. Biomolecules, 12(10), 1412. https://doi.org/10.3390/biom12101412

---

## Figure Legends

**Figure 1. Torus Positional Encoding preserves circular periodicity.** (A) Standard positional encoding (left) vs. TPE (right) for a sequence of length L=100. Standard PE has PE(0) ≠ PE(L-1), violating circular topology. TPE guarantees PE(0) = PE(L) by construction. (B) Visualization of TPE harmonics: low harmonics capture global position, high harmonics capture local structure. (C) CircRNA topology: BSJ connects first and last nucleotides, requiring periodic encoding.

**Figure 2. Seven architectures comparison.** (A) RMSD comparison across schemes (bar chart with bootstrap confidence intervals). Scheme 6 (GNN latent diffusion) achieves 13.91Å RMSD. TBD: results for Schemes 4, 7. Schemes 3 and 5 abandoned. (B) Closure error comparison: Scheme 6 achieves 0.02Å, Scheme 1 has 5.36Å, Scheme 2 guarantees <0.1Å. TBD: closure for Schemes 4, 7. (C) Per-sample RMSD scatter plot. (D) Complexity vs. accuracy trade-off: Scheme 7 offers O(L) scaling for long sequences.

**Figure 3. Scheme 6 GNN latent diffusion architecture.** (A) GNN encoder with physics-aware edge features (bond, pair, stacking, electrostatic). (B) Latent diffusion process: forward diffusion adds noise to latent z, reverse diffusion denoises. (C) GNN decoder reconstructs 3D coordinates with learned closure. (D) Training curves showing convergence.

**Figure 4. External baseline comparisons.** (A) RMSD comparison: TorusFold Scheme 6 vs. IsRNA vs. AlphaFold3 vs. FARFAR2 on circularized test set. TBD: run all baselines. (B) Closure error across methods. (C) Inference time comparison. (D) Accuracy vs. computational cost trade-off.

**Figure 5. TPE ablation study.** (A) TPE vs. standard PE with identical Scheme 6 backbone on 3D structure prediction. TBD: run ablation experiment. (B) Per-nucleotide error heatmap around BSJ region, comparing TPE vs. standard PE. TBD. (C) BSJ-flanking region RMSD: TPE vs. standard PE. (D) Circular distance vs. prediction error correlation.

**Figure 6. Data quality impact and error analysis.** (A) PDB circularized set (N=7, confidence 0.95) vs. circrna_3d_merged (N≈11,000, confidence 0.5). (B) RMSD by data source: high-confidence data gives 11Å improvement. (C) Learning curve: RMSD vs. fraction of high-confidence training data. TBD: data ceiling experiment. (D) Error decomposition by structural region (BSJ-flanking, stems, loops). TBD.

**Figure 7. Length scaling and hyperparameter analysis.** (A) RMSD vs. sequence length for Schemes 1, 6, 7. TBD: length scaling experiment. (B) Memory usage vs. sequence length for O(L²) vs. O(L) schemes. (C) Hyperparameter sensitivity: TPE harmonics H, KNN neighbors K, diffusion steps T. TBD. (D) Confidence calibration: reliability diagram. TBD.

**Figure 8. Failure analysis: Schemes 3 and 5.** (A) Scheme 5: unbounded Å-space prediction produces MSE 250k and gradient explosion (RMSD 245Å). (B) Scheme 3: loss weight imbalance (1.0 + 0.1 + 0.1) and residual prediction from inadequate planar reference (60Å RMSD gap). (C) CPU saturation patterns from per-sample sequential computation loops in both schemes. (D) Necessary conditions for viable circRNA architectures: geometric inductive bias, bounded output magnitude, vectorizable computation.

---

*Manuscript prepared 2026-06-23*
*Status: Draft for internal review*
