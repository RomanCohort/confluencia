# TorusFold Supplementary Experiments Plan
## Response to Peer Review Major Revision

**Document Date:** 2026-06-27
**Response to:** Nature Methods Round 1 Review (3 reviewers)
**Target Completion:** 4-6 weeks

---

## Summary of Critical Issues

| Priority | Issue | Reviewer(s) | Current Status |
|----------|-------|-------------|----------------|
| 🔴 Critical | Test set size N=7 → N≥30 | All 3 | Pending |
| 🔴 Critical | External baselines missing | All 3 | TBD |
| 🔴 Critical | TPE ablation incomplete | All 3 | TBD |
| 🔴 Critical | Closure error validation | R1, R2 | Partial (added text) |
| 🟡 Important | Architectural details | R2 | Added in revision |
| 🟡 Important | Error analysis by region | R1, R2 | Pending |
| 🟡 Important | Computational cost comparison | R1, R3 | Pending |

---

## Experiment 1: Test Set Expansion (N=7 → N≥30)

### Objective
Expand test set from 7 sequences to ≥30 sequences with diverse lengths and structural motifs to enable statistical significance testing.

### Current Test Set
| Sequence | Length | Source | Structural Class |
|----------|--------|--------|------------------|
| PDB_1 | 20-27 nt | Circularized | Hairpin |
| ... | ... | ... | ... |
| Total: 7 | 20-27 nt | PDB circularized | Limited diversity |

### Proposed Expanded Test Set

#### Part A: PDB Circularized (N=15-20)
1. **Short sequences (20-50 nt):** 5-7 samples
   - PDB IDs: 1RNA, 2RNA, 5RNA, 7RNA (tRNA fragments, ribozymes)
   
2. **Medium sequences (50-100 nt):** 5-7 samples
   - PDB IDs: RNA aptamers, small ribozymes
   
3. **Longer sequences (100-200 nt):** 3-5 samples
   - PDB IDs: RNA domains, ribosomal RNA fragments

#### Part B: Synthetic Benchmark (N=10-15)
1. **Hairpin variants:** 3-4 samples with different stem lengths
2. **Internal loops:** 2-3 samples
3. **Pseudoknots:** 2-3 samples (critical for circRNA function)
4. **Multi-branch junctions:** 2-3 samples

#### Part C: Experimental Validation (if available)
- Any experimental circRNA structures from recent PDB entries
- icSHAPE-constrained sequences with known secondary structure

### Implementation Protocol

```python
# Step 1: Download additional PDB RNA structures
python fetch_pdb_rna.py --output data/pdb_expanded --min_len 20 --max_len 200

# Step 2: Circularize using GeometricConstraintSolver
python pdb_rna_circularize.py --input data/pdb_expanded --output data/pdb_circularized_expanded

# Step 3: Generate synthetic benchmark
python generate_benchmark_structures.py --output data/synthetic_benchmark --n_samples 15

# Step 4: Merge expanded test set
python merge_expanded_dataset.py --pdb-dir data/pdb_circularized_expanded --syn-dir data/synthetic_benchmark --output data/test_set_expanded

# Step 5: Evaluate on expanded set
python evaluate_scheme6_expanded.py --test-set data/test_set_expanded --output results/scheme6_expanded.json
```

### Statistical Analysis Protocol

```python
import numpy as np
from scipy import stats

# Bootstrap confidence intervals
def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = np.mean(bootstrap_samples, axis=1)
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper

# Paired t-test for TPE vs standard PE
def paired_comparison(rmsd_tpe, rmsd_std):
    t_stat, p_value = stats.ttest_rel(rmsd_tpe, rmsd_std)
    return t_stat, p_value

# Report format
print(f"RMSD: {mean:.2f} Å [{lower:.2f}, {upper:.2f}] 95% CI")
print(f"Paired t-test TPE vs StdPE: t={t_stat:.3f}, p={p_value:.4f}")
```

### Expected Outputs

| Metric | Current (N=7) | Target (N=30) |
|--------|---------------|---------------|
| RMSD mean ± CI | 13.91 ± N/A | 13-15 ± [X, Y] 95% CI |
| Closure mean ± CI | 0.02 ± N/A | 0.02-0.5 ± [X, Y] 95% CI |
| Significance test | N/A | Paired t-test available |
| Length scaling | N/A | Correlation analysis |

### Timeline
- Week 1-2: PDB structure collection and circularization
- Week 2-3: Synthetic benchmark generation
- Week 3-4: Evaluation runs on all schemes
- Week 4: Statistical analysis and figure generation

---

## Experiment 2: External Baseline Comparisons

### Objective
Compare TorusFold with established RNA 3D prediction methods: IsRNAcirc, FARFAR2, and AlphaFold3.

### 2A: IsRNAcirc Comparison

**Method:** IsRNAcirc web server or local installation
**Input:** Same N=30 test sequences
**Metrics:** RMSD, closure error, inference time

```bash
# Protocol
# 1. Prepare sequences for IsRNAcirc
python prepare_isrna_input.py --test-set data/test_set_expanded --output isrna_input.fasta

# 2. Run IsRNAcirc (web server batch mode or local)
# Web: Submit to http://isrna.circRNA.org/batch
# Local:
python run_isrna_local.py --input isrna_input.fasta --output results/isrna_predictions

# 3. Evaluate IsRNA outputs
python evaluate_external.py --pred results/isrna_predictions --true data/test_set_expanded --output results/isrna_eval.json
```

**Expected Result Format:**
| Method | RMSD (Å) | Closure (Å) | Inference (s) |
|--------|----------|-------------|---------------|
| IsRNAcirc | 18-25? | 8-12? | 300 |
| Scheme 6 | 13.91 | 0.02 | 45 |

**Critical Finding to Document:** IsRNAcirc may achieve lower RMSD but higher closure error (no circRNA-specific mode).

### 2B: FARFAR2 Comparison

**Method:** FARFAR2 from Rosetta package
**Configuration:**
- Standard mode (linear RNA)
- Circ-constrained mode (if available)

```bash
# Protocol
# 1. Install FARFAR2
# Download from: https://github.com/RosettaCommons/RNA

# 2. Run FARFAR2 on test sequences
python run_farfar2.py --input data/test_set_expanded --output results/farfar2_predictions --n-structures 10

# 3. Evaluate
python evaluate_external.py --pred results/farfar2_predictions --true data/test_set_expanded --output results/farfar2_eval.json
```

### 2C: AlphaFold3 Comparison

**Method:** AlphaFold3 (if API available) or ColabFold
**Critical:** AF3 cannot enforce BSJ closure - document this as failure mode

```bash
# Protocol
# 1. Prepare AF3 input
python prepare_af3_input.py --test-set data/test_set_expanded --output af3_input.fasta

# 2. Run AF3 (ColabFold or local if available)
python run_alphafold3.py --input af3_input.fasta --output results/af3_predictions

# 3. Evaluate with special focus on closure failure
python evaluate_af3_closure.py --pred results/af3_predictions --output results/af3_closure_analysis.json
```

**Expected Findings:**
- AF3 RMSD: competitive (5-15 Å on RNA)
- AF3 Closure: **HIGH (>10 Å)** - AF3 treats circRNA as linear, producing open structures
- **This validates the need for torus-aware encoding**

### Timeline
- Week 1: IsRNAcirc runs (web server)
- Week 2-3: FARFAR2 installation and runs
- Week 3-4: AF3 runs and closure analysis
- Week 4: Figure generation and manuscript update

---

## Experiment 3: TPE Ablation Study

### Objective
Empirically validate that TPE provides benefit over standard positional encoding.

### 3A: Same Backbone Comparison

**Setup:** Train identical Scheme 6 architecture with:
- Condition A: TPE (current implementation)
- Condition B: Standard sinusoidal PE (Vaswani et al.)

```python
# Training script
# Condition A: TPE (existing)
python train_scheme6.py --pe-type tpe --output models/scheme6_tpe --epochs 500

# Condition B: Standard PE
python train_scheme6.py --pe-type standard --output models/scheme6_stdpe --epochs 500

# Evaluate both
python evaluate_scheme6.py --model models/scheme6_tpe --output results/scheme6_tpe.json
python evaluate_scheme6.py --model models/scheme6_stdpe --output results/scheme6_stdpe.json
```

### 3B: BSJ-Flanking Region Analysis

**Definition:** BSJ-flanking region = positions within 10-15 nt of junction (positions 0-15 and L-15 to L-1)

```python
def analyze_bsj_flanking(rmsd_per_position, L):
    """Compute RMSD specifically for BSJ-flanking region"""
    flanking_positions = list(range(0, 15)) + list(range(L-15, L))
    flanking_rmsd = np.mean([rmsd_per_position[i] for i in flanking_positions])
    global_rmsd = np.mean(rmsd_per_position)
    return flanking_rmsd, global_rmsd

# Compare
tpe_flanking, tpe_global = analyze_bsj_flanking(results_tpe['rmsd_per_pos'], L)
stdpe_flanking, stdpe_global = analyze_bsj_flanking(results_stdpe['rmsd_per_pos'], L)

print(f"TPE: Global={tpe_global:.2f}Å, BSJ-flanking={tpe_flanking:.2f}Å")
print(f"StdPE: Global={stdpe_global:.2f}Å, BSJ-flanking={stdpe_flanking:.2f}Å")
```

### 3C: Circular Distance Correlation

**Hypothesis:** TPE should reduce prediction error at positions with high circular distance (near BSJ).

```python
def circular_distance_correlation(rmsd_per_position, L):
    """Correlation between circular distance from BSJ and prediction error"""
    circular_dist = np.array([min(i, L-i) for i in range(L)])
    correlation = np.corrcoef(circular_dist, rmsd_per_position)[0, 1]
    return correlation

# Compare
corr_tpe = circular_distance_correlation(results_tpe['rmsd_per_pos'], L)
corr_stdpe = circular_distance_correlation(results_stdpe['rmsd_per_pos'], L)

print(f"Correlation (error vs circular distance):")
print(f"  TPE: r={corr_tpe:.3f}")
print(f"  StdPE: r={corr_stdpe:.3f}")
# Expect: StdPE has stronger correlation (higher error near BSJ)
```

### Expected Results Table

| Metric | TPE | Standard PE | Δ Improvement |
|--------|-----|-------------|---------------|
| Global RMSD | 13.91 Å | 14-15 Å | 0.5-1 Å |
| BSJ-flanking RMSD | 12-13 Å | 16-18 Å | **3-5 Å** |
| Circular distance correlation | ~0.1 | ~0.4 | Lower is better |
| Closure error | 0.02 Å | 0.5-2 Å | Significant |

### Timeline
- Week 1-2: Train both conditions
- Week 3: Evaluate and analyze
- Week 3-4: Statistical testing and figure generation

---

## Experiment 4: Closure Error Validation

### Objective
Verify that closure constraint is learned from data, not memorized.

### 4A: Training on Broken Structures

**Setup:** Train Scheme 6 on synthetic data with deliberately broken closure.

```python
# Generate broken closure data
def generate_broken_closure_data(sequences, coords):
    """Remove BSJ constraint by breaking circular bond"""
    broken_coords = []
    for c in coords:
        # Shift last nucleotide away from first
        c[-1] = c[-1] + np.random.randn(3) * 20  # Random displacement
        broken_coords.append(c)
    return broken_coords

# Train on broken data
python train_scheme6.py --data broken_closure_data --output models/scheme6_broken --epochs 100

# Evaluate on normal circular test set
python evaluate_scheme6.py --model models/scheme6_broken --test-set data/test_set_expanded --output results/broken_closure_eval.json
```

### Expected Findings

| Training Data | Test Closure Error | Interpretation |
|---------------|--------------------|----------------|
| Circular (closed) | 0.02 Å | Learned constraint |
| Linear/broken | 2-5 Å | Constraint NOT learned |
| Random/no structure | >10 Å | No constraint |

**If broken training still produces closed structures:** Constraint is architectural/implicit
**If broken training produces open structures:** Constraint is learned from data distribution

### 4B: Cross-Length Generalization

**Test:** Evaluate closure error on sequences of different lengths than training.

```python
# Train on lengths 20-50 nt
python train_scheme6.py --min-len 20 --max-len 50 --output models/scheme6_short

# Evaluate on lengths 50-100 nt (held out during training)
python evaluate_scheme6.py --model models/scheme6_short --test-set data/test_long --output results/long_generalization.json

# Report closure variance across lengths
```

### Timeline
- Week 1: Generate broken closure data
- Week 2: Training experiments
- Week 3: Cross-length evaluation
- Week 3-4: Analysis and reporting

---

## Experiment 5: Error Analysis by Structural Region

### Objective
Quantify prediction error in different structural regions to understand where TPE helps.

### Structural Regions Classification

```python
def classify_structural_regions(sequence, secondary_structure):
    """Classify each position into structural region"""
    regions = {
        'bsj_flanking': [],      # Within 10 nt of junction
        'stem': [],              # Base-paired, helical region
        'loop_hairpin': [],      # Unpaired in hairpin
        'internal_loop': [],     # Unpaired between stems
        'single_stranded': [],   # Unpaired, not in defined loop
    }
    
    # Parse dot-bracket notation
    pairs = extract_pairs_from_dotbracket(secondary_structure)
    
    # Classify each position
    for i in range(len(sequence)):
        if i < 10 or i >= len(sequence) - 10:
            regions['bsj_flanking'].append(i)
        elif is_paired(i, pairs):
            regions['stem'].append(i)
        elif in_hairpin(i, secondary_structure):
            regions['loop_hairpin'].append(i)
        elif in_internal_loop(i, secondary_structure):
            regions['internal_loop'].append(i)
        else:
            regions['single_stranded'].append(i)
    
    return regions

# Analyze RMSD by region
def rmsd_by_region(pred_coords, true_coords, regions):
    rmsd_per_region = {}
    for region_name, positions in regions.items():
        region_pred = pred_coords[positions]
        region_true = true_coords[positions]
        rmsd = compute_rmsd(region_pred, region_true)
        rmsd_per_region[region_name] = rmsd
    return rmsd_per_region
```

### Expected Results Format

| Region | Mean RMSD (Å) | Std | % of Total Error |
|--------|---------------|-----|------------------|
| BSJ-flanking | 12.5 | 1.5 | 35% |
| Stem | 10.2 | 2.0 | 25% |
| Loop/hairpin | 15.0 | 3.0 | 20% |
| Internal loop | 18.0 | 4.0 | 15% |
| Single-stranded | 12.0 | 2.5 | 5% |

**Hypothesis:** BSJ-flanking region should show largest TPE improvement.

### Timeline
- Week 1: Implement region classification
- Week 2-3: Run analysis on all test samples
- Week 3-4: Generate heatmap figure and update manuscript

---

## Experiment 6: Computational Cost Comparison

### Objective
Report inference time and memory usage for each scheme.

### Protocol

```python
import time
import torch

def benchmark_inference(model, sequence, device='cuda'):
    """Measure inference time and memory"""
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    with torch.no_grad():
        output = model(sequence)
    end_time = time.time()
    
    inference_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    
    return inference_time, peak_memory

# Benchmark across lengths
lengths = [50, 100, 200, 500, 1000]
for L in lengths:
    for scheme in [1, 2, 6, 7]:
        time, mem = benchmark_inference(models[scheme], generate_sequence(L))
        print(f"Scheme {scheme}, L={L}: time={time:.2f}s, mem={mem:.2f}GB")
```

### Expected Results Table

| Scheme | L=50 | L=100 | L=200 | L=500 | L=1000 |
|--------|------|-------|-------|-------|--------|
| **Time (s)** |
| Scheme 1 | 0.1 | 0.3 | 1.2 | 6.0 | OOM |
| Scheme 2 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
| Scheme 6 | 0.2 | 0.5 | 2.0 | 10.0 | OOM |
| Scheme 7 | 0.1 | 0.2 | 0.4 | 1.0 | 2.0 |
| **Memory (GB)** |
| Scheme 1 | 0.5 | 1.0 | 4.0 | 16.0 | OOM |
| Scheme 2 | 0.1 | 0.2 | 0.4 | 1.0 | 2.0 |
| Scheme 6 | 1.0 | 2.0 | 8.0 | 32.0 | OOM |
| Scheme 7 | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |

### Timeline
- Week 1: Implement benchmarking script
- Week 2: Run on all schemes
- Week 2-3: Generate Figure 7 update

---

## Summary Timeline

| Week | Experiments | Deliverables |
|------|-------------|--------------|
| 1 | Test set expansion, IsRNAcirc runs | Expanded test set, preliminary external results |
| 2 | FARFAR2 runs, TPE training start | External baseline figure draft |
| 3 | AF3 runs, TPE eval, closure validation | TPE ablation results |
| 4 | Error analysis, benchmarking | All figures, manuscript revision |

**Total: 4 weeks minimum, 6 weeks buffer for complications**

---

## Resource Requirements

### Computational Resources
- GPU: 1-2 NVIDIA RTX 3090 or A100 (for training)
- Storage: 50-100 GB for expanded datasets
- Time: ~200 GPU hours for all training runs

### External Resources
- IsRNAcirc web server (free)
- FARFAR2 license (Rosetta, academic free)
- AlphaFold3 ColabFold (free tier sufficient)

### Personnel
- 1-2 people for experiment execution
- Weekly sync meetings for progress tracking

---

## Expected Manuscript Updates

### New Figures
- **Figure 4 (revised):** External baseline comparisons with actual data
- **Figure 5 (revised):** TPE ablation with BSJ-flanking heatmap
- **Figure 6 (revised):** Error decomposition by structural region
- **Figure 7 (revised):** Computational cost benchmark

### New Tables
- **Table 2:** Expanded test set statistics (N=30)
- **Table 3:** Statistical significance tests (paired comparisons)

### Text Updates
- Abstract: Explicit N=7 → N=30 transition with timeline
- Results: Bootstrap CI for all metrics
- Discussion: External baseline positioning
- Methods: Full implementation details (already added)

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Test set expansion | N ≥ 30 achieved |
| External baselines | 3 methods compared |
| TPE ablation | Significant improvement at BSJ-flanking (p<0.05) |
| Closure validation | Broken data → higher closure error |
| Statistical rigor | All results with 95% CI |

---

## Appendix: Implementation Code Templates

### A. Expanded Test Set Generation
```bash
#!/bin/bash
# generate_expanded_testset.sh

# PDB circularization
python fetch_pdb_rna.py --output data/pdb_raw --min_len 20 --max_len 200
python pdb_rna_circularize.py --input data/pdb_raw --output data/pdb_circ --n_workers 8

# Synthetic benchmark
python generate_benchmark_structures.py --output data/synth_bench --n_samples 15 --motifs hairpin,internal_loop,pseudoknot

# Merge
python merge_test_set.py --pdb data/pdb_circ --synth data/synth_bench --output data/test_expanded
```

### B. External Baseline Runner
```python
# run_external_baselines.py
import subprocess
import json

def run_isrna(sequences, output_dir):
    """Run IsRNAcirc web server batch mode"""
    # Prepare input
    with open(f"{output_dir}/input.fasta", "w") as f:
        for seq in sequences:
            f.write(f">seq_{seq['id']}\n{seq['sequence']}\n")
    
    # Submit to web server (mock - actual implementation depends on API)
    # results = submit_isrna_batch(f"{output_dir}/input.fasta")
    return results

def run_farfar2(sequences, output_dir):
    """Run FARFAR2 locally"""
    cmd = f"rosetta_scripts.default.linuxgccrelease -parser:protocol farfar2.xml -in:file:fasta {output_dir}/input.fasta"
    subprocess.run(cmd, shell=True)
    return parse_farfar2_output(output_dir)

def run_af3(sequences, output_dir):
    """Run AlphaFold3 via ColabFold"""
    # Mock implementation - actual depends on AF3 API availability
    pass
```

### C. Statistical Analysis Script
```python
# statistical_analysis.py
import numpy as np
from scipy import stats

def full_statistical_report(results_tpe, results_stdpe, results_external):
    """Generate complete statistical report for manuscript"""
    
    # Bootstrap CI
    rmsd_ci = bootstrap_ci(results_tpe['rmsd'], n_bootstrap=10000)
    closure_ci = bootstrap_ci(results_tpe['closure'], n_bootstrap=10000)
    
    # Paired comparison
    t_stat, p_value = stats.ttest_rel(results_tpe['rmsd'], results_stdpe['rmsd'])
    
    # Effect size
    cohens_d = (np.mean(results_tpe['rmsd']) - np.mean(results_stdpe['rmsd'])) / np.std(results_tpe['rmsd'] + results_stdpe['rmsd'])
    
    # ANOVA for external baselines
    f_stat, p_anova = stats.f_oneway(results_tpe['rmsd'], results_isrna['rmsd'], results_af3['rmsd'])
    
    report = {
        'rmsd_mean': rmsd_ci[0],
        'rmsd_95_ci': (rmsd_ci[1], rmsd_ci[2]),
        'closure_mean': closure_ci[0],
        'closure_95_ci': (closure_ci[1], closure_ci[2]),
        'tpe_vs_stdpe': {'t': t_stat, 'p': p_value, 'd': cohens_d},
        'external_anova': {'F': f_stat, 'p': p_anova},
    }
    
    return report
```

---

**Document End**
*Contact: iGEM FBH Team 2026*
*Repository: github.com/RomanCohort/confluencia*