# circRNA 3D Data Generation Pipeline

High-quality circRNA 3D structure generation for TorusFold training.

## Architecture

```
Input (FASTA) → Stage 1 (ViennaRNA) → Stage 2 (RoseTTAFold2NA) → Stage 3 (OpenMM) → Stage 4 (MD) → Stage 5 (Quality) → Output
```

## Quick Start

```bash
# Single sequence
python pipeline.py --sequence "ACGUACGU..." --bsj-start 0 --bsj-end 100

# Batch from FASTA
python parallel_worker.py --fasta input.fasta --num-workers 8 --export-torusfold
```

## Stages

| Stage | Tool | Purpose | Output |
|-------|------|---------|--------|
| 1 | ViennaRNA | Secondary structure + BSJ constraint | dot-bracket, bp_probs |
| 2 | RoseTTAFold2NA | 3D prediction (linear) | PDB files |
| 3 | OpenMM | BSJ cyclization | Cyclized PDB |
| 4 | OpenMM | MD relaxation | Snapshots + trajectories |
| 5 | Quality filter | Confidence scoring | Filtered dataset |

## Dependencies

```bash
# Core
conda install -c bioconda viennarna
pip install openmm ray pyyaml numpy

# RoseTTAFold2NA (separate install)
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
```

## Configuration

Edit `config.yaml` to adjust:
- BSJ restraint strength
- MD duration
- Quality thresholds
- Confidence weights

## Output Format

### Dataset JSON
```json
{
  "pdb_path": "path/to.pdb",
  "confidence": 0.85,
  "energy_kjmol": -850.2,
  "bsj_distance_angstrom": 3.8,
  "sequence": "ACGU...",
  "bsj_start": 0,
  "bsj_end": 100
}
```

### TorusFold Format
```
torusfold_format/
├── coords.npy          # (N, L, 3) coordinates
├── confidences.npy     # (N,) confidence scores
└── metadata.json       # Dataset info
```

## Confidence Score

```
confidence = 0.3 × energy_score
           + 0.3 × rmsd_plateau_score
           + 0.2 × bsj_score
           + 0.2 × ss_preservation_score
```

| Confidence | TorusFold Weight |
|------------|------------------|
| ≥ 0.8 | 2.0 (high quality) |
| 0.5-0.8 | 1.0 (medium) |
| < 0.5 | 0.1 (low) |

## DGX Spark Deployment

```bash
# On DGX Spark with 8 GPUs
chmod +x run_dgx.sh
./run_dgx.sh input.fasta
```

Expected throughput:
- ~5 min per sequence (10ns MD)
- 8 parallel workers
- ~100 sequences/hour
- 50,000+ structures from 10,000 sequences
