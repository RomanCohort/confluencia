# circRNA 3D Structure Data Generation Pipeline

## Overview

This pipeline generates high-quality circRNA 3D structures using RoseTTAFold2NA for initial prediction and OpenMM for BSJ cyclization and MD relaxation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        circRNA 3D Data Generation Pipeline                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │  Input   │───▶│  Stage 1     │───▶│   Stage 2     │───▶│   Stage 3    │ │
│  │  FASTA   │    │  2D Struct   │    │   3D Pred     │    │   Cyclize    │ │
│  └──────────┘    │  ViennaRNA   │    │ RoseTTAFold2NA│    │   OpenMM     │ │
│                  └──────────────┘    └───────────────┘    └──────────────┘ │
│                         │                    │                   │         │
│                         ▼                    ▼                   ▼         │
│                  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐ │
│                  │  dot-bracket │    │   PDB coords  │    │  Cyclized    │ │
│                  │   + bp prob  │    │   (linear)    │    │   PDB coords │ │
│                  └──────────────┘    └───────────────┘    └──────────────┘ │
│                                              │                   │         │
│                                              ▼                   ▼         │
│                                        ┌───────────────┐    ┌──────────────┐│
│                                        │   Stage 4     │    │   Stage 5    ││
│                                        │   MD Relax    │───▶│   Quality    ││
│                                        │   OpenMM      │    │   Filter     ││
│                                        └───────────────┘    └──────────────┘│
│                                              │                   │        │
│                                              ▼                   ▼        │
│                                        ┌───────────────┐    ┌──────────────┐│
│                                        │  Relaxed PDB  │    │  Confidence  ││
│                                        │   + energies  │    │   Score      ││
│                                        └───────────────┘    └──────────────┘│
│                                                                   │        │
│                                                                   ▼        │
│                                                            ┌──────────────┐│
│                                                            │   Output     ││
│                                                            │   Dataset    ││
│                                                            └──────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Secondary Structure Prediction (ViennaRNA)

### Purpose
Predict circRNA secondary structure to guide 3D folding.

### Input
- FASTA file with circRNA sequence
- BSJ position (start, end indices)

### Process
```python
import RNA

def predict_secondary_structure(sequence, bsj_start, bsj_end):
    """
    Predict secondary structure with BSJ constraint.

    The BSJ connects the end of the sequence back to the start,
    forming a circular topology. We model this as a hard constraint.
    """
    # Set folding parameters
    md = RNA.md()
    md.max_bp_span = len(sequence)  # Allow full-span base pairs
    md.window_size = len(sequence)

    # Create fold compound
    fc = RNA.fold_compound(sequence, md)

    # Add BSJ constraint: last base pairs with first base
    # This enforces circular connectivity in 2D structure
    fc.hc_add_bp(bsj_end - 1, bsj_start, RNA.CONSTRAINT_CONTEXT_BP_ENFORCED)

    # Predict MFE structure
    ss, mfe = fc.mfe()

    # Get base pair probabilities
    probs = fc.bpp()

    return {
        'sequence': sequence,
        'dot_bracket': ss,
        'mfe': mfe,
        'bp_probs': probs,
        'bsj_start': bsj_start,
        'bsj_end': bsj_end
    }
```

### Output
- `dot_bracket`: Secondary structure in dot-bracket notation
- `bp_probs`: Base pair probability matrix (N x N)
- `mfe`: Minimum free energy
- `bsj_constraint`: BSJ position annotation

### Key Considerations
- BSJ constraint enforces circular topology at secondary structure level
- ViennaRNA's probabilistic partition function gives bp_probs for uncertainty quantification
- Low MFE structures are more stable → higher confidence

---

## Stage 2: 3D Structure Prediction (RoseTTAFold2NA)

### Purpose
Predict initial 3D coordinates for the linear RNA sequence.

### Input
- Sequence from Stage 1
- Secondary structure (dot_bracket) from Stage 1
- Base pair probabilities from Stage 1

### Process

RoseTTAFold2NA is a RoseTTAFold2 variant specialized for nucleic acids.
It takes sequence + secondary structure and outputs 3D coordinates.

```bash
# RoseTTAFold2NA command-line invocation
run_rosettafold2na.py \
    --sequence "${sequence}" \
    --ss "${dot_bracket}" \
    --bp_probs "${bp_probs_npy}" \
    --output "${output_dir}" \
    --num_samples 5  # Generate multiple conformations
```

### Output
- `pdb_linear_*.pdb`: 5 sampled 3D conformations (linear RNA)
- `confidence_score.npy`: Per-residue confidence (pLDDT-like metric)
- `distance_matrix.npy`: Predicted inter-residue distances

### Key Considerations
- RoseTTAFold2NA predicts *linear* RNA, not circRNA
- The BSJ ends are spatially separated in the linear prediction
- We need Stage 3 to cyclize the structure
- Multiple samples capture conformational diversity

---

## Stage 3: BSJ Cyclization (OpenMM)

### Purpose
Connect BSJ ends to form circular topology and resolve steric clashes.

### Input
- Linear PDB from Stage 2
- BSJ position (start, end indices)
- Secondary structure constraints

### Process

```python
import openmm as mm
import openmm.app as app
from openmm import unit

def cyclize_bsj(pdb_file, bsj_start, bsj_end, ss_constraint):
    """
    Connect BSJ ends and resolve clashes using OpenMM constraints.

    Strategy:
    1. Load linear structure
    2. Add distance restraint between BSJ ends (target: 3.5Å for phosphodiester bond)
    3. Add base pair restraints from secondary structure
    4. Minimize energy
    5. Short MD to relax
    """
    # Load PDB
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Create system
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,  # Full electrostatics for small system
        constraints=app.HBonds
    )

    # Add BSJ distance restraint
    bsj_force = mm.CustomBondForce('k*(r - r0)^2')
    bsj_force.addPerBondParameter('k')
    bsj_force.addPerBondParameter('r0')

    # BSJ atoms: last nucleotide O3' to first nucleotide P
    # Typical phosphodiester bond length: 3.5Å
    bsj_force.addBond(
        bsj_end_atom, bsj_start_atom,
        [1000.0, 3.5]  # Strong restraint: k=1000 kJ/mol/nm², r0=3.5Å
    )
    system.addForce(bsj_force)

    # Add secondary structure restraints (base pairs)
    ss_force = mm.CustomBondForce('k*(r - r0)^2')
    for bp in get_base_pairs(ss_constraint):
        ss_force.addBond(bp.atom1, bp.atom2, [50.0, 3.0])  # Moderate restraint
    system.addForce(ss_force)

    # Energy minimization
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=500)

    # Short MD relaxation (1 ns)
    simulation.step(500000)  # 2fs steps × 500k = 1ns

    # Save cyclized structure
    state = simulation.context.getState(getPositions=True)
    app.PDBFile.writeFile(pdb.topology, state.getPositions(), open('cyclized.pdb', 'w'))

    return 'cyclized.pdb'
```

### Output
- `cyclized.pdb`: Cyclized 3D structure
- `cyclization_energy.json`: Energy during cyclization process

### Key Considerations
- Strong BSJ restraint (k=1000) ensures correct cyclization
- Secondary structure restraints preserve folding pattern
- Minimization + short MD resolves steric clashes

---

## Stage 4: MD Relaxation (OpenMM)

### Purpose
Extended MD simulation to achieve thermodynamically stable circRNA structure.

### Input
- Cyclized PDB from Stage 3
- Simulation parameters (temperature, duration)

### Process

```python
def md_relaxation(pdb_file, duration_ns=10, temperature=300):
    """
    Extended MD simulation for circRNA relaxation.

    Protocol:
    1. Equilibration: 1ns at target temperature with restraints on backbone
    2. Production: 9ns unrestrained MD
    3. Extract snapshots every 100ps
    """
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Solvate in explicit water box
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*unit.nanometer)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,  # Particle Mesh Ewald for electrostatics
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.HBonds
    )

    # Add BSJ restraint during equilibration (prevent decyclization)
    bsj_restraint = create_bsj_restraint(bsj_start, bsj_end, k=500.0)
    system.addForce(bsj_restraint)

    integrator = mm.LangevinMiddleIntegrator(
        temperature*unit.kelvin,
        1/unit.picosecond,
        2*unit.femtosecond
    )

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Equilibration phase (1ns with restraints)
    simulation.step(500000)

    # Remove BSJ restraint for production
    system.removeForce(bsj_restraint)
    simulation.context.reinitialize()

    # Production phase (9ns)
    snapshots = []
    for i in range(90):
        simulation.step(100000)  # 100ps
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        snapshots.append({
            'positions': state.getPositions(),
            'energy': state.getPotentialEnergy(),
            'time': i * 100 * unit.picosecond
        })

    return snapshots
```

### Output
- `snapshots/*.pdb`: 90 snapshots from production MD
- `energy_trajectory.csv`: Energy over time
- `rmsd_trajectory.csv`: RMSD to initial structure

### Key Considerations
- Explicit solvent (TIP3P water) for realistic dynamics
- PME for long-range electrostatics
- Equilibration with BSJ restraint prevents decyclization
- Production without restraint tests cyclization stability

---

## Stage 5: Quality Filtering & Confidence Scoring

### Purpose
Select high-quality structures and compute confidence metrics.

### Input
- MD snapshots from Stage 4
- Energy trajectory
- RMSD trajectory

### Process

```python
def quality_filter_and_score(snapshots, energy_traj, rmsd_traj, ss_reference):
    """
    Filter low-quality structures and compute confidence score.

    Quality criteria:
    1. Energy < threshold (thermodynamically stable)
    2. RMSD plateau (converged structure)
    3. BSJ intact (distance < 5Å)
    4. Secondary structure preserved (base pair RMSD < 2Å)
    """
    quality_snapshots = []

    for snap in snapshots:
        # Check energy
        if snap['energy'] > ENERGY_THRESHOLD:
            continue

        # Check BSJ intact
        bsj_dist = compute_bsj_distance(snap['positions'], bsj_start, bsj_end)
        if bsj_dist > 5.0:  # Å
            continue

        # Check secondary structure preservation
        bp_rmsd = compute_base_pair_rmsd(snap['positions'], ss_reference)
        if bp_rmsd > 2.0:  # Å
            continue

        # Compute confidence score
        confidence = compute_confidence(snap, energy_traj, rmsd_traj)

        quality_snapshots.append({
            'pdb': snap['positions'],
            'energy': snap['energy'],
            'bsj_distance': bsj_dist,
            'bp_rmsd': bp_rmsd,
            'confidence': confidence
        })

    return quality_snapshots

def compute_confidence(snapshot, energy_traj, rmsd_traj):
    """
    Confidence score = weighted combination of:
    - Energy score (normalized to [0,1])
    - RMSD plateau score (how stable is the structure)
    - BSJ score (cyclization quality)
    - SS preservation score

    Final score ∈ [0, 1], higher = more confidence
    """
    # Energy score: lower energy = higher confidence
    energy_norm = normalize_energy(snapshot['energy'], energy_traj)
    energy_score = 1 - energy_norm

    # RMSD plateau score: stable RMSD = higher confidence
    rmsd_score = compute_rmsd_plateau_score(rmsd_traj)

    # BSJ score: shorter distance = higher confidence
    bsj_dist = snapshot['bsj_distance']
    bsj_score = max(0, 1 - (bsj_dist - 3.5) / 1.5)  # Ideal: 3.5Å

    # SS preservation score
    ss_score = max(0, 1 - snapshot['bp_rmsd'] / 2.0)

    # Weighted combination
    confidence = (
        0.3 * energy_score +
        0.3 * rmsd_score +
        0.2 * bsj_score +
        0.2 * ss_score
    )

    return confidence
```

### Output
- `filtered_dataset.json`: High-quality structures with confidence scores
- `confidence_scores.npy`: Confidence ∈ [0, 1]
- `quality_report.txt`: Statistics on filtering

### Quality Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Energy | < 1000 kJ/mol | Thermodynamically stable |
| BSJ distance | < 5 Å | Cyclization intact |
| Base pair RMSD | < 2 Å | Secondary structure preserved |
| RMSD plateau | < 1 Å variance | Structure converged |

### Confidence Score Formula

```
confidence = 0.3 × energy_score
           + 0.3 × rmsd_plateau_score
           + 0.2 × bsj_score
           + 0.2 × ss_preservation_score

where:
  energy_score = 1 - normalized_energy
  rmsd_plateau_score = variance_threshold / actual_variance
  bsj_score = max(0, 1 - (bsj_dist - 3.5) / 1.5)
  ss_score = max(0, 1 - bp_rmsd / 2.0)
```

---

## Full Pipeline Integration

```python
class CircRNA3DPipeline:
    """
    Full pipeline for circRNA 3D structure generation.
    """

    def __init__(self, config):
        self.config = config
        self.vienna = ViennaRNAModule(config['vienna'])
        self.rosetta = RoseTTAFold2NAModule(config['rosetta'])
        self.openmm_cyclize = CyclizationModule(config['cyclize'])
        self.openmm_md = MDModule(config['md'])
        self.quality = QualityModule(config['quality'])

    def run(self, sequence, bsj_start, bsj_end):
        """
        Run full pipeline for one circRNA.
        """
        # Stage 1: Secondary structure
        ss_result = self.vienna.predict(sequence, bsj_start, bsj_end)

        # Stage 2: 3D prediction (linear)
        linear_pdbs = self.rosetta.predict(
            sequence,
            ss_result['dot_bracket'],
            ss_result['bp_probs']
        )

        # Stage 3-5 for each linear conformation
        final_structures = []
        for linear_pdb in linear_pdbs:
            # Stage 3: Cyclization
            cyclized_pdb = self.openmm_cyclize.run(
                linear_pdb,
                bsj_start,
                bsj_end,
                ss_result['dot_bracket']
            )

            # Stage 4: MD relaxation
            snapshots = self.openmm_md.run(cyclized_pdb)

            # Stage 5: Quality filter
            quality_structures = self.quality.filter(snapshots)

            final_structures.extend(quality_structures)

        # Return best structure or ensemble
        return self.select_best(final_structures)
```

---

## Parallelization Strategy

### DGX Spark Deployment

DGX Spark has 8× A100/H100 GPUs. Parallelization strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DGX Spark Parallel Execution                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Queue: 10,000 circRNA sequences                          │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ... (8 workers)│
│  │  Worker 1  │  │  Worker 2  │  │  Worker 3  │                 │
│  │  GPU 0     │  │  GPU 1     │  │  GPU 2     │                 │
│  └────────────┘  └────────────┘  └────────────┘                 │
│       │              │              │                            │
│       ▼              ▼              ▼                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ ViennaRNA  │  │ ViennaRNA  │  │ ViennaRNA  │  (CPU parallel) │
│  │ RoseTTAFold│  │ RoseTTAFold│  │ RoseTTAFold│  (GPU parallel) │
│  │ OpenMM     │  │ OpenMM     │  │ OpenMM     │  (CPU parallel) │
│  └────────────┘  └────────────┘  └────────────┘                 │
│       │              │              │                            │
│       ┼──────────────┼──────────────┼──────────────▶ Output Queue│
│                                                                  │
│  Output: 50,000+ high-quality circRNA 3D structures              │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import ray

@ray.remote(num_gpus=1)
class PipelineWorker:
    def __init__(self, gpu_id):
        self.pipeline = CircRNA3DPipeline(config)
        self.gpu_id = gpu_id

    def process(self, sequence, bsj_start, bsj_end):
        return self.pipeline.run(sequence, bsj_start, bsj_end)

def parallel_pipeline(sequences, num_workers=8):
    """
    Run pipeline on DGX Spark with 8 GPU workers.
    """
    workers = [PipelineWorker.remote(i) for i in range(num_workers)]

    # Distribute sequences across workers
    futures = []
    for i, (seq, bsj_start, bsj_end) in enumerate(sequences):
        worker_id = i % num_workers
        future = workers[worker_id].process.remote(seq, bsj_start, bsj_end)
        futures.append(future)

    # Collect results
    results = ray.get(futures)
    return results
```

---

## Expected Output

### Dataset Statistics

| Metric | Target |
|--------|--------|
| Input sequences | 10,000 |
| Structures per sequence | 5 (ensemble) |
| High-quality structures | 30,000+ (60% pass rate) |
| Confidence score | 0.7-1.0 (high quality) |
| Processing time | ~5 min/sequence (10ns MD) |

### Data Format

```json
{
  "sequence_id": "circRNA_001",
  "sequence": "ACGUACGU...",
  "bsj_position": [0, 100],
  "structures": [
    {
      "pdb_path": "circRNA_001_struct_1.pdb",
      "confidence": 0.85,
      "energy": -850.2,
      "bsj_distance": 3.8,
      "bp_rmsd": 1.2
    },
    ...
  ]
}
```

---

## Integration with TorusFold

### Training Data Format

Convert pipeline output to TorusFold format:

```python
def convert_to_torusfold_format(pipeline_output):
    """
    Convert pipeline JSON to TorusFold training format.
    """
    data = {
        'sequence': pipeline_output['sequence'],
        'coords': load_pdb_coords(pipeline_output['pdb_path']),
        'confidence': pipeline_output['confidence'],
        'ss': predict_ss_from_coords(pipeline_output['pdb_path']),
        'bsj_mask': create_bsj_mask(pipeline_output['bsj_position'])
    }
    return data
```

### Confidence Usage

Pipeline-generated confidence scores directly feed into TorusFold's tiered confidence weighting:

| Confidence | TorusFold Weight | Pipeline Source |
|------------|------------------|-----------------|
| ≥ 0.8 | 2.0 | Stable MD, low energy, intact BSJ |
| 0.5-0.8 | 1.0 | Moderate quality |
| < 0.5 | 0.1 | Filtered out by quality module |

---

## Implementation Files

```
circrna_3d_pipeline/
├── config.yaml              # Pipeline configuration
├── stage1_vienna.py         # Secondary structure prediction
├── stage2_rosetta.py        # RoseTTAFold2NA integration
├── stage3_cyclize.py        # OpenMM cyclization
├── stage4_md.py             # OpenMM MD relaxation
├── stage5_quality.py        # Quality filtering & scoring
├── pipeline.py              # Full pipeline orchestration
├── parallel_worker.py       # Ray parallelization
├── torusfold_converter.py   # Convert to TorusFold format
└── run_dgx.sh               # DGX Spark deployment script
```

---

## Dependencies

- ViennaRNA 2.6+
- RoseTTAFold2NA (from Baker Lab)
- OpenMM 8.0+
- Ray 2.0+ (for parallelization)
- Amber14 force field

---

## Next Steps

1. DGX Spark arrives → install dependencies
2. Install RoseTTAFold2NA from Baker Lab GitHub
3. Test pipeline on 10 circRNA sequences
4. Batch process 10,000 sequences
5. Convert to TorusFold format → train with confidence weighting