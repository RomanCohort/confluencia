# Confluencia Shared Library - Bio-Mimetic Module Appendix

## Overview

This document describes the brain-inspired AI modules added to Confluencia for drug discovery applications.

## Modules

### 1. BioGatedMOE (MoE with Membrane Potential)

Located in `confluencia_shared.moe`

Replaces standard MOE with bio-mimetic gating mechanisms:

```python
from confluencia_shared.moe import BioGatedMOERegressor

model = BioGatedMOERegressor(
    expert_names=['ridge', 'hgb', 'rf', 'mlp'],
    folds=4,
    membrane_decay=0.85,
    membrane_boost=0.35,
    refractory_duration=3,
)
```

Key features:
- **MembranePotential**: Tracks expert activation history with exponential decay
- **RefractoryPeriod**: Prevents over-use of certain experts
- **EmotionalState**: Modulates routing based on molecule properties (lipinski_score, novelty, admet_risk)

### 2. TopologyPharmacophoreNetwork

Located in `confluencia_shared.bio_mimetic`

Scale-free network representation of pharmacophores:

```python
from confluencia_shared.bio_mimetic import TopologyPharmacophoreNetwork

tpn = TopologyPharmacophoreNetwork(decay_alpha=1.5)
tpn.build_from_features(features, feature_names)
topo_features = tpn.get_topology_features()
```

### 3. TissueSpecificAttention

Located in `confluencia_shared.bio_mimetic`

Patient physiology-based ADMET modulation:

```python
from confluencia_shared.bio_mimetic import TissueSpecificAttention, PhysiologicalState, TissueType

# For liver metabolism
liver_attention = TissueSpecificAttention(n_features=64, tissue_type=TissueType.LIVER)

# Patient physiology
phys_state = PhysiologicalState(
    liver_function=0.8,
    inflammation=0.2,
    enzyme_activity=0.5,
)

# Modulate ADMET prediction
modulated_clearance = liver_attention.predict_admet_modulation(base_clearance, phys_state)
```

### 4. NeuroplasticClosedLoop

Located in `confluencia_shared.bio_mimetic`

Clinical feedback-driven model adaptation:

```python
from confluencia_shared.bio_mimetic import NeuroplasticClosedLoop, ClinicalFeedback

neuro = NeuroplasticClosedLoop(adaptation_rate=0.1, plasticity_threshold=0.2)

feedback = ClinicalFeedback(
    patient_id='P001',
    predicted_outcome=0.7,
    actual_outcome=0.85,
)
adjustment = neuro.incorporate_feedback(feedback)
summary = neuro.get_adaptation_summary()
```

### 5. AdversarialPruningOptimizer

Located in `confluencia_shared.bio_mimetic`

Multi-objective molecular optimization with synaptic pruning:

```python
from confluencia_shared.bio_mimetic import AdversarialPruningOptimizer

optimizer = AdversarialPruningOptimizer(
    n_generations=100,
    population_size=50,
    pruning_ratio=0.2,
)
result = optimizer.optimize(objectives)
```

## CLI Commands

Test all bio-mimetic modules:

```bash
confluencia bio test --verbose
confluencia bio membrane
confluencia bio tissue
confluencia bio feedback
```

## Integration Examples

See `confluencia_shared.bio_integration_examples.py` for full pipeline integration.

```python
from confluencia_shared.bio_integration_examples import run_full_bio_pipeline

predictions = run_full_bio_pipeline(
    df=df,
    mol_smiles=smiles_list,
    patient_physiology=patient_phys,
    enable_clinical_feedback=True,
    clinical_results=clinical_df,
)
```

## Dependencies

- numpy
- pandas
- scikit-learn
- lightgbm
- rdkit (for molecular features)

## Version

Added: 2026-05