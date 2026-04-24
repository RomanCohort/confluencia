# Breast Cancer Drug Dataset Metadata

## Data Sources
- **Compounds**: 28 FDA-approved/clinical breast cancer drugs
- **SMILES**: Canonical SMILES from PubChem/DrugBank
- **IC50**: Literature-reported values from ChEMBL
- **Dose/Freq**: FDA-approved dosing regimens
- **Epitopes**: 15 breast cancer-associated epitopes (HER2, MUC1, CEA, MAGE-A3, WT1)
- **Toxicity**: CTCAE grade ≥3 adverse event rates from clinical trials
- **Inflammation**: Inflammation-related AE rates from clinical trials

## Derived Labels
- `target_binding`: Sigmoid transform of IC50 (lower IC50 → higher binding)
- `efficacy`: Composite of binding, dose saturation, treatment duration, frequency
- `immune_activation`: Epitope immunogenicity × mechanism modifier
- `immune_cell_activation`: Derived from immune_activation + antigen type
- `toxicity_risk`: Weighted combination of grade≥3 AE rate + cardiotoxicity rate
- `inflammation_risk`: Inflammation-related AE rate

## Scaling
- All labels in [0, 1]
- Dose in mg, freq in times/day, treatment_time in hours
- Gaussian noise (σ=0.02–0.05) added for realistic variance

## Augmentation
- 3 samples per (drug, epitope) combination with dose jitter
- Additional 2 dose-variant samples (0.5× and 2× standard dose)

## Generated
- Date: 2026-04-13 14:51
- Samples: 2100
- Unique SMILES: 27
- Unique epitopes: 15