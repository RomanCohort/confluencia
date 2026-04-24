# Data Dictionary

## Overview

This directory contains example and benchmark datasets for Confluencia.
All data used in publications should be documented here with source, version,
and field definitions.

## Datasets

### example_drug.csv

Drug efficacy prediction dataset. Each row represents a drug candidate with
experimental conditions.

| Field | Type | Required | Description | Range |
|-------|------|----------|-------------|-------|
| smiles | string | Yes | Molecular SMILES representation | Valid SMILES |
| dose | float | Yes | Administration dose (mg/kg) | (0, +inf) |
| freq | float | Yes | Administration frequency (times/day) | (0, +inf) |
| treatment_time | float | Yes | Treatment duration (hours) | [0, +inf) |
| epitope_seq | string | No | Epitope amino acid sequence | Standard AA letters |
| group_id | string | No | Cohort/experiment group identifier | Any string |
| efficacy | float | No | Primary efficacy score (target) | [0, 1] typical |
| target_binding | float | No | Target binding affinity label | [0, 1] typical |
| immune_activation | float | No | Immune activation score label | [0, 1] typical |
| inflammation_risk | float | No | Inflammation risk score label | [0, 1] typical |
| toxicity_risk | float | No | Toxicity risk score label | [0, 1] typical |

### example_epitope.csv

Epitope immune efficacy prediction dataset. Each row represents a peptide
candidate with immunological context.

| Field | Type | Required | Description | Range |
|-------|------|----------|-------------|-------|
| epitope_seq | string | Yes | Amino acid sequence | Standard AA letters (ACDEFGHIKLMNPQRSTVWY) |
| dose | float | No | Dose (ug/mL) | (0, +inf) |
| freq | float | No | Administration frequency (times/day) | (0, +inf) |
| treatment_time | float | No | Treatment duration (hours) | [0, +inf) |
| circ_expr | float | No | circRNA expression level | [0, +inf) |
| ifn_score | float | No | Interferon response score | [0, +inf) |
| efficacy | float | No | Immune efficacy score (target) | [0, 1] typical |

### External Data Sources

| Source | URL | Usage | License |
|--------|-----|-------|---------|
| PubChem | https://pubchem.ncbi.nlm.nih.gov/ | Molecular structures & properties | CC-BY-NC-SA |
| ChEMBL | https://www.ebi.ac.uk/chembl/ | Bioactivity data | CC-BY-SA 3.0 |
| IEDB | https://www.iedb.org/ | Immune epitope data | CC-BY 2.5 |
| PDB | https://www.rcsb.org/ | Protein 3D structures | CC0 |
| UniProt | https://www.uniprot.org/ | Protein annotations | CC-BY 4.0 |

## Preprocessing Notes

1. All sequences are uppercased and whitespace-stripped before featurization
2. SMILES strings are validated (optional RDKit validation)
3. Numeric columns with >20% missing values are flagged
4. Duplicates are preserved (same sequence with different conditions = valid)

## Usage License

Example data in this repository is provided under CC-BY-4.0 unless otherwise noted.
External data is subject to their respective licenses (see above).
