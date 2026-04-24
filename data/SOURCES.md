# Confluencia Data Sources

## External Data Retrieval Log

Record all external data downloads here for reproducibility.

| Source | Query/Filters | Download Date | Version | Records |
|--------|--------------|---------------|---------|---------|
| PubChem | bioactivity assay | 2026-01-15 | v1 | ~200 |
| ChEMBL | target: circRNA-related | 2026-02-01 | ChEMBL_33 | ~500 |
| IEDB | T-cell epitope, MHC-I | 2026-01-20 | 2026-01 | ~1000 |
| Example data | synthetic | 2026-01-10 | v1 | 40/160 |

## Notes

- PubChem and ChEMBL data were filtered for unique SMILES with valid bioactivity values.
- IEDB data was filtered for peptide length 8-15 amino acids with quantitative binding affinity.
- All external data has been deduplicated and standardised.
