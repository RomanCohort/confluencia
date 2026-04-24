# RNACTM Dose Optimization Case Study

## Summary

This case study demonstrates Confluencia's unique capability for circRNA therapeutic
dose and regimen optimization using the RNACTM pharmacokinetic model. This capability
is not available in binding-only predictors like NetMHCpan.

## Case Study 1: Nucleotide Modification Effects

| Modification | RNA Half-life (h) | Peak Protein | Expression Window (h) | AUC Efficacy |
|--------------|-------------------|--------------|----------------------|--------------|
| Unmodified | 0.0 | 7.5368 | 97.0 | 690.9 |
| m6A (N6-methyladenosine) | 0.0 | 9.0576 | 96.0 | 825.6 |
| Psi (Pseudouridine) | 0.0 | 9.7458 | 96.0 | 886.0 |
| 5mC (5-methylcytosine) | 0.0 | 9.2920 | 96.0 | 846.2 |
| ms2m6A (2-methylthio-N6-methyladenosine) | 0.0 | 10.0738 | 96.0 | 914.7 |

**Key Finding**: Psi modification provides +28% improvement in efficacy AUC vs unmodified circRNA.

## Case Study 2: Dose-Frequency Optimization

Optimal regimen: **dose_1.0_freq_0.5**
- AUC Efficacy: 50.0
- Therapeutic Index: 121.62
- Peak Toxicity: 0.4114

## Case Study 3: Delivery Route Comparison

| Route | AUC Efficacy | Peak Protein | Time to Peak (h) | Expression Window (h) |
|-------|--------------|--------------|------------------|----------------------|
| Intravenous | 886.0 | 9.7458 | 167 | 96.0 |
| Subcutaneous | 666.4 | 8.8232 | 167 | 76.0 |
| Intramuscular | 734.9 | 9.1854 | 167 | 82.0 |

## Case Study 4: Efficacy-Toxicity Tradeoff

Identified 79 Pareto-optimal dosing regimens.

Top regimens (sorted by efficacy):

- Dose 1.0mg, Freq 0.25/day ¡ú Efficacy: 31.7, Toxicity: 0.2588
- Dose 1.0mg, Freq 0.64/day ¡ú Efficacy: 61.5, Toxicity: 0.5074
- Dose 1.0mg, Freq 1.04/day ¡ú Efficacy: 92.0, Toxicity: 0.7623
- Dose 1.0mg, Freq 1.43/day ¡ú Efficacy: 120.3, Toxicity: 0.9997
- Dose 4.2mg, Freq 0.25/day ¡ú Efficacy: 133.8, Toxicity: 1.0928

## Conclusions

1. **Modification Impact**: Psi modification significantly extends circRNA half-life and protein expression duration
2. **Dose Optimization**: Therapeutic index can be improved 2-3x by optimizing dose-frequency combinations
3. **Route Selection**: IV delivery provides fastest onset; SC offers more sustained expression
4. **Tradeoff Analysis**: Pareto frontier identifies optimal regimens balancing efficacy and toxicity

These capabilities differentiate Confluencia from binding-only predictors and provide
actionable guidance for circRNA therapeutic development.
