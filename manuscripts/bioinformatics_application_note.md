# Confluencia circRNA: A Platform for Circular RNA Immunogenicity Prediction and Sequence Design

**Running Title:** Confluencia circRNA Platform

**Keywords:** circRNA, immunogenicity, RIG-I, vaccine design, RNA secondary structure

---

## Abstract (100 words)

Confluencia circRNA is an open-source platform for circular RNA (circRNA) immunogenicity prediction and sequence optimization. It implements literature-backed scoring for innate immune sensors RIG-I, TLR7/8, and PKR, integrates ViennaRNA-based secondary structure analysis, and predicts post-transcriptional modifications including m6A, IRES, and miRNA binding sites. An evolutionary optimization module enables multi-objective sequence design for vaccine or therapeutic cargo applications. The platform is available as a Python package (pip install confluencia-circrna) with a Streamlit web interface and R package bridge, requiring only a CPU and standard bioinformatics dependencies.

---

## Introduction (150 words)

Circular RNAs (circRNAs) have emerged as promising vaccine platforms due to their inherent stability and protein translation capacity (Wesselhoeft et al., 2018). However, predicting circRNA immunogenicity remains challenging: the closed-loop structure eliminates 5'/3' termini recognition by RIG-I, shifting innate sensing to dsRNA backbone structures formed by inverted repeats. Existing tools address linear RNA structure prediction (ViennaRNA; Lorenz et al., 2011) or mRNA design, but no platform integrates circRNA-specific immunogenicity prediction with sequence optimization.

We present Confluencia circRNA, a comprehensive platform addressing this gap. Its key contributions are: (1) literature-weighted scoring for RIG-I, TLR7/8, and PKR pathways calibrated for circRNA's unique topology; (2) integrated structure, kinetics, and modification prediction; (3) Pareto-based evolutionary sequence optimization for multi-objective design; and (4) a modular API accessible via Python, R, CLI, and web interfaces.

---

## Methods (300 words)

**Immunogenicity scoring.** Four innate immune pathways are scored using literature-derived formulas. RIG-I activation (weight 0.35) is predicted from ViennaRNA-derived dsRNA content and MFE, reflecting that circRNA's closed loop precludes blunt-end recognition—only dsRNA backbone structures activate RIG-I (Schlee et al., 2009). TLR7 (0.20) scores GU-rich ssRNA motifs; TLR8 (0.15) scores AU-rich motifs (Forsbach et al., 2008). PKR activation (0.30) uses the >33 bp dsRNA threshold (Nallagatla et al., 2007). The overall score is a weighted sum normalized to [0,1] with pre-defined decision thresholds.

**Structure and modification prediction.** Secondary structure is computed via ViennaRNA RNAfold with a GC-content fallback for environments without ViennaRNA. m6A sites are identified via DRACH motif scanning with GC-context probability estimation. IRES activity is assessed from polypyrimidine tract content; miRNA binding sites are scanned against 15 oncogenic/regulatory miRNA seeds.

**Evolutionary optimization.** A REINFORCE-based policy optimizes sequences over 5-6 rounds using four operators: backbone mutation (preserving the backsplice junction), IRES motif insertion, flanking region shuffling, and modification selection (m6A, Psi, 5mC, etc.). Four objectives—stability, translation potential, immune evasion, and delivery compatibility—are balanced via Pareto front selection. Default runtime is under 30 seconds on CPU.

---

## Results (200 words)

The platform provides ten core analysis modules accessible through a unified Python API. Key features demonstrated on circBase-derived sequences include:

**Multi-pathway immunogenicity scoring.** The pipeline outputs overall and per-pathway scores. On a 300 nt high-GC sequence (GC=93%), the overall score was 0.35 (RIG-I=0.45, PKR=0.56), suitable for vaccine development; a 150 nt low-GC sequence (GC=30%) scored 0.19, suitable as a therapeutic delivery vehicle. The GC-immunogenicity correlation (r=0.85, p<0.001, N=50 circBase sequences) is consistent with PKR activation by GC-rich dsRNA structures.

**Literature consistency.** On 7 published IFN-β measurements, the predicted scores correlated with observed immunogenicity at r=0.91 (95% CI: 0.62-0.99, p=0.004). PK half-life predictions for four modification types showed r=0.94 (95% CI: 0.54-0.99, p=0.06).

**Sequence evolution.** Starting from a 200 nt seed, Pareto optimization selects candidates spanning the stability-immune evasion trade-off in under 30 seconds. Multi-round REINFORCE learning shifts operator selection toward IRES optimization and modification selection.

Performance: immunogenicity scoring <100 ms/seq; full pipeline <3 s/seq on CPU.

---

## Comparison with Existing Tools

| Feature | Confluencia | ViennaRNA | General RNA tools |
|---------|-------------|-----------|-------------------|
| Immunogenicity scoring | ✓ (4 pathways) | ✗ | ✗ |
| circRNA-specific design | ✓ | ✗ | ✗ |
| Structure prediction | ✓ (ViennaRNA) | ✓ | ✓ |
| Modification prediction | ✓ (m6A/IRES/miRNA) | ✗ | Partial |
| Sequence optimization | ✓ (Pareto RL) | ✗ | Codon optimization |
| Clinical translation | Survival/biomarker | ✗ | ✗ |

---

## Availability and Implementation

**Name:** Confluencia circRNA
**Version:** 2.6.0
**License:** MIT
**URL:** https://github.com/IGEM-FBH/confluencia
**Programming language:** Python 3.8+
**Dependencies:** numpy, pandas, plotly, streamlit; Optional: ViennaRNA
**Operating systems:** Linux, macOS, Windows
**Interfaces:** Python API, Streamlit web, R package (27 functions), VS Code extension, CLI

---

## Limitations

Immunogenicity weights are literature-derived heuristics, not empirically calibrated. CircRNA validation is limited to literature data (N=7 IFN-β, N=50 circBase pseudo-labels). MHC-II binding prediction is experimental. Clinical outcome prediction uses Cox approximations with unvalidated C-index.

---

## Acknowledgements

We thank the ViennaRNA team for the structure prediction library.

## References

1. Lorenz R, et al. *Algorithms Mol Biol*. 2011;6:26.
2. Schlee M, et al. *Nature*. 2009;458:514-518.
3. Nallagatla SR, et al. *RNA*. 2007;13:1234-1247.
4. Forsbach A, et al. *J Immunol*. 2008;181:6852-6863.
5. Wesselhoeft RA, et al. *Nat Commun*. 2018;9:2629.
6. Chen YG, et al. *Mol Cell*. 2019;74:598-609.
7. Liu CX, et al. *Nature*. 2022;600:658-666.
