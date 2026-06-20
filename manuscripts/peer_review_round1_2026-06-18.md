# Confluencia 3.0 — Three-Reviewer Report

**Journal:** Bioinformatics (Original Paper)
**Manuscript:** Confluencia 3.0: Integrated circRNA Vaccine Design with TNBC Subtype Simulation and Proposed TorusFold Structure Architecture

---

## Editorial Summary

| Reviewer | Expertise | Verdict |
|----------|-----------|---------|
| R1 | Computational biology methodology, ML, software engineering | **Major Revision** |
| R2 | circRNA biology, innate immune sensing, RNA therapeutics | **Major Revision** |
| R3 | Statistical methodology, benchmarking, reproducibility | **Major Revision** |

**Consensus: Major Revision required.** All three reviewers recognize the ambition and biological insight but identify critical gaps in validation, biological parameterization, and statistical claims.

---

## R1: Computational Biology Methodologist

### Major Concerns (7)

1. **Validation sample sizes critically insufficient** — N=4 PK, N=7 immunogenicity cannot support "validation" claims. Recast as "preliminary" or "proof-of-concept"; add power analysis.
2. **TNBC simulation is circular** — Results recapitulate parameterization, not novel predictions. Need external TCGA/METABRIC validation.
3. **TorusFold non-functional** — ~0% pair predictions. Move to Discussion/Future Directions or separate manuscript; do not claim as validated contribution.
4. **GC confound inadequately addressed** — r=0.85 GC-immunogenicity correlation; partial r=0.42 modest. Need head-to-head comparison: pathway scores vs. simple GC baseline (AIC/BIC).
5. **Wet-lab validation pending** — Problematic for computational methods paper. Include at least N=3-5 pilot data, or submit as Methods manuscript.
6. **Pathway weights uncalibrated** — 20% rank inversion under perturbation. Perform Bayesian optimization or CV against Chen 2019 data.
7. **TorusFold theoretical claims overstated** — Theorems are mathematically trivial (TPE periodicity guaranteed by construction; equivariance follows from invariant bias). Recast as design properties, not contributions.

### Minor Concerns (7)

1. Verify GitHub repository is public and functional
2. R package CRAN claim needs verification
3. hub.confluencia.org appears non-functional
4. Bio-mimetic architecture in Code Availability but not Methods
5. "Claude Code skill" inappropriate for academic manuscript
6. Bidirectional m6A model lacks validation
7. RL convergence needs quantitative support (learning curves, variance)

### Key Strengths

- circRNA-specific RIG-I modeling is biologically important
- Differential m6A suppression is insightful
- EventBus architecture well-designed for extensibility
- TorusFold mathematically sound (though biologically unvalidated)
- Exceptional limitations transparency
- Multi-interface accessibility addresses real need
- Hub federated sharing is thoughtful

---

## R2: RNA Biology Expert

### Major Concerns (4)

1. **RIG-I pathway model is mechanistically problematic** — RIG-I requires 5'-ppp blunt ends; circRNA lacks these. If circRNA activates via dsRNA structures, this is primarily **MDA5 signaling**, not RIG-I. Chen 2019 showed immunogenicity correlates with **intron identity and splicing pathway**, not simply dsRNA content. **Required: Rename "RIG-I pathway" → "MDA5/dsRNA pathway"** with appropriate downstream signaling (MAVS, IRF3/7).

2. **Differential m6A suppression lacks literature basis** — 90%/30%/20% values are model assumptions, not literature-derived. No published quantitative suppression percentages exist. Internal inconsistency: m6A "destabilizes dsRNA structures required for MDA5 co-sensing" contradicts earlier RIG-I framing. **Required: Cite sources or clearly label as hypothesized values.**

3. **PK validation statistically insufficient** — N=4 cannot distinguish six-compartment from simpler models. Need CI on 12% error, AIC/BIC model comparison, and discussion of inter-individual variability.

4. **circRNA production method not addressed** — Chen 2019 demonstrated immunogenicity depends on production method (intron retention, purification). Sequence-only model cannot capture this. **Required: Add production-method parameter or explicitly state model assumptions.**

### Minor Concerns (6)

1. GC confound analysis could use mediation analysis
2. TLR circularity correction factor (0.70) is heuristic, needs justification
3. IRES efficiency range (0.02-0.32/h) not sourced with specific examples
4. Shannon diversity assumes discrete subclones; reality is continuous
5. N=7 LOO range should report median, not just range
6. BSJ-proximal dsRNA scoring lacks biological justification

### Key Recommendation: Pathway Clarification Table

| Pathway | Sensor | circRNA Mechanism | Literature | Confidence |
|---------|--------|-------------------|------------|------------|
| MDA5/dsRNA | MDA5 | Long dsRNA structures (>16 bp) | Chen 2019, Peisley 2013 | Medium |
| TLR7 | TLR7 | GU-rich ssRNA motifs | Gilleron 2013 | Medium |
| TLR8 | TLR8 | AU-rich ssRNA motifs | Gilleron 2013 | Medium |
| PKR | PKR | dsRNA > 33 bp | Nallagatla 2007 | High |

---

## R3: Statistical/Benchmarking Expert

### Major Concerns (6)

1. **N=7 statistical fragility** — SE of r≈0.18; single observations swing correlation ±0.07. This is exploratory, not validation. **Remove r=0.91 from abstract as standalone claim.** Power analysis: need ~N=25 to distinguish r=0.91 from r=0.50 at α=0.05.

2. **Circular validation in TNBC simulation** — Tautology undermines module contribution. **Test: swap BLIS parameters into IM initialization.** If swapped-BLIS shows better response, confirms circular dependency.

3. **Correlation benchmarks insufficient for predictive claims** — Correlation ≠ prediction. Need MAE/RMSE in IFN-β units, not just Spearman r. For PK: AIC/BIC against simpler models. Add held-out sequence test.

4. **Code repository returns HTTP 404** — github.com/IGEM-FBH/confluencia-3.0 not accessible. **Provide working URL or Zenodo archive.**

5. **Sample-size-adaptive methodology not demonstrated** — Immunogenicity weights and RL reward coefficients are fixed, not adaptive. No shrinkage, Bayesian prior adjustment, or N-aware regularization. **Remove "sample-size-adaptive" claims or implement.**

6. **Hypothesis-generation vs. validation mismatch** — Limitations say "all claims are hypothesis-generation" but Discussion presents 11 validated contributions. **Distinguish: (A) implemented features requiring validation, (B) preliminary correlation support, (C) verified mathematical properties.**

### Minor Concerns (6)

1. HEK293 CI 0.26-0.88 (width 0.62) should be explicitly noted as insufficient
2. TorusFold pair head failure mode needs investigation (loss curves, alternative initialization)
3. GC partial correlation r=0.42 sample size unclear (N=7 or N=50?)
4. m6A suppression percentages uncited
5. RL reward coefficients (0.35/0.30/0.20/0.15) lack derivation
6. Abstract 223 words, too dense; restructure

### Key Recommendation: Power Analysis Table

| Module | Current N | Required N (α=0.05, power=0.80) | Current Power |
|--------|-----------|-------------------------------|---------------|
| Immunogenicity | 7 | ~25 | ~0.35 |
| PK | 4 | ~12 | ~0.20 |

---

## Cross-Reviewer Consensus: Top 5 Required Revisions

| Priority | Issue | Reviewers | Action |
|----------|-------|-----------|--------|
| **1** | RIG-I → MDA5 pathway correction | R2 (major), R1 (minor) | Rename pathway, add MAVS/IRF3/7 downstream, cite Peisley 2013 |
| **2** | Statistical claims downgrade | R1, R3 (both major) | Remove r=0.91 from abstract, add power analysis, recast as "preliminary" |
| **3** | TorusFold reframing | R1, R3 (both major) | Move from Results to Discussion/Future Directions; recast as design properties not contributions |
| **4** | m6A suppression values | R2 (major), R3 (minor) | Label as estimated parameters with uncertainty ranges; add ±50% sensitivity analysis |
| **5** | Code repository accessibility | R1 (minor), R3 (major) | Make public or provide Zenodo DOI; verify Hub/CRAN claims |

---

## Cross-Reviewer Consensus: Top 5 Recommended Additions

| Priority | Addition | Proposed By | Value |
|----------|----------|-------------|-------|
| **1** | Pathway clarification table | R2 | Resolves RIG-I/MDA5 confusion |
| **2** | Power analysis table | R3 | Quantifies validation insufficiency |
| **3** | GC baseline comparison (AIC/BIC) | R1, R3 | Proves pathway decomposition value over simple GC |
| **4** | TNBC parameter-swap experiment | R3 | Tests circular validation |
| **5** | circRNA production method parameter | R2 | Addresses missing biological variable |

---

## Questions Requiring Author Response

| # | Question | From |
|---|----------|------|
| 1 | RIG-I vs MDA5: Have you verified RIG-I involvement, or is this a placeholder label? | R2 |
| 2 | m6A suppression sources: What literature supports 90%/30%/20%? | R2, R3 |
| 3 | GC partial correlation sample: Was r=0.42 computed on N=7 or N=50? | R3 |
| 4 | Wet-lab timeline: Will results be available during revision? | R1 |
| 5 | TorusFold loss trajectory: Does loss decrease during 1 epoch or plateau? | R3 |
| 6 | Code repository: Can you provide reviewer access? | R1, R3 |
| 7 | RL reward derivation: Why 0.35/0.30/0.20/0.15? Sensitivity analysis? | R3 |
| 8 | Production method: Does model account for IVT vs ribozymatic vs spliceosome? | R2 |
| 9 | Clinical workflow: Lead optimization, de novo design, or both? | R1 |
| 10 | BSJ-proximal dsRNA: Biological evidence for higher immunogenicity? | R2 |

---

## Suggested Revision Strategy

### Phase 1: Critical Fixes (required for resubmission)

1. **Rename RIG-I → MDA5/dsRNA pathway** throughout manuscript
2. **Rewrite abstract** — remove standalone r=0.91, add "preliminary," restructure
3. **Move TorusFold** from Results to Discussion as "Proposed Future Direction"
4. **Label m6A values** as estimated parameters with ±50% sensitivity analysis
5. **Make code repository accessible** (public GitHub or Zenodo archive)
6. **Add power analysis table** (immunogenicity N=7→25 needed; PK N=4→12 needed)
7. **Add GC baseline comparison** (pathway scores vs. simple GC; AIC/BIC)
8. **Add production method caveat** to immunogenicity model description
9. **Harmonize language** — "hypothesis-generation" in Limitations must match Discussion tone

### Phase 2: Strengthening (recommended for competitive revision)

1. Add pathway clarification table (R2)
2. Add TNBC parameter-swap experiment (R3)
3. Add CI to PK 12% error claim; compare vs simpler models (AIC/BIC)
4. Add m6A ±50% sensitivity analysis
5. Verify Hub URL and CRAN availability; remove non-functional claims
6. Move bio-mimetic architecture from Code Availability to Methods
7. Report RL learning curves with confidence bands across 5+ seeds
8. Add MAE/RMSE for immunogenicity, not just Spearman r
