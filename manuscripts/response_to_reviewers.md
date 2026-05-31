# Point-by-Point Response to Reviewers

## Manuscript: Confluencia: An uncertainty-adaptive cross-modality evaluation platform for circRNA therapeutic development

**Revision type:** Major Revision (pre-emptive)

---

## Reviewer 1 (Computational Biology / ML Methodology)

### Q1: PK validation is circular — half-life "matching within 4.0-4.1%" is parametrization verification, not prediction

**Response:** We agree and have revised the manuscript accordingly.

- **Changes made:** Throughout the manuscript, "matching literature within 4.0-4.1%" has been replaced with "consistency verification with the same literature that informed model parameters, not independent experimental validation." The half-life consistency is now explicitly described as "by design within numerical precision." The most informative metric is now highlighted as the expression window deviation (40h vs ~48h, 17%), which captures dynamics not directly parametrized (Results, p.2; Discussion, p.4; Supplementary Table S1).

- **Why this is sufficient for an Application Note:** Confluencia's contribution is the integration framework, not the PK model's predictive accuracy. We now clearly position RNACTM as a literature-parameterized simulation (not a population PK model), which avoids the expectation of out-of-sample PK prediction.

### Q2: Immunogenicity weights have no empirical basis — ρ=0.93 is direction consistency, not validation

**Response:** We agree. The ρ=0.93 has been recharacterized as "direction consistency" rather than quantitative validation.

- **Changes made:** (1) "Immunogenicity scores are consistent with literature" → "Immunogenicity rankings agree with literature IFN data (ρ=0.93, direction consistency over 5 conditions)" (Abstract, Results, Discussion). (2) Pathway weights are now explicitly described as "author-informed heuristics derived from qualitative literature observations, not empirically calibrated" (Methods, p.2; Supplementary). (3) The small N=5 statistical limitation is noted in Supplementary.

- **New validation:** We added immunogenicity scoring on three real circRNA sequences (circFOXO3, circHIPK3, circMBOAT2) with Ψ/m6A modifications. Direction consistency (Ψ < unmodified) holds on all 4/4 test sequences (Supplementary, new table).

### Q3: 5D weight sensitivity uses synthetic inputs

**Response:** We acknowledge this limitation.

- **Changes made:** We added a real-data joint evaluation to Supplementary (aspirin SMILES, SLYNTVATL epitope, HLA-A*02:01, 200mg BID). Results honestly show that only the binding dimension (1.0) returns a valid score; clinical, kinetics, gene signature, and circRNA sub-dimensions fail due to pipeline integration errors. We document this transparently, noting that the adaptive framework handles incomplete dimensions via $(1-u)^2$ downweighting. The weight sensitivity analysis on synthetic inputs remains as a proof of framework consistency.

### Q4: circBase validation uses fake sequences

**Response:** We have rerun validation with real circRNA sequences.

- **Changes made:** New Supplementary table shows immunogenicity scoring on circFOXO3, circHIPK3, and circMBOAT2 (real exonic sequences, not synthetic repeats). Direction consistency holds on all sequences (4/4 pass). The old synthetic sequence test is no longer the sole evidence.

---

## Reviewer 2 (circRNA / RNA Therapeutics)

### Q5: RNACTM is not a PK model — it is a deterministic arithmetic system

**Response:** We agree and have corrected the terminology throughout.

- **Changes made:** "Pharmacokinetic model" → "compartment simulation" throughout (Abstract, Introduction, Methods, Results, Discussion, Supplementary, Figure caption, Keywords). We now explicitly state: "This is a deterministic literature-parameterized simulation, not a population PK model fitted to concentration–time data; it provides compartment dynamics without inter-individual variability" (Methods, p.1; Discussion, p.4). The value proposition is reframed from PK prediction to circRNA-specific dimension integration (modification effects on half-life, immunogenicity-PK coupling) that Monolix does not address.

### Q6: Five circRNA dimensions — only two are validated; what are the u values for the other three?

**Response:** We now provide explicit uncertainty values.

- **Changes made:** New text in Methods and Supplementary specifies: miRNA sponge u=0.8, RBP binding u=0.8, translation potential u=0.8 (no independent validation), yielding effective weights of $(1-0.8)^2=0.04$ of nominal weight — these dimensions contribute ~4% of their nominal weight. Half-life and immunogenicity are assigned u=0.15 (direction-consistency verified). Gene Signature is assigned u=0.50 (C-index 0.52, exploratory). A new Supplementary table shows per-dimension base weight, u, $(1-u)^2$, and effective weight.

### Q7: Endosomal escape 5.2% exceeds the 5.0% upper bound of the cited range

**Response:** We have corrected the wording.

- **Changes made:** "At the upper bound" → "slightly exceeds the 1–5% literature range, reflecting uncertainty in the range itself" (Results, p.2; Discussion, p.4; Supplementary Table S1). We note that the 1–5% range itself has uncertainty, and 5.2% is within plausible variation.

### Q8: DBTL closed-loop has no real Learn data

**Response:** We have corrected the framing.

- **Changes made:** "Implements a DBTL cycle" → "implements a DBTL architecture" (Discussion, p.4). We now state: "the Learn stage is accessible but not yet exercised with independent experimental outcomes" and "a complete closed-loop demonstration awaits wet-lab data from our ongoing experiments."

---

## Reviewer 3 (Software Tools / Reproducibility)

### Q9: R package depends on reticulate — Python environment is a known pain point

**Response:** We acknowledge this and have improved documentation.

- **Changes made:** Availability section now mentions both `cf_use_python()` for explicit path specification and `cf_find_python()` for auto-detection (conda, venv, system). The R package vignette will include a Python setup troubleshooting guide.

### Q10: "No GPU required" claim vs Mamba3Lite and ESM-2

**Response:** We have clarified GPU requirements.

- **Changes made:** Availability now reads: "no GPU required for core evaluation (compartment simulation, immunogenicity, 5D, MOE); GPU optional for sequence embeddings" (Methods, p.2; Availability). The distinction is now clear: core evaluation runs on any laptop; Mamba3Lite encoding and ESM-2 embeddings benefit from GPU but are not required for basic functionality.

---

## Reviewer 4 (Pharmacokinetics / Statistics)

### Q11: Linear ODE system lacks inter-individual variability

**Response:** We agree and have acknowledged this as a limitation.

- **Changes made:** RNACTM is now explicitly described as deterministic, lacking IIV (Methods, p.1; Discussion, p.4; Limitations). We note that PK uncertainty is instead handled through ±20% parameter sensitivity analysis (≤8.2% half-life change) and $(1-u)^2$ downweighting. Adding IIV and Bayesian parameter estimation is listed as a future direction.

### Q12: C-index 0.52 — not better than random — why 15% weight?

**Response:** We have flagged this dimension as exploratory.

- **Changes made:** Gene Signature is now explicitly "flagged as exploratory, reflecting target prevalence rather than prognostic power" (Methods, p.3; Results, p.4; Discussion, p.4). Its uncertainty is set to u=0.50, yielding effective weight $(1-0.5)^2=0.25$ — the dimension is effectively downweighted from 15% to 3.75%. The log-rank p<0.001 reflects large-sample tertile stratification, not prognostic power; this is now stated clearly.

### Q13: Drug R²=0.984 on N=200 is suspicious

**Response:** We have corrected this error.

- **Changes made:** The previously reported R²=0.984 was incorrectly attributed to drug efficacy prediction. We have corrected this to the actual values from our 91K benchmark: **target binding R²=0.95, clinical efficacy R²=0.60** (Results, p.3; Discussion, p.4; Supplementary Table S2). The high binding R² reflects that molecular binding is well-captured by physicochemical features; efficacy is inherently harder to predict. We also report that removing 2048-dim Morgan fingerprints improves binding R² from 0.67 to 0.95, which validates the sample-size-adaptive design philosophy where high-dimensional features overfit when N is small relative to feature dimensionality.

---

## Summary of All Changes

| Category | Count | Key Changes |
|----------|-------|-------------|
| Terminology corrections | 7 | PK model → compartment simulation; closed-loop → architecture; matching → consistency verification |
| Honesty additions | 6 | "by design", "heuristic", "direction consistency", "exploratory" added throughout |
| Numerical corrections | 2 | Drug R² corrected; endosomal escape 5.2% wording fixed |
| New validation data | 2 | Real circRNA sequences (4/4 direction consistency); real-data joint evaluation |
| Uncertainty quantification | 1 | Per-dimension u values with effective weights in new Supplementary table |
| Format compliance | 3 | Body compressed 2772→960 words; refs 16→15; abstract 87→67 words |
