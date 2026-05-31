# Cover Letter — Bioinformatics Application Note

**Manuscript title:** Confluencia: An uncertainty-adaptive cross-modality evaluation platform for circRNA therapeutic development

**Manuscript type:** Application Note

---

Dear Editor,

We submit Confluencia, the first computational platform that evaluates circRNA-specific therapeutic dimensions across both small-molecule and circRNA modalities in a unified framework.

**Novelty.** Existing tools address individual modalities—NetMHCpan for MHC binding, ADMETlab for ADMET screening, Monolix for population PK—but none evaluates circRNA-specific dimensions: nucleotide modification effects on half-life, circRNA immunogenicity pathways, IRES-mediated translation, or miRNA sponge activity. Confluencia integrates these evaluations with uncertainty-adaptive five-dimension assessment, producing cross-modality Go/Conditional/No-Go recommendations.

**Honesty.** We proactively acknowledge that our validations are computational rather than experimental: (1) RNACTM compartment simulation is a deterministic, literature-parameterized system (not a population PK model fitted to concentration–time data); (2) immunogenicity scoring achieves direction consistency with literature IFN data but pathway weights are heuristic; (3) three of five circRNA functional dimensions lack independent evaluation and are auto-downweighted via $(1-u)^2$. Wet-lab validation in our laboratory is underway and will be reported separately.

**Reproducibility.** Confluencia is freely available under MIT license at https://github.com/IGEM-FBH/confluencia with three interfaces (GUI, R package with 27 functions, VS Code extension with 11 commands), offline-first Hub sharing, and plugin extensibility. Core evaluation runs in under 5 minutes on a standard laptop without GPU.

We believe Confluencia addresses an unmet need in the rapidly growing circRNA therapeutics field and is suitable as a Bioinformatics Application Note.

Sincerely,

IGEM-FBH Team
First Bethune Hospital of Jilin University
