# Cover Letter — Bioinformatics Application Note

**Manuscript title:** Confluencia circRNA: A Platform for Circular RNA Immunogenicity Prediction and Sequence Design

**Manuscript type:** Application Note

---

Dear Editor,

We submit this Application Note entitled "Confluencia circRNA: A Platform for Circular RNA Immunogenicity Prediction and Sequence Design" for consideration in Bioinformatics.

**Summary**

Confluencia is an open-source platform purpose-built for circular RNA immunogenicity prediction, structure analysis, and sequence optimization. It integrates literature-backed scoring for RIG-I, TLR7/8, and PKR innate immune pathways with ViennaRNA-based secondary structure prediction, modification site mapping (m6A, IRES, miRNA binding), and Pareto-based evolutionary sequence optimization. The platform is accessible via Python API, Streamlit web interface, R package (27 functions), VS Code extension (11 commands), and CLI.

**Why this matters**

Circular RNA has emerged as a promising vaccine platform and therapeutic delivery vehicle, yet the field lacks computational tools purpose-built for circRNA's unique biology. Unlike linear mRNA, circRNA's covalently closed loop eliminates 5'/3' terminus recognition by RIG-I, shifting innate sensing to dsRNA backbone structures—a distinction no existing tool addresses. Confluencia is the first platform, to our knowledge, to provide circRNA-specific immunogenicity scoring with integrated sequence design capabilities.

**What the Note delivers**

- Multi-pathway immunogenicity scoring (RIG-I/TLR/PKR) with literature-derived circRNA-specific weights, validated against published IFN-β measurements (r=0.91, N=7)
- Integrated structure, modification, and clinical prediction in a single pipeline
- REINFORCE-based evolutionary optimization with Pareto front multi-objective selection
- Modular architecture with six interfaces—Python API, R package, Streamlit, VS Code, CLI, and Docker
- All evaluations run on CPU within seconds; no GPU required

**Beyond this Note**

The present Note focuses on the core immunogenicity prediction and sequence design pipeline at the 1000-word limit. However, the platform already incorporates more advanced capabilities—including tumor microenvironment simulation, RL-agent-based therapeutic optimization, multi-drug combination modeling, and patient stratification—which we consider the foundation for a substantially extended research article. We are actively pursuing experimental collaborations to calibrate immunogenicity weights against wet-lab data and extend the evolutionary optimization into a TME-aware closed-loop system.

**A glimpse of what lies ahead:** In a pilot application of our evolutionary module, we observed that combining m6A modification with moderate GC content (45–55%) consistently produced circRNA variants with approximately 40% lower predicted immunogenicity while retaining translation potential—a trade-off that existing linear RNA design tools cannot model because they lack circRNA-specific immune pathway scoring. This finding suggests that sequence-modification synergy, rather than either factor alone, may be the key to designing circRNA therapeutics with both low immunogenicity and high expression. We are now pursuing experimental validation of this prediction, alongside calibration of the full platform against wet-lab data. We view this Application Note as establishing the computational foundation for a platform that could meaningfully accelerate circRNA therapeutic development.

**Availability**

Fully open-source under MIT license at https://github.com/IGEM-FBH/confluencia, with comprehensive documentation and quick-start examples.

We believe Confluencia addresses an unmet need in the rapidly growing circRNA therapeutics field and is suitable as a Bioinformatics Application Note. We look forward to your review.

Sincerely,

The Confluencia Team
IGEM-FBH, Jilin University
igem-fbh@jlu.edu.cn
