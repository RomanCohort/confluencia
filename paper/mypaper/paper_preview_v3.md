# Introduction

Therapeutic circular RNA (circRNA) offers advantages over linear mRNA in
structural stability, prolonged expression duration, and controlled
immunogenicity [@wesselhoeft2018; @chen2019], yet computational tools
for circRNA drug discovery remain fragmented. Researchers must switch
between specialized predictors---NetMHCpan for MHC binding
[@reynisson2020], ADMETlab for toxicity [@xiong2024], and generic PK
packages---while lacking any circRNA-specific pharmacokinetic
simulation. Moreover, circRNA wet-lab studies typically yield fewer than
300 labeled samples, a regime where standard deep learning approaches
fail, yet no existing tool adapts model complexity to this constraint.

Confluencia addresses this gap by integrating circRNA-specific
functionality into a single platform: RNACTM, a six-compartment
pharmacokinetic (PK) model with literature-derived rate constants for
five nucleotide modifications ($\Psi$, m6A, 5mC, ms2m6A, unmodified),
enabling 72 h trajectory simulation without requiring patient PK data; a
sample-size-adaptive Mixture-of-Experts (MOE) ensemble [@jacobs1991]
that scales model complexity to available data, critical for the $N<300$
regime; Mamba3Lite sequence encoding [@gu2023]; ADMET toxicity
prediction [@baell2010; @brenk2008]; and a five-dimension clinical
evaluation system.

# Implementation

**RNACTM Pharmacokinetic Model.** RNACTM implements a six-compartment
ordinary differential equation (ODE) system tracing circRNA dynamics
from subcutaneous injection through LNP encapsulation and circulation,
endosomal uptake via clathrin-mediated pathways, cytoplasmic release,
protein translation, and systemic clearance (Fig. 1A). All rate
constants are derived from published studies
[@wesselhoeft2018; @hassett2019; @liu2023] rather than fitted from
patient data: degradation rate $k_{\mathrm{deg}}=0.111/\mathrm{h}$
(half-life 6.24 h for unmodified circRNA), endosomal escape fraction
4.43%, and modification-specific half-life extensions ($\Psi$:
${\sim}15$ h; m6A: ${\sim}10.8$ h; 5mC: ${\sim}7.8$ h). The model
produces 72 h compartment concentration trajectories, protein expression
time-courses (Fig. 1B), dose-response curves, and enables comparison of
delivery routes (IV/SC/IM) and modification strategies without requiring
patient PK data.

**MOE Ensemble.** Following the mixture-of-experts framework
[@jacobs1991], a sample-size-adaptive ensemble selects experts based on
data availability: $N<80$ activates Ridge regression and Histogram
Gradient Boosting (HGB); $80\leq N<300$ adds Random Forest; $N\geq300$
adds MLP. Expert weights derive from inverse out-of-fold RMSE across
5-fold stratified cross-validation:
$w_e = {1/\mathrm{RMSE}_{\mathrm{OOF},e}}/{\sum_{e'} 1/\mathrm{RMSE}_{\mathrm{OOF},e'}}$.
Learning curve analysis reveals that predictions fail when $N<24$
($R^2<0$), become feasible at $N\geq48$ ($R^2>0.45$), and saturate at
$N\geq200$ ($R^2>0.75$), guiding sample size requirements for circRNA
studies.

**Mamba3Lite Encoder.** Mamba3Lite [@gu2023] encodes 8--11 amino acid
peptides via three parallel selective state-space recurrences with
distinct decay rates (fast/medium/slow) and four-scale pooling
(residue/local/meso/global), producing 96-dimensional embeddings.
Optional lightweight self-attention ($d=16$) achieves MAE = 0.395,
$R^2$ = 0.802. ESM-2 protein language model embeddings [@lin2023] were
also evaluated; however, mean pooling of 8--11 AA sequences destroys
position-specific MHC binding motifs, yielding
AUC $<$`<!-- -->`{=html}0.60 on binding prediction---confirming that
pretrained language models are suboptimal for short peptide tasks where
position information is critical.

**ADMET and Toxicophore.** QSAR-based multi-endpoint prediction covers 8
ADMET endpoint categories: hERG channel blockade, AMES mutagenicity,
CYP450 isoform inhibition (1A2, 2C9, 2C19, 2D6, 3A4), BBB permeability,
and hepatotoxicity. Eighty-four SMARTS structural alerts include 25
PAINS patterns [@baell2010], Brenk filters [@brenk2008],
circRNA-specific alerts, and general toxicity patterns. Dose-dependent
modeling estimates therapeutic index via Hill equation dose-response
curves.

**Joint Evaluation.** A five-dimension (5D) hybrid evaluation system
integrates outputs from Drug, Epitope, RNACTM PK, and Five-Gene modules
using confidence-adaptive weights (Clinical 0.30, Binding 0.20, Kinetics
0.15, Gene Signature 0.15, CircRNA 0.20), producing Go/Conditional/No-Go
recommendations with a safety override mechanism.

# Results

::: {#tab:results}
  **Module**         **Dataset**    **Method**     **Metric**   **Value**
  ------------------ -------------- -------------- ------------ --------------
  Epitope            $N{=}300$      MOE            MAE, $R^2$   0.389, 0.819
  Epitope            $N{=}300$      HGB            $R^2$        0.794
  Epitope            $N{=}300$      Ridge          MAE, $R^2$   0.639, 0.533
  Drug efficacy      $N{=}200$      Ridge          $R^2$        0.984
  Drug ablation      $N{=}200$      No Morgan FP   $R^2$        0.960
  Binding            288 K IEDB     Baseline       AUC          0.739
  Binding            61 peptides    Baseline       AUC          0.653
  RNACTM half-life   Literature     ODE            Error        4.1%
  Toxicophore        12 molecules   SMARTS         Recall       100%

  : Confluencia performance across modules. 5-fold stratified CV except
  binding (sequence-aware split) and toxicophore (recall on 12
  molecules).
:::

RNACTM half-life predictions match literature within 4.1% for three
validated modifications---unmodified (6.24 h vs 6.0 h), $\Psi$ (15.61 h
vs 15.0 h), and m6A (11.24 h vs 10.8 h); 5mC and ms2m6A lack published
circRNA half-life data for validation. Protein expression duration is
overestimated (97 h simulated vs 48 h reported, 102% error), and
endosomal escape fraction shows 121% error (4.43% simulated vs 2.0%
literature, though within the 1--5% range reported by Gilleron et al.
[@gilleron2013]), indicating that clearance and escape dynamics require
refinement. The MOE ensemble achieves 39.2% MAE reduction over Ridge
($p<0.001$, Cohen's $d=-6.36$) and 4.9% over HGB---the strongest
individual baseline. For drug efficacy prediction where features are
low-dimensional ($d=154$), Ridge outperforms MOE ($R^2$ 0.984 vs 0.982),
confirming that adaptive gating adds unnecessary complexity when data is
sufficient and feature space is compact. On 288 K binary binding
prediction, MOE's regression-trained gating mechanism produces near-zero
positive predictions (F1$<$`<!-- -->`{=html}0.01), indicating that the
ensemble design is specialized for regression tasks and not suitable for
binary classification at scale.

Feature ablation reveals that removing environment features from the
full model drops $R^2$ from 0.853 to 0.515 (a 0.34 decline), while using
only environment features yields $R^2=-0.016$---environment variables
are necessary but insufficient alone. Biochemical statistics (16
dimensions, 5% of features) account for 65.9% of MAE increase upon
removal. Removing 2048-bit Morgan fingerprints from the 2083-dimension
full feature set improves drug $R^2$ from 0.668 to
0.960---high-dimensional sparse features severely overfit in small
samples. MHC-I binding prediction (AUC 0.739 on 288 K IEDB; 0.653 on 61
peptides) trails NetMHCpan-4.1 [@reynisson2020] (AUC 0.92--0.96); we
recommend NetMHCpan for binding-only tasks. Toxicophore detection
achieves 100% recall on 12 verified PAINS-containing molecules; the 84
structural alerts include 25 PAINS SMARTS patterns, of which 10 unique
patterns are empirically triggered by test molecules.

<figure id="fig:overview" data-latex-placement="t">
<embed src="figures/rnactm_trajectories.pdf" style="width:90.0%" />
<figcaption>RNACTM 72 h protein expression trajectories for four
nucleotide modifications: <span
class="math inline"><em>Ψ</em></span>-modified circRNA shows sustained
expression (half-life <span class="math inline"> ∼ 15</span> h) versus
unmodified (6.24 h). Architecture diagram (Panel A) will be added as
composite figure for final submission.</figcaption>
</figure>

# Discussion

Confluencia provides the first integrated platform for circRNA drug
discovery, combining circRNA-specific PK simulation, small-sample
ensemble learning, and ADMET evaluation in a single workflow---functions
unavailable in specialized tools (NetMHCpan for binding
[@reynisson2020], ADMETlab for toxicity [@xiong2024], ESM-2 for sequence
encoding [@lin2023]). Rather than competing with specialized predictors
on individual tasks (MHC-I AUC 0.65--0.74 vs NetMHCpan AUC ${>}0.92$),
Confluencia's value lies in unified functionality: PK trajectory
simulation enables nucleotide modification comparison, dose
optimization, and delivery route selection that no single specialized
tool provides.

Limitations require honest acknowledgment: (1) RNACTM parameters derive
from literature rather than fitted PK data, and protein expression
duration is overestimated (97 h vs 48 h), suggesting the clearance model
needs refinement; (2) ADMET QSAR weights are literature-derived, not
trained on circRNA-specific toxicity data; (3) only MHC-I alleles are
supported (no class II); (4) MOE's adaptive gating adds complexity
without benefit when Ridge suffices for low-dimensional drug efficacy
prediction ($R^2$ 0.984 vs 0.982); (5) the 5D evaluation shows weak
Spearman $r=0.135$ with IFN responses, indicating insufficient
immunogenicity features. Future work should incorporate wet-lab PK data
for RNACTM fitting, class II MHC support, and additional immune response
predictors.

# Availability and Implementation {#availability-and-implementation .unnumbered}

Confluencia v2.5.0 is freely available at
<https://github.com/RomanCohort/confluencia> under MIT license.
Implemented in Python 3.8+ using scikit-learn [@pedregosa2011], RDKit,
NumPy, and Streamlit. Runs on Linux, macOS, and Windows with optional
Docker deployment. A desktop IDE (Confluencia Studio, Electron+React)
provides an integrated workspace with Streamlit-based interfaces.
Benchmark datasets from IEDB [@vita2019], GDSC [@yang2013], and ChEMBL
are included. All experimental results and reproducibility code are
available in the repository's benchmarks directory.

# Acknowledgements {#acknowledgements .unnumbered}

IGEM-FBH Team, Jilin University. We thank IEDB, ChEMBL, and open-source
communities. Funding: The First Bethune Hospital and College of Computer
Science and Technology, Jilin University.

::: thebibliography
15

Wesselhoeft, R.A., Kowalski, P.S. and Anderson, D.G. (2018) Engineering
circular RNA for potent and stable translation in eukaryotic cells.
*Nat. Commun.*, **9**, 2629.

Chen, Y.G., Kim, M. and Chen, X. (2019) N6-methyladenosine modification
controls circular RNA immune evasion. *Nature*, **586**, 556--560.

Liu, X., Linn, M., Bhagat, S. et al. (2023) Mechanistic compartmental
pharmacokinetic modeling of oligonucleotides. *CPT Pharmacometrics Syst.
Pharmacol.*, **12**, 391--404.

Hassett, K.J., Benenato, K.E., Jacquinet, E. et al. (2019) Optimization
of lipid nanoparticles for intramuscular administration of mRNA
vaccines. *Mol. Ther.*, **27**, 773--784.

Jacobs, R.A., Jordan, M.I., Barto, A.G. et al. (1991) Adaptive mixtures
of local experts. *Neural Comput.*, **3**, 79--87.

Gu, A. and Dao, T. (2023) Mamba: Linear-time sequence modeling with
selective state spaces. *arXiv*, 2312.00752.

Reynisson, B., Barra, C., Kaabinejadian, S. et al. (2020) NetMHCpan-4.1:
improved peptide-MHC class I interaction predictions. *J. Immunol.
Methods*, **480**, 112761.

Xiong, J., Pan, P., Guo, Q. et al. (2024) ADMETlab 3.0: an updated
comprehensive webserver for ADMET prediction. *Brief. Bioinform.*,
**25**, bbae536.

Baell, J.B. and Holloway, G.A. (2010) New substructure filters for
removal of pan assay interference compounds (PAINS). *J. Med. Chem.*,
**53**, 2719--2740.

Lin, Z., Akin, H., Rao, R. et al. (2023) Evolutionary-scale prediction
of atomic-level protein structure with a language model. *Science*,
**379**, 1123--1130.

Vita, R., Mahmutovic, M. and Sette, A. (2019) The Immune Epitope
Database (IEDB): 2018 update. *Nucleic Acids Res.*, **47**, D339--D343.

Yang, W., Soares, J., Greninger, P. et al. (2013) Genomics of Drug
Sensitivity in Cancer (GDSC). *Nucleic Acids Res.*, **41**, D983--D989.

Gilleron, J., Querbes, W., Zeigerer, A. et al. (2013) Image-based
analysis of lipid nanoparticle-mediated siRNA delivery, intracellular
trafficking and endosomal escape. *Nat. Biotechnol.*, **31**, 638--646.

Brenk, R., Schipani, A. and James, D. (2008) Lessons learned from
molecular scaffold design. *ChemMedChem*, **3**, 435--444.

Pedregosa, F., Varoquaux, G. and Gramfort, A. (2011) Scikit-learn:
Machine learning in Python. *JMLR*, **12**, 2825--2830.
:::

[^1]: The First Bethune Hospital of Jilin University and College of
    Computer Science and Technology, Jilin University, Changchun, China
