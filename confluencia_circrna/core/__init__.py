"""
Core analysis modules for circRNA multi-omics.

Modules:
- immune_sensing: RIG-I/TLR/PKR pathway prediction (literature-based weights)
- structure_prediction: ViennaRNA-based secondary structure analysis
- folding_kinetics: Folding rate, barrier, suboptimal structures
- cotrans_folding: Cotranscriptional folding simulation
- folding_pathways: Folding pathway analysis and landscape
- drug_response: Drug response and treatment prediction
- rna_docking: RNA-small molecule docking prediction
- rna_modifications: m6A, IRES, miRNA, RBP modification prediction
- clinical_prediction: Clinical outcome and survival prediction
- cirrna_evolution: circRNA sequence evolutionary optimization
- bsj_features: Back-splice junction feature extraction (NEW)
"""

from confluencia_circrna.core.immune_sensing import (
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
)
from confluencia_circrna.core.structure_prediction import (
    StructurePredictor,
    StructureFeatures,
    compute_pkr_score_from_structure,
)
from confluencia_circrna.core.folding_kinetics import (
    FoldingKineticsPredictor,
    KineticsFeatures,
    SuboptimalStructure,
    predict_folding_kinetics,
    compute_kinetics_score,
)
from confluencia_circrna.core.cotrans_folding import (
    CotranscriptionalFoldingPredictor,
    CotransFeatures,
    IntermediateStructure,
    KineticTrap,
    predict_cotrans_folding,
    compare_transcription_rates,
    compute_cotrans_immunogenicity,
)
from confluencia_circrna.core.folding_pathways import (
    FoldingPathwayAnalyzer,
    PathwayFeatures,
    FoldingPathway,
    StructureTransition,
    TransitionState,
    PathwayType,
    analyze_folding_pathways,
    compute_pathway_immunogenicity,
    find_optimal_folding_conditions,
)
from confluencia_circrna.core.drug_response import (
    DrugResponsePredictor,
    DrugResponseFeatures,
    SynergyScore,
    DrugClass,
    predict_drug_response,
    compute_drug_response_score,
    recommend_treatment,
)
from confluencia_circrna.core.rna_docking import (
    RNADockingPredictor,
    DockingFeatures,
    BindingSite,
    DockingResult,
    DrugCandidate,
    RNAMotifType,
    predict_rna_docking,
    compute_docking_score,
    design_rna_targeting_drug,
)
from confluencia_circrna.core.rna_modifications import (
    ModificationPredictor,
    ModificationFeatures,
    ModificationSite,
    IRESSite,
    MiRNABindingSite,
    RBPBindingSite,
    ModificationType,
    predict_modifications,
    compute_modification_score,
)
from confluencia_circrna.core.clinical_prediction import (
    ClinicalOutcomePredictor,
    ClinicalFeatures,
    SurvivalPrediction,
    BiomarkerScore,
    AdverseEventRisk,
    Endpoint,
    predict_clinical_outcome,
    compute_clinical_score,
    generate_clinical_report,
)
from confluencia_circrna.core.cirrna_evolution import (
    CircRNAEvolutionConfig,
    CircRNAEvolutionArtifacts,
    evolve_cirrna,
    run_cirrna_evolution,
    optimize_for_translation,
    optimize_for_stability,
    optimize_for_immune_safety,
    compute_cirrna_objectives,
    mutate_backbone,
    optimize_ires,
    shuffle_utr,
)
from confluencia_circrna.core.bsj_features import (
    BSJFeatureExtractor,
    BSJFeatures,
    AluMatch,
    SpliceSiteScore,
    detect_alu_elements,
    compute_intron_complementarity,
    score_splice_site,
    predict_circularization_efficiency,
    extract_bsj_features,
    get_bsj_summary,
    # NEW: Real-time detection
    JunctionSignal,
    BSJValidationResult,
    detect_junction_signal,
    validate_bsj_realtime,
    # NEW: Conservation scoring
    ConservationAnnotation,
    compute_bsj_conservation_score,
    get_cross_species_bsj,
    add_conservation_to_features,
)

__all__ = [
    # immune_sensing
    "predict_circrna_immunogenicity",
    "ImmuneSensingConfig",
    # structure_prediction
    "StructurePredictor",
    "StructureFeatures",
    "compute_pkr_score_from_structure",
    # folding_kinetics
    "FoldingKineticsPredictor",
    "KineticsFeatures",
    "SuboptimalStructure",
    "predict_folding_kinetics",
    "compute_kinetics_score",
    # cotrans_folding
    "CotranscriptionalFoldingPredictor",
    "CotransFeatures",
    "IntermediateStructure",
    "KineticTrap",
    "predict_cotrans_folding",
    "compare_transcription_rates",
    "compute_cotrans_immunogenicity",
    # folding_pathways
    "FoldingPathwayAnalyzer",
    "PathwayFeatures",
    "FoldingPathway",
    "StructureTransition",
    "TransitionState",
    "PathwayType",
    "analyze_folding_pathways",
    "compute_pathway_immunogenicity",
    "find_optimal_folding_conditions",
    # drug_response (circRNA vaccine focused)
    "DrugResponsePredictor",
    "DrugResponseFeatures",
    "SynergyScore",
    "DrugClass",
    "predict_drug_response",
    "compute_drug_response_score",
    "recommend_treatment",
    # rna_docking (RNA-targeting drugs)
    "RNADockingPredictor",
    "DockingFeatures",
    "BindingSite",
    "DockingResult",
    "DrugCandidate",
    "RNAMotifType",
    "predict_rna_docking",
    "compute_docking_score",
    "design_rna_targeting_drug",
    # rna_modifications
    "ModificationPredictor",
    "ModificationFeatures",
    "ModificationSite",
    "IRESSite",
    "MiRNABindingSite",
    "RBPBindingSite",
    "ModificationType",
    "predict_modifications",
    "compute_modification_score",
    # clinical_prediction
    "ClinicalOutcomePredictor",
    "ClinicalFeatures",
    "SurvivalPrediction",
    "BiomarkerScore",
    "AdverseEventRisk",
    "Endpoint",
    "predict_clinical_outcome",
    "compute_clinical_score",
    "generate_clinical_report",
    # cirrna_evolution
    "CircRNAEvolutionConfig",
    "CircRNAEvolutionArtifacts",
    "evolve_cirrna",
    "run_cirrna_evolution",
    "optimize_for_translation",
    "optimize_for_stability",
    "optimize_for_immune_safety",
    "compute_cirrna_objectives",
    "mutate_backbone",
    "optimize_ires",
    "shuffle_utr",
    # bsj_features (NEW)
    "BSJFeatureExtractor",
    "BSJFeatures",
    "AluMatch",
    "SpliceSiteScore",
    "detect_alu_elements",
    "compute_intron_complementarity",
    "score_splice_site",
    "predict_circularization_efficiency",
    "extract_bsj_features",
    "get_bsj_summary",
    # bsj_features - NEW (real-time & conservation)
    "JunctionSignal",
    "BSJValidationResult",
    "detect_junction_signal",
    "validate_bsj_realtime",
    "ConservationAnnotation",
    "compute_bsj_conservation_score",
    "get_cross_species_bsj",
    "add_conservation_to_features",
]