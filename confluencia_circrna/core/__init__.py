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
- features: FeatureSpec configuration dataclass
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
    # drug_response
    "DrugResponsePredictor",
    "DrugResponseFeatures",
    "SynergyScore",
    "DrugClass",
    "predict_drug_response",
    "compute_drug_response_score",
    "recommend_treatment",
    # rna_docking
    "RNADockingPredictor",
    "DockingFeatures",
    "BindingSite",
    "DockingResult",
    "DrugCandidate",
    "RNAMotifType",
    "predict_rna_docking",
    "compute_docking_score",
    "design_rna_targeting_drug",
]