"""
Confluencia circRNA multi-omics analysis platform.

Version 3.0 - Literature-based scoring with structure prediction integration.

Key modules:
- core.immune_sensing: RIG-I/TLR/PKR pathway prediction
- core.structure_prediction: ViennaRNA-based secondary structure analysis
- pipeline: Complete analysis workflow
- training: Data loading and normalization
"""

__version__ = "3.0.0"

from confluencia_circrna.core.immune_sensing import (
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
)

from confluencia_circrna.core.structure_prediction import (
    StructurePredictor,
    StructureFeatures,
    compute_pkr_score_from_structure,
)

from confluencia_circrna.pipeline.circrna_pipeline import (
    CircRNAPipeline,
    CircRNAPipelineConfig,
    CircRNAPipelineResult,
    run_pipeline,
)

__all__ = [
    "predict_circrna_immunogenicity",
    "ImmuneSensingConfig",
    "StructurePredictor",
    "StructureFeatures",
    "compute_pkr_score_from_structure",
    "CircRNAPipeline",
    "CircRNAPipelineConfig",
    "CircRNAPipelineResult",
    "run_pipeline",
]