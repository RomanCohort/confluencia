"""
Core analysis modules for circRNA multi-omics.

Modules:
- immune_sensing: RIG-I/TLR/PKR pathway prediction (literature-based weights)
- structure_prediction: ViennaRNA-based secondary structure analysis
- features: FeatureSpec configuration dataclass
"""

from confluencia_circrna.core.immune_sensing import (
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
)
from confluencia_circrna.core.structure_prediction import (
    StructurePredictor,
    StructureFeatures,
)

__all__ = [
    "predict_circrna_immunogenicity",
    "ImmuneSensingConfig",
    "StructurePredictor",
    "StructureFeatures",
]
