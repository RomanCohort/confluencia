"""
circRNA 3D Data Generation Pipeline package.

Generates high-quality circRNA 3D structures using:
- ViennaRNA for secondary structure prediction
- RoseTTAFold2NA for initial 3D coordinates
- OpenMM for BSJ cyclization and MD relaxation
- Quality filtering with confidence scoring
"""

from .stage1_vienna import ViennaRNAPredictor
from .stage2_rosetta import RoseTTAFold2NAPredictor
from .stage3_cyclize import BSJCyclizer
from .stage4_md import MDRelaxation
from .stage5_quality import QualityFilter, save_dataset, convert_to_torusfold_format
from .pipeline import CircRNA3DPipeline, ParallelPipeline

__version__ = '0.1.0'
__all__ = [
    'ViennaRNAPredictor',
    'RoseTTAFold2NAPredictor',
    'BSJCyclizer',
    'MDRelaxation',
    'QualityFilter',
    'CircRNA3DPipeline',
    'ParallelPipeline'
]
