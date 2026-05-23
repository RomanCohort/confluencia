"""
Core modules for circRNA analysis.
"""

from .encoder import CircRNAEncoder, CircRNAEncoderConfig
from .predictor import CircRNAPredictor
from .pipeline import CircRNAPipeline

__all__ = [
    "CircRNAEncoder",
    "CircRNAEncoderConfig",
    "CircRNAPredictor",
    "CircRNAPipeline",
]