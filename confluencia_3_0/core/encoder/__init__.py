"""
confluencia_circrna.encoder — Neural circRNA sequence encoder.

Converts circRNA nucleotide sequence + gene expression into the 13-key
scoring dict consumed by JointScoringEngine._score_circrna(), enabling
end-to-end prediction without manual sub-score input.
"""

from confluencia_circrna.encoder.config import CircRNAEncoderConfig
from confluencia_circrna.encoder.adapter import CircRNAEncoderAdapter

__all__ = ["CircRNAEncoderConfig", "CircRNAEncoderAdapter"]
