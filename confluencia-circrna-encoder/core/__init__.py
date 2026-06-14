"""
Core modules for circRNA analysis.

Includes all components adapted from drug 2.0:
- encoder: Sequence encoding
- predictor: Prediction interface
- pipeline: Processing pipeline
- scoring: Immunogenicity scoring
- features: Feature extraction
- training: Model training
- innate_immune: RIG-I/TLR/PKR activation
- dose_tox: Dose-response and toxicity
- immune_abm: Agent-based immune simulation
- generative: Sequence generation/optimization
- evolution: Evolutionary optimization
- reliability: Prediction confidence
- pkpd: Pharmacokinetic/pharmacodynamic modeling
- multiscale: Multi-scale modeling
- admet: ADMET-like properties
- moe: Mixture of Experts
"""

from .encoder import CircRNAEncoder, CircRNAEncoderConfig
from .predictor import CircRNAPredictor
from .pipeline import CircRNAPipeline
from .scoring import CircRNAScorer
from .features import CircRNAFeatureExtractor
from .training import train_model

# New modules adapted from drug 2.0
from .innate_immune import InnateImmunePredictor, quick_predict
from .dose_tox import CircRNADoseResponse, quick_dose_predict
from .immune_abm import ImmuneABM, simulate_circrna_response
from .generative import CircRNAGenerator, generate_optimized_sequence
from .evolution import CircRNAEvolution, evolve_sequence
from .reliability import ReliabilityEstimator, assess_prediction_reliability
from .pkpd import CircRNAPKPD, simulate_pkpd
from .multiscale import MultiscaleModel, multiscale_simulation
from .admet import CircRNAADMET, quick_admet
from .moe import CircRNAMOE, quick_moe_predict

# TorusFold v2 components (AF3-inspired, 2026-06-14)
from .tpe import TorusPositionalEncoding, TorusPositionalEncoding2D, CircularRelativeBias
from .equivariant_backbone import CircEquivariantBackbone, TorusTransformerLayer
from .triangle_update import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    PairTransition,
    CircPairformerBlock,
    CircPairformerStack,
)
from .diffusion_structure import (
    CircDiffusionStructure,
    SimpleStructureHead,
    DiffusionConditioning,
    DiffusionDenoiser,
    FlexibleStructureHead,  # New: multi-conformation output
    ClosureConstrainedDiffusion,  # New: flexibility-aware diffusion
)
from .irs_pair import BSJPairAnalyzer, circular_distance_matrix
from .torusfold import TorusFold, TorusFoldConfig, PairInitialization, PairPredictionHead

# Tertiary interaction modules (future use, not enabled by default)
from .tertiary_interaction import (
    LongRangeAttention,
    LoopCrossAttention,
    PseudoknotUpdater,
    TertiaryInteractionModule,
    circ_contact_from_linear,
)

__all__ = [
    # Base components
    "CircRNAEncoder",
    "CircRNAEncoderConfig",
    "CircRNAPredictor",
    "CircRNAPipeline",
    "CircRNAScorer",
    "CircRNAFeatureExtractor",
    "train_model",

    # New components from drug 2.0
    "InnateImmunePredictor",
    "quick_predict",
    "CircRNADoseResponse",
    "quick_dose_predict",
    "ImmuneABM",
    "simulate_circrna_response",
    "CircRNAGenerator",
    "generate_optimized_sequence",
    "CircRNAEvolution",
    "evolve_sequence",
    "ReliabilityEstimator",
    "assess_prediction_reliability",
    "CircRNAPKPD",
    "simulate_pkpd",
    "MultiscaleModel",
    "multiscale_simulation",
    "CircRNAADMET",
    "quick_admet",
    "CircRNAMOE",
    "quick_moe_predict",

    # TorusFold v2 (AF3-inspired)
    "TorusFold",
    "TorusFoldConfig",
    "PairInitialization",
    "PairPredictionHead",
    "TorusPositionalEncoding",
    "TorusPositionalEncoding2D",
    "CircularRelativeBias",
    "CircEquivariantBackbone",
    "TorusTransformerLayer",
    # Triangle update (AF3-style)
    "TriangleMultiplicativeUpdate",
    "TriangleAttention",
    "PairTransition",
    "CircPairformerBlock",
    "CircPairformerStack",
    # Structure modules
    "CircDiffusionStructure",
    "SimpleStructureHead",
    "DiffusionConditioning",
    "DiffusionDenoiser",
    "FlexibleStructureHead",
    "ClosureConstrainedDiffusion",
    # IRS/BSJ
    "BSJPairAnalyzer",
    "circular_distance_matrix",
    # Tertiary interaction (future use)
    "LongRangeAttention",
    "LoopCrossAttention",
    "PseudoknotUpdater",
    "TertiaryInteractionModule",
    "circ_contact_from_linear",
]