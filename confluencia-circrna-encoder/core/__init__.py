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
]