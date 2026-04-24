"""
Joint Drug-Epitope-PK Evaluation Module
========================================

Unifies drug efficacy prediction, MHC binding prediction, and PK simulation
into a single three-dimensional evaluation pipeline.
"""

from .joint_input import JointInput
from .scoring import (
    ClinicalScore,
    BindingScore,
    KineticsScore,
    JointScore,
    JointScoringEngine,
)
from .fusion_layer import JointFusionLayer, FusionStrategy
from .joint_evaluator import (
    JointEvaluationEngine,
    JointEvaluationResult,
)

__all__ = [
    # Input
    "JointInput",
    # Scoring
    "ClinicalScore",
    "BindingScore",
    "KineticsScore",
    "JointScore",
    "JointScoringEngine",
    # Fusion
    "JointFusionLayer",
    "FusionStrategy",
    # Evaluation
    "JointEvaluationEngine",
    "JointEvaluationResult",
]
