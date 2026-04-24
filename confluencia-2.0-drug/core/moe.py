"""
Drug MOE module - re-exports from shared with drug-specific defaults.

This module provides backwards compatibility while centralizing the MOE
implementation in confluencia_shared.
"""
from __future__ import annotations

# Import everything from shared
from confluencia_shared.moe import (
    MOERegressor,
    ComputeProfile,
    ExpertConfig,
    choose_compute_profile,
    EXPERT_CONFIG_DRUG,
)

# Drug-specific configuration
DRUG_EXPERT_CONFIG = EXPERT_CONFIG_DRUG

__all__ = [
    "MOERegressor",
    "ComputeProfile",
    "ExpertConfig",
    "choose_compute_profile",
    "DRUG_EXPERT_CONFIG",
]
