"""
weight_loader.py — Centralized scoring weight loader.

All scoring weights are loaded from scoring_weights.json.
If the JSON file is missing, hardcoded defaults are used as fallback.
This ensures the system works even without the config file, while
making weights easy to audit and modify in one place.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional


_CONFIG_PATH = Path(__file__).parent / "scoring_weights.json"

# Inline defaults — used when JSON is missing
_DEFAULTS = {
    "version": 1,
    "fusion": {
        "clinical": 0.30, "binding": 0.20, "kinetics": 0.15,
        "gene_signature": 0.15, "circ_rna": 0.20,
    },
    "clinical_sub": {
        "efficacy": 0.40, "binding": 0.35, "immune": 0.25,
    },
    "clinical_safety": {
        "toxicity": 0.6, "inflammation": 0.4,
    },
    "binding_uncertainty_penalty": 0.3,
    "kinetics_sub": {
        "half_life": 0.25, "auc": 0.30, "therapeutic_index": 0.30, "cmax": 0.15,
    },
    "gene_signature_sub": {
        "efficacy": 0.30, "immune": 0.15, "proliferation": 0.15,
        "mito": 0.15, "risk_inverse": 0.15, "tide_inverse": 0.10,
    },
    "circ_rna_sub": {
        "immunotherapy": 0.20, "tumor_killing_index": 0.15,
        "immunogenicity": 0.15, "immune_cycle": 0.10,
        "tme": 0.10, "therapeutic_window": 0.10,
        "tide_inverse": 0.10, "ips_fraction": 0.10,
    },
    "pathway": {
        "proliferation": {"TROP2": 0.4, "NECTIN4": 0.3, "MKI67": 0.15, "MYC": 0.15},
        "immune": {"B7-H4": 0.6, "TROP2": 0.2, "LIV-1": 0.2},
        "mitochondria": {"LIV-1": 0.3, "NECTIN4": 0.1, "BAX": 0.3, "BCL2": 0.3},
    },
    "tide_ips": {
        "tide_risk": 0.3, "tide_tmem65": 0.4, "tide_immune": -0.3,
        "ips_risk_inverse": 0.5, "ips_immune": 0.3,
    },
    "risk_adjustment": {
        "TROP2_high": 0.30, "TROP2_low": 0.15,
        "NECTIN4_high": 0.20, "NECTIN4_low": 0.10,
        "LIV-1_high": 0.15, "LIV-1_low": 0.08,
        "B7-H4_high": 0.10, "B7-H4_low": 0.05,
        "TMEM65_high": 0.25, "TMEM65_low": 0.13,
        "ddr_base": 0.2, "ddr_risk_weight": 0.4,
    },
    "gene_signature_4gene": {
        "trop2": 0.35, "nectin4": 0.25, "liv1": 0.20, "b7h4": 0.20,
        "metastasis_nectin4": 0.5, "metastasis_liv1": 0.5,
    },
    "combined_4gene": {
        "proliferation": 0.30, "immune": 0.25,
        "metastasis": 0.15, "efficacy": 0.30,
    },
    "clinical_uncertainty": {
        "inflammation": 0.2, "toxicity": 0.2,
    },
    "binding_uncertainty": {
        "default": 0.3,
    },
    "kinetics_uncertainty": {
        "hl_low": 0.5, "hl_high": 72.0, "implausible_penalty": 0.3,
        "cmax_high": 1000.0, "extreme_cmax_penalty": 0.2,
    },
    "gene_signature_uncertainty": {
        "extreme_high": 0.8, "extreme_low": 0.2, "extreme_penalty": 0.15,
    },
    "circ_rna_uncertainty": {
        "conflict_high": 0.6, "conflict_penalty": 0.2,
    },
    "go_threshold": 0.65,
    "conditional_threshold": 0.40,
    "safety_floor": 0.30,
}

_cached_weights: Optional[Dict] = None


def _load_json() -> Dict:
    """Load scoring_weights.json, return dict."""
    global _cached_weights
    if _cached_weights is not None:
        return _cached_weights

    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            _cached_weights = json.load(f)
        return _cached_weights

    # Fallback to defaults
    warnings.warn(
        f"scoring_weights.json not found at {_CONFIG_PATH}, using inline defaults"
    )
    _cached_weights = _DEFAULTS.copy()
    return _cached_weights


def load_scoring_weights() -> Dict:
    """Load all scoring weights from JSON config."""
    return _load_json()


def get_weight(group: str, key: str, default: float = 0.0) -> float:
    """Get a single weight value by group and key.

    Parameters
    ----------
    group : str
        Weight group (e.g., "fusion", "clinical_sub", "pathway").
    key : str
        Weight key within the group.
    default : float
        Default value if not found.

    Returns
    -------
    float
    """
    w = _load_json()
    group_val = w.get(group)
    if group_val is None:
        return default
    # If group is a scalar (not a dict), return it directly
    if not isinstance(group_val, dict):
        return float(group_val)
    return group_val.get(key, default)


def get_sub_weights(group: str) -> Dict[str, float]:
    """Get all weights in a group as a dict.

    Parameters
    ----------
    group : str
        Weight group name.

    Returns
    -------
    dict
        {key: weight} mapping.
    """
    w = _load_json()
    val = w.get(group)
    if val is None:
        return {}
    # If group is a scalar, wrap it in a dict
    if not isinstance(val, dict):
        return {"value": float(val)}
    return val.copy()


def get_pathway_weights() -> Dict[str, Dict[str, float]]:
    """Get pathway weights (proliferation, immune, mitochondria).

    Returns
    -------
    dict
        {pathway_name: {gene: weight}} mapping.
    """
    w = _load_json()
    return w.get("pathway", _DEFAULTS["pathway"]).copy()


def get_fusion_weights() -> Dict[str, float]:
    """Get fusion modality weights."""
    return get_sub_weights("fusion")


def get_thresholds() -> Dict[str, float]:
    """Get Go/Conditional/No-Go thresholds."""
    w = _load_json()
    return {
        "go": w.get("go_threshold", 0.65),
        "conditional": w.get("conditional_threshold", 0.40),
        "safety_floor": w.get("safety_floor", 0.30),
    }


def reload():
    """Force reload weights from JSON (useful after editing the config)."""
    global _cached_weights
    _cached_weights = None
