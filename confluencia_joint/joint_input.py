"""
Unified input format for joint drug-epitope-PK evaluation.

Converts between the joint schema and the separate drug/epitope DataFrames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Amino-acid validation
# ---------------------------------------------------------------------------
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Supported HLA alleles (from confluencia-2.0-epitope/core/mhc_features.py)
SUPPORTED_ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:03", "HLA-A*02:06",
    "HLA-A*03:01", "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:02",
    "HLA-A*26:01", "HLA-A*30:02", "HLA-A*31:01", "HLA-A*33:01",
    "HLA-A*68:01", "HLA-A*68:02",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*15:01", "HLA-B*27:02",
    "HLA-B*27:05", "HLA-B*35:01", "HLA-B*39:01", "HLA-B*40:01",
    "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01", "HLA-B*46:01",
    "HLA-B*48:01", "HLA-B*51:01", "HLA-B*53:01", "HLA-B*57:01",
    "HLA-B*58:01",
    "HLA-C*01:02", "HLA-C*02:02", "HLA-C*03:03", "HLA-C*04:01",
    "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02",
    "HLA-C*08:02", "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02",
]


def _validate_epitope(seq: str) -> bool:
    """Return True if sequence contains only standard amino acids."""
    return bool(seq) and all(c in _VALID_AA for c in seq.upper())


def _validate_allele(allele: str) -> bool:
    """Return True if allele is in the supported list."""
    return allele in SUPPORTED_ALLELES


def _normalize_allele(allele: str) -> str:
    """Try to normalize an allele string to the supported format."""
    # Already exact match
    if allele in SUPPORTED_ALLELES:
        return allele
    # Try uppercase
    upper = allele.upper().strip()
    if upper in SUPPORTED_ALLELES:
        return upper
    # Try adding HLA- prefix
    if not upper.startswith("HLA"):
        candidate = "HLA-" + upper
        if candidate in SUPPORTED_ALLELES:
            return candidate
    return allele  # return as-is; scoring will handle unknown


# ---------------------------------------------------------------------------
# JointInput dataclass
# ---------------------------------------------------------------------------

@dataclass
class JointInput:
    """Unified input for joint drug-epitope-PK evaluation.

    Attributes
    ----------
    smiles : str
        SMILES molecular string for the drug.
    epitope_seq : str
        Peptide amino-acid sequence (e.g. "SLYNTVATL").
    mhc_allele : str
        MHC allele (e.g. "HLA-A*02:01").
    dose_mg : float
        Drug dose in mg.
    freq_per_day : float
        Administrations per day.
    treatment_time : float
        Treatment duration in hours.
    circ_expr : float
        circRNA expression level (optional, default 0).
    ifn_score : float
        IFN response score (optional, default 0).
    group_id : str
        Group identifier for batch tracking.
    """

    smiles: str
    epitope_seq: str
    mhc_allele: str
    dose_mg: float
    freq_per_day: float
    treatment_time: float
    circ_expr: float = 0.0
    ifn_score: float = 0.0
    group_id: str = "G0"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors: List[str] = []
        if not self.smiles or not self.smiles.strip():
            errors.append("SMILES is empty")
        if not _validate_epitope(self.epitope_seq):
            bad = [c for c in self.epitope_seq.upper() if c not in _VALID_AA]
            errors.append(f"Epitope contains invalid amino acids: {bad}")
        if not _validate_allele(self.mhc_allele):
            errors.append(
                f"MHC allele '{self.mhc_allele}' not in supported list "
                f"({len(SUPPORTED_ALLELES)} alleles)"
            )
        if self.dose_mg <= 0:
            errors.append("dose_mg must be > 0")
        if self.freq_per_day <= 0:
            errors.append("freq_per_day must be > 0")
        if self.treatment_time <= 0:
            errors.append("treatment_time must be > 0")
        return errors

    def is_valid(self) -> bool:
        """Return True if all fields pass validation."""
        return len(self.validate()) == 0

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------
    def to_drug_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame compatible with drug pipeline.

        Columns: smiles, epitope_seq, dose, freq, treatment_time, group_id
        """
        return pd.DataFrame([{
            "smiles": self.smiles,
            "epitope_seq": self.epitope_seq,
            "dose": self.dose_mg,
            "freq": self.freq_per_day,
            "treatment_time": self.treatment_time,
            "group_id": self.group_id,
        }])

    def to_epitope_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame compatible with epitope pipeline.

        Columns: epitope_seq, mhc_allele, dose, freq, treatment_time,
                 circ_expr, ifn_score
        """
        return pd.DataFrame([{
            "epitope_seq": self.epitope_seq,
            "mhc_allele": _normalize_allele(self.mhc_allele),
            "dose": self.dose_mg,
            "freq": self.freq_per_day,
            "treatment_time": self.treatment_time,
            "circ_expr": self.circ_expr,
            "ifn_score": self.ifn_score,
        }])

    # ------------------------------------------------------------------
    # Batch factory
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["JointInput"]:
        """Parse a unified CSV into a list of JointInput instances.

        Required columns: smiles, epitope_seq, mhc_allele, dose, freq,
                          treatment_time
        Optional columns: circ_expr, ifn_score, group_id
        """
        # Column aliases
        col_map = {
            "dose_mg": "dose", "dosage": "dose",
            "frequency": "freq", "dose_freq": "freq",
            "epitope": "epitope_seq", "sequence": "epitope_seq",
            "peptide_seq": "epitope_seq", "peptide": "epitope_seq",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        required = ["smiles", "epitope_seq", "mhc_allele", "dose", "freq", "treatment_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        inputs = []
        for _, row in df.iterrows():
            inputs.append(cls(
                smiles=str(row["smiles"]),
                epitope_seq=str(row["epitope_seq"]),
                mhc_allele=_normalize_allele(str(row["mhc_allele"])),
                dose_mg=float(row.get("dose", 0)),
                freq_per_day=float(row.get("freq", 0)),
                treatment_time=float(row.get("treatment_time", 0)),
                circ_expr=float(row.get("circ_expr", 0)),
                ifn_score=float(row.get("ifn_score", 0)),
                group_id=str(row.get("group_id", "G0")),
            ))
        return inputs
