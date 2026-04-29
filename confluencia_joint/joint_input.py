"""
Unified input format for joint drug-epitope-PK evaluation.

Converts between the joint schema and the separate drug/epitope DataFrames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------
_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _validate_smiles(smiles: str) -> bool:
    """Validate SMILES string chemically.

    Uses RDKit if available, otherwise does basic format check.
    """
    if not smiles or not smiles.strip():
        return False
    if _RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    # Fallback: non-empty, printable chars, no obvious nonsense
    return len(smiles.strip()) >= 2


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
    """Unified input for joint drug-epitope-PK-circRNA evaluation.

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
    gene_signature_outputs : dict or None
        Five-target gene signature outputs (pre-computed). If None but
        trop2/nectin4/liv1/b7h4/tmem65 are provided, will be auto-computed.
    trop2 : float
        TROP2 expression level (0-1 normalized). Optional.
    nectin4 : float
        NECTIN4 expression level (0-1 normalized). Optional.
    liv1 : float
        LIV-1 expression level (0-1 normalized). Optional.
    b7h4 : float
        B7-H4 expression level (0-1 normalized). Optional.
    tmem65 : float
        TMEM65 expression level (0-1 normalized). Optional.
    circ_sequence : str or None
        circRNA nucleotide sequence for innate immune sensing.
    circ_expression_matrix : pd.DataFrame or None
        Gene expression matrix for immune cycle / TME / evasion analysis.
    mutation_data : pd.DataFrame or None
        Mutation data for genomic features (TMB / DDR).
    cnv_data : pd.DataFrame or None
        CNV data for genomic features.
    survival_data : dict or None
        Survival data for risk scoring (keys: features, time, event).
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
    gene_signature_outputs: Optional[Dict[str, float]] = None
    trop2: float = 0.5
    nectin4: float = 0.5
    liv1: float = 0.5
    b7h4: float = 0.5
    tmem65: float = 0.5
    circ_sequence: Optional[str] = None
    circ_expression_matrix: Optional[pd.DataFrame] = None
    mutation_data: Optional[pd.DataFrame] = None
    cnv_data: Optional[pd.DataFrame] = None
    survival_data: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors: List[str] = []
        if not _validate_smiles(self.smiles):
            errors.append(
                "SMILES is empty or chemically invalid"
                + ("" if _RDKIT_AVAILABLE else " (RDKit not available, only basic format check)")
            )
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

    def to_gene_signature_dict(self) -> Dict[str, float]:
        """Get gene expression values as a dict for scoring.

        Returns dict: {trop2, nectin4, liv1, b7h4, tmem65}
        """
        return {
            "TROP2": self.trop2,
            "NECTIN4": self.nectin4,
            "LIV-1": self.liv1,
            "B7-H4": self.b7h4,
            "TMEM65": self.tmem65,
        }

    # ------------------------------------------------------------------
    # Batch factory
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["JointInput"]:
        """Parse a unified CSV into a list of JointInput instances.

        Required columns: smiles, epitope_seq, mhc_allele, dose, freq,
                          treatment_time
        Optional columns: circ_expr, ifn_score, group_id, trop2, nectin4,
                         liv1, b7h4, tmem65, gene_signature_outputs,
                         circ_sequence, circ_expr, ifn_score
        """
        # Column aliases
        col_map = {
            "dose_mg": "dose", "dosage": "dose",
            "frequency": "freq", "dose_freq": "freq",
            "epitope": "epitope_seq", "sequence": "epitope_seq",
            "peptide_seq": "epitope_seq", "peptide": "epitope_seq",
            "allele": "mhc_allele",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        required = ["smiles", "epitope_seq", "mhc_allele", "dose", "freq", "treatment_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Gene expression column aliases
        gene_map = {
            "TROP2": "trop2", "trop2_expr": "trop2",
            "NECTIN4": "nectin4", "nectin4_expr": "nectin4",
            "LIV_1": "liv1", "liv1_expr": "liv1", "LIV-1": "liv1",
            "B7_H4": "b7h4", "b7h4_expr": "b7h4", "B7-H4": "b7h4",
            "TMEM65": "tmem65", "tmem65_expr": "tmem65",
        }
        df = df.rename(columns={k: v for k, v in gene_map.items() if k in df.columns})

        inputs = []
        for _, row in df.iterrows():
            # Parse gene_signature_outputs if present (JSON string)
            gs_out = None
            gs_raw = row.get("gene_signature_outputs", None)
            if gs_raw is not None and isinstance(gs_raw, str) and gs_raw.strip():
                import json as _json
                try:
                    gs_out = _json.loads(gs_raw)
                except Exception:
                    pass

            # Parse circ_sequence if present
            circ_seq = row.get("circ_sequence", None)
            if circ_seq is not None and not isinstance(circ_seq, str):
                circ_seq = str(circ_seq)

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
                gene_signature_outputs=gs_out,
                trop2=float(row.get("trop2", 0.5)),
                nectin4=float(row.get("nectin4", 0.5)),
                liv1=float(row.get("liv1", 0.5)),
                b7h4=float(row.get("b7h4", 0.5)),
                tmem65=float(row.get("tmem65", 0.5)),
                circ_sequence=circ_seq,
            ))
        return inputs
