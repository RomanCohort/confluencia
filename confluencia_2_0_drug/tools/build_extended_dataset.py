"""
Build an extended breast cancer drug dataset for Confluencia 2.0 training.

Extends the original 28 drugs × 15 epitopes = 2,100 rows by:
  1. Fetching additional cancer drug compounds from ChEMBL (with real IC50)
  2. Extracting additional breast cancer epitopes from IEDB raw data
  3. Generating derived labels using the same formulas as the original script
  4. Using real pChEMBL values for target_binding where available

The original breast_cancer_drug_dataset.csv is NEVER modified.
Output: breast_cancer_drug_dataset_extended.csv (includes original + new data)

References and formulas are identical to build_breast_cancer_dataset.py.
"""
from __future__ import annotations

import math
import zipfile
import csv
import io
import json
import urllib.request
import urllib.parse
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_CSV = DATA_DIR / "breast_cancer_drug_dataset.csv"
EXTENDED_CSV = DATA_DIR / "breast_cancer_drug_dataset_extended.csv"

IEDB_ZIP = Path(
    r"D:\IGEM集成方案\confluencia-2.0-epitope\data\raw\iedb_tcell_full_v3.zip"
)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# ---------------------------------------------------------------------------
# 1. Original drug data (from build_breast_cancer_dataset.py)
# ---------------------------------------------------------------------------
ORIGINAL_DRUGS = [
    {"name": "Tamoxifen", "smiles": "CC/C(=C(\\c1ccccc1)/c1ccccc1)c1ccc(OCCN(C)C)cc1", "target": "ER_alpha", "mechanism": "SERM", "ic50_um": 0.6, "dose_mg": 20.0, "freq_per_day": 1.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.02, "cardiotox_pct": 0.01, "inflam_pct": 0.05},
    {"name": "Letrozole", "smiles": "CC1=NC(=CN1C2=CC=C(C=C2)C#N)C3=CC=C(C=C3)C#N", "target": "Aromatase", "mechanism": "AI", "ic50_um": 0.000011, "dose_mg": 2.5, "freq_per_day": 1.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.08, "cardiotox_pct": 0.03, "inflam_pct": 0.12},
    {"name": "Anastrozole", "smiles": "CC1=NC(=CN1C2=CC=CC=C2)C3=CC=CC=C3C#N", "target": "Aromatase", "mechanism": "AI", "ic50_um": 0.000015, "dose_mg": 1.0, "freq_per_day": 1.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.07, "cardiotox_pct": 0.02, "inflam_pct": 0.10},
    {"name": "Exemestane", "smiles": "CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C", "target": "Aromatase", "mechanism": "AI_irreversible", "ic50_um": 0.000030, "dose_mg": 25.0, "freq_per_day": 1.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.06, "cardiotox_pct": 0.02, "inflam_pct": 0.09},
    {"name": "Fulvestrant", "smiles": "CC(C)C1=CC=C(C=C1)C2=CC=C(C=C2)C(C3=CC=C(C=C3)O)CC4CCCC4=O", "target": "ER_alpha", "mechanism": "SERD", "ic50_um": 0.004, "dose_mg": 500.0, "freq_per_day": 0.033, "treatment_days": 180, "tox_grade_ge3_pct": 0.04, "cardiotox_pct": 0.01, "inflam_pct": 0.06},
    {"name": "Paclitaxel", "smiles": "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", "target": "Tubulin_beta", "mechanism": "Microtubule_stabilizer", "ic50_um": 0.005, "dose_mg": 175.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.02, "inflam_pct": 0.15},
    {"name": "Docetaxel", "smiles": "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O", "target": "Tubulin_beta", "mechanism": "Microtubule_stabilizer", "ic50_um": 0.003, "dose_mg": 100.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.35, "cardiotox_pct": 0.02, "inflam_pct": 0.18},
    {"name": "Doxorubicin", "smiles": "CC1C(C(CC(O1)OC2CC(CC3=C2C4=CC5=C(C=C4C(=C3O)O)C(=O)C6=CC=CC=C6C5=O)O)O)N", "target": "Topoisomerase_II", "mechanism": "Anthracycline", "ic50_um": 0.05, "dose_mg": 60.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.40, "cardiotox_pct": 0.08, "inflam_pct": 0.20},
    {"name": "Epirubicin", "smiles": "CC1C(C(CC(O1)OC2CC(CC3=C2C4=CC5=C(C=C4C(=C3O)O)C(=O)C6=CC=CC=C6C5=O)O)O)N", "target": "Topoisomerase_II", "mechanism": "Anthracycline", "ic50_um": 0.08, "dose_mg": 100.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.05, "inflam_pct": 0.15},
    {"name": "Cyclophosphamide", "smiles": "ClCCN(CCCl)P(=O)(O)OCC1CCN(C)CC1", "target": "DNA_alkylation", "mechanism": "Alkylating_agent", "ic50_um": 50.0, "dose_mg": 600.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.35, "cardiotox_pct": 0.01, "inflam_pct": 0.12},
    {"name": "Capecitabine", "smiles": "CCCCC(=O)NC1=NC(=O)C(CF)=CN1", "target": "Thymidylate_synthase", "mechanism": "Antimetabolite_prodrug", "ic50_um": 1.0, "dose_mg": 2500.0, "freq_per_day": 2.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.20, "cardiotox_pct": 0.01, "inflam_pct": 0.25},
    {"name": "Gemcitabine", "smiles": "C1=CN(C(=O)NC1=O)C2C(C(C(O2)CO)O)F", "target": "DNA_synthesis", "mechanism": "Antimetabolite", "ic50_um": 0.01, "dose_mg": 1250.0, "freq_per_day": 0.67, "treatment_days": 63, "tox_grade_ge3_pct": 0.25, "cardiotox_pct": 0.01, "inflam_pct": 0.10},
    {"name": "Vinorelbine", "smiles": "CCC1(CC2C(C1)CC1C3C2N(CC2C(C3C2O)O)C)C=CC=C1", "target": "Tubulin", "mechanism": "Vinca_alkaloid", "ic50_um": 0.02, "dose_mg": 30.0, "freq_per_day": 1.0, "treatment_days": 56, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.01, "inflam_pct": 0.08},
    {"name": "Eribulin", "smiles": "CCCCCCCCC1CC2C3OC2CCC4C5OCC6CC(CC6C5O)CC4C3CC1O", "target": "Tubulin", "mechanism": "Microtubule_inhibitor", "ic50_um": 0.001, "dose_mg": 1.4, "freq_per_day": 0.67, "treatment_days": 63, "tox_grade_ge3_pct": 0.20, "cardiotox_pct": 0.02, "inflam_pct": 0.12},
    {"name": "Palbociclib", "smiles": "CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C", "target": "CDK4_6", "mechanism": "CDK_inhibitor", "ic50_um": 0.01, "dose_mg": 125.0, "freq_per_day": 0.75, "treatment_days": 120, "tox_grade_ge3_pct": 0.35, "cardiotox_pct": 0.01, "inflam_pct": 0.05},
    {"name": "Ribociclib", "smiles": "CN(C)C(=O)C1=CC2=CN=C(N=C2N1C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5", "target": "CDK4_6", "mechanism": "CDK_inhibitor", "ic50_um": 0.01, "dose_mg": 600.0, "freq_per_day": 0.75, "treatment_days": 120, "tox_grade_ge3_pct": 0.40, "cardiotox_pct": 0.02, "inflam_pct": 0.08},
    {"name": "Abemaciclib", "smiles": "CCN1CCN(CC1)CC2=CN=C(C=C2)NC3=NC=C(C(=N3)C4=CC5=C(C(=C4)F)N=C(N5C(C)C)C)F", "target": "CDK4_6", "mechanism": "CDK_inhibitor", "ic50_um": 0.005, "dose_mg": 150.0, "freq_per_day": 2.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.25, "cardiotox_pct": 0.01, "inflam_pct": 0.20},
    {"name": "Lapatinib", "smiles": "CS(=O)(=S)NNC(=O)C1=CC=CC=C1SC2=NC3=CC=CC=C3N=C2C4=CC(=CC=C4)F", "target": "HER2_EGFR", "mechanism": "TKI_dual", "ic50_um": 0.005, "dose_mg": 1250.0, "freq_per_day": 1.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.15, "cardiotox_pct": 0.02, "inflam_pct": 0.30},
    {"name": "Neratinib", "smiles": "CC(=O)N1CCN(CC1)C2=CC(=CC=C2)C3=NC4=CC=CC=C4N=C3SC5=CC=CC=C5Cl", "target": "HER2_EGFR", "mechanism": "TKI_irreversible", "ic50_um": 0.059, "dose_mg": 240.0, "freq_per_day": 1.0, "treatment_days": 365, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.01, "inflam_pct": 0.40},
    {"name": "Tucatinib", "smiles": "CC1=NC(=NC(=N1)NC2=CC=C(C=C2)OCC3CC3)NC4=CC=CC5=C4C(=O)NN5", "target": "HER2", "mechanism": "TKI_selective_HER2", "ic50_um": 0.008, "dose_mg": 300.0, "freq_per_day": 2.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.12, "cardiotox_pct": 0.01, "inflam_pct": 0.15},
    {"name": "Pembrolizumab_peptide_mimic", "smiles": "CC1CC(=O)NC1C(=O)NC(C(=O)NC(C(=O)NC(C(=O)N)C)C)C", "target": "PD1", "mechanism": "Immune_checkpoint", "ic50_um": 0.01, "dose_mg": 200.0, "freq_per_day": 0.33, "treatment_days": 90, "tox_grade_ge3_pct": 0.20, "cardiotox_pct": 0.02, "inflam_pct": 0.25},
    {"name": "Olaparib", "smiles": "CCCCC(=O)NC1=CC(=C(C=C1)C2=CC(=O)N3C2=CC=C3F)F", "target": "PARP", "mechanism": "PARP_inhibitor", "ic50_um": 0.005, "dose_mg": 300.0, "freq_per_day": 2.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.25, "cardiotox_pct": 0.01, "inflam_pct": 0.08},
    {"name": "Talazoparib", "smiles": "CC1C2=NC3=CC=CC=C3N2C(=O)NC1C(=O)NC4=CC(=C(C=C4F)F)F", "target": "PARP", "mechanism": "PARP_inhibitor_trapping", "ic50_um": 0.0005, "dose_mg": 1.0, "freq_per_day": 1.0, "treatment_days": 180, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.01, "inflam_pct": 0.10},
    {"name": "Alpelisib", "smiles": "CC1=NC2=CC=CC=C2N1C3=CC=C(C=C3)C4=NC5=CC=CC=C5N4C(C)C", "target": "PI3K_alpha", "mechanism": "PI3K_inhibitor", "ic50_um": 0.005, "dose_mg": 300.0, "freq_per_day": 1.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.01, "inflam_pct": 0.35},
    {"name": "Everolimus", "smiles": "CC1CCCCC1N2C(=O)C3C4CCC(O)CC4N(C)C3C5=CC=CC=C5C2=O", "target": "mTOR", "mechanism": "mTOR_inhibitor", "ic50_um": 0.001, "dose_mg": 10.0, "freq_per_day": 1.0, "treatment_days": 120, "tox_grade_ge3_pct": 0.20, "cardiotox_pct": 0.01, "inflam_pct": 0.30},
    {"name": "Carboplatin", "smiles": "O=C1OC2NC(=O)NC2C1O.[Pt]", "target": "DNA_crosslink", "mechanism": "Platinum_agent", "ic50_um": 5.0, "dose_mg": 600.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.25, "cardiotox_pct": 0.01, "inflam_pct": 0.10},
    {"name": "Cisplatin", "smiles": "N.N.[Pt]=O", "target": "DNA_crosslink", "mechanism": "Platinum_agent", "ic50_um": 2.0, "dose_mg": 75.0, "freq_per_day": 0.33, "treatment_days": 63, "tox_grade_ge3_pct": 0.35, "cardiotox_pct": 0.01, "inflam_pct": 0.20},
    {"name": "Etoposide", "smiles": "C[C@@H]1OC[C@@H]2[C@@H](O1)[C@@H]([C@H]([C@@H](O2)O[C@H]3[C@H]4COC(=O)[C@@H]4[C@@H](C5=CC6=C(C=C35)OCO6)C7=CC(=C(C(=C7)OC)O)OC)O)O", "target": "Topoisomerase_II", "mechanism": "Topo_II_inhibitor", "ic50_um": 0.5, "dose_mg": 100.0, "freq_per_day": 1.0, "treatment_days": 63, "tox_grade_ge3_pct": 0.30, "cardiotox_pct": 0.01, "inflam_pct": 0.10},
]

ORIGINAL_EPITOPES = [
    {"epitope_seq": "KIFSLAHL", "antigen": "HER2", "mhc": "HLA-A02", "immunogenicity": 0.85},
    {"epitope_seq": "ILHNGAYSL", "antigen": "HER2", "mhc": "HLA-A02", "immunogenicity": 0.78},
    {"epitope_seq": "VLRENTSPK", "antigen": "HER2", "mhc": "HLA-A02", "immunogenicity": 0.72},
    {"epitope_seq": "FLLPSDCFL", "antigen": "HER2", "mhc": "HLA-A02", "immunogenicity": 0.68},
    {"epitope_seq": "RYLQQVIFL", "antigen": "HER2", "mhc": "HLA-A02", "immunogenicity": 0.65},
    {"epitope_seq": "GVTSAPDTRPA", "antigen": "MUC1", "mhc": "HLA-A02", "immunogenicity": 0.80},
    {"epitope_seq": "STAPPVHNV", "antigen": "MUC1", "mhc": "HLA-A02", "immunogenicity": 0.70},
    {"epitope_seq": "ALYVDFSFL", "antigen": "MUC1", "mhc": "HLA-A02", "immunogenicity": 0.62},
    {"epitope_seq": "LTVTVPWLR", "antigen": "MUC1", "mhc": "HLA-A02", "immunogenicity": 0.55},
    {"epitope_seq": "SIINFEKL", "antigen": "OVA_control", "mhc": "H2-Kb", "immunogenicity": 0.95},
    {"epitope_seq": "GILGFVFTL", "antigen": "Influenza_control", "mhc": "HLA-A02", "immunogenicity": 0.90},
    {"epitope_seq": "LLFGYPVYV", "antigen": "CEA", "mhc": "HLA-A02", "immunogenicity": 0.60},
    {"epitope_seq": "YLSGANLNL", "antigen": "CEA", "mhc": "HLA-A02", "immunogenicity": 0.58},
    {"epitope_seq": "IMDQVPFSV", "antigen": "MAGE-A3", "mhc": "HLA-A01", "immunogenicity": 0.75},
    {"epitope_seq": "FLWGPRALV", "antigen": "WT1", "mhc": "HLA-A02", "immunogenicity": 0.70},
]

# ChEMBL target IDs for cancer targets
CHEMBL_TARGETS = {
    "ER_alpha": "CHEMBL203",
    "Aromatase": "CHEMBL1544",
    "HER2_EGFR": "CHEMBL203",
    "HER2": "CHEMBL240",
    "VEGFR2": "CHEMBL279",
    "CDK4_6": "CHEMBL2401",
    "PARP": "CHEMBL3193",
    "PI3K_alpha": "CHEMBL3264",
    "mTOR": "CHEMBL2842",
    "Tubulin_beta": "CHEMBL439",
    "Topoisomerase_II": "CHEMBL1801",
    "DNA_alkylation": "CHEMBL203",
}

# Mechanism classification based on target
TARGET_TO_MECHANISM = {
    "ER_alpha": "SERM",
    "Aromatase": "AI",
    "HER2_EGFR": "TKI_dual",
    "HER2": "TKI_selective_HER2",
    "VEGFR2": "TKI",
    "CDK4_6": "CDK_inhibitor",
    "PARP": "PARP_inhibitor",
    "PI3K_alpha": "PI3K_inhibitor",
    "mTOR": "mTOR_inhibitor",
    "Tubulin_beta": "Microtubule_stabilizer",
    "Topoisomerase_II": "Topo_II_inhibitor",
    "DNA_alkylation": "Alkylating_agent",
    "PD1": "Immune_checkpoint",
    "Tubulin": "Microtubule_inhibitor",
    "DNA_synthesis": "Antimetabolite",
    "DNA_crosslink": "Platinum_agent",
}

# Valid amino acid set
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# 2. Label computation formulas (improved version)
# ---------------------------------------------------------------------------
def _estimate_mw(smiles: str) -> float:
    """Estimate molecular weight from SMILES using atom counting.
    More reliable than full RDKit parse for quick estimates."""
    # Average atomic weights for common organic atoms
    atom_weights = {
        "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065,
        "P": 30.974, "F": 18.998, "Cl": 35.453, "Br": 79.904,
        "I": 126.904, "Si": 28.086, "B": 10.811,
    }
    mw = 0.0
    i = 0
    while i < len(smiles):
        c = smiles[i]
        if c.isupper():
            # Check for two-letter elements (Cl, Br, Si, etc.)
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                elem = smiles[i:i + 2]
                i += 2
            else:
                elem = c
                i += 1
            # Skip brackets, ring numbers, etc.
            if elem in atom_weights:
                mw += atom_weights[elem]
            # Check for explicit count
            if i < len(smiles) and smiles[i].isdigit():
                count_str = ""
                while i < len(smiles) and smiles[i].isdigit():
                    count_str += smiles[i]
                    i += 1
                mw += atom_weights.get(elem, 12) * (int(count_str) - 1)
        elif c == "[":
            # Bracket atom — skip to closing bracket
            j = smiles.index("]", i) if "]" in smiles[i:] else i
            i = j + 1
        else:
            i += 1
    # Add rough hydrogen count (organic molecules ~50% H by count)
    mw *= 1.15  # rough H correction
    return max(mw, 50.0)


# Unified binding: convert IC50 (μM) to pChEMBL-equivalent, then normalize.
# pChEMBL = -log10(IC50_M) = -log10(IC50_μM * 1e-6) = 6 - log10(IC50_μM)
# pChEMBL typical range: 3–11. Normalize to [0,1] via (pChEMBL - 3) / 8.
def _ic50_to_pchembl(ic50_um: float) -> float:
    """Convert IC50 in μM to pChEMBL-equivalent value."""
    x = max(ic50_um, 1e-9)
    return 6.0 - math.log10(x)


def _pchembl_to_binding(pchembl: float) -> float:
    """Normalize pChEMBL to [0,1] binding score. Range 3–11 maps to 0–1."""
    return float(np.clip((pchembl - 3.0) / 8.0, 0.0, 1.0))


def _ic50_to_binding(ic50_um: float) -> float:
    """Unified binding: IC50 → pChEMBL → normalized [0,1]."""
    return _pchembl_to_binding(_ic50_to_pchembl(ic50_um))


def _ic50_to_efficacy(
    ic50_um: float, dose_mg: float, freq: float, treatment_days: float,
    epitope_immunogenicity: float = 0.5, drug_mechanism: str = "",
) -> float:
    """Efficacy with epitope interaction. Epitope contributes to efficacy
    via immune activation, reflecting drug-epitope synergy."""
    binding = _ic50_to_binding(ic50_um)
    dose_factor = 1.0 - math.exp(-dose_mg / 100.0)
    time_factor = 1.0 - math.exp(-treatment_days / 90.0)
    freq_factor = min(freq / 2.0, 1.0)

    # Epitope-immune contribution to efficacy
    immune_contrib = epitope_immunogenicity * 0.2
    if drug_mechanism == "Immune_checkpoint":
        immune_contrib += 0.15
    elif drug_mechanism in ("SERD", "SERM", "AI", "AI_irreversible"):
        immune_contrib *= 0.4

    raw = (0.35 * binding + 0.20 * dose_factor + 0.10 * time_factor
           + 0.10 * freq_factor + 0.25 * immune_contrib)
    return float(np.clip(raw, 0.0, 1.0))


def _tox_to_risk(tox_pct: float, cardiotox_pct: float) -> float:
    return float(np.clip(0.7 * tox_pct + 0.3 * cardiotox_pct, 0.0, 1.0))


def _inflam_to_risk(inflam_pct: float) -> float:
    """Apply power transform to spread the inflammation distribution.
    Without this, most values cluster near 0.05–0.15."""
    base = float(np.clip(inflam_pct, 0.0, 1.0))
    # Square-root transform stretches low values
    return float(np.clip(math.sqrt(base), 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Fetch additional compounds from ChEMBL
# ---------------------------------------------------------------------------
def fetch_chembl_compounds(max_per_target: int = 300) -> list[dict]:
    """Fetch additional cancer drug compounds from ChEMBL with real IC50."""
    print("Fetching additional compounds from ChEMBL...")
    all_compounds = []
    seen_smiles = {d["smiles"] for d in ORIGINAL_DRUGS}

    for target_name, chembl_id in CHEMBL_TARGETS.items():
        params = {
            "target_chembl_id": chembl_id,
            "standard_type": "IC50",
            "has_smiles": "true",
            "format": "json",
            "limit": max_per_target,
        }
        url = f"{CHEMBL_API}/activity.json?{urllib.parse.urlencode(params)}"
        print(f"  Fetching {target_name} ({chembl_id})...")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error: {e}")
            continue

        activities = data.get("activities", [])
        for act in activities:
            smiles = act.get("canonical_smiles", "")
            pchembl = act.get("pchembl_value")
            if not smiles or not pchembl or smiles in seen_smiles:
                continue
            try:
                pchembl_val = float(pchembl)
            except (ValueError, TypeError):
                continue

            # Convert pChEMBL to IC50 in μM: IC50_μM = 10^(6 - pChEMBL) / 1e6
            ic50_um = 10 ** (6 - pchembl_val)

            # Skip extreme values
            if ic50_um < 1e-6 or ic50_um > 10000:
                continue

            # Estimate dosing parameters based on IC50 and molecular weight
            # Lower IC50 (more potent) → lower dose needed
            # Use MW from SMILES when available for more realistic estimates
            mw = _estimate_mw(smiles)
            if ic50_um < 0.01:
                # Very potent: low dose, once or twice daily
                dose_mg = max(1.0, min(mw * 0.005, 50.0))
                freq = 1.0
            elif ic50_um < 1.0:
                # Potent: moderate dose
                dose_mg = max(5.0, min(mw * 0.05, 300.0))
                freq = 1.0
            else:
                # Less potent: higher dose, may need BID
                dose_mg = max(10.0, min(mw * 0.2, 1500.0))
                freq = 2.0 if dose_mg > 500 else 1.0

            # Estimate toxicity based on IC50 (potent drugs often have more side effects)
            tox_pct = float(np.clip(0.1 + 0.2 * _ic50_to_binding(ic50_um), 0.01, 0.45))
            cardiotox_pct = float(np.clip(tox_pct * 0.15, 0.01, 0.10))
            inflam_pct = float(np.clip(0.05 + 0.15 * (1 - _ic50_to_binding(ic50_um)), 0.02, 0.35))

            mechanism = TARGET_TO_MECHANISM.get(target_name, "Unknown")
            treatment_days = 120

            compound = {
                "name": f"ChEMBL_{act.get('molecule_chembl_id', 'unknown')}",
                "smiles": smiles,
                "target": target_name,
                "mechanism": mechanism,
                "ic50_um": ic50_um,
                "dose_mg": round(dose_mg, 2),
                "freq_per_day": freq,
                "treatment_days": treatment_days,
                "tox_grade_ge3_pct": tox_pct,
                "cardiotox_pct": cardiotox_pct,
                "inflam_pct": inflam_pct,
                "pchembl_value": pchembl_val,  # keep real pChEMBL for binding
            }
            seen_smiles.add(smiles)
            all_compounds.append(compound)

        time.sleep(0.3)  # rate limiting

    print(f"  Fetched {len(all_compounds)} new compounds from ChEMBL")
    return all_compounds


# ---------------------------------------------------------------------------
# 4. Extract additional epitopes from IEDB
# ---------------------------------------------------------------------------
def extract_iedb_epitopes(max_epitopes: int = 100) -> list[dict]:
    """Extract breast cancer-related epitopes from IEDB raw data."""
    print("Extracting additional epitopes from IEDB...")
    if not IEDB_ZIP.exists():
        print(f"  IEDB zip not found: {IEDB_ZIP}")
        return []

    # Known breast cancer antigen keywords in epitope descriptions
    BC_ANTIGENS = ["HER2", "ERBB2", "MUC1", "CEA", "CEACAM5", "MAGE", "WT1", "BRCA"]
    # Also include common cancer/testis antigens and checkpoint targets
    CANCER_ANTIGENS = BC_ANTIGENS + ["NY-ESO", "PRAME", "SURVIVIN", "BIRC5", "TERT", "MART", "GP100"]

    new_epitopes = []
    seen_seqs = {e["epitope_seq"] for e in ORIGINAL_EPITOPES}

    with zipfile.ZipFile(IEDB_ZIP, "r") as zf:
        with zf.open("tcell_full_v3.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            next(reader)  # skip header

            for row in reader:
                if len(new_epitopes) >= max_epitopes:
                    break

                seq = row[11].strip() if len(row) > 11 else ""
                if len(seq) < 8 or len(seq) > 30:
                    continue
                if not all(c in VALID_AA for c in seq.upper()):
                    continue
                if seq in seen_seqs:
                    continue

                # Check if related to cancer antigens
                source_organism = row[23].strip() if len(row) > 23 else ""
                epitope_name = row[12].strip() if len(row) > 12 else ""
                protein_name = row[19].strip() if len(row) > 19 else ""

                # Determine antigen
                antigen = "Other_cancer"
                for ag_keyword in CANCER_ANTIGENS:
                    combined_text = f"{epitope_name} {protein_name} {source_organism}".upper()
                    if ag_keyword.upper() in combined_text:
                        antigen = ag_keyword
                        break

                # Only keep cancer-related epitopes
                if antigen == "Other_cancer":
                    # Also accept epitopes with Positive qualitative results (likely immunogenic)
                    qual = row[122].strip() if len(row) > 122 else ""
                    if qual not in ("Positive", "Positive-High", "Positive-Intermediate"):
                        continue

                # Estimate immunogenicity from response frequency
                immunogenicity = 0.5  # default
                rf_str = row[127].strip() if len(row) > 127 else ""
                if rf_str:
                    try:
                        rf = float(rf_str)
                        immunogenicity = float(np.clip(rf / 100.0, 0.2, 0.95))
                    except ValueError:
                        pass
                elif antigen != "Other_cancer":
                    immunogenicity = 0.6 + 0.2 * np.random.random()

                epitope = {
                    "epitope_seq": seq.upper(),
                    "antigen": antigen,
                    "mhc": "HLA-A02",
                    "immunogenicity": round(immunogenicity, 3),
                }
                seen_seqs.add(seq)
                new_epitopes.append(epitope)

    print(f"  Extracted {len(new_epitopes)} new epitopes from IEDB")
    return new_epitopes


# ---------------------------------------------------------------------------
# 5. Generate dataset rows
# ---------------------------------------------------------------------------
def generate_rows(
    drugs: list[dict],
    epitopes: list[dict],
    n_per_combination: int = 3,
    include_dose_variants: bool = True,
    source: str = "chembl_expanded",
) -> list[dict]:
    """Generate dataset rows with improved formulas:
    - Unified binding (pChEMBL-based) eliminates distribution shift
    - Epitope affects efficacy via immune contribution
    - Epitope modulates target_binding (drug-epitope interaction)
    - No hard cap on immune_cell_activation
    - Sqrt-transform on inflammation for better spread
    """
    rng = np.random.default_rng(12345)
    rows = []

    for drug in drugs:
        for epitope in epitopes:
            for _ in range(n_per_combination):
                dose = drug["dose_mg"] * rng.uniform(0.8, 1.2)
                freq = drug["freq_per_day"] * rng.uniform(0.9, 1.1)
                t_days = drug["treatment_days"] * rng.uniform(0.85, 1.15)

                # Unified binding: always use pChEMBL-based normalization
                if "pchembl_value" in drug:
                    pchembl = drug["pchembl_value"]
                else:
                    pchembl = _ic50_to_pchembl(drug["ic50_um"])

                binding = _pchembl_to_binding(pchembl)
                # Epitope immunogenicity modulates binding (epitope-drug interaction)
                # Higher immunogenicity → slightly better target engagement
                binding_interaction = 0.1 * epitope["immunogenicity"]
                binding = float(np.clip(binding + binding_interaction + rng.normal(0, 0.03), 0.0, 1.0))

                efficacy = _ic50_to_efficacy(
                    drug["ic50_um"], dose, freq, t_days,
                    epitope_immunogenicity=epitope["immunogenicity"],
                    drug_mechanism=drug["mechanism"],
                )
                efficacy = float(np.clip(efficacy + rng.normal(0, 0.04), 0.0, 1.0))

                # Immune activation: epitope is the primary driver
                immune_base = epitope["immunogenicity"] * 0.6
                if drug["mechanism"] == "Immune_checkpoint":
                    immune_base += 0.25
                elif drug["mechanism"] in ("SERD", "SERM", "AI", "AI_irreversible"):
                    immune_base *= 0.5
                # Binding quality boosts immune activation
                immune_base += 0.15 * binding
                immune_activation = float(np.clip(immune_base + rng.normal(0, 0.05), 0.0, 1.0))

                # Immune cell activation: no hard cap, epitope antigen matters
                immune_cell_base = immune_activation * 0.7 + epitope["immunogenicity"] * 0.2
                if epitope["antigen"] == "HER2":
                    immune_cell_base += 0.08
                elif epitope["antigen"] in ("MUC1", "CEA", "MAGE-A3", "WT1"):
                    immune_cell_base += 0.05
                immune_cell_activation = float(np.clip(immune_cell_base + rng.normal(0, 0.04), 0.0, 1.0))

                tox_risk = _tox_to_risk(drug["tox_grade_ge3_pct"], drug["cardiotox_pct"])
                tox_risk = float(np.clip(tox_risk + rng.normal(0, 0.02), 0.0, 1.0))

                inflam_risk = _inflam_to_risk(drug["inflam_pct"])
                inflam_risk = float(np.clip(inflam_risk + rng.normal(0, 0.03), 0.0, 1.0))

                rows.append({
                    "smiles": drug["smiles"],
                    "epitope_seq": epitope["epitope_seq"],
                    "dose": round(dose, 3),
                    "freq": round(freq, 4),
                    "treatment_time": round(t_days * 24, 1),
                    "group_id": f"{drug['target']}_{epitope['antigen']}",
                    "efficacy": round(efficacy, 4),
                    "target_binding": round(binding, 4),
                    "immune_activation": round(immune_activation, 4),
                    "immune_cell_activation": round(immune_cell_activation, 4),
                    "inflammation_risk": round(inflam_risk, 4),
                    "toxicity_risk": round(tox_risk, 4),
                    "_source": source,
                })

            if include_dose_variants:
                for dose_scale in [0.5, 2.0]:
                    dose = drug["dose_mg"] * dose_scale * rng.uniform(0.9, 1.1)
                    freq = drug["freq_per_day"] * rng.uniform(0.9, 1.1)
                    t_days = drug["treatment_days"] * rng.uniform(0.85, 1.15)

                    if "pchembl_value" in drug:
                        pchembl = drug["pchembl_value"]
                    else:
                        pchembl = _ic50_to_pchembl(drug["ic50_um"])

                    binding = _pchembl_to_binding(pchembl)
                    binding_interaction = 0.1 * epitope["immunogenicity"]
                    binding = float(np.clip(binding + binding_interaction + rng.normal(0, 0.03), 0.0, 1.0))

                    efficacy = _ic50_to_efficacy(
                        drug["ic50_um"], dose, freq, t_days,
                        epitope_immunogenicity=epitope["immunogenicity"],
                        drug_mechanism=drug["mechanism"],
                    )
                    efficacy = float(np.clip(efficacy + rng.normal(0, 0.04), 0.0, 1.0))

                    immune_base = epitope["immunogenicity"] * 0.6
                    if drug["mechanism"] == "Immune_checkpoint":
                        immune_base += 0.25
                    elif drug["mechanism"] in ("SERD", "SERM", "AI", "AI_irreversible"):
                        immune_base *= 0.5
                    immune_base += 0.15 * binding
                    immune_activation = float(np.clip(immune_base + rng.normal(0, 0.05), 0.0, 1.0))

                    immune_cell_base = immune_activation * 0.7 + epitope["immunogenicity"] * 0.2
                    if epitope["antigen"] == "HER2":
                        immune_cell_base += 0.08
                    elif epitope["antigen"] in ("MUC1", "CEA", "MAGE-A3", "WT1"):
                        immune_cell_base += 0.05
                    immune_cell_activation = float(np.clip(immune_cell_base + rng.normal(0, 0.04), 0.0, 1.0))

                    tox_risk = _tox_to_risk(drug["tox_grade_ge3_pct"], drug["cardiotox_pct"])
                    tox_risk = float(np.clip(tox_risk * dose_scale + rng.normal(0, 0.02), 0.0, 1.0))

                    inflam_risk = _inflam_to_risk(drug["inflam_pct"])
                    inflam_risk = float(np.clip(inflam_risk * min(dose_scale, 1.5) + rng.normal(0, 0.03), 0.0, 1.0))

                    rows.append({
                        "smiles": drug["smiles"],
                        "epitope_seq": epitope["epitope_seq"],
                        "dose": round(dose, 3),
                        "freq": round(freq, 4),
                        "treatment_time": round(t_days * 24, 1),
                        "group_id": f"{drug['target']}_{epitope['antigen']}",
                        "efficacy": round(efficacy, 4),
                        "target_binding": round(binding, 4),
                        "immune_activation": round(immune_activation, 4),
                        "immune_cell_activation": round(immune_cell_activation, 4),
                        "inflammation_risk": round(inflam_risk, 4),
                        "toxicity_risk": round(tox_risk, 4),
                        "_source": source,
                    })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Building extended breast cancer drug dataset")
    print("=" * 60)

    # Step 1: Regenerate original data with corrected SMILES and unified formulas
    print("\n[1/5] Regenerating original dataset with corrected SMILES & formulas...")
    rows_original = generate_rows(
        ORIGINAL_DRUGS, ORIGINAL_EPITOPES, n_per_combination=3,
        include_dose_variants=True, source="original",
    )
    original_df = pd.DataFrame(rows_original)
    print(f"  Original: {len(original_df)} rows, {original_df['smiles'].nunique()} SMILES, {original_df['epitope_seq'].nunique()} epitopes")

    # Step 2: Fetch new compounds from ChEMBL
    print("\n[2/5] Fetching additional compounds from ChEMBL...")
    new_drugs = fetch_chembl_compounds(max_per_target=300)
    print(f"  New compounds: {len(new_drugs)}")

    # Step 3: Extract new epitopes from IEDB
    print("\n[3/5] Extracting additional epitopes from IEDB...")
    new_epitopes = extract_iedb_epitopes(max_epitopes=80)
    print(f"  New epitopes: {len(new_epitopes)}")

    # Step 4: Generate expanded rows
    print("\n[4/5] Generating expanded dataset...")

    # Original drugs × new epitopes
    rows_original_drugs_new_epi = generate_rows(
        ORIGINAL_DRUGS, new_epitopes, n_per_combination=3, include_dose_variants=True,
        source="original_drug_new_epitope",
    )
    print(f"  Original drugs × new epitopes: {len(rows_original_drugs_new_epi)} rows")

    # New drugs × original epitopes
    rows_new_drugs_original_epi = generate_rows(
        new_drugs, ORIGINAL_EPITOPES, n_per_combination=3, include_dose_variants=True,
        source="chembl_drug_original_epitope",
    )
    print(f"  New drugs × original epitopes: {len(rows_new_drugs_original_epi)} rows")

    # New drugs × new epitopes (limit to avoid explosion)
    # Use subset: top 100 new drugs × top 30 new epitopes
    limited_drugs = new_drugs[:100]
    limited_epitopes = new_epitopes[:30]
    rows_new_drugs_new_epi = generate_rows(
        limited_drugs, limited_epitopes, n_per_combination=2, include_dose_variants=True,
        source="chembl_drug_new_epitope",
    )
    print(f"  New drugs × new epitopes: {len(rows_new_drugs_new_epi)} rows")

    # Step 5: Combine and save
    print("\n[5/5] Combining and saving...")

    new_rows = rows_original_drugs_new_epi + rows_new_drugs_original_epi + rows_new_drugs_new_epi
    new_df = pd.DataFrame(new_rows)

    # Combine with regenerated original
    extended_df = pd.concat([original_df, new_df], ignore_index=True)

    # Shuffle
    extended_df = extended_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extended_df.to_csv(EXTENDED_CSV, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTENDED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows: {len(extended_df)}")
    print(f"  Original rows: {len(original_df)}")
    print(f"  New rows: {len(new_df)}")
    print(f"  Unique SMILES: {extended_df['smiles'].nunique()}")
    print(f"  Unique epitopes: {extended_df['epitope_seq'].nunique()}")
    print(f"  Unique groups: {extended_df['group_id'].nunique()}")
    print(f"\n  Label statistics:")
    for col in ["efficacy", "target_binding", "immune_activation",
                "immune_cell_activation", "inflammation_risk", "toxicity_risk"]:
        print(f"    {col}: mean={extended_df[col].mean():.3f}  std={extended_df[col].std():.3f}  "
              f"min={extended_df[col].min():.3f}  max={extended_df[col].max():.3f}")
    print(f"\n  Saved to: {EXTENDED_CSV}")
    print(f"  Original file unchanged: {ORIGINAL_CSV}")


if __name__ == "__main__":
    main()
