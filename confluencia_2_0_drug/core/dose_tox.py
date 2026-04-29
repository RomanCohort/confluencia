"""
Dose-Dependent Toxicity Modeling
=================================
Estimates toxicity metrics as a function of dose:
- TD50: Median toxic dose
- LD50: Median lethal dose (acute toxicity)
- MTD: Maximum tolerated dose
- NOAEL: No observed adverse effect level
- Organ-specific dose-response curves
- Therapeutic index (TI = TD50 / ED50)

References:
-ogat & Takatsuki (1999) Food Chem Toxicol 37:91-97 (BMD modeling)
- Slob (2002) Crit Rev Toxicol 32:329-350 (benchmark dose)
- EFSA (2022) Guidance on BMD approach in risk assessment
- Medinsky & Jarabek (1995) Toxicol Lett 79:155-166 (PK/PD-toxicity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

from .admet import ADMETResult, predict_admet
from .toxicophore import detect_toxicophores
from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Organ systems with toxicity endpoints
# ---------------------------------------------------------------------------

ORGAN_SYSTEMS = [
    "cardiac",       # Heart / cardiotoxicity
    "hepatic",       # Liver / hepatotoxicity
    "renal",         # Kidney / nephrotoxicity
    "neurological",  # CNS / neurotoxicity
    "hematologic",   # Blood / hematotoxicity
    "pulmonary",     # Lungs / pulmonary toxicity
    "gastrointestinal", # GI tract
    "dermal",        # Skin
]

# Organ sensitivity multipliers based on ADMET endpoints
_ORGAN_ADMET_MAP: Dict[str, Dict[str, float]] = {
    "cardiac": {"hERG_risk": 0.55, "CYP3A4_inhibition": 0.20, "CYP2D6_inhibition": 0.15},
    "hepatic": {"hepatotoxicity_risk": 0.50, "CYP_total_risk": 0.30, "AMES_positive": 0.10},
    "renal": {"aqueous_solubility": -0.20, "CYP_total_risk": 0.15, "hepatotoxicity_risk": 0.25},
    "neurological": {"BBB_positive": 0.40, "hERG_risk": 0.15, "CYP2D6_inhibition": 0.20},
    "hematologic": {"AMES_positive": 0.35, "skin_sensitization": 0.20, "reactive_groups": 0.25},
    "pulmonary": {"skin_sensitization": 0.30, "hepatotoxicity_risk": 0.20, "AMES_positive": 0.15},
    "gastrointestinal": {"caco2_permeability": -0.15, "hepatotoxicity_risk": 0.35, "CYP_total_risk": 0.20},
    "dermal": {"skin_sensitization": 0.55, "AMES_positive": 0.15, "hERG_risk": 0.10},
}

# Toxicophore severity multipliers per organ
_TOXOPHORE_ORGAN_FACTOR = {
    "high": 1.5,
    "medium": 1.2,
    "low": 1.05,
}


# ---------------------------------------------------------------------------
# Dose-response curve models
# ---------------------------------------------------------------------------

def _hill_equation(dose: np.ndarray, td50: float, hill: float, background: float = 0.0) -> np.ndarray:
    """Hill equation for dose-response: response = background + dose^hill / (td50^hill + dose^hill)."""
    dose = np.asarray(dose, dtype=np.float64)
    return background + dose**hill / (td50**hill + dose**hill)


def _sigmoid_emax(dose: np.ndarray, emax: float, ec50: float, hill: float) -> np.ndarray:
    """Sigmoid Emax model: E = Emax * D^Hill / (EC50^Hill + D^Hill)."""
    dose = np.asarray(dose, dtype=np.float64)
    return emax * dose**hill / (ec50**hill + dose**hill)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrganToxicity:
    """Toxicity metrics for a specific organ system."""
    organ: str
    sensitivity: float         # 0-1, how sensitive this organ is
    td50_mg: float             # Median toxic dose (mg/kg)
    mtd_mg: float              # Maximum tolerated dose (mg/kg)
    noael_mg: float            # No observed adverse effect level (mg/kg)
    severity: str              # "low", "medium", "high"
    dose_response: Dict[float, float]  # dose -> toxicity probability


@dataclass
class DoseToxicityReport:
    """Complete dose-dependent toxicity report."""
    smiles: str
    admet: ADMETResult
    ld50_mgkg: float           # Acute toxicity LD50 (mg/kg)
    td50_mgkg: float           # Median toxic dose (mg/kg)
    mtd_mgkg: float            # Maximum tolerated dose (mg/kg)
    noael_mgkg: float          # No observed adverse effect level
    therapeutic_index: float   # TI = TD50 / ED50 (higher = safer)
    safety_margin: float       # NOAEL / expected therapeutic dose
    organ_toxicities: Dict[str, OrganToxicity]
    dose_response_curve: Dict[float, float]  # dose_mgkg -> toxicity_prob
    risk_classification: str   # "safe", "caution", "dangerous"
    summary: str

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "ld50_mgkg": self.ld50_mgkg,
            "td50_mgkg": self.td50_mgkg,
            "mtd_mgkg": self.mtd_mgkg,
            "noael_mgkg": self.noael_mgkg,
            "therapeutic_index": self.therapeutic_index,
            "safety_margin": self.safety_margin,
            "risk_classification": self.risk_classification,
            "organ_toxicities": {
                k: {
                    "sensitivity": v.sensitivity,
                    "td50_mg": v.td50_mg,
                    "mtd_mg": v.mtd_mg,
                    "noael_mg": v.noael_mg,
                    "severity": v.severity,
                }
                for k, v in self.organ_toxicities.items()
            },
            "summary": self.summary,
        }


class DoseToxicityModel:
    """
    Dose-dependent toxicity estimation combining ADMET predictions,
    toxicophore alerts, and literature-derived dose-response models.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def estimate(
        self,
        smiles: str,
        ed50_mgkg: float = 10.0,       # Expected effective dose
        expected_dose_mgkg: float = 5.0,  # Planned therapeutic dose
        species: str = "human",
    ) -> DoseToxicityReport:
        """
        Estimate dose-dependent toxicity for a molecule.

        Args:
            smiles: SMILES string
            ed50_mgkg: Median effective dose (mg/kg)
            expected_dose_mgkg: Expected therapeutic dose (mg/kg)
            species: Target species for scaling

        Returns:
            DoseToxicityReport with full dose-response analysis.
        """
        # Step 1: Get ADMET predictions
        admet = predict_admet(smiles)

        # Step 2: Get toxicophore report
        tox_report = detect_toxicophores(smiles)

        # Step 3: Estimate LD50 from ADMET composite risk
        # Based on literature: LD50 typically 10-100x TD50 for small molecules
        overall_risk = admet.overall_risk

        # Adjust for toxicophores
        tox_adjustment = 1.0
        if tox_report.high_risk_count > 0:
            tox_adjustment *= 0.5 ** tox_report.high_risk_count
        if tox_report.medium_risk_count > 0:
            tox_adjustment *= 0.8 ** tox_report.medium_risk_count
        tox_adjustment = max(tox_adjustment, 0.01)

        # Base LD50 estimate (mg/kg) - higher risk = lower LD50
        # Typical range: 10 mg/kg (very toxic) to 5000 mg/kg (practically non-toxic)
        base_ld50 = 5000.0 * (1.0 - overall_risk) * tox_adjustment
        noise = self.rng.normal(1.0, 0.15)
        ld50_mgkg = float(np.clip(base_ld50 * noise, 1.0, 15000.0))

        # TD50 ≈ LD50 / 10 (general rule for most drugs)
        td50_mgkg = float(ld50_mgkg / 10.0)

        # MTD ≈ TD50 / 3 (conservative estimate)
        mtd_mgkg = float(td50_mgkg / 3.0)

        # NOAEL ≈ MTD / 2
        noael_mgkg = float(mtd_mgkg / 2.0)

        # Therapeutic Index = TD50 / ED50
        ti = td50_mgkg / max(ed50_mgkg, 0.01)

        # Safety margin = NOAEL / expected dose
        safety_margin = noael_mgkg / max(expected_dose_mgkg, 0.01)

        # Step 4: Organ-specific toxicity
        organ_toxicities = self._estimate_organ_toxicity(
            admet, tox_report, td50_mgkg, mtd_mgkg, noael_mgkg
        )

        # Step 5: Generate dose-response curve
        dose_range = np.logspace(-2, np.log10(ld50_mgkg * 2), 50)
        dose_response = self._generate_dose_response(
            dose_range, td50_mgkg, overall_risk, tox_adjustment
        )

        # Step 6: Risk classification
        if ti >= 10 and overall_risk < 0.3:
            risk_class = "safe"
        elif ti >= 3 and overall_risk < 0.5:
            risk_class = "caution"
        else:
            risk_class = "dangerous"

        # Summary
        summary = self._generate_summary(
            smiles, ld50_mgkg, td50_mgkg, mtd_mgkg, noael_mgkg,
            ti, safety_margin, risk_class, organ_toxicities, admet, tox_report
        )

        return DoseToxicityReport(
            smiles=smiles,
            admet=admet,
            ld50_mgkg=ld50_mgkg,
            td50_mgkg=td50_mgkg,
            mtd_mgkg=mtd_mgkg,
            noael_mgkg=noael_mgkg,
            therapeutic_index=ti,
            safety_margin=safety_margin,
            organ_toxicities=organ_toxicities,
            dose_response_curve=dose_response,
            risk_classification=risk_class,
            summary=summary,
        )

    def _estimate_organ_toxicity(
        self,
        admet: ADMETResult,
        tox_report,
        td50: float,
        mtd: float,
        noael: float,
    ) -> Dict[str, OrganToxicity]:
        """Estimate organ-specific toxicity sensitivities."""
        result = {}

        for organ in ORGAN_SYSTEMS:
            admet_map = _ORGAN_ADMET_MAP.get(organ, {})

            # Calculate organ sensitivity from ADMET endpoints
            sensitivity = 0.0
            for endpoint, weight in admet_map.items():
                val = getattr(admet, endpoint, 0.0)
                if endpoint == "aqueous_solubility":
                    # Higher solubility = lower toxicity
                    sensitivity += weight * max(0, 1.0 - (val + 2.0) / 4.0)
                else:
                    sensitivity += weight * val
            sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

            # Adjust for toxicophore alerts
            tox_factor = 1.0
            for match in tox_report.matches:
                if match.severity in _TOXOPHORE_ORGAN_FACTOR:
                    # Some toxicophores are organ-specific
                    if match.category in ("hepatotoxic",) and organ == "hepatic":
                        tox_factor *= _TOXOPHORE_ORGAN_FACTOR[match.severity]
                    elif match.category in ("mutagenic", "carcinogenic") and organ == "hematologic":
                        tox_factor *= _TOXOPHORE_ORGAN_FACTOR[match.severity]

            # Organ-specific dose adjustments
            organ_td50 = td50 / max(sensitivity * tox_factor, 0.1)
            organ_mtd = mtd / max(sensitivity * tox_factor, 0.1)
            organ_noael = noael / max(sensitivity * tox_factor, 0.1)

            # Severity classification
            if sensitivity > 0.6:
                severity = "high"
            elif sensitivity > 0.3:
                severity = "medium"
            else:
                severity = "low"

            # Generate organ dose-response
            doses = np.array([0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
            hill = 1.0 + sensitivity * 2.0
            resp = _hill_equation(doses, organ_td50, hill)
            dose_resp = {float(d): float(r) for d, r in zip(doses, resp)}

            result[organ] = OrganToxicity(
                organ=organ,
                sensitivity=sensitivity,
                td50_mg=organ_td50,
                mtd_mg=organ_mtd,
                noael_mg=organ_noael,
                severity=severity,
                dose_response=dose_resp,
            )

        return result

    def _generate_dose_response(
        self,
        dose_range: np.ndarray,
        td50: float,
        overall_risk: float,
        tox_adjustment: float,
    ) -> Dict[float, float]:
        """Generate full dose-response curve."""
        hill = 1.5 + overall_risk * 2.0
        response = _hill_equation(dose_range, td50, hill)
        return {float(d): float(r) for d, r in zip(dose_range, response)}

    def _generate_summary(
        self, smiles, ld50, td50, mtd, noael, ti, safety_margin,
        risk_class, organ_toxicities, admet, tox_report
    ) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Dose-Dependent Toxicity Report: {smiles[:40]}...")
        lines.append("=" * 55)

        # Risk classification
        risk_emoji = {"safe": "[OK]", "caution": "[!!]", "dangerous": "[XX]"}
        lines.append(f"\n  Overall: {risk_emoji.get(risk_class, '??')} {risk_class.upper()}")

        lines.append(f"\n  LD50:     {ld50:.1f} mg/kg")
        lines.append(f"  TD50:     {td50:.1f} mg/kg")
        lines.append(f"  MTD:      {mtd:.1f} mg/kg")
        lines.append(f"  NOAEL:    {noael:.1f} mg/kg")
        lines.append(f"  TI:       {ti:.1f} {'(wide margin)' if ti > 10 else '(narrow - caution!)'}")

        # Most sensitive organs
        sorted_organs = sorted(
            organ_toxicities.items(),
            key=lambda x: x[1].sensitivity,
            reverse=True
        )
        lines.append(f"\n  Organ Sensitivity (top 3):")
        for organ, data in sorted_organs[:3]:
            lines.append(f"    {organ}: {data.sensitivity:.2f} ({data.severity})")

        # Toxicophore alerts
        if tox_report.total_alerts > 0:
            lines.append(f"\n  Structural Alerts: {tox_report.total_alerts}")
            for m in tox_report.matches[:5]:
                lines.append(f"    - {m.name}: {m.description}")
        else:
            lines.append(f"\n  Structural Alerts: None")

        return "\n".join(lines)


# Global instance
_model: Optional[DoseToxicityModel] = None


def get_dose_tox_model() -> DoseToxicityModel:
    global _model
    if _model is None:
        _model = DoseToxicityModel()
    return _model


def estimate_dose_toxicity(
    smiles: str,
    ed50_mgkg: float = 10.0,
    expected_dose_mgkg: float = 5.0,
) -> DoseToxicityReport:
    """Quick function to estimate dose-dependent toxicity."""
    return get_dose_tox_model().estimate(smiles, ed50_mgkg, expected_dose_mgkg)
