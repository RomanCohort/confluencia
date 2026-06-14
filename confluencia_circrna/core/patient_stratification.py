"""
Patient Stratification for circRNA Immunotherapy

Stratifies patients into response subgroups based on:
1. Immune phenotype (hot/cold/excluded/mixed TME)
2. Gene expression signatures
3. Biomarker profiles (TROP2, B7-H4, PD-L1, TMB)
4. Predicted drug response patterns

Literature basis:
- Fridman et al., 2012: TME classification
- Chen & Mellman, 2017: Cancer-Immunity Cycle
- Cristescu et al., 2018: Immune tumor microenvironment signatures
- McGrail et al., 2020: Patient stratification for immunotherapy
- Yuan et al., 2022: Immune phenotypes and treatment selection

Key concepts:
- Hot TME: High T cell infiltration, responsive to checkpoint inhibitors
- Cold TME: Low immune infiltration, needs priming (circRNA vaccine)
- Excluded TME: T cells at margin, needs stroma-targeting
- Mixed TME: Intermediate, personalized approach
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# =====================================================================
# Stratification Definitions
# =====================================================================

@dataclass
class BiomarkerThresholds:
    """Thresholds for biomarker-based stratification."""
    # TME markers
    t_cell_infiltration_high: float = 0.6
    t_cell_infiltration_low: float = 0.3

    pd_l1_high: float = 0.5
    pd_l1_low: float = 0.2

    tmb_high: float = 10.0  # mutations/Mb
    tmb_low: float = 5.0

    # Immunosuppression markers
    treg_high: float = 0.4
    mdsc_high: float = 0.3
    m2_high: float = 0.4

    # Target expression
    trop2_high: float = 0.6
    b7h4_high: float = 0.5

    # Gene signatures
    ifng_signature_high: float = 0.5
    tgf_beta_signature_high: float = 0.4


@dataclass
class PatientStratum:
    """A patient stratum with associated treatment recommendations."""
    stratum_id: str
    stratum_name: str
    description: str

    # Characteristics
    tme_type: str
    immune_score_range: Tuple[float, float]
    key_biomarkers: List[str]

    # Treatment recommendations
    recommended_primary: str
    recommended_combinations: List[str]
    expected_response_range: Tuple[float, float]
    priority_actions: List[str]

    # Monitoring
    key_monitoring_params: List[str]
    response_timeline_weeks: int


# Predefined strata based on literature
PATIENT_STRATA: Dict[str, PatientStratum] = {
    "S1_HOT_RESPONSIVE": PatientStratum(
        stratum_id="S1",
        stratum_name="Hot TME / High Response Expected",
        description="High T cell infiltration, PD-L1 positive, inflamed TME",
        tme_type="hot",
        immune_score_range=(70.0, 100.0),
        key_biomarkers=["CD8A", "GZMB", "PRF1", "IFNG", "PDCD1"],
        recommended_primary="checkpoint_inhibitor",
        recommended_combinations=["circrna_vaccine", "anti_pd1"],
        expected_response_range=(0.6, 0.85),
        priority_actions=["Start checkpoint inhibitor", "Add circRNA for enhancement"],
        key_monitoring_params=["t_cell_count", "pd_l1_expression", "tumor_size"],
        response_timeline_weeks=8,
    ),
    "S2_COLD_PRIMING_NEEDED": PatientStratum(
        stratum_id="S2",
        stratum_name="Cold TME / Priming Required",
        description="Low T cell infiltration, needs immune activation",
        tme_type="cold",
        immune_score_range=(0.0, 30.0),
        key_biomarkers=["low_CD8A", "low_IFNG", "high_TGFB1", "high_VEGFA"],
        recommended_primary="circrna_vaccine",
        recommended_combinations=["circrna_vaccine + cyclophosphamide", "circrna_vaccine + pembrolizumab"],
        expected_response_range=(0.3, 0.55),
        priority_actions=["Prime immune system with circRNA", "Deplete Tregs with low-dose cyclophosphamide"],
        key_monitoring_params=["t_cell_activation", "treg_count", "antibody_response"],
        response_timeline_weeks=12,
    ),
    "S3_EXCLUDED_STROMA_TARGET": PatientStratum(
        stratum_id="S3",
        stratum_name="Immune Excluded / Stroma Barrier",
        description="T cells present but excluded by stroma/CAF barrier",
        tme_type="excluded",
        immune_score_range=(30.0, 50.0),
        key_biomarkers=["margin_CD8", "high_FAP", "high_CAF", "high_TGFB1"],
        recommended_primary="stroma_targeting",
        recommended_combinations=["circrna_vaccine + bevacizumab", "bevacizumab + pembrolizumab"],
        expected_response_range=(0.35, 0.60),
        priority_actions=["Normalize vasculature with bevacizumab", "Enable T cell infiltration"],
        key_monitoring_params=["tumor_vascularization", "stroma_density", "t_cell_margin_core_ratio"],
        response_timeline_weeks=10,
    ),
    "S4_IMMunosupPRESSIVE": PatientStratum(
        stratum_id="S4",
        stratum_name="Immunosuppressive Dominant",
        description="High Treg/MDSC/M2, suppressed immune response",
        tme_type="mixed",
        immune_score_range=(20.0, 40.0),
        key_biomarkers=["high_FOXP3", "high_CD163", "high_MDSC", "high_IL10"],
        recommended_primary="suppressor_depletion",
        recommended_combinations=["cyclophosphamide + circrna_vaccine + pembrolizumab"],
        expected_response_range=(0.25, 0.50),
        priority_actions=["Deplete Tregs (low-dose cyclophosphamide)", "Repolarize M2 to M1"],
        key_monitoring_params=["treg_count", "m2_m1_ratio", "mdsc_count"],
        response_timeline_weeks=14,
    ),
    "S5_HIGH_TARGET_LOW_IMMUNE": PatientStratum(
        stratum_id="S5",
        stratum_name="High Target Expression / Low Immune",
        description="TROP2/B7-H4 high but immune context unfavorable",
        tme_type="mixed",
        immune_score_range=(25.0, 45.0),
        key_biomarkers=["high_TROP2", "high_B7H4", "low_CD8A"],
        recommended_primary="targeted_therapy_circrna",
        recommended_combinations=["circrna_vaccine + pembrolizumab", "targeted_therapy"],
        expected_response_range=(0.40, 0.65),
        priority_actions=["Target high antigen expression", "Combine with checkpoint"],
        key_monitoring_params=["target_expression", "tumor_response", "immune_activation"],
        response_timeline_weeks=12,
    ),
    "S6_MIXED_INTERMEDIATE": PatientStratum(
        stratum_id="S6",
        stratum_name="Mixed TME / Intermediate Response",
        description="Mixed features, needs personalized approach",
        tme_type="mixed",
        immune_score_range=(40.0, 70.0),
        key_biomarkers=["mixed_profile"],
        recommended_primary="personalized_combination",
        recommended_combinations=["circrna_vaccine + pembrolizumab + bevacizumab"],
        expected_response_range=(0.45, 0.70),
        priority_actions=["Comprehensive profiling", "Adaptive therapy selection"],
        key_monitoring_params=["all_parameters", "weekly_assessment"],
        response_timeline_weeks=10,
    ),
}


# =====================================================================
# Stratification Engine
# =====================================================================

@dataclass
class StratificationResult:
    """Result from patient stratification."""
    stratum: PatientStratum
    confidence: float

    # Detailed scores
    immune_score: float
    tme_type: str
    response_prediction: float

    # Biomarker summary
    key_positive_biomarkers: List[str]
    key_negative_biomarkers: List[str]

    # Treatment recommendation
    recommended_treatment: str
    recommended_combinations: List[str]
    expected_response: float

    # Monitoring plan
    monitoring_schedule: List[str]
    check_points: List[float]  # Weeks


class PatientStratifier:
    """
    Stratifies patients into treatment response subgroups.

    Uses:
    1. Biomarker thresholds (gene expression, TME markers)
    2. Immune score calculation
    3. TME type classification
    4. Response prediction
    5. Treatment recommendation matching
    """

    def __init__(self, thresholds: Optional[BiomarkerThresholds] = None):
        self.thresholds = thresholds or BiomarkerThresholds()
        self.strata = PATIENT_STRATA

    def stratify(
        self,
        patient_profile: Dict[str, float],
        gene_expression: Optional[Dict[str, float]] = None,
        immune_cell_counts: Optional[Dict[str, float]] = None,
    ) -> StratificationResult:
        """
        Stratify patient into response subgroup.

        Args:
            patient_profile: Patient features (trop2, b7h4, pd_l1, etc.)
            gene_expression: Gene expression values (normalized)
            immune_cell_counts: Immune cell counts (cells/mm³)

        Returns:
            StratificationResult with stratum and recommendations
        """
        # Merge all data
        profile = patient_profile.copy()
        if gene_expression:
            profile.update(gene_expression)
        if immune_cell_counts:
            profile.update(immune_cell_counts)

        # Step 1: Compute immune score
        immune_score = self._compute_immune_score(profile)

        # Step 2: Classify TME type
        tme_type = self._classify_tme_type(profile, immune_score)

        # Step 3: Identify key biomarkers
        positive_biomarkers, negative_biomarkers = self._identify_key_biomarkers(profile)

        # Step 4: Predict response
        response_prediction = self._predict_response(profile, immune_score, tme_type)

        # Step 5: Match to stratum
        stratum, confidence = self._match_stratum(
            immune_score=immune_score,
            tme_type=tme_type,
            positive_biomarkers=positive_biomarkers,
            profile=profile,
        )

        # Step 6: Generate recommendations
        recommended_treatment, recommended_combinations = self._recommend_treatment(stratum, profile)
        expected_response = self._estimate_expected_response(stratum, response_prediction)

        # Step 7: Generate monitoring plan
        monitoring_schedule, check_points = self._generate_monitoring(stratum)

        return StratificationResult(
            stratum=stratum,
            confidence=confidence,
            immune_score=immune_score,
            tme_type=tme_type,
            response_prediction=response_prediction,
            key_positive_biomarkers=positive_biomarkers,
            key_negative_biomarkers=negative_biomarkers,
            recommended_treatment=recommended_treatment,
            recommended_combinations=recommended_combinations,
            expected_response=expected_response,
            monitoring_schedule=monitoring_schedule,
            check_points=check_points,
        )

    def _compute_immune_score(self, profile: Dict[str, float]) -> float:
        """Compute immune score (0-100) based on multiple factors."""
        # T cell infiltration
        cd8 = profile.get("CD8A", profile.get("t_cell_infiltration", 0.5))
        gzmb = profile.get("GZMB", 0.3)
        prf1 = profile.get("PRF1", 0.3)

        t_cell_score = 0.4 * cd8 + 0.3 * gzmb + 0.3 * prf1

        # IFN-γ signature
        ifng = profile.get("IFNG", profile.get("ifng_signature", 0.5))
        ifng_score = ifng

        # Immunosuppression penalty
        treg = profile.get("FOXP3", profile.get("treg_ratio", 0.3))
        mdsc = profile.get("MDSC", profile.get("mdsc_count", 0.2))
        m2 = profile.get("CD163", profile.get("m2_ratio", 0.3))
        tgfb = profile.get("TGFB1", profile.get("tgf_beta", 0.3))

        suppression_penalty = 0.3 * treg + 0.2 * mdsc + 0.2 * m2 + 0.3 * tgfb

        # Combined score
        raw_score = 100.0 * (t_cell_score * 0.4 + ifng_score * 0.3 - suppression_penalty * 0.3)
        return float(np.clip(raw_score, 0.0, 100.0))

    def _classify_tme_type(
        self,
        profile: Dict[str, float],
        immune_score: float,
    ) -> str:
        """Classify TME type (hot/cold/excluded/mixed)."""
        cd8 = profile.get("CD8A", profile.get("t_cell_infiltration", 0.5))

        # Check for exclusion markers
        fap = profile.get("FAP", 0.0)
        tgfb = profile.get("TGFB1", profile.get("tgf_beta", 0.3))
        vegf = profile.get("VEGFA", 0.0)

        exclusion_score = 0.4 * fap + 0.3 * tgfb + 0.3 * vegf

        if immune_score >= 70.0 and cd8 >= self.thresholds.t_cell_infiltration_high:
            return "hot"
        elif immune_score < 30.0 and cd8 < self.thresholds.t_cell_infiltration_low:
            return "cold"
        elif exclusion_score > 0.5 and cd8 >= 0.4:
            # T cells present but excluded
            return "excluded"
        else:
            return "mixed"

    def _identify_key_biomarkers(
        self,
        profile: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        """Identify positive and negative biomarkers."""
        positive = []
        negative = []

        # Check thresholds
        checks = [
            ("CD8A", self.thresholds.t_cell_infiltration_high, positive, "high_CD8"),
            ("CD8A", self.thresholds.t_cell_infiltration_low, negative, "low_CD8"),
            ("PDCD1LG2", self.thresholds.pd_l1_high, positive, "high_PD_L1"),
            ("PDCD1LG2", self.thresholds.pd_l1_low, negative, "low_PD_L1"),
            ("FOXP3", self.thresholds.treg_high, positive, "high_Treg"),
            ("CD163", self.thresholds.m2_high, positive, "high_M2"),
            ("TROP2", self.thresholds.trop2_high, positive, "high_TROP2"),
            ("B7H4", self.thresholds.b7h4_high, positive, "high_B7H4"),
            ("IFNG", self.thresholds.ifng_signature_high, positive, "high_IFNG"),
            ("TGFB1", self.thresholds.tgf_beta_signature_high, positive, "high_TGFB"),
        ]

        for marker, threshold, target_list, label in checks:
            value = profile.get(marker, profile.get(label.replace("high_", "").replace("low_", "").lower(), 0.5))
            if value >= threshold:
                target_list.append(label)
            elif value < threshold * 0.5:
                if label.startswith("high"):
                    target_list.append(label.replace("high_", "low_"))
                elif label.startswith("low"):
                    target_list.append(label.replace("low_", "high_"))

        return positive, negative

    def _predict_response(
        self,
        profile: Dict[str, float],
        immune_score: float,
        tme_type: str,
    ) -> float:
        """Predict response probability."""
        # Base response from immune score
        base = immune_score / 100.0 * 0.6

        # TME type adjustment
        tme_factors = {
            "hot": 0.15,
            "cold": -0.10,
            "excluded": -0.05,
            "mixed": 0.0,
        }
        base += tme_factors.get(tme_type, 0.0)

        # PD-L1 adjustment
        pd_l1 = profile.get("PDCD1LG2", profile.get("pd_l1", 0.5))
        if pd_l1 >= self.thresholds.pd_l1_high:
            base += 0.1

        # TMB adjustment (if available)
        tmb = profile.get("TMB", 0.0)
        if tmb >= self.thresholds.tmb_high:
            base += 0.15
        elif tmb < self.thresholds.tmb_low:
            base -= 0.05

        return float(np.clip(base, 0.0, 1.0))

    def _match_stratum(
        self,
        immune_score: float,
        tme_type: str,
        positive_biomarkers: List[str],
        profile: Dict[str, float],
    ) -> Tuple[PatientStratum, float]:
        """Match patient to best stratum."""
        best_stratum = None
        best_confidence = 0.0

        for stratum_id, stratum in self.strata.items():
            # Check TME type match
            if stratum.tme_type != tme_type:
                continue

            # Check immune score range
            min_score, max_score = stratum.immune_score_range
            if not (min_score <= immune_score <= max_score):
                continue

            # Compute confidence based on biomarker match
            matched_biomarkers = sum(
                1 for bm in stratum.key_biomarkers
                if bm in positive_biomarkers or bm.replace("high_", "low_") in positive_biomarkers
            )
            confidence = matched_biomarkers / max(len(stratum.key_biomarkers), 1)

            if confidence > best_confidence:
                best_stratum = stratum
                best_confidence = confidence

        # Fallback to mixed if no match
        if best_stratum is None:
            best_stratum = self.strata["S6_MIXED_INTERMEDIATE"]
            best_confidence = 0.5

        return best_stratum, best_confidence

    def _recommend_treatment(
        self,
        stratum: PatientStratum,
        profile: Dict[str, float],
    ) -> Tuple[str, List[str]]:
        """Generate specific treatment recommendation."""
        primary = stratum.recommended_primary

        # Adjust based on specific profile features
        trop2 = profile.get("TROP2", profile.get("trop2", 0.5))
        b7h4 = profile.get("B7H4", profile.get("b7h4", 0.5))
        pd_l1 = profile.get("PDCD1LG2", profile.get("pd_l1", 0.5))

        combinations = stratum.recommended_combinations.copy()

        # High target expression: prioritize circRNA
        if trop2 >= self.thresholds.trop2_high or b7h4 >= self.thresholds.b7h4_high:
            if "circrna_vaccine" not in combinations:
                combinations.insert(0, "circrna_vaccine + pembrolizumab")

        # High PD-L1: prioritize checkpoint
        if pd_l1 >= self.thresholds.pd_l1_high:
            if primary == "circrna_vaccine":
                primary = "checkpoint_inhibitor"

        return primary, combinations

    def _estimate_expected_response(
        self,
        stratum: PatientStratum,
        response_prediction: float,
    ) -> float:
        """Estimate expected response for matched stratum."""
        min_resp, max_resp = stratum.expected_response_range
        # Interpolate based on response_prediction
        return min_resp + (max_resp - min_resp) * response_prediction

    def _generate_monitoring(
        self,
        stratum: PatientStratum,
    ) -> Tuple[List[str], float]:
        """Generate monitoring schedule."""
        schedule = []

        # Standard monitoring
        schedule.append("Baseline: Comprehensive tumor profiling + immune assessment")
        schedule.append(f"Week {stratum.response_timeline_weeks // 2}: Response assessment + adverse event monitoring")
        schedule.append(f"Week {stratum.response_timeline_weeks}: Full response evaluation")

        # Stratum-specific monitoring
        for param in stratum.key_monitoring_params:
            if param == "t_cell_count":
                schedule.append("Monthly: T cell subset analysis (CD8, CD4, Treg)")
            elif param == "pd_l1_expression":
                schedule.append("Every 8 weeks: PD-L1 expression assessment")
            elif param == "treg_count":
                schedule.append("Biweekly: Treg count monitoring")
            elif param == "m2_m1_ratio":
                schedule.append("Monthly: Macrophage polarization assessment")

        check_points = [
            float(stratum.response_timeline_weeks // 3),
            float(stratum.response_timeline_weeks // 2),
            float(stratum.response_timeline_weeks),
        ]

        return schedule, check_points


# =====================================================================
# Batch Stratification
# =====================================================================

def stratify_patients(
    patients: pd.DataFrame,
    stratifier: Optional[PatientStratifier] = None,
) -> pd.DataFrame:
    """
    Stratify multiple patients.

    Args:
        patients: DataFrame with patient profiles
        stratifier: Stratifier instance

    Returns:
        DataFrame with stratification results
    """
    stratifier = stratifier or PatientStratifier()

    results = []
    for idx, row in patients.iterrows():
        profile = row.to_dict()

        result = stratifier.stratify(profile)

        results.append({
            "patient_id": idx,
            "stratum_id": result.stratum.stratum_id,
            "stratum_name": result.stratum.stratum_name,
            "confidence": result.confidence,
            "immune_score": result.immune_score,
            "tme_type": result.tme_type,
            "response_prediction": result.response_prediction,
            "recommended_primary": result.recommended_treatment,
            "top_combination": result.recommended_combinations[0] if result.recommended_combinations else "",
            "expected_response": result.expected_response,
        })

    return pd.DataFrame(results)


def get_stratum_summary(stratification_results: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for each stratum."""
    return stratification_results.groupby("stratum_name").agg({
        "patient_id": "count",
        "immune_score": ["mean", "std"],
        "response_prediction": ["mean", "std"],
        "expected_response": ["mean", "std"],
    }).round(2)


# =====================================================================
# Convenience Functions
# =====================================================================

def stratify_patient(
    trop2: float = 0.5,
    b7h4: float = 0.5,
    pd_l1: float = 0.5,
    t_cell_infiltration: float = 0.5,
    treg_ratio: float = 0.3,
    tgfb: float = 0.3,
    tmb: float = 5.0,
    **kwargs,
) -> StratificationResult:
    """Quick stratification for a single patient."""
    stratifier = PatientStratifier()

    profile = {
        "TROP2": trop2,
        "B7H4": b7h4,
        "PDCD1LG2": pd_l1,
        "CD8A": t_cell_infiltration,
        "FOXP3": treg_ratio,
        "TGFB1": tgfb,
        "TMB": tmb,
        **kwargs,
    }

    return stratifier.stratify(profile)


__all__ = [
    "BiomarkerThresholds",
    "PatientStratum",
    "PATIENT_STRATA",
    "StratificationResult",
    "PatientStratifier",
    "stratify_patients",
    "get_stratum_summary",
    "stratify_patient",
]