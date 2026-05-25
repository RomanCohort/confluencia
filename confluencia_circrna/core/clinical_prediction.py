"""
clinical_prediction.py — circRNA Clinical Outcome Prediction

Predicts clinical outcomes for circRNA-based therapies:
1. Survival prognosis (OS, PFS)
2. Response biomarkers
3. Adverse event prediction
4. Quality of life impact
5. Treatment optimization

Literature basis:
- Salzman et al., 2013: circRNA expression in cancer
- Li et al., 2015: circRNA as prognostic marker
- Vo et al., 2019: circRNA signature for survival
- Kristensen et al., 2019: circRNA in cancer diagnosis
- Chen & Mellman, 2017: Immunotherapy biomarkers

Clinical endpoints:
- OS (Overall Survival): time to death
- PFS (Progression Free Survival): time to progression
- ORR (Objective Response Rate): tumor shrinkage
- DCR (Disease Control Rate): stable + response
- QoL (Quality of Life): patient-reported outcomes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum

# Clinical endpoints
class Endpoint(Enum):
    OS = "overall_survival"           # Overall survival
    PFS = "progression_free_survival" # Progression-free
    ORR = "objective_response_rate"   # Objective response
    DCR = "disease_control_rate"      # Disease control
    QOL = "quality_of_life"           # Quality of life


@dataclass
class SurvivalPrediction:
    """Survival time prediction."""
    endpoint: Endpoint
    median_months: float             # Median survival time
    hazard_ratio: float             # Risk ratio vs baseline
    confidence_interval: Tuple[float, float]  # 95% CI
    survival_probability_1yr: float  # 1-year survival rate
    survival_probability_5yr: float  # 5-year survival rate
    risk_group: str                  # low/intermediate/high


@dataclass
class BiomarkerScore:
    """Clinical biomarker score."""
    biomarker_name: str
    value: float
    threshold: float                 # Clinical threshold
    interpretation: str              # Positive/negative
    evidence_level: str              # Strong/moderate/weak
    therapeutic_implication: str     # Actionable insight


@dataclass
class AdverseEventRisk:
    """Adverse event risk prediction."""
    event_type: str                  # AE type
    risk_probability: float          # Probability of occurrence
    severity: str                    # mild/moderate/severe
    time_to_onset: float             # Days to onset
    duration: float                  # Duration in days
    management_strategy: str         # Recommended management


@dataclass
class ClinicalFeatures:
    """Complete clinical outcome prediction."""
    survival_predictions: List[SurvivalPrediction]
    biomarker_scores: List[BiomarkerScore]
    adverse_event_risks: List[AdverseEventRisk]

    overall_risk_score: float        # Combined risk
    prognosis_category: str          # favorable/intermediate/poor
    recommended_followup: List[str]  # Monitoring schedule

    treatment_benefit_score: float   # Net benefit
    quality_of_life_estimate: float  # QoL prediction

    clinical_method: str


class ClinicalOutcomePredictor:
    """
    Predict clinical outcomes for circRNA immunotherapy.

    Methods:
    - Cox regression-based survival prediction
    - Biomarker threshold analysis
    - AE risk modeling
    """

    def __init__(self):
        """Initialize clinical predictor."""
        # Clinical thresholds
        self.biomarker_thresholds = {
            "IPS": {"positive": 7.0, "negative": 4.0},
            "TIDE": {"positive": 0.6, "negative": 0.3},
            "TROP2": {"positive": 8.0, "negative": 5.0},
            "B7-H4": {"positive": 7.0, "negative": 4.0},
            "MKI67": {"positive": 7.0, "negative": 4.0},
            "RIG-I": {"positive": 0.5, "negative": 0.2},
            "TLR": {"positive": 0.4, "negative": 0.1},
        }

    def predict(
        self,
        immune_scores: Dict[str, float],
        gene_expr: Dict[str, float],
        drug_response: Optional[Dict] = None,
        patient_age: int = 60,
        cancer_stage: str = "III",
    ) -> ClinicalFeatures:
        """
        Predict clinical outcomes.

        Args:
            immune_scores: Immune pathway scores
            gene_expr: Gene expression values
            drug_response: Drug response features (optional)
            patient_age: Patient age
            cancer_stage: Cancer stage (I-IV)

        Returns:
            ClinicalFeatures with predictions
        """
        # Survival predictions
        survival = self._predict_survival(
            immune_scores,
            gene_expr,
            cancer_stage
        )

        # Biomarker analysis
        biomarkers = self._analyze_biomarkers(immune_scores, gene_expr)

        # AE risk prediction
        ae_risks = self._predict_adverse_events(immune_scores, gene_expr, patient_age)

        # Overall risk
        risk = self._compute_overall_risk(survival, biomarkers)

        # Prognosis category
        prognosis = self._determine_prognosis(risk, survival)

        # Follow-up schedule
        followup = self._recommend_followup(prognosis, cancer_stage)

        # Treatment benefit
        benefit = self._compute_treatment_benefit(immune_scores, biomarkers)

        # QoL estimate
        qol = self._estimate_quality_of_life(survival, ae_risks)

        return ClinicalFeatures(
            survival_predictions=survival,
            biomarker_scores=biomarkers,
            adverse_event_risks=ae_risks,
            overall_risk_score=risk,
            prognosis_category=prognosis,
            recommended_followup=followup,
            treatment_benefit_score=benefit,
            quality_of_life_estimate=qol,
            clinical_method="cox_regression_approx",
        )

    def _predict_survival(
        self,
        immune_scores: Dict,
        gene_expr: Dict,
        stage: str
    ) -> List[SurvivalPrediction]:
        """Predict survival endpoints."""
        predictions = []

        ips = immune_scores.get("ips", 5.0)
        tide = self._compute_tide(gene_expr)

        # Baseline survival by stage
        stage_baselines = {
            "I": {"OS": 60, "PFS": 50},
            "II": {"OS": 48, "PFS": 36},
            "III": {"OS": 30, "PFS": 18},
            "IV": {"OS": 12, "PFS": 6},
        }

        baseline = stage_baselines.get(stage, {"OS": 24, "PFS": 12})

        # Adjust by IPS and TIDE
        # High IPS = better survival
        # High TIDE = worse survival
        ips_factor = (ips - 5.0) / 5.0 * 0.3
        tide_factor = -tide * 0.2

        # OS prediction
        os_median = baseline["OS"] * (1.0 + ips_factor + tide_factor)
        os_hr = 1.0 - ips_factor  # Lower HR = better

        os_1yr = np.exp(-12 / os_median)  # Simplified exponential
        os_5yr = np.exp(-60 / os_median)

        risk_group = "low" if os_hr < 0.7 else "intermediate" if os_hr < 1.0 else "high"

        predictions.append(SurvivalPrediction(
            endpoint=Endpoint.OS,
            median_months=np.clip(os_median, 6, 120),
            hazard_ratio=np.clip(os_hr, 0.3, 2.0),
            confidence_interval=(os_median * 0.8, os_median * 1.2),
            survival_probability_1yr=np.clip(os_1yr, 0.1, 0.95),
            survival_probability_5yr=np.clip(os_5yr, 0.0, 0.8),
            risk_group=risk_group,
        ))

        # PFS prediction
        pfs_median = baseline["PFS"] * (1.0 + ips_factor + tide_factor * 1.5)
        pfs_hr = os_hr * 1.2

        predictions.append(SurvivalPrediction(
            endpoint=Endpoint.PFS,
            median_months=np.clip(pfs_median, 3, 60),
            hazard_ratio=np.clip(pfs_hr, 0.4, 2.5),
            confidence_interval=(pfs_median * 0.7, pfs_median * 1.3),
            survival_probability_1yr=np.clip(np.exp(-12 / pfs_median), 0.05, 0.9),
            survival_probability_5yr=np.clip(np.exp(-60 / pfs_median), 0.0, 0.6),
            risk_group=risk_group,
        ))

        return predictions

    def _compute_tide(self, gene_expr: Dict) -> float:
        """Compute TIDE score."""
        trop2 = gene_expr.get("TROP2", 0.5)
        b7h4 = gene_expr.get("B7-H4", 0.5)
        mki67 = gene_expr.get("MKI67", 0.5)

        return 0.4 * b7h4 + 0.3 * mki67 + 0.5 * trop2

    def _analyze_biomarkers(
        self,
        immune_scores: Dict,
        gene_expr: Dict
    ) -> List[BiomarkerScore]:
        """Analyze clinical biomarkers."""
        biomarkers = []

        # IPS biomarker
        ips = immune_scores.get("ips", 5.0)
        threshold = self.biomarker_thresholds["IPS"]

        if ips >= threshold["positive"]:
            interpretation = "positive"
            implication = "Predict favorable immunotherapy response"
        elif ips <= threshold["negative"]:
            interpretation = "negative"
            implication = "Consider alternative treatments"
        else:
            interpretation = "indeterminate"
            implication = "Requires additional testing"

        biomarkers.append(BiomarkerScore(
            biomarker_name="IPS (Immunotherapy Potential Score)",
            value=ips,
            threshold=threshold["positive"],
            interpretation=interpretation,
            evidence_level="strong",
            therapeutic_implication=implication,
        ))

        # RIG-I biomarker
        rig_i = immune_scores.get("rig_i_score", 0.3)
        threshold = self.biomarker_thresholds["RIG-I"]

        interpretation = "positive" if rig_i >= threshold["positive"] else "negative"
        implication = "High RIG-I = strong innate activation" if interpretation == "positive" else "Low RIG-I = weak innate response"

        biomarkers.append(BiomarkerScore(
            biomarker_name="RIG-I Activation Score",
            value=rig_i,
            threshold=threshold["positive"],
            interpretation=interpretation,
            evidence_level="moderate",
            therapeutic_implication=implication,
        ))

        # TROP2 biomarker
        trop2 = gene_expr.get("TROP2", 0.5)
        threshold = self.biomarker_thresholds["TROP2"]

        interpretation = "positive" if trop2 >= threshold["positive"] else "negative"
        implication = "High TROP2 = tumor aggressiveness, may need combination"

        biomarkers.append(BiomarkerScore(
            biomarker_name="TROP2 Expression",
            value=trop2,
            threshold=threshold["positive"],
            interpretation=interpretation,
            evidence_level="strong",
            therapeutic_implication=implication,
        ))

        return biomarkers

    def _predict_adverse_events(
        self,
        immune_scores: Dict,
        gene_expr: Dict,
        age: int
    ) -> List[AdverseEventRisk]:
        """Predict adverse event risks."""
        events = []

        immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)

        # Immune-related adverse events (irAEs)
        # Higher immunogenicity = higher irAE risk but also better efficacy

        # Colitis
        colitis_prob = 0.05 + immunogenicity * 0.1
        events.append(AdverseEventRisk(
            event_type="Immune-mediated colitis",
            risk_probability=np.clip(colitis_prob, 0.01, 0.3),
            severity="moderate",
            time_to_onset=45.0,
            duration=21.0,
            management_strategy="Corticosteroids, consider infliximab if severe",
        ))

        # Dermatitis
        dermatitis_prob = 0.1 + immunogenicity * 0.15
        events.append(AdverseEventRisk(
            event_type="Immune-mediated dermatitis",
            risk_probability=np.clip(dermatitis_prob, 0.02, 0.4),
            severity="mild",
            time_to_onset=21.0,
            duration=14.0,
            management_strategy="Topical steroids, antihistamines",
        ))

        # Fatigue
        fatigue_prob = 0.3 + age / 100.0 * 0.2
        events.append(AdverseEventRisk(
            event_type="Fatigue",
            risk_probability=np.clip(fatigue_prob, 0.1, 0.6),
            severity="mild",
            time_to_onset=7.0,
            duration=90.0,
            management_strategy="Rest, exercise, supportive care",
        ))

        # Hepatitis (liver inflammation)
        hepatitis_prob = 0.03 + immunogenicity * 0.05
        events.append(AdverseEventRisk(
            event_type="Immune-mediated hepatitis",
            risk_probability=np.clip(hepatitis_prob, 0.01, 0.2),
            severity="moderate",
            time_to_onset=30.0,
            duration=28.0,
            management_strategy="LFT monitoring, corticosteroids if elevated",
        ))

        return events

    def _compute_overall_risk(
        self,
        survival: List[SurvivalPrediction],
        biomarkers: List[BiomarkerScore]
    ) -> float:
        """Compute overall risk score."""
        # Survival risk
        os_pred = next(s for s in survival if s.endpoint == Endpoint.OS)
        survival_risk = os_pred.hazard_ratio

        # Biomarker risk
        negative_markers = sum(1 for b in biomarkers if b.interpretation == "negative")
        biomarker_risk = negative_markers / len(biomarkers) * 0.5

        return np.clip(survival_risk * 0.6 + biomarker_risk, 0.0, 1.0)

    def _determine_prognosis(self, risk: float, survival: List) -> str:
        """Determine prognosis category."""
        if risk < 0.5:
            return "favorable"
        elif risk < 0.8:
            return "intermediate"
        else:
            return "poor"

    def _recommend_followup(self, prognosis: str, stage: str) -> List[str]:
        """Recommend follow-up schedule."""
        base_schedule = [
            "Monthly: Clinical assessment and labs",
            "Quarterly: Imaging (CT/MRI)",
            "Biannual: Circulating tumor markers",
            "Annual: Comprehensive evaluation",
        ]

        if prognosis == "poor" or stage == "IV":
            base_schedule.insert(0, "Weekly: Close monitoring for first 3 months")

        if prognosis == "favorable":
            base_schedule[1] = "Biannual: Imaging"  # Less frequent

        return base_schedule

    def _compute_treatment_benefit(
        self,
        immune_scores: Dict,
        biomarkers: List[BiomarkerScore]
    ) -> float:
        """Compute treatment benefit score."""
        ips = immune_scores.get("ips", 5.0)

        # Benefit from IPS
        ips_benefit = (ips - 4.0) / 6.0  # Normalized

        # Positive biomarkers increase benefit
        positive_count = sum(1 for b in biomarkers if b.interpretation == "positive")
        biomarker_benefit = positive_count / len(biomarkers) * 0.2

        return np.clip(ips_benefit * 0.7 + biomarker_benefit, 0.0, 1.0)

    def _estimate_quality_of_life(
        self,
        survival: List[SurvivalPrediction],
        ae_risks: List[AdverseEventRisk]
    ) -> float:
        """Estimate quality of life."""
        # Longer survival = better QoL
        os_pred = next(s for s in survival if s.endpoint == Endpoint.OS)
        survival_qol = os_pred.median_months / 60.0

        # Fewer AEs = better QoL
        severe_ae_prob = sum(ae.risk_probability for ae in ae_risks if ae.severity == "severe")
        ae_qol = 1.0 - severe_ae_prob

        return np.clip(survival_qol * 0.5 + ae_qol * 0.5, 0.0, 1.0)


def compute_clinical_score(features: ClinicalFeatures) -> Dict[str, float]:
    """Compute clinical outcome scores."""
    os_pred = next(s for s in features.survival_predictions if s.endpoint == Endpoint.OS)

    return {
        "median_os_months": os_pred.median_months,
        "hazard_ratio": os_pred.hazard_ratio,
        "1yr_survival": os_pred.survival_probability_1yr,
        "5yr_survival": os_pred.survival_probability_5yr,
        "risk_group": os_pred.risk_group,
        "prognosis": features.prognosis_category,
        "treatment_benefit": features.treatment_benefit_score,
        "quality_of_life": features.quality_of_life_estimate,
        "positive_biomarkers": sum(1 for b in features.biomarker_scores if b.interpretation == "positive"),
    }


def generate_clinical_report(
    immune_scores: Dict[str, float],
    gene_expr: Dict[str, float],
    patient_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive clinical report.

    Returns:
        Full report with predictions, recommendations, risks
    """
    predictor = ClinicalOutcomePredictor()

    age = patient_data.get("age", 60) if patient_data else 60
    stage = patient_data.get("stage", "III") if patient_data else "III"

    features = predictor.predict(immune_scores, gene_expr, None, age, stage)

    report = {
        "summary": {
            "prognosis": features.prognosis_category,
            "risk_score": features.overall_risk_score,
            "treatment_benefit": features.treatment_benefit_score,
        },
        "survival": {
            "os_median_months": next(s.median_months for s in features.survival_predictions if s.endpoint == Endpoint.OS),
            "pfs_median_months": next(s.median_months for s in features.survival_predictions if s.endpoint == Endpoint.PFS),
            "1yr_survival": next(s.survival_probability_1yr for s in features.survival_predictions if s.endpoint == Endpoint.OS),
        },
        "biomarkers": [
            {
                "name": b.biomarker_name,
                "value": b.value,
                "interpretation": b.interpretation,
                "implication": b.therapeutic_implication,
            }
            for b in features.biomarker_scores
        ],
        "adverse_events": [
            {
                "type": ae.event_type,
                "probability": ae.risk_probability,
                "severity": ae.severity,
                "management": ae.management_strategy,
            }
            for ae in features.adverse_event_risks
        ],
        "recommendations": {
            "followup_schedule": features.recommended_followup,
            "monitoring": [
                "IPS score monitoring",
                "Gene expression profiling every 3 months",
                "Immune activation markers",
            ],
        },
    }

    return report


def predict_clinical_outcome(
    immune_scores: Dict[str, float],
    gene_expr: Dict[str, float],
) -> ClinicalFeatures:
    """Predict clinical outcomes."""
    predictor = ClinicalOutcomePredictor()
    return predictor.predict(immune_scores, gene_expr)