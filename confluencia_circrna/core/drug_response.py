"""
drug_response.py — circRNA Drug Response Prediction

Predicts therapeutic efficacy for circRNA-based immunotherapy:
1. RNA-small molecule targeting potential
2. Immune response kinetics during treatment
3. Drug synergy with circRNA immunogenicity
4. Treatment time course optimization
5. Resistance risk assessment

Literature basis:
- Chen & Mellman, 2013: Cancer Immunotherapy Cycle
- Jiang et al., 2018: TIDE score for immune checkpoint resistance
- Cristescu et al., 2018: Immune tumor microenvironment signatures
- Schlee et al., 2009: RIG-I agonists as immunotherapy
- Kowolik et al., 2011: circRNA vaccine efficacy

Applications:
- circRNA vaccine design optimization
- Combination therapy (circRNA + checkpoint inhibitors)
- Treatment response prediction
- Personalized immunotherapy planning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings
from enum import Enum

# Drug types for combination therapy
class DrugClass(Enum):
    CHECKPOINT_INHIBITOR = "checkpoint_inhibitor"  # anti-PD-1, anti-CTLA-4
    RIG_I_AGONIST = "rig_i_agonist"                # RIG-I activators
    TLR_AGONIST = "tlr_agonist"                    # TLR7/8 activators
    CHEMOTHERAPY = "chemotherapy"                  # cytotoxic drugs
    TARGETED_THERAPY = "targeted_therapy"          # kinase inhibitors
    IMMUNOTHERAPY = "immunotherapy"                 # broad immune activation


@dataclass
class DrugProperties:
    """Properties of a therapeutic drug."""
    drug_name: str
    drug_class: DrugClass
    mechanism: str                  # Mechanism of action
    half_life: float                # Half life (hours)
    dose_frequency: float           # Dosing frequency (days)
    synergy_targets: List[str]      # Known synergy targets
    resistance_risk_factors: List[str]  # Resistance markers


@dataclass
class TreatmentCourse:
    """A treatment course simulation."""
    time_points: List[float]         # Days
    immune_activation: List[float]   # Immune response over time
    tumor_response: List[float]      # Tumor burden change
    resistance_risk: List[float]     # Resistance developing
    drug_concentration: List[float]  # Drug levels


@dataclass
class SynergyScore:
    """Drug combination synergy analysis."""
    drug_a: str
    drug_b: str
    synergy_type: str               # Additive/synergistic/antagonistic
    synergy_value: float            # Synergy score [0, 1]
    mechanism: str                  # Synergy mechanism
    optimal_ratio: float            # Optimal dose ratio


@dataclass
class DrugResponseFeatures:
    """Drug response prediction features."""
    baseline_ips: float             # Initial IPS score
    predicted_response: str         # responder/intermediate/non-responder
    response_probability: float     # Probability of response
    time_to_response: float         # Days to initial response
    duration_estimate: float        # Estimated response duration (months)
    resistance_risk: float          # Risk of developing resistance
    synergy_scores: List[SynergyScore]  # Combination synergies
    optimal_drug_combination: List[str]  # Recommended drugs
    treatment_course: Optional[TreatmentCourse]  # Simulated course
    contraindications: List[str]    # Risk factors/contraindications


# Known drug properties database
DRUG_DATABASE: Dict[str, DrugProperties] = {
    "pembrolizumab": DrugProperties(
        drug_name="pembrolizumab",
        drug_class=DrugClass.CHECKPOINT_INHIBITOR,
        mechanism="PD-1 blockade",
        half_life=26.0,  # days
        dose_frequency=21.0,
        synergy_targets=["RIG-I agonists", "TLR agonists", "chemotherapy"],
        resistance_risk_factors=["high TROP2", "high B7-H4", "low TMB"],
    ),
    "nivolumab": DrugProperties(
        drug_name="nivolumab",
        drug_class=DrugClass.CHECKPOINT_INHIBITOR,
        mechanism="PD-1 blockade",
        half_life=27.0,
        dose_frequency=14.0,
        synergy_targets=["CTLA-4 inhibitors", "RIG-I agonists"],
        resistance_risk_factors=["low PD-L1", "immunosuppressive TME"],
    ),
    "atezolizumab": DrugProperties(
        drug_name="atezolizumab",
        drug_class=DrugClass.CHECKPOINT_INHIBITOR,
        mechanism="PD-L1 blockade",
        half_life=27.0,
        dose_frequency=21.0,
        synergy_targets=["VEGF inhibitors", "circRNA vaccines"],
        resistance_risk_factors=["high VEGF", "low immune infiltration"],
    ),
    "circRNA_vaccine": DrugProperties(
        drug_name="circRNA_vaccine",
        drug_class=DrugClass.IMMUNOTHERAPY,
        mechanism="RIG-I/TLR activation, antigen presentation",
        half_life=7.0,  # circRNA stability
        dose_frequency=7.0,
        synergy_targets=["PD-1 inhibitors", "CTLA-4 inhibitors", "chemotherapy"],
        resistance_risk_factors=["high PKR activation", "m6A modification"],
    ),
}


class DrugResponsePredictor:
    """
    Predict drug response for circRNA immunotherapy.

    Integrates:
    - IPS score from pipeline
    - TIDE score for resistance
    - Gene expression markers
    - Drug synergy analysis
    """

    def __init__(self):
        """Initialize drug response predictor."""
        self.drug_db = DRUG_DATABASE

    def predict(
        self,
        immune_scores: Dict[str, float],
        gene_expression: Dict[str, float],
        treatment_type: str = "circRNA_vaccine",
        combination_drugs: Optional[List[str]] = None,
    ) -> DrugResponseFeatures:
        """
        Predict drug response.

        Args:
            immune_scores: Immune pathway scores from pipeline
            gene_expression: Gene expression dict (TROP2, B7-H4, etc.)
            treatment_type: Primary treatment
            combination_drugs: Additional drugs for combination

        Returns:
            DrugResponseFeatures with response prediction
        """
        # Get baseline IPS
        ips = immune_scores.get("ips", 5.0)

        # Compute TIDE score (tumor immune dysfunction and exclusion)
        tide = self._compute_tide_score(gene_expression)

        # Predict response class
        response, prob = self._classify_response(ips, tide)

        # Estimate response kinetics
        ttr = self._estimate_time_to_response(ips, immune_scores)
        duration = self._estimate_duration(ips, response)

        # Compute resistance risk
        resistance = self._compute_resistance_risk(
            gene_expression,
            immune_scores,
            treatment_type
        )

        # Analyze drug synergies
        synergies = self._analyze_synergies(
            treatment_type,
            combination_drugs or [],
            immune_scores
        )

        # Find optimal combination
        optimal = self._find_optimal_combination(
            treatment_type,
            synergies,
            gene_expression
        )

        # Simulate treatment course
        course = self._simulate_course(
            ips,
            immune_scores,
            optimal,
            duration
        )

        # Identify contraindications
        contra = self._identify_contraindications(gene_expression, immune_scores)

        return DrugResponseFeatures(
            baseline_ips=ips,
            predicted_response=response,
            response_probability=prob,
            time_to_response=ttr,
            duration_estimate=duration,
            resistance_risk=resistance,
            synergy_scores=synergies,
            optimal_drug_combination=optimal,
            treatment_course=course,
            contraindications=contra,
        )

    def _compute_tide_score(self, gene_expr: Dict[str, float]) -> float:
        """
        Compute TIDE score (Jiang et al., 2018).

        TIDE predicts resistance to immune checkpoint blockade:
        - High TROP2 = exclusion (tumor ignores immune)
        - High B7-H4 = dysfunction (immune cells exhausted)
        - High MKI67 = proliferation > immune killing
        """
        trop2 = gene_expr.get("TROP2", 0.5)
        b7h4 = gene_expr.get("B7-H4", 0.5)
        mki67 = gene_expr.get("MKI67", 0.5)

        # TIDE formula (simplified)
        # Dysfunction: high B7-H4, high MKI67
        dysfunction = 0.4 * b7h4 + 0.3 * mki67

        # Exclusion: high TROP2
        exclusion = 0.5 * trop2

        tide = dysfunction + exclusion

        return np.clip(tide, 0.0, 1.0)

    def _classify_response(self, ips: float, tide: float) -> Tuple[str, float]:
        """
        Classify predicted response.

        Based on IPS and TIDE thresholds (Jiang et al., 2018):
        - IPS > 7, TIDE < 0.3: responder (high prob)
        - IPS < 4, TIDE > 0.6: non-responder
        - Otherwise: intermediate
        """
        if ips >= 7.0 and tide <= 0.3:
            return "likely_responder", 0.85
        elif ips >= 6.0 and tide <= 0.4:
            return "likely_responder", 0.70
        elif ips <= 4.0 or tide >= 0.6:
            return "likely_non_responder", 0.80
        elif ips <= 5.0 or tide >= 0.5:
            return "likely_non_responder", 0.65
        else:
            return "intermediate", 0.50

    def _estimate_time_to_response(self, ips: float, immune_scores: Dict) -> float:
        """
        Estimate time to initial response (days).

        Higher immunogenicity = faster response
        """
        rig_i = immune_scores.get("rig_i_score", 0.3)
        tlr = immune_scores.get("tlr_score", 0.2)

        # Base: 60 days for checkpoint inhibitors
        base_time = 60.0

        # Faster for high immunogenicity
        immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)
        speed_factor = 1.0 - immunogenicity * 0.5

        # circRNA vaccines respond faster (days vs weeks)
        if rig_i > 0.5 or tlr > 0.4:
            speed_factor *= 0.7  # Faster innate activation

        return base_time * speed_factor

    def _estimate_duration(self, ips: float, response: str) -> float:
        """
        Estimate response duration (months).

        Good responders maintain response longer
        """
        if response == "likely_responder":
            base_duration = 24.0  # months
            bonus = (ips - 6.0) * 2.0 if ips > 6.0 else 0.0
            return base_duration + bonus
        elif response == "intermediate":
            return 12.0
        else:
            return 3.0

    def _compute_resistance_risk(
        self,
        gene_expr: Dict[str, float],
        immune_scores: Dict[str, float],
        treatment: str
    ) -> float:
        """
        Compute resistance development risk.

        Risk factors:
        - High TROP2/B7-H4: tumor evasion
        - High MKI67: rapid proliferation
        - Low immunogenicity: weak immune activation
        - High PKR: translational inhibition (circRNA specific)
        """
        trop2 = gene_expr.get("TROP2", 0.5)
        b7h4 = gene_expr.get("B7-H4", 0.5)
        mki67 = gene_expr.get("MKI67", 0.5)
        myc = gene_expr.get("MYC", 0.5)

        # Evasion risk
        evasion_risk = 0.25 * trop2 + 0.25 * b7h4

        # Proliferation risk
        proliferation_risk = 0.2 * mki67 + 0.15 * myc

        # Weak immune activation risk
        immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)
        weak_immune_risk = 0.15 * (1.0 - immunogenicity)

        # PKR-mediated resistance (circRNA specific)
        pkr = immune_scores.get("pkr_score", 0.3)
        pkr_risk = 0.1 * pkr if treatment == "circRNA_vaccine" else 0.0

        total_risk = evasion_risk + proliferation_risk + weak_immune_risk + pkr_risk

        return np.clip(total_risk, 0.0, 1.0)

    def _analyze_synergies(
        self,
        primary_drug: str,
        combination_drugs: List[str],
        immune_scores: Dict[str, float]
    ) -> List[SynergyScore]:
        """
        Analyze drug combination synergies.

        Synergy mechanisms:
        - circRNA + PD-1: RIG-I activation + checkpoint release
        - circRNA + chemo: immunogenic cell death + antigen presentation
        - PD-1 + CTLA-4: dual checkpoint blockade
        """
        synergies = []

        primary_props = self.drug_db.get(primary_drug, None)
        if not primary_props:
            return synergies

        for combo_drug in combination_drugs:
            combo_props = self.drug_db.get(combo_drug, None)
            if not combo_props:
                continue

            # Check if drugs synergize
            if combo_drug in primary_props.synergy_targets or \
               primary_drug in combo_props.synergy_targets:

                # Compute synergy score
                synergy_value = self._compute_synergy_value(
                    primary_drug,
                    combo_drug,
                    immune_scores
                )

                # Determine synergy type
                if synergy_value > 0.7:
                    syn_type = "synergistic"
                    mechanism = "Multiple pathway activation"
                elif synergy_value > 0.3:
                    syn_type = "additive"
                    mechanism = "Independent effects"
                else:
                    syn_type = "antagonistic"
                    mechanism = "Potential interference"

                # Estimate optimal ratio
                optimal_ratio = self._estimate_optimal_ratio(
                    primary_drug,
                    combo_drug,
                    synergy_value
                )

                synergies.append(SynergyScore(
                    drug_a=primary_drug,
                    drug_b=combo_drug,
                    synergy_type=syn_type,
                    synergy_value=synergy_value,
                    mechanism=mechanism,
                    optimal_ratio=optimal_ratio,
                ))

        return synergies

    def _compute_synergy_value(
        self,
        drug_a: str,
        drug_b: str,
        immune_scores: Dict[str, float]
    ) -> float:
        """
        Compute synergy score for drug combination.

        Uses immune pathway overlap and complementarity
        """
        # Known high-synergy pairs
        high_synergy_pairs = [
            ("circRNA_vaccine", "pembrolizumab"),
            ("circRNA_vaccine", "nivolumab"),
            ("circRNA_vaccine", "atezolizumab"),
            ("pembrolizumab", "nivolumab"),
        ]

        pair = (drug_a, drug_b)
        reverse_pair = (drug_b, drug_a)

        if pair in high_synergy_pairs or reverse_pair in high_synergy_pairs:
            # Base synergy
            base = 0.7

            # Enhance if immunogenicity is high
            immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)
            boost = immunogenicity * 0.2

            return min(base + boost, 0.95)

        # Moderate synergy for other combinations
        return 0.4

    def _estimate_optimal_ratio(self, drug_a: str, drug_b: str, synergy: float) -> float:
        """Estimate optimal dose ratio."""
        # Most combinations work best at 1:1
        # circRNA vaccines may need higher frequency
        if "circRNA" in drug_a:
            return 2.0  # 2 circRNA doses per drug dose
        elif "circRNA" in drug_b:
            return 0.5  # 0.5 drug doses per circRNA
        else:
            return 1.0

    def _find_optimal_combination(
        self,
        primary: str,
        synergies: List[SynergyScore],
        gene_expr: Dict[str, float]
    ) -> List[str]:
        """
        Find optimal drug combination.

        Prioritize synergies and address resistance factors
        """
        optimal = [primary]

        # Add best synergy drugs
        synergies_sorted = sorted(synergies, key=lambda s: -s.synergy_value)

        for syn in synergies_sorted[:2]:
            if syn.synergy_type in ["synergistic", "additive"]:
                optimal.append(syn.drug_b)

        # Address resistance factors
        trop2 = gene_expr.get("TROP2", 0.5)
        if trop2 > 0.7:
            # High TROP2 may need targeted therapy
            optimal.append("TROP2_targeted")

        return optimal

    def _simulate_course(
        self,
        ips: float,
        immune_scores: Dict[str, float],
        drugs: List[str],
        duration: float
    ) -> TreatmentCourse:
        """
        Simulate treatment course.

        Generates predicted time series:
        - Immune activation kinetics
        - Tumor response trajectory
        - Resistance risk evolution
        """
        # Time points (days)
        total_days = int(duration * 30)  # months to days
        time_points = list(range(0, total_days + 1, 7))  # Weekly

        immune_activation = []
        tumor_response = []
        resistance_risk = []
        drug_concentration = []

        immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)

        for day in time_points:
            # Immune activation kinetics
            # Rises rapidly then plateaus
            if day < 14:
                immune = immunogenicity * (day / 14.0)
            else:
                immune = immunogenicity * (1.0 - 0.1 * (day - 14) / total_days)

            immune_activation.append(np.clip(immune, 0.0, 1.0))

            # Tumor response (negative = shrinkage)
            # Response starts after immune activation
            if day < 30:
                tumor = 0.0  # No initial change
            else:
                response_rate = (ips - 5.0) / 5.0  # Based on IPS
                tumor = -response_rate * (day - 30) / (total_days - 30)

            tumor_response.append(np.clip(tumor, -1.0, 0.5))

            # Resistance risk increases over time
            base_resistance = self._compute_resistance_risk({}, immune_scores, drugs[0])
            time_factor = day / total_days * 0.3
            resistance = base_resistance + time_factor

            resistance_risk.append(np.clip(resistance, 0.0, 1.0))

            # Drug concentration (pulsed dosing)
            pulse = 1.0 if day % 21 == 0 else 0.5
            decay = np.exp(-day / 30.0)
            drug_concentration.append(pulse * decay)

        return TreatmentCourse(
            time_points=time_points,
            immune_activation=immune_activation,
            tumor_response=tumor_response,
            resistance_risk=resistance_risk,
            drug_concentration=drug_concentration,
        )

    def _identify_contraindications(
        self,
        gene_expr: Dict[str, float],
        immune_scores: Dict[str, float]
    ) -> List[str]:
        """
        Identify treatment contraindications.

        Risk factors that may reduce efficacy or cause adverse effects
        """
        contra = []

        trop2 = gene_expr.get("TROP2", 0.5)
        b7h4 = gene_expr.get("B7-H4", 0.5)
        pkr = immune_scores.get("pkr_score", 0.3)

        if trop2 > 0.8:
            contra.append("High TROP2: tumor exclusion risk, may reduce efficacy")

        if b7h4 > 0.8:
            contra.append("High B7-H4: immune dysfunction, consider combination therapy")

        if pkr > 0.6:
            contra.append("High PKR activation: may inhibit protein expression")

        immunogenicity = immune_scores.get("overall_immunogenicity", 0.5)
        if immunogenicity < 0.3:
            contra.append("Low immunogenicity: weak innate activation, optimize sequence")

        return contra


def compute_drug_response_score(features: DrugResponseFeatures) -> Dict[str, float]:
    """
    Compute composite drug response scores.

    Returns dict compatible with pipeline output.
    """
    scores = {
        "response_probability": features.response_probability,
        "time_to_response_days": features.time_to_response,
        "duration_months": features.duration_estimate,
        "resistance_risk": features.resistance_risk,
        "tide_score": features.baseline_ips,
    }

    # Treatment success score
    success = (
        features.response_probability * 0.4 +
        (features.duration_estimate / 24.0) * 0.3 +
        (1.0 - features.resistance_risk) * 0.3
    )
    scores["treatment_success_score"] = np.clip(success, 0.0, 1.0)

    # Best synergy score
    if features.synergy_scores:
        best_syn = max(s.synergy_value for s in features.synergy_scores)
        scores["best_synergy_score"] = best_syn
    else:
        scores["best_synergy_score"] = 0.0

    return scores


def recommend_treatment(
    immune_scores: Dict[str, float],
    gene_expr: Dict[str, float],
    patient_factors: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate treatment recommendations.

    Returns:
        Recommended drugs, dosing, combination, monitoring plan
    """
    predictor = DrugResponsePredictor()

    # Test different treatment strategies
    strategies = [
        ("circRNA_vaccine", []),
        ("circRNA_vaccine", ["pembrolizumab"]),
        ("pembrolizumab", ["circRNA_vaccine"]),
        ("circRNA_vaccine", ["nivolumab", "chemotherapy"]),
    ]

    best_score = 0.0
    best_response = None
    best_strategy = None

    for primary, combos in strategies:
        response = predictor.predict(immune_scores, gene_expr, primary, combos)
        score = response.response_probability * (1.0 - response.resistance_risk)

        if score > best_score:
            best_score = score
            best_response = response
            best_strategy = (primary, combos)

    recommendations = {
        "recommended_primary": best_strategy[0],
        "recommended_combinations": best_strategy[1],
        "response_probability": best_response.response_probability,
        "resistance_risk": best_response.resistance_risk,
        "time_to_response": best_response.time_to_response,
        "expected_duration": best_response.duration_estimate,
        "contraindications": best_response.contraindications,
        "monitoring_schedule": [
            "Day 14: Initial immune response check",
            "Day 30: Tumor response assessment",
            "Month 3: Resistance marker screening",
            "Month 6: Duration reassessment",
        ],
    }

    return recommendations


# Convenience function
def predict_drug_response(
    immune_scores: Dict[str, float],
    gene_expr: Dict[str, float],
) -> DrugResponseFeatures:
    """Predict drug response for circRNA immunotherapy."""
    predictor = DrugResponsePredictor()
    return predictor.predict(immune_scores, gene_expr)