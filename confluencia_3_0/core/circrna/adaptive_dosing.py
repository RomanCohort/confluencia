"""
Adaptive Dosing Strategy for circRNA Immunotherapy

Implements closed-loop adaptive dosing that:
1. Monitors patient response in real-time
2. Adjusts doses based on efficacy/toxicity signals
3. Switches combinations if response inadequate
4. Optimizes for individual patient dynamics

Literature basis:
- Bates & Planta, 2018: Adaptive dose optimization
- Wei et al., 2021: Model-informed precision dosing
- Lin et al., 2022: Adaptive immunotherapy dosing
- Feng et al., 2023: Real-time dose adjustment for oncology

Key concepts:
- Therapeutic window: Balance efficacy vs toxicity
- Biomarker-driven adjustment: Use real-time markers
- Adaptive switching: Change strategy if non-response
- Bayesian updating: Learn from patient response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.stats import norm


# =====================================================================
# Adaptive Dosing Definitions
# =====================================================================

@dataclass
class DoseConstraints:
    """Safety constraints for dosing."""
    min_dose: float
    max_dose: float
    min_interval_days: float
    max_interval_days: float

    # Toxicity thresholds
    grade3_toxicity_threshold: float = 0.15
    grade4_toxicity_threshold: float = 0.05

    # Response thresholds
    partial_response_threshold: float = 0.30  # 30% tumor reduction
    progressive_disease_threshold: float = 0.20  # 20% increase


@dataclass
class MonitoringPoint:
    """A single monitoring timepoint."""
    time_days: float
    tumor_size: Optional[float] = None
    tumor_change_pct: Optional[float] = None

    # Immune markers
    cd8_count: Optional[float] = None
    treg_ratio: Optional[float] = None
    pd1_expression: Optional[float] = None
    antibody_titer: Optional[float] = None

    # Toxicity
    toxicity_grade: int = 0
    adverse_events: List[str] = field(default_factory=list)

    # Biomarkers
    biomarkers: Dict[str, float] = field(default_factory=dict)


@dataclass
class DoseAdjustment:
    """A dose adjustment decision."""
    time_days: float
    drug_name: str
    previous_dose: float
    new_dose: float
    previous_interval: float
    new_interval: float
    reason: str
    confidence: float

    # Predicted outcomes
    predicted_efficacy_change: float
    predicted_toxicity_change: float


@dataclass
class AdaptiveStrategy:
    """Complete adaptive dosing strategy."""
    drug_name: str
    initial_dose: float
    initial_interval: float
    constraints: DoseConstraints

    # Adjustment rules
    dose_adjustment_rules: Dict[str, Tuple[float, str]]  # condition -> (factor, reason)
    switch_triggers: Dict[str, str]  # condition -> alternative_drug

    # Monitoring requirements
    monitoring_frequency_days: float
    key_biomarkers: List[str]


# Predefined strategies
ADAPTIVE_STRATEGIES: Dict[str, AdaptiveStrategy] = {
    "circrna_vaccine": AdaptiveStrategy(
        drug_name="circrna_vaccine",
        initial_dose=2.0,
        initial_interval=7.0,
        constraints=DoseConstraints(
            min_dose=0.5,
            max_dose=5.0,
            min_interval_days=3.0,
            max_interval_days=14.0,
        ),
        dose_adjustment_rules={
            "response_no_change_14d": (1.2, "Increase dose - no early response"),
            "response_partial": (1.0, "Maintain dose - partial response"),
            "response_good": (0.9, "Reduce dose - good response"),
            "toxicity_grade2": (0.8, "Reduce dose - grade 2 toxicity"),
            "toxicity_grade3": (0.5, "Halve dose - grade 3 toxicity"),
            "cd8_increase_>50%": (1.1, "Boost dose - immune activation"),
            "antibody_high": (0.85, "Reduce dose - high antibody titer"),
        },
        switch_triggers={
            "no_response_4wks": "circrna_vaccine + pembrolizumab",
            "progressive_disease": "alternative_combination",
            "severe_toxicity": "dose_hold",
        },
        monitoring_frequency_days=7.0,
        key_biomarkers=["cd8_count", "antibody_titer", "pd1_expression", "toxicity_grade"],
    ),
    "pembrolizumab": AdaptiveStrategy(
        drug_name="pembrolizumab",
        initial_dose=200.0,  # mg
        initial_interval=21.0,  # days
        constraints=DoseConstraints(
            min_dose=100.0,
            max_dose=400.0,
            min_interval_days=14.0,
            max_interval_days=28.0,
        ),
        dose_adjustment_rules={
            "response_good": (1.0, "Maintain standard dosing"),
            "toxicity_grade2": (1.0, "Maintain - close monitoring"),
            "toxicity_grade3": (0.0, "Hold dose - grade 3 irAE"),
            "progressive_disease": (0.0, "Consider combination therapy"),
            "pd1_high_expression": (1.1, "May increase - target engagement"),
        },
        switch_triggers={
            "progressive_disease_12wks": "add_circrna_vaccine",
            "grade3_irAE": "dose_hold_rechallenge",
            "grade4_irAE": "permanent_discontinue",
        },
        monitoring_frequency_days=21.0,
        key_biomarkers=["pd1_expression", "treg_ratio", "toxicity_grade"],
    ),
    "combination": AdaptiveStrategy(
        drug_name="circrna_vaccine + pembrolizumab",
        initial_dose=2.0,  # circRNA dose
        initial_interval=7.0,  # circRNA interval (pembrolizumab fixed)
        constraints=DoseConstraints(
            min_dose=0.5,
            max_dose=5.0,
            min_interval_days=3.0,
            max_interval_days=14.0,
        ),
        dose_adjustment_rules={
            "response_synergistic": (0.9, "Reduce circRNA - synergy working"),
            "response_no_change": (1.3, "Increase circRNA - prime immune"),
            "cd8_increase_>100%": (0.85, "Reduce circRNA - strong activation"),
            "toxicity_grade2": (0.75, "Reduce circRNA - manage toxicity"),
        },
        switch_triggers={
            "no_response_6wks": "add_cyclophosphamide",
            "toxicity_unmanageable": "reduce_to_monotherapy",
        },
        monitoring_frequency_days=7.0,
        key_biomarkers=["cd8_count", "antibody_titer", "tumor_change", "toxicity_grade"],
    ),
}


# =====================================================================
# Adaptive Dosing Engine
# =====================================================================

class AdaptiveDosingEngine:
    """
    Implements real-time adaptive dosing.

    Uses:
    1. Bayesian updating of patient response model
    2. Safety-constrained optimization
    3. Rule-based adjustments
    4. Predictive modeling for next dose
    """

    def __init__(
        self,
        strategy: Optional[AdaptiveStrategy] = None,
        strategy_name: str = "circrna_vaccine",
    ):
        self.strategy = strategy or ADAPTIVE_STRATEGIES.get(strategy_name, ADAPTIVE_STRATEGIES["circrna_vaccine"])
        self.current_dose = self.strategy.initial_dose
        self.current_interval = self.strategy.initial_interval

        # History tracking
        self.monitoring_history: List[MonitoringPoint] = []
        self.adjustment_history: List[DoseAdjustment] = []

        # Bayesian prior (efficacy and toxicity parameters)
        self.prior_efficacy_mean = 0.5
        self.prior_efficacy_var = 0.1
        self.prior_toxicity_mean = 0.1
        self.prior_toxicity_var = 0.05

        # Response model parameters (learned from patient)
        self.response_model = {
            "baseline_tumor": None,
            "tumor_trajectory": [],
            "cd8_trajectory": [],
            "antibody_trajectory": [],
        }

    def update(
        self,
        monitoring_point: MonitoringPoint,
    ) -> DoseAdjustment:
        """
        Update dosing based on new monitoring data.

        Args:
            monitoring_point: New monitoring data

        Returns:
            DoseAdjustment decision
        """
        # Add to history
        self.monitoring_history.append(monitoring_point)

        # Update response model
        self._update_response_model(monitoring_point)

        # Assess current state
        state = self._assess_state(monitoring_point)

        # Determine adjustment
        dose_factor, reason, confidence = self._determine_adjustment(state, monitoring_point)

        # Calculate new dose
        new_dose = np.clip(
            self.current_dose * dose_factor,
            self.strategy.constraints.min_dose,
            self.strategy.constraints.max_dose,
        )

        # Adjust interval if needed
        new_interval = self.current_interval
        if state.get("dose_interval_adjustment"):
            interval_factor = state["dose_interval_adjustment"]
            new_interval = np.clip(
                self.current_interval * interval_factor,
                self.strategy.constraints.min_interval_days,
                self.strategy.constraints.max_interval_days,
            )

        # Create adjustment record
        adjustment = DoseAdjustment(
            time_days=monitoring_point.time_days,
            drug_name=self.strategy.drug_name,
            previous_dose=self.current_dose,
            new_dose=new_dose,
            previous_interval=self.current_interval,
            new_interval=new_interval,
            reason=reason,
            confidence=confidence,
            predicted_efficacy_change=self._predict_efficacy_change(dose_factor),
            predicted_toxicity_change=self._predict_toxicity_change(dose_factor),
        )

        # Update current dose
        self.current_dose = new_dose
        self.current_interval = new_interval
        self.adjustment_history.append(adjustment)

        return adjustment

    def _update_response_model(self, point: MonitoringPoint):
        """Update internal response model."""
        if self.response_model["baseline_tumor"] is None:
            self.response_model["baseline_tumor"] = point.tumor_size

        if point.tumor_size is not None:
            self.response_model["tumor_trajectory"].append(point.tumor_size)

        if point.cd8_count is not None:
            self.response_model["cd8_trajectory"].append(point.cd8_count)

        if point.antibody_titer is not None:
            self.response_model["antibody_trajectory"].append(point.antibody_titer)

        # Bayesian update
        self._bayesian_update(point)

    def _bayesian_update(self, point: MonitoringPoint):
        """Update Bayesian prior with new observation."""
        # Efficacy signal: tumor reduction
        if point.tumor_change_pct is not None:
            # Negative change = tumor shrinkage = good efficacy
            efficacy_signal = -point.tumor_change_pct / 100.0

            # Update efficacy posterior
            prior_var = self.prior_efficacy_var
            likelihood_var = 0.05  # Observation noise

            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
            posterior_mean = posterior_var * (
                self.prior_efficacy_mean / prior_var +
                efficacy_signal / likelihood_var
            )

            self.prior_efficacy_mean = posterior_mean
            self.prior_efficacy_var = posterior_var

        # Toxicity signal
        if point.toxicity_grade > 0:
            toxicity_signal = point.toxicity_grade / 5.0

            prior_var = self.prior_toxicity_var
            likelihood_var = 0.1

            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
            posterior_mean = posterior_var * (
                self.prior_toxicity_mean / prior_var +
                toxicity_signal / likelihood_var
            )

            self.prior_toxicity_mean = posterior_mean
            self.prior_toxicity_var = posterior_var

    def _assess_state(self, point: MonitoringPoint) -> Dict[str, Any]:
        """Assess current patient state."""
        state = {}

        # Response category
        if point.tumor_change_pct is not None:
            if point.tumor_change_pct <= -self.strategy.constraints.partial_response_threshold * 100:
                state["response"] = "good"
            elif point.tumor_change_pct >= self.strategy.constraints.progressive_disease_threshold * 100:
                state["response"] = "progressive_disease"
            elif -15 <= point.tumor_change_pct < 0:
                state["response"] = "partial"
            else:
                state["response"] = "no_change"

        # Toxicity category
        state["toxicity_grade"] = point.toxicity_grade
        if point.toxicity_grade >= 3:
            state["toxicity"] = "severe"
        elif point.toxicity_grade == 2:
            state["toxicity"] = "moderate"
        else:
            state["toxicity"] = "acceptable"

        # Immune activation
        if len(self.response_model["cd8_trajectory"]) >= 2:
            cd8_change = (
                self.response_model["cd8_trajectory"][-1] /
                self.response_model["cd8_trajectory"][-2] - 1.0
            )
            if cd8_change > 0.5:
                state["cd8_increase"] = ">50%"
            elif cd8_change > 0.2:
                state["cd8_increase"] = ">20%"

        # Antibody response
        if point.antibody_titer is not None:
            if point.antibody_titer > 100:
                state["antibody"] = "high"

        # Duration-based rules
        if len(self.monitoring_history) >= 2:
            time_on_treatment = point.time_days - self.monitoring_history[0].time_days
            if time_on_treatment >= 14 and state.get("response") == "no_change":
                state["response_no_change_14d"] = True
            if time_on_treatment >= 28 and state.get("response") in ["no_change", "partial"]:
                state["no_response_4wks"] = True

        return state

    def _determine_adjustment(
        self,
        state: Dict[str, Any],
        point: MonitoringPoint,
    ) -> Tuple[float, str, float]:
        """Determine dose adjustment factor."""
        rules = self.strategy.dose_adjustment_rules

        matched_rules = []

        # Check each rule condition
        for condition, (factor, reason) in rules.items():
            if self._check_condition(condition, state, point):
                matched_rules.append((condition, factor, reason))

        if not matched_rules:
            return 1.0, "Maintain current dose", 0.8

        # Prioritize rules (toxicity > response > biomarker)
        priority_order = ["toxicity", "response", "cd8", "antibody"]

        for priority in priority_order:
            for condition, factor, reason in matched_rules:
                if priority in condition:
                    return factor, reason, 0.85

        # Default: use first matched rule
        _, factor, reason = matched_rules[0]
        return factor, reason, 0.75

    def _check_condition(self, condition: str, state: Dict, point: MonitoringPoint) -> bool:
        """Check if a condition is met."""
        # Parse condition string
        if condition == "response_no_change_14d":
            return state.get("response_no_change_14d", False)
        elif condition == "response_partial":
            return state.get("response") == "partial"
        elif condition == "response_good":
            return state.get("response") == "good"
        elif condition == "toxicity_grade2":
            return point.toxicity_grade == 2
        elif condition == "toxicity_grade3":
            return point.toxicity_grade == 3
        elif condition.startswith("cd8_increase"):
            if ">50%" in condition:
                return state.get("cd8_increase") == ">50%"
            elif ">100%" in condition:
                return state.get("cd8_increase", "0%").replace(">", "").strip("%") and float(state.get("cd8_increase", "0%").replace(">", "").strip("%")) > 100
        elif condition == "antibody_high":
            return state.get("antibody") == "high"
        elif condition == "response_synergistic":
            # Check for synergy indicators
            return state.get("response") == "good" and state.get("cd8_increase", "0%") in [">50%", ">20%"]

        return False

    def _predict_efficacy_change(self, dose_factor: float) -> float:
        """Predict efficacy change from dose adjustment."""
        # Simple model: efficacy ~ dose^0.5 (diminishing returns)
        return np.sqrt(dose_factor) - 1.0

    def _predict_toxicity_change(self, dose_factor: float) -> float:
        """Predict toxicity change from dose adjustment."""
        # Toxicity ~ dose (linear)
        return dose_factor - 1.0

    def check_switch_trigger(self) -> Optional[str]:
        """Check if therapy switch is needed."""
        if not self.monitoring_history:
            return None

        latest = self.monitoring_history[-1]
        state = self._assess_state(latest)

        for trigger, action in self.strategy.switch_triggers.items():
            if trigger == "no_response_4wks" and state.get("no_response_4wks"):
                return action
            elif trigger == "progressive_disease" and state.get("response") == "progressive_disease":
                return action
            elif trigger == "severe_toxicity" and latest.toxicity_grade >= 3:
                return action
            elif trigger == "grade3_irAE" and latest.toxicity_grade == 3:
                return action

        return None

    def get_optimal_dose(self) -> Tuple[float, float]:
        """
        Get current optimal dose based on learned model.

        Returns:
            (optimal_dose, confidence)
        """
        if len(self.monitoring_history) < 2:
            return self.strategy.initial_dose, 0.5

        # Optimize dose based on efficacy/toxicity balance
        def objective(dose):
            # Predicted efficacy
            efficacy = self.prior_efficacy_mean * np.sqrt(dose / self.strategy.initial_dose)

            # Predicted toxicity
            toxicity = self.prior_toxicity_mean * (dose / self.strategy.initial_dose)

            # Objective: maximize efficacy - penalty for toxicity
            return -efficacy + 0.5 * toxicity

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            objective,
            bounds=(self.strategy.constraints.min_dose, self.strategy.constraints.max_dose),
            method='bounded',
        )

        optimal_dose = result.x
        confidence = 1.0 / (1.0 + self.prior_efficacy_var + self.prior_toxicity_var)

        return optimal_dose, confidence

    def simulate_trajectory(
        self,
        n_weeks: int = 12,
        true_response_type: str = "responsive",
    ) -> pd.DataFrame:
        """
        Simulate treatment trajectory for testing.

        Args:
            n_weeks: Simulation duration
            true_response_type: "responsive", "partial", "resistant"

        Returns:
            DataFrame with simulated trajectory
        """
        np.random.seed(42)

        records = []
        baseline_tumor = 100.0

        for week in range(n_weeks + 1):
            time_days = week * 7

            # Simulate tumor response
            if true_response_type == "responsive":
                tumor_change = -5.0 * week + np.random.normal(0, 2)
            elif true_response_type == "partial":
                tumor_change = -2.0 * week + np.random.normal(0, 2)
            else:  # resistant
                tumor_change = 3.0 * week + np.random.normal(0, 3)

            tumor_size = baseline_tumor + tumor_change
            tumor_change_pct = tumor_change / baseline_tumor * 100

            # Simulate immune markers
            cd8_increase = 1.5 if true_response_type == "responsive" else 1.1
            cd8_count = 100 * (cd8_increase ** (week / 4)) + np.random.normal(0, 5)

            antibody_titer = 20 * (1.5 ** (week / 2)) if true_response_type != "resistant" else 10

            # Simulate toxicity
            toxicity_grade = np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05])

            # Create monitoring point
            point = MonitoringPoint(
                time_days=time_days,
                tumor_size=max(0, tumor_size),
                tumor_change_pct=tumor_change_pct,
                cd8_count=cd8_count,
                antibody_titer=antibody_titer,
                toxicity_grade=toxicity_grade,
            )

            # Get dose adjustment
            adjustment = self.update(point)

            records.append({
                "week": week,
                "time_days": time_days,
                "tumor_size": tumor_size,
                "tumor_change_pct": tumor_change_pct,
                "cd8_count": cd8_count,
                "antibody_titer": antibody_titer,
                "toxicity_grade": toxicity_grade,
                "dose": adjustment.new_dose,
                "interval": adjustment.new_interval,
                "adjustment_reason": adjustment.reason,
            })

        return pd.DataFrame(records)


# =====================================================================
# Convenience Functions
# =====================================================================

def create_adaptive_dosing(
    drug_name: str = "circrna_vaccine",
) -> AdaptiveDosingEngine:
    """Create adaptive dosing engine for a drug."""
    return AdaptiveDosingEngine(strategy_name=drug_name)


def run_adaptive_simulation(
    drug_name: str = "circrna_vaccine",
    response_type: str = "responsive",
    n_weeks: int = 12,
) -> Tuple[pd.DataFrame, AdaptiveDosingEngine]:
    """Run simulation with adaptive dosing."""
    engine = AdaptiveDosingEngine(strategy_name=drug_name)
    trajectory = engine.simulate_trajectory(n_weeks=n_weeks, true_response_type=response_type)
    return trajectory, engine


def get_dose_recommendation(
    current_dose: float,
    tumor_change_pct: float,
    toxicity_grade: int,
    cd8_change_pct: float = 0.0,
    drug_name: str = "circrna_vaccine",
) -> Tuple[float, str]:
    """Quick dose recommendation based on current state."""
    engine = AdaptiveDosingEngine(strategy_name=drug_name)

    point = MonitoringPoint(
        time_days=14.0,  # Assume 2 weeks in
        tumor_change_pct=tumor_change_pct,
        toxicity_grade=toxicity_grade,
        cd8_count=100 * (1 + cd8_change_pct / 100),
    )

    engine.current_dose = current_dose
    adjustment = engine.update(point)

    return adjustment.new_dose, adjustment.reason


__all__ = [
    "DoseConstraints",
    "MonitoringPoint",
    "DoseAdjustment",
    "AdaptiveStrategy",
    "ADAPTIVE_STRATEGIES",
    "AdaptiveDosingEngine",
    "create_adaptive_dosing",
    "run_adaptive_simulation",
    "get_dose_recommendation",
]