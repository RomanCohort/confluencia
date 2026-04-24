"""
Three-dimensional scoring engine for joint Drug-Epitope-PK evaluation.

Produces ClinicalScore, BindingScore, KineticsScore, and a composite
JointScore from drug pipeline outputs, epitope pipeline outputs, and
PK simulation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual score dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClinicalScore:
    """Clinical-level score from the drug pipeline.

    Attributes
    ----------
    efficacy : float
        Predicted drug efficacy (0-1).
    target_binding : float
        Predicted target binding strength (0-1).
    immune_activation : float
        Predicted immune activation level (0-1).
    safety_penalty : float
        Penalty from toxicity/inflammation (0-1, lower = safer).
    overall : float
        Weighted clinical score (0-1).
    interpretation : str
        Human-readable interpretation.
    """
    efficacy: float
    target_binding: float
    immune_activation: float
    safety_penalty: float
    overall: float
    interpretation: str


@dataclass
class BindingScore:
    """MHC-epitope binding score from the epitope pipeline.

    Attributes
    ----------
    epitope_efficacy : float
        Predicted epitope efficacy (0-1).
    uncertainty : float
        Prediction uncertainty (0-1, lower = more confident).
    mhc_affinity_class : str
        Binding affinity class:
        - "strong_binder"   (efficacy >= 0.80)
        - "moderate_binder" (efficacy >= 0.50)
        - "weak_binder"     (efficacy >= 0.30)
        - "non_binder"      (efficacy < 0.30)
    overall : float
        Adjusted binding score accounting for uncertainty.
    interpretation : str
        Human-readable interpretation.
    """
    epitope_efficacy: float
    uncertainty: float
    mhc_affinity_class: str
    overall: float
    interpretation: str


@dataclass
class KineticsScore:
    """Pharmacokinetic / pharmacodynamic score from PK simulation.

    Attributes
    ----------
    cmax : float
        Peak plasma concentration (mg/L).
    tmax : float
        Time to peak concentration (hours).
    half_life : float
        Terminal half-life (hours).
    auc_conc : float
        AUC of concentration curve (mg/L·h).
    auc_effect : float
        AUC of PD effect curve (effect·h).
    therapeutic_index : float
        Ratio auc_effect / auc_conc (higher = better).
    overall : float
        Normalized kinetics score (0-1).
    interpretation : str
        Human-readable interpretation.
    """
    cmax: float
    tmax: float
    half_life: float
    auc_conc: float
    auc_effect: float
    therapeutic_index: float
    overall: float
    interpretation: str


@dataclass
class JointScore:
    """Complete joint score combining all three dimensions.

    Attributes
    ----------
    clinical : ClinicalScore
        Clinical-level assessment.
    binding : BindingScore
        MHC-epitope binding assessment.
    kinetics : KineticsScore
        PK/PD kinetics assessment.
    composite : float
        Weighted composite score (0-1).
    recommendation : str
        One of "Go", "Conditional", or "No-Go".
    recommendation_reason : str
        Brief explanation of the recommendation.
    effective_weights : dict
        Actual weights used after uncertainty adaptation.
        Keys: "clinical", "binding", "kinetics".
    uncertainties : dict
        Per-dimension uncertainty used for adaptive weighting.
        Keys: "clinical", "binding", "kinetics".
        Values in [0, 1], higher = less confident.
    """
    clinical: ClinicalScore
    binding: BindingScore
    kinetics: KineticsScore
    composite: float
    recommendation: str
    recommendation_reason: str
    effective_weights: Dict[str, float] = None
    uncertainties: Dict[str, float] = None


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

@dataclass
class JointScoringEngine:
    """Three-dimensional scoring engine.

    Combines drug efficacy, MHC binding, and PK kinetics into a unified
    composite score and recommendation.

    Parameters
    ----------
    clinical_weight : float (default 0.40)
        Weight for clinical score in composite.
    binding_weight : float (default 0.35)
        Weight for binding score in composite.
    kinetics_weight : float (default 0.25)
        Weight for kinetics score in composite.
    go_threshold : float (default 0.65)
        Composite score threshold for "Go" recommendation.
    conditional_threshold : float (default 0.40)
        Composite score threshold for "Conditional" recommendation.
        Below this → "No-Go".
    safety_floor : float (default 0.30)
        If clinical.safety_penalty > this value, force "No-Go".

    Examples
    --------
    >>> engine = JointScoringEngine()
    >>> # drug_out from drug pipeline, epi_out from epitope pipeline,
    >>> # pk_summary from PK simulation
    >>> score = engine.score(drug_out, epi_out, pk_summary)
    >>> print(score.recommendation, score.composite)
    """

    clinical_weight: float = 0.40
    binding_weight: float = 0.35
    kinetics_weight: float = 0.25
    go_threshold: float = 0.65
    conditional_threshold: float = 0.40
    safety_floor: float = 0.30

    def __post_init__(self):
        total = self.clinical_weight + self.binding_weight + self.kinetics_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}"
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score(
        self,
        drug_outputs: Dict[str, float],
        epitope_outputs: Dict[str, float],
        pk_summary: Dict[str, float],
    ) -> JointScore:
        """Compute the complete joint score.

        Parameters
        ----------
        drug_outputs : dict
            Output from drug pipeline run_pipeline(). Keys include:
            - efficacy_pred
            - target_binding_pred
            - immune_activation_pred
            - inflammation_risk_pred (optional)
            - genotoxicity_risk_pred (optional)
        epitope_outputs : dict
            Output from epitope pipeline. Keys include:
            - efficacy_pred
            - pred_uncertainty
        pk_summary : dict
            Output from pkpd.summarize_pkpd_curve(). Keys include:
            - pkpd_cmax_mg_per_l
            - pkpd_tmax_h
            - pkpd_half_life_h
            - pkpd_auc_conc
            - pkpd_auc_effect

        Returns
        -------
        JointScore
            Complete joint scoring result.
        """
        clinical = self._score_clinical(drug_outputs)
        binding = self._score_binding(epitope_outputs)
        kinetics = self._score_kinetics(pk_summary)

        # --- Uncertainty-adaptive dynamic weights ---
        uncertainties = {
            "clinical": self._uncertainty_clinical(drug_outputs),
            "binding": self._uncertainty_binding(epitope_outputs),
            "kinetics": self._uncertainty_kinetics(pk_summary),
        }
        base = {
            "clinical": self.clinical_weight,
            "binding": self.binding_weight,
            "kinetics": self.kinetics_weight,
        }
        effective = self._adaptive_weights(base, uncertainties)

        # Weighted composite with adaptive weights
        composite = (
            effective["clinical"] * clinical.overall
            + effective["binding"] * binding.overall
            + effective["kinetics"] * kinetics.overall
        )

        # Recommendation logic
        rec, reason = self._recommend(
            composite, clinical.safety_penalty, effective
        )

        return JointScore(
            clinical=clinical,
            binding=binding,
            kinetics=kinetics,
            composite=composite,
            recommendation=rec,
            recommendation_reason=reason,
            effective_weights=effective,
            uncertainties=uncertainties,
        )

    # ------------------------------------------------------------------
    # Per-dimension scoring
    # ------------------------------------------------------------------

    def _score_clinical(
        self, drug: Dict[str, float]
    ) -> ClinicalScore:
        """Compute clinical score from drug pipeline outputs."""
        eff = self._clamp(drug.get("efficacy_pred", np.nan))
        bind = self._clamp(drug.get("target_binding_pred", np.nan))
        immune = self._clamp(drug.get("immune_activation_pred", np.nan))
        infl = self._clamp(drug.get("inflammation_risk_pred", 0.0))
        tox = self._clamp(drug.get("genotoxicity_risk_pred", 0.0))

        # Safety penalty: weighted sum of risk signals
        # Lower penalty = safer (higher safety)
        safety_penalty = 0.5 * tox + 0.3 * infl + 0.2 * (1 - immune)
        safety_penalty = max(0.0, min(1.0, safety_penalty))

        # Overall clinical score
        # Weighted combination, adjusted for safety
        overall = (
            0.35 * eff
            + 0.30 * bind
            + 0.20 * immune
            + 0.15 * (1 - safety_penalty)
        )
        overall = max(0.0, min(1.0, overall))

        interpretation = self._interpret_clinical(eff, bind, immune, safety_penalty)

        return ClinicalScore(
            efficacy=eff,
            target_binding=bind,
            immune_activation=immune,
            safety_penalty=safety_penalty,
            overall=overall,
            interpretation=interpretation,
        )

    def _score_binding(
        self, epitope: Dict[str, float]
    ) -> BindingScore:
        """Compute binding score from epitope pipeline outputs."""
        eff = self._clamp(epitope.get("efficacy_pred", np.nan))
        unc = self._clamp(epitope.get("pred_uncertainty", 0.0))

        # MHC affinity classification
        if eff >= 0.80:
            affinity = "strong_binder"
        elif eff >= 0.50:
            affinity = "moderate_binder"
        elif eff >= 0.30:
            affinity = "weak_binder"
        else:
            affinity = "non_binder"

        # Adjusted score: penalize high uncertainty
        overall = eff * (1 - 0.3 * unc)
        overall = max(0.0, min(1.0, overall))

        interpretation = self._interpret_binding(eff, unc, affinity)

        return BindingScore(
            epitope_efficacy=eff,
            uncertainty=unc,
            mhc_affinity_class=affinity,
            overall=overall,
            interpretation=interpretation,
        )

    def _score_kinetics(
        self, pk: Dict[str, float]
    ) -> KineticsScore:
        """Compute kinetics score from PK simulation summary."""
        cmax = float(pk.get("pkpd_cmax_mg_per_l", np.nan))
        tmax = float(pk.get("pkpd_tmax_h", np.nan))
        hl = float(pk.get("pkpd_half_life_h", np.nan))
        auc_conc = float(pk.get("pkpd_auc_conc", np.nan))
        auc_effect = float(pk.get("pkpd_auc_effect", np.nan))

        # Therapeutic index (avoid division by zero)
        if auc_conc > 0:
            ti = auc_effect / auc_conc
        else:
            ti = 0.0

        # Normalize overall kinetics score
        # - Half-life: ideal range 4-24h; outside penalized
        # - AUC conc: moderate is best (too high = toxicity risk)
        # - Therapeutic index: higher is better
        # - Cmax: moderate is best
        half_life_score = self._score_half_life(hl)
        auc_score = self._score_auc_conc(auc_conc)
        ti_score = self._score_therapeutic_index(ti)
        cmax_score = self._score_cmax(cmax)

        overall = 0.25 * half_life_score + 0.30 * auc_score + 0.30 * ti_score + 0.15 * cmax_score
        overall = max(0.0, min(1.0, overall))

        interpretation = self._interpret_kinetics(cmax, tmax, hl, ti)

        return KineticsScore(
            cmax=cmax,
            tmax=tmax,
            half_life=hl,
            auc_conc=auc_conc,
            auc_effect=auc_effect,
            therapeutic_index=ti,
            overall=overall,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    def _recommend(
        self, composite: float, safety_penalty: float,
        effective_weights: Dict[str, float] = None
    ) -> tuple[str, str]:
        """Determine recommendation from composite score and safety."""
        # Safety override: unsafe drugs always No-Go
        if safety_penalty > self.safety_floor:
            return (
                "No-Go",
                f"Low safety (penalty={safety_penalty:.2f} > floor={self.safety_floor:.2f})"
            )

        # Weight shift warning
        weight_msg = ""
        if effective_weights:
            shifts = []
            if abs(effective_weights["binding"] - 0.35) > 0.05:
                shifts.append(f"binding {effective_weights['binding']:.2f}")
            if abs(effective_weights["kinetics"] - 0.25) > 0.05:
                shifts.append(f"kinetics {effective_weights['kinetics']:.2f}")
            if shifts:
                weight_msg = f" [weights adjusted: {', '.join(shifts)}]"

        if composite >= self.go_threshold:
            return (
                "Go",
                f"Strong composite score ({composite:.3f} ≥ {self.go_threshold:.2f}){weight_msg}"
            )
        elif composite >= self.conditional_threshold:
            return (
                "Conditional",
                f"Moderate composite score ({composite:.3f} in [{self.conditional_threshold:.2f}, {self.go_threshold:.2f}]){weight_msg}"
            )
        else:
            return (
                "No-Go",
                f"Weak composite score ({composite:.3f} < {self.conditional_threshold:.2f})"
            )

    # ------------------------------------------------------------------
    # Uncertainty-adaptive dynamic weighting
    # ------------------------------------------------------------------

    def _uncertainty_clinical(self, drug: Dict[str, float]) -> float:
        """Compute uncertainty for the clinical dimension.

        Uncertainty is derived from missing predictions and high risk signals:
        - Missing efficacy_pred → high uncertainty
        - High toxicity/inflammation → moderate uncertainty (risk signal)
        - All predictions present → low uncertainty
        """
        has_eff = not np.isnan(drug.get("efficacy_pred", np.nan))
        has_bind = not np.isnan(drug.get("target_binding_pred", np.nan))
        has_immune = not np.isnan(drug.get("immune_activation_pred", np.nan))

        # Fraction of missing predictions
        present = sum([has_eff, has_bind, has_immune])
        missing_unc = 1.0 - present / 3.0

        # Risk signal uncertainty from inflammation and toxicity
        infl = self._clamp(drug.get("inflammation_risk_pred", 0.0))
        tox = self._clamp(drug.get("genotoxicity_risk_pred", 0.0))
        risk_unc = 0.2 * infl + 0.2 * tox

        # Combined uncertainty: missing dominates
        return min(1.0, missing_unc + risk_unc)

    def _uncertainty_binding(self, epitope: Dict[str, float]) -> float:
        """Uncertainty for binding dimension.

        Directly uses pred_uncertainty from epitope pipeline (0-1).
        Falls back to 0.3 if unavailable.
        """
        unc = epitope.get("pred_uncertainty")
        if unc is None or np.isnan(unc):
            return 0.3  # moderate default
        return float(unc)

    def _uncertainty_kinetics(self, pk: Dict[str, float]) -> float:
        """Uncertainty for kinetics dimension.

        Derived from how many PK parameters are available and physiologically plausible:
        - Missing PK params → high uncertainty
        - Extreme half-life (<0.5h or >72h) → moderate uncertainty
        - Negative/invalid AUC → high uncertainty
        """
        has_cmax = pk.get("pkpd_cmax_mg_per_l") is not None and not np.isnan(pk.get("pkpd_cmax_mg_per_l"))
        has_auc = pk.get("pkpd_auc_conc") is not None and not np.isnan(pk.get("pkpd_auc_conc"))
        has_hl = pk.get("pkpd_half_life_h") is not None and not np.isnan(pk.get("pkpd_half_life_h"))

        present = sum([has_cmax, has_auc, has_hl])
        missing_unc = 1.0 - present / 3.0

        # Physiologically implausible half-life
        hl = pk.get("pkpd_half_life_h", np.nan)
        phys_unc = 0.0
        if not np.isnan(hl):
            if hl < 0.5 or hl > 72.0:
                phys_unc = 0.3

        # Extreme Cmax (potential numerical instability)
        cmax = pk.get("pkpd_cmax_mg_per_l", np.nan)
        conc_unc = 0.0
        if not np.isnan(cmax):
            if cmax > 1000.0 or cmax <= 0.0:
                conc_unc = 0.2

        return min(1.0, missing_unc + phys_unc + conc_unc)

    @staticmethod
    def _adaptive_weights(
        base: Dict[str, float],
        uncertainties: Dict[str, float],
        penalty_slope: float = 2.0,
    ) -> Dict[str, float]:
        """Adjust weights based on per-dimension uncertainty.

        Uncertainty is converted to a credibility score (1 - unc).
        Weights are then re-normalized so that high-uncertainty dimensions
        contribute proportionally less, while the full budget is redistributed
        to confident dimensions.

        Formula:
            w'_i = w_i * (1 - unc_i)^penalty_slope
            w'_i_normalized = w'_i / sum(w'_j)

        Example:
            base = {clinical:0.40, binding:0.35, kinetics:0.25}
            uncertainties = {clinical:0.2, binding:0.7, kinetics:0.1}
            credibility = {clinical:0.8, binding:0.3, kinetics:0.9}
            adjusted = {clinical:0.40*(0.8^2)=0.256, binding:0.35*(0.3^2)=0.0315, kinetics:0.25*(0.9^2)=0.2025}
            sum = 0.490 → normalized
        """
        # Compute adjusted (pre-normalization) weights
        adjusted = {}
        for dim in ["clinical", "binding", "kinetics"]:
            cred = 1.0 - uncertainties[dim]
            adjusted[dim] = base[dim] * (cred ** penalty_slope)

        total = sum(adjusted.values())
        if total <= 0.0:
            # Fallback to uniform if all uncertain
            return {k: 1.0 / 3.0 for k in base}

        # Normalize so they sum to 1.0
        return {dim: adjusted[dim] / total for dim in adjusted}

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        if np.isnan(x):
            return 0.0
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _score_half_life(hl: float) -> float:
        """Score half-life: ideal 4-24h."""
        if np.isnan(hl) or hl <= 0:
            return 0.0
        if 4 <= hl <= 24:
            return 1.0
        if hl < 4:
            return hl / 4.0
        # hl > 24: penalize exponentially
        return np.exp(-0.1 * (hl - 24))

    @staticmethod
    def _score_auc_conc(auc: float) -> float:
        """Score AUC conc: moderate range is best."""
        if np.isnan(auc) or auc <= 0:
            return 0.0
        # Ideal AUC: 10-200 mg/L·h
        if 10 <= auc <= 200:
            return 1.0
        if auc < 10:
            return auc / 10.0
        # AUC > 200: penalize
        return np.exp(-0.005 * (auc - 200))

    @staticmethod
    def _score_therapeutic_index(ti: float) -> float:
        """Score therapeutic index: higher is better."""
        if np.isnan(ti) or ti < 0:
            return 0.0
        # Normalize to 0-1; TI=1.0 → score 0.5; TI=10 → score 1.0
        return min(1.0, ti / (1.0 + ti))

    @staticmethod
    def _score_cmax(cmax: float) -> float:
        """Score Cmax: moderate range is best."""
        if np.isnan(cmax) or cmax <= 0:
            return 0.0
        # Ideal Cmax: 0.5-10 mg/L
        if 0.5 <= cmax <= 10:
            return 1.0
        if cmax < 0.5:
            return cmax / 0.5
        return np.exp(-0.2 * (cmax - 10))

    # ------------------------------------------------------------------
    # Interpretation strings
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_clinical(
        eff: float, bind: float, immune: float, safety: float
    ) -> str:
        eff_level = _qual(eff)
        bind_level = _qual(bind)
        immune_level = _qual(immune)
        safety_level = "safe" if safety < 0.3 else ("caution" if safety < 0.6 else "risky")
        return (
            f"Efficacy: {eff:.2f} ({eff_level}), "
            f"Target binding: {bind:.2f} ({bind_level}), "
            f"Immune activation: {immune:.2f} ({immune_level}), "
            f"Safety: {safety_level}"
        )

    @staticmethod
    def _interpret_binding(eff: float, unc: float, affinity: str) -> str:
        unc_level = "high confidence" if unc < 0.2 else ("moderate" if unc < 0.5 else "low confidence")
        return (
            f"Epitope efficacy: {eff:.2f} ({affinity}), "
            f"Uncertainty: {unc:.2f} ({unc_level})"
        )

    @staticmethod
    def _interpret_kinetics(
        cmax: float, tmax: float, hl: float, ti: float
    ) -> str:
        hl_str = f"{hl:.1f}h" if not np.isnan(hl) else "N/A"
        ti_str = f"{ti:.3f}" if not np.isnan(ti) else "N/A"
        return (
            f"Cmax: {cmax:.2f} mg/L, "
            f"Tmax: {tmax:.1f}h, "
            f"Half-life: {hl_str}, "
            f"Therapeutic index: {ti_str}"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _qual(v: float) -> str:
    """Convert a 0-1 score to a qualitative label."""
    if v >= 0.8:
        return "excellent"
    if v >= 0.6:
        return "good"
    if v >= 0.4:
        return "moderate"
    if v >= 0.2:
        return "poor"
    return "very low"