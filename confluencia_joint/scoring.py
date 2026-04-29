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
class GeneSignatureScore:
    """Five-target gene signature score (TROP2, NECTIN4, LIV-1, B7-H4, TMEM65).

    Based on Yang et al. 2025 acRGBS methodology.

    Attributes
    ----------
    trop2 : float
        TROP2 expression level (0-1).
    nectin4 : float
        NECTIN4 expression level (0-1).
    liv1 : float
        LIV-1 expression level (0-1).
    b7h4 : float
        B7-H4 expression level (0-1).
    tmem65 : float
        TMEM65 expression level (0-1).
    risk_score : float
        Weighted risk score (0-1, higher = higher risk).
    efficacy_score : float
        Predicted efficacy from gene signature (0-1, inverse of risk).
    proliferation_score : float
        Proliferation signature (TROP2 + NECTIN4 + TMEM65).
    immune_score : float
        Immune signature (B7-H4 + TROP2).
    mito_score : float
        Mitochondrial/metabolism signature (TMEM65 + LIV-1).
    tide_score : float
        TIDE immunotherapy response score (0-1, higher = less ICI benefit).
    ips_estimate : float
        Immunophenoscore estimate (0-1, higher = more immunogenic).
    predicted_response : str
        Predicted immunotherapy response: "CR/PR", "SD", or "PD".
    dhe_recommended : bool
        Whether DHE (TMEM65 inhibitor) is recommended.
    overall : float
        Gene signature overall score (0-1).
    interpretation : str
        Human-readable interpretation.
    """
    trop2: float
    nectin4: float
    liv1: float
    b7h4: float
    tmem65: float
    risk_score: float
    efficacy_score: float
    proliferation_score: float
    immune_score: float
    mito_score: float
    tide_score: float
    ips_estimate: float
    predicted_response: str
    dhe_recommended: bool
    overall: float
    interpretation: str


@dataclass
class CircRNAScore:
    """circRNA multi-omics score: innate immune sensing + Cancer Immunity Cycle.

    Captures the circRNA dimension (confluencia_circrna module) that the drug
    pipeline alone cannot assess:
    - Innate immune sensing (RIG-I / TLR7-8 / PKR) from nucleotide sequence
    - Cancer Immunity Cycle 7-step ssGSEA scores
    - Immunotherapy response (TIDE / IPS / TKI)
    - Therapeutic potential composite

    Attributes
    ----------
    immunotherapy_score : float
        Composite immunotherapy score (0-1, higher = better response).
    therapeutic_window : float
        Therapeutic window estimate (0-1, higher = better window).
    tumor_killing_index : float
        Tumor killing index from immune cycle (0-1).
    overall_immunogenicity : float
        circRNA innate immune sensing score (0-1, higher = more immunogenic).
    rig_i_score : float
        RIG-I pathway activation score (0-1).
    tlr_score : float
        TLR7/8 pathway activation score (0-1).
    pkr_score : float
        PKR pathway activation score (0-1).
    tide_score : float
        TIDE immunotherapy response score (0-1, higher = less benefit).
    ips : float
        Immunophenoscore (0-10).
    predicted_response : str
        Predicted response: "likely_responder", "intermediate",
        or "likely_non_responder".
    immune_cycle_score : float
        Aggregate Cancer Immunity Cycle score (0-1).
    tme_score : float
        Tumor microenvironment score from deconvolution (0-1).
    overall : float
        circRNA composite score (0-1).
    interpretation : str
        Human-readable interpretation.
    """
    immunotherapy_score: float
    therapeutic_window: float
    tumor_killing_index: float
    overall_immunogenicity: float
    rig_i_score: float
    tlr_score: float
    pkr_score: float
    tide_score: float
    ips: float
    predicted_response: str
    immune_cycle_score: float
    tme_score: float
    overall: float
    interpretation: str
    trained_model_risk: float


@dataclass
class JointScore:
    """Complete joint score combining five dimensions.

    Attributes
    ----------
    clinical : ClinicalScore
        Clinical-level assessment.
    binding : BindingScore
        MHC-epitope binding assessment.
    kinetics : KineticsScore
        PK/PD kinetics assessment.
    gene_signature : GeneSignatureScore or None
        Five-target gene signature assessment (optional).
    circrna : CircRNAScore or None
        circRNA multi-omics assessment (innate sensing, immune cycle,
        TME, immunotherapy response) (optional).
    composite : float
        Weighted composite score (0-1).
    recommendation : str
        One of "Go", "Conditional", or "No-Go".
    recommendation_reason : str
        Brief explanation of the recommendation.
    effective_weights : dict
        Actual weights used after uncertainty adaptation.
        Keys: "clinical", "binding", "kinetics", "gene_signature", "circrna".
    uncertainties : dict
        Per-dimension uncertainty used for adaptive weighting.
        Keys: "clinical", "binding", "kinetics", "gene_signature", "circrna".
        Values in [0, 1], higher = less confident.
    """
    clinical: ClinicalScore
    binding: BindingScore
    kinetics: KineticsScore
    gene_signature: Optional[GeneSignatureScore] = None
    circrna: Optional[CircRNAScore] = None
    composite: float = 0.0
    recommendation: str = "No-Go"
    recommendation_reason: str = ""
    effective_weights: Dict[str, float] = None
    uncertainties: Dict[str, float] = None


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

@dataclass
class JointScoringEngine:
    """Five-dimensional scoring engine.

    Combines drug efficacy, MHC binding, PK kinetics, gene signature,
    and circRNA multi-omics into a unified composite score and recommendation.

    Parameters
    ----------
    clinical_weight : float (default 0.30)
        Weight for clinical score in composite.
    binding_weight : float (default 0.20)
        Weight for binding score in composite.
    kinetics_weight : float (default 0.15)
        Weight for kinetics score in composite.
    gene_signature_weight : float (default 0.15)
        Weight for gene signature score in composite.
    circrna_weight : float (default 0.20)
        Weight for circRNA multi-omics score in composite.
        Reflects the immunotherapy / innate sensing dimension
        that drug-only pipelines miss.
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
    >>> score = engine.score(drug_out, epi_out, pk_summary,
    ...                      gene_sig_outputs, circrna_outputs)
    >>> print(score.recommendation, score.composite)
    """

    clinical_weight: float = 0.30
    binding_weight: float = 0.20
    kinetics_weight: float = 0.15
    gene_signature_weight: float = 0.15
    circrna_weight: float = 0.20
    go_threshold: float = 0.65
    conditional_threshold: float = 0.40
    safety_floor: float = 0.30

    def __post_init__(self):
        total = (self.clinical_weight + self.binding_weight +
                 self.kinetics_weight + self.gene_signature_weight +
                 self.circrna_weight)
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
        gene_signature_outputs: Optional[Dict[str, float]] = None,
        circrna_outputs: Optional[Dict[str, float]] = None,
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
        gene_signature_outputs : dict, optional
            Output from five-target gene signature module. Keys include:
            - trop2, nectin4, liv1, b7h4, tmem65 (expression levels 0-1)
            - risk_score, efficacy_score, proliferation_score
            - immune_score, mito_score
            - tide_score, ips_estimate, predicted_response
            - dhe_recommended
        circrna_outputs : dict, optional
            Output from confluencia_circrna pipeline. Keys include:
            - immunotherapy_score, therapeutic_window, tumor_killing_index
            - overall_immunogenicity, rig_i_score, tlr_score, pkr_score
            - tide_score, ips, predicted_response
            - immune_cycle_score, tme_score
            - overall

        Returns
        -------
        JointScore
            Complete joint scoring result.
        """
        clinical = self._score_clinical(drug_outputs)
        binding = self._score_binding(epitope_outputs)
        kinetics = self._score_kinetics(pk_summary)
        gene_sig = self._score_gene_signature(gene_signature_outputs) if gene_signature_outputs else None
        circrna = self._score_circrna(circrna_outputs) if circrna_outputs else None

        # --- Uncertainty-adaptive dynamic weights ---
        uncertainties = {
            "clinical": self._uncertainty_clinical(drug_outputs),
            "binding": self._uncertainty_binding(epitope_outputs),
            "kinetics": self._uncertainty_kinetics(pk_summary),
            "gene_signature": self._uncertainty_gene_signature(gene_signature_outputs) if gene_signature_outputs else 1.0,
            "circrna": self._uncertainty_circrna(circrna_outputs) if circrna_outputs else 1.0,
        }
        base = {
            "clinical": self.clinical_weight,
            "binding": self.binding_weight,
            "kinetics": self.kinetics_weight,
            "gene_signature": self.gene_signature_weight,
            "circrna": self.circrna_weight,
        }
        effective = self._adaptive_weights(base, uncertainties)

        # Weighted composite with adaptive weights
        composite = (
            effective["clinical"] * clinical.overall
            + effective["binding"] * binding.overall
            + effective["kinetics"] * kinetics.overall
            + (effective["gene_signature"] * gene_sig.overall if gene_sig else 0.0)
            + (effective["circrna"] * circrna.overall if circrna else 0.0)
        )

        # Recommendation logic
        rec, reason = self._recommend(
            composite, clinical.safety_penalty, effective, gene_sig, circrna
        )

        return JointScore(
            clinical=clinical,
            binding=binding,
            kinetics=kinetics,
            gene_signature=gene_sig,
            circrna=circrna,
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

        # Safety penalty: weighted sum of risk signals (toxicity + inflammation)
        # Note: immune activation is NOT a safety penalty; it's scored separately.
        # Lower penalty = safer (higher safety)
        safety_penalty = 0.6 * tox + 0.4 * infl
        safety_penalty = max(0.0, min(1.0, safety_penalty))

        # Safety penalty used only as override in recommendation logic,
        # not in overall clinical score (avoids double-penalty).
        # Safety override: if safety_penalty > safety_floor → No-Go.
        overall = (
            0.40 * eff
            + 0.35 * bind
            + 0.25 * immune
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

    def _score_gene_signature(
        self, gs: Dict[str, float]
    ) -> GeneSignatureScore:
        """Compute gene signature score from five-target module outputs.

        Parameters
        ----------
        gs : dict
            Keys: trop2, nectin4, liv1, b7h4, tmem65,
                  risk_score, efficacy_score, proliferation_score,
                  immune_score, mito_score, tide_score, ips_estimate,
                  predicted_response, dhe_recommended
        """
        trop2 = self._clamp(gs.get("trop2", 0.5))
        nectin4 = self._clamp(gs.get("nectin4", 0.5))
        liv1 = self._clamp(gs.get("liv1", 0.5))
        b7h4 = self._clamp(gs.get("b7h4", 0.5))
        tmem65 = self._clamp(gs.get("tmem65", 0.5))

        # Computed signature scores (from enhanced module or direct input)
        risk = self._clamp(gs.get("risk_score", 0.5))
        eff = self._clamp(gs.get("efficacy_score", 1.0 - risk))
        prolif = self._clamp(gs.get("proliferation_score", 0.5))
        immune = self._clamp(gs.get("immune_score", 0.5))
        mito = self._clamp(gs.get("mito_score", 0.5))
        tide = self._clamp(gs.get("tide_score", 0.5))
        ips = self._clamp(gs.get("ips_estimate", 0.5))
        response = str(gs.get("predicted_response", "SD"))
        dhe = bool(gs.get("dhe_recommended", False))

        # Overall gene signature score:
        # efficacy (primary) + immune + proliferation - TIDE penalty
        # Weights: eff=0.30, immune=0.15, proliferation=0.15, (1-risk)=0.15, (1-tide)=0.15, mito=0.10
        overall = (
            0.30 * eff
            + 0.15 * immune
            + 0.15 * prolif
            + 0.15 * (1.0 - risk)
            + 0.15 * (1.0 - tide)
            + 0.10 * mito
        )
        overall = max(0.0, min(1.0, overall))

        interpretation = (
            f"Gene signature: efficacy={eff:.2f}, risk={risk:.2f}, "
            f"immune={immune:.2f}, TIDE={tide:.2f}, "
            f"TMEM65={tmem65:.2f}{' (DHE recommended)' if dhe else ''}"
        )

        return GeneSignatureScore(
            trop2=trop2,
            nectin4=nectin4,
            liv1=liv1,
            b7h4=b7h4,
            tmem65=tmem65,
            risk_score=risk,
            efficacy_score=eff,
            proliferation_score=prolif,
            immune_score=immune,
            mito_score=mito,
            tide_score=tide,
            ips_estimate=ips,
            predicted_response=response,
            dhe_recommended=dhe,
            overall=overall,
            interpretation=interpretation,
        )

    def _score_circrna(
        self, cr: Dict[str, float]
    ) -> CircRNAScore:
        """Compute circRNA multi-omics score from confluencia_circrna pipeline outputs.

        Parameters
        ----------
        cr : dict
            Output from CircRNAPipeline. Keys include:
            - immunotherapy_score, therapeutic_window, tumor_killing_index
            - overall_immunogenicity, rig_i_score, tlr_score, pkr_score
            - tide_score, ips, predicted_response
            - immune_cycle_score, tme_score
            - overall
        """
        imm = self._clamp(cr.get("immunotherapy_score", 0.5))
        tw = self._clamp(cr.get("therapeutic_window", 0.5))
        tki = self._clamp(cr.get("tumor_killing_index", 0.5))
        immu = self._clamp(cr.get("overall_immunogenicity", 0.5))
        rig = self._clamp(cr.get("rig_i_score", 0.5))
        tlr = self._clamp(cr.get("tlr_score", 0.5))
        pkr = self._clamp(cr.get("pkr_score", 0.5))
        tide = self._clamp(cr.get("tide_score", 0.5))
        ips = self._clamp(cr.get("ips", 5.0), 0.0, 10.0)
        resp = str(cr.get("predicted_response", "intermediate"))
        cycle = self._clamp(cr.get("immune_cycle_score", 0.5))
        tme = self._clamp(cr.get("tme_score", 0.5))
        trained_risk = self._clamp(cr.get("trained_model_risk", 0.5))

        # Compute overall from sub-indicators (independent of upstream pipeline)
        # Weights: imm=0.20, tki=0.15, immu=0.15, cycle=0.10, tme=0.10, tw=0.10, tide=0.10, ips=0.10
        overall = (
            0.20 * imm
            + 0.15 * tki
            + 0.15 * immu
            + 0.10 * cycle
            + 0.10 * tme
            + 0.10 * tw
            + 0.10 * (1.0 - tide)
            + 0.10 * (ips / 10.0)
        )
        overall = max(0.0, min(1.0, overall))

        interpretation = (
            f"circRNA: immunotherapy={imm:.2f}, TKI={tki:.2f}, "
            f"TIDE={tide:.2f}, IPS={ips:.1f}/10, "
            f"immune cycle={cycle:.2f}, TME={tme:.2f}, "
            f"response={resp}, immunogenicity={immu:.2f}, "
            f"trained_risk={trained_risk:.2f}"
        )

        return CircRNAScore(
            immunotherapy_score=imm,
            therapeutic_window=tw,
            tumor_killing_index=tki,
            overall_immunogenicity=immu,
            rig_i_score=rig,
            tlr_score=tlr,
            pkr_score=pkr,
            tide_score=tide,
            ips=ips,
            predicted_response=resp,
            immune_cycle_score=cycle,
            tme_score=tme,
            overall=overall,
            interpretation=interpretation,
            trained_model_risk=trained_risk,
        )

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    def _recommend(
        self, composite: float, safety_penalty: float,
        effective_weights: Dict[str, float] = None,
        gene_sig: Optional[GeneSignatureScore] = None,
        circrna: Optional[CircRNAScore] = None,
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
            if abs(effective_weights.get("gene_signature", 0.20) - 0.20) > 0.05:
                shifts.append(f"gene_sig {effective_weights['gene_signature']:.2f}")
            if shifts:
                weight_msg = f" [weights adjusted: {', '.join(shifts)}]"

        # Gene signature context
        gs_msg = ""
        if gene_sig:
            if gene_sig.dhe_recommended:
                gs_msg = f" DHE (TMEM65 inhibitor) recommended for high TMEM65 ({gene_sig.tmem65:.2f})."
            elif gene_sig.predicted_response in ("CR/PR",):
                gs_msg = f" Immunotherapy response predicted: {gene_sig.predicted_response}."
            elif gene_sig.risk_score > 0.6:
                gs_msg = f" High gene risk ({gene_sig.risk_score:.2f})."

        # circRNA context
        cr_msg = ""
        if circrna:
            if circrna.predicted_response == "likely_responder":
                cr_msg = f" circRNA predicts likely ICI responder (IPS={circrna.ips:.1f})."
            elif circrna.predicted_response == "likely_non_responder":
                cr_msg = f" circRNA predicts likely ICI non-responder (TIDE={circrna.tide_score:.2f})."
            if circrna.immunotherapy_score > 0.7:
                cr_msg += " Strong circRNA immunotherapy signal."

        if composite >= self.go_threshold:
            return (
                "Go",
                f"Strong composite score ({composite:.3f} ≥ {self.go_threshold:.2f}){weight_msg}{gs_msg}{cr_msg}"
            )
        elif composite >= self.conditional_threshold:
            return (
                "Conditional",
                f"Moderate composite score ({composite:.3f} in [{self.conditional_threshold:.2f}, {self.go_threshold:.2f}]){weight_msg}{gs_msg}{cr_msg}"
            )
        else:
            return (
                "No-Go",
                f"Weak composite score ({composite:.3f} < {self.conditional_threshold:.2f}){gs_msg}{cr_msg}"
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

    def _uncertainty_gene_signature(self, gs: Optional[Dict[str, float]]) -> float:
        """Uncertainty for gene signature dimension.

        Uncertainty arises from:
        - Missing gene expression values → high uncertainty
        - Extreme risk/tide scores → moderate uncertainty
        - High disagreement between proliferation and immune scores
        """
        if gs is None:
            return 1.0  # No gene signature data = maximum uncertainty

        present_count = 0
        total = 5  # 5 genes
        for gene in ["trop2", "nectin4", "liv1", "b7h4", "tmem65"]:
            val = gs.get(gene)
            if val is not None and not np.isnan(float(val)):
                present_count += 1

        missing_unc = 1.0 - present_count / total

        # Extreme risk or tide
        risk = self._clamp(gs.get("risk_score", 0.5))
        tide = self._clamp(gs.get("tide_score", 0.5))
        extreme_unc = 0.15 if (risk > 0.8 or risk < 0.2 or tide > 0.8 or tide < 0.2) else 0.0

        return min(1.0, missing_unc + extreme_unc)

    def _uncertainty_circrna(self, cr: Optional[Dict[str, float]]) -> float:
        """Uncertainty for circRNA dimension.

        Uncertainty arises from:
        - Missing pipeline outputs → high uncertainty
        - Conflicting immunotherapy signals → moderate uncertainty
        """
        if cr is None:
            return 1.0

        present_count = 0
        key_fields = [
            "immunotherapy_score", "therapeutic_window", "tumor_killing_index",
            "tide_score", "ips", "immune_cycle_score", "tme_score",
        ]
        for key in key_fields:
            val = cr.get(key)
            if val is not None:
                try:
                    if not np.isnan(float(val)):
                        present_count += 1
                except (TypeError, ValueError):
                    present_count += 1
        missing_unc = 1.0 - present_count / len(key_fields)

        # Conflicting signals: high TIDE but high IPS
        tide = self._clamp(cr.get("tide_score", 0.5))
        ips = self._clamp(cr.get("ips", 5.0), 0.0, 10.0) / 10.0
        conflict_unc = 0.2 if (tide > 0.6 and ips > 0.6) else 0.0

        return min(1.0, missing_unc + conflict_unc)

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
            base = {clinical:0.30, binding:0.20, kinetics:0.15,
                    gene_signature:0.15, circrna:0.20}
            uncertainties = {clinical:0.2, binding:0.7, kinetics:0.1,
                             gene_signature:0.0, circrna:0.3}
            credibility = {clinical:0.8, binding:0.3, kinetics:0.9,
                           gene_signature:1.0, circrna:0.7}
            adjusted = proportional to base * cred^2
            → normalized so weights sum to 1.0
        """
        # Compute adjusted (pre-normalization) weights
        adjusted = {}
        for dim in ["clinical", "binding", "kinetics", "gene_signature", "circrna"]:
            cred = 1.0 - uncertainties[dim]
            adjusted[dim] = base[dim] * (cred ** penalty_slope)

        total = sum(adjusted.values())
        if total <= 0.0:
            # Fallback to uniform if all uncertain
            n = len(base)
            return {k: 1.0 / n for k in base}

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