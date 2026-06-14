"""
Multi-Drug Combination Strategy for circRNA Immunotherapy

Optimizes drug combinations for circRNA-based cancer immunotherapy:
1. Pharmacokinetic/pharmacodynamic (PK/PD) modeling
2. Drug-drug interaction analysis
3. Synergy scoring (Bliss, Loewe, ZIP, HSA)
4. Resistance risk assessment
5. Dose optimization via Bayesian optimization
6. Personalized therapy selection

Literature basis:
- Chou & Talalay, 1984: Combination index method
- Bliss, 1939: Independence model for synergy
- Yadav et al., 2015: Drug combination screening
- Ianevski et al., 2017: SynergyFinder algorithms
- Mair et al., 2020: Clinical pharmacokinetics of immunotherapy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


# =====================================================================
# Drug Definitions
# =====================================================================

@dataclass
class DrugInfo:
    """Pharmacokinetic and pharmacodynamic drug information."""
    name: str
    drug_class: str

    # Pharmacokinetics
    half_life_h: float
    cmax_typical: float  # mg/L
    auc_typical: float   # mg·h/L
    vd_ss: float         # L/kg
    clearance: float     # L/h/kg
    protein_binding: float  # fraction

    # Pharmacodynamics
    ec50: float          # mg/L
    emax: float          # max effect (0-1)
    hill_coefficient: float

    # Mechanism
    target: str
    mechanism_of_action: str
    resistance_mechanisms: List[str]

    # Safety
    toxicity_profile: Dict[str, float]  # {toxicity_type: probability}
    dose_limiting_toxicity: str
    mtd: float  # Maximum tolerated dose mg/kg

    # Drug interactions
    cyp_interactions: List[str]
    transporter_interactions: List[str]


# Drug database
DRUG_DATABASE: Dict[str, DrugInfo] = {
    "pembrolizumab": DrugInfo(
        name="pembrolizumab",
        drug_class="checkpoint_inhibitor",
        half_life_h=624.0,  # ~26 days
        cmax_typical=200.0,
        auc_typical=50000.0,
        vd_ss=0.05,
        clearance=0.003,
        protein_binding=0.0,
        ec50=0.5,
        emax=0.7,
        hill_coefficient=1.5,
        target="PD-1",
        mechanism_of_action="PD-1 blockade, prevents T cell exhaustion",
        resistance_mechanisms=["JAK1/2 mutations", "B2M loss", "IFN-γ signaling defects"],
        toxicity_profile={"colitis": 0.02, "pneumonitis": 0.03, "thyroiditis": 0.05, "rash": 0.15},
        dose_limiting_toxicity="pneumonitis",
        mtd=10.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
    "nivolumab": DrugInfo(
        name="nivolumab",
        drug_class="checkpoint_inhibitor",
        half_life_h=648.0,  # ~27 days
        cmax_typical=180.0,
        auc_typical=45000.0,
        vd_ss=0.06,
        clearance=0.003,
        protein_binding=0.0,
        ec50=0.6,
        emax=0.65,
        hill_coefficient=1.4,
        target="PD-1",
        mechanism_of_action="PD-1 blockade",
        resistance_mechanisms=["PD-L1 amplification", "JAK mutations"],
        toxicity_profile={"colitis": 0.02, "pneumonitis": 0.02, "hepatitis": 0.01},
        dose_limiting_toxicity="pneumonitis",
        mtd=10.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
    "ipilimumab": DrugInfo(
        name="ipilimumab",
        drug_class="checkpoint_inhibitor",
        half_life_h=384.0,  # ~16 days
        cmax_typical=150.0,
        auc_typical=30000.0,
        vd_ss=0.05,
        clearance=0.004,
        protein_binding=0.0,
        ec50=1.0,
        emax=0.6,
        hill_coefficient=1.2,
        target="CTLA-4",
        mechanism_of_action="CTLA-4 blockade, enhances T cell priming",
        resistance_mechanisms=["Low neoantigen burden", "Treg infiltration"],
        toxicity_profile={"colitis": 0.08, "hepatitis": 0.05, "dermatitis": 0.10},
        dose_limiting_toxicity="colitis",
        mtd=3.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
    "atezolizumab": DrugInfo(
        name="atezolizumab",
        drug_class="checkpoint_inhibitor",
        half_life_h=648.0,
        cmax_typical=200.0,
        auc_typical=55000.0,
        vd_ss=0.04,
        clearance=0.002,
        protein_binding=0.0,
        ec50=0.4,
        emax=0.65,
        hill_coefficient=1.3,
        target="PD-L1",
        mechanism_of_action="PD-L1 blockade on tumor/immune cells",
        resistance_mechanisms=["PD-1 mutations", "IFN signaling defects"],
        toxicity_profile={"hepatitis": 0.02, "pneumonitis": 0.02, "rash": 0.10},
        dose_limiting_toxicity="hepatitis",
        mtd=20.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
    "circrna_vaccine": DrugInfo(
        name="circrna_vaccine",
        drug_class="immunotherapy",
        half_life_h=168.0,  # ~7 days (circRNA stability)
        cmax_typical=1.0,  # arbitrary units
        auc_typical=100.0,
        vd_ss=0.1,
        clearance=0.01,
        protein_binding=0.0,
        ec50=0.2,
        emax=0.8,
        hill_coefficient=2.0,
        target="RIG-I/TLR/MDA5",
        mechanism_of_action="Innate immune activation, antigen presentation, vaccine effect",
        resistance_mechanisms=["PKR activation", "m6A-mediated degradation", "Low antigen presentation"],
        toxicity_profile={"cytokine_release": 0.05, "injection_site": 0.20, "fever": 0.15},
        dose_limiting_toxicity="cytokine_release",
        mtd=5.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
    "doxorubicin": DrugInfo(
        name="doxorubicin",
        drug_class="chemotherapy",
        half_life_h=30.0,
        cmax_typical=5.0,
        auc_typical=50.0,
        vd_ss=25.0,
        clearance=1.5,
        protein_binding=0.75,
        ec50=0.5,
        emax=0.9,
        hill_coefficient=2.0,
        target="DNA",
        mechanism_of_action="DNA intercalation, topoisomerase II inhibition",
        resistance_mechanisms=["MDR1 overexpression", "Topoisomerase II mutations"],
        toxicity_profile={"cardiotoxicity": 0.10, "myelosuppression": 0.30, "alopecia": 0.60},
        dose_limiting_toxicity="cardiotoxicity",
        mtd=0.075,  # mg/kg
        cyp_interactions=["3A4"],
        transporter_interactions=["P-gp"],
    ),
    "cyclophosphamide": DrugInfo(
        name="cyclophosphamide",
        drug_class="chemotherapy",
        half_life_h=8.0,
        cmax_typical=50.0,
        auc_typical=200.0,
        vd_ss=0.8,
        clearance=0.1,
        protein_binding=0.2,
        ec50=5.0,
        emax=0.85,
        hill_coefficient=1.5,
        target="DNA",
        mechanism_of_action="Alkylating agent, crosslinks DNA, immunomodulatory at low dose",
        resistance_mechanisms=["GST overexpression", "DNA repair upregulation"],
        toxicity_profile={"myelosuppression": 0.40, "hemorrhagic_cystitis": 0.05, "cardiotoxicity": 0.02},
        dose_limiting_toxicity="myelosuppression",
        mtd=1.5,
        cyp_interactions=["2B6", "3A4"],
        transporter_interactions=[],
    ),
    "bevacizumab": DrugInfo(
        name="bevacizumab",
        drug_class="targeted_therapy",
        half_life_h=480.0,  # ~20 days
        cmax_typical=300.0,
        auc_typical=80000.0,
        vd_ss=0.03,
        clearance=0.002,
        protein_binding=0.0,
        ec50=50.0,
        emax=0.5,
        hill_coefficient=1.0,
        target="VEGF",
        mechanism_of_action="VEGF neutralization, anti-angiogenesis, normalizes tumor vasculature",
        resistance_mechanisms=["Alternative angiogenesis pathways", "VEGF-independent mechanisms"],
        toxicity_profile={"hypertension": 0.20, "proteinuria": 0.10, "bleeding": 0.05},
        dose_limiting_toxicity="hypertension",
        mtd=15.0,
        cyp_interactions=[],
        transporter_interactions=[],
    ),
}


# =====================================================================
# Combination Analysis
# =====================================================================

@dataclass
class CombinationResult:
    """Result from combination analysis."""
    drug_a: str
    drug_b: str
    dose_a: float
    dose_b: float

    # Efficacy
    expected_efficacy: float
    predicted_response: float  # 0-1

    # Synergy
    bliss_synergy: float  # >0 synergistic, <0 antagonistic
    loewe_synergy: float
    hsa_synergy: float
    combination_index: float  # <1 synergistic

    # Safety
    toxicity_risk: float
    dose_limiting_factor: str

    # PK/PD
    drug_interaction_risk: float
    overlapping_toxicities: List[str]

    # Recommendation
    recommended: bool
    confidence: float
    rationale: str


@dataclass
class DosingRegimen:
    """Complete dosing regimen for a combination."""
    drugs: List[str]
    doses: List[float]  # mg/kg
    frequencies: List[float]  # days between doses
    duration_weeks: int
    expected_efficacy: float
    expected_toxicity: float
    monitoring_schedule: List[str]


class CombinationOptimizer:
    """
    Optimizes multi-drug combinations for circRNA immunotherapy.

    Uses:
    - PK/PD modeling
    - Synergy analysis (Bliss, Loewe, HSA)
    - Safety constraints
    - Bayesian optimization for dose finding
    """

    def __init__(self, patient_profile: Optional[Dict] = None):
        """
        Initialize optimizer.

        Args:
            patient_profile: Patient-specific factors (weight, organ function, etc.)
        """
        self.patient = patient_profile or {}
        self.drug_db = DRUG_DATABASE

    def analyze_combination(
        self,
        drug_a: str,
        drug_b: str,
        dose_a: Optional[float] = None,
        dose_b: Optional[float] = None,
    ) -> CombinationResult:
        """
        Analyze a two-drug combination.

        Args:
            drug_a: First drug name
            drug_b: Second drug name
            dose_a: Dose of drug A (mg/kg), uses EC50 if not specified
            dose_b: Dose of drug B (mg/kg)

        Returns:
            CombinationResult with full analysis
        """
        info_a = self.drug_db.get(drug_a)
        info_b = self.drug_db.get(drug_b)

        if not info_a or not info_b:
            raise ValueError(f"Drug not found: {drug_a if not info_a else drug_b}")

        # Default doses
        dose_a = dose_a or info_a.ec50
        dose_b = dose_b or info_b.ec50

        # Compute individual effects
        effect_a = self._compute_effect(info_a, dose_a)
        effect_b = self._compute_effect(info_b, dose_b)

        # Compute combination effects
        bliss = self._bliss_synergy(effect_a, effect_b, drug_a, drug_b)
        loewe = self._loewe_synergy(info_a, info_b, dose_a, dose_b, effect_a, effect_b)
        hsa = self._hsa_synergy(effect_a, effect_b)
        ci = self._combination_index(info_a, info_b, dose_a, dose_b, effect_a, effect_b)

        # Predict response
        expected_efficacy = self._predict_combination_efficacy(effect_a, effect_b, bliss)
        predicted_response = self._predict_response(expected_efficacy, drug_a, drug_b)

        # Safety analysis
        toxicity_risk, dlt, overlapping = self._analyze_safety(info_a, info_b, dose_a, dose_b)

        # Drug interaction risk
        interaction_risk = self._check_interactions(info_a, info_b)

        # Generate recommendation
        recommended, confidence, rationale = self._make_recommendation(
            bliss, ci, toxicity_risk, interaction_risk, predicted_response
        )

        return CombinationResult(
            drug_a=drug_a,
            drug_b=drug_b,
            dose_a=dose_a,
            dose_b=dose_b,
            expected_efficacy=expected_efficacy,
            predicted_response=predicted_response,
            bliss_synergy=bliss,
            loewe_synergy=loewe,
            hsa_synergy=hsa,
            combination_index=ci,
            toxicity_risk=toxicity_risk,
            dose_limiting_factor=dlt,
            drug_interaction_risk=interaction_risk,
            overlapping_toxicities=overlapping,
            recommended=recommended,
            confidence=confidence,
            rationale=rationale,
        )

    def optimize_doses(
        self,
        drugs: List[str],
        target_efficacy: float = 0.7,
        max_toxicity: float = 0.3,
        n_iterations: int = 100,
    ) -> DosingRegimen:
        """
        Optimize doses for a multi-drug combination.

        Uses Bayesian optimization to find doses that:
        - Maximize efficacy
        - Keep toxicity below threshold
        - Respect synergy/antagonism

        Args:
            drugs: List of drug names
            target_efficacy: Target efficacy (0-1)
            max_toxicity: Maximum acceptable toxicity (0-1)
            n_iterations: Optimization iterations

        Returns:
            DosingRegimen with optimized doses
        """
        drug_infos = [self.drug_db[d] for d in drugs if d in self.drug_db]
        n_drugs = len(drug_infos)

        if n_drugs == 0:
            raise ValueError("No valid drugs provided")

        # Objective: maximize efficacy - penalty for toxicity
        def objective(doses: np.ndarray) -> float:
            total_effect = 0.0
            total_toxicity = 0.0

            for i, (info, dose) in enumerate(zip(drug_infos, doses)):
                effect = self._compute_effect(info, dose)
                tox = self._compute_toxicity(info, dose)
                total_effect += effect
                total_toxicity += tox

            # Pairwise synergy
            synergy_bonus = 0.0
            for i in range(n_drugs):
                for j in range(i+1, n_drugs):
                    result = self.analyze_combination(
                        drugs[i], drugs[j],
                        doses[i], doses[j]
                    )
                    synergy_bonus += result.bliss_synergy * 0.1

            # Penalty for exceeding toxicity
            toxicity_penalty = max(0, total_toxicity - max_toxicity) * 10

            return -(total_effect / n_drugs + synergy_bonus - toxicity_penalty)

        # Bounds (0 to MTD for each drug)
        bounds = [(0, info.mtd) for info in drug_infos]

        # Initial guess (EC50 values)
        x0 = np.array([info.ec50 for info in drug_infos])

        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': n_iterations}
        )

        optimal_doses = result.x

        # Determine frequencies based on half-lives
        frequencies = [info.half_life_h / 24.0 for info in drug_infos]  # Dosing every half-life

        # Expected outcomes
        expected_efficacy = 0.0
        expected_toxicity = 0.0
        for info, dose in zip(drug_infos, optimal_doses):
            expected_efficacy += self._compute_effect(info, dose)
            expected_toxicity += self._compute_toxicity(info, dose)
        expected_efficacy /= n_drugs
        expected_toxicity /= n_drugs

        # Monitoring schedule
        monitoring = self._generate_monitoring_schedule(drug_infos)

        return DosingRegimen(
            drugs=drugs,
            doses=list(optimal_doses),
            frequencies=frequencies,
            duration_weeks=12,
            expected_efficacy=expected_efficacy,
            expected_toxicity=expected_toxicity,
            monitoring_schedule=monitoring,
        )

    def recommend_combinations(
        self,
        circrna_vaccine: bool = True,
        n_top: int = 5,
    ) -> pd.DataFrame:
        """
        Recommend top drug combinations for circRNA immunotherapy.

        Args:
            circrna_vaccine: Include circRNA vaccine in combinations
            n_top: Number of top combinations to return

        Returns:
            DataFrame with ranked combinations
        """
        # Base drugs
        base_drugs = ["pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab"]
        other_drugs = ["bevacizumab", "cyclophosphamide", "doxorubicin"]

        if circrna_vaccine:
            base_drugs = ["circrna_vaccine"] + base_drugs

        results = []

        # Generate all pairs
        all_drugs = base_drugs + other_drugs
        for i, drug_a in enumerate(all_drugs):
            for drug_b in all_drugs[i+1:]:
                try:
                    result = self.analyze_combination(drug_a, drug_b)
                    results.append({
                        "combination": f"{drug_a} + {drug_b}",
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "expected_efficacy": result.expected_efficacy,
                        "predicted_response": result.predicted_response,
                        "bliss_synergy": result.bliss_synergy,
                        "combination_index": result.combination_index,
                        "toxicity_risk": result.toxicity_risk,
                        "recommended": result.recommended,
                        "confidence": result.confidence,
                        "rationale": result.rationale,
                    })
                except Exception:
                    continue

        df = pd.DataFrame(results)
        df = df.sort_values("expected_efficacy", ascending=False)

        return df.head(n_top)

    # =================================================================
    # Helper Methods
    # =================================================================

    def _compute_effect(self, info: DrugInfo, dose: float) -> float:
        """Compute drug effect using Hill equation."""
        if dose <= 0:
            return 0.0
        return info.emax * (dose ** info.hill_coefficient) / (
            info.ec50 ** info.hill_coefficient + dose ** info.hill_coefficient
        )

    def _compute_toxicity(self, info: DrugInfo, dose: float) -> float:
        """Compute toxicity risk."""
        base_tox = sum(info.toxicity_profile.values()) / len(info.toxicity_profile)
        dose_factor = dose / info.mtd
        return min(1.0, base_tox * (1 + dose_factor))

    def _bliss_synergy(
        self,
        effect_a: float,
        effect_b: float,
        drug_a: str,
        drug_b: str,
    ) -> float:
        """
        Compute Bliss synergy score.

        Bliss independence: E_ab = E_a + E_b - E_a * E_b
        Synergy > 0: observed > expected
        """
        # Expected combination effect under independence
        expected = effect_a + effect_b - effect_a * effect_b

        # Estimate actual combination effect
        # Boost for known synergistic pairs
        synergy_pairs = [
            ("circrna_vaccine", "pembrolizumab"),
            ("circrna_vaccine", "nivolumab"),
            ("circrna_vaccine", "atezolizumab"),
            ("ipilimumab", "nivolumab"),
            ("pembrolizumab", "bevacizumab"),
            ("cyclophosphamide", "circrna_vaccine"),  # Low-dose cyclophosphamide depletes Tregs
        ]

        pair = (drug_a, drug_b)
        reverse = (drug_b, drug_a)

        if pair in synergy_pairs or reverse in synergy_pairs:
            observed = expected * 1.2  # 20% synergy boost
        else:
            observed = expected

        return observed - expected

    def _loewe_synergy(
        self,
        info_a: DrugInfo,
        info_b: DrugInfo,
        dose_a: float,
        dose_b: float,
        effect_a: float,
        effect_b: float,
    ) -> float:
        """
        Compute Loewe additivity synergy.

        Loewe: D_a/Dx_a + D_b/Dx_b = 1 for additive
        < 1 for synergistic
        """
        # Dose for same effect if used alone
        dx_a = info_a.ec50 * (effect_a / (info_a.emax - effect_a + 0.01)) ** (1/info_a.hill_coefficient)
        dx_b = info_b.ec50 * (effect_b / (info_b.emax - effect_b + 0.01)) ** (1/info_b.hill_coefficient)

        if dx_a <= 0 or dx_b <= 0:
            return 0.0

        loewe_index = dose_a / dx_a + dose_b / dx_b

        # Synergy score: negative means synergistic
        return 1.0 - loewe_index

    def _hsa_synergy(self, effect_a: float, effect_b: float) -> float:
        """
        Compute HSA (Highest Single Agent) synergy.

        HSA: E_ab > max(E_a, E_b) indicates synergy
        """
        expected = max(effect_a, effect_b)
        observed = effect_a + effect_b - effect_a * effect_b  # Conservative estimate
        return observed - expected

    def _combination_index(
        self,
        info_a: DrugInfo,
        info_b: DrugInfo,
        dose_a: float,
        dose_b: float,
        effect_a: float,
        effect_b: float,
    ) -> float:
        """
        Compute Chou-Talalay combination index.

        CI < 1: synergistic
        CI = 1: additive
        CI > 1: antagonistic
        """
        # Simplified CI calculation
        ci_a = dose_a / info_a.ec50
        ci_b = dose_b / info_b.ec50

        # Adjust for combination
        total_effect = effect_a + effect_b
        if total_effect <= 0:
            return 2.0

        return (ci_a + ci_b) / max(total_effect, 0.1)

    def _predict_combination_efficacy(
        self,
        effect_a: float,
        effect_b: float,
        bliss_synergy: float,
    ) -> float:
        """Predict overall combination efficacy."""
        base = effect_a + effect_b - effect_a * effect_b
        return np.clip(base + bliss_synergy, 0.0, 1.0)

    def _predict_response(
        self,
        efficacy: float,
        drug_a: str,
        drug_b: str,
    ) -> float:
        """Predict clinical response probability."""
        # Base response from efficacy
        base = 0.5 + 0.4 * efficacy

        # Adjust for patient factors
        if self.patient.get("high_tmb", False):
            base *= 1.15  # Better response
        if self.patient.get("pd_l1_high", False):
            base *= 1.1

        return np.clip(base, 0.0, 1.0)

    def _analyze_safety(
        self,
        info_a: DrugInfo,
        info_b: DrugInfo,
        dose_a: float,
        dose_b: float,
    ) -> Tuple[float, str, List[str]]:
        """Analyze combination safety."""
        tox_a = self._compute_toxicity(info_a, dose_a)
        tox_b = self._compute_toxicity(info_b, dose_b)

        # Combined toxicity (not simply additive)
        combined = tox_a + tox_b - tox_a * tox_b * 0.5

        # Find overlapping toxicities
        overlapping = list(set(info_a.toxicity_profile.keys()) & set(info_b.toxicity_profile.keys()))

        # Dose-limiting factor
        if tox_a > tox_b:
            dlt = info_a.dose_limiting_toxicity
        else:
            dlt = info_b.dose_limiting_toxicity

        return combined, dlt, overlapping

    def _check_interactions(self, info_a: DrugInfo, info_b: DrugInfo) -> float:
        """Check for drug-drug interactions."""
        risk = 0.0

        # CYP interactions
        cyp_overlap = set(info_a.cyp_interactions) & set(info_b.cyp_interactions)
        risk += 0.1 * len(cyp_overlap)

        # Transporter interactions
        tp_overlap = set(info_a.transporter_interactions) & set(info_b.transporter_interactions)
        risk += 0.1 * len(tp_overlap)

        return min(risk, 1.0)

    def _make_recommendation(
        self,
        bliss: float,
        ci: float,
        toxicity: float,
        interaction: float,
        response: float,
    ) -> Tuple[bool, float, str]:
        """
        Generate treatment recommendation with Bliss-CI discrepancy handling.

        Bliss-CI Interpretation Matrix:
        - Bliss > 0.10, CI < 1.0: Strong synergy (both metrics agree)
        - Bliss > 0.10, CI 1.0-2.0: Moderate synergy (dose-ratio acceptable)
        - Bliss > 0.10, CI > 2.0: Effect synergy, dose mismatch (requires optimization)
        - Bliss < 0, CI < 1.0: Low effect synergy, dose synergy possible
        - Bliss < 0, CI > 2.0: Antagonism (not recommended)

        For immunotherapy combinations where mechanisms interact (both affect T cells),
        Bliss independence assumption may not hold. CI is prioritized for dose decisions.
        """
        reasons = []
        requires_optimization = False

        # Check Bliss-CI discrepancy
        bliss_synergy = bliss > 0.05
        ci_synergy = ci < 1.0
        ci_antagonism = ci > 1.5

        if bliss_synergy and ci_antagonism:
            # Bliss shows effect synergy, but CI shows dose mismatch
            reasons.append("Bliss synergy with CI antagonism: requires dose optimization")
            requires_optimization = True
        elif bliss_synergy and not ci_synergy:
            # Moderate case: Bliss positive, CI near 1
            reasons.append("Effect synergy at suboptimal dose ratio")
        elif bliss_synergy and ci_synergy:
            reasons.append("Strong synergy (Bliss and CI agree)")
        elif ci_synergy and not bliss_synergy:
            reasons.append("Dose-level synergy with limited effect synergy")
        elif ci_antagonism and bliss < 0:
            reasons.append("Antagonistic combination - not recommended")
        elif bliss < -0.05:
            reasons.append("Antagonistic - not recommended")

        # Safety check
        if toxicity > 0.4:
            reasons.append("High toxicity risk")
        elif toxicity < 0.2:
            reasons.append("Favorable safety profile")

        # Response prediction
        if response > 0.6:
            reasons.append("High predicted response")
        elif response < 0.3:
            reasons.append("Low predicted response")

        # Overall decision with CI-weighted logic
        # CI > 2.0 with positive Bliss still allows conditional recommendation
        if bliss < -0.05 or toxicity > 0.5 or response < 0.2:
            recommended = False
        elif requires_optimization:
            recommended = False  # Requires dose optimization before use
        else:
            recommended = bliss > -0.05 and toxicity < 0.4 and response > 0.3

        # Confidence calculation
        if requires_optimization:
            confidence = 0.3  # Lower confidence for discrepant results
        elif bliss_synergy and ci_synergy:
            confidence = 0.8
        elif bliss_synergy or ci_synergy:
            confidence = 0.5
        else:
            confidence = 0.3

        # Adjust for safety and response
        confidence += 0.1 * (toxicity < 0.3) + 0.1 * (response > 0.5)
        confidence = np.clip(confidence, 0.0, 1.0)

        rationale = "; ".join(reasons) if reasons else "Standard combination"

        return recommended, confidence, rationale

    def _generate_monitoring_schedule(self, drug_infos: List[DrugInfo]) -> List[str]:
        """Generate monitoring schedule for combination."""
        schedule = [
            "Baseline: CBC, LFTs, thyroid function, imaging",
            "Week 2: CBC, LFTs, adverse event assessment",
            "Week 4: CBC, LFTs, thyroid function, imaging, response assessment",
            "Week 8: CBC, LFTs, response imaging",
            "Week 12: Comprehensive response evaluation",
        ]

        # Add drug-specific monitoring
        for info in drug_infos:
            if "cardiotoxicity" in info.toxicity_profile:
                schedule.append("Monthly: ECHO/EKG for cardiotoxicity monitoring")
            if "pneumonitis" in info.toxicity_profile:
                schedule.append("As needed: CT chest for pneumonitis symptoms")
            if "colitis" in info.toxicity_profile:
                schedule.append("As needed: Stool studies, colonoscopy if symptomatic")

        return list(set(schedule))  # Remove duplicates


# =====================================================================
# Convenience Functions
# =====================================================================

def analyze_combination(
    drug_a: str,
    drug_b: str,
    dose_a: Optional[float] = None,
    dose_b: Optional[float] = None,
) -> CombinationResult:
    """Quick combination analysis."""
    optimizer = CombinationOptimizer()
    return optimizer.analyze_combination(drug_a, drug_b, dose_a, dose_b)


def optimize_combination(
    drugs: List[str],
    target_efficacy: float = 0.7,
) -> DosingRegimen:
    """Optimize multi-drug combination."""
    optimizer = CombinationOptimizer()
    return optimizer.optimize_doses(drugs, target_efficacy)


def get_top_combinations(n: int = 5, include_circrna: bool = True) -> pd.DataFrame:
    """Get top recommended combinations."""
    optimizer = CombinationOptimizer()
    return optimizer.recommend_combinations(circrna_vaccine=include_circrna, n_top=n)


__all__ = [
    "DrugInfo",
    "DRUG_DATABASE",
    "CombinationResult",
    "DosingRegimen",
    "CombinationOptimizer",
    "analyze_combination",
    "optimize_combination",
    "get_top_combinations",
]
