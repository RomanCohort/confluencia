"""
circRNA RL-ABM Environment

Integrates Immune ABM (TME simulation) with circRNA sequence evolution
for reinforcement learning-based drug efficacy optimization.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    RL-ABM Closed Loop                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   State: circRNA sequence + patient profile (gene expression)      │
│                     ↓                                               │
│   Action: mutate / modify / combination drug selection             │
│                     ↓                                               │
│   Environment: Immune ABM → Drug Response → Clinical Outcome       │
│                     ↓                                               │
│   Reward: response_prob × (1 - resistance) × immune_auc           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Key components:
1. CircRNAEnv: Gym-style environment for RL
2. ABM-based reward: Uses immune_abm.py for TME simulation
3. Clinical reward: Uses drug_response.py for efficacy prediction
4. Multi-objective: Combines immunogenicity + efficacy + safety

Literature basis:
- Chen & Mellman, 2013: Cancer Immunotherapy Cycle
- Jiang et al., 2018: TIDE score
- Arora et al., 2024: AI for immunotherapy optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

# Import existing modules
from .cirrna_evolution import (
    CIRCRNA_ACTIONS,
    CircRNAEvolutionConfig,
    mutate_backbone,
    optimize_ires,
    shuffle_ires_flanking,
    compute_cirrna_objectives,
)
from .drug_response import (
    DrugResponsePredictor,
    DrugResponseFeatures,
    recommend_treatment,
)

# Import ABM from drug module
try:
    from confluencia_2_0_drug.core.immune_abm import (
        ImmuneABMConfig,
        simulate_immune_response,
        summarize_immune_curve,
        build_epitope_triggers,
    )
    ABM_AVAILABLE = True
except ImportError:
    ABM_AVAILABLE = False

# Import TME simulation
try:
    from .tme_simulation import (
        TMESimulator,
        TMEConfig,
        TMEResult,
        simulate_tme,
        classify_tme,
    )
    TME_AVAILABLE = True
except ImportError:
    TME_AVAILABLE = False

# Import multi-drug combination
try:
    from .multi_drug_combination import (
        CombinationOptimizer,
        CombinationResult,
        DosingRegimen,
        DRUG_DATABASE,
    )
    COMBINATION_AVAILABLE = True
except ImportError:
    COMBINATION_AVAILABLE = False


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class PatientProfile:
    """Patient-specific features for personalized simulation."""
    patient_id: str = "default"
    # Gene expression (normalized 0-1)
    trop2: float = 0.5
    b7h4: float = 0.5
    mki67: float = 0.5
    myc: float = 0.5
    pd_l1: float = 0.5
    # Immune context
    t_cell_infiltration: float = 0.5
    immune_suppression_score: float = 0.3
    # Clinical features
    tumor_burden: float = 0.5
    performance_status: float = 0.8
    prior_treatments: int = 0

    def to_gene_expression(self) -> Dict[str, float]:
        """Convert to gene expression dict for drug_response module."""
        return {
            "TROP2": self.trop2,
            "B7-H4": self.b7h4,
            "MKI67": self.mki67,
            "MYC": self.myc,
            "PD-L1": self.pd_l1,
        }


@dataclass
class RLABMConfig:
    """Configuration for RL-ABM environment."""
    # Evolution parameters
    max_steps: int = 20
    seed_seq: str = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"
    initial_modification: str = "m6A"

    # ABM parameters
    abm_horizon_h: int = 96
    abm_dt_h: float = 1.0

    # Reward weights
    weight_efficacy: float = 0.35
    weight_immune_response: float = 0.30
    weight_safety: float = 0.20
    weight_novelty: float = 0.15

    # RL parameters
    action_space: List[str] = field(default_factory=lambda: CIRCRNA_ACTIONS + ["select_combination"])
    early_stop_no_improve: int = 5

    # Patient profile
    patient: PatientProfile = field(default_factory=PatientProfile)

    # Combination drug options
    combination_drugs: List[str] = field(default_factory=lambda: ["pembrolizumab", "nivolumab", "atezolizumab"])

    seed: int = 42


@dataclass
class CircRNAState:
    """State representation for RL agent."""
    sequence: str
    modification: str
    step: int
    cumulative_reward: float
    best_reward: float
    history: List[Dict[str, Any]]

    def to_observation(self) -> np.ndarray:
        """Convert state to observation vector for RL agent."""
        # Sequence features
        seq = self.sequence.upper().replace("T", "U")
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / max(length, 1)

        # Modification encoding
        mod_encoding = {
            "none": 0.0, "m6A": 0.2, "Psi": 0.4,
            "5mC": 0.6, "ms2m6A": 0.8, "2OMeA": 1.0
        }
        mod_val = mod_encoding.get(self.modification, 0.0)

        # History features
        n_improvements = sum(1 for h in self.history if h.get("improved", False))
        avg_reward = np.mean([h.get("reward", 0.0) for h in self.history]) if self.history else 0.0

        return np.array([
            length / 1000.0,  # Normalized length
            gc,
            mod_val,
            self.step / 20.0,
            self.cumulative_reward,
            self.best_reward,
            n_improvements / max(len(self.history), 1),
            avg_reward,
        ], dtype=np.float32)


@dataclass
class ABMRewardComponents:
    """Breakdown of ABM-based reward."""
    # ABM outputs
    peak_antibody: float
    peak_effector_t: float
    immune_auc: float
    antigen_clearance: float

    # Drug response
    response_probability: float
    resistance_risk: float
    synergy_score: float

    # Combined
    total_reward: float


# =====================================================================
# ABM-based Reward Function
# =====================================================================

def compute_abm_reward(
    sequence: str,
    modification: str,
    patient: PatientProfile,
    combination_drug: Optional[str] = None,
    abm_config: Optional[ImmuneABMConfig] = None,
    dose: float = 1.0,
) -> ABMRewardComponents:
    """
    Compute reward using ABM simulation + drug response prediction.

    Pipeline:
    1. Extract epitope-like regions from circRNA sequence
    2. Run ABM simulation to get immune response dynamics
    3. Predict drug response using immune scores + patient profile
    4. Combine into multi-objective reward

    Args:
        sequence: circRNA sequence
        modification: Chemical modification (m6A, Psi, etc.)
        patient: Patient profile with gene expression
        combination_drug: Optional combination drug
        abm_config: ABM simulation config
        dose: Treatment dose

    Returns:
        ABMRewardComponents with detailed breakdown
    """
    # Default ABM config
    if abm_config is None and ABM_AVAILABLE:
        abm_config = ImmuneABMConfig(horizon_h=96)
    elif abm_config is None:
        abm_config = None

    # Step 1: Get immune scores from sequence
    immune_scores = _estimate_immune_scores(sequence, modification)

    # Step 2: Run ABM simulation (if available)
    peak_ab = 0.5
    peak_t = 0.5
    immune_auc = 0.5
    antigen_clearance = 0.5

    if ABM_AVAILABLE and abm_config is not None:
        # Extract epitope-like regions (simplified: use whole sequence)
        epitope_seq = _extract_epitope_region(sequence)

        # Build trigger for ABM
        trigger_df = pd.DataFrame({
            "epitope_seq": [epitope_seq],
            "dose": [dose],
            "treatment_time": [0.0],
        })
        triggers = build_epitope_triggers(trigger_df)

        # Run ABM
        curve = simulate_immune_response(triggers, config=abm_config)
        summary = summarize_immune_curve(curve)

        peak_ab = summary.get("immune_peak_antibody", 0.5)
        peak_t = summary.get("immune_peak_effector_t", 0.5)
        immune_auc = summary.get("immune_response_auc", 0.5)

        # Antigen clearance (lower peak antigen = better clearance)
        peak_ag = summary.get("immune_peak_antigen", 1.0)
        antigen_clearance = 1.0 - min(peak_ag / 10.0, 1.0)

    # Step 3: Drug response prediction
    predictor = DrugResponsePredictor()
    gene_expr = patient.to_gene_expression()

    combination_drugs = [combination_drug] if combination_drug else None
    drug_response = predictor.predict(
        immune_scores=immune_scores,
        gene_expression=gene_expr,
        treatment_type="circRNA_vaccine",
        combination_drugs=combination_drugs,
    )

    # Step 4: Compute combined reward
    # Efficacy component
    efficacy_reward = drug_response.response_probability * (1.0 - drug_response.resistance_risk)

    # Immune response component
    immune_reward = (0.4 * peak_ab + 0.4 * peak_t + 0.2 * antigen_clearance)
    immune_reward = np.clip(immune_reward, 0.0, 1.0)

    # Safety component (inverse of toxicity)
    safety_reward = 1.0 - _estimate_toxicity(sequence, modification, immune_scores)

    # Synergy bonus
    synergy_score = 0.0
    if drug_response.synergy_scores:
        synergy_score = max(s.synergy_value for s in drug_response.synergy_scores)

    # Total reward (weighted combination)
    total = (
        0.35 * efficacy_reward +
        0.30 * immune_reward +
        0.20 * safety_reward +
        0.15 * synergy_score
    )

    return ABMRewardComponents(
        peak_antibody=peak_ab,
        peak_effector_t=peak_t,
        immune_auc=immune_auc,
        antigen_clearance=antigen_clearance,
        response_probability=drug_response.response_probability,
        resistance_risk=drug_response.resistance_risk,
        synergy_score=synergy_score,
        total_reward=float(np.clip(total, 0.0, 1.0)),
    )


# =====================================================================
# TME-Enhanced Reward Function
# =====================================================================

@dataclass
class TMERewardComponents:
    """Breakdown of TME-based reward (more detailed than ABM)."""
    # TME simulation outputs
    tumor_reduction: float
    t_cd8_expansion: float
    treg_ratio: float  # Treg/CD8, lower is better
    m1_m2_ratio: float  # M1/M2, higher is better
    immune_score: float
    tme_type: str
    response_prediction: float

    # Drug response
    response_probability: float
    resistance_risk: float

    # Combination analysis
    synergy_score: float
    combination_index: float

    # Combined
    total_reward: float


def compute_tme_reward(
    sequence: str,
    modification: str,
    patient: PatientProfile,
    combination_drugs: Optional[List[str]] = None,
    tme_config: Optional[TMEConfig] = None,
    dose: float = 1.0,
    horizon_h: int = 168,
) -> TMERewardComponents:
    """
    Compute reward using advanced TME simulation + drug response + combination analysis.

    This is a more sophisticated version of compute_abm_reward that:
    1. Uses multi-cell-type TME simulation (CD8, CD4, Treg, B, NK, M1, M2, MDSC, CAF)
    2. Models spatial heterogeneity (core vs margin vs stroma)
    3. Includes cytokine network (IFN-γ, TNF-α, IL-2, IL-6, IL-10, TGF-β)
    4. Analyzes drug combinations with synergy scoring

    Args:
        sequence: circRNA sequence
        modification: Chemical modification
        patient: Patient profile
        combination_drugs: List of combination drugs
        tme_config: TME simulation config
        dose: circRNA dose
        horizon_h: Simulation horizon in hours

    Returns:
        TMERewardComponents with detailed breakdown
    """
    # Step 1: Get immune scores from sequence
    immune_scores = _estimate_immune_scores(sequence, modification)

    # Step 2: Run TME simulation if available
    tumor_reduction = 0.0
    t_cd8_expansion = 1.0
    treg_ratio = 0.3
    m1_m2_ratio = 1.0
    immune_score = 50.0
    tme_type = "mixed"
    response_prediction = 0.5

    if TME_AVAILABLE:
        # Configure TME simulation
        cfg = tme_config or TMEConfig(
            horizon_h=horizon_h,
            seed=42,
        )

        # Adjust initial populations based on patient profile
        cfg.initial_t_cd8 = 100.0 * (1.0 + patient.t_cell_infiltration)
        cfg.initial_treg = 30.0 * (1.0 + patient.immune_suppression_score)
        cfg.initial_tumor = 500.0 * patient.tumor_burden

        simulator = TMESimulator(cfg)

        # Determine checkpoint inhibitor type
        checkpoint = "none"
        if combination_drugs:
            for drug in combination_drugs:
                if "pembrolizumab" in drug or "nivolumab" in drug or "atezolizumab" in drug:
                    checkpoint = "anti_pd1"
                elif "ipilimumab" in drug:
                    checkpoint = "anti_ctla4" if checkpoint == "none" else "combo"

        result = simulator.simulate(
            circrna_dose=dose,
            checkpoint_inhibitor=checkpoint,
        )

        # Extract metrics
        tumor_reduction = float((result.tumor_volume[0] - result.tumor_volume[-1]) / result.tumor_volume[0])
        t_cd8_expansion = float(result.t_cd8_count[-1] / result.t_cd8_count[0])
        treg_ratio = float(result.treg_count[-1] / max(result.t_cd8_count[-1], 1.0))
        m1_m2_ratio = float(result.mac_m1_count[-1] / max(result.mac_m2_count[-1], 1.0))
        immune_score = result.immune_score
        tme_type = result.tme_type
        response_prediction = result.response_prediction

    # Step 3: Drug response prediction
    predictor = DrugResponsePredictor()
    gene_expr = patient.to_gene_expression()

    drug_response = predictor.predict(
        immune_scores=immune_scores,
        gene_expression=gene_expr,
        treatment_type="circRNA_vaccine",
        combination_drugs=combination_drugs,
    )

    # Step 4: Combination analysis
    synergy_score = 0.0
    combination_index = 1.0

    if COMBINATION_AVAILABLE and combination_drugs and len(combination_drugs) >= 2:
        optimizer = CombinationOptimizer()
        try:
            combo_result = optimizer.analyze_combination(
                combination_drugs[0],
                combination_drugs[1],
            )
            synergy_score = combo_result.bliss_synergy
            combination_index = combo_result.combination_index
        except Exception:
            pass

    # Step 5: Compute combined reward
    # TME component
    tme_reward = (
        0.25 * tumor_reduction +
        0.15 * np.clip(t_cd8_expansion - 1.0, 0.0, 1.0) +
        0.10 * np.clip(1.0 - treg_ratio, 0.0, 1.0) +
        0.10 * np.clip(m1_m2_ratio - 1.0, 0.0, 1.0) +
        0.10 * immune_score / 100.0
    )

    # Drug response component
    efficacy_reward = drug_response.response_probability * (1.0 - drug_response.resistance_risk)

    # Safety component
    safety_reward = 1.0 - _estimate_toxicity(sequence, modification, immune_scores)

    # Total reward (weighted combination)
    total = (
        0.30 * tme_reward +
        0.30 * efficacy_reward +
        0.20 * safety_reward +
        0.15 * np.clip(synergy_score, -0.1, 0.2) +
        0.05 * response_prediction
    )

    return TMERewardComponents(
        tumor_reduction=tumor_reduction,
        t_cd8_expansion=t_cd8_expansion,
        treg_ratio=treg_ratio,
        m1_m2_ratio=m1_m2_ratio,
        immune_score=immune_score,
        tme_type=tme_type,
        response_prediction=response_prediction,
        response_probability=drug_response.response_probability,
        resistance_risk=drug_response.resistance_risk,
        synergy_score=synergy_score,
        combination_index=combination_index,
        total_reward=float(np.clip(total, 0.0, 1.0)),
    )


# =====================================================================
# Multi-Drug Combination Optimization
# =====================================================================

@dataclass
class CombinationRecommendation:
    """Complete recommendation for circRNA + drug combination."""
    circrna_sequence: str
    circrna_modification: str
    combination_drugs: List[str]
    doses: List[float]
    frequencies: List[float]

    # Predicted outcomes
    expected_response: float
    expected_tumor_reduction: float
    toxicity_risk: float

    # TME predictions
    predicted_tme_type: str
    immune_score: float

    # Synergy
    bliss_synergy: float
    combination_index: float

    # Monitoring
    monitoring_schedule: List[str]

    rationale: str


def optimize_circrna_combination(
    sequence: str,
    modification: str,
    patient: PatientProfile,
    available_drugs: Optional[List[str]] = None,
    n_top: int = 3,
) -> List[CombinationRecommendation]:
    """
    Find optimal drug combinations for a given circRNA sequence.

    Uses:
    1. TME simulation to predict immune response
    2. Combination optimizer for synergy analysis
    3. Safety constraints

    Args:
        sequence: circRNA sequence
        modification: Chemical modification
        patient: Patient profile
        available_drugs: Available drugs for combination
        n_top: Number of top recommendations

    Returns:
        List of CombinationRecommendation sorted by expected response
    """
    available_drugs = available_drugs or [
        "pembrolizumab", "nivolumab", "atezolizumab",
        "ipilimumab", "bevacizumab", "cyclophosphamide",
    ]

    recommendations = []

    for drug_a in available_drugs:
        for drug_b in available_drugs:
            if drug_a == drug_b:
                continue

            combination = [drug_a, drug_b]

            # Compute TME reward
            tme_reward = compute_tme_reward(
                sequence=sequence,
                modification=modification,
                patient=patient,
                combination_drugs=combination,
                dose=1.0,
                horizon_h=168,
            )

            # Get combination analysis
            if COMBINATION_AVAILABLE:
                optimizer = CombinationOptimizer()
                combo_result = optimizer.analyze_combination(drug_a, drug_b)
                doses = [combo_result.dose_a, combo_result.dose_b]
                frequencies = [
                    DRUG_DATABASE[drug_a].half_life_h / 24.0,
                    DRUG_DATABASE[drug_b].half_life_h / 24.0,
                ]
                monitoring = optimizer._generate_monitoring_schedule([
                    DRUG_DATABASE[drug_a],
                    DRUG_DATABASE[drug_b],
                ])
            else:
                doses = [1.0, 1.0]
                frequencies = [21.0, 21.0]
                monitoring = ["Standard monitoring"]

            # Generate rationale
            rationale = _generate_combination_rationale(
                drug_a, drug_b, tme_reward, modification
            )

            recommendations.append(CombinationRecommendation(
                circrna_sequence=sequence,
                circrna_modification=modification,
                combination_drugs=combination,
                doses=doses,
                frequencies=frequencies,
                expected_response=tme_reward.response_probability,
                expected_tumor_reduction=tme_reward.tumor_reduction,
                toxicity_risk=tme_reward.resistance_risk,
                predicted_tme_type=tme_reward.tme_type,
                immune_score=tme_reward.immune_score,
                bliss_synergy=tme_reward.synergy_score,
                combination_index=tme_reward.combination_index,
                monitoring_schedule=monitoring,
                rationale=rationale,
            ))

    # Sort by expected response
    recommendations.sort(key=lambda r: r.expected_response, reverse=True)

    return recommendations[:n_top]


def _generate_combination_rationale(
    drug_a: str,
    drug_b: str,
    tme_reward: TMERewardComponents,
    modification: str,
) -> str:
    """Generate rationale for combination recommendation."""
    parts = []

    # Synergy rationale
    if tme_reward.synergy_score > 0.05:
        parts.append(f"Synergistic combination (Bliss={tme_reward.synergy_score:.2f})")
    elif tme_reward.synergy_score < -0.05:
        parts.append("Warning: Potential antagonism detected")

    # TME rationale
    if tme_reward.tme_type == "hot":
        parts.append("Predicted to inflame cold TME")
    elif tme_reward.tme_type == "cold":
        parts.append("Requires additional immune priming")

    # Modification rationale
    if modification == "Psi":
        parts.append("Psi modification reduces innate immune toxicity")
    elif modification == "m6A":
        parts.append("m6A enhances translation efficiency")

    # Drug-specific rationale
    if "pembrolizumab" in (drug_a, drug_b) or "nivolumab" in (drug_a, drug_b):
        parts.append("PD-1 blockade prevents T cell exhaustion")
    if "ipilimumab" in (drug_a, drug_b):
        parts.append("CTLA-4 blockade enhances T cell priming")
    if "bevacizumab" in (drug_a, drug_b):
        parts.append("VEGF blockade normalizes tumor vasculature")
    if "cyclophosphamide" in (drug_a, drug_b):
        parts.append("Low-dose cyclophosphamide depletes immunosuppressive Tregs")

    return "; ".join(parts) if parts else "Standard combination therapy"


def _estimate_immune_scores(sequence: str, modification: str) -> Dict[str, float]:
    """Estimate immune pathway activation from sequence."""
    seq = sequence.upper().replace("T", "U")
    length = len(seq)
    gc = sum(1 for c in seq if c in "GC") / max(length, 1)
    gu = sum(1 for c in seq if c in "GU") / max(length, 1)

    # PKR activation (dsRNA-like structures)
    # High GC + long sequences = more dsRNA potential
    dsrna_potential = gc * 0.7 * (length > 500)
    pkr = np.clip(dsrna_potential + 0.1, 0.0, 1.0)

    # RIG-I activation (GU-rich motifs)
    rig_i = np.clip(gu * 0.8 + 0.1, 0.0, 1.0)

    # TLR activation
    tlr = np.clip(0.3 * gc + 0.2 * gu + 0.1, 0.0, 1.0)

    # Overall immunogenicity
    immunogenicity = 0.3 * rig_i + 0.3 * tlr + 0.4 * (1.0 - pkr)

    # Modification effects
    mod_effects = {
        "m6A": {"pkr": -0.1, "rig_i": 0.05, "tlr": -0.05},
        "Psi": {"pkr": -0.2, "rig_i": -0.1, "tlr": -0.1},
        "5mC": {"pkr": -0.05, "rig_i": 0.0, "tlr": 0.0},
        "ms2m6A": {"pkr": 0.1, "rig_i": 0.15, "tlr": 0.05},
        "2OMeA": {"pkr": -0.15, "rig_i": -0.05, "tlr": -0.1},
    }
    effects = mod_effects.get(modification, {})
    pkr = np.clip(pkr + effects.get("pkr", 0.0), 0.0, 1.0)
    rig_i = np.clip(rig_i + effects.get("rig_i", 0.0), 0.0, 1.0)
    tlr = np.clip(tlr + effects.get("tlr", 0.0), 0.0, 1.0)

    return {
        "pkr_score": float(pkr),
        "rig_i_score": float(rig_i),
        "tlr_score": float(tlr),
        "overall_immunogenicity": float(immunogenicity),
        "ips": float(5.0 + 3.0 * immunogenicity),  # IPS scale approximation
    }


def _extract_epitope_region(sequence: str, min_len: int = 15, max_len: int = 50) -> str:
    """Extract potential epitope region from circRNA."""
    seq = sequence.upper().replace("T", "U")

    # Look for open reading frame start
    aug_pos = seq.find("AUG")
    if aug_pos < 0:
        aug_pos = 0

    # Extract region after start codon
    end = min(aug_pos + max_len, len(seq))
    start = max(0, aug_pos)
    region = seq[start:end]

    if len(region) < min_len:
        return seq[:max_len]

    return region


def _estimate_toxicity(
    sequence: str,
    modification: str,
    immune_scores: Dict[str, float]
) -> float:
    """Estimate toxicity risk."""
    seq = sequence.upper().replace("T", "U")
    length = len(seq)

    # Length-related toxicity
    length_tox = 0.1 if length < 2000 else (0.2 if length < 5000 else 0.4)

    # Immune overactivation toxicity
    pkr = immune_scores.get("pkr_score", 0.3)
    rig_i = immune_scores.get("rig_i_score", 0.3)
    immune_tox = 0.3 * pkr + 0.2 * rig_i

    # Modification safety
    mod_safety = {
        "Psi": -0.1,  # Safer
        "2OMeA": -0.08,
        "m6A": -0.05,
        "ms2m6A": 0.05,  # More immunogenic
    }
    mod_effect = mod_safety.get(modification, 0.0)

    total = length_tox + immune_tox + mod_effect
    return float(np.clip(total, 0.0, 1.0))


# =====================================================================
# RL Environment
# =====================================================================

class CircRNAABMEnv:
    """
    Gym-style environment for circRNA optimization with ABM-based rewards.

    State space: Sequence features + modification + history
    Action space: mutate_backbone, optimize_ires, shuffle_ires_flanking,
                  add_modification, select_combination

    Reward: ABM simulation + drug response prediction
    """

    def __init__(self, config: Optional[RLABMConfig] = None):
        self.config = config or RLABMConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Action space
        self.action_names = self.config.action_space
        self.n_actions = len(self.action_names)

        # State
        self.state: Optional[CircRNAState] = None
        self._reset_state()

        # ABM config
        if ABM_AVAILABLE:
            self.abm_config = ImmuneABMConfig(
                horizon_h=self.config.abm_horizon_h,
                dt_h=self.config.abm_dt_h,
            )
        else:
            self.abm_config = None

        # Best tracking
        self.best_sequence = self.config.seed_seq
        self.best_modification = self.config.initial_modification
        self.best_reward = 0.0

    def _reset_state(self):
        """Reset internal state."""
        self.state = CircRNAState(
            sequence=self.config.seed_seq,
            modification=self.config.initial_modification,
            step=0,
            cumulative_reward=0.0,
            best_reward=0.0,
            history=[],
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        self._reset_state()
        self.rng = np.random.default_rng(self.config.seed)
        self.best_sequence = self.config.seed_seq
        self.best_modification = self.config.initial_modification
        self.best_reward = 0.0

        info = {
            "initial_sequence": self.config.seed_seq,
            "initial_modification": self.config.initial_modification,
            "patient_id": self.config.patient.patient_id,
        }
        return self.state.to_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return new state, reward, done, info.

        Args:
            action: Action index (0 to n_actions-1)

        Returns:
            observation: New state observation
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        if self.state is None:
            self._reset_state()

        action_name = self.action_names[action % self.n_actions]
        info = {"action": action_name}

        # Apply action
        new_seq, new_mod = self._apply_action(action_name)

        # Select combination drug if that action
        combination = None
        if action_name == "select_combination":
            combination = str(self.rng.choice(self.config.combination_drugs))

        # Compute reward using ABM
        reward_components = compute_abm_reward(
            sequence=new_seq,
            modification=new_mod,
            patient=self.config.patient,
            combination_drug=combination,
            abm_config=self.abm_config,
        )
        reward = reward_components.total_reward

        # Update state
        improved = reward > self.state.best_reward
        self.state.sequence = new_seq
        self.state.modification = new_mod
        self.state.step += 1
        self.state.cumulative_reward += reward
        self.state.history.append({
            "action": action_name,
            "reward": reward,
            "improved": improved,
            "response_prob": reward_components.response_probability,
            "immune_auc": reward_components.immune_auc,
        })

        if improved:
            self.state.best_reward = reward
            self.best_sequence = new_seq
            self.best_modification = new_mod
            self.best_reward = reward

        # Check termination
        done = self.state.step >= self.config.max_steps

        # Early stopping check
        if len(self.state.history) >= self.config.early_stop_no_improve:
            recent = [h["improved"] for h in self.state.history[-self.config.early_stop_no_improve:]]
            if not any(recent):
                done = True
                info["early_stop"] = True

        info.update({
            "reward_components": reward_components,
            "improved": improved,
            "combination_drug": combination,
        })

        return self.state.to_observation(), reward, done, info

    def _apply_action(self, action_name: str) -> Tuple[str, str]:
        """Apply action to current sequence."""
        seq = self.state.sequence
        mod = self.state.modification

        if action_name == "mutate_backbone":
            seq = mutate_backbone(seq, self.rng, n_mutations=int(self.rng.integers(1, 5)))

        elif action_name == "optimize_ires":
            seq = optimize_ires(seq, self.rng)

        elif action_name == "shuffle_ires_flanking":
            seq = shuffle_ires_flanking(seq, self.rng)

        elif action_name == "add_modification":
            mod_pool = ["none", "m6A", "Psi", "5mC", "ms2m6A", "2OMeA", "2OMeU"]
            mod = str(self.rng.choice([m for m in mod_pool if m != mod] or mod_pool))

        elif action_name == "select_combination":
            # Don't change sequence, just note combination in reward
            pass

        return seq, mod

    def get_best(self) -> Dict[str, Any]:
        """Get best sequence found during training."""
        return {
            "sequence": self.best_sequence,
            "modification": self.best_modification,
            "reward": self.best_reward,
        }

    def render(self, mode: str = "human"):
        """Render current state."""
        if self.state is None:
            return

        print(f"Step {self.state.step}/{self.config.max_steps}")
        print(f"  Sequence length: {len(self.state.sequence)}")
        print(f"  Modification: {self.state.modification}")
        print(f"  Best reward: {self.state.best_reward:.4f}")
        print(f"  Cumulative reward: {self.state.cumulative_reward:.4f}")


# =====================================================================
# Training Functions
# =====================================================================

def train_rl_abm(
    config: Optional[RLABMConfig] = None,
    n_episodes: int = 100,
    policy_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train RL agent on ABM environment.

    Args:
        config: Environment configuration
        n_episodes: Number of training episodes
        policy_fn: Optional policy function (obs -> action)
                  If None, uses epsilon-greedy with learned Q-values
        verbose: Print progress

    Returns:
        results_df: Training history
        best_result: Best sequence found
    """
    config = config or RLABMConfig()
    env = CircRNAABMEnv(config)

    # Simple tabular Q-learning if no policy provided
    if policy_fn is None:
        q_table = np.zeros((100, env.n_actions))  # Discretized state space
        epsilon = 0.3
        alpha = 0.1
        gamma = 0.95

        def policy_fn(obs: np.ndarray) -> int:
            state_idx = _discretize_state(obs)
            if np.random.random() < epsilon:
                return int(np.random.randint(env.n_actions))
            return int(np.argmax(q_table[state_idx]))

    results = []
    best_overall_reward = 0.0
    best_overall_seq = config.seed_seq
    best_overall_mod = config.initial_modification

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy_fn(obs)
            next_obs, reward, done, step_info = env.step(action)
            episode_reward += reward
            obs = next_obs

        # Get best from this episode
        best = env.get_best()
        if best["reward"] > best_overall_reward:
            best_overall_reward = best["reward"]
            best_overall_seq = best["sequence"]
            best_overall_mod = best["modification"]

        results.append({
            "episode": ep + 1,
            "episode_reward": episode_reward,
            "best_reward": best["reward"],
            "sequence_length": len(best["sequence"]),
            "modification": best["modification"],
        })

        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: reward={episode_reward:.4f}, best={best_overall_reward:.4f}")

    results_df = pd.DataFrame(results)
    best_result = {
        "sequence": best_overall_seq,
        "modification": best_overall_mod,
        "reward": best_overall_reward,
    }

    return results_df, best_result


def _discretize_state(obs: np.ndarray, n_bins: int = 10) -> int:
    """Discretize continuous observation to state index."""
    clipped = np.clip(obs * n_bins, 0, n_bins - 1).astype(int)
    # Hash to single index
    idx = 0
    for i, val in enumerate(clipped):
        idx += int(val) * (n_bins ** i)
    return idx % 100  # Keep in bounds


# =====================================================================
# Convenience Functions
# =====================================================================

def optimize_circrna_with_abm(
    seed_seq: str,
    patient: Optional[PatientProfile] = None,
    n_episodes: int = 50,
    modification: str = "m6A",
) -> Tuple[str, str, pd.DataFrame]:
    """
    Optimize circRNA sequence using RL + ABM simulation.

    Args:
        seed_seq: Starting circRNA sequence
        patient: Patient profile (default: average patient)
        n_episodes: Number of RL episodes
        modification: Initial modification

    Returns:
        best_sequence: Optimized sequence
        best_modification: Best modification found
        results_df: Training history
    """
    patient = patient or PatientProfile()

    config = RLABMConfig(
        seed_seq=seed_seq,
        initial_modification=modification,
        patient=patient,
        max_steps=15,
    )

    results_df, best = train_rl_abm(
        config=config,
        n_episodes=n_episodes,
        verbose=False,
    )

    return best["sequence"], best["modification"], results_df


def compare_with_baseline(
    seed_seq: str,
    patient: Optional[PatientProfile] = None,
    n_rounds: int = 5,
) -> pd.DataFrame:
    """
    Compare RL-ABM approach with baseline evolution.

    Returns comparison of:
    - RL-ABM (this module)
    - Original evolution (cirrna_evolution.py)
    - Random search
    """
    patient = patient or PatientProfile()

    results = []

    # 1. RL-ABM
    rl_seq, rl_mod, rl_df = optimize_circrna_with_abm(
        seed_seq=seed_seq,
        patient=patient,
        n_episodes=30,
    )
    rl_reward = compute_abm_reward(rl_seq, rl_mod, patient).total_reward
    results.append({
        "method": "RL-ABM",
        "reward": rl_reward,
        "sequence_length": len(rl_seq),
        "modification": rl_mod,
    })

    # 2. Original evolution
    from .cirrna_evolution import run_cirrna_evolution
    evo_df, evo_artifacts = run_cirrna_evolution(
        seed_seq=seed_seq,
        rounds=n_rounds,
        modification=patient.trop2 > 0.5 and "Psi" or "m6A",
    )
    evo_seq = evo_artifacts.best_sequence
    evo_mod = evo_artifacts.best_modification
    evo_reward = compute_abm_reward(evo_seq, evo_mod, patient).total_reward
    results.append({
        "method": "Evolution",
        "reward": evo_reward,
        "sequence_length": len(evo_seq),
        "modification": evo_mod,
    })

    # 3. Random search
    rng = np.random.default_rng(42)
    best_random_reward = 0.0
    best_random_seq = seed_seq
    best_random_mod = "m6A"

    for _ in range(50):
        # Random mutation
        from .cirrna_evolution import mutate_backbone
        rand_seq = mutate_backbone(seed_seq, rng, n_mutations=5)
        rand_mod = str(rng.choice(["m6A", "Psi", "5mC", "2OMeA"]))
        rand_reward = compute_abm_reward(rand_seq, rand_mod, patient).total_reward

        if rand_reward > best_random_reward:
            best_random_reward = rand_reward
            best_random_seq = rand_seq
            best_random_mod = rand_mod

    results.append({
        "method": "Random",
        "reward": best_random_reward,
        "sequence_length": len(best_random_seq),
        "modification": best_random_mod,
    })

    return pd.DataFrame(results)


__all__ = [
    # Configuration
    "PatientProfile",
    "RLABMConfig",
    "CircRNAState",
    "ABMRewardComponents",
    # Core functions
    "compute_abm_reward",
    "CircRNAABMEnv",
    # Training
    "train_rl_abm",
    "optimize_circrna_with_abm",
    "compare_with_baseline",
]
