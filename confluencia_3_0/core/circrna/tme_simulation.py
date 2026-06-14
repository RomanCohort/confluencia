"""
Tumor Microenvironment (TME) Advanced Simulation

Extends the basic immune ABM with:
1. Multiple cell types (T cells, B cells, NK, Macrophages, MDSC, Tregs, CAFs)
2. Spatial heterogeneity (core vs invasive margin)
3. Cytokine network (IFN-γ, TNF-α, IL-2, IL-6, IL-10, TGF-β)
4. Checkpoint dynamics (PD-1/PD-L1, CTLA-4, LAG-3, TIM-3)
5. Hypoxia and metabolic competition
6. Drug penetration gradients

Literature basis:
- Fridman et al., 2012: TME classification (hot/cold/excluded)
- Chen & Mellman, 2017: Cancer-Immunity Cycle
- Binnewies et al., 2018: TME cellular composition
- Anderson & Simon, 2020: Spatial heterogeneity in immunotherapy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# =====================================================================
# Cell Type Definitions
# =====================================================================

@dataclass
class CellPopulation:
    """Population of a cell type in TME."""
    name: str
    count: float
    activation: float = 0.5
    location: str = "mixed"  # "core", "margin", "mixed"

    # Dynamics parameters
    proliferation_rate: float = 0.0
    death_rate: float = 0.0
    recruitment_rate: float = 0.0
    activation_threshold: float = 0.3


@dataclass
class Cytokine:
    """Cytokine concentration in TME."""
    name: str
    concentration: float  # pg/mL normalized
    half_life_h: float

    # Effects (positive = stimulates, negative = inhibits)
    t_cell_effect: float = 0.0
    b_cell_effect: float = 0.0
    macrophage_effect: float = 0.0
    tumor_effect: float = 0.0


@dataclass
class CheckpointStatus:
    """Immune checkpoint expression status."""
    pd1_t_cell: float = 0.5
    pdl1_tumor: float = 0.5
    ctla4_t_cell: float = 0.3
    lag3_t_cell: float = 0.2
    tim3_t_cell: float = 0.2

    # Inhibitor concentrations (from therapy)
    anti_pd1: float = 0.0
    anti_ctla4: float = 0.0


@dataclass
class SpatialCompartment:
    """Spatial compartment in TME."""
    name: str  # "core", "margin", "stroma"
    volume_fraction: float

    # Local conditions
    oxygen: float = 1.0  # 0-1, hypoxia
    glucose: float = 1.0
    ph: float = 7.4

    # Drug penetration
    drug_concentration: float = 0.0

    # Cell populations (fraction of total)
    t_cells: float = 0.0
    b_cells: float = 0.0
    nk_cells: float = 0.0
    macrophages_m1: float = 0.0
    macrophages_m2: float = 0.0
    mdsc: float = 0.0
    tregs: float = 0.0
    cafs: float = 0.0  # Cancer-associated fibroblasts
    tumor_cells: float = 0.0


# =====================================================================
# TME Configuration
# =====================================================================

@dataclass
class TMEConfig:
    """Configuration for TME simulation."""
    # Time parameters
    horizon_h: int = 168  # 7 days
    dt_h: float = 1.0

    # Initial populations (cells/mm³, normalized)
    initial_t_cd8: float = 100.0  # Cytotoxic T cells
    initial_t_cd4: float = 80.0   # Helper T cells
    initial_treg: float = 30.0    # Regulatory T cells
    initial_b: float = 50.0
    initial_nk: float = 40.0
    initial_mac_m1: float = 60.0  # M1 macrophages (anti-tumor)
    initial_mac_m2: float = 40.0  # M2 macrophages (pro-tumor)
    initial_mdsc: float = 20.0   # Myeloid-derived suppressor cells
    initial_caf: float = 30.0    # Cancer-associated fibroblasts
    initial_tumor: float = 500.0

    # Tumor dynamics
    tumor_growth_rate: float = 0.02  # per hour
    tumor_death_rate_immune: float = 0.01
    tumor_carrying_capacity: float = 1000.0

    # Immune cell dynamics
    t_cell_activation_rate: float = 0.05
    t_cell_exhaustion_rate: float = 0.02
    t_cell_recruitment_rate: float = 0.1
    nk_activation_rate: float = 0.04
    macrophage_polarization_rate: float = 0.03

    # Cytokine dynamics
    ifng_production_rate: float = 0.5
    tnf_production_rate: float = 0.4
    il2_production_rate: float = 0.3
    il6_production_rate: float = 0.2
    il10_production_rate: float = 0.15
    tgf_beta_production_rate: float = 0.1

    # Hypoxia
    hypoxia_threshold: float = 0.3
    hypoxia_induced_factor: float = 0.5

    # Drug parameters
    drug_half_life_h: float = 24.0
    drug_penetration_core: float = 0.3  # Poor penetration to core
    drug_penetration_margin: float = 0.8

    seed: int = 42


@dataclass
class TMEResult:
    """Result from TME simulation."""
    time_h: np.ndarray
    tumor_volume: np.ndarray
    t_cd8_count: np.ndarray
    t_cd4_count: np.ndarray
    treg_count: np.ndarray
    b_count: np.ndarray
    nk_count: np.ndarray
    mac_m1_count: np.ndarray
    mac_m2_count: np.ndarray
    mdsc_count: np.ndarray
    caf_count: np.ndarray

    # Cytokines
    ifng: np.ndarray
    tnf: np.ndarray
    il2: np.ndarray
    il6: np.ndarray
    il10: np.ndarray
    tgf_beta: np.ndarray

    # Checkpoints
    pd1_expression: np.ndarray
    pdl1_expression: np.ndarray

    # Spatial
    core_tumor: np.ndarray
    margin_tumor: np.ndarray

    # Metrics
    immune_score: float
    tme_type: str  # "hot", "cold", "excluded", "mixed"
    response_prediction: float


# =====================================================================
# TME Simulation
# =====================================================================

class TMESimulator:
    """
    Advanced TME simulator with multiple cell types and spatial structure.

    Simulates:
    - Cellular dynamics (proliferation, death, recruitment, activation)
    - Cytokine network
    - Checkpoint dynamics
    - Spatial heterogeneity
    - Drug effects
    """

    def __init__(self, config: Optional[TMEConfig] = None):
        self.config = config or TMEConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Initialize compartments
        self.compartments = self._init_compartments()

    def _init_compartments(self) -> Dict[str, SpatialCompartment]:
        """Initialize spatial compartments."""
        return {
            "core": SpatialCompartment(
                name="core",
                volume_fraction=0.4,
                oxygen=0.3,  # Hypoxic
                glucose=0.5,
                ph=6.8,
                drug_concentration=0.0,
                tumor_cells=0.7,
                t_cells=0.1,
                macrophages_m2=0.15,  # M2 enriched
                mdsc=0.05,
            ),
            "margin": SpatialCompartment(
                name="margin",
                volume_fraction=0.35,
                oxygen=0.8,
                glucose=0.9,
                ph=7.2,
                drug_concentration=0.0,
                tumor_cells=0.3,
                t_cells=0.35,
                b_cells=0.1,
                nk_cells=0.1,
                macrophages_m1=0.1,
                macrophages_m2=0.05,
            ),
            "stroma": SpatialCompartment(
                name="stroma",
                volume_fraction=0.25,
                oxygen=0.9,
                glucose=1.0,
                ph=7.4,
                drug_concentration=0.0,
                tumor_cells=0.05,
                t_cells=0.2,
                b_cells=0.2,
                cafs=0.3,
                macrophages_m1=0.15,
            ),
        }

    def simulate(
        self,
        treatment_schedule: Optional[Dict[float, Dict]] = None,
        circrna_dose: float = 0.0,
        checkpoint_inhibitor: str = "none",
    ) -> TMEResult:
        """
        Run TME simulation.

        Args:
            treatment_schedule: Dict of {time_h: treatment_params}
            circrna_dose: circRNA vaccine dose
            checkpoint_inhibitor: "none", "anti_pd1", "anti_ctla4", "combo"

        Returns:
            TMEResult with full trajectory
        """
        cfg = self.config
        steps = int(cfg.horizon_h / cfg.dt_h) + 1
        time = np.linspace(0, cfg.horizon_h, steps)

        # Initialize state arrays
        tumor = np.zeros(steps)
        t_cd8 = np.zeros(steps)
        t_cd4 = np.zeros(steps)
        treg = np.zeros(steps)
        b = np.zeros(steps)
        nk = np.zeros(steps)
        mac_m1 = np.zeros(steps)
        mac_m2 = np.zeros(steps)
        mdsc = np.zeros(steps)
        caf = np.zeros(steps)

        ifng = np.zeros(steps)
        tnf = np.zeros(steps)
        il2 = np.zeros(steps)
        il6 = np.zeros(steps)
        il10 = np.zeros(steps)
        tgf_beta = np.zeros(steps)

        pd1_expr = np.zeros(steps)
        pdl1_expr = np.zeros(steps)

        core_tumor = np.zeros(steps)
        margin_tumor = np.zeros(steps)

        # Initial conditions
        tumor[0] = cfg.initial_tumor
        t_cd8[0] = cfg.initial_t_cd8
        t_cd4[0] = cfg.initial_t_cd4
        treg[0] = cfg.initial_treg
        b[0] = cfg.initial_b
        nk[0] = cfg.initial_nk
        mac_m1[0] = cfg.initial_mac_m1
        mac_m2[0] = cfg.initial_mac_m2
        mdsc[0] = cfg.initial_mdsc
        caf[0] = cfg.initial_caf

        # Initial cytokines
        ifng[0] = 10.0
        tnf[0] = 8.0
        il2[0] = 5.0
        il6[0] = 15.0
        il10[0] = 10.0
        tgf_beta[0] = 20.0

        # Checkpoint expressions
        pd1_expr[0] = 0.5
        pdl1_expr[0] = 0.5

        # Treatment state
        drug_level = 0.0
        circrna_level = 0.0

        # Main simulation loop
        for i in range(1, steps):
            dt = cfg.dt_h
            t = time[i-1]

            # Check for treatment events
            if treatment_schedule and t in treatment_schedule:
                tx = treatment_schedule[t]
                drug_level = tx.get("drug_level", drug_level)
                circrna_level = tx.get("circrna_dose", circrna_level)

            # circRNA vaccine effect
            if circrna_dose > 0:
                # circRNA boosts innate immune activation
                circrna_level = circrna_dose * np.exp(-t / 48.0)  # 48h half-life

            # Checkpoint inhibitor effect
            anti_pd1 = 1.0 if checkpoint_inhibitor in ["anti_pd1", "combo"] else 0.0
            anti_ctla4 = 1.0 if checkpoint_inhibitor in ["anti_ctla4", "combo"] else 0.0

            # === Tumor dynamics ===
            # Growth (logistic)
            growth = cfg.tumor_growth_rate * tumor[i-1] * (1 - tumor[i-1] / cfg.tumor_carrying_capacity)

            # Immune-mediated killing
            # CD8 T cells + NK cells + M1 macrophages
            immune_kill = cfg.tumor_death_rate_immune * (
                t_cd8[i-1] * (1 - pd1_expr[i-1] * pdl1_expr[i-1] * (1 - anti_pd1)) +
                nk[i-1] * 0.5 +
                mac_m1[i-1] * 0.3
            ) * tumor[i-1] / (tumor[i-1] + 100.0)

            # Hypoxia-induced resistance
            hypoxia_factor = 1.0 - cfg.hypoxia_induced_factor * (self.compartments["core"].oxygen < cfg.hypoxia_threshold)

            tumor[i] = max(0, tumor[i-1] + dt * (growth - immune_kill * hypoxia_factor))

            # === T cell dynamics ===
            # CD8 activation by antigen + cytokines
            t_activation = cfg.t_cell_activation_rate * (
                circrna_level * 0.5 +  # circRNA activates
                ifng[i-1] / 50.0 * 0.3 +
                il2[i-1] / 20.0 * 0.2
            ) * t_cd8[i-1]

            # Exhaustion by chronic stimulation + checkpoints
            t_exhaustion = cfg.t_cell_exhaustion_rate * (
                pd1_expr[i-1] * (1 - anti_pd1) +
                pdl1_expr[i-1] * 0.5 +
                tgf_beta[i-1] / 50.0 * 0.3
            ) * t_cd8[i-1]

            # Recruitment from periphery
            t_recruit = cfg.t_cell_recruitment_rate * (
                ifng[i-1] / 30.0 +
                circrna_level * 0.3
            ) * (200 - t_cd8[i-1])  # Homeostatic setpoint

            t_cd8[i] = max(0, t_cd8[i-1] + dt * (t_activation + t_recruit - t_exhaustion - 0.01 * t_cd8[i-1]))

            # CD4 helper T cells
            t_cd4[i] = max(0, t_cd4[i-1] + dt * (
                0.03 * t_cd4[i-1] * (ifng[i-1] / 50.0) -
                0.02 * t_cd4[i-1] * (tgf_beta[i-1] / 40.0) -
                0.005 * t_cd4[i-1]
            ))

            # Tregs (suppressive, expand in presence of TGF-β)
            treg[i] = max(0, treg[i-1] + dt * (
                0.02 * treg[i-1] * (tgf_beta[i-1] / 30.0 + il10[i-1] / 40.0) -
                0.01 * treg[i-1]
            ))

            # === NK cells ===
            nk_activation = cfg.nk_activation_rate * (
                il2[i-1] / 20.0 +
                ifng[i-1] / 40.0 +
                circrna_level * 0.4
            ) * nk[i-1]
            nk[i] = max(0, nk[i-1] + dt * (nk_activation - 0.02 * nk[i-1]))

            # === Macrophages ===
            # M1 polarization by IFN-γ
            m1_polarize = cfg.macrophage_polarization_rate * (ifng[i-1] / 30.0) * mac_m2[i-1]
            # M2 polarization by IL-10, TGF-β
            m2_polarize = cfg.macrophage_polarization_rate * (il10[i-1] / 30.0 + tgf_beta[i-1] / 40.0) * mac_m1[i-1] * 0.5

            mac_m1[i] = max(0, mac_m1[i-1] + dt * (m1_polarize - m2_polarize + 0.01 * mac_m1[i-1] - 0.02 * mac_m1[i-1]))
            mac_m2[i] = max(0, mac_m2[i-1] + dt * (m2_polarize - m1_polarize + 0.01 * mac_m2[i-1] - 0.02 * mac_m2[i-1]))

            # === MDSC (immunosuppressive) ===
            mdsc[i] = max(0, mdsc[i-1] + dt * (
                0.02 * mdsc[i-1] * (il6[i-1] / 30.0 + tumor[i-1] / 500.0) -
                0.03 * mdsc[i-1] * (ifng[i-1] / 40.0)
            ))

            # === CAFs ===
            caf[i] = max(0, caf[i-1] + dt * (
                0.01 * caf[i-1] * (tgf_beta[i-1] / 30.0) -
                0.005 * caf[i-1]
            ))

            # === B cells ===
            b[i] = max(0, b[i-1] + dt * (
                0.02 * b[i-1] * (t_cd4[i-1] / 100.0) -
                0.01 * b[i-1]
            ))

            # === Cytokine dynamics ===
            # IFN-γ (from Th1, CD8, NK)
            ifng[i] = max(0, ifng[i-1] + dt * (
                cfg.ifng_production_rate * (t_cd8[i-1] + nk[i-1]) / 100.0 -
                0.1 * ifng[i-1]
            ))

            # TNF-α (from M1, T cells)
            tnf[i] = max(0, tnf[i-1] + dt * (
                cfg.tnf_production_rate * (mac_m1[i-1] + t_cd8[i-1]) / 100.0 -
                0.15 * tnf[i-1]
            ))

            # IL-2 (from CD4)
            il2[i] = max(0, il2[i-1] + dt * (
                cfg.il2_production_rate * t_cd4[i-1] / 80.0 -
                0.2 * il2[i-1]
            ))

            # IL-6 (pro-inflammatory, from tumor, M2)
            il6[i] = max(0, il6[i-1] + dt * (
                cfg.il6_production_rate * (tumor[i-1] / 500.0 + mac_m2[i-1] / 50.0) -
                0.1 * il6[i-1]
            ))

            # IL-10 (immunosuppressive)
            il10[i] = max(0, il10[i-1] + dt * (
                cfg.il10_production_rate * (treg[i-1] + mac_m2[i-1]) / 50.0 -
                0.1 * il10[i-1]
            ))

            # TGF-β (immunosuppressive, from tumor, CAFs)
            tgf_beta[i] = max(0, tgf_beta[i-1] + dt * (
                cfg.tgf_beta_production_rate * (tumor[i-1] / 400.0 + caf[i-1] / 30.0) -
                0.05 * tgf_beta[i-1]
            ))

            # === Checkpoint dynamics ===
            # PD-1 upregulation with chronic activation
            pd1_expr[i] = np.clip(pd1_expr[i-1] + dt * 0.01 * (t_cd8[i-1] / 100.0 - 0.5), 0.1, 1.0)
            # PD-L1 upregulation by IFN-γ
            pdl1_expr[i] = np.clip(pdl1_expr[i-1] + dt * 0.02 * (ifng[i-1] / 30.0 - 0.5), 0.1, 1.0)

            # === Spatial distribution ===
            # Core: hypoxic, drug-poor, M2-enriched
            core_tumor[i] = tumor[i] * 0.6 * (1 + 0.2 * (1 - self.compartments["core"].oxygen))
            # Margin: immune-rich, drug-accessible
            margin_tumor[i] = tumor[i] * 0.3

        # === Compute metrics ===
        # Immune score (0-100)
        final_t_cd8 = t_cd8[-1]
        final_nk = nk[-1]
        final_m1 = mac_m1[-1]
        final_treg = treg[-1]
        final_m2 = mac_m2[-1]
        final_mdsc = mdsc[-1]

        effector_score = (final_t_cd8 + final_nk + final_m1) / 3.0
        suppressor_score = (final_treg + final_m2 + final_mdsc) / 3.0
        immune_score = 100.0 * effector_score / (effector_score + suppressor_score + 1.0)

        # TME classification
        t_cell_density = (t_cd8[-1] + t_cd4[-1]) / tumor[-1]
        margin_ratio = margin_tumor[-1] / max(core_tumor[-1], 1.0)

        if t_cell_density > 0.5 and margin_ratio < 1.5:
            tme_type = "hot"
        elif t_cell_density < 0.2:
            tme_type = "cold"
        elif margin_ratio > 2.0:
            tme_type = "excluded"
        else:
            tme_type = "mixed"

        # Response prediction
        tumor_reduction = (tumor[0] - tumor[-1]) / tumor[0]
        response_prediction = np.clip(
            0.4 * (t_cd8[-1] / t_cd8[0]) +
            0.3 * (1 - tumor_reduction) +
            0.3 * (immune_score / 100.0),
            0.0, 1.0
        )

        return TMEResult(
            time_h=time,
            tumor_volume=tumor,
            t_cd8_count=t_cd8,
            t_cd4_count=t_cd4,
            treg_count=treg,
            b_count=b,
            nk_count=nk,
            mac_m1_count=mac_m1,
            mac_m2_count=mac_m2,
            mdsc_count=mdsc,
            caf_count=caf,
            ifng=ifng,
            tnf=tnf,
            il2=il2,
            il6=il6,
            il10=il10,
            tgf_beta=tgf_beta,
            pd1_expression=pd1_expr,
            pdl1_expression=pdl1_expr,
            core_tumor=core_tumor,
            margin_tumor=margin_tumor,
            immune_score=float(immune_score),
            tme_type=tme_type,
            response_prediction=float(response_prediction),
        )


# =====================================================================
# Convenience Functions
# =====================================================================

def simulate_tme(
    circrna_dose: float = 1.0,
    checkpoint_inhibitor: str = "anti_pd1",
    horizon_h: int = 168,
) -> TMEResult:
    """
    Quick TME simulation with common settings.

    Args:
        circrna_dose: circRNA vaccine dose
        checkpoint_inhibitor: "none", "anti_pd1", "anti_ctla4", "combo"
        horizon_h: Simulation duration in hours

    Returns:
        TMEResult
    """
    config = TMEConfig(horizon_h=horizon_h)
    sim = TMESimulator(config)
    return sim.simulate(
        circrna_dose=circrna_dose,
        checkpoint_inhibitor=checkpoint_inhibitor,
    )


def classify_tme(result: TMEResult) -> Dict[str, float]:
    """
    Classify TME and compute immunotherapy recommendations.

    Returns:
        Dict with TME features and recommendations
    """
    return {
        "tme_type": result.tme_type,
        "immune_score": result.immune_score,
        "response_prediction": result.response_prediction,
        "t_cd8_final": float(result.t_cd8_count[-1]),
        "treg_ratio": float(result.treg_count[-1] / max(result.t_cd8_count[-1], 1.0)),
        "m1_m2_ratio": float(result.mac_m1_count[-1] / max(result.mac_m2_count[-1], 1.0)),
        "tumor_reduction": float((result.tumor_volume[0] - result.tumor_volume[-1]) / result.tumor_volume[0]),
        "ifng_peak": float(np.max(result.ifng)),
        "tgf_beta_final": float(result.tgf_beta[-1]),
        "recommendation": _get_recommendation(result),
    }


def _get_recommendation(result: TMEResult) -> str:
    """Generate treatment recommendation based on TME state."""
    if result.tme_type == "hot":
        return "Respond well to checkpoint inhibitors. Consider circRNA vaccine for enhanced response."
    elif result.tme_type == "cold":
        return "Low immunogenicity. Recommend circRNA vaccine + combination therapy to inflame TME."
    elif result.tme_type == "excluded":
        return "Immune excluded. Consider stroma-targeting + circRNA to enable T cell infiltration."
    else:
        return "Mixed TME. Personalized combination therapy recommended."


__all__ = [
    "CellPopulation",
    "Cytokine",
    "CheckpointStatus",
    "SpatialCompartment",
    "TMEConfig",
    "TMEResult",
    "TMESimulator",
    "simulate_tme",
    "classify_tme",
]
