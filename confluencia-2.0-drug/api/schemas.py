"""Pydantic request/response schemas for Confluencia 2.0 Drug API.

Mirrors the core dataclass structures for JSON serialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Artifacts & Report Schemas
# ---------------------------------------------------------------------------


class ArtifactsSchema(BaseModel):
    compute_profile: str = "low"
    moe_weights: Dict[str, float] = Field(default_factory=dict)
    moe_metrics: Dict[str, float] = Field(default_factory=dict)
    used_proxy_micro_labels: bool = False
    smiles_backend: str = "hash"
    ctm_param_source: str = "heuristic"
    model_backend: str = "moe"
    dynamics_model: str = "ctm"
    shap_ready: bool = False
    shap_message: str = ""
    shap_feature_count: int = 0
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.0
    adaptive_samples: int = 0
    adaptive_message: str = "disabled"


class TrainingReportSchema(BaseModel):
    sample_count: int = 0
    used_labels: Dict[str, bool] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)


class EvolutionArtifactsSchema(BaseModel):
    final_policy_logits: Dict[str, float] = Field(default_factory=dict)
    reflections: List[str] = Field(default_factory=list)
    used_ed2mol: bool = False
    selected_objective_weights: Dict[str, float] = Field(default_factory=dict)
    rounds_ran: int = 0
    best_reward: float = 0.0
    per_round_best: List[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Drug Config Schemas
# ---------------------------------------------------------------------------


class DrugTrainPredictRequest(BaseModel):
    data_csv: str
    compute_mode: str = "auto"
    model_backend: str = "moe"
    dynamics_model: str = "ctm"
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.2


class DrugTrainRequest(BaseModel):
    data_csv: str
    compute_mode: str = "auto"
    model_backend: str = "moe"
    dynamics_model: str = "ctm"
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.2


class DrugPredictRequest(BaseModel):
    model_id: str
    data_csv: str
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.2


class DrugTrainPredictResponse(BaseModel):
    result_csv: str = ""
    curve_csv: str = ""
    artifacts: ArtifactsSchema = Field(default_factory=ArtifactsSchema)
    report: TrainingReportSchema = Field(default_factory=TrainingReportSchema)


class DrugTrainResponse(BaseModel):
    model_id: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class DrugPredictResponse(BaseModel):
    result_csv: str = ""
    curve_csv: str = ""
    artifacts: ArtifactsSchema = Field(default_factory=ArtifactsSchema)
    report: TrainingReportSchema = Field(default_factory=TrainingReportSchema)


# ---------------------------------------------------------------------------
# Model Management Schemas
# ---------------------------------------------------------------------------


class ModelExportRequest(BaseModel):
    model_id: str


class ModelImportResponse(BaseModel):
    model_id: str


class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(default_factory=list)


class ModelMetadataResponse(BaseModel):
    model_id: str
    metadata: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evolution Config Schemas
# ---------------------------------------------------------------------------


class EvolutionConfigSchema(BaseModel):
    rounds: int = 5
    top_k: int = 12
    candidates_per_round: int = 48
    epsilon: float = 0.15
    lr: float = 0.06
    dose: float = 2.0
    freq: float = 1.0
    treatment_time: float = 24.0
    group_id: str = "EVO"
    epitope_seq: str = "SLYNTVATL"
    compute_mode: str = "low"
    use_pareto_search: bool = True
    pareto_weight_samples: int = 64
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.2
    use_adaptive_gate_penalty: bool = True
    risk_gate_threshold: float = 0.70
    risk_gate_penalty: float = 0.20
    risk_gate_threshold_mode: str = "fixed"
    risk_gate_threshold_quantile: float = 0.80

    def to_dataclass(self):
        from core.evolution import EvolutionConfig
        return EvolutionConfig(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj) -> "EvolutionConfigSchema":
        from dataclasses import fields
        return cls(**{f.name: getattr(obj, f.name) for f in fields(obj)})


class CircRNAEvolutionConfigSchema(BaseModel):
    rounds: int = 5
    top_k: int = 8
    candidates_per_round: int = 24
    epsilon: float = 0.15
    lr: float = 0.06
    seed_seq: str = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"
    modification: str = "m6A"
    delivery_vector: str = "LNP_liver"
    route: str = "IV"
    ires_type: str = "EMCV"
    dose: float = 2.0
    freq: float = 1.0
    treatment_time: float = 24.0
    use_pareto_search: bool = True
    pareto_weight_samples: int = 32
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    seed: int = 42

    def to_dataclass(self):
        from core.evolution import CircRNAEvolutionConfig
        return CircRNAEvolutionConfig(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj) -> "CircRNAEvolutionConfigSchema":
        from dataclasses import fields
        return cls(**{f.name: getattr(obj, f.name) for f in fields(obj)})


class EvolveMoleculesRequest(BaseModel):
    seed_smiles: List[str]
    config: EvolutionConfigSchema = Field(default_factory=EvolutionConfigSchema)
    ed2mol_repo_dir: str = ""
    ed2mol_config_path: str = ""
    ed2mol_python_cmd: str = "python"


class EvolveMoleculesResponse(BaseModel):
    results_csv: str = ""
    artifacts: EvolutionArtifactsSchema = Field(default_factory=EvolutionArtifactsSchema)


class EvolveCirrnaRequest(BaseModel):
    config: CircRNAEvolutionConfigSchema = Field(default_factory=CircRNAEvolutionConfigSchema)


class EvolveCirrnaResponse(BaseModel):
    results_csv: str = ""
    artifacts: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trial Config Schemas
# ---------------------------------------------------------------------------


class CohortConfigSchema(BaseModel):
    n_patients: int = 100
    age_range: List[float] = Field(default_factory=lambda: [18.0, 75.0])
    age_mean: float = 55.0
    age_std: float = 12.0
    female_frac: float = 0.45
    weight_mean: float = 70.0
    weight_std: float = 15.0
    disease_stages: List[str] = Field(default_factory=lambda: ["I", "II", "III"])
    stage_probs: List[float] = Field(default_factory=lambda: [0.3, 0.4, 0.3])
    biomarker_positive_frac: float = 0.4
    ecog_scores: List[int] = Field(default_factory=lambda: [0, 1, 2])
    ecog_probs: List[float] = Field(default_factory=lambda: [0.3, 0.5, 0.2])
    seed: int = 42

    def to_dataclass(self):
        from core.trial_sim import CohortConfig
        d = self.model_dump()
        d["age_range"] = tuple(d["age_range"])
        return CohortConfig(**d)


class CohortRequest(BaseModel):
    config: Optional[CohortConfigSchema] = None


class CohortResponse(BaseModel):
    cohort_csv: str = ""


class PhaseIConfigSchema(BaseModel):
    design: str = "3+3"
    dose_levels: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    start_level: int = 0
    max_level: int = 5
    dlt_threshold_3p3: float = 0.33
    dlt_threshold_boin: float = 0.30
    n_per_cohort: int = 3
    seed: int = 42

    def to_dataclass(self):
        from core.trial_sim import PhaseIConfig
        return PhaseIConfig(**self.model_dump())


class PhaseIRequest(BaseModel):
    dlt_prob_fn_name: str = "default_dlt_prob"
    cohort_csv: Optional[str] = None
    cohort_config: Optional[CohortConfigSchema] = None
    config: PhaseIConfigSchema = Field(default_factory=PhaseIConfigSchema)


class PhaseIResponse(BaseModel):
    mtd_estimate: float = 0.0
    rp2d: float = 0.0
    dose_toxicity_csv: str = ""
    dose_levels_tested: List[float] = Field(default_factory=list)
    patients_per_level: List[int] = Field(default_factory=list)
    dlts_per_level: List[int] = Field(default_factory=list)
    decision_log: List[str] = Field(default_factory=list)


class PhaseIIConfigSchema(BaseModel):
    n_arm_treatment: int = 50
    n_arm_control: int = 25
    primary_endpoint: str = "ORR"
    orr_threshold: float = 0.20
    alpha: float = 0.05
    seed: int = 42

    def to_dataclass(self):
        from core.trial_sim import PhaseIIConfig
        return PhaseIIConfig(**self.model_dump())


class PhaseIIRequest(BaseModel):
    efficacy_fn_name: str = "default_efficacy_fn"
    rp2d: float = 4.0
    soc_efficacy: Dict[str, float] = Field(default_factory=lambda: {
        "ORR": 0.15, "DCR": 0.45, "PFS_rate_6mo": 0.30,
    })
    cohort_csv: Optional[str] = None
    cohort_config: Optional[CohortConfigSchema] = None
    config: PhaseIIConfigSchema = Field(default_factory=PhaseIIConfigSchema)


class PhaseIIResponse(BaseModel):
    treatment_arm: Dict[str, float] = Field(default_factory=dict)
    control_arm: Dict[str, float] = Field(default_factory=dict)
    p_value: float = 1.0
    statistically_significant: bool = False
    power_estimate: float = 0.0
    biomarker_subgroup: Dict[str, float] = Field(default_factory=dict)
    km_data_treatment_csv: str = ""
    km_data_control_csv: str = ""


class PhaseIIIConfigSchema(BaseModel):
    n_arm_treatment: int = 300
    n_arm_control: int = 300
    stratification_factors: List[str] = Field(default_factory=lambda: ["disease_stage", "biomarker_positive"])
    alpha: float = 0.05
    seed: int = 42

    def to_dataclass(self):
        from core.trial_sim import PhaseIIIConfig
        return PhaseIIIConfig(**self.model_dump())


class PhaseIIIRequest(BaseModel):
    survival_fn_name: str = "default_survival_fn"
    rp2d: float = 4.0
    soc_median_survival: float = 12.0
    cohort_csv: Optional[str] = None
    cohort_config: Optional[CohortConfigSchema] = None
    config: PhaseIIIConfigSchema = Field(default_factory=PhaseIIIConfigSchema)


class PhaseIIIResponse(BaseModel):
    hazard_ratio: float = 1.0
    hr_ci_lower: float = 1.0
    hr_ci_upper: float = 1.0
    p_value: float = 1.0
    significant: bool = False
    subgroup_analysis_csv: str = ""
    km_data_treatment_csv: str = ""
    km_data_control_csv: str = ""
    median_survival_treatment: float = 0.0
    median_survival_control: float = 0.0


class FullTrialRequest(BaseModel):
    drug_name: str = "circRNA-X"
    cohort_config: Optional[CohortConfigSchema] = None
    phase_i_config: PhaseIConfigSchema = Field(default_factory=PhaseIConfigSchema)
    phase_ii_config: PhaseIIConfigSchema = Field(default_factory=PhaseIIConfigSchema)
    phase_iii_config: PhaseIIIConfigSchema = Field(default_factory=PhaseIIIConfigSchema)
    dlt_prob_fn_name: str = "default_dlt_prob"
    efficacy_fn_name: str = "default_efficacy_fn"
    survival_fn_name: str = "default_survival_fn"
    soc_efficacy: Dict[str, float] = Field(default_factory=lambda: {
        "ORR": 0.15, "DCR": 0.45, "PFS_rate_6mo": 0.30,
    })
    soc_median_survival: float = 12.0


class FullTrialResponse(BaseModel):
    phase_i: PhaseIResponse = Field(default_factory=PhaseIResponse)
    phase_ii: PhaseIIResponse = Field(default_factory=PhaseIIResponse)
    phase_iii: PhaseIIIResponse = Field(default_factory=PhaseIIIResponse)
    report: str = ""


class TrialReportRequest(BaseModel):
    phase_i: Optional[PhaseIResponse] = None
    phase_ii: Optional[PhaseIIResponse] = None
    phase_iii: Optional[PhaseIIIResponse] = None
    cohort_summary_csv: Optional[str] = None
    drug_name: str = "circRNA-X"


class TrialReportResponse(BaseModel):
    report: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0"
