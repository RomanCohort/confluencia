"""Computation slots for Confluencia 2.0 Drug API.

Defines Protocol-based interfaces (slots) for drug computation,
molecular evolution, and clinical trial simulation. Each slot has two
implementations:
  - Local*Slot: directly calls core/*.py functions
  - Remote*Slot: calls the FastAPI server over HTTP

Usage:
    from api.slots import create_slots
    slots = create_slots("local")              # local computation
    slots = create_slots("remote", "http://host:8000")  # remote server
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Ensure project root is on sys.path so `core` package is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.schemas import (
    ArtifactsSchema,
    CohortRequest,
    CohortResponse,
    DrugPredictRequest,
    DrugPredictResponse,
    DrugTrainPredictRequest,
    DrugTrainPredictResponse,
    DrugTrainRequest,
    DrugTrainResponse,
    EvolutionArtifactsSchema,
    EvolveCirrnaRequest,
    EvolveCirrnaResponse,
    EvolveMoleculesRequest,
    EvolveMoleculesResponse,
    FullTrialRequest,
    FullTrialResponse,
    PhaseIRequest,
    PhaseIResponse,
    PhaseIIRequest,
    PhaseIIResponse,
    PhaseIIIRequest,
    PhaseIIIResponse,
    TrainingReportSchema,
    TrialReportRequest,
    TrialReportResponse,
)
from api.serialization import csv_str_to_df, dataclass_to_dict, df_to_csv_str


# ---------------------------------------------------------------------------
# Protocol Interfaces (Slots)
# ---------------------------------------------------------------------------


@runtime_checkable
class DrugComputationSlot(Protocol):
    def train_and_predict(self, request: DrugTrainPredictRequest) -> DrugTrainPredictResponse: ...
    def train(self, request: DrugTrainRequest) -> DrugTrainResponse: ...
    def predict(self, request: DrugPredictRequest) -> DrugPredictResponse: ...


@runtime_checkable
class EvolutionComputationSlot(Protocol):
    def evolve_molecules(self, request: EvolveMoleculesRequest) -> EvolveMoleculesResponse: ...
    def evolve_cirrna(self, request: EvolveCirrnaRequest) -> EvolveCirrnaResponse: ...


@runtime_checkable
class TrialComputationSlot(Protocol):
    def generate_cohort(self, request: CohortRequest) -> CohortResponse: ...
    def simulate_phase_i(self, request: PhaseIRequest) -> PhaseIResponse: ...
    def simulate_phase_ii(self, request: PhaseIIRequest) -> PhaseIIResponse: ...
    def simulate_phase_iii(self, request: PhaseIIIRequest) -> PhaseIIIResponse: ...
    def full_pipeline(self, request: FullTrialRequest) -> FullTrialResponse: ...
    def generate_report(self, request: TrialReportRequest) -> TrialReportResponse: ...


@dataclass
class SlotBundle:
    drug: DrugComputationSlot
    evolution: EvolutionComputationSlot
    trial: TrialComputationSlot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifacts_to_schema(artifacts) -> ArtifactsSchema:
    """Convert ConfluenciaArtifacts dataclass to ArtifactsSchema."""
    return ArtifactsSchema(
        compute_profile=artifacts.compute_profile,
        moe_weights=dict(artifacts.moe_weights),
        moe_metrics=dict(artifacts.moe_metrics),
        used_proxy_micro_labels=bool(artifacts.used_proxy_micro_labels),
        smiles_backend=str(artifacts.smiles_backend),
        ctm_param_source=str(artifacts.ctm_param_source),
        model_backend=str(artifacts.model_backend),
        dynamics_model=str(artifacts.dynamics_model),
        shap_ready=bool(artifacts.shap_ready),
        shap_message=str(artifacts.shap_message),
        shap_feature_count=int(artifacts.shap_feature_count),
        adaptive_enabled=bool(artifacts.adaptive_enabled),
        adaptive_strength=float(artifacts.adaptive_strength),
        adaptive_samples=int(artifacts.adaptive_samples),
        adaptive_message=str(artifacts.adaptive_message),
    )


def _report_to_schema(report) -> TrainingReportSchema:
    """Convert DrugTrainingReport dataclass to TrainingReportSchema."""
    return TrainingReportSchema(
        sample_count=report.sample_count,
        used_labels=dict(report.used_labels),
        metrics=dict(report.metrics),
    )


def _evo_artifacts_to_schema(artifacts) -> EvolutionArtifactsSchema:
    """Convert EvolutionArtifacts dataclass to EvolutionArtifactsSchema."""
    return EvolutionArtifactsSchema(
        final_policy_logits=dict(artifacts.final_policy_logits),
        reflections=list(artifacts.reflections),
        used_ed2mol=bool(artifacts.used_ed2mol),
        selected_objective_weights=dict(artifacts.selected_objective_weights),
        rounds_ran=int(artifacts.rounds_ran),
        best_reward=float(artifacts.best_reward),
        per_round_best=list(artifacts.per_round_best),
    )


# ---------------------------------------------------------------------------
# Local Implementations
# ---------------------------------------------------------------------------


class LocalDrugSlot:
    """Calls core.training functions directly."""

    def __init__(self, model_store: Optional[Dict] = None):
        self._model_store = model_store if model_store is not None else {}

    def train_and_predict(self, request: DrugTrainPredictRequest) -> DrugTrainPredictResponse:
        from core.training import train_and_predict_drug
        df = csv_str_to_df(request.data_csv)
        result_df, curve_df, artifacts, report = train_and_predict_drug(
            df=df,
            compute_mode=request.compute_mode,
            model_backend=request.model_backend,
            dynamics_model=request.dynamics_model,
            adaptive_enabled=request.adaptive_enabled,
            adaptive_strength=request.adaptive_strength,
        )
        return DrugTrainPredictResponse(
            result_csv=df_to_csv_str(result_df),
            curve_csv=df_to_csv_str(curve_df),
            artifacts=_artifacts_to_schema(artifacts),
            report=_report_to_schema(report),
        )

    def train(self, request: DrugTrainRequest) -> DrugTrainResponse:
        import uuid
        from core.training import train_drug_model, get_drug_model_metadata
        df = csv_str_to_df(request.data_csv)
        trained = train_drug_model(
            df=df,
            compute_mode=request.compute_mode,
            model_backend=request.model_backend,
            dynamics_model=request.dynamics_model,
            adaptive_enabled=request.adaptive_enabled,
            adaptive_strength=request.adaptive_strength,
        )
        model_id = str(uuid.uuid4())
        self._model_store[model_id] = trained
        metadata = get_drug_model_metadata(trained)
        return DrugTrainResponse(model_id=model_id, metadata=metadata)

    def predict(self, request: DrugPredictRequest) -> DrugPredictResponse:
        from core.training import predict_drug_with_model
        trained = self._model_store.get(request.model_id)
        if trained is None:
            raise ValueError(f"Model not found: {request.model_id}")
        df = csv_str_to_df(request.data_csv)
        result_df, curve_df, artifacts, report = predict_drug_with_model(
            df=df,
            trained_model=trained,
            adaptive_enabled=request.adaptive_enabled,
            adaptive_strength=request.adaptive_strength,
        )
        return DrugPredictResponse(
            result_csv=df_to_csv_str(result_df),
            curve_csv=df_to_csv_str(curve_df),
            artifacts=_artifacts_to_schema(artifacts),
            report=_report_to_schema(report),
        )


class LocalEvolutionSlot:
    """Calls core.evolution functions directly."""

    def evolve_molecules(self, request: EvolveMoleculesRequest) -> EvolveMoleculesResponse:
        from core.evolution import evolve_molecules_with_reflection
        cfg = request.config.to_dataclass()
        result_df, artifacts = evolve_molecules_with_reflection(
            seed_smiles=request.seed_smiles,
            cfg=cfg,
            ed2mol_repo_dir=request.ed2mol_repo_dir,
            ed2mol_config_path=request.ed2mol_config_path,
            ed2mol_python_cmd=request.ed2mol_python_cmd,
        )
        return EvolveMoleculesResponse(
            results_csv=df_to_csv_str(result_df),
            artifacts=_evo_artifacts_to_schema(artifacts),
        )

    def evolve_cirrna(self, request: EvolveCirrnaRequest) -> EvolveCirrnaResponse:
        from core.evolution import evolve_cirrna_sequences
        cfg = request.config.to_dataclass()
        result_df, artifacts = evolve_cirrna_sequences(cfg=cfg)
        return EvolveCirrnaResponse(
            results_csv=df_to_csv_str(result_df),
            artifacts=dataclass_to_dict(artifacts) if artifacts else {},
        )


class LocalTrialSlot:
    """Calls core.trial_sim functions directly."""

    def __init__(self):
        from core.trial_sim import default_dlt_prob, default_efficacy_fn, default_survival_fn
        self._fn_registry = {
            "default_dlt_prob": default_dlt_prob,
            "default_efficacy_fn": default_efficacy_fn,
            "default_survival_fn": default_survival_fn,
        }

    def _resolve_cohort(self, request) -> "pd.DataFrame":
        """Resolve cohort from CSV string or config."""
        from core.trial_sim import generate_cohort
        cohort_csv = getattr(request, "cohort_csv", None)
        if cohort_csv:
            return csv_str_to_df(cohort_csv)
        cohort_config = getattr(request, "cohort_config", None)
        cfg = cohort_config.to_dataclass() if cohort_config else None
        return generate_cohort(cfg)

    def generate_cohort(self, request: CohortRequest) -> CohortResponse:
        from core.trial_sim import generate_cohort
        cfg = request.config.to_dataclass() if request.config else None
        cohort = generate_cohort(cfg)
        return CohortResponse(cohort_csv=df_to_csv_str(cohort))

    def simulate_phase_i(self, request: PhaseIRequest) -> PhaseIResponse:
        from core.trial_sim import simulate_phase_i
        cohort = self._resolve_cohort(request)
        dlt_fn = self._fn_registry.get(request.dlt_prob_fn_name)
        if dlt_fn is None:
            raise ValueError(f"Unknown function: {request.dlt_prob_fn_name}")
        cfg = request.config.to_dataclass()
        result = simulate_phase_i(dlt_fn, cohort, cfg)
        return PhaseIResponse(
            mtd_estimate=result.mtd_estimate,
            rp2d=result.rp2d,
            dose_toxicity_csv=df_to_csv_str(result.dose_toxicity_curve),
            dose_levels_tested=list(result.dose_levels_tested),
            patients_per_level=list(result.patients_per_level),
            dlts_per_level=list(result.dlts_per_level),
            decision_log=list(result.decision_log),
        )

    def simulate_phase_ii(self, request: PhaseIIRequest) -> PhaseIIResponse:
        from core.trial_sim import simulate_phase_ii
        cohort = self._resolve_cohort(request)
        eff_fn = self._fn_registry.get(request.efficacy_fn_name)
        if eff_fn is None:
            raise ValueError(f"Unknown function: {request.efficacy_fn_name}")
        cfg = request.config.to_dataclass()
        result = simulate_phase_ii(eff_fn, cohort, request.rp2d, request.soc_efficacy, cfg)
        return PhaseIIResponse(
            treatment_arm=dict(result.treatment_arm),
            control_arm=dict(result.control_arm),
            p_value=float(result.p_value),
            statistically_significant=bool(result.statistically_significant),
            power_estimate=float(result.power_estimate),
            biomarker_subgroup=dict(result.biomarker_subgroup),
            km_data_treatment_csv=df_to_csv_str(result.km_data_treatment),
            km_data_control_csv=df_to_csv_str(result.km_data_control),
        )

    def simulate_phase_iii(self, request: PhaseIIIRequest) -> PhaseIIIResponse:
        from core.trial_sim import simulate_phase_iii
        cohort = self._resolve_cohort(request)
        surv_fn = self._fn_registry.get(request.survival_fn_name)
        if surv_fn is None:
            raise ValueError(f"Unknown function: {request.survival_fn_name}")
        cfg = request.config.to_dataclass()
        result = simulate_phase_iii(surv_fn, cohort, request.rp2d, request.soc_median_survival, cfg)
        return PhaseIIIResponse(
            hazard_ratio=float(result.hazard_ratio),
            hr_ci_lower=float(result.hr_ci_lower),
            hr_ci_upper=float(result.hr_ci_upper),
            p_value=float(result.p_value),
            significant=bool(result.significant),
            subgroup_analysis_csv=df_to_csv_str(result.subgroup_analysis),
            km_data_treatment_csv=df_to_csv_str(result.km_data_treatment),
            km_data_control_csv=df_to_csv_str(result.km_data_control),
            median_survival_treatment=float(result.median_survival_treatment),
            median_survival_control=float(result.median_survival_control),
        )

    def full_pipeline(self, request: FullTrialRequest) -> FullTrialResponse:
        from core.trial_sim import generate_cohort, simulate_phase_i, simulate_phase_ii, simulate_phase_iii, generate_trial_report

        cfg_cohort = request.cohort_config.to_dataclass() if request.cohort_config else None
        cohort = generate_cohort(cfg_cohort)
        rp2d = 4.0  # will be updated after phase I

        # Phase I
        dlt_fn = self._fn_registry.get(request.dlt_prob_fn_name)
        phase_i_result = simulate_phase_i(dlt_fn, cohort, request.phase_i_config.to_dataclass())
        rp2d = phase_i_result.rp2d
        phase_i_resp = PhaseIResponse(
            mtd_estimate=phase_i_result.mtd_estimate,
            rp2d=phase_i_result.rp2d,
            dose_toxicity_csv=df_to_csv_str(phase_i_result.dose_toxicity_curve),
            dose_levels_tested=list(phase_i_result.dose_levels_tested),
            patients_per_level=list(phase_i_result.patients_per_level),
            dlts_per_level=list(phase_i_result.dlts_per_level),
            decision_log=list(phase_i_result.decision_log),
        )

        # Phase II
        eff_fn = self._fn_registry.get(request.efficacy_fn_name)
        phase_ii_result = simulate_phase_ii(eff_fn, cohort, rp2d, request.soc_efficacy, request.phase_ii_config.to_dataclass())
        phase_ii_resp = PhaseIIResponse(
            treatment_arm=dict(phase_ii_result.treatment_arm),
            control_arm=dict(phase_ii_result.control_arm),
            p_value=float(phase_ii_result.p_value),
            statistically_significant=bool(phase_ii_result.statistically_significant),
            power_estimate=float(phase_ii_result.power_estimate),
            biomarker_subgroup=dict(phase_ii_result.biomarker_subgroup),
            km_data_treatment_csv=df_to_csv_str(phase_ii_result.km_data_treatment),
            km_data_control_csv=df_to_csv_str(phase_ii_result.km_data_control),
        )

        # Phase III
        surv_fn = self._fn_registry.get(request.survival_fn_name)
        phase_iii_result = simulate_phase_iii(surv_fn, cohort, rp2d, request.soc_median_survival, request.phase_iii_config.to_dataclass())
        phase_iii_resp = PhaseIIIResponse(
            hazard_ratio=float(phase_iii_result.hazard_ratio),
            hr_ci_lower=float(phase_iii_result.hr_ci_lower),
            hr_ci_upper=float(phase_iii_result.hr_ci_upper),
            p_value=float(phase_iii_result.p_value),
            significant=bool(phase_iii_result.significant),
            subgroup_analysis_csv=df_to_csv_str(phase_iii_result.subgroup_analysis),
            km_data_treatment_csv=df_to_csv_str(phase_iii_result.km_data_treatment),
            km_data_control_csv=df_to_csv_str(phase_iii_result.km_data_control),
            median_survival_treatment=float(phase_iii_result.median_survival_treatment),
            median_survival_control=float(phase_iii_result.median_survival_control),
        )

        # Report
        report = generate_trial_report(phase_i_result, phase_ii_result, phase_iii_result, None, request.drug_name)

        return FullTrialResponse(
            phase_i=phase_i_resp,
            phase_ii=phase_ii_resp,
            phase_iii=phase_iii_resp,
            report=report,
        )

    def generate_report(self, request: TrialReportRequest) -> TrialReportResponse:
        from core.trial_sim import (
            PhaseIResult, PhaseIIResult, PhaseIIIResult,
            generate_trial_report,
        )
        phase_i = None
        phase_ii = None
        phase_iii = None
        cohort_summary = csv_str_to_df(request.cohort_summary_csv) if request.cohort_summary_csv else None

        if request.phase_i:
            phase_i = PhaseIResult(
                mtd_estimate=request.phase_i.mtd_estimate,
                rp2d=request.phase_i.rp2d,
                dose_toxicity_curve=csv_str_to_df(request.phase_i.dose_toxicity_csv),
                dose_levels_tested=request.phase_i.dose_levels_tested,
                patients_per_level=request.phase_i.patients_per_level,
                dlts_per_level=request.phase_i.dlts_per_level,
                decision_log=request.phase_i.decision_log,
            )
        if request.phase_ii:
            phase_ii = PhaseIIResult(
                treatment_arm=request.phase_ii.treatment_arm,
                control_arm=request.phase_ii.control_arm,
                p_value=request.phase_ii.p_value,
                statistically_significant=request.phase_ii.statistically_significant,
                power_estimate=request.phase_ii.power_estimate,
                biomarker_subgroup=request.phase_ii.biomarker_subgroup,
                km_data_treatment=csv_str_to_df(request.phase_ii.km_data_treatment_csv),
                km_data_control=csv_str_to_df(request.phase_ii.km_data_control_csv),
            )
        if request.phase_iii:
            phase_iii = PhaseIIIResult(
                hazard_ratio=request.phase_iii.hazard_ratio,
                hr_ci_lower=request.phase_iii.hr_ci_lower,
                hr_ci_upper=request.phase_iii.hr_ci_upper,
                p_value=request.phase_iii.p_value,
                significant=request.phase_iii.significant,
                subgroup_analysis=csv_str_to_df(request.phase_iii.subgroup_analysis_csv),
                km_data_treatment=csv_str_to_df(request.phase_iii.km_data_treatment_csv),
                km_data_control=csv_str_to_df(request.phase_iii.km_data_control_csv),
                median_survival_treatment=request.phase_iii.median_survival_treatment,
                median_survival_control=request.phase_iii.median_survival_control,
            )

        report = generate_trial_report(phase_i, phase_ii, phase_iii, cohort_summary, request.drug_name)
        return TrialReportResponse(report=report)


# ---------------------------------------------------------------------------
# Remote Implementations
# ---------------------------------------------------------------------------


class RemoteDrugSlot:
    """Calls the FastAPI server over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        resp = httpx.post(f"{self._base_url}{path}", json=payload, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    def train_and_predict(self, request: DrugTrainPredictRequest) -> DrugTrainPredictResponse:
        data = self._post("/api/drug/train-and-predict", request.model_dump())
        return DrugTrainPredictResponse(**data)

    def train(self, request: DrugTrainRequest) -> DrugTrainResponse:
        data = self._post("/api/drug/train", request.model_dump())
        return DrugTrainResponse(**data)

    def predict(self, request: DrugPredictRequest) -> DrugPredictResponse:
        data = self._post("/api/drug/predict", request.model_dump())
        return DrugPredictResponse(**data)


class RemoteEvolutionSlot:
    """Calls the FastAPI server over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        resp = httpx.post(f"{self._base_url}{path}", json=payload, timeout=600.0)
        resp.raise_for_status()
        return resp.json()

    def evolve_molecules(self, request: EvolveMoleculesRequest) -> EvolveMoleculesResponse:
        data = self._post("/api/evolution/molecules", request.model_dump())
        return EvolveMoleculesResponse(**data)

    def evolve_cirrna(self, request: EvolveCirrnaRequest) -> EvolveCirrnaResponse:
        data = self._post("/api/evolution/cirrna", request.model_dump())
        return EvolveCirrnaResponse(**data)


class RemoteTrialSlot:
    """Calls the FastAPI server over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        resp = httpx.post(f"{self._base_url}{path}", json=payload, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    def generate_cohort(self, request: CohortRequest) -> CohortResponse:
        data = self._post("/api/trial/cohort", request.model_dump())
        return CohortResponse(**data)

    def simulate_phase_i(self, request: PhaseIRequest) -> PhaseIResponse:
        data = self._post("/api/trial/phase-i", request.model_dump())
        return PhaseIResponse(**data)

    def simulate_phase_ii(self, request: PhaseIIRequest) -> PhaseIIResponse:
        data = self._post("/api/trial/phase-ii", request.model_dump())
        return PhaseIIResponse(**data)

    def simulate_phase_iii(self, request: PhaseIIIRequest) -> PhaseIIIResponse:
        data = self._post("/api/trial/phase-iii", request.model_dump())
        return PhaseIIIResponse(**data)

    def full_pipeline(self, request: FullTrialRequest) -> FullTrialResponse:
        data = self._post("/api/trial/full-pipeline", request.model_dump())
        return FullTrialResponse(**data)

    def generate_report(self, request: TrialReportRequest) -> TrialReportResponse:
        data = self._post("/api/trial/report", request.model_dump())
        return TrialReportResponse(**data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_slots(mode: str = "local", server_url: str = "http://localhost:8000", model_store: Optional[Dict] = None) -> SlotBundle:
    """Create a bundle of computation slots.

    Parameters
    ----------
    mode : str
        "local" for direct computation, "remote" for HTTP calls.
    server_url : str
        Base URL of the remote server (used only in remote mode).
    model_store : dict, optional
        Shared model store for LocalDrugSlot. If None, creates a new one.
    """
    if mode == "local":
        store = model_store if model_store is not None else {}
        return SlotBundle(
            drug=LocalDrugSlot(model_store=store),
            evolution=LocalEvolutionSlot(),
            trial=LocalTrialSlot(),
        )
    elif mode == "remote":
        return SlotBundle(
            drug=RemoteDrugSlot(base_url=server_url),
            evolution=RemoteEvolutionSlot(base_url=server_url),
            trial=RemoteTrialSlot(base_url=server_url),
        )
    raise ValueError(f"Unknown slot mode: {mode!r}. Use 'local' or 'remote'.")
