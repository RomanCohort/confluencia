"""
Joint evaluation engine — orchestrates drug + epitope + PK pipelines.

Provides the single entry point `JointEvaluationEngine.evaluate()` that
runs the full three-dimensional evaluation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure drug / epitope / shared are importable
# ---------------------------------------------------------------------------

import importlib

_PROJECT = Path(__file__).resolve().parents[1]
_DRUG_DIR = _PROJECT / "confluencia-2.0-drug"
_EPITOPE_DIR = _PROJECT / "confluencia-2.0-epitope"
_SHARED_DIR = _PROJECT / "confluencia_shared"

for _dir in [_DRUG_DIR, _EPITOPE_DIR, _SHARED_DIR]:
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# ---------------------------------------------------------------------------
# Imports — use absolute import after path setup
# ---------------------------------------------------------------------------

from confluencia_joint.joint_input import JointInput
from confluencia_joint.scoring import (
    ClinicalScore,
    BindingScore,
    KineticsScore,
    JointScore,
    JointScoringEngine,
)
from confluencia_joint.fusion_layer import JointFusionLayer, FusionStrategy


# ---------------------------------------------------------------------------
# Lazy imports to avoid drug/epitope "core" package name conflict
# ---------------------------------------------------------------------------

def _import_drug_pipeline():
    """Import drug pipeline module (lazy)."""
    spec = importlib.util.spec_from_file_location(
        "drug_pipeline",
        _DRUG_DIR / "core" / "pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pkpd():
    """Import pkpd module (lazy)."""
    spec = importlib.util.spec_from_file_location(
        "pkpd",
        _DRUG_DIR / "core" / "pkpd.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_epitope_pipeline():
    """Import epitope pipeline module (lazy)."""
    spec = importlib.util.spec_from_file_location(
        "epitope_pipeline",
        _EPITOPE_DIR / "core" / "pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class JointEvaluationResult:
    """Container for a complete joint evaluation.

    Attributes
    ----------
    input : JointInput
        The original input that was evaluated.
    joint_score : JointScore
        The three-dimensional scoring result.
    drug_outputs : dict
        Raw outputs from the drug pipeline.
    epitope_outputs : dict
        Raw outputs from the epitope pipeline.
    pk_summary : dict
        PK/PD summary from simulate_pkpd + summarize_pkpd_curve.
    pk_curve : pd.DataFrame
        Time-series PK curve (time_h, pkpd_conc_mg_per_l, pkpd_effect).
    fused_vector : np.ndarray
        Fused representation from JointFusionLayer.
    evaluation_time_s : float
        Wall-clock time for the evaluation in seconds.
    errors : list[str]
        Any errors or warnings encountered.
    """

    def __init__(
        self,
        input: JointInput,
        joint_score: JointScore,
        drug_outputs: Dict[str, float],
        epitope_outputs: Dict[str, float],
        pk_summary: Dict[str, float],
        pk_curve: pd.DataFrame,
        fused_vector: np.ndarray,
        evaluation_time_s: float,
        errors: List[str],
    ):
        self.input = input
        self.joint_score = joint_score
        self.drug_outputs = drug_outputs
        self.epitope_outputs = epitope_outputs
        self.pk_summary = pk_summary
        self.pk_curve = pk_curve
        self.fused_vector = fused_vector
        self.evaluation_time_s = evaluation_time_s
        self.errors = errors

    def to_dict(self) -> dict:
        """Serialize to a plain dict (useful for JSON export)."""
        return {
            "input": {
                "smiles": self.input.smiles,
                "epitope_seq": self.input.epitope_seq,
                "mhc_allele": self.input.mhc_allele,
                "dose_mg": self.input.dose_mg,
                "freq_per_day": self.input.freq_per_day,
                "treatment_time": self.input.treatment_time,
                "circ_expr": self.input.circ_expr,
                "ifn_score": self.input.ifn_score,
                "group_id": self.input.group_id,
            },
            "clinical_score": {
                "efficacy": self.joint_score.clinical.efficacy,
                "target_binding": self.joint_score.clinical.target_binding,
                "immune_activation": self.joint_score.clinical.immune_activation,
                "safety_penalty": self.joint_score.clinical.safety_penalty,
                "overall": self.joint_score.clinical.overall,
                "interpretation": self.joint_score.clinical.interpretation,
            },
            "binding_score": {
                "epitope_efficacy": self.joint_score.binding.epitope_efficacy,
                "uncertainty": self.joint_score.binding.uncertainty,
                "mhc_affinity_class": self.joint_score.binding.mhc_affinity_class,
                "overall": self.joint_score.binding.overall,
                "interpretation": self.joint_score.binding.interpretation,
            },
            "kinetics_score": {
                "cmax": self.joint_score.kinetics.cmax,
                "tmax": self.joint_score.kinetics.tmax,
                "half_life": self.joint_score.kinetics.half_life,
                "auc_conc": self.joint_score.kinetics.auc_conc,
                "auc_effect": self.joint_score.kinetics.auc_effect,
                "therapeutic_index": self.joint_score.kinetics.therapeutic_index,
                "overall": self.joint_score.kinetics.overall,
                "interpretation": self.joint_score.kinetics.interpretation,
            },
            "composite": self.joint_score.composite,
            "recommendation": self.joint_score.recommendation,
            "recommendation_reason": self.joint_score.recommendation_reason,
            "evaluation_time_s": self.evaluation_time_s,
            "errors": self.errors,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a flat DataFrame row."""
        row = {
            "smiles": self.input.smiles,
            "epitope_seq": self.input.epitope_seq,
            "mhc_allele": self.input.mhc_allele,
            "composite": self.joint_score.composite,
            "recommendation": self.joint_score.recommendation,
            # Clinical
            "clinical_efficacy": self.joint_score.clinical.efficacy,
            "clinical_binding": self.joint_score.clinical.target_binding,
            "clinical_immune": self.joint_score.clinical.immune_activation,
            "clinical_safety_penalty": self.joint_score.clinical.safety_penalty,
            "clinical_overall": self.joint_score.clinical.overall,
            # Binding
            "binding_epitope_eff": self.joint_score.binding.epitope_efficacy,
            "binding_uncertainty": self.joint_score.binding.uncertainty,
            "binding_affinity_class": self.joint_score.binding.mhc_affinity_class,
            "binding_overall": self.joint_score.binding.overall,
            # Kinetics
            "kin_cmax": self.joint_score.kinetics.cmax,
            "kin_tmax": self.joint_score.kinetics.tmax,
            "kin_half_life": self.joint_score.kinetics.half_life,
            "kin_auc_conc": self.joint_score.kinetics.auc_conc,
            "kin_auc_effect": self.joint_score.kinetics.auc_effect,
            "kin_therapeutic_index": self.joint_score.kinetics.therapeutic_index,
            "kin_overall": self.joint_score.kinetics.overall,
            # PK
            "eval_time_s": self.evaluation_time_s,
        }
        return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class JointEvaluationEngine:
    """Three-dimensional joint drug-epitope-PK evaluator.

    Orchestrates the drug pipeline, epitope pipeline, and PK simulation
    into a unified evaluation.

    Parameters
    ----------
    scoring_engine : JointScoringEngine, optional
        Custom scoring engine. If None, uses defaults.
    fusion_layer : JointFusionLayer, optional
        Custom fusion layer. If None, uses weighted concat.
    epitope_backend : str, optional
        Epitope pipeline backend. Default: "sklearn-moe".
    pk_horizon : int, optional
        PK simulation horizon in hours. Default: 72.
    drug_compute_profile : str, optional
        Drug pipeline compute profile. Default: "medium".

    Examples
    --------
    >>> engine = JointEvaluationEngine()
    >>> inp = JointInput(
    ...     smiles="CC(=O)Oc1ccccc1C(=O)O",
    ...     epitope_seq="SLYNTVATL",
    ...     mhc_allele="HLA-A*02:01",
    ...     dose_mg=200.0,
    ...     freq_per_day=2.0,
    ...     treatment_time=72.0,
    ... )
    >>> result = engine.evaluate_single(inp)
    >>> print(result.joint_score.recommendation)
    """

    def __init__(
        self,
        scoring_engine: Optional[JointScoringEngine] = None,
        fusion_layer: Optional[JointFusionLayer] = None,
        epitope_backend: str = "sklearn-moe",
        pk_horizon: int = 72,
        drug_compute_profile: str = "medium",
        use_mhc: bool = True,
    ):
        self.scoring_engine = scoring_engine or JointScoringEngine()
        self.fusion_layer = fusion_layer or JointFusionLayer()
        self.epitope_backend = epitope_backend
        self.pk_horizon = pk_horizon
        self.drug_compute_profile = drug_compute_profile
        self.use_mhc = use_mhc
        # Modules loaded lazily at first use
        self._drug_pipeline = None
        self._pkpd = None
        self._epitope_pipeline = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_modules(self):
        """Lazily load drug/epitope/pkpd modules."""
        if self._drug_pipeline is None:
            self._drug_pipeline = _import_drug_pipeline()
        if self._pkpd is None:
            self._pkpd = _import_pkpd()
        if self._epitope_pipeline is None:
            self._epitope_pipeline = _import_epitope_pipeline()

    def evaluate(self, inputs: List[JointInput]) -> List[JointEvaluationResult]:
        """Evaluate a batch of JointInput instances.

        Parameters
        ----------
        inputs : list[JointInput]
            List of inputs to evaluate.

        Returns
        -------
        list[JointEvaluationResult]
            One result per input, in the same order.
        """
        if not inputs:
            return []

        import time
        self._ensure_modules()

        # --- Step 1: Split into drug_df and epitope_df ---
        drug_dfs = []
        epi_dfs = []
        for inp in inputs:
            drug_dfs.append(inp.to_drug_dataframe())
            epi_dfs.append(inp.to_epitope_dataframe())

        drug_df = pd.concat(drug_dfs, ignore_index=True)
        epi_df = pd.concat(epi_dfs, ignore_index=True)

        # --- Step 2: Run drug pipeline ---
        t_drug = time.time()
        drug_errors: List[str] = []
        try:
            drug_out, drug_curve, drug_artifacts = self._drug_pipeline.run_pipeline(
                drug_df,
                compute_profile=self.drug_compute_profile,
            )
        except Exception as ex:
            drug_errors.append(f"Drug pipeline error: {ex}")
            drug_out = drug_df.copy()
            drug_out["efficacy_pred"] = np.nan
            drug_out["target_binding_pred"] = np.nan
            drug_out["immune_activation_pred"] = np.nan
            drug_out["inflammation_risk_pred"] = np.nan
            drug_out["genotoxicity_risk_pred"] = np.nan
        drug_elapsed = time.time() - t_drug

        # --- Step 3: Run epitope pipeline ---
        t_epi = time.time()
        epi_errors: List[str] = []
        try:
            epi_out, epi_artifacts, epi_sens = self._epitope_pipeline.run_pipeline(
                epi_df,
                model_backend=self.epitope_backend,
                feature_spec=self._epitope_pipeline.FeatureSpec(use_mhc=self.use_mhc),
            )
        except Exception as ex:
            epi_errors.append(f"Epitope pipeline error: {ex}")
            epi_out = epi_df.copy()
            epi_out["efficacy_pred"] = np.nan
            epi_out["pred_uncertainty"] = np.nan
        epi_elapsed = time.time() - t_epi

        # --- Step 4: PK simulation ---
        pk_errors: List[str] = []
        all_pk_summaries: List[Dict[str, float]] = []
        all_pk_curves: List[pd.DataFrame] = []

        for i, inp in enumerate(inputs):
            try:
                pk_sum, pk_curve = self._run_pk_simulation(inp)
            except Exception as ex:
                pk_errors.append(f"PK simulation error for row {i}: {ex}")
                pk_sum = {
                    "pkpd_cmax_mg_per_l": np.nan,
                    "pkpd_tmax_h": np.nan,
                    "pkpd_half_life_h": np.nan,
                    "pkpd_auc_conc": np.nan,
                    "pkpd_auc_effect": np.nan,
                }
                pk_curve = pd.DataFrame({
                    "time_h": [0, self.pk_horizon],
                    "pkpd_conc_mg_per_l": [0, 0],
                    "pkpd_effect": [0, 0],
                })
            all_pk_summaries.append(pk_sum)
            all_pk_curves.append(pk_curve)

        # --- Step 5: Score and fuse per row ---
        results: List[JointEvaluationResult] = []
        for i, inp in enumerate(inputs):
            # Extract outputs as dicts
            drug_dict = self._row_to_dict(drug_out, i, [
                "efficacy_pred", "target_binding_pred", "immune_activation_pred",
                "inflammation_risk_pred", "genotoxicity_risk_pred",
            ])
            epi_dict = self._row_to_dict(epi_out, i, [
                "efficacy_pred", "pred_uncertainty",
            ])

            # Score
            all_errors = list(drug_errors) + list(epi_errors)
            if pk_errors:
                all_errors.append(pk_errors[i] if i < len(pk_errors) else "")

            joint_score = self.scoring_engine.score(drug_dict, epi_dict, all_pk_summaries[i])

            # Fuse
            fused = self.fusion_layer.fuse(
                self.joint_to_arrays(joint_score, "clinical"),
                self.joint_to_arrays(joint_score, "binding"),
                self.joint_to_arrays(joint_score, "kinetics"),
            )

            # Eval time (rough proportional share)
            total_time = drug_elapsed + epi_elapsed + 0.1 * len(inputs)
            eval_time = total_time * (1.0 / len(inputs))

            results.append(JointEvaluationResult(
                input=inp,
                joint_score=joint_score,
                drug_outputs=drug_dict,
                epitope_outputs=epi_dict,
                pk_summary=all_pk_summaries[i],
                pk_curve=all_pk_curves[i],
                fused_vector=fused,
                evaluation_time_s=round(eval_time, 3),
                errors=[e for e in all_errors if e],
            ))

        return results

    def evaluate_single(self, inp: JointInput) -> JointEvaluationResult:
        """Convenience wrapper for evaluating a single input."""
        results = self.evaluate([inp])
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_pk_simulation(
        self, inp: JointInput
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Run PK simulation for a single input.

        Uses inferred PK parameters from drug efficacy + binding + immune.
        """
        self._ensure_modules()
        params = self._infer_pk_params(inp)

        # Simulate
        curve_df = self._pkpd.simulate_pkpd(
            dose_mg=inp.dose_mg,
            freq_per_day=inp.freq_per_day,
            params=params,
            horizon=self.pk_horizon,
            dt=1.0,
        )

        # Summarize
        summary = self._pkpd.summarize_pkpd_curve(curve_df)

        return summary, curve_df

    def _infer_pk_params(self, inp: JointInput):
        """Infer PK parameters from input and defaults.

        Uses `infer_pkpd_params` from the PKPD module with estimated
        absorption / distribution / elimination rates derived from the
        epitope binding and immune activation (used as bioactivity proxy).
        """
        self._ensure_modules()
        return self._pkpd.infer_pkpd_params(
            target_binding=inp.circ_expr if inp.circ_expr > 0 else 0.3,
            immune_activation=inp.ifn_score if inp.ifn_score > 0 else 0.5,
            inflammation=0.1,  # default low
            dose_mg=inp.dose_mg,
            freq_per_day=inp.freq_per_day,
        )

    @staticmethod
    def _row_to_dict(
        df: pd.DataFrame, row_idx: int, columns: List[str]
    ) -> Dict[str, float]:
        """Extract a single row as a dict."""
        if row_idx >= len(df):
            return {col: np.nan for col in columns}
        row = df.iloc[row_idx]
        return {col: float(row.get(col, np.nan)) for col in columns}

    @staticmethod
    def joint_to_arrays(
        score: JointScore, dim: Literal["clinical", "binding", "kinetics"]
    ) -> Dict[str, float]:
        """Convert a JointScore sub-object to a flat dict for fusion."""
        if dim == "clinical":
            c = score.clinical
            return {
                "efficacy": c.efficacy,
                "target_binding": c.target_binding,
                "immune_activation": c.immune_activation,
                "safety_penalty": c.safety_penalty,
                "overall": c.overall,
            }
        elif dim == "binding":
            b = score.binding
            return {
                "epitope_efficacy": b.epitope_efficacy,
                "uncertainty": b.uncertainty,
                "overall": b.overall,
            }
        else:  # kinetics
            k = score.kinetics
            return {
                "cmax": k.cmax,
                "tmax": k.tmax,
                "half_life": k.half_life,
                "auc_conc": k.auc_conc,
                "auc_effect": k.auc_effect,
                "therapeutic_index": k.therapeutic_index,
                "overall": k.overall,
            }