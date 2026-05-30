"""
Joint evaluation engine — orchestrates drug + epitope + PK pipelines.

Provides the single entry point `JointEvaluationEngine.evaluate()` that
runs the full three-dimensional evaluation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

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
_CIRCRNA_DIR = _PROJECT / "confluencia_circrna"

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
    GeneSignatureScore,
    CircRNAScore,
    JointScore,
    JointScoringEngine,
)
from confluencia_joint.fusion_layer import JointFusionLayer, FusionStrategy

try:
    from confluencia_shared.five_gene_scorer import FiveGeneMOEScorer
except ImportError:
    FiveGeneMOEScorer = None

try:
    from confluencia_joint.trained_survival_model import get_default_model, predict_patient_risk
    HAS_TRAINED_MODEL = True
except ImportError:
    HAS_TRAINED_MODEL = False


# ---------------------------------------------------------------------------
# Lazy imports to avoid drug/epitope "core" package name conflict
# ---------------------------------------------------------------------------

def _register_hyphenated_package(hyphen_dir: Path, pkg_name: str):
    """Register a hyphenated directory as a Python package in sys.modules.

    This allows relative imports (from .xxx import yyy) within the package's
    modules to work correctly, even though the directory name contains hyphens
    which are not valid in Python identifiers.

    We create a minimal namespace package (no __init__.py execution) to avoid
    issues with relative imports in the original __init__.py files.

    Parameters
    ----------
    hyphen_dir : Path
        The directory containing the package (e.g., confluencia-2.0-drug).
    pkg_name : str
        A valid Python identifier to use as the package name (e.g., confluencia_drug).
    """
    import types

    if pkg_name in sys.modules:
        return sys.modules[pkg_name]

    # Create a namespace package (no __init__.py execution)
    mod = types.ModuleType(pkg_name)
    mod.__path__ = [str(hyphen_dir)]
    mod.__package__ = pkg_name
    mod.__file__ = None  # Namespace packages have no __file__
    sys.modules[pkg_name] = mod
    return mod


def _register_submodule(parent_pkg_name: str, sub_name: str, sub_path: Path):
    """Register a submodule under a parent package in sys.modules.

    Creates a namespace subpackage (no __init__.py execution) to avoid
    relative import issues.

    Parameters
    ----------
    parent_pkg_name : str
        The parent package name (e.g., "confluencia_drug").
    sub_name : str
        The submodule name (e.g., "core").
    sub_path : Path
        Path to the submodule directory.
    """
    import types

    full_name = f"{parent_pkg_name}.{sub_name}"

    if full_name in sys.modules:
        return sys.modules[full_name]

    # Create a namespace subpackage
    mod = types.ModuleType(full_name)
    mod.__path__ = [str(sub_path)]
    mod.__package__ = full_name
    mod.__file__ = None
    sys.modules[full_name] = mod
    # Register in parent's attributes
    parent = sys.modules.get(parent_pkg_name)
    if parent is not None:
        setattr(parent, sub_name, mod)
    return mod


def _register_leaf_module(parent_pkg_name: str, module_name: str, module_path: Path):
    """Register a leaf .py module under a parent package in sys.modules.

    Parameters
    ----------
    parent_pkg_name : str
        The parent package name (e.g., "confluencia_drug.core").
    module_name : str
        The module name without .py (e.g., "pipeline").
    module_path : Path
        Full path to the .py file.
    """
    import importlib.util

    full_name = f"{parent_pkg_name}.{module_name}"

    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = parent_pkg_name
    sys.modules[full_name] = mod
    # Register in parent's attributes
    parent = sys.modules.get(parent_pkg_name)
    if parent is not None:
        setattr(parent, module_name, mod)
    spec.loader.exec_module(mod)
    return mod


def _setup_drug_package():
    """Set up the confluencia-2.0-drug package hierarchy in sys.modules.

    Registers: confluencia_drug → confluencia_drug.core → confluencia_drug.core.xxx
    Only loads modules that pipeline.py transitively depends on, in dependency order.

    Dependency chain for pipeline.py (from line 9-18 of pipeline.py):
      from .ctm import simulate_ctm, summarize_curve
      from .ctm_param_model import CTMParamModel, heuristic_param_targets
      from .features import MixedFeatureSpec, build_feature_matrix, ...
      from .legacy_algorithms import LegacyAlgorithmConfig, train_predict_legacy_backend
      from .micro_predictors import MICRO_TARGETS, MicroPredictor, proxy_micro_labels
      from .moe import MOERegressor, choose_compute_profile
      from .ndp4pd import ndp4pd_from_ctm_like, simulate_ndp4pd
      from .immune_abm import simulate_single_epitope_response
      from .pkpd import infer_pkpd_params, simulate_pkpd, summarize_pkpd_curve

    ctm_param_model.py depends on: from .ctm import CTMParams
    All other modules have only absolute imports (confluencia_shared.xxx).
    """
    # Step 1: Register top-level namespace package
    _register_hyphenated_package(_DRUG_DIR, "confluencia_drug")
    # Step 2: Register core namespace subpackage
    _register_submodule("confluencia_drug", "core", _DRUG_DIR / "core")

    core_dir = _DRUG_DIR / "core"

    # Step 3: Load Tier 0 modules (no relative imports within core)
    # Order matters: ctm must be loaded before ctm_param_model
    tier0_no_deps = [
        "ctm", "pkpd", "ndp4pd", "immune_abm", "micro_predictors",
        "features", "legacy_algorithms",
    ]
    for mod_name in tier0_no_deps:
        mod_path = core_dir / f"{mod_name}.py"
        if mod_path.exists():
            _register_leaf_module("confluencia_drug.core", mod_name, mod_path)

    # Step 4: Load ctm_param_model (depends on .ctm which is now loaded)
    _register_leaf_module("confluencia_drug.core", "ctm_param_model", core_dir / "ctm_param_model.py")

    # Step 5: Load moe (depends on confluencia_shared.moe, absolute import)
    _register_leaf_module("confluencia_drug.core", "moe", core_dir / "moe.py")

    # Step 6: Load pipeline.py (depends on all above)
    _register_leaf_module("confluencia_drug.core", "pipeline", core_dir / "pipeline.py")


def _setup_epitope_package():
    """Set up the confluencia-2.0-epitope package hierarchy in sys.modules.

    Dependency chain for epitope pipeline.py:
      from .features import FeatureSpec, build_feature_matrix, ensure_columns
      from .moe import MOERegressor, choose_compute_profile
      from .sensitivity import neighborhood_importance, numerical_input_gradient, top_features
      from .torch_mamba import TorchMambaConfig, predict_torch_mamba, ...

    features.py depends on: from .mamba3 import Mamba3Config, Mamba3LiteEncoder
    moe.py, sensitivity.py: no relative imports
    torch_mamba.py: no relative imports (has optional torch/mamba_ssm)
    mamba3.py: no relative imports
    """
    _register_hyphenated_package(_EPITOPE_DIR, "confluencia_epitope")
    _register_submodule("confluencia_epitope", "core", _EPITOPE_DIR / "core")

    core_dir = _EPITOPE_DIR / "core"

    # Tier 0: no relative imports
    tier0 = ["mamba3", "moe", "sensitivity"]
    for mod_name in tier0:
        mod_path = core_dir / f"{mod_name}.py"
        if mod_path.exists():
            _register_leaf_module("confluencia_epitope.core", mod_name, mod_path)

    # Tier 1: features depends on .mamba3
    _register_leaf_module("confluencia_epitope.core", "features", core_dir / "features.py")

    # torch_mamba: optional, may fail if torch not available
    torch_mamba_path = core_dir / "torch_mamba.py"
    if torch_mamba_path.exists():
        try:
            _register_leaf_module("confluencia_epitope.core", "torch_mamba", torch_mamba_path)
        except Exception:
            pass

    # Tier 2: pipeline depends on all above
    _register_leaf_module("confluencia_epitope.core", "pipeline", core_dir / "pipeline.py")


def _import_drug_pipeline():
    """Import drug pipeline module (lazy).

    Sets up the full package hierarchy so relative imports work.
    """
    _setup_drug_package()
    return sys.modules["confluencia_drug.core.pipeline"]


def _import_pkpd():
    """Import pkpd module (lazy)."""
    _setup_drug_package()
    return sys.modules["confluencia_drug.core.pkpd"]


def _import_epitope_pipeline():
    """Import epitope pipeline module (lazy).

    Sets up the full package hierarchy so relative imports work.
    """
    _setup_epitope_package()
    return sys.modules["confluencia_epitope.core.pipeline"]


def _import_circrna_pipeline():
    """Import circRNA pipeline module (lazy).

    Uses normal import after temporarily adding project root to sys.path,
    since confluencia_circrna uses package-relative imports.
    """
    project_root = str(_PROJECT)
    path_added = False
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            path_added = True
        import confluencia_circrna.pipeline.circrna_pipeline as mod
        return mod
    finally:
        if path_added and project_root in sys.path:
            sys.path.remove(project_root)


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
        The five-dimensional scoring result.
    drug_outputs : dict
        Raw outputs from the drug pipeline.
    epitope_outputs : dict
        Raw outputs from the epitope pipeline.
    circrna_outputs : dict
        Raw outputs from the circRNA pipeline.
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
        input,
        joint_score: JointScore,
        drug_outputs: Dict[str, float],
        epitope_outputs: Dict[str, float],
        circrna_outputs: Dict[str, float],
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
        self.circrna_outputs = circrna_outputs
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
            "gene_signature_score": {
                "trop2": self.joint_score.gene_signature.trop2 if self.joint_score.gene_signature else None,
                "nectin4": self.joint_score.gene_signature.nectin4 if self.joint_score.gene_signature else None,
                "liv1": self.joint_score.gene_signature.liv1 if self.joint_score.gene_signature else None,
                "b7h4": self.joint_score.gene_signature.b7h4 if self.joint_score.gene_signature else None,
                "tmem65": self.joint_score.gene_signature.tmem65 if self.joint_score.gene_signature else None,
                "risk_score": self.joint_score.gene_signature.risk_score if self.joint_score.gene_signature else None,
                "efficacy_score": self.joint_score.gene_signature.efficacy_score if self.joint_score.gene_signature else None,
                "proliferation_score": self.joint_score.gene_signature.proliferation_score if self.joint_score.gene_signature else None,
                "immune_score": self.joint_score.gene_signature.immune_score if self.joint_score.gene_signature else None,
                "mito_score": self.joint_score.gene_signature.mito_score if self.joint_score.gene_signature else None,
                "tide_score": self.joint_score.gene_signature.tide_score if self.joint_score.gene_signature else None,
                "ips_estimate": self.joint_score.gene_signature.ips_estimate if self.joint_score.gene_signature else None,
                "predicted_response": self.joint_score.gene_signature.predicted_response if self.joint_score.gene_signature else None,
                "dhe_recommended": self.joint_score.gene_signature.dhe_recommended if self.joint_score.gene_signature else None,
                "overall": self.joint_score.gene_signature.overall if self.joint_score.gene_signature else None,
                "interpretation": self.joint_score.gene_signature.interpretation if self.joint_score.gene_signature else None,
            },
            "circrna_score": {
                "immunotherapy_score": self.joint_score.circrna.immunotherapy_score if self.joint_score.circrna else None,
                "therapeutic_window": self.joint_score.circrna.therapeutic_window if self.joint_score.circrna else None,
                "tumor_killing_index": self.joint_score.circrna.tumor_killing_index if self.joint_score.circrna else None,
                "overall_immunogenicity": self.joint_score.circrna.overall_immunogenicity if self.joint_score.circrna else None,
                "rig_i_score": self.joint_score.circrna.rig_i_score if self.joint_score.circrna else None,
                "tlr_score": self.joint_score.circrna.tlr_score if self.joint_score.circrna else None,
                "pkr_score": self.joint_score.circrna.pkr_score if self.joint_score.circrna else None,
                "tide_score": self.joint_score.circrna.tide_score if self.joint_score.circrna else None,
                "ips": self.joint_score.circrna.ips if self.joint_score.circrna else None,
                "predicted_response": self.joint_score.circrna.predicted_response if self.joint_score.circrna else None,
                "immune_cycle_score": self.joint_score.circrna.immune_cycle_score if self.joint_score.circrna else None,
                "tme_score": self.joint_score.circrna.tme_score if self.joint_score.circrna else None,
                "overall": self.joint_score.circrna.overall if self.joint_score.circrna else None,
                "interpretation": self.joint_score.circrna.interpretation if self.joint_score.circrna else None,
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
            # Gene Signature
            "gs_trop2": self.joint_score.gene_signature.trop2 if self.joint_score.gene_signature else np.nan,
            "gs_nectin4": self.joint_score.gene_signature.nectin4 if self.joint_score.gene_signature else np.nan,
            "gs_liv1": self.joint_score.gene_signature.liv1 if self.joint_score.gene_signature else np.nan,
            "gs_b7h4": self.joint_score.gene_signature.b7h4 if self.joint_score.gene_signature else np.nan,
            "gs_tmem65": self.joint_score.gene_signature.tmem65 if self.joint_score.gene_signature else np.nan,
            "gs_risk": self.joint_score.gene_signature.risk_score if self.joint_score.gene_signature else np.nan,
            "gs_efficacy": self.joint_score.gene_signature.efficacy_score if self.joint_score.gene_signature else np.nan,
            "gs_tide": self.joint_score.gene_signature.tide_score if self.joint_score.gene_signature else np.nan,
            "gs_ips": self.joint_score.gene_signature.ips_estimate if self.joint_score.gene_signature else np.nan,
            "gs_response": self.joint_score.gene_signature.predicted_response if self.joint_score.gene_signature else "N/A",
            "gs_dhe": self.joint_score.gene_signature.dhe_recommended if self.joint_score.gene_signature else False,
            "gs_overall": self.joint_score.gene_signature.overall if self.joint_score.gene_signature else np.nan,
            # CircRNA
            "cr_immunotherapy": self.joint_score.circrna.immunotherapy_score if self.joint_score.circrna else np.nan,
            "cr_therapeutic_window": self.joint_score.circrna.therapeutic_window if self.joint_score.circrna else np.nan,
            "cr_tki": self.joint_score.circrna.tumor_killing_index if self.joint_score.circrna else np.nan,
            "cr_immunogenicity": self.joint_score.circrna.overall_immunogenicity if self.joint_score.circrna else np.nan,
            "cr_rig_i": self.joint_score.circrna.rig_i_score if self.joint_score.circrna else np.nan,
            "cr_tlr": self.joint_score.circrna.tlr_score if self.joint_score.circrna else np.nan,
            "cr_pkr": self.joint_score.circrna.pkr_score if self.joint_score.circrna else np.nan,
            "cr_tide": self.joint_score.circrna.tide_score if self.joint_score.circrna else np.nan,
            "cr_ips": self.joint_score.circrna.ips if self.joint_score.circrna else np.nan,
            "cr_response": self.joint_score.circrna.predicted_response if self.joint_score.circrna else "N/A",
            "cr_immune_cycle": self.joint_score.circrna.immune_cycle_score if self.joint_score.circrna else np.nan,
            "cr_tme": self.joint_score.circrna.tme_score if self.joint_score.circrna else np.nan,
            "cr_overall": self.joint_score.circrna.overall if self.joint_score.circrna else np.nan,
            # PK
            "eval_time_s": self.evaluation_time_s,
        }
        return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class JointEvaluationEngine:
    """Five-dimensional joint drug-epitope-PK-circRNA evaluator.

    Orchestrates the drug pipeline, epitope pipeline, PK simulation,
    and circRNA multi-omics pipeline into a unified evaluation.

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
    use_circrna : bool, optional
        Whether to run the circRNA pipeline. Default: True.

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
        use_circrna: bool = True,
    ):
        self.scoring_engine = scoring_engine or JointScoringEngine()
        self.fusion_layer = fusion_layer or JointFusionLayer()
        self.epitope_backend = epitope_backend
        self.pk_horizon = pk_horizon
        self.drug_compute_profile = drug_compute_profile
        self.use_mhc = use_mhc
        self.use_circrna = use_circrna
        # Optional: circRNA neural encoder for automatic sequence→scoring
        self._circrna_encoder = None
        # Modules loaded lazily at first use
        self._drug_pipeline = None
        self._pkpd = None
        self._epitope_pipeline = None
        self._circrna_pipeline = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_circrna_encoder(self, checkpoint_path: str, device: str = "cpu"):
        """Load a trained circRNA sequence encoder for automatic scoring.

        When set, _run_circrna_pipeline will use the encoder to produce
        the 13-key scoring dict from raw sequence + gene expression,
        instead of the rule-based pipeline. This does NOT modify
        JointScoringEngine or _score_circrna() in any way.

        Parameters
        ----------
        checkpoint_path : str
            Path to the .pt checkpoint saved by train_circrna_encoder.py.
        device : str
            Device for inference ("cpu" or "cuda").
        """
        try:
            from confluencia_circrna.encoder.adapter import CircRNAEncoderAdapter
            self._circrna_encoder = CircRNAEncoderAdapter.from_pretrained(
                checkpoint_path, device=device,
            )
            print(f"[JointEval] circRNA encoder loaded from {checkpoint_path}")
        except Exception as e:
            print(f"[JointEval] Failed to load circRNA encoder: {e}")
            self._circrna_encoder = None

    def _ensure_modules(self):
        """Lazily load drug/epitope/pkpd/circrna modules."""
        if self._drug_pipeline is None:
            try:
                self._drug_pipeline = _import_drug_pipeline()
            except Exception:
                self._drug_pipeline = None
        if self._pkpd is None:
            try:
                self._pkpd = _import_pkpd()
            except Exception:
                self._pkpd = None
        if self._epitope_pipeline is None:
            try:
                self._epitope_pipeline = _import_epitope_pipeline()
            except Exception:
                self._epitope_pipeline = None
        if self._circrna_pipeline is None and self.use_circrna:
            try:
                self._circrna_pipeline = _import_circrna_pipeline()
            except Exception:
                self._circrna_pipeline = None

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

        # --- Step 5: Run circRNA pipeline ---
        circrna_errors: List[str] = []
        all_circrna_outputs: List[Optional[Dict[str, float]]] = []

        if self._circrna_pipeline is not None and self.use_circrna:
            for i, inp in enumerate(inputs):
                try:
                    circrna_out = self._run_circrna_pipeline(inp)
                    all_circrna_outputs.append(circrna_out)
                except Exception as ex:
                    circrna_errors.append(f"circRNA pipeline error for row {i}: {ex}")
                    all_circrna_outputs.append(None)
        else:
            all_circrna_outputs = [None] * len(inputs)

        # --- Step 6: Score and fuse per row ---
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

            # Gene signature outputs (if available on JointInput)
            gene_sig_outputs = getattr(inp, "gene_signature_outputs", None)

            # Auto-compute gene signature from expression values if not provided
            if gene_sig_outputs is None and FiveGeneMOEScorer is not None:
                has_genes = any(getattr(inp, g, 0.5) != 0.5 for g in ["trop2", "nectin4", "liv1", "b7h4", "tmem65"])
                if has_genes:
                    try:
                        gene_dict = inp.to_gene_signature_dict()
                        scorer = FiveGeneMOEScorer()
                        gene_sig_outputs = scorer.to_scoring_dict(gene_dict)
                    except Exception:
                        pass  # model not found, skip

            # CircRNA outputs
            circrna_out = all_circrna_outputs[i]

            # Score
            all_errors = list(drug_errors) + list(epi_errors) + list(circrna_errors)
            if pk_errors:
                all_errors.append(pk_errors[i] if i < len(pk_errors) else "")

            joint_score = self.scoring_engine.score(
                drug_dict, epi_dict, all_pk_summaries[i],
                gene_sig_outputs, circrna_out,
            )

            # Fuse
            gene_sig_arr = None
            if joint_score.gene_signature is not None:
                gene_sig_arr = self.joint_to_arrays(joint_score, "gene_signature")

            circrna_arr = None
            if joint_score.circrna is not None:
                circrna_arr = self.joint_to_arrays(joint_score, "circrna")

            fused = self.fusion_layer.fuse(
                self.joint_to_arrays(joint_score, "clinical"),
                self.joint_to_arrays(joint_score, "binding"),
                self.joint_to_arrays(joint_score, "kinetics"),
                gene_sig_arr,
                circrna_arr,
            )

            # Eval time per sample (proportional share)
            eval_time = (drug_elapsed + epi_elapsed) / len(inputs)

            results.append(JointEvaluationResult(
                input=inp,
                joint_score=joint_score,
                drug_outputs=drug_dict,
                epitope_outputs=epi_dict,
                circrna_outputs=circrna_out or {},
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
        """Run PK simulation for a single input."""
        self._ensure_modules()
        if self._pkpd is None:
            return {
                "pkpd_cmax_mg_per_l": np.nan,
                "pkpd_tmax_h": np.nan,
                "pkpd_half_life_h": np.nan,
                "pkpd_auc_conc": np.nan,
                "pkpd_auc_effect": np.nan,
            }, pd.DataFrame({
                "time_h": [0, self.pk_horizon],
                "pkpd_conc_mg_per_l": [0, 0],
                "pkpd_effect": [0, 0],
            })
        params = self._infer_pk_params(inp)
        curve_df = self._pkpd.simulate_pkpd(
            dose_mg=inp.dose_mg,
            freq_per_day=inp.freq_per_day,
            params=params,
            horizon=self.pk_horizon,
            dt=1.0,
        )
        summary = self._pkpd.summarize_pkpd_curve(curve_df)
        return summary, curve_df

    def _infer_pk_params(self, inp: JointInput):
        """Infer PK parameters from input and defaults."""
        self._ensure_modules()
        if self._pkpd is None:
            return dict(
                ka=0.5, vc=10.0, ke=0.05, kin=0.1,
                target_binding=inp.circ_expr if inp.circ_expr > 0 else 0.3,
                immune_activation=inp.ifn_score if inp.ifn_score > 0 else 0.5,
            )
        return self._pkpd.infer_pkpd_params(
            target_binding=inp.circ_expr if inp.circ_expr > 0 else 0.3,
            immune_activation=inp.ifn_score if inp.ifn_score > 0 else 0.5,
            inflammation=0.1,
            dose_mg=inp.dose_mg,
            freq_per_day=inp.freq_per_day,
        )

    def _run_circrna_pipeline(
        self, inp: JointInput
    ) -> Optional[Dict[str, float]]:
        """Run the circRNA multi-omics pipeline for a single input.

        Uses sequence data and expression data from the JointInput / inputs
        to produce a dict of circRNA outputs consumable by the scoring engine.

        Returns a dict with keys: immunotherapy_score, therapeutic_window,
        tumor_killing_index, overall_immunogenicity, rig_i_score,
        tlr_score, pkr_score, tide_score, ips, predicted_response,
        immune_cycle_score, tme_score, trained_model_risk, overall.
        """
        self._ensure_modules()

        seq = getattr(inp, "circ_sequence", None)
        expr = getattr(inp, "circ_expression_matrix", None)

        if seq is None and expr is None:
            return None

        # Run the circRNA pipeline
        circrna_out = {}
        if self._circrna_pipeline is not None:
            pipeline_mod = self._circrna_pipeline
            pipeline = pipeline_mod.CircRNAPipeline(
                immune_sensing_config=pipeline_mod.ImmuneSensingConfig(),
                immune_cycle_config=pipeline_mod.ImmuneCycleConfig(),
            )

            result = pipeline.run(
                sequences=[seq] if seq else None,
                expr_matrix=expr,
                mutation_data=getattr(inp, "mutation_data", None),
                cnv_data=getattr(inp, "cnv_data", None),
                survival_data=getattr(inp, "survival_data", None),
            )

            if result.sample_scores:
                score = result.sample_scores[0]
                circrna_out = self._circrna_score_to_dict(score)

        # Compute trained survival model risk score
        trained_risk = self._compute_trained_model_risk(inp, expr)
        if trained_risk is not None:
            circrna_out["trained_model_risk"] = trained_risk

        return circrna_out if circrna_out else None

    def _compute_trained_model_risk(
        self,
        inp: JointInput,
        expr_matrix: Optional[pd.DataFrame] = None,
    ) -> Optional[float]:
        """Compute risk score from trained Stepwise Cox model.

        Extracts gene expression values from available sources:
        1. inp.gene_signature_outputs (if precomputed)
        2. expr_matrix columns (if gene symbol columns exist)
        3. Direct gene attributes on JointInput
        """
        if not HAS_TRAINED_MODEL:
            return None

        from confluencia_joint.trained_survival_model import FEATURE_ORDER

        # Source 1: gene_signature_outputs
        gene_sig = getattr(inp, "gene_signature_outputs", None)
        if gene_sig and any(g in gene_sig for g in FEATURE_ORDER):
            gene_expr = {}
            for g in FEATURE_ORDER:
                val = gene_sig.get(g, None)
                if val is not None:
                    gene_expr[g] = float(val)
            if len(gene_expr) >= 5:
                risk, _ = predict_patient_risk(gene_expr)
                return risk

        # Source 2: expression matrix (if provided and has gene columns)
        if expr_matrix is not None and isinstance(expr_matrix, pd.DataFrame):
            available = [g for g in FEATURE_ORDER if g in expr_matrix.columns]
            if len(available) >= 5:
                if len(expr_matrix) > 0:
                    row = expr_matrix.iloc[0]
                    gene_expr = {g: float(row[g]) for g in available}
                    risk, _ = predict_patient_risk(gene_expr)
                    return risk

        return None

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
        score: JointScore, dim: Literal["clinical", "binding", "kinetics", "gene_signature", "circrna"]
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
        elif dim == "kinetics":
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
        elif dim == "circrna":
            cr = score.circrna
            if cr is None:
                return {}
            return {
                "immunotherapy_score": cr.immunotherapy_score,
                "therapeutic_window": cr.therapeutic_window,
                "tumor_killing_index": cr.tumor_killing_index,
                "overall_immunogenicity": cr.overall_immunogenicity,
                "rig_i_score": cr.rig_i_score,
                "tlr_score": cr.tlr_score,
                "pkr_score": cr.pkr_score,
                "tide_score": cr.tide_score,
                "ips": cr.ips,
                "immune_cycle_score": cr.immune_cycle_score,
                "tme_score": cr.tme_score,
                "overall": cr.overall,
            }
        else:  # gene_signature
            g = score.gene_signature
            if g is None:
                return {}
            return {
                "risk_score": g.risk_score,
                "efficacy_score": g.efficacy_score,
                "proliferation_score": g.proliferation_score,
                "immune_score": g.immune_score,
                "mito_score": g.mito_score,
                "tide_score": g.tide_score,
                "ips_estimate": g.ips_estimate,
                "overall": g.overall,
            }

    @staticmethod
    def _circrna_score_to_dict(score) -> Dict[str, float]:
        """Convert a CircRNAScore from the circRNA pipeline to a flat dict."""
        result = {}

        # Immune sensing
        if score.immune_sensing is not None:
            result["rig_i_score"] = score.immune_sensing.rig_i_score
            result["tlr_score"] = score.immune_sensing.tlr_score
            result["pkr_score"] = score.immune_sensing.pkr_score
            result["overall_immunogenicity"] = score.immune_sensing.overall_immunogenicity
        else:
            result["rig_i_score"] = 0.0
            result["tlr_score"] = 0.0
            result["pkr_score"] = 0.0
            result["overall_immunogenicity"] = 0.5

        # Immune cycle
        if score.immune_cycle is not None:
            result["immune_cycle_score"] = float(np.mean([
                score.immune_cycle.antigen_release,
                score.immune_cycle.dc_priming,
                score.immune_cycle.t_cell_priming,
                score.immune_cycle.trafficking,
                score.immune_cycle.infiltration,
                score.immune_cycle.recognition,
                score.immune_cycle.killing,
            ]))
            result["tumor_killing_index"] = score.immune_cycle.tumor_killing_index
            result["therapeutic_window"] = score.immune_cycle.therapeutic_window
        else:
            result["immune_cycle_score"] = 0.5
            result["tumor_killing_index"] = 0.5
            result["therapeutic_window"] = 0.5

        # TME
        if score.tme is not None and score.tme.cell_fractions:
            fracs = list(score.tme.cell_fractions.values())
            cd8_frac = score.tme.cell_fractions.get("T cells CD8", 0.0)
            result["tme_score"] = float(np.clip(cd8_frac, 0, 1))
        else:
            result["tme_score"] = 0.5

        # Immune evasion
        if score.immune_evasion is not None:
            result["tide_score"] = score.immune_evasion.tide_score
            result["ips"] = score.immune_evasion.ips
            result["predicted_response"] = score.immune_evasion.predicted_response
        else:
            result["tide_score"] = 0.5
            result["ips"] = 5.0
            result["predicted_response"] = "intermediate"

        # Composite scores
        result["immunotherapy_score"] = score.immunotherapy_score
        result["overall"] = score.therapeutic_potential

        return result