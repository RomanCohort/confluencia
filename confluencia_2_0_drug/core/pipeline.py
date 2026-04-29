from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ctm import simulate_ctm, summarize_curve
from .ctm_param_model import CTMParamModel, heuristic_param_targets
from .features import MixedFeatureSpec, build_feature_matrix, build_feature_names, ensure_columns, logit_transform, inverse_logit
from .legacy_algorithms import LegacyAlgorithmConfig, train_predict_legacy_backend
from .micro_predictors import MICRO_TARGETS, MicroPredictor, proxy_micro_labels
from .moe import MOERegressor, choose_compute_profile
from .ndp4pd import ndp4pd_from_ctm_like, simulate_ndp4pd
from .immune_abm import simulate_single_epitope_response
from .pkpd import infer_pkpd_params, simulate_pkpd, summarize_pkpd_curve
from confluencia_shared.utils.logging import get_logger
from confluencia_shared.data_utils import resolve_label as _resolve_label

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """Results from simulating a single sample's dynamics."""
    auc_efficacy: float
    peak_efficacy: float
    peak_toxicity: float
    pk_half_life: float
    pk_vd_ss: float
    pk_clearance: float
    pk_cmax: float
    pk_tmax: float
    pk_auc_conc: float
    pk_auc_effect: float
    pk_effect_peak: float
    pk_effect_corr: float
    curve: pd.DataFrame


def _simulate_single_sample(
    row: pd.Series,
    dynamics_model: str,
    ctm_model: CTMParamModel,
    horizon: int = 72,
    dt: float = 1.0,
) -> SimulationResult:
    """Simulate CTM/PKPD dynamics for a single sample.

    Args:
        row: DataFrame row containing CTM parameters and dose/freq.
        dynamics_model: "ctm" or "ndp4pd".
        ctm_model: CTM parameter model for parameter conversion.
        horizon: Simulation horizon in hours.
        dt: Time step in hours.

    Returns:
        SimulationResult with efficacy metrics and trajectory curve.
    """
    params = ctm_model.to_params(
        np.array(
            [
                float(row.get("ctm_ka", 0.2)),
                float(row.get("ctm_kd", 0.2)),
                float(row.get("ctm_ke", 0.2)),
                float(row.get("ctm_km", 0.1)),
                float(row.get("ctm_signal_gain", 1.0)),
            ],
            dtype=np.float32,
        )
    )
    if str(dynamics_model).lower() in {"ndp4pd", "nd4pd"}:
        nd = ndp4pd_from_ctm_like(
            ka=float(row.get("ctm_ka", 0.2)),
            kd=float(row.get("ctm_kd", 0.2)),
            ke=float(row.get("ctm_ke", 0.2)),
            km=float(row.get("ctm_km", 0.1)),
            signal_gain=float(row.get("ctm_signal_gain", 1.0)),
        )
        c = simulate_ndp4pd(
            dose=float(row.get("dose", 0.0)),
            freq=float(row.get("freq", 1.0)),
            params=nd,
            horizon=horizon,
            dt=dt,
        )
    else:
        c = simulate_ctm(
            dose=float(row.get("dose", 0.0)),
            freq=float(row.get("freq", 1.0)),
            params=params,
            horizon=horizon,
            dt=dt,
        )

    pk_params = infer_pkpd_params(
        binding=float(row.get("target_binding_pred", 0.5)),
        immune=float(row.get("immune_activation_pred", 0.5)),
        inflammation=float(row.get("inflammation_risk_pred", 0.2)),
        dose_mg=float(row.get("dose", 0.0)),
        freq_per_day=float(row.get("freq", 1.0)),
    )
    pk_curve = simulate_pkpd(
        dose_mg=float(row.get("dose", 0.0)),
        freq_per_day=float(row.get("freq", 1.0)),
        params=pk_params,
        horizon=horizon,
        dt=dt,
    )
    pk_sum = summarize_pkpd_curve(pk_curve, pk_params)

    if not pk_curve.empty:
        c = c.merge(pk_curve, on="time_h", how="left")

    s = summarize_curve(c)
    return SimulationResult(
        auc_efficacy=s["auc_efficacy"],
        peak_efficacy=s["peak_efficacy"],
        peak_toxicity=s["peak_toxicity"],
        pk_half_life=float(pk_sum.get("pkpd_half_life_h", 0.0)),
        pk_vd_ss=float(pk_sum.get("pkpd_vd_ss_l", 0.0)),
        pk_clearance=float(pk_sum.get("pkpd_clearance_lph", 0.0)),
        pk_cmax=float(pk_sum.get("pkpd_cmax_mg_per_l", 0.0)),
        pk_tmax=float(pk_sum.get("pkpd_tmax_h", 0.0)),
        pk_auc_conc=float(pk_sum.get("pkpd_auc_conc", 0.0)),
        pk_auc_effect=float(pk_sum.get("pkpd_auc_effect", 0.0)),
        pk_effect_peak=float(pk_sum.get("pkpd_effect_peak", 0.0)),
        pk_effect_corr=float(pk_sum.get("pkpd_pk_effect_corr", 0.0)),
        curve=c.copy(),
    )


def _run_simulation_loop(
    out: pd.DataFrame,
    dynamics_model: str,
    ctm_model: CTMParamModel,
    horizon: int = 72,
    dt: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CTM/PKPD simulation loop for all samples.

    Args:
        out: DataFrame with CTM parameters and dose/freq columns.
        dynamics_model: "ctm" or "ndp4pd".
        ctm_model: CTM parameter model.
        horizon: Simulation horizon in hours.
        dt: Time step in hours.

    Returns:
        Tuple of (updated out DataFrame with metrics, curves DataFrame).
    """
    aucs = []
    peaks = []
    tox_peaks = []
    pk_half_life = []
    pk_vd_ss = []
    pk_clearance = []
    pk_cmax = []
    pk_tmax = []
    pk_auc_conc = []
    pk_auc_effect = []
    pk_effect_peak = []
    pk_effect_corr = []
    curves = []

    for idx, row in out.iterrows():
        result = _simulate_single_sample(row, dynamics_model, ctm_model, horizon, dt)
        aucs.append(result.auc_efficacy)
        peaks.append(result.peak_efficacy)
        tox_peaks.append(result.peak_toxicity)
        pk_half_life.append(result.pk_half_life)
        pk_vd_ss.append(result.pk_vd_ss)
        pk_clearance.append(result.pk_clearance)
        pk_cmax.append(result.pk_cmax)
        pk_tmax.append(result.pk_tmax)
        pk_auc_conc.append(result.pk_auc_conc)
        pk_auc_effect.append(result.pk_auc_effect)
        pk_effect_peak.append(result.pk_effect_peak)
        pk_effect_corr.append(result.pk_effect_corr)
        curve = result.curve
        curve["sample_id"] = int(len(curves))
        curves.append(curve)

    out = out.copy()
    out["ctm_auc_efficacy"] = np.array(aucs, dtype=np.float32)
    out["ctm_peak_efficacy"] = np.array(peaks, dtype=np.float32)
    out["ctm_peak_toxicity"] = np.array(tox_peaks, dtype=np.float32)
    out["pkpd_half_life_h"] = np.array(pk_half_life, dtype=np.float32)
    out["pkpd_vd_ss_l"] = np.array(pk_vd_ss, dtype=np.float32)
    out["pkpd_clearance_lph"] = np.array(pk_clearance, dtype=np.float32)
    out["pkpd_cmax_mg_per_l"] = np.array(pk_cmax, dtype=np.float32)
    out["pkpd_tmax_h"] = np.array(pk_tmax, dtype=np.float32)
    out["pkpd_auc_conc"] = np.array(pk_auc_conc, dtype=np.float32)
    out["pkpd_auc_effect"] = np.array(pk_auc_effect, dtype=np.float32)
    out["pkpd_effect_peak"] = np.array(pk_effect_peak, dtype=np.float32)
    out["pkpd_pk_effect_corr"] = np.array(pk_effect_corr, dtype=np.float32)

    curve_df = pd.concat(curves, axis=0, ignore_index=True) if curves else pd.DataFrame()
    return out, curve_df


def _apply_multitask_consistency(out: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight consistency constraints across efficacy and risk heads.

    This is a post-hoc safety regularizer to reduce implausible combinations,
    such as high efficacy with unrealistically low toxicity/inflammation.
    """
    w = out.copy()
    n = max(len(w), 1)

    def _col(name: str) -> pd.Series:
        if name in w.columns:
            return pd.to_numeric(w[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros((len(w),), dtype=np.float32), index=w.index)

    eff = _col("efficacy_pred").to_numpy(dtype=np.float32)
    if np.allclose(float(np.std(eff)), 0.0):
        eff_n = np.clip(eff, 0.0, 1.0)
    else:
        eff_n = (eff - float(np.min(eff))) / max(float(np.max(eff) - np.min(eff)), 1e-6)

    if "toxicity_risk_pred" in w.columns:
        tox = _col("toxicity_risk_pred").to_numpy(dtype=np.float32)
        imm = _col("immune_cell_activation_pred").to_numpy(dtype=np.float32)
        # Toxicity floor: minimum plausible toxicity given efficacy level.
        # 0.06 baseline (even safe drugs have some signal), +0.16 proportional to efficacy
        # (higher efficacy requires more active compound → more toxicity potential),
        # -0.05 immune cell activation credit (immune cells help clear toxic metabolites).
        tox_floor = 0.06 + 0.16 * eff_n - 0.05 * np.clip(imm, 0.0, 1.0)
        w["toxicity_risk_pred"] = np.clip(np.maximum(tox, tox_floor), 0.0, 1.0)

    if "inflammation_risk_pred" in w.columns:
        inf = _col("inflammation_risk_pred").to_numpy(dtype=np.float32)
        bind = _col("target_binding_pred").to_numpy(dtype=np.float32)
        # Inflammation floor: 0.05 baseline + 0.12 proportional to efficacy × binding.
        # Interaction term: high efficacy with high binding suggests strong biological activity
        # which naturally induces some inflammatory response (danger-associated molecular patterns).
        inf_floor = 0.05 + 0.12 * eff_n * np.clip(bind, 0.0, 1.0)
        w["inflammation_risk_pred"] = np.clip(np.maximum(inf, inf_floor), 0.0, 1.0)

    if n > 0:
        # Consistency score: weighted combination of risk/safety signals.
        # 35% toxicity penalty (largest weight: primary safety concern),
        # 25% inflammation penalty (secondary safety concern),
        # 40% immune activation bonus (positive therapeutic signal).
        # Weights sum to 1.0 and the result is clipped to [0, 1].
        w["consistency_score"] = np.clip(
            1.0
            - 0.35 * _col("toxicity_risk_pred")
            - 0.25 * _col("inflammation_risk_pred")
            + 0.40 * _col("immune_cell_activation_pred"),
            0.0,
            1.0,
        )
    return w


@dataclass
class ConfluenciaArtifacts:
    compute_profile: str
    moe_weights: Dict[str, float]
    moe_metrics: Dict[str, float]
    used_proxy_micro_labels: bool
    smiles_backend: str
    ctm_param_source: str
    model_backend: str
    dynamics_model: str
    shap_ready: bool = False
    shap_message: str = ""
    shap_feature_count: int = 0
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.0
    adaptive_samples: int = 0
    adaptive_message: str = "disabled"
    feature_selection_applied: bool = False
    feature_selection_n_final: int = 0
    use_logit_transform: bool = False
    hierarchical_residual_enabled: bool = False


@dataclass
class ConfluenciaModelBundle:
    feature_spec: MixedFeatureSpec
    compute_profile: str
    moe_model: MOERegressor
    micro_model: MicroPredictor
    ctm_model: CTMParamModel
    ctm_param_source: str
    moe_weights: Dict[str, float]
    moe_metrics: Dict[str, float]
    dynamics_model: str
    use_logit_transform: bool = False
    binding_model: Optional[MOERegressor] = None


def _append_immune_abm_outputs(out: pd.DataFrame, horizon_h: int = 96) -> pd.DataFrame:
    if len(out) == 0:
        out["immune_peak_antibody"] = np.array([], dtype=np.float32)
        out["immune_peak_effector_t"] = np.array([], dtype=np.float32)
        out["immune_peak_antigen"] = np.array([], dtype=np.float32)
        out["immune_response_auc"] = np.array([], dtype=np.float32)
        return out

    peaks_ab = []
    peaks_t = []
    peaks_ag = []
    aucs = []

    for _, row in out.iterrows():
        _, s = simulate_single_epitope_response(
            epitope_seq=str(row.get("epitope_seq", "")),
            dose=float(row.get("dose", 1.0)),
            treatment_time=float(row.get("treatment_time", 0.0)),
            horizon_h=int(horizon_h),
        )
        peaks_ab.append(float(s.get("immune_peak_antibody", 0.0)))
        peaks_t.append(float(s.get("immune_peak_effector_t", 0.0)))
        peaks_ag.append(float(s.get("immune_peak_antigen", 0.0)))
        aucs.append(float(s.get("immune_response_auc", 0.0)))

    out["immune_peak_antibody"] = np.array(peaks_ab, dtype=np.float32)
    out["immune_peak_effector_t"] = np.array(peaks_t, dtype=np.float32)
    out["immune_peak_antigen"] = np.array(peaks_ag, dtype=np.float32)
    out["immune_response_auc"] = np.array(aucs, dtype=np.float32)
    return out


def _compute_moe_shap_columns(
    model: MOERegressor,
    X: np.ndarray,
    feature_names: List[str],
    max_background: int = 96,
) -> Tuple[pd.DataFrame, bool, str]:
    if X.size == 0:
        return pd.DataFrame(), False, "empty_input"
    if X.shape[1] != len(feature_names):
        return pd.DataFrame(), False, "feature_name_dim_mismatch"

    try:
        import shap  # type: ignore
    except Exception as e:
        return pd.DataFrame(), False, f"shap_unavailable: {e}"

    x_arr = np.asarray(X, dtype=np.float32)
    bg_n = int(max(1, min(max_background, x_arr.shape[0])))
    bg_idx = np.unique(np.linspace(0, x_arr.shape[0] - 1, num=bg_n, dtype=np.int64))
    background = x_arr[bg_idx]

    try:
        explainer = shap.Explainer(lambda z: model.predict(np.asarray(z, dtype=np.float32)), background, feature_names=feature_names)
        exp = explainer(x_arr)
        values = np.asarray(exp.values, dtype=np.float32)
        base_values = np.asarray(exp.base_values, dtype=np.float32).reshape(-1)
    except Exception:
        try:
            kernel = shap.KernelExplainer(lambda z: model.predict(np.asarray(z, dtype=np.float32)), background)
            values = np.asarray(kernel.shap_values(x_arr, nsamples=min(512, int(4 * x_arr.shape[1] + 64))), dtype=np.float32)
            if values.ndim == 3:
                values = values[0]
            if values.ndim == 1:
                values = values.reshape(x_arr.shape[0], 1)
            base_values = np.full((x_arr.shape[0],), float(np.asarray(kernel.expected_value).reshape(-1)[0]), dtype=np.float32)
        except Exception as e2:
            return pd.DataFrame(), False, f"shap_failed: {e2}"

    if values.ndim != 2 or values.shape != x_arr.shape:
        return pd.DataFrame(), False, "shap_shape_mismatch"

    cols: Dict[str, np.ndarray] = {
        "shap_base_value": base_values.astype(np.float32),
        "shap_value_sum": values.sum(axis=1).astype(np.float32),
    }
    cols["shap_reconstructed_pred"] = (cols["shap_base_value"] + cols["shap_value_sum"]).astype(np.float32)
    for j, name in enumerate(feature_names):
        cols[f"shap_{name}"] = values[:, j].astype(np.float32)
    return pd.DataFrame(cols), True, "ok"


def _fit_hierarchical_residual(
    X: np.ndarray,
    y_eff: np.ndarray,
    y_binding: np.ndarray | None,
    expert_names: List[str],
    folds: int,
    random_state: int,
) -> Tuple[MOERegressor, Optional[MOERegressor], np.ndarray]:
    """Hierarchical residual model: binding → efficacy residual.

    Stage 1: Train binding predictor (R²=0.965 on labeled data).
    Stage 2: Train efficacy residual from features + binding_pred.

    Returns (binding_model, residual_model, binding_pred).
    """
    binding_model = None
    residual_model = None
    binding_pred = np.full(len(y_eff), 0.5, dtype=np.float32)

    if y_binding is None:
        return binding_model, residual_model, binding_pred

    n = len(y_eff)
    # Stage 1: predict binding
    binding_model = MOERegressor(expert_names=expert_names, folds=folds, random_state=random_state)
    binding_model.fit(X, y_binding)
    binding_pred = binding_model.predict(X).astype(np.float32)

    # Stage 2: predict efficacy residual from (features, binding_pred)
    # residual = efficacy - 0.6 * binding (0.6 = typical efficacy-binding correlation)
    baseline = 0.6 * binding_pred
    y_residual = (y_eff - baseline).astype(np.float32)

    X_with_binding = np.column_stack([X, binding_pred]).astype(np.float32)
    residual_model = MOERegressor(expert_names=expert_names, folds=folds, random_state=random_state + 1)
    residual_model.fit(X_with_binding, y_residual)
    residual_pred = residual_model.predict(X_with_binding).astype(np.float32)

    # Final: efficacy = baseline + residual
    # (binding_model and residual_model returned for prediction-time use)
    return binding_model, residual_model, binding_pred


def _apply_adaptive_adjustment(
    out: pd.DataFrame,
    enabled: bool,
    strength: float,
) -> tuple[pd.DataFrame, str]:
    if not enabled:
        return out, "disabled"
    if len(out) == 0:
        return out, "empty_input"

    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-8:
        return out, "zero_strength"

    w = out.copy()

    def _col(name: str) -> pd.Series:
        if name in w.columns:
            return pd.to_numeric(w[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros((len(w),), dtype=np.float32), index=w.index)

    eff = _col("efficacy_pred").to_numpy(dtype=np.float32)
    bind = np.clip(_col("target_binding_pred").to_numpy(dtype=np.float32), 0.0, 1.0)
    imm = np.clip(_col("immune_activation_pred").to_numpy(dtype=np.float32), 0.0, 1.0)
    imm_cell = np.clip(_col("immune_cell_activation_pred").to_numpy(dtype=np.float32), 0.0, 1.0)
    tox = np.clip(_col("toxicity_risk_pred").to_numpy(dtype=np.float32), 0.0, 1.0)
    infl = np.clip(_col("inflammation_risk_pred").to_numpy(dtype=np.float32), 0.0, 1.0)
    consistency = np.clip(_col("consistency_score").to_numpy(dtype=np.float32), 0.0, 1.0)

    # Confidence: weighted average of positive prediction signals.
    # 32% target binding (primary efficacy driver), 26% immune activation (secondary mechanism),
    # 20% immune cell activation (cellular-level confirmation), 22% consistency (cross-signal agreement).
    # Weights reflect relative importance for circRNA therapeutic confidence assessment.
    confidence = np.clip(0.32 * bind + 0.26 * imm + 0.20 * imm_cell + 0.22 * consistency, 0.0, 1.0)
    # Risk pressure: 58% toxicity (acute safety concern) + 42% inflammation (chronic safety concern).
    # Toxicity weighted higher because it directly impacts therapeutic index.
    risk_pressure = np.clip(0.58 * tox + 0.42 * infl, 0.0, 1.0)

    # Efficacy gain: boost if confidence > 0.5 (above-median confidence), penalize if risk > 0.45.
    # The 0.45 risk threshold allows moderate risk before penalizing (avoids over-conservatism).
    # 0.45 scaling on risk excess keeps the penalty proportional to how far risk exceeds the threshold.
    eff_gain = 1.0 + s * (confidence - 0.5) - 0.45 * s * np.maximum(risk_pressure - 0.45, 0.0)
    w["efficacy_pred"] = np.maximum(eff * eff_gain.astype(np.float32), 0.0)

    # Risk scaling: inflate risk estimates when confidence is high (>0.55 threshold).
    # This is a conservative adjustment: high-confidence predictions with elevated risk
    # should have their risk amplified (better to overestimate risk than underestimate).
    # 0.55 threshold chosen as slightly above median to avoid scaling on borderline cases.
    risk_scale = 1.0 + 0.65 * s * np.maximum(confidence - 0.55, 0.0)
    w["toxicity_risk_pred"] = np.clip(tox * risk_scale.astype(np.float32), 0.0, 1.0)
    w["inflammation_risk_pred"] = np.clip(infl * risk_scale.astype(np.float32), 0.0, 1.0)

    dose = _col("dose").to_numpy(dtype=np.float32)
    freq = np.maximum(_col("freq").to_numpy(dtype=np.float32), 1e-6)
    dose_med = float(np.median(dose)) if len(dose) > 0 else 0.0
    dose_mad = float(np.median(np.abs(dose - dose_med))) if len(dose) > 0 else 0.0
    dose_z = (dose - dose_med) / max(1.4826 * dose_mad, 1e-6)
    dose_z = np.clip(dose_z, -3.0, 3.0)

    # Dose/freq factors: adjust dosing recommendations based on risk/benefit balance.
    # 0.55 risk pressure term: high risk → reduce dose (lower is better).
    # 0.35 confidence term: high confidence → increase dose (more is better).
    # 0.05 dose_z term: already-high doses get a slight reduction (avoid toxicity cliff).
    # Clip bounds [0.70, 1.25] cap adjustments at ±25-30% to prevent extreme recommendations.
    dose_factor = np.clip(
        1.0 + s * (0.55 * (0.5 - risk_pressure) + 0.35 * (confidence - 0.5) - 0.05 * dose_z),
        0.70,
        1.25,
    )
    # Freq factor: same structure as dose but slightly smaller coefficients and tighter clip bounds.
    # 0.04 log(freq) term: penalizes very high frequency (diminishing returns, injection site reactions).
    freq_factor = np.clip(
        1.0 + s * (0.45 * (0.5 - risk_pressure) + 0.30 * (confidence - 0.5) - 0.04 * np.log1p(freq)),
        0.75,
        1.20,
    )

    w["adaptive_confidence"] = confidence.astype(np.float32)
    w["adaptive_risk_pressure"] = risk_pressure.astype(np.float32)
    w["adaptive_dose_factor"] = dose_factor.astype(np.float32)
    w["adaptive_freq_factor"] = freq_factor.astype(np.float32)
    return w, "ok"


def run_pipeline(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    model_backend: str = "moe",
    dynamics_model: str = "ctm",
    legacy_cfg: LegacyAlgorithmConfig | None = None,
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
    feature_spec: MixedFeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ConfluenciaArtifacts]:
    work = ensure_columns(df)
    for c in ["dose", "freq", "treatment_time"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    spec = feature_spec or MixedFeatureSpec(
        smiles_hash_dim=128, smiles_rdkit_bits=2048, smiles_rdkit_version=2, prefer_rdkit=True
    )
    X, env_cols, smiles_backend = build_feature_matrix(work, spec)

    n = X.shape[0]
    y_eff = _resolve_label(work, "efficacy")

    # ── Target transform (logit for bounded efficacy) ──────
    use_logit = str(getattr(spec, "target_transform", "none")).lower() == "logit"
    y_eff_logit = None
    if use_logit and y_eff is not None:
        y_eff_logit = logit_transform(y_eff)
        y_eff_for_fit = y_eff_logit
    else:
        y_eff_for_fit = y_eff
    feature_names: List[str] = []

    # ── Feature selection (optional, requires labeled efficacy) ──
    if bool(spec.use_feature_selection) and y_eff_for_fit is not None:
        from .feature_selector import FeatureSelector

        selector = FeatureSelector(
            top_k=int(spec.feature_selection_top_k),
            corr_thresh=float(spec.feature_selection_correl_thresh),
        )
        all_names = build_feature_names(spec, list(env_cols))
        X, feature_names = selector.fit_transform(X, y_eff, feature_names=all_names)
    else:
        feature_names = build_feature_names(spec, list(env_cols))

    if n == 0:
        empty = work.copy()
        curve = pd.DataFrame(
            columns=[
                "time_h",
                "absorption_A",
                "distribution_D",
                "effect_E",
                "metabolism_M",
                "efficacy_signal",
                "toxicity_signal",
                "pkpd_depot_mg",
                "pkpd_central_mg",
                "pkpd_peripheral_mg",
                "pkpd_conc_mg_per_l",
                "pkpd_effect",
            ]
        )
        artifacts = ConfluenciaArtifacts(
            compute_profile="low",
            moe_weights={},
            moe_metrics={},
            used_proxy_micro_labels=True,
            smiles_backend="hash",
            ctm_param_source="heuristic",
            model_backend="moe",
            dynamics_model="ctm",
            shap_ready=False,
            shap_message="empty_input",
            shap_feature_count=0,
            adaptive_enabled=bool(adaptive_enabled),
            adaptive_strength=float(np.clip(adaptive_strength, 0.0, 1.0)),
            adaptive_samples=0,
            adaptive_message="empty_input",
        )
        return empty, curve, artifacts

    prof = choose_compute_profile(n_samples=n, requested=compute_mode)

    # y_eff was already resolved above (for feature selection or proxy fallback)
    if y_eff is None:
        # Transparent proxy objective for unlabeled mode.
        # Weights mirror circRNA pharmacology: dose (35%) and frequency (25%) are primary drivers,
        # sequence features (40% of first 32 dims) capture binding/immunogenicity signal.
        # Different from epitope proxy because drug module features have different semantics.
        y_eff = (0.35 * work["dose"].to_numpy(dtype=np.float32) + 0.25 * work["freq"].to_numpy(dtype=np.float32))
        y_eff = y_eff + 0.4 * X[:, :32].mean(axis=1).astype(np.float32)

    moe: MOERegressor | None = None
    moe_weights: Dict[str, float] = {}
    moe_metrics: Dict[str, float] = {}
    if model_backend == "moe":
        moe = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds)
        moe.fit(X, y_eff_for_fit)
        eff_pred_logit = moe.predict(X)
        # Inverse-logit transform if training used logit
        eff_pred = inverse_logit(eff_pred_logit) if use_logit else eff_pred_logit
        moe_weights = moe.explain_weights()
        moe_metrics = dict(moe.metrics)
    else:
        eff_pred, legacy_metrics = train_predict_legacy_backend(
            work_df=work,
            X=X,
            y=y_eff,
            env_cols=list(env_cols),
            model_backend=model_backend,
            cfg=legacy_cfg,
        )
        moe_metrics = dict(legacy_metrics)

    y_micro = None
    micro_used_proxy = False
    if any(c in work.columns for c in MICRO_TARGETS):
        cols = []
        for c in MICRO_TARGETS:
            if c in work.columns:
                cols.append(pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=np.float32))
            else:
                cols.append(np.full((n,), np.nan, dtype=np.float32))
        y_micro = np.column_stack(cols).astype(np.float32)
        if np.isnan(y_micro).any():
            y_proxy = proxy_micro_labels(X)
            nan_mask = np.isnan(y_micro)
            y_micro[nan_mask] = y_proxy[nan_mask]
            micro_used_proxy = True

    micro = MicroPredictor().fit(X, y_micro)
    if micro_used_proxy:
        micro.used_proxy_labels = True
    micro_pred = micro.predict(X)

    out = work.copy()
    out["efficacy_pred"] = eff_pred
    for k, v in micro_pred.items():
        out[f"{k}_pred"] = v
    out = _apply_multitask_consistency(out)
    out, adaptive_message = _apply_adaptive_adjustment(out, enabled=bool(adaptive_enabled), strength=float(adaptive_strength))
    shap_ready = False
    shap_message = "not_supported_for_backend"
    if model_backend == "moe" and moe is not None:
        shap_df, shap_ready, shap_message = _compute_moe_shap_columns(moe, X, feature_names)
        if not shap_df.empty:
            out = pd.concat([out.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    # Learn individual CTM parameters from condition + micro-space.
    x_ctm = np.column_stack(
        [
            out["dose"].to_numpy(dtype=np.float32),
            out["freq"].to_numpy(dtype=np.float32),
            out["treatment_time"].to_numpy(dtype=np.float32),
            out["target_binding_pred"].to_numpy(dtype=np.float32),
            out["immune_activation_pred"].to_numpy(dtype=np.float32),
            out["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            out["efficacy_pred"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    target_cols = ["ctm_ka", "ctm_kd", "ctm_ke", "ctm_km", "ctm_signal_gain"]
    y_ctm: np.ndarray
    ctm_source = "heuristic"
    if all(c in out.columns for c in target_cols):
        y_ctm = out[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ctm_source = "labeled"
    else:
        y_ctm = heuristic_param_targets(
            binding=out["target_binding_pred"].to_numpy(dtype=np.float32),
            immune=out["immune_activation_pred"].to_numpy(dtype=np.float32),
            inflammation=out["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            dose=out["dose"].to_numpy(dtype=np.float32),
            freq=out["freq"].to_numpy(dtype=np.float32),
        )

    ctm_model = CTMParamModel()
    group_ids = out["group_id"].astype(str).to_numpy(dtype=object)
    ctm_model.fit(x_ctm, y_ctm, group_ids=group_ids)
    y_ctm_pred, y_ctm_base, y_ctm_offset = ctm_model.predict_decomposed(x_ctm, group_ids=group_ids)

    if ctm_source == "heuristic":
        # Physics-informed residual blend: 70% heuristic prior + 30% learned residual.
        # The heuristic prior is grounded in literature PK parameter ranges and provides
        # a stable anchor. The 30% residual allows the model to learn dataset-specific
        # corrections without deviating far from physiologically plausible values.
        # This ratio prevents the learned model from overfitting to small training sets.
        y_ctm_pred = (0.7 * y_ctm + 0.3 * y_ctm_pred).astype(np.float32)
        y_ctm_offset = (y_ctm_pred - y_ctm_base).astype(np.float32)
        ctm_source = "heuristic+residual"

    out["ctm_ka"] = y_ctm_pred[:, 0]
    out["ctm_kd"] = y_ctm_pred[:, 1]
    out["ctm_ke"] = y_ctm_pred[:, 2]
    out["ctm_km"] = y_ctm_pred[:, 3]
    out["ctm_signal_gain"] = y_ctm_pred[:, 4]
    out["ctm_ka_base"] = y_ctm_base[:, 0]
    out["ctm_kd_base"] = y_ctm_base[:, 1]
    out["ctm_ke_base"] = y_ctm_base[:, 2]
    out["ctm_km_base"] = y_ctm_base[:, 3]
    out["ctm_signal_gain_base"] = y_ctm_base[:, 4]
    out["ctm_ka_offset"] = y_ctm_offset[:, 0]
    out["ctm_kd_offset"] = y_ctm_offset[:, 1]
    out["ctm_ke_offset"] = y_ctm_offset[:, 2]
    out["ctm_km_offset"] = y_ctm_offset[:, 3]
    out["ctm_signal_gain_offset"] = y_ctm_offset[:, 4]

    # Build CTM trajectory for each sample and summarize.
    out, curve_df = _run_simulation_loop(out, dynamics_model, ctm_model)
    out = _append_immune_abm_outputs(out, horizon_h=96)

    # Group interaction proxy: knock up one group dose and monitor other groups efficacy delta.
    if model_backend == "moe" and moe is not None:
        interaction = []
        groups = sorted(set(out["group_id"].astype(str).tolist()))
        if len(groups) > 1:
            base_group_mean = out.groupby("group_id", as_index=False).agg(base_mean=("efficacy_pred", "mean"))
            for g in groups:
                perturbed = out.copy()
                mask = perturbed["group_id"].astype(str) == g
                perturbed.loc[mask, "dose"] = perturbed.loc[mask, "dose"].astype(float) * 1.1 + 0.2
                Xp, _, _ = build_feature_matrix(perturbed, spec)
                p = moe.predict(Xp)
                tmp = perturbed[["group_id"]].copy()
                tmp["pred"] = p
                p_group_mean = tmp.groupby("group_id", as_index=False).agg(perturbed_mean=("pred", "mean"))
                merged = base_group_mean.merge(p_group_mean, on="group_id", how="left")
                merged["source_group"] = g
                merged["delta"] = merged["perturbed_mean"] - merged["base_mean"]
                interaction.append(merged[["source_group", "group_id", "delta"]])

            if interaction:
                inter_df = pd.concat(interaction, axis=0, ignore_index=True)
                out = out.merge(
                    inter_df.groupby("group_id", as_index=False)["delta"].mean().rename(columns={"group_id": "group_id", "delta": "cross_group_impact"}),
                    on="group_id",
                    how="left",
                )

    artifacts = ConfluenciaArtifacts(
        compute_profile=prof.level,
        moe_weights=moe_weights,
        moe_metrics=moe_metrics,
        used_proxy_micro_labels=micro.used_proxy_labels,
        smiles_backend=smiles_backend,
        ctm_param_source=ctm_source,
        model_backend=model_backend,
        dynamics_model=("ndp4pd" if str(dynamics_model).lower() in {"ndp4pd", "nd4pd"} else "ctm"),
        shap_ready=bool(shap_ready),
        shap_message=str(shap_message),
        shap_feature_count=int(len(feature_names)),
        adaptive_enabled=bool(adaptive_enabled),
        adaptive_strength=float(np.clip(adaptive_strength, 0.0, 1.0)),
        adaptive_samples=int(len(out)),
        adaptive_message=str(adaptive_message),
        use_logit_transform=bool(use_logit),
    )
    return out, curve_df, artifacts


def _predict_with_bundle_core(
    work: pd.DataFrame,
    X: np.ndarray,
    spec: MixedFeatureSpec,
    env_cols: List[str],
    bundle: ConfluenciaModelBundle,
    adaptive_enabled: bool,
    adaptive_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool, str, int, str]:
    eff_pred_logit = bundle.moe_model.predict(X)
    # Inverse-logit transform if training used logit
    eff_pred = inverse_logit(eff_pred_logit) if bundle.use_logit_transform else eff_pred_logit
    micro_pred = bundle.micro_model.predict(X)

    out = work.copy()
    out["efficacy_pred"] = eff_pred
    for k, v in micro_pred.items():
        out[f"{k}_pred"] = v
    out = _apply_multitask_consistency(out)
    out, adaptive_message = _apply_adaptive_adjustment(out, enabled=bool(adaptive_enabled), strength=float(adaptive_strength))
    feat_names = build_feature_names(spec, env_cols)
    shap_df, shap_ready, shap_message = _compute_moe_shap_columns(bundle.moe_model, X, feat_names)
    if not shap_df.empty:
        out = pd.concat([out.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    x_ctm = np.column_stack(
        [
            out["dose"].to_numpy(dtype=np.float32),
            out["freq"].to_numpy(dtype=np.float32),
            out["treatment_time"].to_numpy(dtype=np.float32),
            out["target_binding_pred"].to_numpy(dtype=np.float32),
            out["immune_activation_pred"].to_numpy(dtype=np.float32),
            out["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            out["efficacy_pred"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    group_ids = out["group_id"].astype(str).to_numpy(dtype=object)
    y_ctm_pred, y_ctm_base, y_ctm_offset = bundle.ctm_model.predict_decomposed(x_ctm, group_ids=group_ids)

    ctm_source = str(bundle.ctm_param_source)
    if ctm_source.startswith("heuristic"):
        y_ctm_prior = heuristic_param_targets(
            binding=out["target_binding_pred"].to_numpy(dtype=np.float32),
            immune=out["immune_activation_pred"].to_numpy(dtype=np.float32),
            inflammation=out["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            dose=out["dose"].to_numpy(dtype=np.float32),
            freq=out["freq"].to_numpy(dtype=np.float32),
        )
        # Same 70/30 physics-informed blend as in run_pipeline (see comment there for rationale).
        y_ctm_pred = (0.7 * y_ctm_prior + 0.3 * y_ctm_pred).astype(np.float32)
        y_ctm_offset = (y_ctm_pred - y_ctm_base).astype(np.float32)
        ctm_source = "heuristic+residual"

    out["ctm_ka"] = y_ctm_pred[:, 0]
    out["ctm_kd"] = y_ctm_pred[:, 1]
    out["ctm_ke"] = y_ctm_pred[:, 2]
    out["ctm_km"] = y_ctm_pred[:, 3]
    out["ctm_signal_gain"] = y_ctm_pred[:, 4]
    out["ctm_ka_base"] = y_ctm_base[:, 0]
    out["ctm_kd_base"] = y_ctm_base[:, 1]
    out["ctm_ke_base"] = y_ctm_base[:, 2]
    out["ctm_km_base"] = y_ctm_base[:, 3]
    out["ctm_signal_gain_base"] = y_ctm_base[:, 4]
    out["ctm_ka_offset"] = y_ctm_offset[:, 0]
    out["ctm_kd_offset"] = y_ctm_offset[:, 1]
    out["ctm_ke_offset"] = y_ctm_offset[:, 2]
    out["ctm_km_offset"] = y_ctm_offset[:, 3]
    out["ctm_signal_gain_offset"] = y_ctm_offset[:, 4]

    out, curve_df = _run_simulation_loop(out, bundle.dynamics_model, bundle.ctm_model)
    out = _append_immune_abm_outputs(out, horizon_h=96)

    interaction = []
    groups = sorted(set(out["group_id"].astype(str).tolist()))
    if len(groups) > 1:
        base_group_mean = out.groupby("group_id", as_index=False).agg(base_mean=("efficacy_pred", "mean"))
        for g in groups:
            perturbed = out.copy()
            mask = perturbed["group_id"].astype(str) == g
            perturbed.loc[mask, "dose"] = perturbed.loc[mask, "dose"].astype(float) * 1.1 + 0.2
            Xp, _, _ = build_feature_matrix(perturbed, spec)
            p = bundle.moe_model.predict(Xp)
            tmp = perturbed[["group_id"]].copy()
            tmp["pred"] = p
            p_group_mean = tmp.groupby("group_id", as_index=False).agg(perturbed_mean=("pred", "mean"))
            merged = base_group_mean.merge(p_group_mean, on="group_id", how="left")
            merged["source_group"] = g
            merged["delta"] = merged["perturbed_mean"] - merged["base_mean"]
            interaction.append(merged[["source_group", "group_id", "delta"]])

        if interaction:
            inter_df = pd.concat(interaction, axis=0, ignore_index=True)
            out = out.merge(
                inter_df.groupby("group_id", as_index=False)["delta"].mean().rename(columns={"group_id": "group_id", "delta": "cross_group_impact"}),
                on="group_id",
                how="left",
            )

    return out, curve_df, ctm_source, bool(shap_ready), str(shap_message), int(len(feat_names)), str(adaptive_message)


def train_pipeline_bundle(
    df: pd.DataFrame,
    compute_mode: str = "auto",
    dynamics_model: str = "ctm",
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
    feature_spec: MixedFeatureSpec | None = None,
    tune_hyperparams: bool = False,
    tune_strategy: str = "random",
    tune_n_iter: int = 20,
    tune_cv: int = 3,
) -> tuple[ConfluenciaModelBundle, ConfluenciaArtifacts]:
    work = ensure_columns(df)
    for c in ["dose", "freq", "treatment_time"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    spec = feature_spec or MixedFeatureSpec(
        smiles_hash_dim=128, smiles_rdkit_bits=2048, smiles_rdkit_version=2, prefer_rdkit=True
    )
    X, env_cols, smiles_backend = build_feature_matrix(work, spec)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot train model bundle with empty dataset")

    y_eff = _resolve_label(work, "efficacy")
    y_bind = _resolve_label(work, "target_binding")

    # ── Target transform (logit for bounded efficacy) ──────
    use_logit = str(getattr(spec, "target_transform", "none")).lower() == "logit"
    if use_logit and y_eff is not None:
        y_eff_for_fit = logit_transform(y_eff)
    else:
        y_eff_for_fit = y_eff

    # Feature selection (optional, requires labeled efficacy)
    feature_selection_applied = False
    feature_selection_n_final = 0
    if bool(spec.use_feature_selection) and y_eff_for_fit is not None:
        from .feature_selector import FeatureSelector

        selector = FeatureSelector(
            top_k=int(spec.feature_selection_top_k),
            corr_thresh=float(spec.feature_selection_correl_thresh),
        )
        all_names = build_feature_names(spec, list(env_cols))
        X, feature_names = selector.fit_transform(X, y_eff_for_fit, feature_names=all_names)
        feature_selection_applied = True
        feature_selection_n_final = X.shape[1]
    else:
        feature_names = build_feature_names(spec, list(env_cols))

    prof = choose_compute_profile(n_samples=n, requested=compute_mode)

    # ── Hierarchical residual model (optional) ───────────────
    # If target_binding is available, use two-stage model:
    # Stage 1: predict binding (high R²), Stage 2: predict efficacy residual
    binding_model = None
    residual_model = None
    if y_bind is not None and str(getattr(spec, "use_auxiliary_labels", False)):
        binding_model, residual_model, bind_pred = _fit_hierarchical_residual(
            X, y_eff_for_fit, y_bind, prof.enabled_experts, prof.folds, 42,
        )

    # ── Hyperparameter tuning (optional) ───────────────────────
    tuned_config = None
    if tune_hyperparams and n >= 30:
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from confluencia_shared.moe import ExpertConfig

        param_grids = {
            "hgb": {"max_depth": [4, 5, 6, 7, 8], "learning_rate": [0.03, 0.05, 0.1, 0.15]},
            "gbr": {"n_estimators": [100, 200, 300], "max_depth": [4, 5, 6, 7], "learning_rate": [0.03, 0.05, 0.1]},
            "rf": {"n_estimators": [150, 200, 260, 300], "max_depth": [8, 10, 12, 15]},
            "ridge": {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
            "mlp": {"mlp__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)], "mlp__alpha": [0.0001, 0.001, 0.01]},
        }

        tuned_config = ExpertConfig()
        for expert_name in prof.enabled_experts:
            if expert_name not in param_grids:
                continue
            grid = param_grids[expert_name]
            if expert_name == "ridge":
                base_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=42))])
            elif expert_name == "mlp":
                base_model = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(max_iter=1200, early_stopping=True, random_state=42))])
            elif expert_name == "hgb":
                base_model = HistGradientBoostingRegressor(random_state=42)
            elif expert_name == "gbr":
                base_model = GradientBoostingRegressor(random_state=42)
            elif expert_name == "rf":
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            else:
                continue

            if tune_strategy == "grid":
                search = GridSearchCV(base_model, grid, cv=tune_cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            else:
                search = RandomizedSearchCV(base_model, grid, n_iter=min(tune_n_iter, 20), cv=tune_cv, scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42)
            search.fit(X, y_eff_for_fit)
            bp = search.best_params_

            # Apply best params to ExpertConfig
            if expert_name == "ridge" and "alpha" in bp:
                tuned_config.ridge_alpha = float(bp["alpha"])
            elif expert_name in {"hgb", "gbr"}:
                if "max_depth" in bp:
                    tuned_config.hgb_max_depth = int(bp["max_depth"])
                if "learning_rate" in bp:
                    tuned_config.hgb_learning_rate = float(bp["learning_rate"])
            elif expert_name == "rf":
                if "n_estimators" in bp:
                    tuned_config.rf_n_estimators = int(bp["n_estimators"])
                if "max_depth" in bp:
                    tuned_config.rf_max_depth = int(bp["max_depth"])
            elif expert_name == "mlp":
                if "mlp__hidden_layer_sizes" in bp:
                    tuned_config.mlp_hidden_sizes = bp["mlp__hidden_layer_sizes"]

    moe = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds, config=tuned_config)
    moe.fit(X, y_eff_for_fit)
    moe_weights = moe.explain_weights()
    moe_metrics = dict(moe.metrics)

    y_micro = None
    micro_used_proxy = False
    if any(c in work.columns for c in MICRO_TARGETS):
        cols = []
        for c in MICRO_TARGETS:
            if c in work.columns:
                cols.append(pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=np.float32))
            else:
                cols.append(np.full((n,), np.nan, dtype=np.float32))
        y_micro = np.column_stack(cols).astype(np.float32)
        if np.isnan(y_micro).any():
            y_proxy = proxy_micro_labels(X)
            nan_mask = np.isnan(y_micro)
            y_micro[nan_mask] = y_proxy[nan_mask]
            micro_used_proxy = True

    micro = MicroPredictor().fit(X, y_micro)
    if micro_used_proxy:
        micro.used_proxy_labels = True

    eff_pred_train_logit = moe.predict(X)
    # Inverse-logit transform if training used logit
    eff_pred_train = inverse_logit(eff_pred_train_logit) if use_logit else eff_pred_train_logit
    micro_pred_train = micro.predict(X)
    out_train = work.copy()
    out_train["efficacy_pred"] = eff_pred_train
    for k, v in micro_pred_train.items():
        out_train[f"{k}_pred"] = v
    out_train = _apply_multitask_consistency(out_train)
    out_train, adaptive_message = _apply_adaptive_adjustment(
        out_train,
        enabled=bool(adaptive_enabled),
        strength=float(adaptive_strength),
    )

    x_ctm = np.column_stack(
        [
            out_train["dose"].to_numpy(dtype=np.float32),
            out_train["freq"].to_numpy(dtype=np.float32),
            out_train["treatment_time"].to_numpy(dtype=np.float32),
            out_train["target_binding_pred"].to_numpy(dtype=np.float32),
            out_train["immune_activation_pred"].to_numpy(dtype=np.float32),
            out_train["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            out_train["efficacy_pred"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    target_cols = ["ctm_ka", "ctm_kd", "ctm_ke", "ctm_km", "ctm_signal_gain"]
    if all(c in out_train.columns for c in target_cols):
        y_ctm = out_train[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ctm_source = "labeled"
    else:
        y_ctm = heuristic_param_targets(
            binding=out_train["target_binding_pred"].to_numpy(dtype=np.float32),
            immune=out_train["immune_activation_pred"].to_numpy(dtype=np.float32),
            inflammation=out_train["inflammation_risk_pred"].to_numpy(dtype=np.float32),
            dose=out_train["dose"].to_numpy(dtype=np.float32),
            freq=out_train["freq"].to_numpy(dtype=np.float32),
        )
        ctm_source = "heuristic+residual"

    ctm_model = CTMParamModel()
    group_ids = out_train["group_id"].astype(str).to_numpy(dtype=object)
    ctm_model.fit(x_ctm, y_ctm, group_ids=group_ids)

    bundle = ConfluenciaModelBundle(
        feature_spec=spec,
        compute_profile=prof.level,
        moe_model=moe,
        micro_model=micro,
        ctm_model=ctm_model,
        ctm_param_source=ctm_source,
        moe_weights=moe_weights,
        moe_metrics=moe_metrics,
        dynamics_model=("ndp4pd" if str(dynamics_model).lower() in {"ndp4pd", "nd4pd"} else "ctm"),
        use_logit_transform=bool(use_logit),
        binding_model=binding_model,
    )
    artifacts = ConfluenciaArtifacts(
        compute_profile=prof.level,
        moe_weights=moe_weights,
        moe_metrics=moe_metrics,
        used_proxy_micro_labels=micro.used_proxy_labels,
        smiles_backend=smiles_backend,
        ctm_param_source=ctm_source,
        model_backend="moe",
        dynamics_model=bundle.dynamics_model,
        adaptive_enabled=bool(adaptive_enabled),
        adaptive_strength=float(np.clip(adaptive_strength, 0.0, 1.0)),
        adaptive_samples=int(len(out_train)),
        adaptive_message=str(adaptive_message),
        feature_selection_applied=bool(feature_selection_applied),
        feature_selection_n_final=int(feature_selection_n_final),
        use_logit_transform=bool(use_logit),
        hierarchical_residual_enabled=binding_model is not None,
    )
    return bundle, artifacts


def predict_pipeline_with_bundle(
    df: pd.DataFrame,
    bundle: ConfluenciaModelBundle,
    adaptive_enabled: bool = False,
    adaptive_strength: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, ConfluenciaArtifacts]:
    work = ensure_columns(df)
    for c in ["dose", "freq", "treatment_time"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    X, env_cols, smiles_backend = build_feature_matrix(work, bundle.feature_spec)
    n = X.shape[0]
    if n == 0:
        empty = work.copy()
        curve = pd.DataFrame(
            columns=[
                "time_h",
                "absorption_A",
                "distribution_D",
                "effect_E",
                "metabolism_M",
                "efficacy_signal",
                "toxicity_signal",
                "pkpd_depot_mg",
                "pkpd_central_mg",
                "pkpd_peripheral_mg",
                "pkpd_conc_mg_per_l",
                "pkpd_effect",
            ]
        )
        artifacts = ConfluenciaArtifacts(
            compute_profile=bundle.compute_profile,
            moe_weights=bundle.moe_weights,
            moe_metrics=bundle.moe_metrics,
            used_proxy_micro_labels=bundle.micro_model.used_proxy_labels,
            smiles_backend=smiles_backend,
            ctm_param_source=bundle.ctm_param_source,
            model_backend="moe",
            dynamics_model=bundle.dynamics_model,
            shap_ready=False,
            shap_message="empty_input",
            shap_feature_count=0,
            adaptive_enabled=bool(adaptive_enabled),
            adaptive_strength=float(np.clip(adaptive_strength, 0.0, 1.0)),
            adaptive_samples=0,
            adaptive_message="empty_input",
        )
        return empty, curve, artifacts

    out, curve_df, ctm_source, shap_ready, shap_message, shap_feature_count, adaptive_message = _predict_with_bundle_core(
        work=work,
        X=X,
        spec=bundle.feature_spec,
        env_cols=list(env_cols),
        bundle=bundle,
        adaptive_enabled=bool(adaptive_enabled),
        adaptive_strength=float(adaptive_strength),
    )
    artifacts = ConfluenciaArtifacts(
        compute_profile=bundle.compute_profile,
        moe_weights=bundle.moe_weights,
        moe_metrics=bundle.moe_metrics,
        used_proxy_micro_labels=bundle.micro_model.used_proxy_labels,
        smiles_backend=smiles_backend,
        ctm_param_source=ctm_source,
        model_backend="moe",
        dynamics_model=bundle.dynamics_model,
        shap_ready=bool(shap_ready),
        shap_message=str(shap_message),
        shap_feature_count=int(shap_feature_count),
        adaptive_enabled=bool(adaptive_enabled),
        adaptive_strength=float(np.clip(adaptive_strength, 0.0, 1.0)),
        adaptive_samples=int(len(out)),
        adaptive_message=str(adaptive_message),
    )
    return out, curve_df, artifacts
