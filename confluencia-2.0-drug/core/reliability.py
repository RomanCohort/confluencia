from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    CircRNAFeatureSpec,
    MixedFeatureSpec,
    build_cirrna_feature_matrix,
    build_combined_feature_matrix,
    build_feature_matrix,
    ensure_columns,
    ensure_cirrna_columns,
)
from .moe import MOERegressor, choose_compute_profile


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if y_true.size < 2:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _mean_std_ci(values: list[float], n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Compute mean, std, and 95% CI using bootstrap or t-distribution.

    For n >= 10: bootstrap percentile method
    For n < 10: t-distribution (more reliable for small samples)
    """
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    n = arr.size

    if n < 10:
        # Use t-distribution for small samples
        from scipy import stats
        t_val = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
        ci95 = float(t_val * std / max(np.sqrt(float(n)), 1e-8)) if n > 1 else 0.0
    else:
        # Bootstrap percentile CI
        rng = np.random.default_rng(seed)
        boot_means = rng.choice(arr, size=(n_bootstrap, n), replace=True).mean(axis=1)
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
        ci95 = float((hi - lo) / 2.0)  # half-width

    return mean, std, ci95


def _stratified_kfold_for_regression(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified CV splits for regression by binning target into quantiles.

    Returns list of (train_idx, test_idx) tuples.
    """
    n_bins = min(n_splits * 2, 10)  # At most 10 bins
    if len(y) < n_bins * 3:
        # Too few samples for stratification, fall back to KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(len(y))))

    # Create quantile-based bins
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    # Use StratifiedKFold on bins
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(y)), bins))


def _interval_calibration(y_true: np.ndarray, y_pred: np.ndarray, val_abs_err: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for q in [0.5, 0.8, 0.9]:
        half_width = float(np.quantile(val_abs_err, q)) if val_abs_err.size > 0 else 0.0
        lo = y_pred - half_width
        hi = y_pred + half_width
        covered = np.logical_and(y_true >= lo, y_true <= hi)
        coverage = float(np.mean(covered.astype(np.float32))) if covered.size > 0 else 0.0
        rows.append(
            {
                "target_coverage": float(q),
                "empirical_coverage": coverage,
                "half_width": half_width,
                "coverage_gap": float(abs(coverage - q)),
            }
        )
    return pd.DataFrame(rows)


def _leakage_audit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    key_cols = [
        c
        for c in ["smiles", "epitope_seq", "dose", "freq", "treatment_time", "group_id"]
        if c in train_df.columns and c in test_df.columns
    ]
    if not key_cols:
        return {"overlap_count": 0.0, "overlap_ratio": 0.0}

    tr_sig = train_df[key_cols].astype(str).apply(lambda r: "|".join(r.tolist()), axis="columns")
    te_sig = test_df[key_cols].astype(str).apply(lambda r: "|".join(r.tolist()), axis="columns")
    overlap = int(te_sig.isin(set(tr_sig.tolist())).sum())
    ratio = float(overlap) / float(max(len(test_df), 1))
    return {"overlap_count": float(overlap), "overlap_ratio": ratio}


def _build_sklearn_regressor(name: str, seed: int):
    if name == "linear":
        return Pipeline(steps=[("scaler", StandardScaler()), ("linear", LinearRegression())])
    if name == "ridge":
        return Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))])
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, max_depth=6)
    if name == "rf":
        return RandomForestRegressor(n_estimators=260, max_depth=12, random_state=seed, n_jobs=1)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1000, early_stopping=True, random_state=seed),
                ),
            ]
        )
    if name == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unsupported regressor: {name}")


def _predict_backend(backend: str, compute_mode: str, seed: int, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    b = str(backend).strip().lower()
    if b == "moe":
        prof = choose_compute_profile(n_samples=int(len(y_tr)), requested=compute_mode)
        m = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds, random_state=seed)
        m.fit(x_tr, y_tr)
        return m.predict(x_te).astype(np.float32)

    if b in {"hgb", "gbr", "rf", "ridge", "mlp"}:
        m = _build_sklearn_regressor(b, seed=seed)
        m.fit(x_tr, y_tr)
        return np.asarray(m.predict(x_te), dtype=np.float32).reshape(-1)

    raise ValueError(f"可信评估暂不支持当前后端: {backend}")


def credible_eval_drug(
    df: pd.DataFrame,
    backend: str,
    compute_mode: str,
    seed: int,
    test_ratio: float,
    val_ratio: float,
    cv_folds: int,
    top_n_failures: int,
    external_df: pd.DataFrame | None,
    feature_spec: "MixedFeatureSpec | None" = None,
) -> dict[str, Any]:
    work = ensure_columns(df)
    if "efficacy" not in work.columns:
        return {"enabled": False, "reason": "missing_label"}

    y_series = pd.to_numeric(work["efficacy"], errors="coerce")
    if isinstance(y_series, pd.Series):
        y = y_series.fillna(0.0).to_numpy(dtype=np.float32)
    else:
        y = np.full((len(work),), float(y_series), dtype=np.float32)

    _spec = feature_spec or MixedFeatureSpec(smiles_hash_dim=128, smiles_rdkit_bits=2048, smiles_rdkit_version=2, prefer_rdkit=True)
    x, _, _ = build_feature_matrix(work, _spec)
    n = int(len(work))
    if n < max(30, cv_folds * 4):
        return {"enabled": False, "reason": "too_few_samples"}

    idx = np.arange(n)
    trva_idx, te_idx = train_test_split(idx, test_size=float(test_ratio), random_state=seed, shuffle=True)
    val_ratio_in_trva = float(val_ratio) / max(1.0 - float(test_ratio), 1e-6)
    tr_idx, va_idx = train_test_split(trva_idx, test_size=val_ratio_in_trva, random_state=seed, shuffle=True)

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    backend_used = str(backend)
    backend_supported = backend_used in {"moe", "hgb", "gbr", "rf", "ridge", "mlp"}
    if not backend_supported:
        backend_used = "hgb"

    pred_va = _predict_backend(backend_used, compute_mode, seed, x_tr, y_tr, x_va)
    pred_te = _predict_backend(backend_used, compute_mode, seed, x_tr, y_tr, x_te)
    val_metrics = _safe_metrics(y_va, pred_va)
    test_metrics = _safe_metrics(y_te, pred_te)
    val_abs_err = np.abs(pred_va - y_va).astype(np.float32)
    interval_df = _interval_calibration(y_true=y_te, y_pred=pred_te, val_abs_err=val_abs_err)

    cv_splits = _stratified_kfold_for_regression(y[trva_idx], n_splits=max(2, int(cv_folds)), seed=seed)
    cv_mae: list[float] = []
    cv_rmse: list[float] = []
    cv_r2: list[float] = []
    for cv_tr, cv_te in cv_splits:
        x_cv_tr = x[trva_idx][cv_tr]
        y_cv_tr = y[trva_idx][cv_tr]
        x_cv_te = x[trva_idx][cv_te]
        y_cv_te = y[trva_idx][cv_te]
        p = _predict_backend(backend_used, compute_mode, seed, x_cv_tr, y_cv_tr, x_cv_te)
        m = _safe_metrics(y_cv_te, p)
        cv_mae.append(float(m["mae"]))
        cv_rmse.append(float(m["rmse"]))
        cv_r2.append(float(m["r2"]))

    mae_mean, mae_std, mae_ci = _mean_std_ci(cv_mae)
    rmse_mean, rmse_std, rmse_ci = _mean_std_ci(cv_rmse)
    r2_mean, r2_std, r2_ci = _mean_std_ci(cv_r2)

    baseline_rows: list[dict[str, float | str]] = []
    for b in ["linear", "rf", "hgb"]:
        bm = _build_sklearn_regressor(b, seed=seed)
        bm.fit(x_tr, y_tr)
        bp = np.asarray(bm.predict(x_te), dtype=np.float32).reshape(-1)
        mm = _safe_metrics(y_te, bp)
        baseline_rows.append({"model": b, "mae": mm["mae"], "rmse": mm["rmse"], "r2": mm["r2"]})
    baseline_df = pd.DataFrame(baseline_rows).sort_values(["rmse", "mae"], ascending=[True, True])

    baseline_best_rmse = float(baseline_df["rmse"].min()) if not baseline_df.empty else float("inf")
    pass_gate = bool(float(test_metrics["rmse"]) < baseline_best_rmse)

    fail_df = work.iloc[te_idx].copy()
    fail_df["y_true"] = y_te
    fail_df["y_pred"] = pred_te
    fail_df["abs_error"] = np.abs(pred_te - y_te)
    fail_df = fail_df.sort_values("abs_error", ascending=False).head(max(1, int(top_n_failures)))
    leakage = _leakage_audit(work.iloc[tr_idx], work.iloc[te_idx])

    external_metrics = None
    if external_df is not None and (not external_df.empty) and ("efficacy" in external_df.columns):
        ext_work = ensure_columns(external_df)
        x_ext, _, _ = build_feature_matrix(
            ext_work,
            _spec,
        )
        y_ext_series = pd.to_numeric(ext_work["efficacy"], errors="coerce")
        if isinstance(y_ext_series, pd.Series):
            y_ext = y_ext_series.fillna(0.0).to_numpy(dtype=np.float32)
        else:
            y_ext = np.full((len(ext_work),), float(y_ext_series), dtype=np.float32)
        pred_ext = _predict_backend(backend_used, compute_mode, seed, x[trva_idx], y[trva_idx], x_ext)
        external_metrics = _safe_metrics(y_ext, pred_ext)

    return {
        "enabled": True,
        "backend_supported": backend_supported,
        "backend_used": backend_used,
        "split_sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cv_summary": {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mae_ci95": mae_ci,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_ci95": rmse_ci,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "r2_ci95": r2_ci,
        },
        "baseline_df": baseline_df,
        "pass_gate": pass_gate,
        "interval_calibration_df": interval_df,
        "leakage_audit": leakage,
        "failure_df": fail_df,
        "external_metrics": external_metrics,
    }


# ===================================================================
# circRNA-specific credibility evaluation
# ===================================================================

def credible_eval_cirrna(
    df: pd.DataFrame,
    backend: str = "hgb",
    compute_mode: str = "low",
    seed: int = 42,
    test_ratio: float = 0.20,
    cv_folds: int = 5,
    target_col: str = "efficacy",
) -> dict[str, Any]:
    """Evaluate circRNA model credibility with circRNA-specific validation.

    Extends standard credible_eval with:
      - circRNA feature importance decomposition (IRES vs modification vs backbone)
      - Innate immune score cross-validation
      - Feature correlation analysis for circRNA features
      - Modified vs unmodified subgroup analysis

    Returns a dict with evaluation results.
    """
    from .innate_immune import batch_assess_innate_immune

    work = ensure_columns(df)
    work = ensure_cirrna_columns(work)

    if target_col not in work.columns:
        return {"enabled": False, "reason": "missing_label"}

    y_series = pd.to_numeric(work[target_col], errors="coerce")
    y = y_series.fillna(0.0).to_numpy(dtype=np.float32)
    n = len(work)

    # Build combined features (small-molecule + circRNA)
    x, all_names, backend_used = build_combined_feature_matrix(work)

    if n < max(30, cv_folds * 4):
        return {"enabled": False, "reason": "too_few_samples"}

    # Standard train/test split
    idx = np.arange(n)
    trva_idx, te_idx = train_test_split(idx, test_size=float(test_ratio), random_state=seed, shuffle=True)
    x_tr, y_tr = x[trva_idx], y[trva_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    # Fit model and predict
    pred_te = _predict_backend(backend, compute_mode, seed, x_tr, y_tr, x_te)
    test_metrics = _safe_metrics(y_te, pred_te)

    # --- circRNA Feature Importance Decomposition ---
    crna_feature_groups = {
        "sequence": [n for n in all_names if n.startswith("crna_seq_") or n.startswith("crna_frac_") or n.startswith("crna_palindrome") or n.startswith("crna_bsj_") or n == "crna_gc_content"],
        "structure": [n for n in all_names if n.startswith("crna_mfe") or n.startswith("crna_stem") or n.startswith("crna_struct_")],
        "functional": [n for n in all_names if n.startswith("crna_ires") or n.startswith("crna_orf") or n.startswith("crna_kozak") or n.startswith("crna_utr_")],
        "modification": [n for n in all_names if n.startswith("crna_mod_")],
        "delivery": [n for n in all_names if n.startswith("crna_bio") or n.startswith("crna_liver") or n.startswith("crna_spleen") or n.startswith("crna_muscle") or n.startswith("crna_endo") or n.startswith("crna_half_")],
    }

    # Use permutation importance on each feature group
    group_importance: dict[str, float] = {}
    baseline_rmse = test_metrics["rmse"]
    for group_name, group_cols in crna_feature_groups.items():
        group_indices = [all_names.index(c) for c in group_cols if c in all_names]
        if not group_indices:
            group_importance[group_name] = 0.0
            continue
        x_te_perm = x_te.copy()
        rng = np.random.default_rng(seed)
        for gi in group_indices:
            rng.shuffle(x_te_perm[:, gi])
        pred_perm = _predict_backend(backend, compute_mode, seed, x_tr, y_tr, x_te_perm)
        perm_rmse = float(np.sqrt(np.mean((y_te - pred_perm) ** 2)))
        group_importance[group_name] = float(perm_rmse - baseline_rmse)

    total_importance = sum(max(v, 0.0) for v in group_importance.values())
    if total_importance > 0:
        group_importance_pct = {k: max(v, 0.0) / total_importance for k, v in group_importance.items()}
    else:
        group_importance_pct = {k: 0.2 for k in group_importance}

    # --- Innate Immune Score Cross-Validation ---
    innate_results = batch_assess_innate_immune(work)
    innate_df = pd.DataFrame(innate_results)
    innate_cv_scores: list[float] = []
    if not innate_df.empty and "innate_immune_score" in innate_df.columns:
        innate_scores_arr = innate_df["innate_immune_score"].to_numpy(dtype=np.float32)
        cv = KFold(n_splits=max(2, int(cv_folds)), shuffle=True, random_state=seed)
        for cv_tr, cv_te in cv.split(np.arange(n)):
            tr_mean = float(np.mean(innate_scores_arr[cv_tr]))
            te_val = float(np.mean(innate_scores_arr[cv_te]))
            innate_cv_scores.append(abs(te_val - tr_mean))

    innate_cv_mean, innate_cv_std, innate_cv_ci = _mean_std_ci(innate_cv_scores)

    # --- Modification Subgroup Analysis ---
    has_cirrna = (work["circrna_seq"].notna() & (work["circrna_seq"].str.len() > 0)).any()
    mod_subgroup = {}
    if has_cirrna:
        for mod_val in work["modification"].unique():
            mod_mask = work["modification"] == mod_val
            mod_idx = np.where(mod_mask.to_numpy())[0]
            if len(mod_idx) >= 5:
                mod_idx_in_test = [i for i in mod_idx if i in te_idx]
                if len(mod_idx_in_test) >= 3:
                    y_mod = y_te[[te_idx.tolist().index(i) for i in mod_idx_in_test]]
                    p_mod = pred_te[[te_idx.tolist().index(i) for i in mod_idx_in_test]]
                    mod_subgroup[str(mod_val)] = _safe_metrics(y_mod, p_mod)

    # --- Cross-validation on combined features ---
    cv_rmse_list: list[float] = []
    cv_splits_circ = _stratified_kfold_for_regression(y[trva_idx], n_splits=max(2, int(cv_folds)), seed=seed)
    for cv_tr, cv_te in cv_splits_circ:
        p_cv = _predict_backend(backend, compute_mode, seed, x[trva_idx][cv_tr], y[trva_idx][cv_tr], x[trva_idx][cv_te])
        cv_rmse_list.append(float(np.sqrt(np.mean((y[trva_idx][cv_te] - p_cv) ** 2))))

    cv_rmse_mean, cv_rmse_std, cv_rmse_ci = _mean_std_ci(cv_rmse_list)

    return {
        "enabled": True,
        "backend": backend,
        "n_samples": n,
        "test_metrics": test_metrics,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "cv_rmse_ci95": cv_rmse_ci,
        "feature_importance_by_group": group_importance,
        "feature_importance_pct": group_importance_pct,
        "innate_immune_cv_mean": innate_cv_mean,
        "innate_immune_cv_std": innate_cv_std,
        "innate_immune_cv_ci95": innate_cv_ci,
        "modification_subgroup": mod_subgroup,
        "has_cirrna_data": has_cirrna,
    }
