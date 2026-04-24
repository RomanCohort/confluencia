from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import HYDROPHOBIC, POLAR, ACIDIC, BASIC, FeatureSpec, build_feature_matrix, ensure_columns
from .moe import MOERegressor, choose_compute_profile
from .torch_mamba import TorchMambaConfig, predict_torch_mamba, torch_available, train_torch_mamba

AROMATIC = set("WFY")
CHARGED = set("KRHDE")


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else 0.0
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


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail_k = int(min(k, n - k))
    p = 0.0
    for i in range(0, tail_k + 1):
        p += math.comb(n, i)
    p = p / float(2**n)
    return float(min(1.0, max(0.0, 2.0 * p)))


def _paired_significance(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> dict[str, float | int | str]:
    err_a = np.abs(np.asarray(pred_a, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1))
    err_b = np.abs(np.asarray(pred_b, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1))
    diff = err_a - err_b

    nz = np.abs(diff) > 1e-12
    d_nz = diff[nz]
    if d_nz.size == 0:
        return {
            "model_a": str(name_a),
            "model_b": str(name_b),
            "n_pairs": int(diff.size),
            "n_nonzero": 0,
            "p_value": 1.0,
            "effect_size_dz": 0.0,
            "mae_a": float(err_a.mean()) if err_a.size > 0 else 0.0,
            "mae_b": float(err_b.mean()) if err_b.size > 0 else 0.0,
            "delta_mae_b_minus_a": float(err_b.mean() - err_a.mean()) if err_a.size > 0 else 0.0,
        }

    wins_b = int((d_nz > 0).sum())
    n_nz = int(d_nz.size)
    p_val = _binom_two_sided_pvalue(wins_b, n_nz)

    mean_d = float(d_nz.mean())
    std_d = float(d_nz.std(ddof=1)) if d_nz.size > 1 else 0.0
    dz = float(mean_d / std_d) if std_d > 1e-12 else 0.0

    return {
        "model_a": str(name_a),
        "model_b": str(name_b),
        "n_pairs": int(diff.size),
        "n_nonzero": n_nz,
        "wins_model_b": wins_b,
        "p_value": float(p_val),
        "effect_size_dz": dz,
        "mae_a": float(err_a.mean()) if err_a.size > 0 else 0.0,
        "mae_b": float(err_b.mean()) if err_b.size > 0 else 0.0,
        "delta_mae_b_minus_a": float(err_b.mean() - err_a.mean()) if err_a.size > 0 else 0.0,
    }


def _ood_subset_eval(df_train: pd.DataFrame, df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    te = df_test.copy()
    tr = df_train.copy()
    seq_len_tr = tr["epitope_seq"].astype(str).str.len().to_numpy(dtype=np.float32)
    seq_len_te = te["epitope_seq"].astype(str).str.len().to_numpy(dtype=np.float32)

    signals: list[np.ndarray] = []
    rows: list[dict[str, float | str]] = []

    def _append_signal(name: str, train_vals: np.ndarray, test_vals: np.ndarray) -> None:
        lo = float(np.quantile(train_vals, 0.05)) if train_vals.size > 0 else 0.0
        hi = float(np.quantile(train_vals, 0.95)) if train_vals.size > 0 else 0.0
        sig = np.logical_or(test_vals < lo, test_vals > hi)
        signals.append(sig)
        rows.append({"feature": name, "train_q05": lo, "train_q95": hi, "ood_ratio": float(np.mean(sig.astype(np.float32)))})

    _append_signal("seq_len", seq_len_tr, seq_len_te)
    for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]:
        tr_vals = pd.to_numeric(tr[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        te_vals = pd.to_numeric(te[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        _append_signal(c, tr_vals, te_vals)

    if signals:
        ood_mask = np.logical_or.reduce(signals)
    else:
        ood_mask = np.zeros((len(te),), dtype=bool)
    id_mask = np.logical_not(ood_mask)

    ood_metrics = _safe_metrics(y_true[ood_mask], y_pred[ood_mask]) if int(ood_mask.sum()) > 0 else {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    id_metrics = _safe_metrics(y_true[id_mask], y_pred[id_mask]) if int(id_mask.sum()) > 0 else {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    return {
        "ood_count": int(ood_mask.sum()),
        "id_count": int(id_mask.sum()),
        "ood_ratio": float(np.mean(ood_mask.astype(np.float32))) if ood_mask.size > 0 else 0.0,
        "ood_metrics": ood_metrics,
        "id_metrics": id_metrics,
        "feature_threshold_df": pd.DataFrame(rows),
    }


def _aa_composition_stratification(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Stratify test samples by amino acid composition ratios.

    Bins samples by hydrophobic/charged/aromatic fraction into
    low/mid/high groups and reports per-bin metrics.
    """
    seqs = df_test["epitope_seq"].astype(str)
    rows: list[dict] = []

    for prop_name, aa_set in [("hydrophobic", HYDROPHOBIC), ("charged", CHARGED), ("aromatic", AROMATIC)]:
        fracs = seqs.apply(lambda s: sum(1 for ch in s if ch in aa_set) / max(len(s), 1)).to_numpy(dtype=np.float32)
        q33 = float(np.quantile(fracs, 0.33)) if fracs.size > 2 else 0.0
        q66 = float(np.quantile(fracs, 0.66)) if fracs.size > 2 else 1.0
        bins = np.full(fracs.shape, "mid", dtype=object)
        bins[fracs <= q33] = "low"
        bins[fracs >= q66] = "high"

        for label in ["low", "mid", "high"]:
            mask = bins == label
            n = int(mask.sum())
            if n == 0:
                continue
            m = _safe_metrics(y_true[mask], y_pred[mask])
            rows.append({
                "property": prop_name,
                "bin": label,
                "n": n,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["property", "bin", "n", "mae", "rmse", "r2"])


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
        for c in ["epitope_seq", "dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
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
                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1200, early_stopping=True, random_state=seed),
                ),
            ]
        )
    if name == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unsupported regressor: {name}")


def _predict_epitope_backend(
    backend: str,
    compute_mode: str,
    seed: int,
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_eval: pd.DataFrame,
    torch_cfg: TorchMambaConfig,
    feature_spec: "FeatureSpec | None" = None,
) -> tuple[np.ndarray, bool]:
    b = str(backend).strip().lower()
    _feat_spec = feature_spec or FeatureSpec()
    if b == "torch-mamba":
        env_cols = [c for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"] if c in df_train.columns]
        bundle = train_torch_mamba(df_train, y_train, env_cols=env_cols, cfg=torch_cfg, prefer_real_mamba=True)
        pred = predict_torch_mamba(bundle, df_eval)
        return np.asarray(pred, dtype=np.float32).reshape(-1), bool(bundle.used_real_mamba)

    x_train, _, _ = build_feature_matrix(df_train, _feat_spec)
    x_eval, _, _ = build_feature_matrix(df_eval, _feat_spec)
    if b == "sklearn-moe":
        prof = choose_compute_profile(n_samples=int(len(y_train)), requested=compute_mode)
        m = MOERegressor(expert_names=prof.enabled_experts, folds=prof.folds, random_state=seed)
        m.fit(x_train, y_train)
        return m.predict(x_eval).astype(np.float32), False
    if b in {"hgb", "gbr", "rf", "ridge", "mlp"}:
        m = _build_sklearn_regressor(b, seed)
        m.fit(x_train, y_train)
        return np.asarray(m.predict(x_eval), dtype=np.float32).reshape(-1), False
    raise ValueError(f"可信评估暂不支持当前后端: {backend}")


def credible_eval_epitope(
    df: pd.DataFrame,
    backend: str,
    compute_mode: str,
    seed: int,
    test_ratio: float,
    val_ratio: float,
    cv_folds: int,
    top_n_failures: int,
    torch_cfg: TorchMambaConfig,
    external_df: pd.DataFrame | None,
    feature_spec: "FeatureSpec | None" = None,
) -> dict[str, Any]:
    work = ensure_columns(df)
    if "efficacy" not in work.columns:
        return {"enabled": False, "reason": "missing_label"}
    if len(work) < max(30, int(cv_folds) * 4):
        return {"enabled": False, "reason": "too_few_samples"}

    y_series = pd.to_numeric(work["efficacy"], errors="coerce")
    if isinstance(y_series, pd.Series):
        y = y_series.fillna(0.0).to_numpy(dtype=np.float32)
    else:
        y = np.full((len(work),), float(y_series), dtype=np.float32)

    idx = np.arange(len(work))
    trva_idx, te_idx = train_test_split(idx, test_size=float(test_ratio), random_state=seed, shuffle=True)
    val_ratio_in_trva = float(val_ratio) / max(1.0 - float(test_ratio), 1e-6)
    tr_idx, va_idx = train_test_split(trva_idx, test_size=val_ratio_in_trva, random_state=seed, shuffle=True)

    df_tr = work.iloc[tr_idx].copy()
    y_tr = y[tr_idx]
    df_va = work.iloc[va_idx].copy()
    y_va = y[va_idx]
    df_te = work.iloc[te_idx].copy()
    y_te = y[te_idx]

    backend_used = str(backend)
    backend_supported = backend_used in {"torch-mamba", "sklearn-moe", "hgb", "gbr", "rf", "ridge", "mlp"}
    if not backend_supported:
        backend_used = "hgb"

    pred_va, used_real_val = _predict_epitope_backend(backend_used, compute_mode, seed, df_tr, y_tr, df_va, torch_cfg, feature_spec=_feat_spec)
    pred_te, used_real_test = _predict_epitope_backend(backend_used, compute_mode, seed, df_tr, y_tr, df_te, torch_cfg, feature_spec=_feat_spec)
    val_metrics = _safe_metrics(y_va, pred_va)
    test_metrics = _safe_metrics(y_te, pred_te)
    val_abs_err = np.abs(pred_va - y_va).astype(np.float32)
    interval_df = _interval_calibration(y_true=y_te, y_pred=pred_te, val_abs_err=val_abs_err)

    cv_splits = _stratified_kfold_for_regression(y[trva_idx], n_splits=max(2, int(cv_folds)), seed=seed)
    cv_mae: list[float] = []
    cv_rmse: list[float] = []
    cv_r2: list[float] = []
    for cv_tr, cv_te in cv_splits:
        df_cv_tr = work.iloc[trva_idx].iloc[cv_tr].copy()
        y_cv_tr = y[trva_idx][cv_tr]
        df_cv_te = work.iloc[trva_idx].iloc[cv_te].copy()
        y_cv_te = y[trva_idx][cv_te]
        p_cv, _ = _predict_epitope_backend(backend_used, compute_mode, seed, df_cv_tr, y_cv_tr, df_cv_te, torch_cfg, feature_spec=_feat_spec)
        m_cv = _safe_metrics(y_cv_te, p_cv)
        cv_mae.append(float(m_cv["mae"]))
        cv_rmse.append(float(m_cv["rmse"]))
        cv_r2.append(float(m_cv["r2"]))

    mae_mean, mae_std, mae_ci = _mean_std_ci(cv_mae)
    rmse_mean, rmse_std, rmse_ci = _mean_std_ci(cv_rmse)
    r2_mean, r2_std, r2_ci = _mean_std_ci(cv_r2)

    _feat_spec = feature_spec or FeatureSpec()
    x_tr, _, _ = build_feature_matrix(df_tr, _feat_spec)
    x_te, _, _ = build_feature_matrix(df_te, _feat_spec)
    baseline_rows: list[dict[str, float | str]] = []
    baseline_preds: dict[str, np.ndarray] = {}
    for b in ["linear", "rf", "hgb"]:
        bm = _build_sklearn_regressor(b, seed=seed)
        bm.fit(x_tr, y_tr)
        bp = np.asarray(bm.predict(x_te), dtype=np.float32).reshape(-1)
        baseline_preds[b] = bp
        mm = _safe_metrics(y_te, bp)
        baseline_rows.append({"model": b, "mae": mm["mae"], "rmse": mm["rmse"], "r2": mm["r2"]})
    baseline_df = pd.DataFrame(baseline_rows).sort_values(["rmse", "mae"], ascending=[True, True])
    baseline_best_rmse = float(baseline_df["rmse"].min()) if not baseline_df.empty else float("inf")
    pass_gate = bool(float(test_metrics["rmse"]) < baseline_best_rmse)

    fail_df = df_te.copy()
    fail_df["y_true"] = y_te
    fail_df["y_pred"] = pred_te
    fail_df["abs_error"] = np.abs(pred_te - y_te)
    fail_df["seq_len"] = fail_df["epitope_seq"].astype(str).str.len()
    fail_df = fail_df.sort_values("abs_error", ascending=False).head(max(1, int(top_n_failures)))
    leakage = _leakage_audit(df_tr, df_te)

    seq_len = df_te["epitope_seq"].astype(str).str.len()
    bins = pd.cut(seq_len, bins=[-1, 12, 25, 10**9], labels=["short<=12", "mid13-25", "long>25"])
    strat_rows = []
    for label in ["short<=12", "mid13-25", "long>25"]:
        mask = bins == label
        if int(mask.sum()) == 0:
            continue
        m = _safe_metrics(y_te[mask.to_numpy()], pred_te[mask.to_numpy()])
        strat_rows.append({"length_bin": label, "n": int(mask.sum()), "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
    strat_df = pd.DataFrame(strat_rows)

    significance = None
    if baseline_preds:
        best_baseline = str(baseline_df.iloc[0]["model"]) if not baseline_df.empty else "hgb"
        if best_baseline in baseline_preds:
            significance = _paired_significance(
                y_true=y_te,
                pred_a=baseline_preds[best_baseline],
                pred_b=pred_te,
                name_a=best_baseline,
                name_b=backend_used,
            )

    ood_eval = _ood_subset_eval(df_train=df_tr, df_test=df_te, y_true=y_te, y_pred=pred_te)

    aa_strat_df = _aa_composition_stratification(df_test=df_te, y_true=y_te, y_pred=pred_te)

    mamba_calibration = None
    if backend_used == "torch-mamba" and torch_available():
        env_cols = [c for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"] if c in df_tr.columns]
        b_real = train_torch_mamba(df_tr, y_tr, env_cols=env_cols, cfg=torch_cfg, prefer_real_mamba=True)
        p_real = predict_torch_mamba(b_real, df_te)
        real_rmse = _safe_metrics(y_te, p_real)["rmse"]

        b_fb = train_torch_mamba(df_tr, y_tr, env_cols=env_cols, cfg=torch_cfg, prefer_real_mamba=False)
        p_fb = predict_torch_mamba(b_fb, df_te)
        fb_rmse = _safe_metrics(y_te, p_fb)["rmse"]

        mamba_calibration = {
            "real_mamba_used": bool(b_real.used_real_mamba),
            "real_rmse": float(real_rmse),
            "fallback_rmse": float(fb_rmse),
            "delta_rmse": float(fb_rmse - real_rmse),
        }

    external_metrics = None
    if external_df is not None and (not external_df.empty) and ("efficacy" in external_df.columns):
        ext_work = ensure_columns(external_df)
        y_ext_series = pd.to_numeric(ext_work["efficacy"], errors="coerce")
        if isinstance(y_ext_series, pd.Series):
            y_ext = y_ext_series.fillna(0.0).to_numpy(dtype=np.float32)
        else:
            y_ext = np.full((len(ext_work),), float(y_ext_series), dtype=np.float32)

        df_trva = work.iloc[trva_idx].copy()
        y_trva = y[trva_idx]
        pred_ext, _ = _predict_epitope_backend(backend_used, compute_mode, seed, df_trva, y_trva, ext_work, torch_cfg)
        external_metrics = _safe_metrics(y_ext, pred_ext)

    return {
        "enabled": True,
        "backend_supported": backend_supported,
        "backend_used": backend_used,
        "used_real_mamba": bool(used_real_val or used_real_test),
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
        "length_strat_df": strat_df,
        "significance": significance,
        "ood_eval": ood_eval,
        "aa_composition_strat_df": aa_strat_df,
        "mamba_calibration": mamba_calibration,
        "external_metrics": external_metrics,
    }
