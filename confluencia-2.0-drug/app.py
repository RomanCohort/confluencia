from __future__ import annotations

import io
import json
import hashlib
import platform
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.ed2mol_templates import write_ed2mol_config
from core.evolution import EvolutionConfig, CircRNAEvolutionConfig, evolve_molecules_with_reflection, evolve_cirrna_sequences
from core.immune_abm import build_epitope_triggers, simulate_single_epitope_response
from core.legacy_algorithms import LegacyAlgorithmConfig
from core.reliability import credible_eval_drug as core_credible_eval_drug, credible_eval_cirrna
from core.ctm import CTMParams, simulate_ctm, summarize_curve, infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve
from core.innate_immune import assess_innate_immune, innate_immune_result_to_dict, batch_assess_innate_immune
from core.pkpd import PKPDParams, infer_pkpd_params, simulate_pkpd, summarize_pkpd_curve
from core.features import MixedFeatureSpec, CircRNAFeatureSpec, build_feature_matrix, ensure_columns, ensure_cirrna_columns, build_cirrna_feature_vector, build_cirrna_feature_matrix
from core.moe import MOERegressor, choose_compute_profile
from core.training import export_drug_model_bytes, get_drug_model_metadata, import_drug_model_bytes, predict_drug_with_model, train_and_predict_drug, train_drug_model

# Cloud client for remote computation (optional import)
try:
    from api.frontend_client import CloudClient, create_cloud_client
    CLOUD_CLIENT_AVAILABLE = True
except ImportError:
    CloudClient = None
    create_cloud_client = None
    CLOUD_CLIENT_AVAILABLE = False

st.set_page_config(page_title="Confluencia 2.0 药物模块", layout="wide", page_icon="app.png")

st.title("Confluencia 2.0：药物训练与微机制疗效预测")
st.caption("MOE 自动建模 + CTM 动态仿真 + 靶点/免疫/炎症多指标预测")

doc_mode = "新手版"


@st.cache_data(show_spinner=False)
def _demo_data(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    def rand_seq(k: int = 15) -> str:
        return "".join(rng.choice(aa, size=k).tolist())

    smiles_candidates = [
        "CCO",
        "CCN(CC)CC",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C1=CC=C(C=C1)C=O",
        "CC1=CC(=O)NC(=O)N1",
    ]

    df = pd.DataFrame(
        {
            "smiles": rng.choice(smiles_candidates, size=n),
            "epitope_seq": [rand_seq(int(rng.integers(9, 21))) for _ in range(n)],
            "dose": rng.uniform(0.2, 8.0, size=n),
            "freq": rng.uniform(0.5, 3.0, size=n),
            "treatment_time": rng.uniform(0, 72, size=n),
            "group_id": rng.choice(["G1", "G2", "G3"], size=n),
        }
    )

    # Optional supervised labels in demo data.
    df["efficacy"] = 0.5 * df["dose"] + 0.25 * df["freq"] + rng.normal(0, 0.3, size=n)
    df["target_binding"] = np.clip(0.45 + 0.05 * df["dose"] + rng.normal(0, 0.05, size=n), 0, 1)
    df["immune_activation"] = np.clip(0.4 + 0.08 * df["freq"] + rng.normal(0, 0.05, size=n), 0, 1)
    df["immune_cell_activation"] = np.clip(0.35 + 0.12 * df["freq"] + 0.03 * df["dose"] + rng.normal(0, 0.06, size=n), 0, 1)
    df["inflammation_risk"] = np.clip(0.2 + 0.03 * df["dose"] + rng.normal(0, 0.04, size=n), 0, 1)
    df["toxicity_risk"] = np.clip(0.18 + 0.05 * df["dose"] + 0.02 * df["freq"] + rng.normal(0, 0.04, size=n), 0, 1)
    return df


def _input_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CCO", "CCN(CC)CC"],
            "epitope_seq": ["SLYNTVATL", "GILGFVFTL"],
            "dose": [2.0, 1.2],
            "dose_unit": ["mg", "mg"],
            "freq": [1.0, 0.8],
            "treatment_time": [24.0, 36.0],
            "time_unit": ["h", "h"],
            "group_id": ["G1", "G2"],
            "circrna_seq": ["", ""],
            "modification": ["none", "none"],
            "delivery_vector": ["LNP_standard", "LNP_standard"],
            "route": ["IV", "IV"],
            "ires_type": ["", ""],
        }
    )


def _input_schema_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": [
                "smiles",
                "dose",
                "dose_unit",
                "freq",
                "treatment_time",
                "time_unit",
                "efficacy",
                "epitope_seq",
                "group_id",
                "circrna_seq",
                "modification",
                "delivery_vector",
                "route",
                "ires_type",
            ],
            "required": ["Y", "Y", "N", "Y", "Y", "N", "N", "N", "N", "N", "N", "N", "N", "N"],
            "type": ["string", "float", "string", "float", "float", "string", "float", "string", "string", "string", "string", "string", "string", "string"],
            "rule": [
                "有效 SMILES",
                "> 0",
                "mg/g/ug/mcg（默认按 mg）",
                "> 0",
                ">= 0",
                "h/hr/hour/hours/d/day/days/min/minute/minutes（默认按 h）",
                "严格临床模式下缺失会警告",
                "可选序列特征",
                "可选分组字段",
                "circRNA 全序列或骨架序列（AUGC）",
                "m6A / Ψ / 5mC / ms2m6A / none",
                "LNP_standard / LNP_liver / LNP_spleen / AAV / naked",
                "IV / SC / IM / ID",
                "EMCV / HCV / CVB3 / c-MYC / VEGF / BiP / custom（可选）",
            ],
            "example": ["CCO", "2.0", "mg", "1.0", "24", "h", "0.62", "SLYNTVATL", "G1",
                         "AUGCGCUAUGGC...", "m6A", "LNP_liver", "IV", "EMCV"],
        }
    )


REQUIRED_INPUT_COLUMNS = ["smiles", "dose", "freq", "treatment_time"]


def _read_uploaded_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_err: Exception | None = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(io.BytesIO(raw))


def _read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(uploaded_file.getvalue()))
    return _read_uploaded_csv(uploaded_file)


def _missing_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


@st.cache_data(show_spinner=False)
def _build_numeric_error_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checks = {
        "dose": (0.0, False, "必须 > 0"),
        "freq": (0.0, False, "必须 > 0"),
        "treatment_time": (0.0, True, "必须 >= 0"),
    }
    for col, (lower, allow_equal, rule_text) in checks.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad_parse_mask = series.isna()
        valid_mask = (~bad_parse_mask) & ((series >= lower) if allow_equal else (series > lower))
        q_low = lower
        q_high = lower
        if bool(valid_mask.any()):
            valid_vals = series[valid_mask]
            q_low = float(valid_vals.quantile(0.05))
            q_high = float(valid_vals.quantile(0.95))
            if q_high < q_low:
                q_low, q_high = q_high, q_low
            raw_suggested = float(valid_vals.median())
            suggested_value = float(np.clip(raw_suggested, q_low, q_high))
        else:
            suggested_value = 0.0 if allow_equal else 0.1
            q_high = suggested_value

        if allow_equal:
            out_of_range_mask = (~bad_parse_mask) & (series < lower)
        else:
            out_of_range_mask = (~bad_parse_mask) & (series <= lower)

        for idx in series[bad_parse_mask].index.tolist():
            rows.append(
                {
                    "row_index": int(idx),
                    "column": col,
                    "issue": "无法解析为数值",
                    "rule": rule_text,
                    "value": df.at[idx, col],
                    "suggested_fix": "改为数字，例如 1.0",
                    "suggested_value": suggested_value,
                    "suggested_min": q_low,
                    "suggested_max": q_high,
                }
            )

        for idx in series[out_of_range_mask].index.tolist():
            suggested = "改为 > 0" if not allow_equal else "改为 >= 0"
            rows.append(
                {
                    "row_index": int(idx),
                    "column": col,
                    "issue": "数值越界",
                    "rule": rule_text,
                    "value": float(series.at[idx]),
                    "suggested_fix": suggested,
                    "suggested_value": suggested_value,
                    "suggested_min": q_low,
                    "suggested_max": q_high,
                }
            )

    return pd.DataFrame(rows)


def _normalize_units(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    infos: list[str] = []
    errors: list[str] = []

    if "dose" in out.columns and "dose_unit" in out.columns:
        unit = out["dose_unit"].fillna("").astype(str).str.strip().str.lower()
        dose = pd.to_numeric(out["dose"], errors="coerce")
        dose_factors = {
            "mg": 1.0,
            "g": 1000.0,
            "ug": 0.001,
            "mcg": 0.001,
        }
        known = set(dose_factors.keys())
        unknown_mask = (unit != "") & (~unit.isin(known))
        if bool(unknown_mask.any()):
            bad_vals = sorted({str(v) for v in unit[unknown_mask].head(5).tolist()})
            errors.append(f"dose_unit 存在不支持取值: {', '.join(bad_vals)}；支持: mg/g/ug/mcg")

        converted = 0
        for u, factor in dose_factors.items():
            m = (unit == u) & dose.notna()
            if bool(m.any()):
                out.loc[m, "dose"] = dose.loc[m] * factor
                converted += int(m.sum())
        if converted > 0:
            infos.append(f"dose 已按 dose_unit 归一化到 mg（处理 {converted} 行）。")

    if "treatment_time" in out.columns and "time_unit" in out.columns:
        unit = out["time_unit"].fillna("").astype(str).str.strip().str.lower()
        t = pd.to_numeric(out["treatment_time"], errors="coerce")
        time_factors = {
            "h": 1.0,
            "hr": 1.0,
            "hour": 1.0,
            "hours": 1.0,
            "d": 24.0,
            "day": 24.0,
            "days": 24.0,
            "min": 1.0 / 60.0,
            "minute": 1.0 / 60.0,
            "minutes": 1.0 / 60.0,
        }
        known = set(time_factors.keys())
        unknown_mask = (unit != "") & (~unit.isin(known))
        if bool(unknown_mask.any()):
            bad_vals = sorted({str(v) for v in unit[unknown_mask].head(5).tolist()})
            errors.append(
                "time_unit 存在不支持取值: "
                + f"{', '.join(bad_vals)}；支持: h/hr/hour/hours/d/day/days/min/minute/minutes"
            )

        converted = 0
        for u, factor in time_factors.items():
            m = (unit == u) & t.notna()
            if bool(m.any()):
                out.loc[m, "treatment_time"] = t.loc[m] * factor
                converted += int(m.sum())
        if converted > 0:
            infos.append(f"treatment_time 已按 time_unit 归一化到小时（处理 {converted} 行）。")

    return out, infos, errors


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["smiles", "epitope_seq", "group_id"]:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str).str.strip()
    for c in ["dose", "freq", "treatment_time"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "smiles": ["SMILES", "compound_smiles", "drug_smiles"],
        "epitope_seq": ["epitope", "sequence", "peptide_seq"],
        "dose": ["dosage", "dose_mg"],
        "dose_unit": ["dosage_unit", "dose_units", "doseUnit"],
        "freq": ["frequency", "times", "dose_freq"],
        "treatment_time": ["time", "duration_h", "treatment_hours"],
        "time_unit": ["duration_unit", "time_units", "treatment_unit", "timeUnit"],
        "group_id": ["group", "cohort", "group_name"],
        "immune_cell_activation": ["immune_cell_act", "immune_cell_score", "tcell_activation"],
        "toxicity_risk": ["toxicity", "tox_risk", "safety_risk"],
        "inflammation_risk": ["inflammation", "inflam_risk", "cytokine_risk"],
    }
    for canonical, aliases in alias_map.items():
        if canonical in out.columns:
            continue
        for a in aliases:
            if a in out.columns:
                out = out.rename(columns={a: canonical})
                break
    return out


def _core_ready_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return 1.0
    hit = sum(1 for c in cols if c in df.columns)
    return float(hit) / float(len(cols))


@st.cache_data(show_spinner=False)
def _data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        miss = float(df[c].isna().sum()) / float(n)
        rows.append({"列名": c, "缺失率": miss})

    invalid_ratio = 0.0
    if "smiles" in df.columns:
        smiles = df["smiles"].astype(str).tolist()
        try:
            from rdkit import Chem  # type: ignore
            mol_from_smiles = getattr(Chem, "MolFromSmiles", None)
            if mol_from_smiles is None:
                raise RuntimeError("MolFromSmiles unavailable")

            bad = 0
            for s in smiles:
                if mol_from_smiles(str(s)) is None:
                    bad += 1
            invalid_ratio = float(bad) / float(max(len(smiles), 1))
        except Exception:
            # Fallback heuristic when RDKit is unavailable.
            bad = 0
            for s in smiles:
                t = str(s).strip()
                if (not t) or (" " in t) or ("%" in t):
                    bad += 1
            invalid_ratio = float(bad) / float(max(len(smiles), 1))

    rows.append({"列名": "smiles_invalid_ratio", "缺失率": invalid_ratio})
    return pd.DataFrame(rows)


def _append_experiment_log(module: str, config: dict, metrics: dict) -> None:
    base = Path(__file__).resolve().parents[1] / "logs"
    base.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
    }
    csv_path = base / "experiments.csv"
    line_df = pd.DataFrame([row])
    if csv_path.exists():
        line_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        line_df.to_csv(csv_path, index=False)


def _append_clinical_audit(event: str, payload: dict) -> None:
    base = Path(__file__).resolve().parents[1] / "logs" / "clinical_audit"
    base.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "payload": payload,
    }
    audit_path = base / "drug_clinical_audit.jsonl"
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hash_dataframe(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _snapshot_env_deps() -> dict[str, str]:
    deps: dict[str, str] = {}
    for pkg in ["python", "numpy", "pandas", "scikit-learn", "streamlit", "rdkit", "torch"]:
        if pkg == "python":
            deps[pkg] = platform.python_version()
            continue
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except Exception:
            deps[pkg] = "not-installed"
    return deps


def _save_repro_bundle(module: str, data_df: pd.DataFrame, config: dict, metrics: dict) -> None:
    base = Path(__file__).resolve().parents[1] / "logs" / "reproduce"
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_hash = _hash_dataframe(data_df)
    run_id = f"{module}_{ts}_{data_hash[:8]}"
    env_deps = _snapshot_env_deps()

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "rows": int(len(data_df)),
        "data_sha256": data_hash,
        "python_executable": sys.executable,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "env_deps": json.dumps(env_deps, ensure_ascii=False),
    }

    csv_path = base / "runs.csv"
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)

    md_path = base / f"{run_id}.md"
    md_lines = [
        f"# Repro Report ({module})",
        f"- run_id: {run_id}",
        f"- timestamp: {row['timestamp']}",
        f"- rows: {row['rows']}",
        f"- data_sha256: {data_hash}",
        f"- python_executable: {sys.executable}",
        "",
        "## Config",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Environment Dependencies",
        "```json",
        json.dumps(env_deps, ensure_ascii=False, indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def _summarize_dynamic_metrics(result_df: pd.DataFrame) -> dict[str, float]:
    summary = {
        "ctm_peak_efficacy": 0.0,
        "ctm_auc_efficacy": 0.0,
    }
    if result_df is None or result_df.empty:
        return summary

    for key in ["ctm_peak_efficacy", "ctm_auc_efficacy"]:
        if key in result_df.columns:
            vals = pd.to_numeric(result_df[key], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                summary[key] = float(vals.mean())
    return summary


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


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / max(np.sqrt(float(arr.size)), 1e-8)) if arr.size > 1 else 0.0
    return mean, std, ci95


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


def _credible_eval_drug(
    df: pd.DataFrame,
    backend: str,
    compute_mode: str,
    seed: int,
    test_ratio: float,
    val_ratio: float,
    cv_folds: int,
    top_n_failures: int,
    external_df: pd.DataFrame | None,
    feature_spec: MixedFeatureSpec | None = None,
) -> dict:
    _spec = feature_spec or MixedFeatureSpec(smiles_hash_dim=128, smiles_rdkit_bits=2048, smiles_rdkit_version=2, prefer_rdkit=True)
    work = ensure_columns(df)
    if "efficacy" not in work.columns:
        return {"enabled": False, "reason": "missing_label"}

    y_series = pd.to_numeric(work["efficacy"], errors="coerce")
    if isinstance(y_series, pd.Series):
        y = y_series.fillna(0.0).to_numpy(dtype=np.float32)
    else:
        y = np.full((len(work),), float(y_series), dtype=np.float32)
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

    cv = KFold(n_splits=max(2, int(cv_folds)), shuffle=True, random_state=seed)
    cv_mae: list[float] = []
    cv_rmse: list[float] = []
    cv_r2: list[float] = []
    for cv_tr, cv_te in cv.split(x[trva_idx]):
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

    baseline_rows: list[dict] = []
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

    external_metrics = None
    if external_df is not None and (not external_df.empty) and ("efficacy" in external_df.columns):
        ext_work = ensure_columns(external_df)
        x_ext, _, _ = build_feature_matrix(ext_work, _spec)
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
        "failure_df": fail_df,
        "external_metrics": external_metrics,
    }


pkpd_override_enabled = False
pkpd_use_manual_params = False
pkpd_manual_ka = 0.3
pkpd_manual_k12 = 0.2
pkpd_manual_k21 = 0.12
pkpd_manual_ke = 0.08
pkpd_manual_v1 = 4.0
pkpd_manual_emax = 1.2
pkpd_manual_ec50 = 0.5
pkpd_manual_hill = 1.2
pkpd_horizon = 72
pkpd_dt = 1.0

with st.sidebar:
    st.header("运行设置")
    doc_mode = st.radio("说明模式", ["新手版", "专家版"], index=0, horizontal=True)
    preset = st.selectbox("参数预设", ["平衡", "快速", "高精度"], index=0)
    compute_mode = st.selectbox("计算资源档位", ["auto", "low", "medium", "high"], index=0)
    backend_map = {
        "Confluencia 2.0 MOE": "moe",
        "Confluencia 1.0 HGB": "hgb",
        "Confluencia 1.0 GBR": "gbr",
        "Confluencia 1.0 RF": "rf",
        "Confluencia 1.0 Ridge": "ridge",
        "Confluencia 1.0 MLP": "mlp",
        "Confluencia 1.0 Torch-MLP": "torch_mlp",
        "Confluencia 1.0 Transformer": "transformer",
    }
    backend_label = st.selectbox("算法后端", list(backend_map.keys()), index=1)
    model_backend = backend_map[backend_label]
    st.caption("默认使用 Confluencia 1.0 HGB，可切换到其他 1.0 算法。")
    dynamics_backend_map = {
        "CTM（默认）": "ctm",
        "NDP4PD（吉林大学动力学）": "ndp4pd",
    }
    dynamics_label = st.selectbox("动力学后端", list(dynamics_backend_map.keys()), index=0)
    dynamics_model = dynamics_backend_map[dynamics_label]
    with st.expander("circRNA 模式（v2.2）", expanded=False):
        crna_mode = st.checkbox("启用 circRNA 模式", value=False, key="crna_mode")
        if crna_mode:
            crna_modification = st.selectbox("修饰类型", ["none", "m6A", "Psi", "5mC", "ms2m6A"], index=1, key="crna_mod")
            crna_delivery = st.selectbox("递送系统", ["LNP_standard", "LNP_liver", "LNP_spleen", "AAV", "naked"], index=1, key="crna_del")
            crna_route = st.selectbox("给药途径", ["IV", "SC", "IM", "ID"], index=0, key="crna_route")
            crna_ires = st.selectbox("IRES 类型", ["", "EMCV", "HCV", "CVB3", "c-MYC", "VEGF", "BiP", "custom"], index=0, key="crna_ires")
            st.caption("启用后自动附加 circRNA 特征并运行先天免疫评估和 RNA PK 仿真。")
    with st.expander("自适应系统", expanded=False):
        adaptive_enabled = st.checkbox("启用自适应校准", value=False)
        adaptive_strength = st.slider("自适应强度", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        st.caption("对疗效/风险预测进行分布感知校准，并给出剂量/频次自适应系数。")

    with st.expander("超参数调优", expanded=False):
        st.caption("启用后将对 MOE 专家 (HGB/RF/Ridge/MLP) 执行交叉验证超参数搜索")
        tune_hyperparams = st.checkbox("启用超参数调优", value=False, help="在训练前对 MOE 专家参数执行 RandomizedSearchCV，耗时较长")
        if tune_hyperparams:
            tune_strategy = st.selectbox("调优策略", ["random", "grid"], index=0, help="random: 随机采样; grid: 穷举搜索")
            tune_n_iter = st.slider("随机搜索迭代次数", min_value=5, max_value=50, value=20, step=5)
            tune_cv = st.slider("调优交叉验证折数", min_value=2, max_value=5, value=3, step=1)
        else:
            tune_strategy = "random"
            tune_n_iter = 20
            tune_cv = 3

    # --- Cloud Mode Configuration ---
    with st.expander("云服务器接口", expanded=False):
        st.caption("切换到远程云服务器进行计算，减轻本地计算压力。")
        if not CLOUD_CLIENT_AVAILABLE:
            st.warning("云客户端模块未安装。请确保 api/frontend_client.py 存在。")
            cloud_mode_enabled = False
            cloud_server_url = "http://localhost:8000"
        else:
            cloud_mode_enabled = st.checkbox("启用云服务器模式", value=False, key="cloud_mode_enabled")
            if cloud_mode_enabled:
                cloud_server_url = st.text_input("服务器地址", value="http://localhost:8000", key="cloud_server_url")
                st.caption("请确保服务器已启动：`python server.py --port 8000`")

                # Connection status indicator
                if st.button("测试连接", key="cloud_test_connection"):
                    client = create_cloud_client("remote", cloud_server_url)
                    status = client.check_connection()
                    if status.connected:
                        st.success(f"连接成功 ({status.latency_ms:.0f}ms)")
                    else:
                        st.error(f"连接失败: {status.error}")

                # Quick status check (cached per session to avoid ~2s latency on each render)
                if cloud_mode_enabled and not st.session_state.get("_cloud_health_cached"):
                    try:
                        import httpx
                        resp = httpx.get(f"{cloud_server_url}/api/health", timeout=2.0)
                        if resp.status_code == 200:
                            st.caption("状态: 已连接")
                        else:
                            st.caption("状态: 服务器响应异常")
                    except Exception:
                        st.caption("状态: 未连接")
                    st.session_state["_cloud_health_cached"] = True
            else:
                cloud_server_url = "http://localhost:8000"
                st.caption("本地模式：所有计算在本地执行。")

    run_mode = st.selectbox("运行模式", ["训练并预测", "仅训练(保存到会话)", "仅预测(使用最近模型)", "仅查看最近结果"], index=0)
    if run_mode in {"仅训练(保存到会话)", "仅预测(使用最近模型)"} and model_backend != "moe":
        st.info("拆分模式当前仅支持 Confluencia 2.0 MOE；请切换算法后端为 MOE。")
    with st.expander("模型导入/导出（MOE 拆分模式）", expanded=False):
        _cloud_mode_model = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
        if _cloud_mode_model:
            st.info("云服务器模式下，模型存储在服务器端。导入/导出通过服务器 API 执行。")
            _cloud_client_model = create_cloud_client("remote", cloud_server_url)
            session_model = st.session_state.get("drug_trained_model")
            if session_model is None:
                st.caption("当前会话暂无模型引用。请先训练或从服务器导入。")
            else:
                model_id = session_model.get("model_id", "unknown") if isinstance(session_model, dict) else "local"
                st.caption(f"当前会话模型 ID: {model_id}")
                if isinstance(session_model, dict):
                    st.json(session_model.get("metadata", {}), expanded=False)
                if st.button("从服务器下载模型文件"):
                    try:
                        model_bytes = _cloud_client_model.export_model(session_model)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            "下载模型文件",
                            data=model_bytes,
                            file_name=f"confluencia2_drug_model_{ts}.cf2model",
                            mime="application/octet-stream",
                        )
                    except Exception as e:
                        st.error(f"导出失败: {e}")

            st.divider()
            st.caption("从本地文件导入到服务器")
            allow_unsafe_import_cloud = st.checkbox("我已确认来源可信，允许导入", value=False, key="cloud_allow_import")
            if allow_unsafe_import_cloud:
                model_file_cloud = st.file_uploader("导入模型到服务器", type=["cf2model"], key="drug_model_upload_cloud")
                if model_file_cloud is not None:
                    try:
                        loaded_model_cloud = _cloud_client_model.import_model(model_file_cloud.getvalue())
                        st.session_state["drug_trained_model"] = loaded_model_cloud
                        st.success(f"模型导入成功，服务器模型 ID: {loaded_model_cloud.get('model_id', 'unknown')}")
                    except Exception as e:
                        st.error(f"模型导入失败: {e}")
        else:
            st.warning("默认已禁用模型文件导入（pickle 反序列化存在安全风险）。")
            allow_unsafe_import = st.checkbox("我已确认来源可信，允许风险导入", value=False)
            if allow_unsafe_import:
                model_file = st.file_uploader("导入已训练模型（.cf2model）", type=["cf2model"], key="drug_model_upload")
                if model_file is not None:
                    try:
                        loaded_model = import_drug_model_bytes(model_file.getvalue(), allow_unsafe_deserialization=True)
                        st.session_state["drug_trained_model"] = loaded_model
                        st.success("模型导入成功，已写入当前会话。")
                        st.caption("已导入模型元信息")
                        st.json(get_drug_model_metadata(loaded_model), expanded=False)
                    except Exception as e:
                        st.error(f"模型导入失败: {e}")
            else:
                st.caption("如需导入旧模型，请勾选上方确认项后再上传。")

            session_model = st.session_state.get("drug_trained_model")
            if session_model is None:
                st.caption("当前会话暂无已训练模型。")
            else:
                st.caption("当前会话模型元信息")
                st.json(get_drug_model_metadata(session_model), expanded=False)
                model_bytes = export_drug_model_bytes(session_model)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "下载当前会话模型",
                    data=model_bytes,
                    file_name=f"confluencia2_drug_model_{ts}.cf2model",
                    mime="application/octet-stream",
                )
    do_benchmark = st.checkbox("启用算法对比排行榜", value=False)
    strict_clinical_mode = st.checkbox("严格临床模式（无 efficacy 给出警告）", value=True)
    use_demo = st.checkbox("使用内置演示数据", value=True)

    if preset == "快速":
        compute_mode = "low"
    elif preset == "高精度":
        if compute_mode == "auto":
            compute_mode = "high"

    with st.expander("预训练编码器配置", expanded=False):
        st.caption("启用预训练分子/蛋白编码器以提升特征表示能力（需网络下载模型）")
        use_gnn = st.checkbox("启用 GNN 图神经网络编码器", value=False, help="EnhancedGNN 对分子图结构编码，128维，需 RDKit")
        use_chemberta = st.checkbox("启用 ChemBERTa 分子Transformer", value=False, help="seyonec/ChemBERTa-zinc-base-v1，768维，需网络下载")
        use_esm2_drug = st.checkbox("启用 ESM-2 蛋白序列编码（Drug epitope特征）", value=False, help="facebook/esm2_t33_650M_UR50D，1280维，需网络下载")
        use_pk_prior = st.checkbox("启用 PK 先验特征", value=False, help="Lipinski五规则/溶解度/半衰期等药代动力学先验，9维")
        use_dose_response = st.checkbox("启用剂量-响应曲线特征", value=False, help="累积剂量/Emax/EC50/Hill系数等剂量响应特征，12维")
        use_cross_features = st.checkbox("启用剂量-频次交叉特征", value=False, help="dose×binding/freq×immune等交互特征，9维")
        if use_chemberta or use_esm2_drug:
            cache_dir = st.text_input(
                "编码器缓存目录",
                value="D:/IGEM集成方案/data/cache",
                help="ESM-2/ChemBERTa 嵌入缓存存放路径，留空则使用 .cache/"
            )
        else:
            cache_dir = "./.cache"

    with st.expander("1.0 深度模型参数", expanded=False):
        legacy_epochs = st.slider("训练轮数", min_value=5, max_value=120, value=40, step=5)
        legacy_batch = st.select_slider("批大小", options=[16, 32, 64, 96, 128], value=64)
        legacy_lr = st.select_slider("学习率", options=[5e-4, 1e-3, 2e-3, 3e-3], value=1e-3)
        legacy_h1 = st.select_slider("Torch 隐藏层1", options=[128, 192, 256, 384, 512], value=256)
        legacy_h2 = st.select_slider("Torch 隐藏层2", options=[64, 96, 128, 160, 192, 256], value=128)
        legacy_max_len = st.select_slider("Transformer 最大长度", options=[64, 96, 128, 160, 192, 256], value=128)
        legacy_emb = st.select_slider("Transformer Embedding", options=[64, 96, 128, 160], value=96)
        legacy_heads = st.select_slider("Transformer Heads", options=[2, 4, 8], value=4)
        legacy_layers = st.select_slider("Transformer 层数", options=[1, 2, 3, 4], value=2)

    with st.expander("参数介绍", expanded=False):
        if doc_mode == "新手版":
            st.markdown(
                """
                - `算法后端`：可切换 Confluencia 1.0 的全部核心算法族。
                - `计算资源档位`：越高通常效果更好，但耗时更久。
                - `使用内置演示数据`：不准备数据时可先直接跑通。
                - `轮数/候选数`：越大搜索越充分，但速度会变慢。
                - `剂量/频次/治疗时长`：会直接影响疗效与风险预测。
                """
            )
        else:
            st.markdown(
                """
                                - `算法后端`：
                                    `moe/hgb/gbr/rf/ridge/mlp/torch_mlp/transformer` 对应 Confluencia 2.0 与 1.0 算法实现。
                - `计算资源档位`：影响 MOE 拟合与 CTM 仿真预算，决定吞吐与精度平衡。
                - `使用内置演示数据`：用于快速回归测试；关闭后可上传自定义 CSV。
                - `轮数`：进化迭代深度，增加可提升探索但放大计算成本。
                - `每轮保留 Top-k`：控制选择压力，过小易早熟，过大易保守。
                - `每轮候选数`：每轮探索宽度，直接决定策略更新统计稳定性。
                - `剂量/频次/治疗时长`：上下文干预变量，影响微观预测与 CTM 时序响应。
                - `进化计算档位`：作用于进化评估阶段，可与主流程训练档位分离。
                - `Pareto 搜索`：在多目标空间内自适配奖励权重，提升前沿质量。
                """
            )

    with st.expander("PK/PD 参数面板（SciPy）", expanded=False):
        pkpd_override_enabled = st.checkbox("启用实时 PK/PD 重算", value=True)
        pkpd_use_manual_params = st.checkbox("手动覆盖房室参数", value=False)
        pkpd_horizon = st.slider("PK/PD 仿真时长 (h)", min_value=24, max_value=240, value=72, step=12)
        pkpd_dt = st.select_slider("时间步长 (h)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)

        if pkpd_use_manual_params:
            pkpd_manual_ka = st.slider("ka 吸收速率", min_value=0.01, max_value=2.0, value=0.30, step=0.01)
            pkpd_manual_k12 = st.slider("k12 中央->外周", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
            pkpd_manual_k21 = st.slider("k21 外周->中央", min_value=0.01, max_value=1.0, value=0.12, step=0.01)
            pkpd_manual_ke = st.slider("ke 清除速率", min_value=0.005, max_value=0.6, value=0.08, step=0.005)
            pkpd_manual_v1 = st.slider("V1 中央容积 (L)", min_value=1.0, max_value=20.0, value=4.0, step=0.1)
            pkpd_manual_emax = st.slider("Emax 最大药效", min_value=0.1, max_value=3.0, value=1.2, step=0.05)
            pkpd_manual_ec50 = st.slider("EC50 (mg/L)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
            pkpd_manual_hill = st.slider("Hill 系数", min_value=0.5, max_value=3.0, value=1.2, step=0.05)

        st.caption("关闭手动覆盖时，参数会由模型根据结合/免疫/炎症状态自动推断。")

with st.expander("机制解释（药物模块）", expanded=False):
    if doc_mode == "新手版":
        st.markdown(
            """
            - 先预测疗效、靶点结合、免疫激活、炎症风险，再做 CTM 动态模拟。
            - 现在可以在同一页面切换 Confluencia 1.0 的经典算法与深度算法。
            - 强化学习会反复生成和筛选分子，让候选逐轮变好。
            - 严格临床模式开启时，无 `efficacy` 会给出警告。
            """
        )
    else:
        st.markdown(
            """
            - 数据流：输入 `smiles + epitope_seq + 给药条件`，输出微观指标与 CTM 动力学结果。
            - 训练机制：
                            1. `moe` 学习总体疗效 `efficacy_pred`。
                            2. Confluencia 1.0 算法族可选：`hgb/gbr/rf/ridge/mlp/torch_mlp/transformer`。
              2. 微观预测器学习 `target_binding/immune_activation/inflammation_risk`。
              3. CTM 参数模型将微观状态映射到 `ka/kd/ke/km/signal_gain` 并做动态仿真。
            - 进化机制：ED2Mol + 纯强化学习反思更新策略，逐轮优化分子候选。
            - 标签机制：严格临床模式开启时，缺少真实 `efficacy` 会给出警告。
            """
        )

st.subheader("快速上手")
st.markdown("1. 先下载模板并填入数据。 2. 上传 CSV/Parquet 或使用演示数据。 3. 点击`开始训练与预测`查看结果并下载报告。")

tpl_buf = io.StringIO()
_input_template().to_csv(tpl_buf, index=False)
st.download_button(
    "下载输入模板 CSV",
    data=tpl_buf.getvalue(),
    file_name="confluencia2_drug_input_template.csv",
    mime="text/csv",
)

schema_buf = io.StringIO()
_input_schema_template().to_csv(schema_buf, index=False)
st.download_button(
    "下载字段规则 CSV",
    data=schema_buf.getvalue(),
    file_name="confluencia2_drug_input_schema.csv",
    mime="text/csv",
)

uploaded = st.file_uploader(
    "上传数据文件（CSV / Parquet）",
    type=["csv", "parquet", "pq"],
    help="必填: smiles, dose, freq, treatment_time。建议补充: epitope_seq。可选标签: efficacy/target_binding/immune_activation/immune_cell_activation/inflammation_risk/toxicity_risk/group_id",
)
df = pd.DataFrame()
if use_demo:
    df = _demo_data(120)
elif uploaded is not None:
    try:
        df = _read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"文件读取失败，请检查格式/编码: {e}")
        st.stop()
else:
    st.info("请上传 CSV/Parquet 文件，或启用演示数据。")
    st.stop()

df = _apply_column_aliases(df)
df = _normalize_input(df)
df, unit_infos, unit_errors = _normalize_units(df)
if df.empty:
    st.error("输入数据为空，请检查文件内容。")
    st.stop()

if unit_errors:
    for msg in unit_errors:
        st.error(msg)
    st.stop()

for msg in unit_infos:
    st.info(msg)

missing_required = _missing_required_columns(df, REQUIRED_INPUT_COLUMNS)
if missing_required:
    st.error(f"缺少必填列: {', '.join(missing_required)}")
    st.stop()

missing_core = [c for c in ["smiles", "epitope_seq", "dose", "freq", "treatment_time"] if c not in df.columns]
if missing_core:
    st.warning(f"检测到缺少核心列: {', '.join(missing_core)}。系统会自动补全缺失列为默认值，但建议完善后再训练。")

bad_numeric = 0
for c in ["dose", "freq", "treatment_time"]:
    if c in df.columns:
        bad_numeric += int(df[c].isna().sum())
if bad_numeric > 0:
    st.warning(f"检测到 {bad_numeric} 个数值单元无法解析，训练时将按 0 处理。建议修正后重试。")

numeric_errors_df = _build_numeric_error_report(df)
if not numeric_errors_df.empty:
    st.warning(f"检测到 {len(numeric_errors_df)} 个数值问题（解析失败或范围越界）。")

core_cols = ["smiles", "epitope_seq", "dose", "freq", "treatment_time"]
ready_ratio = _core_ready_ratio(df, core_cols)
cq1, cq2, cq3 = st.columns(3)
cq1.metric("数据行数", f"{len(df)}")
cq2.metric("核心列完备度", f"{ready_ratio * 100:.0f}%")
cq3.metric("数值异常单元", f"{bad_numeric}")
if ready_ratio < 1.0:
    st.info("已尝试自动识别常见别名列（如 `SMILES`、`frequency`、`time`）。建议确认映射后的列名是否符合预期。")

st.subheader("输入预览")
st.dataframe(df.head(12), use_container_width=True)

with st.expander("数据质控报告", expanded=False):
    dq = _data_quality_report(df)
    st.dataframe(dq, use_container_width=True)
    if not numeric_errors_df.empty:
        st.markdown("**数值错误明细（可下载给数据团队修复）**")
        st.dataframe(numeric_errors_df.head(200), use_container_width=True)
        err_buf = io.StringIO()
        numeric_errors_df.to_csv(err_buf, index=False)
        st.download_button(
            "下载数值错误报告 CSV",
            data=err_buf.getvalue(),
            file_name="drug_numeric_error_report.csv",
            mime="text/csv",
        )

with st.expander("科研可信评估设置（建议开启）", expanded=False):
    st.caption("用于严格切分、5-fold CV 置信区间、基线对照与失败样本分析。")
    enable_credible_eval = st.checkbox("启用科研可信评估", value=True)
    ce1, ce2, ce3 = st.columns(3)
    credible_seed = ce1.number_input("随机种子", min_value=1, max_value=999999, value=42, step=1)
    credible_test_ratio = ce2.select_slider("测试集比例", options=[0.1, 0.15, 0.2, 0.25], value=0.2)
    credible_val_ratio = ce3.select_slider("验证集比例", options=[0.1, 0.15, 0.2], value=0.2)

    ce4, ce5 = st.columns(2)
    credible_cv_folds = ce4.select_slider("CV 折数", options=[3, 4, 5, 6, 8, 10], value=5)
    credible_top_fail_n = ce5.slider("失败样本导出 Top-N", min_value=5, max_value=100, value=20, step=5)

    uploaded_external = st.file_uploader(
        "上传外部独立测试集（CSV / Parquet，可选，需含 efficacy）",
        type=["csv", "parquet", "pq"],
        key="drug_external_eval_csv",
    )
    external_eval_df: pd.DataFrame | None = None
    if uploaded_external is not None:
        try:
            external_eval_df = _normalize_input(_apply_column_aliases(_read_uploaded_file(uploaded_external)))
            external_eval_df, external_unit_infos, external_unit_errors = _normalize_units(external_eval_df)
            if external_unit_errors:
                for msg in external_unit_errors:
                    st.error(f"外部测试集单位错误: {msg}")
                external_eval_df = None
            else:
                for msg in external_unit_infos:
                    st.info(f"外部测试集: {msg}")
            missing_external = []
            if external_eval_df is not None:
                missing_external = _missing_required_columns(external_eval_df, REQUIRED_INPUT_COLUMNS + ["efficacy"])

            if external_eval_df is not None and missing_external:
                st.error(f"外部测试集缺少必填列: {', '.join(missing_external)}")
                external_eval_df = None
            elif external_eval_df is not None:
                st.success(f"外部测试集已载入: {len(external_eval_df)} 行")
        except Exception as e:
            st.error(f"外部测试集读取失败: {e}")

# =============================================================================
# PopPK 临床分析面板 (v2.1+)
# =============================================================================
with st.expander("PopPK 临床分析 (v2.1+)", expanded=False):
    st.markdown("""
    **RNACTM 群体药代动力学模型验证结果**

    基于 30 个模拟受试者、354 条观测记录的 PopPK 拟合与 VPC 验证。
    """)

    poppk_results_path = Path(__file__).parent / 'benchmarks' / 'results' / 'rnactm_poppk_fit_results.json'
    vpc_plot_path = Path(__file__).parent / 'benchmarks' / 'results' / 'vpc_plot.png'
    clinical_report_path = Path(__file__).parent / 'benchmarks' / 'results' / 'rnactm_clinical_report.html'

    if poppk_results_path.exists():
        with open(poppk_results_path, 'r', encoding='utf-8') as f:
            poppk_results = json.load(f)

        fit_quality = poppk_results.get('fit_quality', {})
        params = poppk_results.get('parameters', {})
        literature = poppk_results.get('literature_comparison', {})
        data_summary = poppk_results.get('data_summary', {})

        # 关键指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{fit_quality.get('r_squared', 0):.4f}")
        c2.metric("Pearson r", f"{fit_quality.get('pearson_r', 0):.4f}")
        c3.metric("受试者数", data_summary.get('n_subjects', 'N/A'))
        c4.metric("观测记录", data_summary.get('n_observations', 'N/A'))

        # 参数估计表
        st.markdown("**参数估计**")
        param_df = pd.DataFrame([
            {"参数": "tv_ka", "描述": "吸收速率常数", "值": f"{params.get('tv_ka', 0):.4f}", "单位": "1/h"},
            {"参数": "tv_ke", "描述": "消除速率常数", "值": f"{params.get('tv_ke', 0):.4f}", "单位": "1/h"},
            {"参数": "tv_V", "描述": "分布容积", "值": f"{params.get('tv_v', 0):.4f}", "单位": "L/kg"},
            {"参数": "tv_F", "描述": "生物利用度", "值": f"{params.get('tv_f', 0):.6f}", "单位": "-"},
            {"参数": "σ_prop", "描述": "比例残差", "值": f"{params.get('sigma_prop', 0):.4f}", "单位": "CV%"},
        ])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        # 文献对比
        st.markdown("**文献对比 (半衰期)**")
        lit_rows = []
        for mod, data in literature.items():
            status = "PASS" if data.get('error_ke_pct', 100) < 30 else "FAIL"
            lit_rows.append({
                "修饰类型": mod,
                "拟合 t½ (h)": f"{data.get('fitted_hl', 0):.1f}",
                "参考 t½ (h)": f"{data.get('ref_hl', 0):.1f}",
                "误差 (%)": f"{data.get('error_ke_pct', 0):.1f}",
                "状态": status,
            })
        st.dataframe(pd.DataFrame(lit_rows), use_container_width=True, hide_index=True)

        # VPC 图
        if vpc_plot_path.exists():
            st.markdown("**VPC 验证图**")
            st.image(str(vpc_plot_path), caption="Visual Predictive Check (90% PI Coverage: 100%)")

        # 报告下载
        if clinical_report_path.exists():
            st.markdown("**FDA/EMA 临床报告**")
            with open(clinical_report_path, 'r', encoding='utf-8') as f:
                report_html = f.read()
            st.download_button(
                "下载 FDA/EMA 临床报告 (HTML)",
                data=report_html,
                file_name="rnactm_clinical_report.html",
                mime="text/html",
            )
    else:
        st.info("PopPK 拟合结果未找到。请先运行 `python core/fit_real_poppk.py` 生成结果。")

run = st.button("开始训练与预测", type="primary")
needs_training = run_mode in {"训练并预测", "仅训练(保存到会话)"}
if run and strict_clinical_mode and needs_training and "efficacy" not in df.columns:
    st.warning("严格临床模式已开启：当前缺少真实标签列 efficacy，将继续训练但结果仅建议用于预筛。")
    _append_clinical_audit(
        event="strict_mode_warn_missing_efficacy",
        payload={
            "run_mode": run_mode,
            "model_backend": model_backend,
            "rows": int(len(df)),
            "has_efficacy": False,
        },
    )

if run and strict_clinical_mode and needs_training and "efficacy" in df.columns:
    _append_clinical_audit(
        event="strict_mode_training_started",
        payload={
            "run_mode": run_mode,
            "model_backend": model_backend,
            "rows": int(len(df)),
            "has_efficacy": True,
        },
    )

legacy_cfg = LegacyAlgorithmConfig(
    epochs=int(legacy_epochs),
    batch_size=int(legacy_batch),
    lr=float(legacy_lr),
    torch_hidden_1=int(legacy_h1),
    torch_hidden_2=int(legacy_h2),
    max_len=int(legacy_max_len),
    emb_dim=int(legacy_emb),
    n_heads=int(legacy_heads),
    n_layers=int(legacy_layers),
)
if run and run_mode == "训练并预测":
    # Determine cloud mode
    _cloud_mode = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
    if _cloud_mode:
        _cloud_client = create_cloud_client("remote", cloud_server_url)
    with st.spinner("正在训练模型并执行 CTM 动态仿真..." + ("（云服务器模式）" if _cloud_mode else "")):
        if _cloud_mode:
            result_df, curve_df, artifacts, train_report = _cloud_client.train_and_predict(
                df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                dynamics_model=dynamics_model,
                legacy_cfg=legacy_cfg,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )
        elif model_backend == "moe":
            trained_model = train_drug_model(
                df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                dynamics_model=dynamics_model,
                legacy_cfg=legacy_cfg,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
                tune_hyperparams=tune_hyperparams,
                tune_strategy=tune_strategy,
                tune_n_iter=tune_n_iter,
                tune_cv=tune_cv,
            )
            st.session_state["drug_trained_model"] = trained_model
            result_df, curve_df, artifacts, train_report = predict_drug_with_model(
                df=df,
                trained_model=trained_model,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )
        else:
            result_df, curve_df, artifacts, train_report = train_and_predict_drug(
                df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                dynamics_model=dynamics_model,
                legacy_cfg=legacy_cfg,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )
    st.session_state["drug_main_result"] = (result_df, curve_df, artifacts, train_report)
    dynamic_metrics = _summarize_dynamic_metrics(result_df)
    _append_experiment_log(
        module="drug-main",
        config={
            "compute_mode": compute_mode,
            "model_backend": model_backend,
            "rows": int(len(df)),
            "adaptive_enabled": bool(adaptive_enabled),
            "adaptive_strength": float(adaptive_strength),
        },
        metrics={
            "efficacy_mae": float(train_report.metrics.get("efficacy_mae", 0.0)),
            "efficacy_rmse": float(train_report.metrics.get("efficacy_rmse", 0.0)),
            "efficacy_r2": float(train_report.metrics.get("efficacy_r2", 0.0)),
            "ctm_peak_efficacy": float(dynamic_metrics.get("ctm_peak_efficacy", 0.0)),
            "ctm_auc_efficacy": float(dynamic_metrics.get("ctm_auc_efficacy", 0.0)),
        },
    )
    _save_repro_bundle(
        module="drug-main",
        data_df=df,
        config={
            "compute_mode": compute_mode,
            "model_backend": model_backend,
            "dynamics_model": dynamics_model,
            "rows": int(len(df)),
            "adaptive_enabled": bool(adaptive_enabled),
            "adaptive_strength": float(adaptive_strength),
        },
        metrics={
            "efficacy_mae": float(train_report.metrics.get("efficacy_mae", 0.0)),
            "efficacy_rmse": float(train_report.metrics.get("efficacy_rmse", 0.0)),
            "efficacy_r2": float(train_report.metrics.get("efficacy_r2", 0.0)),
            "ctm_peak_efficacy": float(dynamic_metrics.get("ctm_peak_efficacy", 0.0)),
            "ctm_auc_efficacy": float(dynamic_metrics.get("ctm_auc_efficacy", 0.0)),
        },
    )

    if enable_credible_eval:
        # 构建预训练编码器配置
        _encoder_spec = MixedFeatureSpec(
            use_gnn=use_gnn,
            use_chemberta=use_chemberta,
            use_esm2=use_esm2_drug,
            use_pk_prior=use_pk_prior,
            use_dose_response=use_dose_response,
            use_cross_features=use_cross_features,
            cache_dir=cache_dir,
            smiles_hash_dim=128,
            smiles_rdkit_bits=2048,
            smiles_rdkit_version=2,
            prefer_rdkit=True,
        )
        with st.spinner("正在执行严格切分 + CV + 基线对照评估..."):
            credible = core_credible_eval_drug(
                df=df,
                backend=model_backend,
                compute_mode=compute_mode,
                seed=int(credible_seed),
                test_ratio=float(credible_test_ratio),
                val_ratio=float(credible_val_ratio),
                cv_folds=int(credible_cv_folds),
                top_n_failures=int(credible_top_fail_n),
                external_df=external_eval_df,
                feature_spec=_encoder_spec,
            )
        st.session_state["drug_credible_eval"] = credible
        _append_experiment_log(
            module="drug-credible-eval",
            config={
                "backend": model_backend,
                "seed": int(credible_seed),
                "test_ratio": float(credible_test_ratio),
                "val_ratio": float(credible_val_ratio),
                "cv_folds": int(credible_cv_folds),
            },
            metrics={
                "enabled": float(1.0 if credible.get("enabled", False) else 0.0),
                "test_rmse": float(credible.get("test_metrics", {}).get("rmse", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_mean": float(credible.get("cv_summary", {}).get("rmse_mean", 0.0)) if credible.get("enabled", False) else 0.0,
            },
        )
        _save_repro_bundle(
            module="drug-credible-eval",
            data_df=df,
            config={
                "backend": model_backend,
                "seed": int(credible_seed),
                "test_ratio": float(credible_test_ratio),
                "val_ratio": float(credible_val_ratio),
                "cv_folds": int(credible_cv_folds),
            },
            metrics={
                "enabled": float(1.0 if credible.get("enabled", False) else 0.0),
                "test_rmse": float(credible.get("test_metrics", {}).get("rmse", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_mean": float(credible.get("cv_summary", {}).get("rmse_mean", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_ci95": float(credible.get("cv_summary", {}).get("rmse_ci95", 0.0)) if credible.get("enabled", False) else 0.0,
            },
        )

if run and run_mode == "仅训练(保存到会话)":
    if model_backend != "moe":
        st.error("仅训练模式当前仅支持 Confluencia 2.0 MOE。")
    else:
        _cloud_mode = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
        if _cloud_mode:
            _cloud_client = create_cloud_client("remote", cloud_server_url)
        with st.spinner("正在训练模型并保存到会话..." + ("（云服务器模式）" if _cloud_mode else "")):
            if _cloud_mode:
                trained_model = _cloud_client.train(
                    df,
                    compute_mode=compute_mode,
                    model_backend=model_backend,
                    dynamics_model=dynamics_model,
                    legacy_cfg=legacy_cfg,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
            else:
                trained_model = train_drug_model(
                    df,
                    compute_mode=compute_mode,
                    model_backend=model_backend,
                    dynamics_model=dynamics_model,
                    legacy_cfg=legacy_cfg,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
        st.session_state["drug_trained_model"] = trained_model
        st.success("训练完成，模型已保存到当前会话，可直接切换到\"仅预测(使用最近模型)\"。" + ("（云模式：模型存储在服务器）" if _cloud_mode else ""))

if run and run_mode == "仅预测(使用最近模型)":
    trained_model = st.session_state.get("drug_trained_model")
    if trained_model is None:
        st.error("未检测到会话模型。请先执行\"仅训练(保存到会话)\"或\"训练并预测\"。")
    else:
        _cloud_mode = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
        if _cloud_mode:
            _cloud_client = create_cloud_client("remote", cloud_server_url)
        with st.spinner("正在使用最近模型执行预测..." + ("（云服务器模式）" if _cloud_mode else "")):
            if _cloud_mode:
                result_df, curve_df, artifacts, train_report = _cloud_client.predict(
                    df=df,
                    trained_model=trained_model,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
            else:
                result_df, curve_df, artifacts, train_report = predict_drug_with_model(
                    df=df,
                    trained_model=trained_model,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
        st.session_state["drug_main_result"] = (result_df, curve_df, artifacts, train_report)
        st.success("预测完成（未重训模型）。")

if run and do_benchmark and run_mode == "训练并预测":
    _cloud_mode = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
    if _cloud_mode:
        st.info("云服务器模式下算法对比将通过网络执行，可能较慢。")
    if strict_clinical_mode and "efficacy" not in df.columns:
        st.warning("严格临床模式已开启：算法对比缺少 efficacy，指标仅供参考。")
    with st.spinner("正在执行算法排行榜对比..." + ("（云服务器模式）" if _cloud_mode else "")):
        compare_backends = ["hgb", "gbr", "rf", "ridge", "mlp", "torch_mlp", "transformer"]
        rows = []
        p = st.progress(0, text="算法对比进行中")
        total = max(len(compare_backends), 1)
        _cloud_client = create_cloud_client("remote", cloud_server_url) if _cloud_mode else None
        for i, b in enumerate(compare_backends):
            if _cloud_mode:
                result_df_b, _, _, report_b = _cloud_client.train_and_predict(
                    df,
                    compute_mode=compute_mode,
                    model_backend=b,
                    dynamics_model=dynamics_model,
                    legacy_cfg=legacy_cfg,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
            else:
                result_df_b, _, _, report_b = train_and_predict_drug(
                    df,
                    compute_mode=compute_mode,
                    model_backend=b,
                    dynamics_model=dynamics_model,
                    legacy_cfg=legacy_cfg,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_strength=adaptive_strength,
                )
            rows.append(
                {
                    "backend": b,
                    "mae": float(report_b.metrics.get("efficacy_mae", 0.0)),
                    "rmse": float(report_b.metrics.get("efficacy_rmse", 0.0)),
                    "r2": float(report_b.metrics.get("efficacy_r2", 0.0)),
                    "samples": int(len(result_df_b)),
                }
            )
            p.progress(int((i + 1) * 100 / total), text=f"已完成: {b}")
        p.empty()
        lb_df = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=[True, True])
        st.session_state["drug_leaderboard"] = lb_df
        _append_experiment_log(
            module="drug-benchmark",
            config={"backends": compare_backends, "compute_mode": compute_mode, "rows": int(len(df))},
            metrics={"top_backend": str(lb_df.iloc[0]["backend"]) if not lb_df.empty else "none"},
        )
        _save_repro_bundle(
            module="drug-benchmark",
            data_df=df,
            config={"backends": compare_backends, "compute_mode": compute_mode, "rows": int(len(df))},
            metrics={"top_backend": str(lb_df.iloc[0]["backend"]) if not lb_df.empty else "none"},
        )

if run and run_mode == "仅查看最近结果":
    st.info("已选择\"仅查看最近结果\"，未触发新的训练。")

if "drug_leaderboard" in st.session_state:
    st.subheader("算法排行榜")
    lb_df = st.session_state["drug_leaderboard"]
    st.dataframe(lb_df, use_container_width=True)
    if not lb_df.empty:
        st.bar_chart(lb_df.set_index("backend")[["rmse", "mae"]])

if "drug_main_result" in st.session_state:
    result_df, curve_df, artifacts, train_report = st.session_state["drug_main_result"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", f"{len(result_df)}")
    c2.metric("计算档位", artifacts.compute_profile)
    c3.metric("算法后端", artifacts.model_backend)
    c4.metric("代理微标签模式", "是" if artifacts.used_proxy_micro_labels else "否")

    c5, c6, c7 = st.columns(3)
    c5.metric("SMILES 特征", artifacts.smiles_backend)
    c6.metric("CTM 参数来源", artifacts.ctm_param_source)
    c7.metric("动力学后端", artifacts.dynamics_model)
    st.caption(
        "自适应系统: "
        + ("已启用" if getattr(artifacts, "adaptive_enabled", False) else "未启用")
        + f" | 强度: {float(getattr(artifacts, 'adaptive_strength', 0.0)):.2f}"
        + f" | 样本: {int(getattr(artifacts, 'adaptive_samples', 0))}"
        + f" | 状态: {str(getattr(artifacts, 'adaptive_message', 'disabled'))}"
    )
    st.caption(
        f"SHAP 状态: {'已启用' if getattr(artifacts, 'shap_ready', False) else '未启用'}"
        + f" | 特征数: {int(getattr(artifacts, 'shap_feature_count', 0))}"
        + f" | 说明: {str(getattr(artifacts, 'shap_message', 'n/a'))}"
    )

    st.subheader("训练模块报告")
    if train_report.used_labels.get("efficacy", False):
        t1, t2, t3 = st.columns(3)
        t1.metric("疗效 MAE", f"{train_report.metrics.get('efficacy_mae', 0.0):.4f}")
        t2.metric("疗效 RMSE", f"{train_report.metrics.get('efficacy_rmse', 0.0):.4f}")
        t3.metric("疗效 R2", f"{train_report.metrics.get('efficacy_r2', 0.0):.4f}")
    else:
        st.warning("当前数据未提供真实 efficacy 标签。已隐藏 MAE/RMSE/R2，结果仅用于筛选/排序，不可作为最终结论。")

    st.caption("微机制标签评估（有真实标签时有效）")
    tm1, tm2, tm3 = st.columns(3)
    tm1.metric("炎症风险 MAE", f"{train_report.metrics.get('inflammation_risk_mae', 0.0):.4f}")
    tm2.metric("毒性风险 MAE", f"{train_report.metrics.get('toxicity_risk_mae', 0.0):.4f}")
    tm3.metric("免疫细胞激活 MAE", f"{train_report.metrics.get('immune_cell_activation_mae', 0.0):.4f}")

    if "drug_credible_eval" in st.session_state:
        st.subheader("科研可信评估（严格切分）")
        ce = st.session_state["drug_credible_eval"]
        if not ce.get("enabled", False):
            reason = str(ce.get("reason", "unknown"))
            if reason == "missing_label":
                st.warning("未检测到真实 efficacy 标签，已跳过严格可信评估。")
            elif reason == "too_few_samples":
                st.warning("样本量不足，无法稳定执行 train/val/test + 5-fold 评估。")
            else:
                st.warning(f"可信评估未启用: {reason}")
        else:
            if not ce.get("backend_supported", True):
                st.info(f"当前后端在可信评估中暂不支持，已使用 {ce.get('backend_used', 'hgb')} 代理评估。")

            sp = ce.get("split_sizes", {})
            s1, s2, s3 = st.columns(3)
            s1.metric("Train", str(sp.get("train", 0)))
            s2.metric("Val", str(sp.get("val", 0)))
            s3.metric("Test", str(sp.get("test", 0)))

            test_m = ce.get("test_metrics", {})
            cv_s = ce.get("cv_summary", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{float(test_m.get('rmse', 0.0)):.4f}")
            c2.metric("CV RMSE 均值", f"{float(cv_s.get('rmse_mean', 0.0)):.4f}")
            c3.metric("CV RMSE 95%CI", f"+/- {float(cv_s.get('rmse_ci95', 0.0)):.4f}")

            st.caption("5-fold 交叉验证汇总（均值 ± 标准差）")
            cv_table = pd.DataFrame(
                [
                    {
                        "metric": "MAE",
                        "mean": float(cv_s.get("mae_mean", 0.0)),
                        "std": float(cv_s.get("mae_std", 0.0)),
                        "ci95": float(cv_s.get("mae_ci95", 0.0)),
                    },
                    {
                        "metric": "RMSE",
                        "mean": float(cv_s.get("rmse_mean", 0.0)),
                        "std": float(cv_s.get("rmse_std", 0.0)),
                        "ci95": float(cv_s.get("rmse_ci95", 0.0)),
                    },
                    {
                        "metric": "R2",
                        "mean": float(cv_s.get("r2_mean", 0.0)),
                        "std": float(cv_s.get("r2_std", 0.0)),
                        "ci95": float(cv_s.get("r2_ci95", 0.0)),
                    },
                ]
            )
            st.dataframe(cv_table, use_container_width=True)

            st.caption("固定基线对照（Linear / RF / HGB，测试集）")
            bdf = ce.get("baseline_df", pd.DataFrame())
            if isinstance(bdf, pd.DataFrame) and (not bdf.empty):
                st.dataframe(bdf, use_container_width=True)

            if bool(ce.get("pass_gate", False)):
                st.success("当前后端在测试集 RMSE 上已超过固定基线最佳模型。")
            else:
                st.error("当前后端未超过固定基线最佳模型，建议先优化再宣称改进有效。")

            ic_df = ce.get("interval_calibration_df", pd.DataFrame())
            if isinstance(ic_df, pd.DataFrame) and (not ic_df.empty):
                st.caption("预测区间校准（由验证集残差校准测试集覆盖率）")
                st.dataframe(ic_df, use_container_width=True)

            leak = ce.get("leakage_audit", None)
            if isinstance(leak, dict):
                l1, l2 = st.columns(2)
                l1.metric("Train/Test 精确重叠数", f"{int(float(leak.get('overlap_count', 0.0)))}")
                l2.metric("Train/Test 重叠比例", f"{float(leak.get('overlap_ratio', 0.0)) * 100:.2f}%")

            ext_m = ce.get("external_metrics", None)
            if isinstance(ext_m, dict):
                st.caption("外部独立测试集指标")
                e1, e2, e3 = st.columns(3)
                e1.metric("External MAE", f"{float(ext_m.get('mae', 0.0)):.4f}")
                e2.metric("External RMSE", f"{float(ext_m.get('rmse', 0.0)):.4f}")
                e3.metric("External R2", f"{float(ext_m.get('r2', 0.0)):.4f}")

            fail_df = ce.get("failure_df", pd.DataFrame())
            if isinstance(fail_df, pd.DataFrame) and (not fail_df.empty):
                st.caption("失败样本 Top-N（按绝对误差）")
                show_fail_cols = [c for c in ["smiles", "epitope_seq", "group_id", "dose", "freq", "treatment_time", "y_true", "y_pred", "abs_error"] if c in fail_df.columns]
                st.dataframe(fail_df[show_fail_cols], use_container_width=True)
                fail_buf = io.StringIO()
                fail_df.to_csv(fail_buf, index=False)
                st.download_button(
                    "下载失败样本 CSV",
                    data=fail_buf.getvalue(),
                    file_name="confluencia2_drug_failure_samples.csv",
                    mime="text/csv",
                )

    st.subheader("模型诊断与可解释性")
    col_w, col_m = st.columns(2)
    with col_w:
        w_df = pd.DataFrame({"expert": list(artifacts.moe_weights.keys()), "weight": list(artifacts.moe_weights.values())})
        if not w_df.empty:
            st.bar_chart(w_df.set_index("expert"))
        else:
            st.info("当前算法后端无专家权重输出")
    with col_m:
        m_df = pd.DataFrame({"metric": list(artifacts.moe_metrics.keys()), "value": list(artifacts.moe_metrics.values())})
        st.dataframe(m_df, use_container_width=True)

    st.subheader("SHAP 样本级特征贡献")
    shap_meta_cols = {"shap_base_value", "shap_value_sum", "shap_reconstructed_pred"}
    shap_feature_cols = [c for c in result_df.columns if c.startswith("shap_") and c not in shap_meta_cols]
    if shap_feature_cols:
        sid = st.number_input("选择样本索引", min_value=0, max_value=max(len(result_df) - 1, 0), value=0, step=1)
        sid_int = int(sid)
        srow = result_df.iloc[sid_int]
        contrib_df = pd.DataFrame(
            {
                "feature": [c.replace("shap_", "", 1) for c in shap_feature_cols],
                "contribution": [float(pd.to_numeric(srow[c], errors="coerce")) for c in shap_feature_cols],
            }
        )
        contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
        st.caption("当前样本贡献 Top-30（正值提升预测分数，负值降低预测分数）")
        st.dataframe(contrib_df.sort_values("abs_contribution", ascending=False).head(30), use_container_width=True)

        with st.expander("导出全部样本-特征 SHAP 明细", expanded=False):
            shap_mat = result_df[shap_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            long_df = shap_mat.copy()
            long_df.insert(0, "sample_id", np.arange(len(long_df), dtype=np.int32))
            long_df = long_df.melt(id_vars=["sample_id"], var_name="feature", value_name="contribution")
            long_df["feature"] = long_df["feature"].str.replace("^shap_", "", regex=True)
            st.dataframe(long_df.head(200), use_container_width=True)
            shap_buf = io.StringIO()
            long_df.to_csv(shap_buf, index=False)
            st.download_button(
                "下载 SHAP 全量明细 CSV",
                data=shap_buf.getvalue(),
                file_name="confluencia2_shap_contributions.csv",
                mime="text/csv",
            )
    else:
        st.info("当前结果未包含 SHAP 特征贡献。若使用非 MOE 后端或环境未安装 shap，将不生成该明细。")

    st.subheader("微观层预测结果")
    micro_cols = [
        "efficacy_pred",
        "target_binding_pred",
        "immune_activation_pred",
        "immune_cell_activation_pred",
        "inflammation_risk_pred",
        "toxicity_risk_pred",
        "immune_peak_antibody",
        "immune_peak_effector_t",
        "immune_peak_antigen",
        "immune_response_auc",
        "ctm_ka",
        "ctm_kd",
        "ctm_ke",
        "ctm_km",
        "ctm_signal_gain",
        "ctm_ka_base",
        "ctm_kd_base",
        "ctm_ke_base",
        "ctm_km_base",
        "ctm_signal_gain_base",
        "ctm_ka_offset",
        "ctm_kd_offset",
        "ctm_ke_offset",
        "ctm_km_offset",
        "ctm_signal_gain_offset",
        "ctm_auc_efficacy",
        "ctm_peak_efficacy",
        "ctm_peak_toxicity",
        "pkpd_half_life_h",
        "pkpd_vd_ss_l",
        "pkpd_clearance_lph",
        "pkpd_cmax_mg_per_l",
        "pkpd_tmax_h",
        "pkpd_auc_conc",
        "pkpd_auc_effect",
        "pkpd_effect_peak",
        "pkpd_pk_effect_corr",
    ]
    show_cols = [c for c in ["smiles", "epitope_seq", "group_id", *micro_cols, "cross_group_impact"] if c in result_df.columns]
    st.dataframe(result_df[show_cols].head(80), use_container_width=True)

    st.subheader("自适应给药建议")
    has_adaptive_cols = all(c in result_df.columns for c in ["adaptive_dose_factor", "adaptive_freq_factor"])
    if has_adaptive_cols and all(c in result_df.columns for c in ["dose", "freq"]):
        rec_df = result_df.copy()
        rec_df["dose"] = pd.to_numeric(rec_df["dose"], errors="coerce").fillna(0.0)
        rec_df["freq"] = pd.to_numeric(rec_df["freq"], errors="coerce").fillna(0.0)
        rec_df["adaptive_dose_factor"] = pd.to_numeric(rec_df["adaptive_dose_factor"], errors="coerce").fillna(1.0)
        rec_df["adaptive_freq_factor"] = pd.to_numeric(rec_df["adaptive_freq_factor"], errors="coerce").fillna(1.0)

        rec_df["recommended_dose"] = np.clip(rec_df["dose"] * rec_df["adaptive_dose_factor"], 0.0, np.inf)
        rec_df["recommended_freq"] = np.clip(rec_df["freq"] * rec_df["adaptive_freq_factor"], 0.0, np.inf)
        rec_df["dose_change_pct"] = np.where(
            rec_df["dose"] > 1e-8,
            (rec_df["recommended_dose"] / rec_df["dose"] - 1.0) * 100.0,
            0.0,
        )
        rec_df["freq_change_pct"] = np.where(
            rec_df["freq"] > 1e-8,
            (rec_df["recommended_freq"] / rec_df["freq"] - 1.0) * 100.0,
            0.0,
        )

        risk_gate = np.maximum(
            pd.to_numeric(rec_df.get("toxicity_risk_pred", 0.0), errors="coerce").fillna(0.0),
            pd.to_numeric(rec_df.get("inflammation_risk_pred", 0.0), errors="coerce").fillna(0.0),
        )
        rec_df["adaptive_gate_flag"] = np.where(risk_gate >= 0.70, "review", "ok")

        for col in [
            "recommended_dose",
            "recommended_freq",
            "dose_change_pct",
            "freq_change_pct",
            "adaptive_gate_flag",
        ]:
            result_df[col] = rec_df[col].to_numpy()

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("建议剂量均值", f"{float(rec_df['recommended_dose'].mean()):.4f}")
        g2.metric("建议频次均值", f"{float(rec_df['recommended_freq'].mean()):.4f}")
        g3.metric("剂量平均变化", f"{float(rec_df['dose_change_pct'].mean()):+.2f}%")
        g4.metric("频次平均变化", f"{float(rec_df['freq_change_pct'].mean()):+.2f}%")

        recommend_cols = [
            c
            for c in [
                "smiles",
                "group_id",
                "dose",
                "freq",
                "recommended_dose",
                "recommended_freq",
                "dose_change_pct",
                "freq_change_pct",
                "adaptive_confidence",
                "adaptive_risk_pressure",
                "adaptive_gate_flag",
            ]
            if c in rec_df.columns
        ]
        st.caption("说明：adaptive_gate_flag=review 表示风险较高，建议人工复核后再采用建议给药参数。")
        st.dataframe(rec_df[recommend_cols].head(80), use_container_width=True)
    else:
        st.info("当前结果未包含自适应建议系数。请在侧边栏开启\"启用自适应校准\"后重跑。")

    st.subheader("风险-收益可视化")
    viz_df = result_df.copy()
    viz_df["benefit_risk_score"] = (
        0.45 * pd.to_numeric(viz_df.get("efficacy_pred", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(viz_df.get("target_binding_pred", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(viz_df.get("immune_cell_activation_pred", 0.0), errors="coerce").fillna(0.0)
        - 0.10 * pd.to_numeric(viz_df.get("inflammation_risk_pred", 0.0), errors="coerce").fillna(0.0)
        - 0.05 * pd.to_numeric(viz_df.get("toxicity_risk_pred", 0.0), errors="coerce").fillna(0.0)
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("平均疗效", f"{float(viz_df['efficacy_pred'].mean()):.4f}" if "efficacy_pred" in viz_df.columns else "0.0000")
    k2.metric("平均炎症风险", f"{float(viz_df['inflammation_risk_pred'].mean()):.4f}" if "inflammation_risk_pred" in viz_df.columns else "0.0000")
    k3.metric("平均毒性风险", f"{float(viz_df['toxicity_risk_pred'].mean()):.4f}" if "toxicity_risk_pred" in viz_df.columns else "0.0000")
    k4.metric("平均免疫细胞激活", f"{float(viz_df['immune_cell_activation_pred'].mean()):.4f}" if "immune_cell_activation_pred" in viz_df.columns else "0.0000")

    if all(c in viz_df.columns for c in ["toxicity_risk_pred", "efficacy_pred", "immune_cell_activation_pred"]):
        if "inflammation_risk_pred" in viz_df.columns:
            st.scatter_chart(
                viz_df,
                x="toxicity_risk_pred",
                y="efficacy_pred",
                size="immune_cell_activation_pred",
                color="inflammation_risk_pred",
            )
        else:
            st.scatter_chart(
                viz_df,
                x="toxicity_risk_pred",
                y="efficacy_pred",
                size="immune_cell_activation_pred",
            )

    rank_cols = [c for c in ["smiles", "group_id", "benefit_risk_score", "efficacy_pred", "immune_cell_activation_pred", "inflammation_risk_pred", "toxicity_risk_pred"] if c in viz_df.columns]
    st.caption("推荐候选（按收益-风险分数排序）")
    st.dataframe(viz_df.sort_values("benefit_risk_score", ascending=False)[rank_cols].head(15), use_container_width=True)

    if "group_id" in result_df.columns and "cross_group_impact" in result_df.columns:
        st.subheader("跨组相互作用")
        gdf = result_df.groupby("group_id", as_index=False)["cross_group_impact"].mean()
        st.bar_chart(gdf.set_index("group_id"))

    st.subheader("CTM 动态轨迹")
    if not curve_df.empty:
        sample_ids = sorted(curve_df["sample_id"].unique().tolist())
        sid = st.selectbox("选择样本 ID", sample_ids, index=0)
        c = curve_df[curve_df["sample_id"] == sid]
        st.line_chart(
            c.set_index("time_h")[["absorption_A", "distribution_D", "effect_E", "metabolism_M", "efficacy_signal", "toxicity_signal"]]
        )

        with st.expander("CTM 曲线解释", expanded=True):
            eff_peak = float(c["efficacy_signal"].max()) if "efficacy_signal" in c.columns and not c.empty else 0.0
            tox_peak = float(c["toxicity_signal"].max()) if "toxicity_signal" in c.columns and not c.empty else 0.0
            t_eff = float(c.loc[c["efficacy_signal"].idxmax(), "time_h"]) if "efficacy_signal" in c.columns and not c.empty else 0.0
            t_tox = float(c.loc[c["toxicity_signal"].idxmax(), "time_h"]) if "toxicity_signal" in c.columns and not c.empty else 0.0
            if all(x in c.columns for x in ["time_h", "efficacy_signal"]) and not c.empty:
                trap = getattr(np, "trapezoid", None)
                auc_eff = float(
                    trap(c["efficacy_signal"].to_numpy(dtype=np.float32), c["time_h"].to_numpy(dtype=np.float32))
                    if callable(trap)
                    else np.trapz(c["efficacy_signal"].to_numpy(dtype=np.float32), c["time_h"].to_numpy(dtype=np.float32))
                )
            else:
                auc_eff = 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("疗效峰值", f"{eff_peak:.4f}")
            m2.metric("毒性峰值", f"{tox_peak:.4f}")
            m3.metric("疗效 AUC", f"{auc_eff:.4f}")

            st.markdown(
                f"""
                - `A (absorption_A)`：吸收阶段，数值上升快通常代表起效更快。
                - `D (distribution_D)`：分布阶段，体现药物在体系中的扩散与可达性。
                - `E (effect_E)`：效应阶段，常与 `efficacy_signal` 的上升相关。
                - `M (metabolism_M)`：代谢阶段，越高通常代表清除更快、作用持续性可能降低。
                - `efficacy_signal`：综合疗效信号，当前样本在 `t={t_eff:.1f}h` 达到峰值。
                - `toxicity_signal`：综合毒性信号，当前样本在 `t={t_tox:.1f}h` 达到峰值。
                """
            )

            if tox_peak > eff_peak:
                st.warning("该样本毒性峰值高于疗效峰值，建议降低剂量或频次并重新评估。")
            elif t_tox < t_eff:
                st.info("该样本毒性先于疗效达峰，建议关注早期给药窗口。")
            else:
                st.success("该样本疗效曲线整体优于毒性曲线，CTM 动态表现相对稳健。")

        with st.expander("CTM 参数敏感性（±10%）", expanded=False):
            rid = int(sid or 0) if len(result_df) > 0 else 0
            rid = max(0, min(rid, len(result_df) - 1)) if len(result_df) > 0 else 0
            row0 = result_df.iloc[rid] if len(result_df) > 0 else pd.Series(dtype=float)

            base_params = CTMParams(
                ka=float(row0.get("ctm_ka", 0.2)),
                kd=float(row0.get("ctm_kd", 0.2)),
                ke=float(row0.get("ctm_ke", 0.2)),
                km=float(row0.get("ctm_km", 0.1)),
                signal_gain=float(row0.get("ctm_signal_gain", 1.0)),
            )
            base_curve = simulate_ctm(
                dose=float(row0.get("dose", 0.0)),
                freq=float(row0.get("freq", 1.0)),
                params=base_params,
                horizon=72,
                dt=1.0,
            )
            base_sum = summarize_curve(base_curve)

            sens_rows = []
            for pname in ["ka", "kd", "ke", "km", "signal_gain"]:
                for scale in [0.9, 1.1]:
                    p = CTMParams(**base_params.__dict__)
                    setattr(p, pname, float(getattr(p, pname)) * float(scale))
                    cv = simulate_ctm(
                        dose=float(row0.get("dose", 0.0)),
                        freq=float(row0.get("freq", 1.0)),
                        params=p,
                        horizon=72,
                        dt=1.0,
                    )
                    sm = summarize_curve(cv)
                    sens_rows.append(
                        {
                            "参数": pname,
                            "扰动": f"{int((scale - 1.0) * 100)}%",
                            "AUC变化": float(sm["auc_efficacy"] - base_sum["auc_efficacy"]),
                            "疗效峰值变化": float(sm["peak_efficacy"] - base_sum["peak_efficacy"]),
                            "毒性峰值变化": float(sm["peak_toxicity"] - base_sum["peak_toxicity"]),
                        }
                    )
            sens_df = pd.DataFrame(sens_rows)
            st.dataframe(sens_df, use_container_width=True)

        if all(x in c.columns for x in ["pkpd_conc_mg_per_l", "pkpd_effect"]):
            st.subheader("PK/PD（SciPy 简化房室模型）")
            c_pk = c.copy()

            if len(result_df) > 0:
                rid = int(sid or 0)
                rid = max(0, min(rid, len(result_df) - 1))
                row_pk = result_df.iloc[rid]

                if bool(pkpd_override_enabled):
                    if bool(pkpd_use_manual_params):
                        rt_params = PKPDParams(
                            ka=float(pkpd_manual_ka),
                            k12=float(pkpd_manual_k12),
                            k21=float(pkpd_manual_k21),
                            ke=float(pkpd_manual_ke),
                            v1_l=float(pkpd_manual_v1),
                            emax=float(pkpd_manual_emax),
                            ec50_mg_per_l=float(pkpd_manual_ec50),
                            hill=float(pkpd_manual_hill),
                        )
                    else:
                        rt_params = infer_pkpd_params(
                            binding=float(row_pk.get("target_binding_pred", 0.5)),
                            immune=float(row_pk.get("immune_activation_pred", 0.5)),
                            inflammation=float(row_pk.get("inflammation_risk_pred", 0.2)),
                            dose_mg=float(row_pk.get("dose", 0.0)),
                            freq_per_day=float(row_pk.get("freq", 1.0)),
                        )

                    c_pk = simulate_pkpd(
                        dose_mg=float(row_pk.get("dose", 0.0)),
                        freq_per_day=float(row_pk.get("freq", 1.0)),
                        params=rt_params,
                        horizon=int(pkpd_horizon),
                        dt=float(pkpd_dt),
                    )
                    pk_sum = summarize_pkpd_curve(c_pk, rt_params)
                    pk_half_life_v = float(pk_sum.get("pkpd_half_life_h", 0.0))
                    pk_vd_ss_v = float(pk_sum.get("pkpd_vd_ss_l", 0.0))
                    pk_cl_v = float(pk_sum.get("pkpd_clearance_lph", 0.0))
                    pk_cmax_v = float(pk_sum.get("pkpd_cmax_mg_per_l", 0.0))
                    pk_tmax_v = float(pk_sum.get("pkpd_tmax_h", 0.0))
                    pk_auc_c_v = float(pk_sum.get("pkpd_auc_conc", 0.0))
                    pk_auc_e_v = float(pk_sum.get("pkpd_auc_effect", 0.0))
                    pk_corr_v = float(pk_sum.get("pkpd_pk_effect_corr", 0.0))
                else:
                    pk_half_life_v = float(row_pk.get("pkpd_half_life_h", 0.0))
                    pk_vd_ss_v = float(row_pk.get("pkpd_vd_ss_l", 0.0))
                    pk_cl_v = float(row_pk.get("pkpd_clearance_lph", 0.0))
                    pk_cmax_v = float(row_pk.get("pkpd_cmax_mg_per_l", 0.0))
                    pk_tmax_v = float(row_pk.get("pkpd_tmax_h", 0.0))
                    pk_auc_c_v = float(row_pk.get("pkpd_auc_conc", 0.0))
                    pk_auc_e_v = float(row_pk.get("pkpd_auc_effect", 0.0))
                    pk_corr_v = float(row_pk.get("pkpd_pk_effect_corr", 0.0))

                st.line_chart(c_pk.set_index("time_h")[["pkpd_conc_mg_per_l", "pkpd_effect"]])

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("终末半衰期 (h)", f"{pk_half_life_v:.2f}")
                p2.metric("分布容积 Vdss (L)", f"{pk_vd_ss_v:.2f}")
                p3.metric("清除率 CL (L/h)", f"{pk_cl_v:.2f}")
                p4.metric("Cmax (mg/L)", f"{pk_cmax_v:.2f}")

                p5, p6, p7, p8 = st.columns(4)
                p5.metric("Tmax (h)", f"{pk_tmax_v:.2f}")
                p6.metric("浓度 AUC", f"{pk_auc_c_v:.2f}")
                p7.metric("药效 AUC", f"{pk_auc_e_v:.2f}")
                p8.metric("PK-PD 相关", f"{pk_corr_v:.3f}")

                st.caption("说明：该模块为简化 PK/PD 估算（吸收-中央-外周房室 + Emax 药效），用于方案早期比较，不替代正式群体药代分析。启用\"实时 PK/PD 重算\"后会使用当前面板参数即时更新。")
    else:
        st.info("当前未生成 CTM 曲线")

    # ------------------------------------------------------------------
    # circRNA v2.2: Innate immune assessment + RNA CTM
    # ------------------------------------------------------------------
    _crna_mode = st.session_state.get("crna_mode", False)
    if _crna_mode and len(result_df) > 0:
        _crna_mod = st.session_state.get("crna_mod", "none")
        _crna_del = st.session_state.get("crna_del", "LNP_standard")
        _crna_route = st.session_state.get("crna_route", "IV")
        _crna_ires = st.session_state.get("crna_ires", "")

        # Ensure circRNA columns exist
        result_df = ensure_cirrna_columns(result_df)

        st.subheader("先天免疫评估（circRNA v2.2）")
        with st.spinner("正在评估先天免疫激活..."):
            innate_results = batch_assess_innate_immune(result_df)
            if innate_results:
                innate_df_display = pd.DataFrame(innate_results)
                # Show key metrics as a bar chart
                if "innate_immune_score" in innate_df_display.columns:
                    innate_df_display.index = result_df.index[:len(innate_df_display)]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("平均先天免疫评分", f"{innate_df_display['innate_immune_score'].mean():.3f}")
                    c2.metric("平均安全评分", f"{innate_df_display['innate_safety_score'].mean():.3f}")
                    c3.metric("IFN 风险 (medium+)", f"{(innate_df_display['innate_ifn_storm_level'] != 'low').sum()}/{len(innate_df_display)}")

                    with st.expander("先天免疫通路详情", expanded=False):
                        pathway_cols = ["innate_tlr3", "innate_tlr7", "innate_tlr8", "innate_rigi", "innate_mda5", "innate_pkr"]
                        avail_cols = [c for c in pathway_cols if c in innate_df_display.columns]
                        if avail_cols:
                            st.bar_chart(innate_df_display[avail_cols])
                        st.dataframe(innate_df_display, use_container_width=True)

        st.subheader("RNA PK 仿真（circRNA 六室模型 v2.2）")
        with st.spinner("正在运行 RNA 药代动力学仿真..."):
            # Use first sample with circRNA data or defaults
            sample_idx = 0
            for i, (_, row) in enumerate(result_df.iterrows()):
                seq = str(row.get("circrna_seq", ""))
                if seq and len(seq) > 10:
                    sample_idx = i
                    break

            crna_seq = str(result_df.iloc[sample_idx].get("circrna_seq", "")) if len(result_df) > 0 else ""
            crna_dose = float(result_df.iloc[sample_idx].get("dose", 2.0)) if len(result_df) > 0 else 2.0
            crna_freq = float(result_df.iloc[sample_idx].get("freq", 1.0)) if len(result_df) > 0 else 1.0

            if crna_seq and len(crna_seq) > 10:
                from core.features import encode_cirrna_structure, encode_cirrna_functional
                struct_feat = encode_cirrna_structure(crna_seq)
                func_feat = encode_cirrna_functional(crna_seq, ires_type=_crna_ires)
                ires_score = func_feat[0]
                gc_content = sum(1 for c in crna_seq.upper() if c in "GC") / max(len(crna_seq), 1)
                struct_stability = struct_feat[2]

                innate_single = assess_innate_immune(crna_seq, _crna_mod, _crna_del)
                innate_score = innate_single.innate_immune_score

                rna_params = infer_rna_ctm_params(
                    modification=_crna_mod, delivery_vector=_crna_del, route=_crna_route,
                    ires_score=ires_score, gc_content=gc_content,
                    struct_stability=struct_stability, innate_immune_score=innate_score,
                )
                rna_curve = simulate_rna_ctm(crna_dose, crna_freq, rna_params, horizon=168, dt=1.0)
                rna_summary = summarize_rna_ctm_curve(rna_curve)

                if not rna_curve.empty:
                    plot_cols = ["rna_cytoplasmic", "protein_translated", "rna_circulating_total",
                                "tissue_liver", "tissue_spleen", "efficacy_signal"]
                    avail_plot = [c for c in plot_cols if c in rna_curve.columns]
                    if avail_plot:
                        st.line_chart(rna_curve.set_index("time_h")[avail_plot])

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("蛋白峰值", f"{rna_summary['rna_ctm_peak_protein']:.3f}")
                    m2.metric("表达窗口", f"{rna_summary['rna_ctm_protein_expression_window_h']:.0f}h")
                    m3.metric("RNA 半衰期", f"{rna_summary['rna_ctm_rna_half_life_h']:.1f}h")
                    m4.metric("生物利用度", f"{rna_summary['rna_ctm_bioavailability_frac']:.1%}")

                    with st.expander("RNA PK 参数详情", expanded=False):
                        st.json({
                            "k_release": rna_params.k_release,
                            "k_escape": rna_params.k_escape,
                            "k_translate": rna_params.k_translate,
                            "k_degrade": rna_params.k_degrade,
                            "k_protein_half_h": rna_params.k_protein_half,
                            "k_immune_clear": rna_params.k_immune_clear,
                            "f_liver": rna_params.f_liver,
                            "f_spleen": rna_params.f_spleen,
                            "f_muscle": rna_params.f_muscle,
                        })

                    if innate_single.interferon_storm_level in ("high", "critical"):
                        st.error("先天免疫评估：该 circRNA 序列存在较高的 IFN 风险（%s），建议优化修饰或序列。" % innate_single.interferon_storm_level)
                    elif innate_single.interferon_storm_level == "medium":
                        st.warning("先天免疫评估：IFN 风险中等，建议考虑添加 Psi 修饰以降低免疫原性。")
                    else:
                        st.success("先天免疫评估：IFN 风险低，序列安全性良好。")
            else:
                st.info("当前数据无 circRNA 序列输入。如需运行 RNA PK 仿真，请在上传数据中包含 circrna_seq 列。")

    st.subheader("免疫 ABM 动态轨迹（APC/T/B/抗体）")
    if len(result_df) > 0:
        abm_sample_ids = list(range(len(result_df)))
        abm_sid = st.selectbox("选择 ABM 样本 ID", abm_sample_ids, index=0)
        abm_sid = max(0, min(int(abm_sid), len(result_df) - 1))
        abm_row = result_df.iloc[abm_sid]

        abm_curve, abm_summary = simulate_single_epitope_response(
            epitope_seq=str(abm_row.get("epitope_seq", "")),
            dose=float(abm_row.get("dose", 1.0)),
            treatment_time=float(abm_row.get("treatment_time", 0.0)),
            horizon_h=96,
        )

        st.line_chart(
            abm_curve.set_index("time_h")[["antigen_load", "apc_presenting", "effector_t", "plasma_b", "antibody_titer"]]
        )

        a1, a2, a3 = st.columns(3)
        a1.metric("抗体峰值", f"{float(abm_summary.get('immune_peak_antibody', 0.0)):.3f}")
        a2.metric("效应 T 峰值", f"{float(abm_summary.get('immune_peak_effector_t', 0.0)):.3f}")
        a3.metric("免疫响应 AUC", f"{float(abm_summary.get('immune_response_auc', 0.0)):.3f}")

        with st.expander("导出 NetLogo 触发事件（当前结果）", expanded=False):
            trigger_df = build_epitope_triggers(result_df[[c for c in ["epitope_seq", "dose", "treatment_time"] if c in result_df.columns]])
            trigger_out = trigger_df.rename(columns={"time_h": "tick"}).copy()
            trigger_out["tick"] = np.round(pd.to_numeric(trigger_out["tick"], errors="coerce").fillna(0.0)).astype(int)
            trigger_out = trigger_out[["sample_id", "tick", "epitope_seq", "immunogenicity", "antigen_input"]]

            st.dataframe(trigger_out.head(50), use_container_width=True)
            trig_buf = io.StringIO()
            trigger_out.to_csv(trig_buf, index=False)
            st.download_button(
                "下载 NetLogo 触发事件 CSV",
                data=trig_buf.getvalue(),
                file_name="epitope_triggers_from_result.csv",
                mime="text/csv",
            )
    else:
        st.info("当前无可用于 ABM 的样本")

    st.subheader("下载结果")
    buf = io.StringIO()
    result_df.to_csv(buf, index=False)
    st.download_button("下载预测 CSV", data=buf.getvalue(), file_name="confluencia2_predictions.csv", mime="text/csv")
    report_lines = [
        "# Confluencia Drug Report",
        f"backend: {artifacts.model_backend}",
        f"samples: {len(result_df)}",
    ]
    if train_report.used_labels.get("efficacy", False):
        report_lines.extend(
            [
                f"efficacy_mae: {train_report.metrics.get('efficacy_mae', 0.0):.6f}",
                f"efficacy_rmse: {train_report.metrics.get('efficacy_rmse', 0.0):.6f}",
                f"efficacy_r2: {train_report.metrics.get('efficacy_r2', 0.0):.6f}",
            ]
        )
    else:
        report_lines.append("label_status: no_real_efficacy_label")
        report_lines.append("note: metrics hidden; ranking-only mode")

    if "drug_credible_eval" in st.session_state:
        ce = st.session_state["drug_credible_eval"]
        if ce.get("enabled", False):
            tm = ce.get("test_metrics", {})
            cv = ce.get("cv_summary", {})
            report_lines.extend(
                [
                    f"credible_backend_used: {ce.get('backend_used', artifacts.model_backend)}",
                    f"credible_test_rmse: {float(tm.get('rmse', 0.0)):.6f}",
                    f"credible_cv_rmse_mean: {float(cv.get('rmse_mean', 0.0)):.6f}",
                    f"credible_cv_rmse_ci95: {float(cv.get('rmse_ci95', 0.0)):.6f}",
                    f"credible_pass_baseline_gate: {bool(ce.get('pass_gate', False))}",
                ]
            )

    report_text = "\n".join(report_lines)
    st.download_button(
        "下载运行报告 (Markdown)",
        data=report_text,
        file_name="confluencia2_drug_report.md",
        mime="text/markdown",
    )
else:
    st.info("点击\"开始训练与预测\"后，将在此处展示训练结果。")

# ── Benchmark Results & Bottleneck Analysis ───────────────────────────
st.markdown("---")
st.header("基准测试结果与瓶颈分析")

with st.expander("📊 Drug 疗效预测基准结果", expanded=False):
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("多任务预测结果 (91k, 2083维特征)")
        st.markdown("""
        | 目标 | 最佳模型 | MAE | R² | Pearson r |
        |------|---------|-----|-----|-----------|
        | **efficacy** | **MOE** | **0.035** | **0.603** | **0.777** |
        | target_binding | Ridge | 0.029 | 0.965 | 0.982 |
        | immune_activation | HGB | 0.045 | 0.737 | 0.864 |
        | immune_cell_activation | HGB | 0.046 | 0.725 | 0.859 |
        | inflammation_risk | RF | 0.049 | 0.698 | 0.839 |
        | toxicity_risk | RF | 0.036 | 0.670 | 0.820 |
        """)
        st.caption("随机分割 80/20 (train=71,745, test=19,405)")

    with col_r2:
        st.subheader("特征增强效果 (91k, 疗效目标)")
        st.markdown("""
        | 配置 | 特征维度 | R² | Δ R² |
        |------|----------|-----|------|
        | Baseline (Morgan FP + RDKit) | 2,083 | 0.587 | — |
        | + Dose-response (DR) | 2,095 | 0.600 | +0.013 |
        | **+ DR + PK prior** | **2,104** | **0.603** | **+0.015** |
        | + GNN embedding | 2,232 | 0.581 | -0.006 |
        | + ChemBERTa | 2,872 | 0.579 | -0.008 |
        | + ESM-2 epitope | 3,363 | 0.576 | -0.011 |
        """)
        st.caption("离线模式：深度学习编码器返回零向量（需联网下载预训练权重）")

with st.expander("🔬 疗效预测瓶颈分析", expanded=False):
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        st.metric("MOE 权重", "0.33 / 0.33 / 0.33", help="Ridge/HGB/RF 权重均等 → 特征瓶颈，非模型瓶颈")

    with col_b2:
        st.metric("分子内方差占比", "48%", help="近半数疗效方差来自剂量/频率/表位组合（上下文相关）")

    with col_b3:
        st.metric("GroupKFold R²", "0.43", help="对未见分子的真实泛化能力（随机分割 R²=0.60 被夸大）")

    st.divider()

    col_b4, col_b5, col_b6 = st.columns(3)
    with col_b4:
        st.metric("独特分子数", "905", help="91,150 行数据仅来自 905 个不同分子")
    with col_b5:
        st.metric("Morgan FP 稀疏度", "~0.05%", help="2048 位指纹，905 个分子 → 稀疏表示")
    with col_b6:
        st.metric("达到 R²≥0.70 需要", "预训练权重", help="GNN/ChemBERTa/ESM-2 需要联网下载预训练模型")

    st.markdown("""
    **诊断结论：** 905 个分子用 2048 位 Morgan 指纹 → 稀疏特征表示。48% 疗效方差来自同分子内
    （剂量、频率、表位组合），表明疗效是上下文相关属性，非纯分子属性。
    GroupKFold（按 SMILES 分组）揭示真实泛化 R²≈0.43，远低于随机分割的 R²=0.60。
    密集分子嵌入（GNN, ChemBERTa, ESM-2）是正确方向，但离线模式返回零向量反而降低性能。
    **实际离线上限 R²≈0.60**，达到 R²≥0.70 需要预训练深度学习权重联网下载。
    """)

with st.expander("🧬 Epitope Binding 基准结果", expanded=False):
    st.markdown("""
    | 数据集 | 模型 | AUC | 说明 |
    |--------|------|-----|------|
    | 288k IEDB (baseline, 317 dims) | RF | 0.739 | 序列感知分割 80/20 |
    | 288k IEDB (MHC pseudo-seq, 1335 dims) | HGB | 0.751 | MHC 等位基因特征增强 |
    | 288k IEDB (HGB best, 317 dims) | HGB | 0.736 | 无 ESM-2 嵌入 |
    | netmhcpan_heldout (61 peptides) | RF | 0.596 | 独立基准集 |
    | iedb_heldout_mhc (2166 peptides) | HGB | 0.917 | 含真实 MHC 等位基因信息 |

    **参考：** NetMHCpan-4.1 AUC=0.92-0.96（专业绑定预测工具）
    **ESM-2 650M 增强基准：** 正在后台运行（预估 ~3.7h 完成）
    """)
    st.caption("AUC=0.917 来自 iedb_heldout_mhc.csv（2,166 peptides，含真实 allele 信息），非完整 288k 评估")

st.markdown("---")

with st.expander("常见问题（Drug）", expanded=False):
    st.markdown(
        """
        - 上传失败: CSV 优先使用 UTF-8；若是 Excel 导出中文 CSV，也可直接上传（已支持 GBK/GB18030 自动识别）。Parquet 需本地安装 pyarrow 或 fastparquet。
        - 提示缺少必填列: 至少需包含 `smiles/dose/freq/treatment_time`，外部测试集还需 `efficacy`。
        - 指标全是 0: 通常是缺少真实标签列（如 `efficacy`），系统会走代理目标，仅用于流程验证。
        - 训练较慢: 可将预设切到`快速`或降低数据量先验证流程。
        - SMILES 质量差: 在\"数据质控报告\"中先看 `smiles_invalid_ratio`，过高时建议先清洗。
        """
    )

st.markdown("---")
st.header("ED2Mol + 纯强化学习反思进化")
st.caption("生成 -> 评估 -> 反思 -> 策略更新 -> 再生成")

with st.expander("进化设置", expanded=False):
    # Use relative paths based on project root
    _project_root = Path(__file__).parent.parent
    _default_ed_repo = str(_project_root / "external" / "ED2Mol")
    _default_ed_cfg = str(_project_root / "external" / "ED2Mol" / "configs" / "hitopt.yml")
    _default_receptor = str(_project_root / "data" / "receptor.pdb")
    _default_core = str(_project_root / "data" / "core.sdf")
    _default_out_dir = str(_project_root / "tmp" / "ed2mol_outputs")

    seed_smiles_text = st.text_area(
        "初始 SMILES（逗号或换行分隔）",
        value="CCO, CCN(CC)CC",
        height=80,
    )
    ed_repo = st.text_input("ED2Mol 仓库目录", value=_default_ed_repo)
    ed_cfg = st.text_input("ED2Mol 配置路径", value=_default_ed_cfg)
    ed_py = st.text_input("ED2Mol Python 命令", value="python")

    st.markdown("**自动生成 ED2Mol 配置（方案1）**")
    t1, t2 = st.columns(2)
    cfg_mode = t1.selectbox("配置模式", ["denovo", "hitopt"], index=1)
    cfg_save = t2.text_input("配置保存路径", value=ed_cfg)
    t3, t4 = st.columns(2)
    receptor_pdb = t3.text_input("受体 PDB 路径", value=_default_receptor)
    ref_core = t4.text_input("参考核心 SDF（hitopt）", value=_default_core)
    t5, t6, t7 = st.columns(3)
    cx = t5.number_input("口袋中心 x", value=0.0, step=0.1)
    cy = t6.number_input("口袋中心 y", value=0.0, step=0.1)
    cz = t7.number_input("口袋中心 z", value=0.0, step=0.1)
    out_dir = st.text_input("ED2Mol 输出目录", value=_default_out_dir)
    gen_cfg_btn = st.button("生成 ED2Mol YAML")
    if gen_cfg_btn:
        try:
            cfg_mode_lit: Literal["denovo", "hitopt"] = "hitopt" if cfg_mode == "hitopt" else "denovo"
            saved = write_ed2mol_config(
                save_path=str(cfg_save),
                mode=cfg_mode_lit,
                output_dir=str(out_dir),
                receptor_pdb=str(receptor_pdb),
                center_x=float(cx),
                center_y=float(cy),
                center_z=float(cz),
                reference_core_sdf=(str(ref_core) if cfg_mode_lit == "hitopt" else None),
            )
            st.success(f"配置已生成: {saved}")
        except Exception as e:
            st.error(f"配置生成失败: {e}")

    e1, e2, e3 = st.columns(3)
    evo_rounds = e1.slider("轮数", min_value=1, max_value=12, value=4)
    evo_topk = e2.slider("每轮保留 Top-k", min_value=2, max_value=32, value=10)
    evo_cands = e3.slider("每轮候选数", min_value=8, max_value=128, value=40, step=4)

    e4, e5, e6 = st.columns(3)
    evo_dose = e4.number_input("剂量", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    evo_freq = e5.number_input("频次", min_value=0.1, max_value=8.0, value=1.0, step=0.1)
    evo_time = e6.number_input("治疗时长", min_value=0.0, max_value=240.0, value=24.0, step=1.0)

    evo_epitope = st.text_input("表位序列", value="SLYNTVATL")
    evo_compute = st.selectbox("进化计算档位", ["low", "medium", "high"], index=0)

    st.markdown("**Pareto 奖励搜索（方案2）**")
    use_pareto = st.checkbox("启用 Pareto 引导的目标权重搜索", value=True)
    pareto_samples = st.slider("每轮权重候选数", min_value=8, max_value=256, value=64, step=8)
    st.markdown("**稳定性控制**")
    rl_patience = st.slider("强化学习早停耐心轮数", min_value=1, max_value=8, value=3)
    rl_min_improve = st.select_slider("最小改进阈值", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)

    st.markdown("**进化评估自适应校准（方案3）**")
    evo_adaptive_enabled = st.checkbox("进化阶段启用自适应校准", value=bool(adaptive_enabled))
    evo_adaptive_strength = st.slider(
        "进化自适应强度",
        min_value=0.0,
        max_value=1.0,
        value=float(adaptive_strength),
        step=0.05,
    )
    evo_gate_penalty_enabled = st.checkbox("启用高风险门控惩罚", value=True)
    e7, e8, e9 = st.columns(3)
    evo_gate_mode_label = e7.selectbox("阈值模式", ["固定阈值", "分位数自适应"], index=0)
    evo_gate_threshold = e8.slider("高风险阈值", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
    evo_gate_quantile = e9.slider("风险分位数", min_value=0.50, max_value=0.95, value=0.80, step=0.05)
    evo_gate_penalty = st.slider("门控惩罚系数", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

run_evo = st.button("运行 ED2Mol 反思进化", type="primary")

if run_evo:
    seeds = [x.strip() for x in seed_smiles_text.replace("\n", ",").split(",") if x.strip()]
    cfg = EvolutionConfig(
        rounds=int(evo_rounds),
        top_k=int(evo_topk),
        candidates_per_round=int(evo_cands),
        dose=float(evo_dose),
        freq=float(evo_freq),
        treatment_time=float(evo_time),
        epitope_seq=str(evo_epitope),
        compute_mode=str(evo_compute),
        group_id="EVO",
        use_pareto_search=bool(use_pareto),
        pareto_weight_samples=int(pareto_samples),
        early_stop_patience=int(rl_patience),
        min_improve=float(rl_min_improve),
        adaptive_enabled=bool(evo_adaptive_enabled),
        adaptive_strength=float(evo_adaptive_strength),
        use_adaptive_gate_penalty=bool(evo_gate_penalty_enabled),
        risk_gate_threshold=float(evo_gate_threshold),
        risk_gate_penalty=float(evo_gate_penalty),
        risk_gate_threshold_mode=("quantile" if evo_gate_mode_label == "分位数自适应" else "fixed"),
        risk_gate_threshold_quantile=float(evo_gate_quantile),
    )
    _cloud_mode = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
    with st.spinner("正在执行 ED2Mol 生成与纯强化学习反思更新..." + ("（云服务器模式）" if _cloud_mode else "")):
        if _cloud_mode:
            _cloud_client = create_cloud_client("remote", cloud_server_url)
            evo_df, evo_art = _cloud_client.evolve_molecules(
                seed_smiles=seeds,
                cfg=cfg,
                ed2mol_repo_dir=str(ed_repo),
                ed2mol_config_path=str(ed_cfg),
                ed2mol_python_cmd=str(ed_py),
            )
        else:
            evo_df, evo_art = evolve_molecules_with_reflection(
                seed_smiles=seeds,
                cfg=cfg,
                ed2mol_repo_dir=str(ed_repo),
                ed2mol_config_path=str(ed_cfg),
                ed2mol_python_cmd=str(ed_py),
            )
    st.session_state["drug_evo_result"] = (evo_df, evo_art)
    st.session_state["drug_evo_cfg"] = cfg
    _append_experiment_log(
        module="drug-evolution",
        config={
            "rounds": int(evo_rounds),
            "top_k": int(evo_topk),
            "candidates": int(evo_cands),
            "adaptive_enabled": bool(evo_adaptive_enabled),
            "adaptive_strength": float(evo_adaptive_strength),
            "gate_mode": ("quantile" if evo_gate_mode_label == "分位数自适应" else "fixed"),
            "gate_threshold": float(evo_gate_threshold),
            "gate_quantile": float(evo_gate_quantile),
            "gate_penalty": float(evo_gate_penalty),
        },
        metrics={"rounds_ran": int(evo_art.rounds_ran), "best_reward": float(evo_art.best_reward)},
    )

if "drug_evo_result" in st.session_state:
    evo_df, evo_art = st.session_state["drug_evo_result"]
    evo_cfg = st.session_state.get("drug_evo_cfg")

    st.success("进化完成")
    c7, c8 = st.columns(2)
    c7.metric("使用 ED2Mol", "是" if evo_art.used_ed2mol else "否（使用回退策略）")
    c8.metric("进化样本数", f"{len(evo_df)}")
    c9, c10 = st.columns(2)
    c9.metric("实际运行轮数", f"{evo_art.rounds_ran}")
    c10.metric("最佳奖励", f"{evo_art.best_reward:.4f}")
    if evo_cfg is not None:
        st.caption(
            f"进化自适应校准: {'已启用' if bool(evo_cfg.adaptive_enabled) else '未启用'}"
            + f" | 强度: {float(evo_cfg.adaptive_strength):.2f}"
        )
        st.caption(
            f"高风险门控惩罚: {'已启用' if bool(evo_cfg.use_adaptive_gate_penalty) else '未启用'}"
            + f" | 阈值: {float(evo_cfg.risk_gate_threshold):.2f}"
            + f" | 系数: {float(evo_cfg.risk_gate_penalty):.2f}"
        )
        st.caption(
            f"阈值模式: {str(getattr(evo_cfg, 'risk_gate_threshold_mode', 'fixed'))}"
            + f" | 分位数: {float(getattr(evo_cfg, 'risk_gate_threshold_quantile', 0.80)):.2f}"
        )

    if evo_art.per_round_best:
        st.subheader("每轮最佳奖励")
        pr_df = pd.DataFrame({"round": np.arange(1, len(evo_art.per_round_best) + 1), "best_reward": evo_art.per_round_best})
        st.line_chart(pr_df.set_index("round"))

    st.subheader("策略 Logits（纯强化学习）")
    p_df = pd.DataFrame({"action": list(evo_art.final_policy_logits.keys()), "logit": list(evo_art.final_policy_logits.values())})
    st.bar_chart(p_df.set_index("action"))

    st.subheader("选中的目标权重")
    w_df = pd.DataFrame({"objective": list(evo_art.selected_objective_weights.keys()), "weight": list(evo_art.selected_objective_weights.values())})
    st.bar_chart(w_df.set_index("objective"))

    st.subheader("反思记录")
    for note in evo_art.reflections[-12:]:
        st.write("- " + str(note))

    if not evo_df.empty:
        if all(c in evo_df.columns for c in ["efficacy_pred", "inflammation_risk_pred", "ctm_peak_toxicity", "pareto_front"]):
            st.subheader("Pareto 前沿交互散点图")
            viz_df = evo_df.copy()
            viz_df["pareto_front"] = viz_df["pareto_front"].astype(bool)
            viz_df["pareto_label"] = np.where(viz_df["pareto_front"], "Pareto", "Non-Pareto")

            f1, f2, f3 = st.columns(3)
            x_candidates = [c for c in ["inflammation_risk_pred", "toxicity_risk_pred", "ctm_peak_toxicity", "reward"] if c in viz_df.columns]
            y_candidates = [c for c in ["efficacy_pred", "target_binding_pred", "immune_activation_pred", "immune_cell_activation_pred"] if c in viz_df.columns]
            x_metric = f1.selectbox("X 轴", x_candidates, index=0)
            y_metric = f2.selectbox("Y 轴", y_candidates, index=0)

            round_opts = []
            if "round" in viz_df.columns:
                round_opts = sorted([int(x) for x in pd.unique(viz_df["round"])])
            selected_rounds = f3.multiselect("轮次筛选", options=round_opts, default=round_opts) if round_opts else []

            action_opts = []
            if "action" in viz_df.columns:
                action_opts = sorted([str(x) for x in pd.unique(viz_df["action"])])
            selected_actions = st.multiselect("动作筛选", options=action_opts, default=action_opts) if action_opts else []

            if round_opts:
                viz_df = viz_df[viz_df["round"].astype(int).isin(selected_rounds)]
            if action_opts:
                viz_df = viz_df[viz_df["action"].astype(str).isin(selected_actions)]

            front_n = int(viz_df["pareto_front"].sum()) if not viz_df.empty else 0
            st.caption(f"筛选后 Pareto 点数: {front_n} / {len(viz_df)}")

            if not viz_df.empty:
                st.scatter_chart(viz_df, x=x_metric, y=y_metric, color="pareto_label")
            else:
                st.info("筛选后无可展示数据点")

        st.subheader("进化分子 Top 列表")
        top_evo = evo_df.sort_values("reward", ascending=False).head(30)
        cols = [
            c
            for c in [
                "round",
                "action",
                "smiles",
                "reward_raw",
                "reward_penalty_gate",
                "reward",
                "adaptive_gate_flag",
                "risk_gate_threshold_used",
                "efficacy_pred",
                "target_binding_pred",
                "immune_activation_pred",
                "immune_cell_activation_pred",
                "inflammation_risk_pred",
                "toxicity_risk_pred",
                "ctm_peak_toxicity",
                "obj_low_gate_excess",
                "pareto_front",
            ]
            if c in top_evo.columns
        ]
        st.dataframe(top_evo[cols], use_container_width=True)

        ebuf = io.StringIO()
        evo_df.to_csv(ebuf, index=False)
        st.download_button(
            "下载进化结果 CSV",
            data=ebuf.getvalue(),
            file_name="confluencia2_evolution.csv",
            mime="text/csv",
        )
else:
    st.info("点击运行 ED2Mol 反思进化后，将在此处展示进化结果。")

# ------------------------------------------------------------------
# circRNA v2.2: Sequence evolution
# ------------------------------------------------------------------
st.markdown("---")
st.header("circRNA 序列进化（v2.2）")
st.caption("骨架突变 / IRES 优化 / UTR 重排 / 修饰策略进化")

_crna_evo_enabled = st.session_state.get("crna_mode", False)
if _crna_evo_enabled:
    with st.expander("circRNA 进化设置", expanded=False):
        crna_evo_seq = st.text_area(
            "初始 circRNA 序列",
            value="AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGCGCUAUGGCUAGCUAUGCGC",
            height=80,
        )
        ce1, ce2, ce3 = st.columns(3)
        crna_evo_rounds = ce1.slider("进化轮数", min_value=1, max_value=10, value=4, key="crna_evo_rounds")
        crna_evo_cands = ce2.slider("每轮候选数", min_value=4, max_value=48, value=16, step=4, key="crna_evo_cands")
        crna_evo_topk = ce3.slider("每轮 Top-k", min_value=2, max_value=16, value=6, key="crna_evo_topk")

        crna_evo_btn = st.button("运行 circRNA 进化", key="crna_evo_run")
        if crna_evo_btn and crna_evo_seq:
            _cloud_mode_crna = CLOUD_CLIENT_AVAILABLE and cloud_mode_enabled
            with st.spinner("正在进化 circRNA 序列..." + ("（云服务器模式）" if _cloud_mode_crna else "")):
                crna_evo_cfg = CircRNAEvolutionConfig(
                    rounds=crna_evo_rounds,
                    candidates_per_round=crna_evo_cands,
                    top_k=crna_evo_topk,
                    seed_seq=crna_evo_seq.strip(),
                    modification=st.session_state.get("crna_mod", "m6A"),
                    delivery_vector=st.session_state.get("crna_del", "LNP_liver"),
                    route=st.session_state.get("crna_route", "IV"),
                    ires_type=st.session_state.get("crna_ires", "EMCV"),
                    dose=2.0, freq=1.0, treatment_time=24.0, seed=42,
                )
                if _cloud_mode_crna:
                    _cloud_client_crna = create_cloud_client("remote", cloud_server_url)
                    crna_evo_df, crna_evo_art = _cloud_client_crna.evolve_cirrna(crna_evo_cfg)
                else:
                    crna_evo_df, crna_evo_art = evolve_cirrna_sequences(crna_evo_cfg)

                if crna_evo_df is not None and not crna_evo_df.empty:
                    st.success("circRNA 进化完成！")
                    st.subheader("进化统计")
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("总轮数", crna_evo_art["rounds_ran"])
                    ec2.metric("最佳 Reward", f"{crna_evo_art['best_reward']:.4f}")
                    ec3.metric("Pareto 前沿", f"{int(crna_evo_df['pareto_front'].sum())}/{len(crna_evo_df)}")

                    st.subheader("各轮 Reward 趋势")
                    if crna_evo_art.get("per_round_best"):
                        st.line_chart({"round_best": crna_evo_art["per_round_best"]})

                    st.subheader("Top 进化序列")
                    top_crna = crna_evo_df.sort_values("reward", ascending=False).head(15)
                    display_cols = [
                        c for c in [
                            "round", "action", "circrna_seq", "modification", "reward",
                            "obj_stability", "obj_translation", "obj_immune_evasion",
                            "innate_safety_score", "innate_ifn_storm_level"
                        ]
                        if c in top_crna.columns
                    ]
                    st.dataframe(top_crna[display_cols], use_container_width=True)

                    if crna_evo_art.get("reflections"):
                        with st.expander("进化反思日志", expanded=False):
                            for ref in crna_evo_art["reflections"]: 
                                st.text(ref)

                    ebuf = io.StringIO()
                    crna_evo_df.to_csv(ebuf, index=False)
                    st.download_button(
                        "下载 circRNA 进化结果 CSV",
                        data=ebuf.getvalue(),
                        file_name="confluencia2_crna_evolution.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("进化未产出有效结果。")
else:
    st.info("请在左侧边栏启用 circRNA 模式以使用此功能。")
