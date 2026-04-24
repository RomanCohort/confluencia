from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HYDROPHOBIC_AA = set("AILMFWYV")


@dataclass(frozen=True)
class ImmuneABMConfig:
    horizon_h: int = 96
    dt_h: float = 1.0
    apc_count: float = 60.0
    naive_t_count: float = 220.0
    naive_b_count: float = 180.0
    antigen_decay: float = 0.08
    apc_capture_rate: float = 0.06
    presentation_decay: float = 0.05
    t_activation_rate: float = 0.10
    t_decay: float = 0.03
    b_activation_rate: float = 0.08
    plasma_decay: float = 0.02
    antibody_secretion_rate: float = 1.6
    antibody_decay: float = 0.07
    neutralization_rate: float = 0.12


def _epitope_immunogenicity(seq: str) -> float:
    s = str(seq or "").strip().upper()
    if not s:
        return 0.1

    length_score = np.clip((len(s) - 8.0) / 14.0, 0.0, 1.0)
    uniq_score = np.clip(len(set(s)) / max(len(s), 1), 0.0, 1.0)
    hydro = sum(1 for x in s if x in HYDROPHOBIC_AA)
    hydro_score = np.clip(hydro / max(len(s), 1), 0.0, 1.0)
    score = 0.45 * length_score + 0.35 * uniq_score + 0.20 * hydro_score
    return float(np.clip(score, 0.05, 1.0))


def build_epitope_triggers(df: pd.DataFrame, default_time_h: float = 0.0) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["sample_id", "time_h", "epitope_seq", "immunogenicity", "antigen_input"])

    work = df.copy()
    if "epitope_seq" not in work.columns:
        work["epitope_seq"] = ""
    if "treatment_time" not in work.columns:
        work["treatment_time"] = float(default_time_h)
    if "dose" not in work.columns:
        work["dose"] = 1.0

    records: list[dict[str, object]] = []
    for i, row in enumerate(work.reset_index(drop=True).to_dict(orient="records")):
        epi = str(row.get("epitope_seq", ""))
        t_h = float(pd.to_numeric(row.get("treatment_time", default_time_h), errors="coerce") or 0.0)
        dose = float(pd.to_numeric(row.get("dose", 1.0), errors="coerce") or 1.0)
        immune = _epitope_immunogenicity(epi)
        antigen_input = float(np.clip(dose, 0.1, 20.0) * (0.5 + 1.5 * immune))
        records.append(
            {
                "sample_id": i,
                "time_h": float(max(0.0, t_h)),
                "epitope_seq": epi,
                "immunogenicity": immune,
                "antigen_input": antigen_input,
            }
        )

    return pd.DataFrame(records)


def export_netlogo_trigger_csv(df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    trig = build_epitope_triggers(df)
    out = trig.rename(columns={"time_h": "tick"}).copy()
    out["tick"] = np.round(pd.to_numeric(out["tick"], errors="coerce").fillna(0.0)).astype(int)
    out = out[["sample_id", "tick", "epitope_seq", "immunogenicity", "antigen_input"]]
    out.to_csv(out_csv, index=False)
    return out


def _group_triggers(triggers: pd.DataFrame) -> dict[int, float]:
    if triggers is None or len(triggers) == 0:
        return {}

    ticks = np.round(pd.to_numeric(triggers["time_h"], errors="coerce").fillna(0.0)).astype(int)
    antigen_input = pd.to_numeric(triggers["antigen_input"], errors="coerce").fillna(0.0).astype(float)
    grouped = pd.DataFrame({"tick": ticks, "antigen_input": antigen_input}).groupby("tick", as_index=False)["antigen_input"].sum()
    return {int(r["tick"]): float(r["antigen_input"]) for _, r in grouped.iterrows()}


def simulate_immune_response(
    triggers: pd.DataFrame,
    config: ImmuneABMConfig | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    cfg = config or ImmuneABMConfig()
    rng = np.random.default_rng(seed)
    _ = rng

    event_map = _group_triggers(triggers)
    steps = int(max(1, round(cfg.horizon_h / max(cfg.dt_h, 1e-6))))

    antigen = 0.0
    apc_presenting = 0.0
    effector_t = 0.0
    plasma_b = 0.0
    antibody = 0.0

    rows: list[dict[str, float]] = []
    for step in range(steps + 1):
        time_h = step * cfg.dt_h
        tick = int(round(time_h))
        antigen += float(event_map.get(tick, 0.0))

        apc_capture_capacity = cfg.apc_count * cfg.apc_capture_rate * cfg.dt_h
        captured = float(min(antigen, apc_capture_capacity))
        antigen = max(0.0, antigen - captured)
        apc_presenting = max(0.0, apc_presenting + captured - cfg.presentation_decay * apc_presenting * cfg.dt_h)

        t_activation = min(cfg.naive_t_count, apc_presenting) * cfg.t_activation_rate * cfg.dt_h
        effector_t = max(0.0, effector_t + t_activation - cfg.t_decay * effector_t * cfg.dt_h)

        b_drive = min(cfg.naive_b_count, effector_t) * (1.0 + np.clip(antigen / 40.0, 0.0, 1.5))
        b_activation = b_drive * cfg.b_activation_rate * cfg.dt_h
        plasma_b = max(0.0, plasma_b + b_activation - cfg.plasma_decay * plasma_b * cfg.dt_h)

        antibody = max(0.0, antibody + cfg.antibody_secretion_rate * plasma_b * cfg.dt_h - cfg.antibody_decay * antibody * cfg.dt_h)

        neutralized = min(antigen, cfg.neutralization_rate * antibody * cfg.dt_h)
        antigen = max(0.0, antigen - neutralized)
        antigen = max(0.0, antigen - cfg.antigen_decay * antigen * cfg.dt_h)

        rows.append(
            {
                "time_h": float(time_h),
                "antigen_load": float(antigen),
                "apc_presenting": float(apc_presenting),
                "effector_t": float(effector_t),
                "plasma_b": float(plasma_b),
                "antibody_titer": float(antibody),
                "neutralized_antigen": float(neutralized),
            }
        )

    return pd.DataFrame(rows)


def summarize_immune_curve(curve: pd.DataFrame) -> dict[str, float]:
    if curve is None or len(curve) == 0:
        return {
            "immune_peak_antibody": 0.0,
            "immune_peak_effector_t": 0.0,
            "immune_peak_antigen": 0.0,
            "immune_response_auc": 0.0,
        }

    t = pd.to_numeric(curve["time_h"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ab = pd.to_numeric(curve["antibody_titer"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    te = pd.to_numeric(curve["effector_t"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ag = pd.to_numeric(curve["antigen_load"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    immune_signal = 0.55 * ab + 0.45 * te
    auc = float(np.trapezoid(immune_signal, t)) if len(t) > 1 else float(np.sum(immune_signal))

    return {
        "immune_peak_antibody": float(np.max(ab)) if len(ab) else 0.0,
        "immune_peak_effector_t": float(np.max(te)) if len(te) else 0.0,
        "immune_peak_antigen": float(np.max(ag)) if len(ag) else 0.0,
        "immune_response_auc": max(0.0, auc),
    }


def simulate_single_epitope_response(
    epitope_seq: str,
    dose: float,
    treatment_time: float,
    horizon_h: int = 96,
) -> tuple[pd.DataFrame, dict[str, float]]:
    one = pd.DataFrame(
        {
            "epitope_seq": [str(epitope_seq or "")],
            "dose": [float(dose)],
            "treatment_time": [float(max(0.0, treatment_time))],
        }
    )
    trig = build_epitope_triggers(one)
    curve = simulate_immune_response(trig, config=ImmuneABMConfig(horizon_h=int(horizon_h)))
    return curve, summarize_immune_curve(curve)


def batch_simulate_epitopes(
    epitopes: Iterable[str],
    doses: Iterable[float],
    treatment_times: Iterable[float],
    horizon_h: int = 96,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "epitope_seq": list(epitopes),
            "dose": list(doses),
            "treatment_time": list(treatment_times),
        }
    )
    trig = build_epitope_triggers(df)

    rows: list[dict[str, float]] = []
    for sid, g in trig.groupby("sample_id"):
        sid_int = int(float(pd.to_numeric(sid, errors="coerce") or 0.0))
        curve = simulate_immune_response(g, config=ImmuneABMConfig(horizon_h=int(horizon_h)), seed=7 + sid_int)
        s = summarize_immune_curve(curve)
        s["sample_id"] = sid_int
        rows.append(s)

    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
