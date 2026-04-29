"""Unified experiment report template for Confluencia 2.0 Epitope.

Generates Markdown + CSV experiment reports with all required fields:
config, data hash, environment dependencies, core metrics,
significance results, OOD eval, and AA composition stratification.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def generate_experiment_report(
    module: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    data_hash: str,
    n_rows: int,
    env_deps: dict[str, str],
    credible_eval: dict[str, Any] | None = None,
    python_executable: str = "",
) -> str:
    """Generate a unified Markdown experiment report.

    Returns the report as a string.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# Experiment Report: {module}",
        "",
        f"- **Timestamp**: {ts}",
        f"- **Module**: {module}",
        f"- **Data rows**: {n_rows}",
        f"- **Data SHA256**: `{data_hash}`",
        f"- **Python**: {python_executable}",
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

    if credible_eval is not None and credible_eval.get("enabled", False):
        lines.extend(_credible_section(credible_eval))

    lines.append("")
    return "\n".join(lines)


def _credible_section(ce: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Credible Evaluation",
        "",
        f"- **Backend**: {ce.get('backend_used', 'unknown')}",
        f"- **Real Mamba**: {ce.get('used_real_mamba', False)}",
    ]

    sp = ce.get("split_sizes", {})
    lines.append(f"- **Split**: train={sp.get('train',0)}, val={sp.get('val',0)}, test={sp.get('test',0)}")

    tm = ce.get("test_metrics", {})
    lines.extend([
        f"- **Test MAE**: {tm.get('mae', 0.0):.6f}",
        f"- **Test RMSE**: {tm.get('rmse', 0.0):.6f}",
        f"- **Test R2**: {tm.get('r2', 0.0):.6f}",
    ])

    cs = ce.get("cv_summary", {})
    lines.extend([
        "",
        "### CV Summary (5-fold)",
        f"- MAE: {cs.get('mae_mean', 0.0):.6f} +/- {cs.get('mae_std', 0.0):.6f} (CI95: +/-{cs.get('mae_ci95', 0.0):.6f})",
        f"- RMSE: {cs.get('rmse_mean', 0.0):.6f} +/- {cs.get('rmse_std', 0.0):.6f} (CI95: +/-{cs.get('rmse_ci95', 0.0):.6f})",
        f"- R2: {cs.get('r2_mean', 0.0):.6f} +/- {cs.get('r2_std', 0.0):.6f} (CI95: +/-{cs.get('r2_ci95', 0.0):.6f})",
    ])

    lines.append(f"- **Pass baseline gate**: {ce.get('pass_gate', False)}")

    leak = ce.get("leakage_audit", {})
    if isinstance(leak, dict):
        lines.extend([
            "",
            "### Leakage Audit",
            f"- Overlap count: {int(float(leak.get('overlap_count', 0)))}",
            f"- Overlap ratio: {float(leak.get('overlap_ratio', 0)) * 100:.4f}%",
        ])

    sig = ce.get("significance", {})
    if isinstance(sig, dict):
        lines.extend([
            "",
            "### Significance Test",
            f"- Model A: {sig.get('model_a', 'na')}",
            f"- Model B: {sig.get('model_b', 'na')}",
            f"- p-value: {float(sig.get('p_value', 1.0)):.8f}",
            f"- effect size (dz): {float(sig.get('effect_size_dz', 0.0)):.8f}",
            f"- non-zero pairs: {int(sig.get('n_nonzero', 0))}",
        ])

    ood = ce.get("ood_eval", {})
    if isinstance(ood, dict):
        lines.extend([
            "",
            "### OOD Subset Evaluation",
            f"- OOD ratio: {float(ood.get('ood_ratio', 0.0)) * 100:.2f}%",
            f"- OOD count: {int(ood.get('ood_count', 0))}",
            f"- ID count: {int(ood.get('id_count', 0))}",
        ])
        ood_m = ood.get("ood_metrics", {}) if isinstance(ood.get("ood_metrics", {}), dict) else {}
        id_m = ood.get("id_metrics", {}) if isinstance(ood.get("id_metrics", {}), dict) else {}
        lines.extend([
            f"- ID RMSE: {float(id_m.get('rmse', 0.0)):.6f}",
            f"- OOD RMSE: {float(ood_m.get('rmse', 0.0)):.6f}",
        ])

    aa_strat = ce.get("aa_composition_strat_df", pd.DataFrame())
    if isinstance(aa_strat, pd.DataFrame) and not aa_strat.empty:
        lines.extend([
            "",
            "### AA Composition Stratification",
        ])
        lines.append("| property | bin | n | mae | rmse | r2 |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in aa_strat.iterrows():
            lines.append(
                f"| {row.get('property','?')} | {row.get('bin','?')} | {int(row.get('n',0))} "
                f"| {float(row.get('mae',0.0)):.6f} | {float(row.get('rmse',0.0)):.6f} "
                f"| {float(row.get('r2',0.0)):.6f} |"
            )

    cal = ce.get("mamba_calibration", {})
    if isinstance(cal, dict):
        lines.extend([
            "",
            "### Mamba Calibration",
            f"- Real Mamba used: {cal.get('real_mamba_used', False)}",
            f"- Real RMSE: {float(cal.get('real_rmse', 0.0)):.6f}",
            f"- Fallback RMSE: {float(cal.get('fallback_rmse', 0.0)):.6f}",
            f"- Delta: {float(cal.get('delta_rmse', 0.0)):.6f}",
        ])

    ext_m = ce.get("external_metrics", {})
    if isinstance(ext_m, dict):
        lines.extend([
            "",
            "### External Test Set",
            f"- MAE: {float(ext_m.get('mae', 0.0)):.6f}",
            f"- RMSE: {float(ext_m.get('rmse', 0.0)):.6f}",
            f"- R2: {float(ext_m.get('r2', 0.0)):.6f}",
        ])

    return lines


def save_report_csv(
    module: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    data_hash: str,
    n_rows: int,
    env_deps: dict[str, str],
    log_dir: str | Path | None = None,
) -> Path:
    """Append a single-row CSV summary to the runs log.

    Returns the path to the CSV file.
    """
    if log_dir is None:
        base = Path(__file__).resolve().parents[1] / "logs" / "reproduce"
    else:
        base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat(timespec="seconds")
    row = {
        "timestamp": ts,
        "module": module,
        "rows": n_rows,
        "data_sha256": data_hash,
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
    return csv_path
