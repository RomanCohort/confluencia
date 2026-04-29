from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    module: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text) if text else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _fmt(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _fmt_ci(v: Optional[float]) -> str:
    return _fmt(v, digits=6)


def load_runs(runs_csv: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    with runs_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            module = str(row.get("module", "")).strip()
            if module != "drug-main":
                continue
            records.append(
                RunRecord(
                    run_id=str(row.get("run_id", "")).strip(),
                    timestamp=str(row.get("timestamp", "")).strip(),
                    module=module,
                    config=_safe_json_loads(str(row.get("config", ""))),
                    metrics=_safe_json_loads(str(row.get("metrics", ""))),
                )
            )
    records.sort(key=lambda r: r.timestamp)
    return records


def extract_metric(record: RunRecord, key: str) -> Optional[float]:
    return _to_float(record.metrics.get(key))


def extract_dynamics_metrics(record: RunRecord) -> Tuple[Optional[float], Optional[float]]:
    peak_candidates = ["ctm_peak_efficacy", "peak_efficacy"]
    auc_candidates = ["ctm_auc_efficacy", "auc_efficacy"]

    peak: Optional[float] = None
    auc: Optional[float] = None
    for k in peak_candidates:
        peak = _to_float(record.metrics.get(k))
        if peak is not None:
            break
    for k in auc_candidates:
        auc = _to_float(record.metrics.get(k))
        if auc is not None:
            break
    return peak, auc


def latest_by_dynamics(records: List[RunRecord], dynamics_model: str) -> Optional[RunRecord]:
    hits = [r for r in records if str(r.config.get("dynamics_model", "")).lower() == dynamics_model.lower()]
    return hits[-1] if hits else None


def mean_std_ci95(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    margin = 1.96 * std / math.sqrt(n)
    return mean, std, mean - margin, mean + margin


def build_markdown(records: List[RunRecord]) -> str:
    ctm = latest_by_dynamics(records, "ctm")
    nd = latest_by_dynamics(records, "ndp4pd")

    def row_for_method(name: str, backend: str, rec: Optional[RunRecord]) -> str:
        if rec is None:
            return f"| {name} | {backend} | N/A | N/A | N/A | N/A | N/A |"
        mae = extract_metric(rec, "efficacy_mae")
        rmse = extract_metric(rec, "efficacy_rmse")
        r2 = extract_metric(rec, "efficacy_r2")
        peak, auc = extract_dynamics_metrics(rec)
        return (
            f"| {name} | {backend} | {_fmt(mae)} | {_fmt(rmse)} | {_fmt(r2)} | "
            f"{_fmt(peak)} | {_fmt(auc)} |"
        )

    lines: List[str] = []
    lines.append("## Auto-Generated Paper Tables")
    lines.append("")
    lines.append("Data source: `dist/logs/reproduce/runs.csv` (module = `drug-main`).")
    lines.append("")

    lines.append("### D.1 主结果表（自动汇总）")
    lines.append("")
    lines.append("| 方法 | Dynamics Backend | MAE (`efficacy_mae`) ↓ | RMSE (`efficacy_rmse`) ↓ | $R^2$ (`efficacy_r2`) ↑ | Peak Efficacy (`ctm_peak_efficacy`) ↑ | AUC(Efficacy) (`ctm_auc_efficacy`) ↑ |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append("| 静态回归基线 | - | N/A | N/A | N/A | N/A | N/A |")
    lines.append("| MOE（无动力学） | - | N/A | N/A | N/A | N/A | N/A |")
    lines.append(row_for_method("本方法", "CTM", ctm))
    lines.append(row_for_method("本方法", "NDP4PD", nd))
    lines.append("")

    lines.append("### D.2 消融实验表（占位模板）")
    lines.append("")
    lines.append("| 配置 | MAE (`efficacy_mae`) ↓ | RMSE (`efficacy_rmse`) ↓ | $R^2$ (`efficacy_r2`) ↑ | Peak Efficacy (`ctm_peak_efficacy`) ↑ | AUC(Efficacy) (`ctm_auc_efficacy`) ↑ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append("| Full Model | N/A | N/A | N/A | N/A | N/A |")
    lines.append("| w/o epitope feature | N/A | N/A | N/A | N/A | N/A |")
    lines.append("| w/o group id | N/A | N/A | N/A | N/A | N/A |")
    lines.append("| CTM -> NDP4PD | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")

    lines.append("### D.3 稳定性统计表（重复实验自动汇总）")
    lines.append("")

    mae_vals = [v for v in (extract_metric(r, "efficacy_mae") for r in records) if v is not None]
    rmse_vals = [v for v in (extract_metric(r, "efficacy_rmse") for r in records) if v is not None]
    r2_vals = [v for v in (extract_metric(r, "efficacy_r2") for r in records) if v is not None]

    peak_vals: List[float] = []
    auc_vals: List[float] = []
    for r in records:
        peak, auc = extract_dynamics_metrics(r)
        if peak is not None:
            peak_vals.append(peak)
        if auc is not None:
            auc_vals.append(auc)

    def stat_row(label: str, vals: List[float]) -> str:
        mean, std, lo, hi = mean_std_ci95(vals)
        return f"| {label} | {_fmt(mean)} | {_fmt(std)} | {_fmt_ci(lo)} | {_fmt_ci(hi)} |"

    lines.append("| 指标 | Mean | Std | 95% CI Lower | 95% CI Upper |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(stat_row("MAE (`efficacy_mae`)", mae_vals))
    lines.append(stat_row("RMSE (`efficacy_rmse`)", rmse_vals))
    lines.append(stat_row("$R^2$ (`efficacy_r2`)", r2_vals))
    lines.append(stat_row("Peak Efficacy (`ctm_peak_efficacy`)", peak_vals))
    lines.append(stat_row("AUC(Efficacy) (`ctm_auc_efficacy`)", auc_vals))
    lines.append("")

    lines.append("Notes:")
    lines.append("- `ctm_peak_efficacy` 与 `ctm_auc_efficacy` 仅在日志 metrics 含对应键时可自动汇总；否则显示 N/A。")
    lines.append("- 统计基于当前 `runs.csv` 中 `module=drug-main` 的全部记录。")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready markdown tables from reproduce logs.")
    parser.add_argument(
        "--runs-csv",
        default="dist/logs/reproduce/runs.csv",
        help="Path to runs.csv generated by reproduce pipeline.",
    )
    parser.add_argument(
        "--out",
        default="dist/logs/reproduce/paper_tables.md",
        help="Output markdown path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_csv = Path(args.runs_csv)
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    records = load_runs(runs_csv)
    md = build_markdown(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"wrote: {out}")
    print(f"records_used: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
