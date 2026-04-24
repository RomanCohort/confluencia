from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np


@dataclass(frozen=True)
class SensitivityResult:
    """Local sensitivity result for a single input point.

    grad: numerical gradient d(pred)/d(x_i)
    importance: default = abs(grad)
    """

    pred: float
    grad: np.ndarray
    importance: np.ndarray
    feature_names: List[str]


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1)


def numerical_input_gradient(
    model,
    x: np.ndarray,
    eps: float = 1e-3,
    batch_size: int = 2048,
) -> Tuple[float, np.ndarray]:
    """Estimate gradient of model.predict at x via central difference.

    Works for any sklearn-like regressor exposing .predict(X)->(n,).

    Note: This is a *local* sensitivity, not a global feature importance.
    """

    x0 = _as_1d(x)
    n = int(x0.shape[0])

    # Baseline prediction
    pred0 = float(np.asarray(model.predict(x0.reshape(1, -1))).reshape(-1)[0])

    if n == 0:
        return pred0, np.zeros((0,), dtype=np.float32)

    grad = np.empty((n,), dtype=np.float32)
    step = max(1, int(batch_size) // 2)
    for start in range(0, n, step):
        end = min(n, start + step)
        width = end - start
        X = np.repeat(x0.reshape(1, -1), repeats=2 * width, axis=0)
        for j, i in enumerate(range(start, end)):
            X[2 * j, i] += eps
            X[2 * j + 1, i] -= eps

        preds = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
        for j, i in enumerate(range(start, end)):
            grad[i] = float((preds[2 * j] - preds[2 * j + 1]) / (2.0 * eps))

    return pred0, grad


def sensitivity_from_bundle(
    bundle,
    x: np.ndarray,
    eps: float = 1e-3,
    importance: str = "abs_grad",
    batch_size: int = 2048,
) -> SensitivityResult:
    pred, grad = numerical_input_gradient(bundle.model, x=x, eps=eps, batch_size=batch_size)

    if importance == "abs_grad":
        imp = np.abs(grad)
    elif importance == "grad_x":
        imp = np.abs(grad * _as_1d(x))
    else:
        raise ValueError(f"Unknown importance: {importance}")

    feature_names = list(getattr(bundle, "feature_names", []))
    if feature_names and len(feature_names) != len(grad):
        # Keep analysis usable even if names mismatch; fall back to indices.
        feature_names = [f"x{i}" for i in range(len(grad))]

    return SensitivityResult(
        pred=float(pred),
        grad=grad,
        importance=imp,
        feature_names=feature_names,
    )


def top_features(
    feature_names_or_result,
    importance_or_k=None,
    grad=None,
    k: int = 15,
) -> List[Tuple[str, float, float]]:
    """Return (feature_name, importance, grad) sorted by importance desc.

    Supports two calling conventions:
      1. top_features(result: SensitivityResult, k=15) — legacy
      2. top_features(feature_names, importance, grad, k=15) — pipeline
    """
    if isinstance(feature_names_or_result, SensitivityResult):
        result = feature_names_or_result
        kk = max(1, int(importance_or_k or k))
        idx = np.argsort(-result.importance)[:kk]
        out: List[Tuple[str, float, float]] = []
        for i in idx:
            out.append((result.feature_names[int(i)], float(result.importance[int(i)]), float(result.grad[int(i)])))
        return out

    # Pipeline convention: (feature_names, importance, grad, k)
    names = feature_names_or_result
    imp = np.asarray(importance_or_k, dtype=np.float32).reshape(-1)
    kk = max(1, int(k))
    idx = np.argsort(-imp)[:kk]
    out = []
    for i in idx:
        g = float(grad[int(i)]) if grad is not None else 0.0
        out.append((str(names[int(i)]), float(imp[int(i)]), g))
    return out


def neighborhood_importance(feature_names, imp) -> Dict[str, float]:
    """Group importance by feature-name neighborhood/prefix.

    Accepts raw (feature_names, importance_array) from the pipeline.
    """
    imp = np.asarray(imp, dtype=np.float32).reshape(-1)

    groups: Dict[str, float] = {
        "hydropathy_middle": 0.0,
        "hydropathy_ends": 0.0,
        "nonpolar_middle": 0.0,
        "nonpolar_ends": 0.0,
        "global_hydropathy": 0.0,
        "aac": 0.0,
        "env": 0.0,
        "other": 0.0,
    }

    for name, v in zip(feature_names, imp):
        v = float(v)
        if str(name).startswith("env_"):
            groups["env"] += v
        elif str(name).startswith("aac_"):
            groups["aac"] += v
        elif name == "hydropathy_mean":
            groups["global_hydropathy"] += v
        elif name in ("hydropathy_mean_mid",):
            groups["hydropathy_middle"] += v
        elif name in ("hydropathy_mean_n", "hydropathy_mean_c"):
            groups["hydropathy_ends"] += v
        elif name in ("frac_nonpolar_mid",):
            groups["nonpolar_middle"] += v
        elif name in ("frac_nonpolar_n", "frac_nonpolar_c"):
            groups["nonpolar_ends"] += v
        else:
            groups["other"] += v

    return groups


def group_importance(result: SensitivityResult) -> Dict[str, float]:
    """Coarse grouping for wet-lab interpretation."""

    groups: Dict[str, float] = {
        "hydropathy_middle": 0.0,
        "hydropathy_ends": 0.0,
        "nonpolar_middle": 0.0,
        "nonpolar_ends": 0.0,
        "global_hydropathy": 0.0,
        "aac": 0.0,
        "env": 0.0,
        "other": 0.0,
    }

    for name, imp in zip(result.feature_names, result.importance):
        v = float(imp)
        if name.startswith("env_"):
            groups["env"] += v
        elif name.startswith("aac_"):
            groups["aac"] += v
        elif name == "hydropathy_mean":
            groups["global_hydropathy"] += v
        elif name in ("hydropathy_mean_mid",):
            groups["hydropathy_middle"] += v
        elif name in ("hydropathy_mean_n", "hydropathy_mean_c"):
            groups["hydropathy_ends"] += v
        elif name in ("frac_nonpolar_mid",):
            groups["nonpolar_middle"] += v
        elif name in ("frac_nonpolar_n", "frac_nonpolar_c"):
            groups["nonpolar_ends"] += v
        else:
            groups["other"] += v

    return groups





def _normalize_importance_array(imp: np.ndarray) -> np.ndarray:
    s = float(np.sum(imp))
    if s <= 0:
        return imp.astype(np.float32)
    return (imp / s).astype(np.float32)


def _detect_input_type_from_features(feature_names: List[str]) -> str:
    """Heuristic to detect input data type from feature names.

    Returns one of: 'epitope', 'drug', 'env_only', 'unknown'.
    """
    fn = [str(n).lower() for n in feature_names]
    if any(n.startswith("aac_") or "hydropathy" in n or "frac_nonpolar" in n for n in fn):
        return "epitope"
    if any("smiles" in n or "mol" in n or "rdkit" in n for n in fn):
        return "drug"
    if any(n.startswith("env_") for n in fn) and len(fn) <= 6:
        return "env_only"
    return "unknown"


def generate_diverse_suggestions(
    result: SensitivityResult,
    max_suggestions: int = 3,
    *,
    options: Optional[Dict[str, bool]] = None,
    input_type: Optional[str] = None,
) -> List[str]:
    """Produce diverse, actionable suggestions.

    options supports toggles (all default True):
      - mutation_hints
      - experimental_design
      - validation
      - model_checks
      - assay_suggestions

    `input_type` can override automatic detection (values from
    `_detect_input_type_from_features`).
    """

    # Default options
    opts = {
        "mutation_hints": True,
        "experimental_design": True,
        "validation": True,
        "model_checks": True,
        "assay_suggestions": True,
    }
    if options:
        opts.update(options)

    # Normalize and pick top features
    norm = _normalize_importance_array(result.importance)
    idx = np.argsort(-norm)[: min(12, len(norm))]
    top_feats = [(result.feature_names[int(i)], float(norm[int(i)]), float(result.grad[int(i)])) for i in idx]

    groups = group_importance(result)
    suggestions: List[str] = []

    # Detect input type if not provided
    itype = input_type or _detect_input_type_from_features(result.feature_names)

    # 1) Sequence / epitope suggestions
    if itype == "epitope":
        mid = groups.get("hydropathy_middle", 0.0) + groups.get("nonpolar_middle", 0.0)
        ends = groups.get("hydropathy_ends", 0.0) + groups.get("nonpolar_ends", 0.0)
        if mid > ends and mid > 0:
            if opts.get("mutation_hints"):
                suggestions.append(
                    "序列设计：中部疏水/非极性特征较强。建议在中部做系统的定点/扫描突变，评估活性与溶解性折衷，并优先验证中部变体。"
                )
        elif ends > mid and ends > 0:
            if opts.get("mutation_hints"):
                suggestions.append(
                    "序列设计：两端疏水/非极性影响更明显。考虑在两端引入亲水化改造或短标签以改善溶解性/可表达性，同时检测免疫原性变化。"
                )
        else:
            if opts.get("mutation_hints"):
                suggestions.append(
                    "序列设计：未观察到明显区域性，建议做 AAC/逐位扫描并结合位点互作分析以定位关键残基。"
                )

    # 2) Drug-like suggestions
    if itype == "drug":
        if opts.get("mutation_hints"):
            suggestions.append(
                "化学修饰建议：若分子疏水性对目标敏感，考虑引入极性基团或短链PEG以改善溶解性；若极性更敏感，可优化疏水面以提高结合。"
            )

    # 3) Experimental design (env features)
    if opts.get("experimental_design"):
        if any(name.startswith("env_") for name, *_ in top_feats):
            suggestions.append(
                "实验设计：环境条件对预测重要。建议构建一组参数梯度（例如 pH、温度、盐强度）并做并行对照，纪录批间变异以分离序列效应与环境效应。"
            )
        else:
            suggestions.append(
                "实验设计：建议以高通量小规模突变/候选替换为主的验证方案，采用随机化与重复以量化不确定性。"
            )

    # 4) Validation and statistical checks
    if opts.get("validation"):
        top_desc = ", ".join([f"{n}({v:.2g})" for n, v, _ in top_feats[:5]]) if top_feats else "无明显特征"
        suggestions.append(
            "验证与分析：做多样本/交叉验证或对不同序列取样的全局敏感性分析，报告前5个影响特征："
            + top_desc
            + "。考虑采用积分梯度或置换重要性作为补充。"
        )

    # 5) Model / assay checks
    if opts.get("model_checks"):
        suggestions.append(
            "模型稳健性：若可能，使用不同模型/随机种子复现敏感性结果；对重要特征做置换检验以评估假阳性。"
        )

    # 6) Assay-specific suggestions (if assay metadata exists)
    if opts.get("assay_suggestions"):
        if any(n.startswith("env_") for n in result.feature_names):
            suggestions.append(
                "测定建议：记录并标准化关键实验条件（pH、温度、缓冲液），并在报告中包含这些参数以便模型解释。"
            )

    # Context-aware extra: per-feature actionable hints
    if opts.get("mutation_hints"):
        for name, v, g in top_feats[:6]:
            if name.startswith("aac_"):
                aa = name.split("aac_")[-1]
                if g > 0:
                    suggestions.append(f"候选替换：增加残基 {aa} 的比例或在关键位点引入{aa}，可能提升响应（局部梯度={g:.3g}）。")
                elif g < 0:
                    suggestions.append(f"候选替换：减少残基 {aa} 或替换为亲水残基，可能提升响应（局部梯度={g:.3g}）。")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out[:max_suggestions]


def wetlab_takeaway(result_or_groups: Dict[str, float] | SensitivityResult) -> str:
    """Generate a more diverse, actionable interpretation message.

    Accepts either the raw `groups` dict (legacy) or a `SensitivityResult`.
    """

    if isinstance(result_or_groups, SensitivityResult):
        result = result_or_groups
        groups = group_importance(result)
    else:
        groups = result_or_groups
        result = None

    # Quick regional summary
    mid = float(groups.get("hydropathy_middle", 0.0) + groups.get("nonpolar_middle", 0.0))
    ends = float(groups.get("hydropathy_ends", 0.0) + groups.get("nonpolar_ends", 0.0))
    global_h = float(groups.get("global_hydropathy", 0.0))

    header = []
    if mid > max(ends, global_h) and mid > 0:
        header.append("分析提示：中部疏水/非极性特征较为突出，可能影响结合/稳定性。")
    else:
        items = sorted(groups.items(), key=lambda kv: -kv[1])
        top_name, top_val = items[0] if items else ("other", 0.0)
        mapping = {
            "hydropathy_middle": "中部疏水性（hydropathy_mean_mid）",
            "hydropathy_ends": "两端疏水性（hydropathy_mean_n/c）",
            "nonpolar_middle": "中部非极性比例（frac_nonpolar_mid）",
            "nonpolar_ends": "两端非极性比例（frac_nonpolar_n/c）",
            "global_hydropathy": "全局疏水性（hydropathy_mean）",
            "aac": "氨基酸组成（AAC）",
            "env": "实验条件参数（env_*）",
            "other": "其他特征",
        }

        header.append(
            f" 当前输入点附近影响更大的因素是：{mapping.get(top_name, top_name)}（重要性≈{top_val:.3g}）。"
        )

    # Generate diverse suggestions
    suggestions: List[str] = []
    if result is not None:
        suggestions = generate_diverse_suggestions(result, max_suggestions=3)
    else:
        # Fallback generic suggestions when only groups are provided
        suggestions = [
            "建议：结合更多样本/多序列做全局统计后再下结论。",
            "建议：若关注序列性能，考虑进行逐位扫描/候选突变验证。",
            "建议：监测并记录实验条件以便在模型中分离环境效应。",
        ]

    body = "\n".join([f"{i+1}. {s}" for i, s in enumerate(suggestions)])
    return "\n".join(header + ["\n建议：", body])


def sensitivity_report(result: SensitivityResult, top_k: int = 10, suggestion_options: Optional[Dict[str, bool]] = None) -> Dict[str, object]:
    """Return a structured report (dict) with enriched data for display.

    The report is JSON-serializable and contains prediction, raw gradients,
    importances, normalized importances, top feature breakdown, group
    importances, generated suggestions, and a human-readable takeaway.
    """

    # Basic arrays
    grad = np.asarray(result.grad, dtype=np.float32).reshape(-1)
    imp = np.asarray(result.importance, dtype=np.float32).reshape(-1)
    imp_norm = _normalize_importance_array(imp)

    # Top features
    k = max(1, int(min(top_k, len(imp))))
    idx = list(np.argsort(-imp)[:k])
    top_list: List[Dict[str, object]] = []
    for i in idx:
        name = result.feature_names[int(i)]
        top_list.append(
            {
                "index": int(i),
                "name": name,
                "importance": float(imp[int(i)]),
                "importance_norm": float(imp_norm[int(i)]),
                "grad": float(grad[int(i)]),
                "grad_sign": int(np.sign(grad[int(i)])),
            }
        )

    groups = group_importance(result)

    # Simple mutation hints based on feature name patterns
    mutation_hints: List[str] = []
    for item in cast(List[Dict[str, Any]], top_list)[:6]:
        n = str(item.get("name", ""))
        g = float(item.get("grad", 0.0))
        if n.startswith("aac_"):
            aa = n.split("aac_")[-1]
            if g > 0:
                mutation_hints.append(f"增加残基 {aa} 的比例可能提升预测响应（梯度={g:.3g}）。")
            elif g < 0:
                mutation_hints.append(f"减少残基 {aa} 的比例可能提升预测响应（梯度={g:.3g}）。")
        elif "hydropathy" in n:
            if g > 0:
                mutation_hints.append(f"增加{n} 对预测有正向影响，考虑引入更疏水残基（梯度={g:.3g}）。")
            elif g < 0:
                mutation_hints.append(f"降低{n} 可能有利，考虑引入亲水残基或标签（梯度={g:.3g}）。")

    # Default suggestion options included in report metadata
    default_options = {
        "mutation_hints": True,
        "experimental_design": True,
        "validation": True,
        "model_checks": True,
        "assay_suggestions": True,
    }
    opts = default_options if suggestion_options is None else {**default_options, **suggestion_options}
    suggestions = generate_diverse_suggestions(result, max_suggestions=4, options=opts)
    takeaway = wetlab_takeaway(result)

    report: Dict[str, object] = {
        "pred": float(result.pred),
        "n_features": int(len(imp)),
        "feature_names": list(result.feature_names),
        "grad": [float(x) for x in grad.tolist()],
        "importance": [float(x) for x in imp.tolist()],
        "importance_norm": [float(x) for x in imp_norm.tolist()],
        "top_features": top_list,
        "groups": groups,
        "detected_input_type": _detect_input_type_from_features(result.feature_names),
        "suggestion_options": opts,
        "mutation_hints": mutation_hints,
        "suggestions": suggestions,
        "takeaway_text": takeaway,
    }

    return report


def format_sensitivity_report(report: Dict[str, object], top_k: int = 10) -> str:
    """Pretty-print the structured report into a readable Chinese summary."""

    lines: List[str] = []
    lines.append(f"预测值: {report.get('pred', 'N/A')}")
    lines.append(f"特征数: {report.get('n_features', 0)}")

    lines.append("\n分组重要性:")
    for k, v in cast(Dict[str, Any], report.get("groups", {})).items():
        try:
            vv = float(v)
        except Exception:
            vv = 0.0
        lines.append(f"- {k}: {vv:.3g}")

    lines.append("\nTop 特征:")
    for t in cast(List[Dict[str, Any]], report.get("top_features", []))[:top_k]:
        td = t
        lines.append(
            f"- {td.get('name', '')} (idx={td.get('index', '')}): importance={float(td.get('importance', 0.0)):.3g}, norm={float(td.get('importance_norm', 0.0)):.3g}, grad={float(td.get('grad', 0.0)):.3g}"
        )

    mutation_hints_list = cast(List[str], report.get("mutation_hints", []))
    if mutation_hints_list:
        lines.append("\n突变提示:")
        for s in mutation_hints_list:
            lines.append(f"- {s}")

    lines.append("\n建议:")
    for s in cast(List[str], report.get("suggestions", [])):
        lines.append(f"- {s}")

    lines.append("\n结论:")
    lines.append(str(report.get("takeaway_text", "")))

    return "\n".join(lines)
