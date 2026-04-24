"""
Confluencia Statistical Testing Module
========================================
Provides statistical significance tests for benchmark comparisons.

Includes:
- Paired t-test with Bonferroni correction
- Wilcoxon signed-rank test (non-parametric alternative)
- Cohen's d effect size
- Bootstrap confidence intervals
- Friedman test (for comparing multiple methods)
- Nemenyi post-hoc test
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired t-test between two sets of scores."""
    from scipy import stats
    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "n_pairs": int(len(diff)),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric)."""
    from scipy import stats
    diff = a - b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False}
    stat, p_value = stats.wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Cohen's d effect size with interpretation."""
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))
    d = float((a.mean() - b.mean()) / s_pooled) if s_pooled > 1e-8 else 0.0
    magnitude = (
        "large" if abs(d) >= 0.8 else
        "medium" if abs(d) >= 0.5 else
        "small" if abs(d) >= 0.2 else
        "negligible"
    )
    return {"d": d, "magnitude": magnitude}


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(float(statistic(sample)))
    stats = np.array(stats)
    alpha = (1 - confidence) / 2
    return {
        "lower": float(np.percentile(stats, alpha * 100)),
        "upper": float(np.percentile(stats, (1 - alpha) * 100)),
        "point_estimate": float(statistic(data)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(p_values)
    corrected_alpha = alpha / max(n_tests, 1)
    results = []
    for i, p in enumerate(p_values):
        results.append({
            "test_index": i,
            "p_value": float(p),
            "significant_before": bool(p < alpha),
            "significant_after": bool(p < corrected_alpha),
            "corrected_alpha": float(corrected_alpha),
        })
    return {"n_tests": n_tests, "corrected_alpha": corrected_alpha, "results": results}


def friedman_test(*score_arrays: np.ndarray) -> Dict[str, Any]:
    """Friedman test for comparing multiple methods across datasets."""
    from scipy import stats
    k = len(score_arrays)
    n = min(len(a) for a in score_arrays)
    if k < 3:
        return {"error": "Friedman test requires at least 3 groups"}

    # Rank within each block
    ranks = np.zeros((n, k))
    for i in range(n):
        scores = [a[i] for a in score_arrays]
        order = np.argsort(scores)
        ranks[i, order] = np.arange(1, k + 1)

    avg_ranks = ranks.mean(axis=0)
    chi2, p_value = stats.friedmanchisquare(*[a[:n] for a in score_arrays])

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n_blocks": n,
        "k_methods": k,
        "average_ranks": avg_ranks.tolist(),
        "significant_005": bool(p_value < 0.05),
    }


def full_comparison_report(
    method_scores: Dict[str, np.ndarray],
    reference_method: str = "moe",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Generate full statistical comparison report.

    Args:
        method_scores: dict mapping method name to array of CV scores (e.g., MAE per fold)
        reference_method: the method to compare others against
        alpha: significance level

    Returns:
        Comprehensive comparison report
    """
    if reference_method not in method_scores:
        raise ValueError(f"Reference method '{reference_method}' not found in scores")

    ref_scores = method_scores[reference_method]
    report: Dict[str, Any] = {
        "reference_method": reference_method,
        "alpha": alpha,
        "comparisons": {},
    }

    p_values = []
    for name, scores in method_scores.items():
        if name == reference_method:
            continue
        n = min(len(ref_scores), len(scores))
        ref = ref_scores[:n]
        other = scores[:n]

        t_test = paired_t_test(ref, other)
        wilcox = wilcoxon_test(ref, other)
        effect = cohens_d(ref, other)
        ci = bootstrap_ci(ref - other)

        p_values.append(t_test["p_value"])

        report["comparisons"][name] = {
            "paired_t_test": t_test,
            "wilcoxon": wilcox,
            "cohens_d": effect,
            "mean_diff_ci_95": ci,
            "ref_mean": float(ref.mean()),
            "other_mean": float(other.mean()),
        }

    # Apply Bonferroni correction
    if p_values:
        report["bonferroni"] = bonferroni_correction(p_values, alpha)

        # Update each comparison with corrected significance
        for i, (name, comp) in enumerate(report["comparisons"].items()):
            bonf = report["bonferroni"]["results"][i]
            comp["paired_t_test"]["significant_bonferroni"] = bonf["significant_after"]

    return report


def format_latex_table(
    method_scores: Dict[str, Dict[str, float]],
    metric: str = "mae",
    best_bold: bool = True,
    include_std: bool = True,
) -> str:
    """Format benchmark results as a LaTeX table string.

    Args:
        method_scores: {method_name: {metric_mean: val, metric_std: val, ...}}
        metric: which metric to display
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance comparison on " + metric.upper() + r"}",
        r"\label{tab:baseline_" + metric + r"}",
        r"\begin{tabular}{l" + "c" * (1 + int(include_std)) + r"}",
        r"\toprule",
    ]

    header = "Method"
    header += " & " + metric.upper() + " Mean"
    if include_std:
        header += " & " + metric.upper() + " Std"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best
    mean_key = f"{metric}_mean" if f"{metric}_mean" in list(method_scores.values())[0] else metric
    std_key = f"{metric}_std" if f"{metric}_std" in list(method_scores.values())[0] else f"{metric}_std"

    best_method = min(method_scores, key=lambda k: method_scores[k].get(mean_key, float("inf")))

    for method, scores in sorted(method_scores.items(), key=lambda x: x[1].get(mean_key, float("inf"))):
        mean_val = scores.get(mean_key, 0.0)
        row = method.replace("_", r"\_")
        if best_bold and method == best_method:
            row += f" & \\textbf{{{mean_val:.4f}}}"
        else:
            row += f" & {mean_val:.4f}"
        if include_std:
            std_val = scores.get(std_key, 0.0)
            row += f" & {std_val:.4f}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)
