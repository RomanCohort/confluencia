"""Residual Analysis and SHAP Visualization Utilities.

Provides diagnostic plots for model evaluation including
residual analysis, QQ plots, and SHAP feature importance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json


def generate_residual_plots(
    y_true: list,
    y_pred: list,
    output_dir: str = ".",
    prefix: str = "residual",
    format: str = "plotly",
) -> Dict[str, Any]:
    """Generate residual analysis plots.

    Creates:
    - Residuals vs Predicted (heteroscedasticity check)
    - Residual histogram (normality check)
    - QQ plot (distribution check)
    - Scale-Location plot

    Args:
        y_true: True values
        y_pred: Predicted values
        output_dir: Output directory
        prefix: File name prefix
        format: "plotly" for interactive, "matplotlib" for static

    Returns:
        Dict with plot data and metrics
    """
    import numpy as np

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    residuals = y_true - y_pred
    standardized = (residuals - residuals.mean()) / (residuals.std() + 1e-10)

    # Compute metrics
    metrics = _compute_regression_metrics(y_true, y_pred)

    if format == "plotly":
        return _generate_plotly_residual(y_true, y_pred, residuals, standardized, metrics, prefix)
    else:
        return _generate_matplotlib_residual(y_true, y_pred, residuals, standardized, output_dir, prefix, metrics)


def _compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics."""
    import numpy as np

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pearson correlation
    if len(y_true) > 1:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(corr),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
    }


def _generate_plotly_residual(y_true, y_pred, residuals, standardized, metrics, prefix) -> Dict:
    """Generate Plotly JSON specs for residual analysis."""
    specs = {}

    # 1. Predicted vs Actual
    specs[f"{prefix}_pred_vs_actual"] = {
        "data": [{
            "x": y_true.tolist(),
            "y": y_pred.tolist(),
            "mode": "markers",
            "type": "scatter",
            "marker": {"color": "#89b4fa", "size": 6, "opacity": 0.6},
            "name": "Predictions",
        }, {
            "x": [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
            "y": [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
            "mode": "lines",
            "line": {"color": "#f38ba8", "dash": "dash"},
            "name": "Perfect",
        }],
        "layout": {
            "title": f"Predicted vs Actual (R²={metrics['r2']:.4f})",
            "xaxis": {"title": "Actual"},
            "yaxis": {"title": "Predicted"},
            "plot_bgcolor": "#1e1e2e",
            "paper_bgcolor": "#1e1e2e",
            "font": {"color": "#cdd6f4"},
        }
    }

    # 2. Residuals vs Predicted
    specs[f"{prefix}_residual_vs_pred"] = {
        "data": [{
            "x": y_pred.tolist(),
            "y": residuals.tolist(),
            "mode": "markers",
            "type": "scatter",
            "marker": {"color": "#a6e3a1", "size": 6, "opacity": 0.6},
            "name": "Residuals",
        }, {
            "x": [y_pred.min(), y_pred.max()],
            "y": [0, 0],
            "mode": "lines",
            "line": {"color": "#f38ba8", "dash": "dash"},
            "name": "Zero line",
        }],
        "layout": {
            "title": "Residuals vs Predicted",
            "xaxis": {"title": "Predicted"},
            "yaxis": {"title": "Residual"},
            "plot_bgcolor": "#1e1e2e",
            "paper_bgcolor": "#1e1e2e",
            "font": {"color": "#cdd6f4"},
        }
    }

    # 3. Residual histogram
    specs[f"{prefix}_residual_hist"] = {
        "data": [{
            "x": residuals.tolist(),
            "type": "histogram",
            "marker": {"color": "#cba6f7"},
            "nbinsx": 30,
            "name": "Residuals",
        }],
        "layout": {
            "title": "Residual Distribution",
            "xaxis": {"title": "Residual"},
            "yaxis": {"title": "Count"},
            "plot_bgcolor": "#1e1e2e",
            "paper_bgcolor": "#1e1e2e",
            "font": {"color": "#cdd6f4"},
        }
    }

    # 4. QQ plot data
    import numpy as np
    sorted_res = np.sort(standardized)
    n = len(sorted_res)
    theoretical_quantiles = [__import__('scipy').stats.norm.ppf((i - 0.5) / n) for i in range(1, n + 1)]

    specs[f"{prefix}_qq"] = {
        "data": [{
            "x": theoretical_quantiles,
            "y": sorted_res.tolist(),
            "mode": "markers",
            "type": "scatter",
            "marker": {"color": "#f9e2af", "size": 6, "opacity": 0.6},
            "name": "QQ",
        }, {
            "x": [min(theoretical_quantiles), max(theoretical_quantiles)],
            "y": [min(theoretical_quantiles), max(theoretical_quantiles)],
            "mode": "lines",
            "line": {"color": "#f38ba8", "dash": "dash"},
            "name": "Reference",
        }],
        "layout": {
            "title": "Q-Q Plot (Normality Check)",
            "xaxis": {"title": "Theoretical Quantiles"},
            "yaxis": {"title": "Standardized Residuals"},
            "plot_bgcolor": "#1e1e2e",
            "paper_bgcolor": "#1e1e2e",
            "font": {"color": "#cdd6f4"},
        }
    }

    return {
        "specs": specs,
        "metrics": metrics,
    }


def _generate_matplotlib_residual(y_true, y_pred, residuals, standardized, output_dir, prefix, metrics) -> Dict:
    """Generate matplotlib static plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        files = {}

        # 1. Predicted vs Actual
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Predicted vs Actual (R²={metrics["r2"]:.4f})')
        ax.legend()
        path = str(output_dir / f"{prefix}_pred_vs_actual.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        files["pred_vs_actual"] = path

        # 2. Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')
        path = str(output_dir / f"{prefix}_residual_vs_pred.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        files["residual_vs_pred"] = path

        # 3. Histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title('Residual Distribution')
        path = str(output_dir / f"{prefix}_residual_hist.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        files["histogram"] = path

        return {
            "files": files,
            "metrics": metrics,
        }

    except ImportError:
        return {
            "error": "matplotlib not available",
            "metrics": metrics,
        }


def generate_shap_plot(
    model,
    X,
    output_path: Optional[str] = None,
    format: str = "plotly",
    max_display: int = 20,
) -> Dict[str, Any]:
    """Generate SHAP feature importance visualization.

    Args:
        model: Trained model (sklearn, xgboost, etc.)
        X: Feature dataframe or array
        output_path: Output file path
        format: "plotly" or "matplotlib"
        max_display: Max features to display

    Returns:
        Dict with plot data
    """
    try:
        import shap
        import numpy as np
    except ImportError:
        return {"error": "SHAP not installed. Install with: pip install shap"}

    # Compute SHAP values
    try:
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') or hasattr(model, 'estimators_') else shap.KernelExplainer(model.predict_proba, X[:100])
        else:
            explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') or hasattr(model, 'estimators_') else shap.KernelExplainer(model.predict, X[:100])

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first class for classification

        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Sort features by importance
        feature_names = getattr(X, 'columns', [f"feature_{i}" for i in range(X.shape[1])])
        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()

        sorted_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_values = mean_abs_shap[sorted_indices].tolist()

        if format == "plotly":
            return {
                "plotly_spec": {
                    "data": [{
                        "x": sorted_values,
                        "y": sorted_features,
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": sorted_values, "colorscale": "Viridis"},
                    }],
                    "layout": {
                        "title": "SHAP Feature Importance",
                        "xaxis": {"title": "Mean |SHAP value|"},
                        "yaxis": {"autorange": "reversed"},
                        "plot_bgcolor": "#1e1e2e",
                        "paper_bgcolor": "#1e1e2e",
                        "font": {"color": "#cdd6f4"},
                        "height": max(400, max_display * 25),
                    }
                },
                "feature_importance": dict(zip(sorted_features, sorted_values)),
            }
        else:
            if output_path:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
                ax.barh(sorted_features, sorted_values)
                ax.set_xlabel('Mean |SHAP value|')
                ax.set_title('SHAP Feature Importance')
                ax.invert_yaxis()
                fig.tight_layout()
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {
                "output_path": output_path,
                "feature_importance": dict(zip(sorted_features, sorted_values)),
            }

    except Exception as e:
        return {"error": f"SHAP computation failed: {str(e)}"}
