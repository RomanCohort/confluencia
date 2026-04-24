from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


def _ensure_out_dir(out_dir: Union[str, Path]) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_regression_diagnostic_plots(
    *,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    out_dir: Union[str, Path],
    prefix: str = "reg",
    title: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Save simple regression diagnostics as PNGs."""

    # Ensure headless-friendly backend
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true shape {yt.shape} != y_pred shape {yp.shape}")

    out = _ensure_out_dir(out_dir)
    residual = yp - yt

    # 1) y_true vs y_pred
    f1, ax1 = plt.subplots(figsize=(5.5, 5.0))
    ax1.scatter(yt, yp, s=10, alpha=0.6)
    mn = float(np.nanmin([yt.min(), yp.min()])) if len(yt) else 0.0
    mx = float(np.nanmax([yt.max(), yp.max()])) if len(yt) else 1.0
    ax1.plot([mn, mx], [mn, mx], color="black", linewidth=1)
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.set_title(title or "y_true vs y_pred")
    p1 = out / f"{prefix}_scatter.png"
    f1.tight_layout()
    f1.savefig(p1, dpi=160)
    plt.close(f1)

    # 2) residual histogram
    f2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    ax2.hist(residual[~np.isnan(residual)], bins=40, alpha=0.85)
    ax2.set_xlabel("residual (y_pred - y_true)")
    ax2.set_ylabel("count")
    ax2.set_title(title or "residual histogram")
    p2 = out / f"{prefix}_residual_hist.png"
    f2.tight_layout()
    f2.savefig(p2, dpi=160)
    plt.close(f2)

    # 3) residual vs pred
    f3, ax3 = plt.subplots(figsize=(5.5, 4.0))
    ax3.scatter(yp, residual, s=10, alpha=0.6)
    ax3.axhline(0.0, color="black", linewidth=1)
    ax3.set_xlabel("y_pred")
    ax3.set_ylabel("residual")
    ax3.set_title(title or "residual vs prediction")
    p3 = out / f"{prefix}_residual_vs_pred.png"
    f3.tight_layout()
    f3.savefig(p3, dpi=160)
    plt.close(f3)

    return p1, p2, p3


def save_series_histogram(
    *,
    values: Sequence[float],
    out_dir: Union[str, Path],
    filename: str,
    title: str,
    bins: int = 60,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = _ensure_out_dir(out_dir)
    v = np.asarray(values, dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(v[~np.isnan(v)], bins=int(bins), alpha=0.85)
    ax.set_title(str(title))
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()

    p = out / str(filename)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p
