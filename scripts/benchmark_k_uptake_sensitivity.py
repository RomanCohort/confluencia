#!/usr/bin/env python3
"""
RNACTM k_uptake Parameter Sensitivity Analysis
===============================================
Reviewer concern: k_uptake (Inj→LNP) was separated from k_release (LNP→Endo).
This script analyzes the sensitivity of key PK outputs to k_uptake variation.

Outputs analyzed:
  - Half-life (terminal elimination half-life)
  - Expression window (time above therapeutic threshold)
  - AUC (area under concentration-time curve)
  - Cmax (peak concentration)

Parameter variations: ±20%, ±50%, ±100% from baseline

Usage:
  # Default (pure Python, no external data needed)
  python benchmark_k_uptake_sensitivity.py

  # Custom output directory
  K_UPTAKE_OUTPUT_DIR=/root/results python benchmark_k_uptake_sensitivity.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers (AutoDL)
import matplotlib.pyplot as plt

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-drug"))
from core.ctm import RNACTMParams, simulate_rna_ctm, infer_rna_ctm_params

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(os.environ.get(
    "K_UPTAKE_OUTPUT_DIR",
    str(SCRIPT_DIR / "benchmark_results"),
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline parameters (from literature-informed defaults)
BASELINE_K_UPTAKE = {
    "IV": 0.80,   # IV: rapid uptake from injection site
    "IM": 0.20,   # IM: moderate uptake
    "SC": 0.15,   # SC: slower uptake from depot
    "ID": 0.10,   # ID: slowest uptake
}

# Parameter variation factors
VARIATION_FACTORS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

# Simulation settings
DOSE = 1.0       # mg/kg
FREQ = 1.0       # QD (once daily)
HORIZON = 168    # 7 days
MODIFICATION = "psi"  # Pseudouridine modification


# ============================================================================
# Sensitivity Analysis Functions
# ============================================================================

def compute_pk_metrics(time_arr: np.ndarray, cyto: np.ndarray, protein: np.ndarray) -> Dict:
    """Compute PK metrics from simulation output.

    Args:
        time_arr: Time points (hours)
        cyto: Cytoplasmic RNA concentration
        protein: Protein concentration

    Returns:
        Dictionary of PK metrics
    """
    # Half-life: terminal elimination phase
    peak_idx = np.argmax(cyto)
    if peak_idx < len(cyto) - 10:
        decay_phase = cyto[peak_idx:]
        time_decay = time_arr[peak_idx:]

        # Log-linear regression for half-life
        log_decay = np.log(decay_phase + 1e-10)
        valid = np.isfinite(log_decay) & (decay_phase > 0.01 * cyto[peak_idx])

        if valid.sum() > 5:
            slope, _ = np.polyfit(time_decay[valid], log_decay[valid], 1)
            half_life = -np.log(2) / slope if slope < 0 else np.inf
        else:
            half_life = np.inf
    else:
        half_life = np.inf

    # Expression window: time above 10% of peak
    threshold = 0.1 * cyto.max()
    above_threshold = cyto > threshold
    if above_threshold.any():
        first_above = np.argmax(above_threshold)
        last_above = len(cyto) - np.argmax(above_threshold[::-1]) - 1
        expression_window = time_arr[last_above] - time_arr[first_above]
    else:
        expression_window = 0.0

    # AUC via trapezoidal rule
    auc_cyto = np.trapz(cyto, time_arr)
    auc_protein = np.trapz(protein, time_arr)

    # Cmax and Tmax
    cmax_cyto = cyto.max()
    tmax_cyto = time_arr[np.argmax(cyto)]
    cmax_protein = protein.max()
    tmax_protein = time_arr[np.argmax(protein)]

    return {
        "half_life_h": float(half_life) if np.isfinite(half_life) else 999.0,
        "expression_window_h": float(expression_window),
        "auc_cyto": float(auc_cyto),
        "auc_protein": float(auc_protein),
        "cmax_cyto": float(cmax_cyto),
        "tmax_cyto_h": float(tmax_cyto),
        "cmax_protein": float(cmax_protein),
        "tmax_protein_h": float(tmax_protein),
    }


def run_sensitivity_analysis(
    route: str = "IV",
    modification: str = "psi",
) -> Tuple[pd.DataFrame, RNACTMParams]:
    """Run k_uptake sensitivity analysis for a given route."""
    print(f"\n[Sensitivity] Route: {route}, Modification: {modification}")

    baseline_k_uptake = BASELINE_K_UPTAKE.get(route, 0.20)
    results = []

    for factor in VARIATION_FACTORS:
        k_uptake = baseline_k_uptake * factor

        # Get baseline params, then override k_uptake
        params = infer_rna_ctm_params(
            route=route,
            modification=modification,
        )

        # Build custom params dict — field names must match RNACTMParams exactly
        params_dict = {
            "k_uptake": k_uptake,
            "k_release": params.k_release,
            "k_escape": params.k_escape,
            "k_translate": params.k_translate,
            "k_degrade": params.k_degrade,
            "k_protein_half": params.k_protein_half,
            "k_immune_clear": params.k_immune_clear,
            "f_liver": params.f_liver,
            "f_spleen": params.f_spleen,
            "f_muscle": params.f_muscle,
            "f_other": params.f_other,
        }

        custom_params = RNACTMParams(**params_dict)

        # Run simulation
        df = simulate_rna_ctm(
            dose=DOSE,
            freq=FREQ,
            params=custom_params,
            horizon=HORIZON,
        )

        # Extract columns — simulate_rna_ctm returns columns named:
        #   time_h, rna_cytoplasmic, protein_translated, ...
        time_arr = df["time_h"].values.astype(np.float64)
        cyto = df["rna_cytoplasmic"].values.astype(np.float64)
        protein = df["protein_translated"].values.astype(np.float64)

        # Compute metrics
        metrics = compute_pk_metrics(time_arr, cyto, protein)

        metrics["k_uptake"] = k_uptake
        metrics["factor"] = factor
        metrics["route"] = route
        metrics["modification"] = modification

        results.append(metrics)

        print(f"  k_uptake={k_uptake:.2f} (factor={factor:.1f}x): "
              f"t1/2={metrics['half_life_h']:.1f}h, "
              f"window={metrics['expression_window_h']:.1f}h, "
              f"AUC={metrics['auc_cyto']:.2f}")

    return pd.DataFrame(results), params


def plot_sensitivity(df: pd.DataFrame, route: str, output_dir: Path):
    """Generate sensitivity plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_list = [
        ("half_life_h", "Half-life (h)"),
        ("expression_window_h", "Expression Window (h)"),
        ("auc_cyto", "AUC (Cytoplasmic RNA)"),
        ("cmax_cyto", "Cmax (Cytoplasmic RNA)"),
    ]

    for ax, (metric, label) in zip(axes.flat, metrics_list):
        baseline = df[df["factor"] == 1.0][metric].values[0]

        ax.plot(df["factor"], df[metric], "o-", linewidth=2, markersize=8)
        ax.axhline(baseline, color="r", linestyle="--", alpha=0.5, label="Baseline")
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("k_uptake Factor (relative to baseline)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs k_uptake ({route})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"k_uptake_sensitivity_{route}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved: {output_file}")


# ============================================================================
# Main Analysis
# ============================================================================

def run_full_analysis():
    """Run full sensitivity analysis across routes."""
    print("=" * 60)
    print("RNACTM k_uptake Parameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Variation factors: {VARIATION_FACTORS}")
    print(f"Baseline k_uptake: {BASELINE_K_UPTAKE}")
    print(f"Output dir: {OUTPUT_DIR}")

    all_results = []

    for route in ["IV", "IM", "SC"]:
        try:
            df, params = run_sensitivity_analysis(route=route, modification=MODIFICATION)
            all_results.append(df)

            # Generate plots
            plot_sensitivity(df, route, OUTPUT_DIR)

        except Exception as e:
            print(f"[Error] Route {route} failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined results
        output_csv = OUTPUT_DIR / "k_uptake_sensitivity_all.csv"
        combined_df.to_csv(output_csv, index=False)
        print(f"\n[Output] Saved: {output_csv}")

        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY: Sensitivity to k_uptake Variation")
        print("=" * 60)

        for route in ["IV", "IM", "SC"]:
            route_df = combined_df[combined_df["route"] == route]
            baseline = route_df[route_df["factor"] == 1.0]

            if len(baseline) > 0:
                print(f"\n{route} Route:")
                print(f"  Baseline k_uptake: {BASELINE_K_UPTAKE[route]:.2f} /h")
                print(f"  Baseline half-life: {baseline['half_life_h'].values[0]:.1f} h")
                print(f"  Baseline expression window: {baseline['expression_window_h'].values[0]:.1f} h")
                print(f"  Baseline AUC: {baseline['auc_cyto'].values[0]:.2f}")

                # Compute % change for key variations
                for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                    variant = route_df[route_df["factor"] == factor]
                    if len(variant) > 0:
                        hl_pct = (variant['half_life_h'].values[0] / baseline['half_life_h'].values[0] - 1) * 100
                        auc_pct = (variant['auc_cyto'].values[0] / baseline['auc_cyto'].values[0] - 1) * 100
                        win_pct = (variant['expression_window_h'].values[0] / baseline['expression_window_h'].values[0] - 1) * 100
                        print(f"  k_uptake x{factor}: t1/2 {hl_pct:+.1f}%, AUC {auc_pct:+.1f}%, window {win_pct:+.1f}%")

        # Save summary JSON
        summary = {
            "config": {
                "variation_factors": VARIATION_FACTORS,
                "baseline_k_uptake": BASELINE_K_UPTAKE,
                "modification": MODIFICATION,
                "dose": DOSE,
                "freq": FREQ,
                "horizon": HORIZON,
            },
            "routes": {},
        }

        for route in ["IV", "IM", "SC"]:
            route_df = combined_df[combined_df["route"] == route]
            summary["routes"][route] = route_df.to_dict("records")

        output_json = OUTPUT_DIR / "k_uptake_sensitivity_summary.json"
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Output] Summary saved: {output_json}")

    return combined_df if all_results else None


if __name__ == "__main__":
    run_full_analysis()