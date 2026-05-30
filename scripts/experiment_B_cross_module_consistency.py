#!/usr/bin/env python3
"""
Confluencia Experiment B: Cross-Module Consistency vs No Consistency

Compares drug pipeline predictions with/without multitask consistency
constraints and adaptive adjustment.

Demonstrates that cross-module constraints produce more physiologically
plausible predictions (lower extreme toxicity, lower extreme inflammation).

Usage: python experiment_B_cross_module_consistency.py
Output: benchmarks/results/experiment_B_cross_module_consistency.json
"""

import sys
import os
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "confluencia-2.0-drug"))

# We need to import and modify the drug pipeline
# The key functions are _apply_multitask_consistency and _apply_adaptive_adjustment
# Since these are internal methods, we'll run the pipeline with adaptive_enabled=True/False
# and compare the outputs.

from confluencia_2_0_drug.core.pipeline import run_pipeline


# ── Test molecules ────────────────────────────────────────────────────

# Mix of molecules with different ADMET profiles
TEST_MOLECULES = [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin", "dose": 200, "freq": 2, "treatment_time": 72},
    {"smiles": "O=C1NC(=S)CS1", "name": "Rhodanine (PAINS)", "dose": 100, "freq": 3, "treatment_time": 48},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen", "dose": 400, "freq": 2, "treatment_time": 72},
    {"smiles": "c1ccccc1", "name": "Benzene (simple)", "dose": 50, "freq": 1, "treatment_time": 24},
    {"smiles": "C1CO1", "name": "Ethylene oxide (reactive)", "dose": 10, "freq": 1, "treatment_time": 24},
]


def make_dataframe(molecules):
    """Create input DataFrame for drug pipeline."""
    rows = []
    for mol in molecules:
        rows.append({
            "smiles": mol["smiles"],
            "dose": mol["dose"],
            "freq": mol["freq"],
            "treatment_time": mol["treatment_time"],
            "group_id": "G0",
        })
    return pd.DataFrame(rows)


def extract_key_outputs(out_df, molecule_names):
    """Extract key prediction columns from pipeline output."""
    results = []
    for i, name in enumerate(molecule_names):
        row = out_df.iloc[i]
        result = {
            "molecule_name": name,
            "smiles": TEST_MOLECULES[i]["smiles"],
            "dose": TEST_MOLECULES[i]["dose"],
        }

        # Core predictions
        key_cols = [
            "efficacy_pred", "target_binding_pred", "immune_activation_pred",
            "inflammation_risk_pred", "genotoxicity_risk_pred",
            "toxicity_risk_pred", "immune_cell_activation_pred",
            "consistency_score",
        ]

        for col in key_cols:
            if col in out_df.columns:
                result[col] = float(row[col])
            else:
                result[col] = None

        # CTM parameters
        ctm_cols = ["ctm_ka", "ctm_kd", "ctm_ke", "ctm_km", "ctm_signal_gain"]
        for col in ctm_cols:
            if col in out_df.columns:
                result[col] = float(row[col])
            else:
                result[col] = None

        # PK outputs
        pk_cols = [
            "pkpd_half_life_h", "pkpd_cmax_mg_per_l", "pkpd_tmax_h",
            "pkpd_auc_conc", "pkpd_auc_effect",
        ]
        for col in pk_cols:
            if col in out_df.columns:
                result[col] = float(row[col])
            else:
                result[col] = None

        # Adaptive adjustment outputs (if available)
        adaptive_cols = [
            "adaptive_confidence", "adaptive_risk_pressure",
            "adaptive_dose_factor", "adaptive_freq_factor",
        ]
        for col in adaptive_cols:
            if col in out_df.columns:
                result[col] = float(row[col])
            else:
                result[col] = None

        results.append(result)

    return results


def compute_physiological_plausibility(results):
    """Evaluate whether predictions are physiologically plausible.

    Checks:
    - Toxicity should not be 0 (all drugs have some toxicity)
    - Inflammation should not be 0
    - Efficacy should not be extremely high (>0.95) unless binding is also high
    - CTM parameters should be in plausible ranges
    """
    implausibility_count = 0
    implausibility_details = []

    for r in results:
        name = r["molecule_name"]

        # Check 1: Zero toxicity
        tox = r.get("toxicity_risk_pred")
        if tox is not None and tox < 0.05:
            implausibility_count += 1
            implausibility_details.append(f"{name}: toxicity {tox:.3f} implausibly low (all drugs have baseline toxicity)")

        # Check 2: Zero inflammation
        inf = r.get("inflammation_risk_pred")
        if inf is not None and inf < 0.05:
            implausibility_count += 1
            implausibility_details.append(f"{name}: inflammation {inf:.3f} implausibly low")

        # Check 3: Extreme efficacy without binding
        eff = r.get("efficacy_pred")
        bind = r.get("target_binding_pred")
        if eff is not None and bind is not None and eff > 0.95 and bind < 0.3:
            implausibility_count += 1
            implausibility_details.append(f"{name}: efficacy {eff:.3f} implausibly high given binding {bind:.3f}")

        # Check 4: CTM parameters out of range
        ka = r.get("ctm_ka")
        kd = r.get("ctm_kd")
        if ka is not None and ka > 0.9:
            implausibility_count += 1
            implausibility_details.append(f"{name}: ka {ka:.3f} exceeds plausible range [0.02, 0.9]")
        if kd is not None and kd > 0.9:
            implausibility_count += 1
            implausibility_details.append(f"{name}: kd {kd:.3f} exceeds plausible range [0.02, 0.9]")

    return {
        "implausibility_count": implausibility_count,
        "implausibility_details": implausibility_details,
        "total_checks": len(results) * 4,
        "plausibility_rate": 1.0 - (implausibility_count / (len(results) * 4)) if len(results) > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Confluencia Experiment B: Cross-Module Consistency Comparison")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    molecule_names = [m["name"] for m in TEST_MOLECULES]
    df = make_dataframe(TEST_MOLECULES)

    all_results = {
        "experiment": "B_cross_module_consistency",
        "timestamp": datetime.now().isoformat(),
        "n_molecules": len(TEST_MOLECULES),
    }

    # ── Run 1: WITH adaptive adjustment (consistency + adaptive enabled) ──
    print("Run 1: Pipeline WITH adaptive adjustment (consistency + adaptive)...")
    try:
        out_with, curves_with, artifacts_with = run_pipeline(
            df,
            compute_mode="auto",
            model_backend="moe",
            adaptive_enabled=True,
            adaptive_strength=0.2,
        )
        results_with = extract_key_outputs(out_with, molecule_names)
        plausibility_with = compute_physiological_plausibility(results_with)

        print(f"  Completed. Plausibility: {plausibility_with['plausibility_rate']:.1%}")
        print(f"  Implausibility count: {plausibility_with['implausibility_count']}")

        # Print key metrics
        for r in results_with:
            print(f"    {r['molecule_name']}: eff={r.get('efficacy_pred', 'N/A')}, "
                  f"tox={r.get('toxicity_risk_pred', 'N/A')}, "
                  f"inf={r.get('inflammation_risk_pred', 'N/A')}, "
                  f"consistency={r.get('consistency_score', 'N/A')}")

    except Exception as e:
        results_with = {"error": str(e), "traceback": traceback.format_exc()}
        plausibility_with = {"error": str(e)}
        print(f"  ERROR: {str(e)}")

    # ── Run 2: WITHOUT adaptive adjustment (baseline) ──
    print("\nRun 2: Pipeline WITHOUT adaptive adjustment (baseline)...")
    try:
        out_without, curves_without, artifacts_without = run_pipeline(
            df,
            compute_mode="auto",
            model_backend="moe",
            adaptive_enabled=False,  # Disable adaptive adjustment
        )
        results_without = extract_key_outputs(out_without, molecule_names)
        plausibility_without = compute_physiological_plausibility(results_without)

        print(f"  Completed. Plausibility: {plausibility_without['plausibility_rate']:.1%}")
        print(f"  Implausibility count: {plausibility_without['implausibility_count']}")

        for r in results_without:
            print(f"    {r['molecule_name']}: eff={r.get('efficacy_pred', 'N/A')}, "
                  f"tox={r.get('toxicity_risk_pred', 'N/A')}, "
                  f"inf={r.get('inflammation_risk_pred', 'N/A')}")

    except Exception as e:
        results_without = {"error": str(e), "traceback": traceback.format_exc()}
        plausibility_without = {"error": str(e)}
        print(f"  ERROR: {str(e)}")

    # ── Comparison ──
    print("\nComparison: WITH vs WITHOUT consistency constraints...")

    comparison = {}

    if isinstance(results_with, list) and isinstance(results_without, list):
        # Per-molecule comparison
        per_molecule_diff = []
        for i in range(len(molecule_names)):
            r_with = results_with[i]
            r_without = results_without[i]

            diff = {"molecule_name": molecule_names[i]}
            for key in ["efficacy_pred", "toxicity_risk_pred", "inflammation_risk_pred",
                        "consistency_score", "immune_cell_activation_pred"]:
                v_with = r_with.get(key)
                v_without = r_without.get(key)
                if v_with is not None and v_without is not None:
                    diff[f"{key}_with"] = v_with
                    diff[f"{key}_without"] = v_without
                    diff[f"{key}_delta"] = v_with - v_without

            per_molecule_diff.append(diff)

        comparison["per_molecule_diff"] = per_molecule_diff

        # Overall comparison
        if isinstance(plausibility_with, dict) and isinstance(plausibility_without, dict):
            comparison["plausibility_improvement"] = {
                "with_rate": plausibility_with.get("plausibility_rate", 0),
                "without_rate": plausibility_without.get("plausibility_rate", 0),
                "improvement": plausibility_with.get("plausibility_rate", 0) - plausibility_without.get("plausibility_rate", 0),
                "with_implausibility_count": plausibility_with.get("implausibility_count", 0),
                "without_implausibility_count": plausibility_without.get("implausibility_count", 0),
                "implausibility_reduction": plausibility_without.get("implausibility_count", 0) - plausibility_with.get("implausibility_count", 0),
            }

            print(f"  Plausibility improvement: {comparison['plausibility_improvement']['improvement']:+.1%}")
            print(f"  Implausibility reduction: {comparison['plausibility_improvement']['implausibility_reduction']} fewer implausible predictions")

    all_results["with_adaptive"] = results_with
    all_results["without_adaptive"] = results_without
    all_results["plausibility_with"] = plausibility_with
    all_results["plausibility_without"] = plausibility_without
    all_results["comparison"] = comparison

    # ── Save ──
    output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_B_cross_module_consistency.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()