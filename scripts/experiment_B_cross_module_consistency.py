#!/usr/bin/env python3
"""
Confluencia Experiment B: Cross-Module Consistency vs No Consistency

Runs as a standalone script that directly imports from the project.
Uses subprocess isolation to avoid import conflicts with hyphenated dirs.

Usage: python experiment_B_cross_module_consistency.py
Output: benchmarks/results/experiment_B_cross_module_consistency.json
"""

import sys
import os
import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Test molecules ────────────────────────────────────────────────────

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
                val = row[col]
                result[col] = float(val) if pd.notna(val) and val is not None else None
            else:
                result[col] = None

        # CTM/PK parameters
        param_cols = [
            "ctm_ka", "ctm_kd", "ctm_ke", "ctm_km", "ctm_signal_gain",
            "pkpd_half_life_h", "pkpd_cmax_mg_per_l", "pkpd_tmax_h",
            "pkpd_auc_conc", "pkpd_auc_effect",
            "adaptive_confidence", "adaptive_risk_pressure",
            "adaptive_dose_factor", "adaptive_freq_factor",
        ]
        for col in param_cols:
            if col in out_df.columns:
                val = row[col]
                result[col] = float(val) if pd.notna(val) and val is not None else None
            else:
                result[col] = None

        results.append(result)
    return results


def compute_physiological_plausibility(results):
    """Evaluate whether predictions are physiologically plausible."""
    implausibility_count = 0
    implausibility_details = []

    for r in results:
        name = r["molecule_name"]

        tox = r.get("toxicity_risk_pred")
        if tox is not None and tox < 0.05:
            implausibility_count += 1
            implausibility_details.append(f"{name}: toxicity {tox:.3f} implausibly low")

        inf = r.get("inflammation_risk_pred")
        if inf is not None and inf < 0.05:
            implausibility_count += 1
            implausibility_details.append(f"{name}: inflammation {inf:.3f} implausibly low")

        eff = r.get("efficacy_pred")
        bind = r.get("target_binding_pred")
        if eff is not None and bind is not None and eff > 0.95 and bind < 0.3:
            implausibility_count += 1
            implausibility_details.append(f"{name}: efficacy too high given binding")

        ka = r.get("ctm_ka")
        if ka is not None and ka > 0.9:
            implausibility_count += 1
            implausibility_details.append(f"{name}: ka {ka:.3f} out of range")

    total = len(results) * 4
    return {
        "implausibility_count": implausibility_count,
        "implausibility_details": implausibility_details,
        "total_checks": total,
        "plausibility_rate": 1.0 - (implausibility_count / total) if total > 0 else 0,
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

    # Import pipeline via importlib (handles hyphenated directory name)
    import importlib.util

    pipeline_path = os.path.join(PROJECT_ROOT, "confluencia-2.0-drug", "core", "pipeline.py")

    try:
        spec = importlib.util.spec_from_file_location("drug_pipeline", pipeline_path)
        pipeline_mod = importlib.util.module_from_spec(spec)
        # Need to set up sys.path before executing the module
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "confluencia-2.0-drug", "core"))
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "confluencia_shared"))
        spec.loader.exec_module(pipeline_mod)
        run_pipeline = pipeline_mod.run_pipeline
        print(f"  Pipeline loaded from: {pipeline_path}")
    except Exception as e:
        print(f"  FATAL: Could not load drug pipeline: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        all_results["error"] = str(e)
        # Save and exit
        output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiment_B_cross_module_consistency.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        return

    # ── Run 1: WITH adaptive ──
    print("\nRun 1: Pipeline WITH adaptive adjustment...")
    try:
        out_with, curves_with, artifacts_with = run_pipeline(
            df, compute_mode="auto", model_backend="moe",
            adaptive_enabled=True, adaptive_strength=0.2,
        )
        if out_with is None:
            raise ValueError("Pipeline returned None for output DataFrame")
        results_with = extract_key_outputs(out_with, molecule_names)
        plausibility_with = compute_physiological_plausibility(results_with)
        print(f"  Completed. Plausibility: {plausibility_with['plausibility_rate']:.1%}")
        for r in results_with:
            print(f"    {r['molecule_name']}: eff={r.get('efficacy_pred', 'N/A'):.3f}, "
                  f"tox={r.get('toxicity_risk_pred', 'N/A')}, "
                  f"inf={r.get('inflammation_risk_pred', 'N/A')}")
    except Exception as e:
        results_with = {"error": str(e), "traceback": traceback.format_exc()}
        plausibility_with = {"error": str(e)}
        print(f"  ERROR: {str(e)}")

    # ── Run 2: WITHOUT adaptive ──
    print("\nRun 2: Pipeline WITHOUT adaptive adjustment (baseline)...")
    try:
        out_without, curves_without, artifacts_without = run_pipeline(
            df, compute_mode="auto", model_backend="moe",
            adaptive_enabled=False,
        )
        if out_without is None:
            raise ValueError("Pipeline returned None for output DataFrame")
        results_without = extract_key_outputs(out_without, molecule_names)
        plausibility_without = compute_physiological_plausibility(results_without)
        print(f"  Completed. Plausibility: {plausibility_without['plausibility_rate']:.1%}")
        for r in results_without:
            print(f"    {r['molecule_name']}: eff={r.get('efficacy_pred', 'N/A'):.3f}, "
                  f"tox={r.get('toxicity_risk_pred', 'N/A')}, "
                  f"inf={r.get('inflammation_risk_pred', 'N/A')}")
    except Exception as e:
        results_without = {"error": str(e), "traceback": traceback.format_exc()}
        plausibility_without = {"error": str(e)}
        print(f"  ERROR: {str(e)}")

    # ── Comparison ──
    comparison = {}
    if isinstance(results_with, list) and isinstance(results_without, list):
        per_molecule_diff = []
        for i in range(len(molecule_names)):
            r_with = results_with[i]
            r_without = results_without[i]
            diff = {"molecule_name": molecule_names[i]}
            for key in ["efficacy_pred", "toxicity_risk_pred", "inflammation_risk_pred",
                        "consistency_score", "immune_cell_activation_pred"]:
                v_w = r_with.get(key)
                v_wo = r_without.get(key)
                if v_w is not None and v_wo is not None:
                    diff[f"{key}_with"] = v_w
                    diff[f"{key}_without"] = v_wo
                    diff[f"{key}_delta"] = v_w - v_wo
            per_molecule_diff.append(diff)
        comparison["per_molecule_diff"] = per_molecule_diff

        if isinstance(plausibility_with, dict) and isinstance(plausibility_without, dict) \
                and "plausibility_rate" in plausibility_with and "plausibility_rate" in plausibility_without:
            comparison["plausibility_improvement"] = {
                "with_rate": plausibility_with["plausibility_rate"],
                "without_rate": plausibility_without["plausibility_rate"],
                "improvement": plausibility_with["plausibility_rate"] - plausibility_without["plausibility_rate"],
            }

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