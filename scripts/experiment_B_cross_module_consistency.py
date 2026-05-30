#!/usr/bin/env python3
"""
Confluencia Experiment B: Cross-Module Consistency vs No Consistency

Compares drug predictions WITH adaptive adjustment (consistency + adaptive)
vs WITHOUT. Uses JointEvaluationEngine which handles all internal imports.

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

from confluencia_joint.joint_input import JointInput
from confluencia_joint.joint_evaluator import JointEvaluationEngine


# ── Test molecules ────────────────────────────────────────────────────

TEST_MOLECULES = [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin", "dose": 200, "freq": 2, "treatment_time": 72,
     "epitope_seq": "SLYNTVATL", "mhc_allele": "HLA-A*02:01"},
    {"smiles": "O=C1NC(=S)CS1", "name": "Rhodanine (PAINS)", "dose": 100, "freq": 3, "treatment_time": 48,
     "epitope_seq": "NLVPMVATV", "mhc_allele": "HLA-A*02:01"},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen", "dose": 400, "freq": 2, "treatment_time": 72,
     "epitope_seq": "ELAGIGILTV", "mhc_allele": "HLA-A*02:01"},
    {"smiles": "c1ccccc1", "name": "Benzene (simple)", "dose": 50, "freq": 1, "treatment_time": 24,
     "epitope_seq": "SYFPEITHI", "mhc_allele": "HLA-A*02:01"},
    {"smiles": "C1CO1", "name": "Ethylene oxide (reactive)", "dose": 10, "freq": 1, "treatment_time": 24,
     "epitope_seq": "LLFGYPVYV", "mhc_allele": "HLA-A*02:01"},
]


def make_joint_inputs(molecules, adaptive=True):
    """Create JointInput objects for each molecule."""
    inputs = []
    for mol in molecules:
        inp = JointInput(
            smiles=mol["smiles"],
            epitope_seq=mol["epitope_seq"],
            mhc_allele=mol["mhc_allele"],
            dose_mg=mol["dose"],
            freq_per_day=mol["freq"],
            treatment_time=mol["treatment_time"],
            circ_expr=0.5,
            ifn_score=0.3,
            trop2=0.7,
            nectin4=0.6,
            liv1=0.5,
            b7h4=0.5,
            tmem65=0.5,
            grade=2,
        )
        inputs.append(inp)
    return inputs


def extract_key_outputs(result, mol_name, smiles):
    """Extract key prediction columns from JointEvaluationResult."""
    js = result.joint_score
    output = {
        "molecule_name": mol_name,
        "smiles": smiles,
    }

    # Clinical sub-score
    output["clinical"] = {
        "efficacy": js.clinical.efficacy,
        "target_binding": js.clinical.target_binding,
        "immune_activation": js.clinical.immune_activation,
        "safety_penalty": js.clinical.safety_penalty,
        "overall": js.clinical.overall,
    }

    # Kinetics
    output["kinetics"] = {
        "half_life": js.kinetics.half_life,
        "cmax": js.kinetics.cmax,
        "therapeutic_index": js.kinetics.therapeutic_index,
        "overall": js.kinetics.overall,
    }

    # Binding
    output["binding"] = {
        "epitope_efficacy": js.binding.epitope_efficacy,
        "mhc_affinity_class": js.binding.mhc_affinity_class,
        "overall": js.binding.overall,
    }

    # Composite
    output["composite"] = js.composite
    output["recommendation"] = js.recommendation
    output["recommendation_reason"] = js.recommendation_reason

    # Drug outputs
    if result.drug_outputs:
        output["drug_outputs"] = result.drug_outputs

    # Gene signature (if present)
    if js.gene_signature is not None:
        output["gene_signature"] = {"overall": js.gene_signature.overall}

    # circRNA (if present)
    if js.circrna is not None:
        output["circrna"] = {
            "overall": js.circrna.overall,
            "overall_immunogenicity": js.circrna.overall_immunogenicity,
        }

    # Pipeline errors
    if result.errors:
        output["pipeline_errors"] = result.errors

    return output


def compute_physiological_plausibility(results):
    """Evaluate whether predictions are physiologically plausible."""
    implausibility_count = 0
    implausibility_details = []

    for r in results:
        name = r["molecule_name"]

        safety = r.get("clinical", {}).get("safety_penalty")
        if safety is not None:
            if safety < 0.05:
                implausibility_count += 1
                implausibility_details.append(f"{name}: safety_penalty {safety:.3f} implausibly low")
            elif safety > 0.80:
                implausibility_count += 1
                implausibility_details.append(f"{name}: safety_penalty {safety:.3f} implausibly high")

        eff = r.get("clinical", {}).get("efficacy")
        bind = r.get("clinical", {}).get("target_binding")
        if eff is not None and bind is not None:
            if eff > 0.95 and bind < 0.3:
                implausibility_count += 1
                implausibility_details.append(f"{name}: efficacy {eff:.3f} implausibly high given binding {bind:.3f}")

        hl = r.get("kinetics", {}).get("half_life")
        if hl is not None:
            if hl < 0.5 or hl > 100:
                implausibility_count += 1
                implausibility_details.append(f"{name}: half-life {hl:.1f}h implausible")

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
    print("(Using JointEvaluationEngine for all imports)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {
        "experiment": "B_cross_module_consistency",
        "timestamp": datetime.now().isoformat(),
        "n_molecules": len(TEST_MOLECULES),
    }

    # ── Create engine ──
    print("Creating JointEvaluationEngine...")
    try:
        engine_adaptive = JointEvaluationEngine()  # Default: adaptive enabled
        print("  Engine created successfully")
    except Exception as e:
        print(f"  FATAL: Could not create engine: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        all_results["engine_error"] = str(e)
        output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiment_B_cross_module_consistency.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        return

    molecule_names = [m["name"] for m in TEST_MOLECULES]
    inputs_adaptive = make_joint_inputs(TEST_MOLECULES, adaptive=True)

    # ── Run 1: WITH adaptive adjustment ──
    print("\nRun 1: Full evaluation WITH adaptive adjustment...")
    results_adaptive = []
    try:
        for i, inp in enumerate(inputs_adaptive):
            print(f"  Evaluating {molecule_names[i]}...")
            result = engine_adaptive.evaluate_single(inp)
            extracted = extract_key_outputs(result, molecule_names[i], TEST_MOLECULES[i]["smiles"])
            results_adaptive.append(extracted)
            print(f"    Composite: {extracted['composite']:.3f}, Rec: {extracted['recommendation']}")
            print(f"    Safety penalty: {extracted['clinical']['safety_penalty']:.3f}")

        plausibility_adaptive = compute_physiological_plausibility(results_adaptive)
        print(f"  Plausibility rate: {plausibility_adaptive['plausibility_rate']:.1%}")
        print(f"  Implausibility count: {plausibility_adaptive['implausibility_count']}")

    except Exception as e:
        results_adaptive = [{"error": str(e), "traceback": traceback.format_exc()}]
        plausibility_adaptive = {"error": str(e)}
        print(f"  ERROR: {str(e)}")

    # ── Run 2: WITHOUT adaptive — use a modified scoring engine ──
    # The JointScoringEngine's adaptive weights are built into the score() method.
    # To disable them, we need to manually compute with fixed weights.
    # We'll reuse the adaptive results and recompute composites with fixed weights.

    print("\nRun 2: Recomputing with FIXED weights (no adaptive adjustment)...")

    # The adaptive results already contain all sub-scores.
    # We just need to recompute the composite with fixed weights.
    base_weights = {"clinical": 0.30, "binding": 0.20, "kinetics": 0.15, "gene_signature": 0.15, "circrna": 0.20}

    results_fixed = []
    comparison_per_molecule = []

    if isinstance(results_adaptive, list) and "error" not in results_adaptive[0]:
        for i, r_adaptive in enumerate(results_adaptive):
            r_fixed = {
                "molecule_name": r_adaptive["molecule_name"],
                "smiles": r_adaptive["smiles"],
            }

            # Same sub-scores (pipeline outputs don't change — only weight aggregation)
            r_fixed["clinical"] = r_adaptive["clinical"]
            r_fixed["kinetics"] = r_adaptive["kinetics"]
            r_fixed["binding"] = r_adaptive["binding"]
            if "gene_signature" in r_adaptive:
                r_fixed["gene_signature"] = r_adaptive["gene_signature"]
            if "circrna" in r_adaptive:
                r_fixed["circrna"] = r_adaptive["circrna"]

            # Recompute composite with fixed weights
            sub_scores = {
                "clinical": r_adaptive["clinical"]["overall"],
                "binding": r_adaptive["binding"]["overall"],
                "kinetics": r_adaptive["kinetics"]["overall"],
                "gene_signature": r_adaptive.get("gene_signature", {}).get("overall", 0.0),
                "circrna": r_adaptive.get("circrna", {}).get("overall", 0.0),
            }

            fixed_composite = sum(base_weights[k] * sub_scores.get(k, 0.0) for k in base_weights)

            # Determine recommendation with fixed weights
            safety = r_adaptive["clinical"]["safety_penalty"]
            if safety > 0.30:
                fixed_rec = "No-Go"
                fixed_reason = "Safety override"
            elif fixed_composite >= 0.65:
                fixed_rec = "Go"
                fixed_reason = f"Composite {fixed_composite:.3f} >= 0.65"
            elif fixed_composite >= 0.40:
                fixed_rec = "Conditional"
                fixed_reason = f"Composite {fixed_composite:.3f} >= 0.40"
            else:
                fixed_rec = "No-Go"
                fixed_reason = f"Composite {fixed_composite:.3f} < 0.40"

            r_fixed["composite"] = fixed_composite
            r_fixed["recommendation"] = fixed_rec
            r_fixed["recommendation_reason"] = fixed_reason

            results_fixed.append(r_fixed)

            # Per-molecule comparison
            composite_delta = r_adaptive["composite"] - fixed_composite
            rec_changed = r_adaptive["recommendation"] != fixed_rec

            comparison_per_molecule.append({
                "molecule_name": molecule_names[i],
                "adaptive_composite": r_adaptive["composite"],
                "fixed_composite": fixed_composite,
                "composite_delta": composite_delta,
                "adaptive_recommendation": r_adaptive["recommendation"],
                "fixed_recommendation": fixed_rec,
                "recommendation_changed": rec_changed,
                "safety_penalty": r_adaptive["clinical"]["safety_penalty"],
            })

            print(f"  {molecule_names[i]}: adaptive={r_adaptive['composite']:.3f} ({r_adaptive['recommendation']}), "
                  f"fixed={fixed_composite:.3f} ({fixed_rec}), delta={composite_delta:+.3f}")

        plausibility_fixed = compute_physiological_plausibility(results_fixed)

        # Overall comparison
        overall_comparison = {
            "per_molecule": comparison_per_molecule,
            "adaptive_plausibility_rate": plausibility_adaptive.get("plausibility_rate", 0),
            "fixed_plausibility_rate": plausibility_fixed.get("plausibility_rate", 0),
            "n_recommendations_changed": sum(1 for c in comparison_per_molecule if c["recommendation_changed"]),
        }
        print(f"\n  Recommendations changed: {overall_comparison['n_recommendations_changed']}/{len(TEST_MOLECULES)}")

    else:
        results_fixed = [{"error": "Could not compute fixed weights — adaptive results unavailable"}]
        overall_comparison = {"error": "Adaptive results unavailable for comparison"}

    all_results["adaptive"] = results_adaptive
    all_results["fixed"] = results_fixed
    all_results["plausibility_adaptive"] = plausibility_adaptive
    all_results["comparison"] = overall_comparison

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