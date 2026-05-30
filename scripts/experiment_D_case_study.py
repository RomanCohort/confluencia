#!/usr/bin/env python3
"""
Confluencia Experiment D: End-to-End Case Study
+ Experiment A: Adaptive vs Fixed Weight Comparison

This script runs 3 circRNA drug candidate scenarios through the full Confluencia
pipeline, demonstrating cross-module reasoning that individual tools cannot produce.

IMPORTANT: This script uses JointEvaluationEngine which handles all internal
imports via lazy loading. No direct imports from sub-modules are needed.

Usage: python experiment_D_case_study.py
Output: benchmarks/results/experiment_D_case_study.json
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Diagnostic: verify weight_loader import works ──
print(f"[DIAG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DIAG] sys.path[0] = {sys.path[0]}")
try:
    from confluencia_shared.weight_loader import get_sub_weights, get_thresholds
    print("[DIAG] weight_loader import: SUCCESS")
except ImportError as e:
    print(f"[DIAG] weight_loader import: FAILED — {e}")
    # Fallback: try adding confluencia_shared directly
    shared_dir = os.path.join(PROJECT_ROOT, "confluencia_shared")
    if os.path.isdir(shared_dir):
        # Create a namespace package by ensuring __init__.py exists
        init_path = os.path.join(shared_dir, "__init__.py")
        if not os.path.exists(init_path):
            print(f"[DIAG] Creating missing __init__.py in {shared_dir}")
            with open(init_path, "w") as f:
                f.write("")
    print(f"[DIAG] Retrying with explicit path...")
    sys.path.insert(0, PROJECT_ROOT)  # Re-insert at position 0
    try:
        from confluencia_shared.weight_loader import get_sub_weights, get_thresholds
        print("[DIAG] weight_loader import retry: SUCCESS")
    except ImportError as e2:
        print(f"[DIAG] weight_loader import retry: STILL FAILED — {e2}")
        print(f"[DIAG] Files in confluencia_shared: {os.listdir(shared_dir) if os.path.isdir(shared_dir) else 'DIR NOT FOUND'}")

from confluencia_joint.joint_input import JointInput
from confluencia_joint.joint_evaluator import JointEvaluationEngine


# ── Case Study Scenarios ──────────────────────────────────────────────

CASE_STUDIES = [
    {
        "name": "Case 1: Ψ-modified circRNA + Aspirin (low-toxicity drug)",
        "description": "Ψ modification extends circRNA half-life (~15h). Aspirin has low ADMET risk. Expected: favorable PK but moderate immunogenicity → Go or Conditional",
        "modification": "Ψ",
        "input": JointInput(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            epitope_seq="SLYNTVATL",
            mhc_allele="HLA-A*02:01",
            dose_mg=200.0,
            freq_per_day=2.0,
            treatment_time=72.0,
            circ_expr=0.7,
            ifn_score=0.3,
            trop2=0.8,
            nectin4=0.7,
            liv1=0.5,
            b7h4=0.4,
            tmem65=0.5,
            grade=3,
            er_positive=False,
            her2_positive=False,
            pr_positive=False,
        ),
        "expected_insight": "Ψ extends half-life → favorable PK; low ADMET risk; moderate binding → likely Go or Conditional",
    },
    {
        "name": "Case 2: Unmodified circRNA + Rhodanine (reactive/toxic scaffold)",
        "description": "Unmodified circRNA has short half-life (6.24h). Rhodanine is a PAINS scaffold. Expected: short expression + high ADMET risk → No-Go",
        "modification": "unmodified",
        "input": JointInput(
            smiles="O=C1NC(=S)CS1",
            epitope_seq="NLVPMVATV",
            mhc_allele="HLA-A*02:01",
            dose_mg=100.0,
            freq_per_day=3.0,
            treatment_time=48.0,
            circ_expr=0.3,
            ifn_score=0.8,
            trop2=0.6,
            nectin4=0.5,
            liv1=0.5,
            b7h4=0.6,
            tmem65=0.4,
            grade=2,
            er_positive=False,
            her2_positive=False,
            pr_positive=False,
        ),
        "expected_insight": "Unmodified → short half-life + high immune activation; PAINS scaffold → No-Go with safety override",
    },
    {
        "name": "Case 3: 5mC-modified circRNA + Ibuprofen (moderate drug)",
        "description": "5mC modification gives intermediate half-life (~8h). Ibuprofen has moderate ADMET profile. Expected: mixed signals → Conditional",
        "modification": "5mC",
        "input": JointInput(
            smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            epitope_seq="ELAGIGILTV",
            mhc_allele="HLA-A*02:01",
            dose_mg=400.0,
            freq_per_day=2.0,
            treatment_time=72.0,
            circ_expr=0.5,
            ifn_score=0.4,
            trop2=0.7,
            nectin4=0.6,
            liv1=0.6,
            b7h4=0.5,
            tmem65=0.5,
            grade=2,
            er_positive=False,
            her2_positive=False,
            pr_positive=False,
        ),
        "expected_insight": "5mC → intermediate half-life; strong MHC binding; moderate ADMET → Conditional with 5D trade-off",
    },
]


def run_case_study(engine, case, use_adaptive=True):
    """Run a single case study through the full Confluencia pipeline."""
    inp = case["input"]

    errors = inp.validate()
    if errors:
        return {"error": f"Input validation failed: {errors}", "case_name": case["name"]}

    try:
        start_time = time.time()
        result = engine.evaluate_single(inp)
        elapsed = time.time() - start_time

        js = result.joint_score

        output = {
            "case_name": case["name"],
            "modification": case["modification"],
            "description": case["description"],
            "expected_insight": case["expected_insight"],
            "composite_score": js.composite,
            "recommendation": js.recommendation,
            "recommendation_reason": js.recommendation_reason,
            "effective_weights": js.effective_weights,
            "uncertainties": js.uncertainties,
            "use_adaptive_weights": use_adaptive,
            "evaluation_time_s": elapsed,
            "clinical": {
                "efficacy": js.clinical.efficacy,
                "binding": js.clinical.target_binding,
                "immune_activation": js.clinical.immune_activation,
                "safety_penalty": js.clinical.safety_penalty,
                "overall": js.clinical.overall,
                "interpretation": js.clinical.interpretation,
            },
            "binding": {
                "epitope_efficacy": js.binding.epitope_efficacy,
                "uncertainty": js.binding.uncertainty,
                "mhc_affinity_class": js.binding.mhc_affinity_class,
                "overall": js.binding.overall,
                "interpretation": js.binding.interpretation,
            },
            "kinetics": {
                "cmax": js.kinetics.cmax,
                "tmax": js.kinetics.tmax,
                "half_life": js.kinetics.half_life,
                "auc_conc": js.kinetics.auc_conc,
                "auc_effect": js.kinetics.auc_effect,
                "therapeutic_index": js.kinetics.therapeutic_index,
                "overall": js.kinetics.overall,
                "interpretation": js.kinetics.interpretation,
            },
            "drug_outputs": result.drug_outputs_dict if hasattr(result, 'drug_output_dict') else result.drug_outputs,
            "epitope_outputs": result.epitope_outputs,
            "pk_summary": result.pk_summary,
        }

        if js.gene_signature is not None:
            output["gene_signature"] = {
                "trop2": js.gene_signature.trop2,
                "nectin4": js.gene_signature.nectin4,
                "overall": js.gene_signature.overall,
                "interpretation": js.gene_signature.interpretation,
            }

        if js.circrna is not None:
            output["circrna"] = {
                "immunotherapy_score": js.circrna.immunotherapy_score,
                "therapeutic_window": js.circrna.therapeutic_window,
                "overall_immunogenicity": js.circrna.overall_immunogenicity,
                "rig_i_score": js.circrna.rig_i_score,
                "overall": js.circrna.overall,
                "interpretation": js.circrna.interpretation,
            }

        if result.circrna_outputs_dict if hasattr(result, 'circrna_output_dict') else result.circrna_outputs:
            output["circrna_outputs"] = result.circrna_output_dict if hasattr(result, 'circrna_output_dict') else result.circrna_outputs

        if result.errors:
            output["pipeline_errors"] = result.errors

        return output

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc(), "case_name": case["name"]}


def run_adaptive_vs_fixed_comparison(case):
    """Compare adaptive vs fixed weighting for a single case study."""
    results = {}

    # 1. ADAPTIVE weights (default behavior)
    try:
        engine = JointEvaluationEngine()
        results["adaptive"] = run_case_study(engine, case, use_adaptive=True)
    except Exception as e:
        results["adaptive"] = {"error": str(e), "traceback": traceback.format_exc()}
        return results

    if "error" in results["adaptive"]:
        results["fixed"] = {"error": "Adaptive failed, cannot compare"}
        results["comparison"] = {"error": "Cannot compare"}
        return results

    # 2. FIXED weights — recompute composite with base weights
    js_adaptive = results["adaptive"]
    base_weights = {"clinical": 0.30, "binding": 0.20, "kinetics": 0.15, "gene_signature": 0.15, "circrna": 0.20}

    sub_scores = {
        "clinical": js_adaptive.get("clinical", {}).get("overall", 0.0),
        "binding": js_adaptive.get("binding", {}).get("overall", 0.0),
        "kinetics": js_adaptive.get("kinetics", {}).get("overall", 0.0),
        "gene_signature": js_adaptive.get("gene_signature", {}).get("overall", 0.0) if js_adaptive.get("gene_signature") else 0.0,
        "circrna": js_adaptive.get("circrna", {}).get("overall", 0.0) if js_adaptive.get("circrna") else 0.0,
    }

    fixed_composite = sum(base_weights[k] * sub_scores.get(k, 0.0) for k in base_weights)

    # Determine recommendation with fixed weights
    safety_penalty = js_adaptive.get("clinical", {}).get("safety_penalty", 0.0)
    if safety_penalty > 0.30:
        fixed_rec = "No-Go"
        fixed_reason = "Safety override: safety penalty exceeds 0.30"
    elif fixed_composite >= 0.65:
        fixed_rec = "Go"
        fixed_reason = f"Composite {fixed_composite:.3f} >= 0.65"
    elif fixed_composite >= 0.40:
        fixed_rec = "Conditional"
        fixed_reason = f"Composite {fixed_composite:.3f} >= 0.40"
    else:
        fixed_rec = "No-Go"
        fixed_reason = f"Composite {fixed_composite:.3f} < 0.40"

    results["fixed"] = {
        "case_name": case["name"],
        "modification": case["modification"],
        "composite_score": fixed_composite,
        "recommendation": fixed_rec,
        "recommendation_reason": fixed_reason,
        "effective_weights": base_weights,
        "uncertainties": js_adaptive.get("uncertainties"),
        "use_adaptive_weights": False,
        "sub_scores": sub_scores,
    }

    # Comparison
    adaptive_composite = js_adaptive.get("composite_score", 0.0)
    composite_delta = adaptive_composite - fixed_composite
    rec_changed = js_adaptive.get("recommendation") != fixed_rec

    results["comparison"] = {
        "adaptive_composite": adaptive_composite,
        "fixed_composite": fixed_composite,
        "composite_delta": composite_delta,
        "adaptive_recommendation": js_adaptive.get("recommendation"),
        "fixed_recommendation": fixed_rec,
        "recommendation_changed": rec_changed,
        "adaptive_weights": js_adaptive.get("effective_weights"),
        "fixed_weights": base_weights,
        "weight_shifts": {
            k: round(js_adaptive.get("effective_weights", {}).get(k, 0) - base_weights[k], 4)
            for k in base_weights
        } if js_adaptive.get("effective_weights") else {},
    }

    return results


def generate_cross_module_insight(case_result):
    """Generate human-readable cross-module reasoning chain."""
    if "error" in case_result:
        return f"Pipeline failed: {case_result['error']}"

    parts = []
    modification = case_result.get("modification", "unknown")
    kin = case_result.get("kinetics", {})
    hl = kin.get("half_life")

    if hl:
        if modification == "Ψ" and hl > 10:
            parts.append(f"PK: Ψ-modification → extended half-life ({hl:.1f}h)")
        elif modification == "unmodified" and hl < 8:
            parts.append(f"PK: Unmodified → short half-life ({hl:.1f}h)")
        elif modification == "5mC" and 7 <= hl <= 9:
            parts.append(f"PK: 5mC → intermediate half-life ({hl:.1f}h)")

    safety = case_result.get("clinical", {}).get("safety_penalty")
    if safety:
        if safety > 0.30:
            parts.append(f"ADMET: Safety penalty {safety:.3f} → safety override (No-Go)")
        elif safety > 0.15:
            parts.append(f"ADMET: Moderate safety concern ({safety:.3f})")
        else:
            parts.append(f"ADMET: Low safety penalty ({safety:.3f})")

    bind_class = case_result.get("binding", {}).get("mhc_affinity_class")
    if bind_class:
        parts.append(f"Binding: {bind_class}")

    cr = case_result.get("circrna", {})
    rig_i = cr.get("rig_i_score")
    if rig_i:
        if rig_i > 0.5 and modification == "unmodified":
            parts.append(f"circRNA: RIG-I {rig_i:.2f} → unmodified triggers immunity")
        elif rig_i < 0.3 and modification == "Ψ":
            parts.append(f"circRNA: Low RIG-I ({rig_i:.2f}) → Ψ reduces immune sensing")

    composite = case_result.get("composite_score", 0)
    rec = case_result.get("recommendation", "N/A")
    reason = case_result.get("recommendation_reason", "")
    parts.append(f"SYNTHESIS: {composite:.3f} → {rec} ({reason})")

    return " | ".join(parts)


def main():
    print("=" * 70)
    print("Confluencia Experiment D: End-to-End Case Study")
    print("Experiment A: Adaptive vs Fixed Weight Comparison")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {
        "experiment": "D_case_study_and_A_weight_comparison",
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(CASE_STUDIES),
    }

    # ── Phase 1: Adaptive weights ──
    print("Phase 1: Running case studies with ADAPTIVE weights...")
    print("-" * 50)

    try:
        engine = JointEvaluationEngine()
    except Exception as e:
        print(f"  FATAL: Could not create JointEvaluationEngine: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        all_results["engine_error"] = str(e)
        output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiment_D_case_study.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        return

    case_results = []
    for i, case in enumerate(CASE_STUDIES):
        print(f"\n  Case {i+1}: {case['name']}")
        print(f"  Modification: {case['modification']}")
        print(f"  SMILES: {case['input'].smiles}")

        result = run_case_study(engine, case, use_adaptive=True)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Composite: {result['composite_score']:.3f}")
            print(f"  Recommendation: {result['recommendation']}")
            print(f"  Reason: {result['recommendation_reason']}")
            print(f"  Effective weights: {result['effective_weights']}")
            print(f"  Uncertainties: {result['uncertainties']}")
            insight = generate_cross_module_insight(result)
            print(f"  Cross-module insight: {insight}")
            result["cross_module_insight"] = insight

        case_results.append(result)

    all_results["case_studies_adaptive"] = case_results

    # ── Phase 2: Adaptive vs Fixed ──
    print("\n\nPhase 2: Adaptive vs Fixed weight comparison...")
    print("-" * 50)

    comparison_results = []
    for i, case in enumerate(CASE_STUDIES):
        print(f"\n  Case {i+1}: {case['name']}")
        comp = run_adaptive_vs_fixed_comparison(case)
        comparison_results.append(comp)

        if "comparison" in comp and "error" not in comp.get("comparison", {}):
            c = comp["comparison"]
            print(f"  Adaptive composite: {c.get('adaptive_composite', 'N/A')}")
            print(f"  Fixed composite: {c.get('fixed_composite', 'N/A')}")
            print(f"  Delta: {c.get('composite_delta', 'N/A')}")
            print(f"  Recommendation changed: {c.get('recommendation_changed', 'N/A')}")
            print(f"  Weight shifts: {c.get('weight_shifts', 'N/A')}")

    all_results["adaptive_vs_fixed_comparison"] = comparison_results

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for i, result in enumerate(case_results):
        if "error" not in result:
            print(f"  Case {i+1}: {result['recommendation']} (composite {result['composite_score']:.3f})")
            print(f"    Cross-module: {result.get('cross_module_insight', 'N/A')}")

    # ── Save ──
    output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_D_case_study.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()