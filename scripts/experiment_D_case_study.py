#!/usr/bin/env python3
"""
Confluencia Experiment D: End-to-End Case Study
+ Experiment A: Adaptive vs Fixed Weight Comparison

This script runs 3 circRNA drug candidate scenarios through the full Confluencia
pipeline, demonstrating cross-module reasoning that individual tools cannot produce.

Case Study Scenarios:
  1. Ψ-modified circRNA + low-toxicity drug → expected Go/Conditional
     Insight: Ψ extends half-life but may increase immunogenicity concern
  2. Unmodified circRNA + reactive/toxic drug → expected No-Go
     Insight: Short expression window + high ADMET risk = definitive No-Go
  3. 5mC-modified circRNA + moderate drug → expected Conditional
     Insight: Trade-off analysis across all 5 dimensions

Plus: Adaptive vs Fixed weight comparison for each scenario.

Usage: python experiment_D_case_study.py
Output: benchmarks/results/experiment_D_case_study.json
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add project root to sys.path — this allows proper package imports
# (from confluencia_shared.xxx, from confluencia_joint.xxx, etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_joint.joint_input import JointInput
from confluencia_joint.joint_evaluator import JointEvaluationEngine
from confluencia_joint.scoring import JointScoringEngine


# ── Case Study Scenarios ──────────────────────────────────────────────

CASE_STUDIES = [
    {
        "name": "Case 1: Ψ-modified circRNA + Aspirin (low-toxicity drug)",
        "description": "Ψ modification extends circRNA half-life (~15h). Aspirin has low ADMET risk. Expected: favorable PK but moderate immunogenicity → Go or Conditional",
        "modification": "Ψ",
        "input": JointInput(
            smiles="CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
            epitope_seq="SLYNTVATL",                    # HIV Gag, known HLA-A*02:01 binder
            mhc_allele="HLA-A*02:01",
            dose_mg=200.0,
            freq_per_day=2.0,
            treatment_time=72.0,
            circ_expr=0.7,                               # Moderate-high circRNA expression
            ifn_score=0.3,                                # Moderate IFN response
            trop2=0.8,                                    # High TROP2 (TNBC marker)
            nectin4=0.7,                                  # High NECTIN4
            liv1=0.5,
            b7h4=0.4,
            tmem65=0.5,
            grade=3,                                     # High-grade TNBC
            er_positive=False,
            her2_positive=False,
            pr_positive=False,
        ),
        "expected_insight": "Ψ extends half-life → favorable PK kinetics; low ADMET risk; moderate binding; combined → likely Go or Conditional with reasoning about Ψ immunogenicity vs expression duration trade-off",
    },
    {
        "name": "Case 2: Unmodified circRNA + Rhodanine (reactive/toxic scaffold)",
        "description": "Unmodified circRNA has short half-life (6.24h). Rhodanine is a known PAINS scaffold with high toxicity risk. Expected: short expression + high ADMET risk → No-Go",
        "modification": "unmodified",
        "input": JointInput(
            smiles="O=C1NC(=S)CS1",                     # Rhodanine (PAINS scaffold)
            epitope_seq="NLVPMVATV",                     # CMV pp65, moderate HLA-A*02:01 binder
            mhc_allele="HLA-A*02:01",
            dose_mg=100.0,
            freq_per_day=3.0,
            treatment_time=48.0,
            circ_expr=0.3,                               # Low expression (short half-life)
            ifn_score=0.8,                                # High IFN (unmodified circRNA triggers immunity)
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
        "expected_insight": "Unmodified circRNA → short half-life (6.24h) + high innate immune activation (RIG-I/TLR); PAINS scaffold detected by toxicophore; ADMET flags hepatotoxicity + reactive alerts → definitive No-Go with safety override",
    },
    {
        "name": "Case 3: 5mC-modified circRNA + Ibuprofen (moderate drug)",
        "description": "5mC modification gives intermediate half-life (~8h). Ibuprofen has moderate ADMET profile. Expected: mixed signals across dimensions → Conditional",
        "modification": "5mC",
        "input": JointInput(
            smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",       # Ibuprofen
            epitope_seq="ELAGIGILTV",                     # Melan-A, strong HLA-A*02:01 binder
            mhc_allele="HLA-A*02:01",
            dose_mg=400.0,
            freq_per_day=2.0,
            treatment_time=72.0,
            circ_expr=0.5,                               # Moderate expression
            ifn_score=0.4,                                # Moderate IFN
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
        "expected_insight": "5mC → intermediate half-life; strong MHC binding (ELAGIGILTV) but ibuprofen has moderate ADMET; gene signature shows moderate TNBC profile → Conditional with detailed trade-off reasoning across 5 dimensions",
    },
]


def run_case_study(engine, case, use_adaptive=True):
    """Run a single case study through the full Confluencia pipeline."""
    inp = case["input"]

    # Validate input
    errors = inp.validate()
    if errors:
        return {
            "error": f"Input validation failed: {errors}",
            "case_name": case["name"],
        }

    try:
        start_time = time.time()
        result = engine.evaluate_single(inp)
        elapsed = time.time() - start_time

        # Extract key results
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

            # Sub-scores
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
            "drug_outputs": result.drug_outputs,
            "epitope_outputs": result.epitope_outputs,
            "pk_summary": result.pk_summary,
        }

        # Gene signature (if present)
        if js.gene_signature is not None:
            output["gene_signature"] = {
                "trop2": js.gene_signature.trop2,
                "nectin4": js.gene_signature.nectin4,
                "liv1": js.gene_signature.liv1,
                "b7h4": js.gene_signature.b7h4,
                "tmem65": js.gene_signature.tmem65,
                "risk_score": js.gene_signature.risk_score,
                "efficacy_score": js.gene_signature.efficacy_score,
                "overall": js.gene_signature.overall,
                "interpretation": js.gene_signature.interpretation,
            }

        # circRNA (if present)
        if js.circrna is not None:
            output["circrna"] = {
                "immunotherapy_score": js.circrna.immunotherapy_score,
                "therapeutic_window": js.circrna.therapeutic_window,
                "tumor_killing_index": js.circrna.tumor_killing_index,
                "overall_immunogenicity": js.circrna.overall_immunogenicity,
                "rig_i_score": js.circrna.rig_i_score,
                "tlr_score": js.circrna.tlr_score,
                "pkr_score": js.circrna.pkr_score,
                "overall": js.circrna.overall,
                "interpretation": js.circrna.interpretation,
            }

        # circRNA pipeline outputs (if present)
        if result.circrna_outputs:
            output["circrna_outputs"] = result.circrna_outputs

        # Errors
        if result.errors:
            output["pipeline_errors"] = result.errors

        return output

    except Exception as e:
        return {
            "error": f"Pipeline exception: {str(e)}",
            "traceback": traceback.format_exc(),
            "case_name": case["name"],
        }


def run_adaptive_vs_fixed_comparison(case):
    """Compare adaptive vs fixed weighting for a single case study."""

    results = {}

    # 1. ADAPTIVE weights (default behavior)
    adaptive_engine = JointEvaluationEngine()
    results["adaptive"] = run_case_study(adaptive_engine, case, use_adaptive=True)

    # 2. FIXED weights (disable adaptive mechanism)
    # To use fixed weights, we need to manually compute scores with base weights
    # JointScoringEngine._adaptive_weights is called inside score()
    # We can override by creating a scoring engine with custom logic

    # Approach: compute manually with fixed weights
    fixed_engine = JointEvaluationEngine()

    # First run the pipeline to get module outputs
    try:
        inp = case["input"]
        start_time = time.time()
        result = fixed_engine.evaluate_single(inp)
        elapsed = time.time() - start_time

        # Now manually recompute with fixed weights (no adaptive)
        js_adaptive = result.joint_score

        # Get base weights from config
        base_weights = {
            "clinical": 0.30,
            "binding": 0.20,
            "kinetics": 0.15,
            "gene_signature": 0.15,
            "circrna": 0.20,
        }

        # Compute composite with fixed weights
        sub_scores = {
            "clinical": js_adaptive.clinical.overall,
            "binding": js_adaptive.binding.overall,
            "kinetics": js_adaptive.kinetics.overall,
        }
        if js_adaptive.gene_signature is not None:
            sub_scores["gene_signature"] = js_adaptive.gene_signature.overall
        else:
            sub_scores["gene_signature"] = 0.0
        if js_adaptive.circrna is not None:
            sub_scores["circrna"] = js_adaptive.circrna.overall
        else:
            sub_scores["circrna"] = 0.0

        fixed_composite = sum(base_weights[k] * sub_scores.get(k, 0.0) for k in base_weights)

        # Determine recommendation with fixed weights
        safety_override = js_adaptive.clinical.safety_penalty > 0.30
        if safety_override:
            fixed_recommendation = "No-Go"
            fixed_reason = "Safety override: clinical safety penalty exceeds threshold"
        elif fixed_composite >= 0.65:
            fixed_recommendation = "Go"
            fixed_reason = f"Composite {fixed_composite:.3f} >= 0.65 threshold"
        elif fixed_composite >= 0.40:
            fixed_recommendation = "Conditional"
            fixed_reason = f"Composite {fixed_composite:.3f} >= 0.40 threshold"
        else:
            fixed_recommendation = "No-Go"
            fixed_reason = f"Composite {fixed_composite:.3f} < 0.40 threshold"

        results["fixed"] = {
            "case_name": case["name"],
            "modification": case["modification"],
            "composite_score": fixed_composite,
            "recommendation": fixed_recommendation,
            "recommendation_reason": fixed_reason,
            "effective_weights": base_weights,
            "uncertainties": js_adaptive.uncertainties,
            "use_adaptive_weights": False,
            "sub_scores": sub_scores,
        }

        # Comparison
        adaptive_composite = js_adaptive.composite
        composite_delta = adaptive_composite - fixed_composite

        # Did recommendation change?
        rec_changed = js_adaptive.recommendation != fixed_recommendation

        results["comparison"] = {
            "adaptive_composite": adaptive_composite,
            "fixed_composite": fixed_composite,
            "composite_delta": composite_delta,
            "adaptive_recommendation": js_adaptive.recommendation,
            "fixed_recommendation": fixed_recommendation,
            "recommendation_changed": rec_changed,
            "adaptive_weights": js_adaptive.effective_weights,
            "fixed_weights": base_weights,
            "weight_shifts": {
                k: round(js_adaptive.effective_weights.get(k, 0) - base_weights[k], 4)
                for k in base_weights
            },
        }

    except Exception as e:
        results["fixed"] = {
            "error": f"Fixed weight computation failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
        results["comparison"] = {"error": "Could not compute comparison"}

    return results


def generate_cross_module_insight(case_result):
    """Generate human-readable cross-module reasoning chain from results."""

    if "error" in case_result:
        return f"Pipeline failed: {case_result['error']}"

    insight_parts = []

    # PK reasoning
    kin = case_result.get("kinetics", {})
    hl = kin.get("half_life", "N/A")
    ti = kin.get("therapeutic_index", "N/A")
    kin_overall = kin.get("overall", "N/A")
    modification = case_result.get("modification", "unknown")

    if hl and hl != "N/A":
        if modification == "Ψ" and hl > 10:
            insight_parts.append(f"PK: Ψ-modification yields extended half-life ({hl:.1f}h) → favorable sustained expression")
        elif modification == "unmodified" and hl < 8:
            insight_parts.append(f"PK: Unmodified circRNA has short half-life ({hl:.1f}h) → limited expression window")
        elif modification == "5mC" and 7 <= hl <= 9:
            insight_parts.append(f"PK: 5mC-modification gives intermediate half-life ({hl:.1f}h) → moderate expression duration")

    # ADMET reasoning
    drug = case_result.get("drug_outputs", {})
    tox = drug.get("genotoxicity_risk_pred", drug.get("toxicity_risk_pred", "N/A"))
    inf = drug.get("inflammation_risk_pred", "N/A")
    safety = case_result.get("clinical", {}).get("safety_penalty", "N/A")

    if safety and safety != "N/A":
        if safety > 0.30:
            insight_parts.append(f"ADMET: Safety penalty {safety:.3f} exceeds 0.30 threshold → triggers safety override (No-Go)")
        elif safety > 0.15:
            insight_parts.append(f"ADMET: Moderate safety concern (penalty {safety:.3f}) → reduces confidence in clinical score")
        else:
            insight_parts.append(f"ADMET: Low safety penalty ({safety:.3f}) → favorable toxicity profile")

    # Binding reasoning
    bind = case_result.get("binding", {})
    bind_class = bind.get("mhc_affinity_class", "N/A")
    bind_unc = bind.get("uncertainty", "N/A")

    if bind_class and bind_class != "N/A":
        if bind_class == "strong_binder":
            insight_parts.append(f"Binding: Strong MHC binder → potential immunogenicity concern but also indicates target engagement")
        elif bind_class == "non_binder":
            insight_parts.append(f"Binding: Non-binder → may lack target engagement")
        else:
            insight_parts.append(f"Binding: {bind_class} → moderate binding affinity")

    # Gene signature reasoning
    gs = case_result.get("gene_signature", {})
    gs_risk = gs.get("risk_score", "N/A")
    gs_eff = gs.get("efficacy_score", "N/A")

    if gs_risk and gs_risk != "N/A":
        if gs_risk > 0.6:
            insight_parts.append(f"Gene: High TNBC risk score ({gs_risk:.2f}) → aggressive tumor profile, may need stronger therapy")
        elif gs_risk < 0.3:
            insight_parts.append(f"Gene: Low risk score ({gs_risk:.2f}) → less aggressive tumor, better prognosis")

    # circRNA reasoning
    cr = case_result.get("circrna", {})
    cr_immuno = cr.get("overall_immunogenicity", "N/A")
    rig_i = cr.get("rig_i_score", "N/A")

    if rig_i and rig_i != "N/A":
        if rig_i > 0.5 and modification == "unmodified":
            insight_parts.append(f"circRNA: RIG-I score {rig_i:.2f} → unmodified circRNA triggers innate immunity (reduces therapeutic window)")
        elif rig_i < 0.3 and modification == "Ψ":
            insight_parts.append(f"circRNA: Low RIG-I ({rig_i:.2f}) → Ψ modification reduces immune sensing (favorable)")

    # Cross-module synthesis
    composite = case_result.get("composite_score", "N/A")
    rec = case_result.get("recommendation", "N/A")
    reason = case_result.get("recommendation_reason", "")

    insight_parts.append(f"SYNTHESIS: Composite {composite:.3f} → {rec} ({reason})")

    return " | ".join(insight_parts)


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

    # ── Phase 1: Run all case studies with adaptive weights ──
    print("Phase 1: Running case studies with ADAPTIVE weights...")
    print("-" * 50)

    engine = JointEvaluationEngine()
    case_results = []

    for i, case in enumerate(CASE_STUDIES):
        print(f"\n  Case {i+1}: {case['name']}")
        print(f"  Modification: {case['modification']}")
        print(f"  SMILES: {case['input'].smiles}")
        print(f"  Epitope: {case['input'].epitope_seq}")

        result = run_case_study(engine, case, use_adaptive=True)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Composite: {result['composite_score']:.3f}")
            print(f"  Recommendation: {result['recommendation']}")
            print(f"  Reason: {result['recommendation_reason']}")
            print(f"  Effective weights: {result['effective_weights']}")
            print(f"  Uncertainties: {result['uncertainties']}")

            # Generate cross-module insight
            insight = generate_cross_module_insight(result)
            print(f"  Cross-module insight: {insight}")
            result["cross_module_insight"] = insight

        case_results.append(result)

    all_results["case_studies_adaptive"] = case_results

    # ── Phase 2: Adaptive vs Fixed weight comparison ──
    print("\n\nPhase 2: Adaptive vs Fixed weight comparison...")
    print("-" * 50)

    comparison_results = []

    for i, case in enumerate(CASE_STUDIES):
        print(f"\n  Case {i+1}: {case['name']}")

        comp = run_adaptive_vs_fixed_comparison(case)
        comparison_results.append(comp)

        if "comparison" in comp and "error" not in comp["comparison"]:
            c = comp["comparison"]
            print(f"  Adaptive composite: {c['adaptive_composite']:.3f}")
            print(f"  Fixed composite: {c['fixed_composite']:.3f}")
            print(f"  Delta: {c['composite_delta']:+.3f}")
            print(f"  Adaptive rec: {c['adaptive_recommendation']}")
            print(f"  Fixed rec: {c['fixed_recommendation']}")
            print(f"  Recommendation changed: {c['recommendation_changed']}")
            print(f"  Weight shifts: {c['weight_shifts']}")
        else:
            print(f"  Comparison failed: {comp.get('comparison', {}).get('error', 'unknown')}")

    all_results["adaptive_vs_fixed_comparison"] = comparison_results

    # ── Phase 3: Summary ──
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for i, result in enumerate(case_results):
        if "error" not in result:
            print(f"\n  Case {i+1}: {result['recommendation']}")
            print(f"    Composite: {result['composite_score']:.3f}")
            print(f"    Cross-module: {result.get('cross_module_insight', 'N/A')}")

    print("\n  Adaptive vs Fixed:")
    for i, comp in enumerate(comparison_results):
        if "comparison" in comp and "error" not in comp["comparison"]:
            c = comp["comparison"]
            print(f"    Case {i+1}: delta={c['composite_delta']:+.3f}, rec_changed={c['recommendation_changed']}")

    # ── Save results ──
    output_dir = os.path.join(PROJECT_ROOT, "benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "experiment_D_case_study.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()