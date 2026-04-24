"""
RNACTM Pharmacokinetic Validation
==================================
Validates the six-compartment circRNA PK model against literature-reported values.

Literature sources:
- Wesselhoeft et al. (2018) Nat Commun 9:2629 - circRNA half-life
- Hassett et al. (2019) Mol Ther 27:1885-1897 - LNP depot kinetics
- Paunovska et al. (2018) ACS Nano 12:8307-8320 - Tissue distribution
- Gilleron et al. (2013) Nat Biotechnol 31:638-646 - Endosomal escape
- Chen et al. (2019) Nature 586:651-655 - m6A modification effects
- Liu et al. (2023) Nat Commun 14:2548 - Modified circRNA therapeutics
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "confluencia-2.0-drug"))
from core.ctm import (
    RNACTMParams,
    infer_rna_ctm_params,
    simulate_rna_ctm,
    summarize_rna_ctm_curve,
)


# ---------------------------------------------------------------------------
# Literature Reference Values
# ---------------------------------------------------------------------------

LITERATURE_VALUES = {
    "rna_half_life_unmodified": {
        "value": 6.0,
        "unit": "hours",
        "source": "Wesselhoeft et al. (2018)",
        "doi": "10.1038/s41467-018-05096-x",
        "notes": "Figure 4D, HeLa cells, RT-qPCR detection",
    },
    "rna_half_life_m6a": {
        "value": 10.8,
        "unit": "hours",
        "source": "Chen et al. (2019)",
        "doi": "10.1038/s41586-019-1016-7",
        "notes": "m6A increases stability ~1.8x",
    },
    "rna_half_life_psi": {
        "value": 15.0,
        "unit": "hours",
        "source": "Liu et al. (2023)",
        "doi": "10.1038/s41467-023-38203-5",
        "notes": "Psi modification enhances stability ~2.5x",
    },
    "endosomal_escape_frac": {
        "value": 0.02,
        "unit": "fraction",
        "source": "Gilleron et al. (2013)",
        "doi": "10.1038/nbt.2612",
        "notes": "1-5% of LNP cargo reaches cytoplasm",
    },
    "tissue_liver_frac": {
        "value": 0.80,
        "unit": "fraction",
        "source": "Paunovska et al. (2018)",
        "doi": "10.1021/acsnano.8b03575",
        "notes": "ApoE-mediated hepatocyte uptake",
    },
    "tissue_spleen_frac": {
        "value": 0.10,
        "unit": "fraction",
        "source": "Paunovska et al. (2018)",
        "doi": "10.1021/acsnano.8b03575",
        "notes": "Macrophage uptake",
    },
    "protein_expression_window": {
        "value": 48.0,
        "unit": "hours",
        "source": "Wesselhoeft et al. (2018)",
        "doi": "10.1038/s41467-018-05096-x",
        "notes": "Sustained expression 48-72h",
    },
}


def run_pk_validation() -> Dict[str, Any]:
    """Run comprehensive PK validation against literature values."""
    validation_items = []

    # Standard simulation conditions - SINGLE DOSE for clean half-life estimation
    dose = 1.0
    freq = 0.01  # Effectively single dose (once every 100 days)
    horizon = 168  # 7 days

    print("\n" + "="*60)
    print("RNACTM Pharmacokinetic Validation")
    print("="*60)

    # =========================================================================
    # Test 1: Unmodified circRNA half-life
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 1: Unmodified circRNA half-life")

    params_unmodified = infer_rna_ctm_params(
        modification="none",
        delivery_vector="LNP_standard",
        route="IV",
    )

    # Calculate half-life from degradation rate: t_half = ln(2) / k_degrade
    k_degrade = params_unmodified.k_degrade
    half_life_sim = math.log(2.0) / k_degrade

    lit_val = LITERATURE_VALUES["rna_half_life_unmodified"]["value"]
    error_pct = abs(half_life_sim - lit_val) / lit_val * 100
    passed = error_pct < 50  # Allow 50% error for biological variability

    validation_items.append({
        "parameter": "RNA half-life (unmodified)",
        "literature_value": lit_val,
        "literature_unit": "hours",
        "literature_source": LITERATURE_VALUES["rna_half_life_unmodified"]["source"],
        "simulated_value": round(half_life_sim, 2),
        "error_pct": round(error_pct, 1),
        "passed": bool(passed),  # Convert numpy.bool_ to Python bool
        "notes": f"k_degrade={k_degrade:.4f}/h, t1/2=ln(2)/k ≈ {half_life_sim:.1f}h",
    })
    print(f"  Literature: {lit_val} h ({LITERATURE_VALUES['rna_half_life_unmodified']['source']})")
    print(f"  Simulated:  {half_life_sim:.2f} h (from k_degrade={k_degrade:.4f}/h)")
    print(f"  Error:      {error_pct:.1f}%")
    print(f"  Status:     {'PASS' if passed else 'FAIL'}")

    # =========================================================================
    # Test 2: m6A-modified circRNA half-life
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 2: m6A-modified circRNA half-life")

    params_m6a = infer_rna_ctm_params(
        modification="m6a",
        delivery_vector="LNP_standard",
        route="IV",
    )

    k_degrade_m6a = params_m6a.k_degrade
    half_life_m6a = math.log(2.0) / k_degrade_m6a

    lit_val_m6a = LITERATURE_VALUES["rna_half_life_m6a"]["value"]
    error_pct_m6a = abs(half_life_m6a - lit_val_m6a) / lit_val_m6a * 100
    passed_m6a = error_pct_m6a < 50

    validation_items.append({
        "parameter": "RNA half-life (m6A-modified)",
        "literature_value": lit_val_m6a,
        "literature_unit": "hours",
        "literature_source": LITERATURE_VALUES["rna_half_life_m6a"]["source"],
        "simulated_value": round(half_life_m6a, 2),
        "error_pct": round(error_pct_m6a, 1),
        "passed": bool(passed_m6a),
        "notes": f"k_degrade={k_degrade_m6a:.4f}/h (1.8x stability boost)",
    })
    print(f"  Literature: {lit_val_m6a} h")
    print(f"  Simulated:  {half_life_m6a:.2f} h")
    print(f"  Error:      {error_pct_m6a:.1f}%")
    print(f"  Status:     {'PASS' if passed_m6a else 'FAIL'}")

    # =========================================================================
    # Test 3: Psi-modified circRNA half-life
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 3: Psi-modified circRNA half-life")

    params_psi = infer_rna_ctm_params(
        modification="Ψ",
        delivery_vector="LNP_standard",
        route="IV",
    )

    k_degrade_psi = params_psi.k_degrade
    half_life_psi = math.log(2.0) / k_degrade_psi

    lit_val_psi = LITERATURE_VALUES["rna_half_life_psi"]["value"]
    error_pct_psi = abs(half_life_psi - lit_val_psi) / lit_val_psi * 100
    passed_psi = error_pct_psi < 50

    validation_items.append({
        "parameter": "RNA half-life (Psi-modified)",
        "literature_value": lit_val_psi,
        "literature_unit": "hours",
        "literature_source": LITERATURE_VALUES["rna_half_life_psi"]["source"],
        "simulated_value": round(half_life_psi, 2),
        "error_pct": round(error_pct_psi, 1),
        "passed": bool(passed_psi),
        "notes": f"k_degrade={k_degrade_psi:.4f}/h (2.5x stability boost)",
    })
    print(f"  Literature: {lit_val_psi} h")
    print(f"  Simulated:  {half_life_psi:.2f} h")
    print(f"  Error:      {error_pct_psi:.1f}%")
    print(f"  Status:     {'PASS' if passed_psi else 'FAIL'}")

    # =========================================================================
    # Test 4: Endosomal escape efficiency
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 4: Endosomal escape efficiency")

    # The k_escape parameter represents the escape rate constant
    # To estimate cumulative escape fraction, run single-dose simulation
    curve = simulate_rna_ctm(dose, freq, params_unmodified, horizon=72)

    # Peak cytoplasmic RNA relative to injected dose
    peak_cyto = curve["rna_cytoplasmic"].max()
    escape_frac_sim = peak_cyto / dose

    lit_val_escape = LITERATURE_VALUES["endosomal_escape_frac"]["value"]
    # Allow wide error margin (2x) due to high variability in literature (1-5%)
    error_pct_escape = abs(escape_frac_sim - lit_val_escape) / lit_val_escape * 100
    passed_escape = error_pct_escape < 200  # 1-5% range is 3x variability

    validation_items.append({
        "parameter": "Endosomal escape fraction",
        "literature_value": lit_val_escape,
        "literature_unit": "fraction",
        "literature_source": LITERATURE_VALUES["endosomal_escape_frac"]["source"],
        "simulated_value": round(escape_frac_sim, 4),
        "error_pct": round(error_pct_escape, 1),
        "passed": bool(passed_escape),
        "notes": f"k_escape={params_unmodified.k_escape:.4f}/h, peak_cyto/dose={escape_frac_sim:.2%}",
    })
    print(f"  Literature: {lit_val_escape:.2%} (range 1-5%)")
    print(f"  Simulated:  {escape_frac_sim:.2%} (peak cytoplasmic RNA / dose)")
    print(f"  Error:      {error_pct_escape:.1f}%")
    print(f"  Status:     {'PASS' if passed_escape else 'FAIL'}")

    # =========================================================================
    # Test 5: Tissue distribution (liver)
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 5: Tissue distribution (liver)")

    lit_val_liver = LITERATURE_VALUES["tissue_liver_frac"]["value"]
    sim_val_liver = params_unmodified.f_liver
    error_pct_liver = abs(sim_val_liver - lit_val_liver) / lit_val_liver * 100
    passed_liver = error_pct_liver < 10

    validation_items.append({
        "parameter": "Tissue distribution (liver)",
        "literature_value": lit_val_liver,
        "literature_unit": "fraction",
        "literature_source": LITERATURE_VALUES["tissue_liver_frac"]["source"],
        "simulated_value": sim_val_liver,
        "error_pct": round(error_pct_liver, 1),
        "passed": bool(passed_liver),
        "notes": "Direct parameter match from Paunovska 2018",
    })
    print(f"  Literature: {lit_val_liver:.0%}")
    print(f"  Simulated:  {sim_val_liver:.0%}")
    print(f"  Error:      {error_pct_liver:.1f}%")
    print(f"  Status:     {'PASS' if passed_liver else 'FAIL'}")

    # =========================================================================
    # Test 6: Tissue distribution (spleen)
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 6: Tissue distribution (spleen)")

    lit_val_spleen = LITERATURE_VALUES["tissue_spleen_frac"]["value"]
    sim_val_spleen = params_unmodified.f_spleen
    error_pct_spleen = abs(sim_val_spleen - lit_val_spleen) / lit_val_spleen * 100
    passed_spleen = error_pct_spleen < 10

    validation_items.append({
        "parameter": "Tissue distribution (spleen)",
        "literature_value": lit_val_spleen,
        "literature_unit": "fraction",
        "literature_source": LITERATURE_VALUES["tissue_spleen_frac"]["source"],
        "simulated_value": sim_val_spleen,
        "error_pct": round(error_pct_spleen, 1),
        "passed": bool(passed_spleen),
        "notes": "Direct parameter match from Paunovska 2018",
    })
    print(f"  Literature: {lit_val_spleen:.0%}")
    print(f"  Simulated:  {sim_val_spleen:.0%}")
    print(f"  Error:      {error_pct_spleen:.1f}%")
    print(f"  Status:     {'PASS' if passed_spleen else 'FAIL'}")

    # =========================================================================
    # Test 7: Protein expression window
    # =========================================================================
    print("\n" + "-"*60)
    print("Test 7: Protein expression window")

    # Run with daily dosing to measure expression window
    curve_daily = simulate_rna_ctm(dose, freq=1.0, params=params_unmodified, horizon=168)
    summary = summarize_rna_ctm_curve(curve_daily)
    sim_val_window = summary.get("rna_ctm_protein_expression_window_h", 0.0)

    lit_val_window = LITERATURE_VALUES["protein_expression_window"]["value"]
    error_pct_window = abs(sim_val_window - lit_val_window) / lit_val_window * 100
    passed_window = error_pct_window < 60  # Allow 60% error (48-72h range)

    validation_items.append({
        "parameter": "Protein expression window",
        "literature_value": lit_val_window,
        "literature_unit": "hours",
        "literature_source": LITERATURE_VALUES["protein_expression_window"]["source"],
        "simulated_value": round(sim_val_window, 1),
        "error_pct": round(error_pct_window, 1),
        "passed": bool(passed_window),
        "notes": "Time above 50% peak protein with daily dosing",
    })
    print(f"  Literature: {lit_val_window} h (range 48-72h)")
    print(f"  Simulated:  {sim_val_window:.1f} h")
    print(f"  Error:      {error_pct_window:.1f}%")
    print(f"  Status:     {'PASS' if passed_window else 'FAIL'}")

    # =========================================================================
    # Summary
    # =========================================================================
    n_passed = sum(1 for item in validation_items if item["passed"])
    n_total = len(validation_items)

    summary_result = {
        "n_passed": n_passed,
        "n_total": n_total,
        "pass_rate": round(n_passed / n_total * 100, 1),
    }

    pass_fail = {
        "all_passed": bool(n_passed == n_total),
        "half_life_validation": bool(all([
            validation_items[0]["passed"],
            validation_items[1]["passed"],
            validation_items[2]["passed"],
        ])),
        "distribution_validation": bool(all([
            validation_items[4]["passed"],
            validation_items[5]["passed"],
        ])),
        "pk_dynamics_validation": bool(validation_items[3]["passed"] and validation_items[6]["passed"]),
    }

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  Passed: {n_passed}/{n_total} ({summary_result['pass_rate']:.1f}%)")
    print(f"  Half-life validation: {'PASS' if pass_fail['half_life_validation'] else 'FAIL'}")
    print(f"  Distribution validation: {'PASS' if pass_fail['distribution_validation'] else 'FAIL'}")
    print(f"  PK dynamics validation: {'PASS' if pass_fail['pk_dynamics_validation'] else 'FAIL'}")

    results = {
        "validation_items": validation_items,
        "summary": summary_result,
        "pass_fail": pass_fail,
        "literature_references": LITERATURE_VALUES,
    }

    return results


def main():
    """Main entry point."""
    results = run_pk_validation()

    # Save results
    output_path = Path(__file__).parent / "results" / "rnactm_pk_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
