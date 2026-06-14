#!/usr/bin/env python3
"""
Compare original heuristic RIG-I scoring vs improved ViennaRNA-based scoring.

Purpose: Evaluate improvement in RIG-I activation prediction accuracy.
"""

import sys
sys.path.append('D:/IGEM集成方案/confluencia_circrna/core')

from rig_i_improved import score_rig_i_improved, score_rig_i_heuristic
from immune_sensing import _score_rig_i, ImmuneSensingConfig

# Test sequences from literature
TEST_CASES = [
    {
        "name": "High dsRNA (GC-rich)",
        "seq": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC" * 2,
        "expected": "high",
        "reference": "Zhang 2016: GC-rich regions form dsRNA stems"
    },
    {
        "name": "Low dsRNA (AU-rich)",
        "seq": "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU" * 2,
        "expected": "low",
        "reference": "AU-rich regions lack stable dsRNA stems"
    },
    {
        "name": "Moderate (mixed)",
        "seq": "GCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAU" * 2,
        "expected": "moderate",
        "reference": "Mixed composition"
    },
    {
        "name": "Inverted repeat",
        "seq": "GCGCGCAUAUAUGCGCGCAUAUAUGCGCGCAUAUAUGCGCGCAUAUAU" * 2,
        "expected": "high",
        "reference": "Zhang 2016: Inverted repeats form dsRNA backbone"
    },
    # Real circRNA examples (simplified)
    {
        "name": "circFOXO3-like",
        "seq": "GCGCAGCAGCAUCGUACGUACGUACGUAGCUAGCUAGCUA" * 3,
        "expected": "moderate",
        "reference": "Wesselhoeft 2018"
    },
    {
        "name": "Short hairpin",
        "seq": "GCGCGCGCGCGCAUAUAUAUAUAUAUAUAGCGCGCGCGCGC",
        "expected": "moderate-high",
        "reference": "Hairpin stem-loop structure"
    },
]

def compare_scores():
    """Compare three scoring methods."""
    print("=" * 70)
    print("RIG-I Scoring Method Comparison")
    print("=" * 70)
    print()

    results = []

    for case in TEST_CASES:
        seq = case["seq"]
        expected = case["expected"]

        # Original heuristic (from immune_sensing.py)
        original_result = _score_rig_i(seq, ImmuneSensingConfig())
        original_score = original_result["rig_i_score"]

        # Improved ViennaRNA-based
        improved_result = score_rig_i_improved(seq, use_viennarna=True)
        improved_score = improved_result["rig_i_score"]

        # Fallback heuristic (from rig_i_improved.py)
        heuristic_result = score_rig_i_heuristic(seq)
        heuristic_score = heuristic_result["rig_i_score"]

        # Check agreement with expected
        def check_agreement(score, exp):
            if exp == "high" and score > 0.55:
                return "[OK]"
            elif exp == "low" and score < 0.35:
                return "[OK]"
            elif exp == "moderate" and 0.35 <= score <= 0.55:
                return "[OK]"
            elif exp == "moderate-high" and score > 0.45:
                return "[OK]"
            else:
                return "[--]"

        orig_agree = check_agreement(original_score, expected)
        improved_agree = check_agreement(improved_score, expected)

        result = {
            "name": case["name"],
            "expected": expected,
            "original": original_score,
            "improved": improved_score,
            "method": improved_result["method"],
            "orig_agree": orig_agree,
            "improved_agree": improved_agree,
            "reference": case["reference"]
        }
        results.append(result)

        print(f"{case['name']} (len={len(seq)}):")
        print(f"  Expected: {expected}")
        print(f"  Original heuristic: {original_score:.3f} {orig_agree}")
        print(f"  Improved ViennaRNA: {improved_score:.3f} {improved_agree}")
        print(f"    Method: {improved_result['method']}")
        if improved_result["method"] == "viennarna_structure":
            print(f"    dsRNA fraction: {improved_result['dsRNA_fraction']:.1%}")
            print(f"    Max stem: {improved_result['max_stem_length']} bp")
            print(f"    MFE: {improved_result['mfe']:.1f} kcal/mol")
        print(f"  Reference: {case['reference']}")
        print()

    # Summary
    orig_correct = sum(1 for r in results if "[OK]" in r["orig_agree"])
    improved_correct = sum(1 for r in results if "[OK]" in r["improved_agree"])

    print("=" * 70)
    print("Summary:")
    print(f"  Original heuristic agreement: {orig_correct}/{len(results)} ({orig_correct/len(results):.1%})")
    print(f"  Improved ViennaRNA agreement: {improved_correct}/{len(results)} ({improved_correct/len(results):.1%})")
    print()

    # Correlation between methods
    import numpy as np
    orig_scores = [r["original"] for r in results]
    improved_scores = [r["improved"] for r in results]
    correlation = np.corrcoef(orig_scores, improved_scores)[0, 1]
    print(f"  Correlation between methods: r = {correlation:.3f}")

    return results


if __name__ == "__main__":
    compare_scores()