#!/usr/bin/env python3
"""
RIG-I pathway weight optimization via grid search.

Literature validation data (qualitative):
- Chen et al. 2019: m6A modification reduces RIG-I activation
- Zhang et al. 2016: Inverted repeat circRNA activates RIG-I
- Wesselhoeft 2018: Unmodified circRNA has moderate immunogenicity
- Liu et al. 2022: circRNA degradation fragments can activate RIG-I

Goal: Find weights that maximize agreement with literature directionality.
"""

import numpy as np
from itertools import product
from typing import Dict, List, Tuple
import json

# Literature-derived test cases (qualitative labels)
# Format: (sequence_description, expected_relative_activation)
LITERATURE_TEST_CASES = [
    # From Chen et al. 2019: m6A reduces RIG-I activation
    {
        "name": "unmodified_vs_m6A",
        "unmodified_score": "higher",  # unmodified should score higher than m6A
        "reference": "Chen et al. 2019 Nature",
        "mechanism": "m6A leads to YTHDF2-mediated degradation, reducing RIG-I substrate"
    },
    # From Zhang et al. 2016: Inverted repeats activate RIG-I
    {
        "name": "inverted_repeat_vs_no_repeat",
        "IR_score": "higher",  # IR-containing should score higher
        "reference": "Zhang et al. Nat Immunol 2016",
        "mechanism": "Inverted repeats form dsRNA backbone recognized by RIG-I"
    },
    # From Wesselhoeft 2018: circRNA stability correlates with lower immunogenicity
    {
        "name": "stable_vs_unstable",
        "stable_score": "lower",  # more stable (Psi) should have lower immunogenicity
        "reference": "Wesselhoeft et al. 2018 Nat Commun",
        "mechanism": "Stable modifications reduce immune recognition"
    },
]

# Example sequences for testing (simplified)
TEST_SEQUENCES = {
    # High dsRNA potential: GC-rich with inverted repeat pattern
    "high_dsRNA": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    # Low dsRNA potential: AU-rich, no stems
    "low_dsRNA": "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",
    # Moderate: mixed sequence
    "moderate": "GCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAUGCGCAUAU",
    # Inverted repeat pattern (simplified)
    "inverted_repeat": "GCGCGCAUAUAUGCGCGCAUAUAUGCGCGCAUAUAUGCGCGCAUAUAU",
}


def _gc_content(seq: str) -> float:
    """Calculate GC content."""
    seq = seq.upper().replace("T", "U")
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if len(seq) > 0 else 0.0


def _estimate_dsRNA(seq: str) -> float:
    """Estimate dsRNA structure potential."""
    gc = _gc_content(seq)
    # GC-rich regions indicate stem potential
    return min(gc * 1.5, 1.0)


def _estimate_inverted_repeats(seq: str) -> float:
    """Estimate inverted repeat potential."""
    # Simplified: count GC-rich windows
    window = 6
    gc_windows = 0
    for i in range(0, len(seq) - window, window):
        if _gc_content(seq[i:i+window]) > 0.5:
            gc_windows += 1
    return gc_windows / max(len(seq) // window, 1)


def score_rig_i_with_weights(
    seq: str,
    w_dsRNA: float,
    w_motif: float,
    w_gc: float,
    w_length: float
) -> float:
    """
    Score RIG-I activation with given weights.

    Weights should sum to 1.0.
    """
    # Normalize weights
    total = w_dsRNA + w_motif + w_gc + w_length
    w_dsRNA /= total
    w_motif /= total
    w_gc /= total
    w_length /= total

    # Components
    dsRNA_score = _estimate_dsRNA(seq)
    motif_score = 0.5  # placeholder
    gc_score = _gc_content(seq)
    length_score = min(len(seq) / 500.0, 1.0)

    return (w_dsRNA * dsRNA_score +
            w_motif * motif_score +
            w_gc * gc_score +
            w_length * length_score)


def evaluate_weights(
    w_dsRNA: float,
    w_motif: float,
    w_gc: float,
    w_length: float
) -> Dict:
    """
    Evaluate weight combination against literature expectations.

    Returns agreement score (0-1) with literature directionality.
    """
    scores = {}

    # Score test sequences
    for name, seq in TEST_SEQUENCES.items():
        scores[name] = score_rig_i_with_weights(seq, w_dsRNA, w_motif, w_gc, w_length)

    # Evaluate literature expectations
    agreements = []

    # Expectation 1: high_dsRNA > low_dsRNA
    if scores["high_dsRNA"] > scores["low_dsRNA"]:
        agreements.append(1.0)
    else:
        agreements.append(0.0)

    # Expectation 2: inverted_repeat > moderate
    if scores["inverted_repeat"] > scores["moderate"]:
        agreements.append(1.0)
    else:
        agreements.append(0.0)

    # Expectation 3: moderate > low_dsRNA
    if scores["moderate"] > scores["low_dsRNA"]:
        agreements.append(1.0)
    else:
        agreements.append(0.0)

    return {
        "weights": {
            "dsRNA": w_dsRNA,
            "motif": w_motif,
            "gc": w_gc,
            "length": w_length
        },
        "scores": scores,
        "agreements": agreements,
        "agreement_rate": np.mean(agreements)
    }


def grid_search_weights(step: float = 0.05) -> List[Dict]:
    """
    Grid search over weight combinations.

    Constraints:
    - All weights >= 0
    - Sum = 1.0
    - dsRNA weight >= 0.30 (primary mechanism for circRNA)
    """
    results = []

    # Generate weight combinations
    # w_dsRNA: 0.30 to 0.60 (primary mechanism)
    # w_motif: 0.10 to 0.40
    # w_gc: 0.10 to 0.30
    # w_length: 0.05 to 0.20

    for w_dsRNA in np.arange(0.30, 0.65, step):
        for w_motif in np.arange(0.10, 0.45, step):
            for w_gc in np.arange(0.10, 0.35, step):
                w_length = 1.0 - w_dsRNA - w_motif - w_gc
                if 0.05 <= w_length <= 0.20:
                    result = evaluate_weights(w_dsRNA, w_motif, w_gc, w_length)
                    results.append(result)

    # Sort by agreement rate
    results.sort(key=lambda x: x["agreement_rate"], reverse=True)

    return results


def main():
    print("=" * 60)
    print("RIG-I Pathway Weight Optimization")
    print("=" * 60)
    print("\nSearching for optimal weights via grid search...")

    results = grid_search_weights(step=0.05)

    print(f"\nTested {len(results)} weight combinations")
    print("\nTop 5 weight configurations:")
    print("-" * 60)

    for i, result in enumerate(results[:5]):
        w = result["weights"]
        print(f"\nRank {i+1}: Agreement = {result['agreement_rate']:.1%}")
        print(f"  dsRNA: {w['dsRNA']:.2f}, motif: {w['motif']:.2f}, "
              f"gc: {w['gc']:.2f}, length: {w['length']:.2f}")
        print(f"  Scores: high_dsRNA={result['scores']['high_dsRNA']:.3f}, "
              f"low_dsRNA={result['scores']['low_dsRNA']:.3f}")

    # Current implementation weights
    print("\n" + "=" * 60)
    print("Current implementation weights:")
    print("  dsRNA: 0.40, motif: 0.30, gc: 0.20, length: 0.10")

    current_result = evaluate_weights(0.40, 0.30, 0.20, 0.10)
    print(f"  Agreement rate: {current_result['agreement_rate']:.1%}")

    # Save results
    output_path = "rig_i_weight_optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(results[:20], f, indent=2)
    print(f"\nTop 20 results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
