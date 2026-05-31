#!/usr/bin/env python3
"""
5D Weight Sensitivity Analysis — P0 experiment for Bioinformatics Application Note.

Three analyses:
A) Weight perturbation robustness: ±50% random perturbation on base weights,
   1000 iterations, track Go/Conditional/No-Go stability for Ψ and unmodified.
B) Threshold scan: sweep Go threshold (0.50-0.80) and Conditional threshold
   (0.30-0.50), record decision boundaries.
C) Per-dimension drop test: remove each dimension one at a time,
   measure composite score change.

Output: results/5d_weight_sensitivity.json + figures/fig_5d_sensitivity.pdf/png
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ── Add project root to path ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import scoring engine ──
from confluencia_joint.scoring import JointScoringEngine

# ── Results directory ──
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "mypaper" / "figures"

# ── Case study inputs (from Table 2 in paper) ──
# Ψ modification: extended half-life (~15h), low immunogenicity → Go
CASE_PSI = {
    "drug_outputs": {
        "efficacy_pred": 0.72,
        "target_binding_pred": 0.65,
        "immune_activation_pred": 0.55,
        "inflammation_risk_pred": 0.05,
        "genotoxicity_risk_pred": 0.03,
    },
    "epitope_outputs": {
        "efficacy_pred": 0.70,
        "pred_uncertainty": 0.15,
    },
    "pk_summary": {
        "pkpd_cmax_mg_per_l": 2.5,
        "pkpd_tmax_h": 12.0,
        "pkpd_half_life_h": 15.61,
        "pkpd_auc_conc": 45.0,
        "pkpd_auc_effect": 35.0,
    },
    "gene_signature_outputs": {
        "trop2": 0.8, "nectin4": 0.7, "liv1": 0.5, "b7h4": 0.4, "tmem65": 0.75,
        "risk_score": 0.35, "efficacy_score": 0.65, "proliferation_score": 0.7,
        "immune_score": 0.55, "mito_score": 0.5, "tide_score": 0.3,
        "ips_estimate": 7.0, "predicted_response": "CR/PR", "dhe_recommended": True,
    },
    "circrna_outputs": {
        "immunotherapy_score": 0.75, "therapeutic_window": 0.65,
        "tumor_killing_index": 0.60, "overall_immunogenicity": 0.25,
        "rig_i_score": 0.20, "tlr_score": 0.15, "pkr_score": 0.30,
        "tide_score": 0.30, "ips": 7.0, "predicted_response": "likely_responder",
        "immune_cycle_score": 0.65, "tme_score": 0.60, "overall": 0.70,
        "trained_model_risk": 0.30,
    },
}

# Unmodified circRNA: short half-life (6.24h), high immunogenicity → No-Go
CASE_UNMOD = {
    "drug_outputs": {
        "efficacy_pred": 0.45,
        "target_binding_pred": 0.40,
        "immune_activation_pred": 0.80,  # high immune activation
        "inflammation_risk_pred": 0.15,
        "genotoxicity_risk_pred": 0.08,
    },
    "epitope_outputs": {
        "efficacy_pred": 0.35,
        "pred_uncertainty": 0.40,
    },
    "pk_summary": {
        "pkpd_cmax_mg_per_l": 0.8,
        "pkpd_tmax_h": 4.0,
        "pkpd_half_life_h": 6.24,
        "pkpd_auc_conc": 8.0,
        "pkpd_auc_effect": 5.0,
    },
    "gene_signature_outputs": {
        "trop2": 0.8, "nectin4": 0.7, "liv1": 0.5, "b7h4": 0.4, "tmem65": 0.75,
        "risk_score": 0.45, "efficacy_score": 0.55, "proliferation_score": 0.7,
        "immune_score": 0.45, "mito_score": 0.45, "tide_score": 0.45,
        "ips_estimate": 5.0, "predicted_response": "SD", "dhe_recommended": False,
    },
    "circrna_outputs": {
        "immunotherapy_score": 0.35, "therapeutic_window": 0.25,
        "tumor_killing_index": 0.30, "overall_immunogenicity": 0.80,  # HIGH
        "rig_i_score": 0.70, "tlr_score": 0.65, "pkr_score": 0.75,
        "tide_score": 0.50, "ips": 4.0, "predicted_response": "likely_non_responder",
        "immune_cycle_score": 0.35, "tme_score": 0.40, "overall": 0.35,
        "trained_model_risk": 0.65,
    },
}

# m6A modification: moderate half-life, moderate immunogenicity → Go
CASE_M6A = {
    "drug_outputs": {
        "efficacy_pred": 0.62,
        "target_binding_pred": 0.55,
        "immune_activation_pred": 0.60,
        "inflammation_risk_pred": 0.10,
        "genotoxicity_risk_pred": 0.05,
    },
    "epitope_outputs": {
        "efficacy_pred": 0.60,
        "pred_uncertainty": 0.25,
    },
    "pk_summary": {
        "pkpd_cmax_mg_per_l": 1.8,
        "pkpd_tmax_h": 8.0,
        "pkpd_half_life_h": 11.24,
        "pkpd_auc_conc": 25.0,
        "pkpd_auc_effect": 20.0,
    },
    "gene_signature_outputs": {
        "trop2": 0.8, "nectin4": 0.7, "liv1": 0.5, "b7h4": 0.4, "tmem65": 0.75,
        "risk_score": 0.40, "efficacy_score": 0.60, "proliferation_score": 0.65,
        "immune_score": 0.50, "mito_score": 0.48, "tide_score": 0.38,
        "ips_estimate": 6.0, "predicted_response": "SD", "dhe_recommended": False,
    },
    "circrna_outputs": {
        "immunotherapy_score": 0.55, "therapeutic_window": 0.50,
        "tumor_killing_index": 0.45, "overall_immunogenicity": 0.50,
        "rig_i_score": 0.45, "tlr_score": 0.40, "pkr_score": 0.55,
        "tide_score": 0.38, "ips": 6.0, "predicted_response": "intermediate",
        "immune_cycle_score": 0.50, "tme_score": 0.50, "overall": 0.52,
        "trained_model_risk": 0.48,
    },
}

CASES = {"Psi": CASE_PSI, "Unmodified": CASE_UNMOD, "m6A": CASE_M6A}

# ── Analysis A: Weight perturbation ──

def analysis_weight_perturbation(n_iter=1000, perturbation_range=0.5):
    """±50% random perturbation on 5D base weights, 1000 iterations."""
    base_weights = [0.30, 0.20, 0.15, 0.15, 0.20]  # clinical, binding, kinetics, gene_sig, circrna
    dim_names = ["clinical", "binding", "kinetics", "gene_signature", "circrna"]

    results = {}
    for case_name, case_inputs in CASES.items():
        decisions = {"Go": 0, "Conditional": 0, "No-Go": 0}
        scores = []

        for seed in range(n_iter):
            rng = np.random.RandomState(seed)
            # Perturb each weight by ±50%
            perturbed = [w * (1 + rng.uniform(-perturbation_range, perturbation_range))
                         for w in base_weights]
            # Normalize to sum to 1.0
            total = sum(perturbed)
            normalized = [w / total for w in perturbed]

            # Create engine with perturbed weights
            engine = JointScoringEngine(
                clinical_weight=normalized[0],
                binding_weight=normalized[1],
                kinetics_weight=normalized[2],
                gene_signature_weight=normalized[3],
                circrna_weight=normalized[4],
            )

            score = engine.score(
                drug_outputs=case_inputs["drug_outputs"],
                epitope_outputs=case_inputs["epitope_outputs"],
                pk_summary=case_inputs["pk_summary"],
                gene_signature_outputs=case_inputs["gene_signature_outputs"],
                circrna_outputs=case_inputs["circrna_outputs"],
            )

            decisions[score.recommendation] += 1
            scores.append(score.composite)

        results[case_name] = {
            "decision_counts": decisions,
            "decision_pcts": {k: v / n_iter * 100 for k, v in decisions.items()},
            "score_mean": np.mean(scores),
            "score_std": np.std(scores),
            "score_min": np.min(scores),
            "score_max": np.max(scores),
            "score_ci_95": [np.percentile(scores, 2.5), np.percentile(scores, 97.5)],
        }

    return results


# ── Analysis B: Threshold scan ──

def analysis_threshold_scan():
    """Sweep Go threshold (0.50-0.80) and Conditional threshold (0.30-0.50)."""
    go_thresholds = np.arange(0.50, 0.85, 0.05)
    cond_thresholds = np.arange(0.30, 0.55, 0.05)

    results = {}
    for case_name, case_inputs in CASES.items():
        grid = []
        # First compute score with default weights
        engine_default = JointScoringEngine()
        score_default = engine_default.score(
            drug_outputs=case_inputs["drug_outputs"],
            epitope_outputs=case_inputs["epitope_outputs"],
            pk_summary=case_inputs["pk_summary"],
            gene_signature_outputs=case_inputs["gene_signature_outputs"],
            circrna_outputs=case_inputs["circrna_outputs"],
        )
        composite = score_default.composite

        for go_t in go_thresholds:
            for cond_t in cond_thresholds:
                if go_t <= cond_t:
                    continue
                # Determine decision based on thresholds
                if composite >= go_t:
                    decision = "Go"
                elif composite >= cond_t:
                    decision = "Conditional"
                else:
                    decision = "No-Go"
                grid.append({
                    "go_threshold": round(go_t, 2),
                    "conditional_threshold": round(cond_t, 2),
                    "composite": round(composite, 4),
                    "decision": decision,
                })

        results[case_name] = {
            "default_composite": round(composite, 4),
            "default_decision": score_default.recommendation,
            "grid": grid,
        }

    return results


# ── Analysis C: Per-dimension drop test ──

def analysis_dimension_drop():
    """Remove each dimension one at a time, measure composite score change."""
    dim_names = ["clinical", "binding", "kinetics", "gene_signature", "circrna"]

    results = {}
    for case_name, case_inputs in CASES.items():
        # Baseline: all 5 dimensions
        engine_full = JointScoringEngine()
        score_full = engine_full.score(
            drug_outputs=case_inputs["drug_outputs"],
            epitope_outputs=case_inputs["epitope_outputs"],
            pk_summary=case_inputs["pk_summary"],
            gene_signature_outputs=case_inputs["gene_signature_outputs"],
            circrna_outputs=case_inputs["circrna_outputs"],
        )
        baseline_composite = score_full.composite
        baseline_decision = score_full.recommendation

        drops = {}
        for dim in dim_names:
            # Set dropped dimension weight to 0, redistribute to others
            base_weights = {"clinical": 0.30, "binding": 0.20, "kinetics": 0.15,
                            "gene_signature": 0.15, "circrna": 0.20}
            base_weights[dim] = 0.0
            total = sum(base_weights.values())
            redistributed = {k: v / total for k, v in base_weights.items()}

            engine_drop = JointScoringEngine(
                clinical_weight=redistributed["clinical"],
                binding_weight=redistributed["binding"],
                kinetics_weight=redistributed["kinetics"],
                gene_signature_weight=redistributed["gene_signature"],
                circrna_weight=redistributed["circrna"],
            )

            # For dropped gene_signature/circrna, omit those inputs
            gs_out = case_inputs["gene_signature_outputs"] if dim != "gene_signature" else None
            cr_out = case_inputs["circrna_outputs"] if dim != "circrna" else None

            score_drop = engine_drop.score(
                drug_outputs=case_inputs["drug_outputs"],
                epitope_outputs=case_inputs["epitope_outputs"],
                pk_summary=case_inputs["pk_summary"],
                gene_signature_outputs=gs_out,
                circrna_outputs=cr_out,
            )

            drops[dim] = {
                "composite": round(score_drop.composite, 4),
                "decision": score_drop.recommendation,
                "delta": round(score_drop.composite - baseline_composite, 4),
                "delta_pct": round((score_drop.composite - baseline_composite) / baseline_composite * 100, 2)
                    if baseline_composite > 0 else 0,
            }

        results[case_name] = {
            "baseline_composite": round(baseline_composite, 4),
            "baseline_decision": baseline_decision,
            "dimension_drops": drops,
        }

    return results


# ── Figure generation ──

def generate_figure(perturb_results, threshold_results, drop_results):
    """Generate publication-quality sensitivity analysis figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Decision stability bar chart
    ax = axes[0]
    case_names = list(perturb_results.keys())
    x = np.arange(len(case_names))
    width = 0.25

    for i, decision in enumerate(["Go", "Conditional", "No-Go"]):
        pcts = [perturb_results[c]["decision_pcts"][decision] for c in case_names]
        colors = ['#A5D6A7', '#FFF9C4', '#EF9A9A']
        ax.bar(x + i * width, pcts, width, label=decision, color=colors[i], edgecolor='#37474F', linewidth=0.8)

    ax.set_xlabel('Modification', fontsize=10)
    ax.set_ylabel('Decision frequency (%)', fontsize=10)
    ax.set_title('A) Decision stability under ±50% weight perturbation', fontsize=10, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(case_names)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add stability annotation
    for ci, cn in enumerate(case_names):
        dominant = max(perturb_results[cn]["decision_pcts"], key=perturb_results[cn]["decision_pcts"].get)
        pct = perturb_results[cn]["decision_pcts"][dominant]
        ax.text(ci + width, pct + 3, f'{pct:.0f}% {dominant}', ha='center', fontsize=7, fontweight='bold')

    # Panel B: Threshold heat map for Ψ
    ax = axes[1]
    psi_grid = threshold_results["Psi"]["grid"]

    # Build matrix
    go_thresholds = sorted(set(round(g["go_threshold"], 2) for g in psi_grid))
    cond_thresholds = sorted(set(round(g["conditional_threshold"], 2) for g in psi_grid))

    matrix = np.full((len(cond_thresholds), len(go_thresholds)), -1)
    for g in psi_grid:
        gi = go_thresholds.index(round(g["go_threshold"], 2))
        ci = cond_thresholds.index(round(g["conditional_threshold"], 2))
        val = {"Go": 2, "Conditional": 1, "No-Go": 0}[g["decision"]]
        matrix[ci, gi] = val

    cmap = plt.cm.colors.ListedColormap(['#EF9A9A', '#FFF9C4', '#A5D6A7'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    ax.set_xlabel('Go threshold', fontsize=10)
    ax.set_ylabel('Conditional threshold', fontsize=10)
    ax.set_title('B) Ψ decision under threshold sweep', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(go_thresholds)))
    ax.set_xticklabels([f'{t:.2f}' for t in go_thresholds], fontsize=8)
    ax.set_yticks(range(len(cond_thresholds)))
    ax.set_yticklabels([f'{t:.2f}' for t in cond_thresholds], fontsize=8)

    # Add composite score annotation
    composite = threshold_results["Psi"]["default_composite"]
    ax.text(0.5, -0.15, f'Ψ composite={composite:.3f} (default thresholds)',
            transform=ax.transAxes, ha='center', fontsize=8, color='#1565C0')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#A5D6A7', label='Go'),
        mpatches.Patch(color='#FFF9C4', label='Conditional'),
        mpatches.Patch(color='#EF9A9A', label='No-Go'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7)

    # Panel C: Dimension drop impact
    ax = axes[2]
    dims = ["clinical", "binding", "kinetics", "gene_signature", "circrna"]
    dim_labels = ["Clinical", "Binding", "Kinetics", "Gene Sig.", "circRNA"]

    for ci, cn in enumerate(case_names):
        deltas = [abs(drop_results[cn]["dimension_drops"][d]["delta"]) for d in dims]
        color = ['#1565C0', '#D32F2F', '#2E7D32'][ci]
        marker = ['o', 's', '^'][ci]
        ax.plot(range(len(dims)), deltas, marker=marker, color=color, linewidth=2,
                markersize=8, label=cn)

    ax.set_xlabel('Dropped dimension', fontsize=10)
    ax.set_ylabel('|Composite score change|', fontsize=10)
    ax.set_title('C) Impact of dropping each dimension', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pdf_path = FIGURES_DIR / "fig_5d_sensitivity.pdf"
    png_path = FIGURES_DIR / "fig_5d_sensitivity.png"

    fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved: {pdf_path}")
    print(f"Figure saved: {png_path}")
    return str(pdf_path)


# ── Main ──

def main():
    print("=" * 60)
    print("5D Weight Sensitivity Analysis")
    print("=" * 60)

    # Analysis A: Weight perturbation
    print("\n[A] Weight perturbation robustness (1000 iterations)...")
    perturb_results = analysis_weight_perturbation(n_iter=1000)
    for case, res in perturb_results.items():
        print(f"  {case}: composite={res['score_mean']:.3f}±{res['score_std']:.3f}, "
              f"95%CI=[{res['score_ci_95'][0]:.3f},{res['score_ci_95'][1]:.3f}]")
        for dec, pct in res["decision_pcts"].items():
            print(f"    {dec}: {pct:.1f}%")

    # Analysis B: Threshold scan
    print("\n[B] Threshold scan...")
    threshold_results = analysis_threshold_scan()
    for case, res in threshold_results.items():
        print(f"  {case}: default composite={res['default_composite']:.3f}, "
              f"default decision={res['default_decision']}")

    # Analysis C: Dimension drop
    print("\n[C] Per-dimension drop test...")
    drop_results = analysis_dimension_drop()
    for case, res in drop_results.items():
        print(f"  {case}: baseline={res['baseline_composite']:.3f} ({res['baseline_decision']})")
        for dim, drop in res["dimension_drops"].items():
            print(f"    drop {dim}: composite={drop['composite']:.3f} ({drop['decision']}), "
                  f"delta={drop['delta']:.3f} ({drop['delta_pct']:.1f}%)")

    # Save results
    all_results = {
        "weight_perturbation": perturb_results,
        "threshold_scan": threshold_results,
        "dimension_drop": drop_results,
        "metadata": {
            "n_iterations": 1000,
            "perturbation_range": 0.5,
            "base_weights": [0.30, 0.20, 0.15, 0.15, 0.20],
            "default_thresholds": {"go": 0.65, "conditional": 0.40, "safety_floor": 0.30},
        }
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = RESULTS_DIR / "5d_weight_sensitivity.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Generate figure
    print("\nGenerating figure...")
    generate_figure(perturb_results, threshold_results, drop_results)

    print("\n" + "=" * 60)
    print("DONE — 5D Weight Sensitivity Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()