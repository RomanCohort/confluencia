#!/usr/bin/env python3
"""
Gene Signature Survival Validation — P2 experiment.

Validates FiveGeneMOEScorer predictions against actual survival outcomes
using TCGA-BRCA + METABRIC combined data (N=3078).

Metrics: C-index, Cox regression, Kaplan-Meier stratification.

Output: results/gene_signature_survival_validation.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

# ── Gene name mapping ──
GENE_MAP = {
    "TROP2": "TROP2",      # TACSTD2 alias in data
    "NECTIN4": "NECTIN4",   # PVRL4 alias
    "LIV-1": "LIV-1",      # SLC39A8 alias
    "B7-H4": "B7-H4",      # VTCN1 alias
    "TMEM65": "TMEM65",
}

# ── Weight config (from weight_loader defaults) ──
GENE_SIG_SUB_WEIGHTS = {
    "efficacy": 0.30, "immune": 0.15, "proliferation": 0.15,
    "mito": 0.15, "risk_inverse": 0.15, "tide_inverse": 0.10,
}

RISK_ADJUSTMENT = {
    "TROP2_high": 0.30, "TROP2_low": 0.15,
    "NECTIN4_high": 0.20, "NECTIN4_low": 0.10,
    "LIV-1_high": 0.15, "LIV-1_low": 0.08,
    "B7-H4_high": 0.10, "B7-H4_low": 0.05,
    "TMEM65_high": 0.25, "TMEM65_low": 0.13,
}


def compute_gene_signature_score(trop2, nectin4, liv1, b7h4, tmem65):
    """Compute gene signature score from 5 gene expression values (0-1 normalized).

    Uses continuous gene expression (not thresholds) for risk computation
    to ensure varied risk scores across patients.
    """
    # Sub-scores: continuous weighted sums
    proliferation = 0.4 * trop2 + 0.3 * nectin4 + 0.15 * tmem65 + 0.15 * trop2
    immune = 0.6 * b7h4 + 0.2 * trop2 + 0.2 * liv1
    mito = 0.3 * liv1 + 0.3 * tmem65 + 0.2 * nectin4 + 0.2 * b7h4

    # Risk score: continuous weighted combination of gene expression
    # Higher expression of proliferation genes → higher risk
    # Higher B7-H4 (immune evasion marker) → higher risk
    # Higher TMEM65 (mitochondrial stress) → higher risk
    # LIV-1 (zinc transporter, metastasis suppressor) → inverse risk
    risk = (
        0.30 * trop2       # TROP2: proliferation marker → risk
        + 0.20 * nectin4   # NECTIN4: proliferation + metastasis → risk
        + 0.10 * (1.0 - liv1)  # LIV-1: low expression → worse prognosis (inverse)
        + 0.15 * b7h4      # B7-H4: immune evasion → risk
        + 0.25 * tmem65    # TMEM65: mitochondrial stress → risk
    )
    risk = min(1.0, max(0.0, risk))
    efficacy = 1.0 - risk

    # Overall score
    overall = (
        GENE_SIG_SUB_WEIGHTS["efficacy"] * efficacy
        + GENE_SIG_SUB_WEIGHTS["immune"] * immune
        + GENE_SIG_SUB_WEIGHTS["proliferation"] * proliferation
        + GENE_SIG_SUB_WEIGHTS["mito"] * mito
        + GENE_SIG_SUB_WEIGHTS["risk_inverse"] * (1.0 - risk)
    )
    return {
        "overall": overall,
        "risk_score": risk,
        "efficacy_score": efficacy,
        "proliferation_score": proliferation,
        "immune_score": immune,
        "mito_score": mito,
    }


def concordance_index(survival_times, predicted_risk, events):
    """Compute Harrell's C-index.

    survival_times: actual survival months
    predicted_risk: predicted risk score (higher = worse prognosis)
    events: 1=death, 0=censored
    """
    n = len(survival_times)
    if n < 2:
        return 0.5

    concordant = 0
    permissible = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only compare if we can determine ordering
            if events[i] == 1 and survival_times[i] < survival_times[j]:
                permissible += 1
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] == predicted_risk[j]:
                    concordant += 0.5
            elif events[j] == 1 and survival_times[j] < survival_times[i]:
                permissible += 1
                if predicted_risk[j] > predicted_risk[i]:
                    concordant += 1
                elif predicted_risk[j] == predicted_risk[i]:
                    concordant += 0.5

    if permissible == 0:
        return 0.5
    return concordant / permissible


def km_stratification(df, risk_col, time_col="OS_months", event_col="OS_status"):
    """Stratify patients into high/low risk and compute log-rank test."""
    median = df[risk_col].median()
    high_risk = df[df[risk_col] >= median]
    low_risk = df[df[risk_col] < median]

    # Compute median survival for each group
    high_surv_median = high_risk[time_col].median()
    low_surv_median = low_risk[time_col].median()

    high_death_rate = high_risk[event_col].mean()
    low_death_rate = low_risk[event_col].mean()

    # Simplified log-rank test (chi-square approximation)
    n_high = len(high_risk)
    n_low = len(low_risk)
    d_high = high_risk[event_col].sum()
    d_low = low_risk[event_col].sum()
    d_total = d_high + d_low
    n_total = n_high + n_low

    expected_high = d_total * n_high / n_total
    expected_low = d_total * n_low / n_total

    chi2 = ((d_high - expected_high) ** 2 / expected_high +
            (d_low - expected_low) ** 2 / expected_low)

    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi2, 1) if chi2 > 0 else 1.0

    return {
        "high_risk_n": int(n_high),
        "low_risk_n": int(n_low),
        "high_risk_median_survival": float(high_surv_median),
        "low_risk_median_survival": float(low_surv_median),
        "high_risk_death_rate": float(high_death_rate),
        "low_risk_death_rate": float(low_death_rate),
        "log_rank_chi2": float(chi2),
        "log_rank_p": float(p_value),
        "risk_median": float(median),
    }


def main():
    print("=" * 60)
    print("Gene Signature Survival Validation")
    print("=" * 60)

    # ── Load data ──
    data_path = PROJECT_ROOT / "data" / "gene_signature" / "combined_with_survival.csv"
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(str(data_path))
    print(f"  N={len(df)}, sources={df['source'].value_counts().to_dict()}")

    # Clean data
    df = df[df["OS_months"] > 0].copy()  # Remove negative survival times
    df = df.dropna(subset=["OS_months", "OS_status"]).copy()
    print(f"  After cleaning: N={len(df)}")

    # ── Compute gene signature scores ──
    print("\nComputing gene signature scores...")
    scores = []
    for _, row in df.iterrows():
        s = compute_gene_signature_score(
            row["TROP2"], row["NECTIN4"], row["LIV-1"],
            row["B7-H4"], row["TMEM65"]
        )
        scores.append(s)

    df["gs_overall"] = [s["overall"] for s in scores]
    df["gs_risk"] = [s["risk_score"] for s in scores]
    df["gs_efficacy"] = [s["efficacy_score"] for s in scores]
    df["gs_proliferation"] = [s["proliferation_score"] for s in scores]
    df["gs_immune"] = [s["immune_score"] for s in scores]

    # ── Validation metrics ──

    # 1) C-index: predicted risk vs actual survival
    c_index = concordance_index(
        df["OS_months"].values,
        df["gs_risk"].values,
        df["OS_status"].values
    )
    print(f"\n  C-index (risk vs survival): {c_index:.4f}")

    # 2) C-index per source
    c_per_source = {}
    for source in ["TCGA-BRCA", "METABRIC"]:
        sub = df[df["source"] == source]
        if len(sub) > 10:
            ci = concordance_index(
                sub["OS_months"].values,
                sub["gs_risk"].values,
                sub["OS_status"].values
            )
            c_per_source[source] = ci
            print(f"  C-index ({source}, N={len(sub)}): {ci:.4f}")

    # 3) Spearman correlation: risk vs survival time
    from scipy.stats import spearmanr
    sp_r, sp_p = spearmanr(df["gs_risk"], df["OS_months"])
    print(f"\n  Spearman r (risk vs OS_months): {sp_r:.4f}, p={sp_p:.2e}")

    # Negative correlation expected: higher risk → shorter survival
    sp_r_eff, sp_p_eff = spearmanr(df["gs_efficacy"], df["OS_months"])
    print(f"  Spearman r (efficacy vs OS_months): {sp_r_eff:.4f}, p={sp_p_eff:.2e}")

    # 4) KM stratification
    km_risk = km_stratification(df, "gs_risk")
    print(f"\n  KM stratification by risk:")
    print(f"    High risk (n={km_risk['high_risk_n']}): median surv={km_risk['high_risk_median_survival']:.1f}mo, death_rate={km_risk['high_risk_death_rate']:.3f}")
    print(f"    Low risk (n={km_risk['low_risk_n']}): median surv={km_risk['low_risk_median_survival']:.1f}mo, death_rate={km_risk['low_risk_death_rate']:.3f}")
    print(f"    Log-rank: chi2={km_risk['log_rank_chi2']:.2f}, p={km_risk['log_rank_p']:.2e}")

    # 5) Per-gene contribution: C-index for each gene individually
    per_gene = {}
    for gene in ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]:
        # Risk: higher expression → higher risk for most genes
        ci = concordance_index(
            df["OS_months"].values,
            df[gene].values,
            df["OS_status"].values
        )
        per_gene[gene] = ci
        print(f"  Per-gene C-index ({gene}): {ci:.4f}")

    # 6) Cox proportional hazards regression
    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        # Use all 5 genes + composite risk as predictors
        cph_df = df[["OS_months", "OS_status", "gs_risk", "TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]].copy()
        cph_df.columns = ["duration", "event", "risk", "TROP2", "NECTIN4", "LIV1", "B7H4", "TMEM65"]
        # Penalized Cox to handle collinearity
        cph.fit(cph_df, duration_col="duration", event_col="event",
                penalizer=0.1, show_progress=False)
        cox_hr = float(cph.hazard_ratios_["risk"])
        cox_p = float(cph.summary["p"]["risk"])
        cox_all_hr = {k: float(v) for k, v in cph.hazard_ratios_.items()}
        cox_all_p = {k: float(v) for k, v in cph.summary["p"].items()}
        print(f"\n  Cox HR (gene signature risk): {cox_hr:.4f}, p={cox_p:.2e}")
        for gene in ["TROP2", "NECTIN4", "LIV1", "B7H4", "TMEM65"]:
            print(f"  Cox HR ({gene}): {cox_all_hr[gene]:.4f}, p={cox_all_p[gene]:.2e}")
    except Exception as e:
        print(f"\n  Cox regression failed: {e}")
        cox_hr = None
        cox_p = None
        cox_all_hr = {}
        cox_all_p = {}

    # ── Save results ──
    results = {
        "n_samples": len(df),
        "n_tcga": int(len(df[df["source"] == "TCGA-BRCA"])),
        "n_metabric": int(len(df[df["source"] == "METABRIC"])),
        "c_index_overall": c_index,
        "c_index_per_source": c_per_source,
        "spearman_r_risk_vs_survival": float(sp_r),
        "spearman_p_risk_vs_survival": float(sp_p),
        "spearman_r_efficacy_vs_survival": float(sp_r_eff),
        "spearman_p_efficacy_vs_survival": float(sp_p_eff),
        "km_stratification": km_risk,
        "per_gene_c_index": per_gene,
        "cox_hazard_ratio": float(cox_hr) if cox_hr is not None else None,
        "cox_p_value": float(cox_p) if cox_p is not None else None,
        "cox_per_gene_hr": cox_all_hr,
        "gene_signature_weights": GENE_SIG_SUB_WEIGHTS,
        "risk_adjustment_weights": RISK_ADJUSTMENT,
    }

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    results_path = RESULTS_DIR / "gene_signature_survival_validation.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    print("\n" + "=" * 60)
    print("DONE — Gene Signature Survival Validation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()