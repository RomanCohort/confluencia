"""
train_pathway_weights.py — Train pathway sub-score weights via Cox regression.

For each pathway (proliferation, immune, mitochondrial), fit a Cox model
on the candidate genes using the combined raw expression + survival data.
The non-zero coefficients are normalized to sum to 1 as pathway weights.

Output: output/pathway_weights.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Auto-detect project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/gene_signature/cache")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Pathway candidate genes (column names in combined_raw_with_survival.csv)
PATHWAY_CANDIDATES = {
    "proliferation": ["TROP2", "NECTIN4", "TMEM65", "MKI67", "MYC"],
    "immune": ["B7-H4", "TROP2", "LIV-1", "CDH1", "ESR1"],
    "mitochondria": ["TMEM65", "LIV-1", "NECTIN4", "BAX", "BCL2"],
}

# Display name mapping (internal column names → output keys)
DISPLAY_NAMES = {
    "TROP2": "TROP2", "NECTIN4": "NECTIN4", "LIV-1": "LIV-1",
    "B7-H4": "B7-H4", "TMEM65": "TMEM65",
    "MKI67": "MKI67", "MYC": "MYC",
    "CDH1": "CDH1", "ESR1": "ESR1",
    "BAX": "BAX", "BCL2": "BCL2",
}


def load_data() -> pd.DataFrame:
    """Load combined raw expression + survival data."""
    path = CACHE_DIR / "combined_raw_with_survival.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run build_raw_expression.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    return df


def fit_pathway_cox(df, genes, duration_col="OS_months", event_col="OS_status"):
    """Fit Cox model for a single pathway's candidate genes.

    Returns dict of gene → normalized weight (abs coefficients, sum to 1).
    """
    from lifelines import CoxPHFitter

    # Filter to available genes
    available = [g for g in genes if g in df.columns]
    if not available:
        print(f"  No candidate genes available!")
        return {}

    # Subset and drop NaN
    cols = available + [duration_col, event_col]
    sub = df[cols].dropna()
    if len(sub) < 50:
        print(f"  Too few samples ({len(sub)}) after dropping NaN")
        return {}

    # Standardize
    for g in available:
        mean = sub[g].mean()
        std = sub[g].std()
        if std > 0:
            sub[g] = (sub[g] - mean) / std
        else:
            sub[g] = 0.0

    # Try LASSO Cox with light regularization to get sparse solution
    best_alpha = None
    best_c = 0.5
    best_coefs = {}

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        try:
            cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.0)
            cph.fit(sub, duration_col=duration_col, event_col=event_col,
                    fit_options={"max_steps": 200})

            # Get C-index
            pred = cph.predict_partial_hazard(sub[available])
            from lifelines.utils import concordance_index
            c = concordance_index(sub[duration_col], -pred.values.flatten(), sub[event_col])

            coefs = {g: float(cph.params_.get(g, 0.0)) for g in available}
            nonzero = sum(1 for v in coefs.values() if abs(v) > 1e-6)

            print(f"    alpha={alpha:.2f}: C={c:.4f}, nonzero={nonzero}/{len(available)}")

            if c > best_c or (c > best_c - 0.01 and nonzero >= 2):
                best_c = c
                best_alpha = alpha
                best_coefs = coefs
        except Exception as e:
            print(f"    alpha={alpha:.2f}: FAILED - {e}")
            continue

    if not best_coefs:
        # Fallback: unpenalized Cox
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col=duration_col, event_col=event_col,
                    fit_options={"max_steps": 200})
            best_coefs = {g: float(cph.params_.get(g, 0.0)) for g in available}
            pred = cph.predict_partial_hazard(sub[available])
            from lifelines.utils import concordance_index
            best_c = concordance_index(sub[duration_col], -pred.values.flatten(), sub[event_col])
            print(f"    Unpenalized Cox: C={best_c:.4f}")
        except Exception as e:
            print(f"    Unpenalized Cox also FAILED: {e}")
            return {}

    # Normalize to weights (abs coefficients sum to 1)
    abs_sum = sum(abs(v) for v in best_coefs.values())
    if abs_sum > 0:
        weights = {g: abs(v) / abs_sum for g, v in best_coefs.items()}
    else:
        # Equal weights fallback
        weights = {g: 1.0 / len(available) for g in available}

    # Filter out near-zero weights (<1%)
    weights = {g: w for g, w in weights.items() if w >= 0.01}
    # Re-normalize
    wsum = sum(weights.values())
    if wsum > 0:
        weights = {g: w / wsum for g, w in weights.items()}

    return weights, best_c


def main():
    print("=" * 60)
    print("Training Pathway Weights via Cox Regression")
    print("=" * 60)

    df = load_data()

    # Convert OS_status if needed
    if df["OS_status"].dtype == object:
        status_map = {"Deceased": 1, "Living": 0, "DECEASED": 1, "LIVING": 0}
        df["OS_status"] = df["OS_status"].map(status_map).fillna(df["OS_status"])

    results = {}
    c_indices = {}

    for pathway, genes in PATHWAY_CANDIDATES.items():
        print(f"\n--- {pathway} pathway ---")
        print(f"  Candidates: {genes}")

        out = fit_pathway_cox(df, genes)
        if out:
            weights, c = out
            results[pathway] = weights
            c_indices[pathway] = c
            print(f"  Best C-index: {c:.4f}")
            print(f"  Weights:")
            for g, w in sorted(weights.items(), key=lambda x: -x[1]):
                print(f"    {DISPLAY_NAMES.get(g, g)}: {w:.4f}")
        else:
            print(f"  FAILED — will use default weights")

    # Save
    report = {
        "method": "Cox_regression_pathway_weights",
        "pathways": results,
        "c_indices": c_indices,
        "n_samples": len(df),
    }

    out_path = OUTPUT_DIR / "pathway_weights.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pathway, weights in results.items():
        print(f"\n{pathway} (C={c_indices[pathway]:.4f}):")
        for g, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {DISPLAY_NAMES.get(g, g)}: {w:.3f}")

    # Show old vs new
    OLD = {
        "proliferation": {"TROP2": 0.4, "NECTIN4": 0.3, "TMEM65": 0.3},
        "immune": {"B7-H4": 0.6, "TROP2": 0.2, "LIV-1": 0.2},
        "mitochondria": {"TMEM65": 0.7, "LIV-1": 0.2, "NECTIN4": 0.1},
    }
    print(f"\n--- Old (hardcoded) vs New (Cox fitted) ---")
    for pathway in results:
        print(f"\n{pathway}:")
        print(f"  Old: {OLD.get(pathway, {})}")
        print(f"  New: {results[pathway]}")

    print("\nDone!")


if __name__ == "__main__":
    main()