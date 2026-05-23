"""
train_five_gene_lasso_v2.py — Retrain with raw log2 values + expanded gene pool + clinical features.

Key improvements over v1:
1. Raw log2 expression values (NOT min-max normalized 0-1)
2. Expanded gene pool: 5 ADC targets + 17 extra genes from 19-gene model = 22 genes
3. Clinical features: grade, ER/HER2/PR status, tumor_stage
4. True StepCox (forward AIC selection)
5. 5-fold CV + external validation (TCGA→METABRIC)

Output: output/lasso_stepcox_v2_report.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/gene_signature/cache")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 5 ADC target genes (display names)
FIVE_GENE_DISPLAY = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]

# Extra 17 genes (display names = gene symbols)
EXTRA_GENE_SYMBOLS = [
    "ERBB2", "PIK3CA", "GATA3", "MKI67", "BRCA1", "LARP6",
    "ESR1", "NR1H3", "BAX", "PSMD2", "CDH1", "CASP3",
    "AKT1", "BCL2", "MYC", "TP53", "PTEN",
]

# Clinical feature columns
CLINICAL_COLS = ["grade", "ER_positive", "HER2_positive", "PR_positive", "tumor_stage"]


def load_combined_data() -> pd.DataFrame:
    """Load combined raw expression + survival + clinical data."""
    path = CACHE_DIR / "combined_raw_with_survival.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run build_raw_expression.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    print(f"  Source distribution: {df['source'].value_counts().to_dict()}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and survival outcome.

    Returns
    -------
    X : pd.DataFrame — feature matrix (samples × genes + clinical)
    duration : pd.Series — OS_months
    event : pd.Series — OS_status (1=deceased, 0=censored)
    gene_cols : list — gene column names in X
    clin_cols : list — clinical column names in X
    """
    # Determine available gene columns
    available_genes = [c for c in df.columns if c in FIVE_GENE_DISPLAY + EXTRA_GENE_SYMBOLS]
    available_clin = [c for c in CLINICAL_COLS if c in df.columns]

    print(f"\nAvailable genes: {len(available_genes)}/{len(FIVE_GENE_DISPLAY + EXTRA_GENE_SYMBOLS)}")
    print(f"Available clinical: {len(available_clin)}/{len(CLINICAL_COLS)}")

    # Check coverage
    for g in available_genes:
        n_valid = df[g].notna().sum()
        print(f"  {g}: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")

    for c in available_clin:
        n_valid = df[c].notna().sum()
        print(f"  {c}: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")

    # OS_status conversion
    if df["OS_status"].dtype == object:
        status_map = {"Deceased": 1, "Living": 0, "DECEASED": 1, "LIVING": 0,
                       "1": 1, "0": 0, 1: 1, 0: 0}
        event = df["OS_status"].map(status_map)
    else:
        event = df["OS_status"]

    duration = df["OS_months"]
    valid_mask = duration.notna() & event.notna() & (duration > 0)
    print(f"\nValid survival: {valid_mask.sum()}/{len(df)}")

    # Keep only samples with valid survival + at least some gene data
    gene_valid = df[available_genes].notna().any(axis=1)
    valid_mask = valid_mask & gene_valid
    print(f"After gene filter: {valid_mask.sum()}/{len(df)}")

    # Feature matrix
    feature_cols = available_genes + available_clin
    X = df.loc[valid_mask, feature_cols].copy()
    duration = duration[valid_mask]
    event = event[valid_mask]
    source = df.loc[valid_mask, "source"].copy()

    # Impute missing clinical features with median
    for col in available_clin:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  Imputed {col} median={median_val:.3f}")

    # Impute missing gene values with per-cohort median
    for col in available_genes:
        if X[col].isna().any():
            for src in source.unique():
                mask = (source == src) & X[col].isna()
                cohort_median = X.loc[source == src, col].median()
                X.loc[mask, col] = cohort_median
            # Any remaining NaN (entire cohort missing) → overall median
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

    print(f"\nFinal feature matrix: {X.shape}")
    print(f"  NaN remaining: {X.isna().sum().sum()}")

    return X, duration, event, available_genes, available_clin, source


# ---------------------------------------------------------------------------
# LASSO Cox with cross-validation
# ---------------------------------------------------------------------------

def lasso_cox_cv(X, duration, event, gene_cols, clin_cols, n_folds=5):
    """LASSO Cox path with 5-fold CV to select optimal alpha.

    Uses lifelines CoxPHFitter with L1 penalizer.

    Returns
    -------
    best_alpha : float
    best_c_index : float
    coefs : dict — gene/coefficient mapping
    """
    from lifelines import CoxPHFitter

    feature_cols = gene_cols + clin_cols
    alphas = np.logspace(-3, 1, 30)  # 0.001 to 10

    df_work = X[feature_cols].copy()
    df_work["duration"] = duration.values
    df_work["event"] = event.values

    # Standardize features for LASSO
    means = df_work[feature_cols].mean()
    stds = df_work[feature_cols].std().replace(0, 1)
    df_work[feature_cols] = (df_work[feature_cols] - means) / stds

    # CV
    np.random.seed(42)
    n = len(df_work)
    fold_ids = np.random.randint(0, n_folds, n)

    results = []
    for alpha in alphas:
        c_indices = []
        for fold in range(n_folds):
            train = df_work[fold_ids != fold]
            test = df_work[fold_ids == fold]

            try:
                cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.0)
                cph.fit(train, duration_col="duration", event_col="event",
                        fit_options={"max_steps": 200})

                # Predict on test
                pred = cph.predict_partial_hazard(test[feature_cols])
                c = concordance_index(test["duration"].values,
                                       -pred.values.flatten(),  # negative: higher hazard = shorter survival
                                       test["event"].values)
                c_indices.append(c)
            except Exception:
                c_indices.append(0.5)

        mean_c = np.mean(c_indices)
        results.append((alpha, mean_c))
        if mean_c > 0.56:
            print(f"    alpha={alpha:.4f}: C-index={mean_c:.4f}")

    # Best alpha
    results.sort(key=lambda x: x[1], reverse=True)
    best_alpha, best_c = results[0]
    print(f"\n  Best LASSO alpha: {best_alpha:.4f}, C-index: {best_c:.4f}")

    # Refit on full data with best alpha
    cph = CoxPHFitter(penalizer=best_alpha, l1_ratio=1.0)
    cph.fit(df_work, duration_col="duration", event_col="event",
            fit_options={"max_steps": 200})

    # Extract non-zero coefficients
    coefs = {}
    for col in feature_cols:
        c = cph.params_.get(col, 0.0)
        if abs(c) > 1e-6:
            coefs[col] = float(c)

    print(f"  Non-zero features: {len(coefs)}/{len(feature_cols)}")
    for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat}: {coef:.4f}")

    return best_alpha, best_c, coefs, means, stds


# ---------------------------------------------------------------------------
# True StepCox — forward stepwise AIC selection
# ---------------------------------------------------------------------------

def stepwise_cox_forward(X, duration, event, candidate_genes, clin_cols):
    """Forward stepwise Cox regression using AIC criterion.

    Starting from clinical features only, add genes one at a time
    choosing the one that most improves AIC.

    Returns
    -------
    selected_genes : list — genes selected by StepCox
    coefs : dict — final coefficients
    c_index : float
    """
    from lifelines import CoxPHFitter

    base_features = list(clin_cols)
    remaining = list(candidate_genes)
    selected = []
    current_aic = np.inf

    # Compute baseline AIC (clinical only)
    df_work = X[base_features].copy()
    df_work["duration"] = duration.values
    df_work["event"] = event.values

    # Standardize
    means = df_work[base_features].mean()
    stds = df_work[base_features].std().replace(0, 1)
    df_work[base_features] = (df_work[base_features] - means) / stds

    try:
        cph = CoxPHFitter()
        cph.fit(df_work, duration_col="duration", event_col="event",
                fit_options={"max_steps": 200})
        current_aic = cph.AIC_partial_
        print(f"  Baseline AIC (clinical only): {current_aic:.2f}")
    except Exception:
        print("  Baseline model failed, starting from empty model")
        current_aic = np.inf

    # Forward selection
    while remaining:
        best_gene = None
        best_aic = current_aic
        best_coefs = {}

        for gene in remaining:
            test_features = base_features + selected + [gene]
            df_test = X[test_features].copy()
            df_test["duration"] = duration.values
            df_test["event"] = event.values

            # Standardize
            means_t = df_test[test_features].mean()
            stds_t = df_test[test_features].std().replace(0, 1)
            df_test[test_features] = (df_test[test_features] - means_t) / stds_t

            try:
                cph = CoxPHFitter()
                cph.fit(df_test, duration_col="duration", event_col="event",
                        fit_options={"max_steps": 200})
                aic = cph.AIC_partial_

                if aic < best_aic:
                    best_aic = aic
                    best_gene = gene
                    best_coefs = {col: float(cph.params_.get(col, 0.0)) for col in test_features}
            except Exception:
                continue

        if best_gene is None:
            break

        selected.append(best_gene)
        remaining.remove(best_gene)
        current_aic = best_aic
        print(f"  + {best_gene}: AIC={current_aic:.2f}")

    # Refit final model to get C-index
    final_features = base_features + selected
    if not selected and not base_features:
        print("  No features selected!")
        return [], {}, 0.5

    df_final = X[final_features].copy()
    df_final["duration"] = duration.values
    df_final["event"] = event.values

    means_f = df_final[final_features].mean()
    stds_f = df_final[final_features].std().replace(0, 1)
    df_final[final_features] = (df_final[final_features] - means_f) / stds_f

    cph = CoxPHFitter()
    cph.fit(df_final, duration_col="duration", event_col="event",
            fit_options={"max_steps": 200})

    pred = cph.predict_partial_hazard(df_final[final_features])
    c_index = concordance_index(
        df_final["duration"].values,
        -pred.values.flatten(),
        df_final["event"].values
    )

    coefs = {col: float(cph.params_.get(col, 0.0)) for col in final_features}

    return selected, coefs, c_index


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------

def concordance_index(y_time, y_pred, y_event):
    """Compute Harrell's concordance index."""
    from lifelines.utils import concordance_index as ci
    return ci(y_time, y_pred, y_event)


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------

def external_validation(X, duration, event, source, features, coefs, means, stds):
    """Train on TCGA, validate on METABRIC (and vice versa)."""
    tcga_mask = source == "TCGA-BRCA"
    metabric_mask = source == "METABRIC"

    results = {}

    # TCGA → METABRIC
    if tcga_mask.sum() > 50 and metabric_mask.sum() > 50:
        from lifelines import CoxPHFitter

        # Train on TCGA
        train_df = X.loc[tcga_mask, features].copy()
        train_df["duration"] = duration[tcga_mask].values
        train_df["event"] = event[tcga_mask].values

        # Standardize on training set
        train_means = train_df[features].mean()
        train_stds = train_df[features].std().replace(0, 1)
        train_df[features] = (train_df[features] - train_means) / train_stds

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train_df, duration_col="duration", event_col="event",
                    fit_options={"max_steps": 200})

            # Predict on METABRIC
            test_df = X.loc[metabric_mask, features].copy()
            test_df[features] = (test_df[features] - train_means) / train_stds
            pred = cph.predict_partial_hazard(test_df)

            c_ext = concordance_index(
                duration[metabric_mask].values,
                -pred.values.flatten(),
                event[metabric_mask].values
            )
            results["TCGA→METABRIC"] = c_ext
            print(f"  TCGA→METABRIC: C-index={c_ext:.4f}")
        except Exception as e:
            print(f"  TCGA→METABRIC failed: {e}")

    # METABRIC → TCGA
    if tcga_mask.sum() > 50 and metabric_mask.sum() > 50:
        train_df = X.loc[metabric_mask, features].copy()
        train_df["duration"] = duration[metabric_mask].values
        train_df["event"] = event[metabric_mask].values

        train_means = train_df[features].mean()
        train_stds = train_df[features].std().replace(0, 1)
        train_df[features] = (train_df[features] - train_means) / train_stds

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train_df, duration_col="duration", event_col="event",
                    fit_options={"max_steps": 200})

            test_df = X.loc[tcga_mask, features].copy()
            test_df[features] = (test_df[features] - train_means) / train_stds
            pred = cph.predict_partial_hazard(test_df)

            c_ext = concordance_index(
                duration[tcga_mask].values,
                -pred.values.flatten(),
                event[tcga_mask].values
            )
            results["METABRIC→TCGA"] = c_ext
            print(f"  METABRIC→TCGA: C-index={c_ext:.4f}")
        except Exception as e:
            print(f"  METABRIC→TCGA failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LASSO Cox + StepCox v2 Training")
    print("Raw log2 values + expanded gene pool + clinical features")
    print("=" * 60)

    # Load data
    df = load_combined_data()

    # Prepare features
    X, duration, event, gene_cols, clin_cols, source = prepare_features(df)

    print(f"\n{'='*60}")
    print(f"Gene pool: {len(gene_cols)} genes")
    print(f"Clinical features: {len(clin_cols)}")
    print(f"Total features: {len(gene_cols) + len(clin_cols)}")
    print(f"Samples: {len(X)} (TCGA={sum(source=='TCGA-BRCA')}, METABRIC={sum(source=='METABRIC')})")
    print(f"{'='*60}")

    # --- Step 1: LASSO Cox CV ---
    print("\n--- Step 1: LASSO Cox CV ---")
    best_alpha, best_c, lasso_coefs, feat_means, feat_stds = lasso_cox_cv(
        X, duration, event, gene_cols, clin_cols
    )

    # Non-zero genes from LASSO
    lasso_genes = [g for g in gene_cols if g in lasso_coefs]
    print(f"\nLASSO selected {len(lasso_genes)} genes: {lasso_genes}")
    print(f"LASSO selected {len([c for c in clin_cols if c in lasso_coefs])} clinical features")

    # --- Step 2: True StepCox (forward AIC) ---
    print("\n--- Step 2: StepCox Forward AIC ---")
    step_genes, step_coefs, step_c = stepwise_cox_forward(
        X, duration, event, lasso_genes, clin_cols
    )
    print(f"\nStepCox selected genes: {step_genes}")
    print(f"StepCox C-index: {step_c:.4f}")

    # --- Step 3: Final model (5 ADC genes + StepCox-selected + clinical) ---
    # IMPORTANT: Force-include all 5 ADC target genes because:
    # 1. They are the drug targets (TROP2, NECTIN4, LIV-1, B7-H4, TMEM65)
    # 2. The acRGBS score requires all 5 genes to be represented
    # 3. Even if LASSO shrinks them, they have biological relevance for DHE ADC
    print("\n--- Step 3: Final Model (5 ADC genes forced + StepCox-selected + clinical) ---")
    from lifelines import CoxPHFitter

    # Force-include 5 ADC genes + StepCox-selected genes (deduplicated)
    forced_genes = [g for g in FIVE_GENE_DISPLAY if g in gene_cols]
    other_genes = [g for g in step_genes if g not in forced_genes]
    final_gene_features = forced_genes + other_genes
    final_clin_features = [c for c in clin_cols if c in step_coefs]

    final_features = final_gene_features + final_clin_features
    if not final_features:
        # Fallback: use all LASSO non-zero features
        final_features = list(lasso_coefs.keys())
    if not final_features:
        print("ERROR: No features selected by either LASSO or StepCox")
        return

    df_final = X[final_features].copy()
    df_final["duration"] = duration.values
    df_final["event"] = event.values

    # Standardize
    final_means = df_final[final_features].mean()
    final_stds = df_final[final_features].std().replace(0, 1)
    df_final[final_features] = (df_final[final_features] - final_means) / final_stds

    cph_final = CoxPHFitter()
    cph_final.fit(df_final, duration_col="duration", event_col="event",
                  fit_options={"max_steps": 200})

    pred_final = cph_final.predict_partial_hazard(df_final[final_features])
    c_train = concordance_index(
        df_final["duration"].values,
        -pred_final.values.flatten(),
        df_final["event"].values
    )
    print(f"  Final C-index (train): {c_train:.4f}")
    print(f"  Final AIC: {cph_final.AIC_partial_:.2f}")

    final_coefs = {col: float(cph_final.params_.get(col, 0.0)) for col in final_features}
    for feat, coef in sorted(final_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat}: {coef:.4f}")

    # --- Step 4: 5-fold CV ---
    print("\n--- Step 4: 5-Fold CV ---")
    np.random.seed(42)
    n = len(df_final)
    fold_ids = np.random.randint(0, 5, n)
    cv_c_indices = []

    for fold in range(5):
        train = df_final[fold_ids != fold]
        test = df_final[fold_ids == fold]

        try:
            cph = CoxPHFitter()
            cph.fit(train, duration_col="duration", event_col="event",
                    fit_options={"max_steps": 200})
            pred = cph.predict_partial_hazard(test[final_features])
            c = concordance_index(test["duration"].values,
                                   -pred.values.flatten(),
                                   test["event"].values)
            cv_c_indices.append(c)
        except Exception:
            cv_c_indices.append(0.5)

    cv_mean = np.mean(cv_c_indices)
    cv_std = np.std(cv_c_indices)
    print(f"  5-Fold CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")

    # --- Step 5: External validation ---
    print("\n--- Step 5: External Validation ---")
    ext_results = external_validation(
        X, duration, event, source, final_features,
        final_coefs, final_means, final_stds
    )

    # --- Step 6: Compute normalized weights for 5-gene acRGBS ---
    print("\n--- Step 6: Normalized Weights for acRGBS ---")
    # Use final model coefficients (5 ADC genes are force-included)
    five_gene_coefs = {g: final_coefs.get(g, 0.0) for g in FIVE_GENE_DISPLAY}
    abs_sum = sum(abs(v) for v in five_gene_coefs.values())

    if abs_sum > 0:
        normalized_weights = {g: abs(v) / abs_sum for g, v in five_gene_coefs.items()}
    else:
        # All 5 genes have zero coefficients — use equal weights as fallback
        normalized_weights = {g: 1.0 / len(FIVE_GENE_DISPLAY) for g in FIVE_GENE_DISPLAY}

    print("  5-gene acRGBS weights (from final model with forced ADC genes):")
    for g, w in sorted(normalized_weights.items(), key=lambda x: -x[1]):
        print(f"    {g}: {w:.4f} (coef={five_gene_coefs.get(g, 0.0):.4f})")

    # --- Step 7: Save report ---
    report = {
        "method": "LASSO_Cox_StepCox_v2",
        "description": "Raw log2 values + expanded gene pool + clinical features",
        "gene_pool_size": len(gene_cols),
        "clinical_features": clin_cols,
        "lasso": {
            "best_alpha": best_alpha,
            "best_c_index": best_c,
            "nonzero_genes": lasso_genes,
            "nonzero_clinical": [c for c in clin_cols if c in lasso_coefs],
            "coefficients": lasso_coefs,
        },
        "stepcox": {
            "selected_genes": step_genes,
            "coefficients": step_coefs,
            "c_index": step_c,
        },
        "final_model": {
            "features": final_features,
            "genes": final_gene_features,
            "adc_genes_forced": forced_genes,
            "stepcox_genes": step_genes,
            "clinical": final_clin_features,
            "coefficients": final_coefs,
            "normalized_weights": normalized_weights,
            "c_index_train": c_train,
            "c_index_cv_mean": cv_mean,
            "c_index_cv_std": cv_std,
        },
        "external_validation": {
            k: float(v) for k, v in ext_results.items()
        },
        "n_samples": len(X),
        "n_tcga": int(sum(source == "TCGA-BRCA")),
        "n_metabric": int(sum(source == "METABRIC")),
        "feature_means": {k: float(v) for k, v in final_means.items()},
        "feature_stds": {k: float(v) for k, v in final_stds.items()},
        "baseline_cumulative_hazard": None,  # Could add cph_final.baseline_cumulative_hazard_
    }

    report_path = OUTPUT_DIR / "lasso_stepcox_v2_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Also save coefficients CSV
    coef_df = pd.DataFrame([
        {"feature": k, "coefficient": v} for k, v in final_coefs.items()
    ])
    coef_path = OUTPUT_DIR / "lasso_stepcox_v2_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Coefficients saved to {coef_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Gene pool: {len(gene_cols)} genes + {len(clin_cols)} clinical")
    print(f"LASSO selected: {len(lasso_genes)} genes, {len([c for c in clin_cols if c in lasso_coefs])} clinical")
    print(f"StepCox selected: {len(step_genes)} genes")
    print(f"Final features: {len(final_features)}")
    print(f"C-index (train): {c_train:.4f}")
    print(f"C-index (5-fold CV): {cv_mean:.4f} ± {cv_std:.4f}")
    for k, v in ext_results.items():
        print(f"C-index ({k}): {v:.4f}")
    print(f"\nacRGBS normalized weights:")
    for g, w in sorted(normalized_weights.items(), key=lambda x: -x[1]):
        print(f"  {g}: {w:.3f}")

    # Comparison with v1
    v1_path = OUTPUT_DIR / "lasso_stepcox_report.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"\nv1 (normalized 0-1, 5 genes): C-index={v1['final_model']['c_index_train']:.4f}")
        print(f"v2 (raw log2, {len(gene_cols)} genes + clinical): C-index={c_train:.4f}")
        improvement = c_train - v1['final_model']['c_index_train']
        print(f"Improvement: {improvement:+.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
