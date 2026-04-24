"""
Full-scale drug efficacy enhancement experiment (91k dataset).
Tests all 6 enhancement strategies with both random and group-aware splits.
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # D:\IGEM集成方案

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from core.features import MixedFeatureSpec, build_feature_matrix, logit_transform, inverse_logit


def quick_moe_predict(X_train, y_train, X_test):
    """2-expert MOE (Ridge + HGB) with OOF-RMSE weighting."""
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_ridge = np.zeros(len(y_train))
    for tr, va in kf.split(Xs_train):
        m = Ridge(alpha=1.0)
        m.fit(Xs_train[tr], y_train[tr])
        oof_ridge[va] = m.predict(Xs_train[va])
    ridge_rmse = np.sqrt(np.mean((y_train - oof_ridge) ** 2))

    oof_hgb = np.zeros(len(y_train))
    for tr, va in kf.split(Xs_train):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.1, max_iter=100, random_state=42)
        m.fit(Xs_train[tr], y_train[tr])
        oof_hgb[va] = m.predict(Xs_train[va])
    hgb_rmse = np.sqrt(np.mean((y_train - oof_hgb) ** 2))

    m_ridge = Ridge(alpha=1.0).fit(Xs_train, y_train)
    m_hgb = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.1, max_iter=100, random_state=42).fit(Xs_train, y_train)
    p_ridge = m_ridge.predict(Xs_test)
    p_hgb = m_hgb.predict(Xs_test)

    w_ridge = 1.0 / max(ridge_rmse, 1e-6)
    w_hgb = 1.0 / max(hgb_rmse, 1e-6)
    w_sum = w_ridge + w_hgb
    return (w_ridge * p_ridge + w_hgb * p_hgb) / w_sum


def run_exp(X_tr, y_tr, X_te, y_te, label, use_logit=False):
    """Run one experiment config."""
    y_tr_t = logit_transform(y_tr) if use_logit else y_tr
    y_p = quick_moe_predict(X_tr, y_tr_t, X_te)
    if use_logit:
        y_p = inverse_logit(y_p)
    y_p = np.clip(y_p, 0, 1)
    r2 = r2_score(y_te, y_p)
    mae = mean_absolute_error(y_te, y_p)
    rmse = np.sqrt(mean_squared_error(y_te, y_p))
    return r2, mae, rmse


def build_features_for_configs(df_train, df_test, configs):
    """Pre-build feature matrices for all configs sharing base settings."""
    results = {}
    for name, spec in configs:
        t0 = time.time()
        X_tr, _, _ = build_feature_matrix(df_train, spec)
        X_te, _, _ = build_feature_matrix(df_test, spec)
        elapsed = time.time() - t0
        print(f"  {name}: {X_tr.shape[1]}d, feature build {elapsed:.1f}s")
        results[name] = (X_tr, X_te)
    return results


def main():
    print("=" * 70)
    print("Full-Scale Drug Enhancement Experiment")
    print("=" * 70)

    print("\nLoading dataset...")
    df_full = pd.read_csv('data/breast_cancer_drug_dataset_extended.csv')
    print(f"N = {len(df_full)}")

    # Use 10k samples for speed
    N = min(10000, len(df_full))
    df = df_full.sample(n=N, random_state=42).reset_index(drop=True)
    print(f"Using {N} samples")

    # Feature configs to test
    configs = [
        ('Baseline', MixedFeatureSpec(), False),
        ('+DR+PK', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True), False),
        ('+Cross', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                    use_cross_features=True), False),
        ('+Aux', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                  use_cross_features=True, use_auxiliary_labels=True), False),
        ('+Logit', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                    use_cross_features=True, use_auxiliary_labels=True,
                                    target_transform='logit'), True),
        ('Full(+ALL)', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                        use_cross_features=True, use_auxiliary_labels=True,
                                        target_transform='logit'), True),
    ]

    # ============================
    # Experiment 1: Random Split
    # ============================
    print("\n" + "=" * 70)
    print("Experiment 1: Random Split (80/20)")
    print("=" * 70)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_tr = train_df['efficacy'].values.astype(np.float32)
    y_te = test_df['efficacy'].values.astype(np.float32)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    random_results = []
    for name, spec, use_logit in configs:
        print(f"\n  [{name}] Building features...")
        t0 = time.time()
        X_tr, _, _ = build_feature_matrix(train_df, spec)
        X_te, _, _ = build_feature_matrix(test_df, spec)
        feat_time = time.time() - t0
        print(f"  Features: {X_tr.shape[1]}d in {feat_time:.1f}s")

        t0 = time.time()
        r2, mae, rmse = run_exp(X_tr, y_tr, X_te, y_te, name, use_logit)
        train_time = time.time() - t0
        random_results.append((name, X_tr.shape[1], r2, mae, rmse, train_time))
        print(f"  R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({train_time:.1f}s)")

    # ============================
    # Experiment 2: Group-Aware Split
    # ============================
    print("\n" + "=" * 70)
    print("Experiment 2: Group-Aware Split (GroupKFold)")
    print("=" * 70)

    # Create groups from smiles (same molecule = same group)
    unique_smiles = df['smiles'].unique()
    smile_to_group = {s: i for i, s in enumerate(unique_smiles)}
    groups = df['smiles'].map(smile_to_group).values

    # Use GroupKFold to split
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, groups=groups))
    # Use last fold as test
    train_idx, test_idx = splits[-1]
    train_df_g = df.iloc[train_idx].reset_index(drop=True)
    test_df_g = df.iloc[test_idx].reset_index(drop=True)
    y_tr_g = train_df_g['efficacy'].values.astype(np.float32)
    y_te_g = test_df_g['efficacy'].values.astype(np.float32)
    print(f"Train: {len(train_df_g)}, Test: {len(test_df_g)}")
    print(f"Train groups: {train_df_g['smiles'].nunique()}, Test groups: {test_df_g['smiles'].nunique()}")

    group_results = []
    for name, spec, use_logit in configs:
        print(f"\n  [{name}] Building features...")
        t0 = time.time()
        X_tr, _, _ = build_feature_matrix(train_df_g, spec)
        X_te, _, _ = build_feature_matrix(test_df_g, spec)
        feat_time = time.time() - t0

        t0 = time.time()
        r2, mae, rmse = run_exp(X_tr, y_tr_g, X_te, y_te_g, name, use_logit)
        train_time = time.time() - t0
        group_results.append((name, X_tr.shape[1], r2, mae, rmse, train_time))
        print(f"  R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({train_time:.1f}s)")

    # ============================
    # Summary
    # ============================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Random Split ---")
    print(f"{'Config':<12} {'Feats':>6} {'R2':>8} {'MAE':>8} {'dR2':>8}")
    print("-" * 50)
    base_r2 = random_results[0][2]
    for name, nf, r2, mae, rmse, t in random_results:
        print(f"{name:<12} {nf:>6} {r2:>8.4f} {mae:>8.4f} {r2 - base_r2:>+8.4f}")

    print("\n--- Group-Aware Split ---")
    print(f"{'Config':<12} {'Feats':>6} {'R2':>8} {'MAE':>8} {'dR2':>8}")
    print("-" * 50)
    base_r2_g = group_results[0][2]
    for name, nf, r2, mae, rmse, t in group_results:
        print(f"{name:<12} {nf:>6} {r2:>8.4f} {mae:>8.4f} {r2 - base_r2_g:>+8.4f}")

    # Generalization gap
    print("\n--- Generalization Gap (Random R2 - Group R2) ---")
    print(f"{'Config':<12} {'Random':>8} {'Group':>8} {'Gap':>8}")
    print("-" * 42)
    for i, (name, _, _, _, _, _) in enumerate(random_results):
        r_r2 = random_results[i][2]
        g_r2 = group_results[i][2]
        print(f"{name:<12} {r_r2:>8.4f} {g_r2:>8.4f} {r_r2 - g_r2:>+8.4f}")

    # Best config recommendation
    print("\n--- Recommendation ---")
    best_random = max(random_results, key=lambda x: x[2])
    best_group = max(group_results, key=lambda x: x[2])
    print(f"Best random split:  {best_random[0]} (R2={best_random[2]:.4f})")
    print(f"Best group split:   {best_group[0]} (R2={best_group[2]:.4f})")


if __name__ == '__main__':
    main()
