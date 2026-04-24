"""
Ultra-fast experiment: Drug efficacy on 5k sample subset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from core.features import MixedFeatureSpec, build_feature_matrix, logit_transform, inverse_logit

def quick_moe_predict(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_ridge = np.zeros(len(y_train))
    for tr, va in kf.split(Xs_train):
        m = Ridge(alpha=1.0)
        m.fit(Xs_train[tr], y_train[tr])
        oof_ridge[va] = m.predict(Xs_train[va])
    ridge_rmse = np.sqrt(np.mean((y_train - oof_ridge)**2))

    oof_hgb = np.zeros(len(y_train))
    for tr, va in kf.split(Xs_train):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.1, max_iter=100, random_state=42)
        m.fit(Xs_train[tr], y_train[tr])
        oof_hgb[va] = m.predict(Xs_train[va])
    hgb_rmse = np.sqrt(np.mean((y_train - oof_hgb)**2))

    m_ridge = Ridge(alpha=1.0).fit(Xs_train, y_train)
    m_hgb = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.1, max_iter=100, random_state=42).fit(Xs_train, y_train)
    p_ridge = m_ridge.predict(Xs_test)
    p_hgb = m_hgb.predict(Xs_test)

    w_ridge = 1.0 / max(ridge_rmse, 1e-6)
    w_hgb = 1.0 / max(hgb_rmse, 1e-6)
    w_sum = w_ridge + w_hgb
    return (w_ridge * p_ridge + w_hgb * p_hgb) / w_sum

def run_exp(df_full, n_train, n_test, spec, label):
    df = df_full.sample(n=n_train+n_test, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=n_test, random_state=42)
    X_tr, _, _ = build_feature_matrix(train_df, spec)
    X_te, _, _ = build_feature_matrix(test_df, spec)
    y_tr = train_df['efficacy'].values.astype(np.float32)
    y_te = test_df['efficacy'].values.astype(np.float32)
    nf = X_tr.shape[1]

    use_logit = str(getattr(spec, 'target_transform', 'none')).lower() == 'logit'
    y_tr_t = logit_transform(y_tr) if use_logit else y_tr
    y_p = quick_moe_predict(X_tr, y_tr_t, X_te)
    if use_logit:
        y_p = inverse_logit(y_p)
    y_p = np.clip(y_p, 0, 1)
    r2 = r2_score(y_te, y_p)
    mae = mean_absolute_error(y_te, y_p)
    print(f"  {label}: Feats={nf}, R2={r2:.4f}, MAE={mae:.4f}")
    return r2, mae, nf

def main():
    print("Loading dataset...")
    df_full = pd.read_csv('data/breast_cancer_drug_dataset_extended.csv')
    print(f"N = {len(df_full)} (will sample 5k for speed)")

    # Quick test with 5k samples
    results = []
    for name, spec in [
        ('Baseline', MixedFeatureSpec()),
        ('+DR+PK', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True)),
        ('+Cross', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True, use_cross_features=True)),
        ('+Aux', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                  use_cross_features=True, use_auxiliary_labels=True)),
        ('+Logit', MixedFeatureSpec(use_dose_response=True, use_pk_prior=True,
                                     use_cross_features=True, use_auxiliary_labels=True,
                                     target_transform='logit')),
    ]:
        r2, mae, nf = run_exp(df_full, 4000, 1000, spec, name)
        results.append((name, r2, mae, nf))

    print("\n" + "="*60)
    print("RESULTS (5k sample, random split)")
    print("="*60)
    print(f"{'Config':<10} {'Feats':>6} {'R2':>8} {'MAE':>8} {'ΔR2':>8}")
    print("-"*60)
    base = results[0][1]
    for name, r2, mae, nf in results:
        print(f"{name:<10} {nf:>6} {r2:>8.4f} {mae:>8.4f} {r2-base:>+8.4f}")

if __name__ == '__main__':
    main()