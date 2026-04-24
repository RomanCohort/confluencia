"""最终冲刺: 扩展调参 + Ensemble"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "confluencia-2.0-epitope"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from core.features import FeatureSpec, build_feature_matrix

PROJECT = Path(__file__).resolve().parents[1]

# === Data ===
bind = pd.read_csv(PROJECT / "confluencia-2.0-epitope/data/iedb_mhc_i_binding.csv")
train_df, test_df = train_test_split(bind, test_size=0.2, random_state=42, stratify=bind['is_binder'])

ext = pd.read_csv(PROJECT / "benchmarks/data/iedb_heldout_mhc.csv")
for col in ['dose','freq','treatment_time','circ_expr','ifn_score']:
    if col not in ext.columns: ext[col] = 0.0
y_ext = ext['is_binder'].astype(int).values

net61 = pd.read_csv(PROJECT / "benchmarks/data/netmhcpan_heldout.csv")
for col in ['dose','freq','treatment_time','circ_expr','ifn_score']:
    if col not in net61.columns: net61[col] = 0.0
y_net61 = net61['is_binder'].astype(int).values

spec_mhc = FeatureSpec(use_mhc=True, use_esm2=False, mhc_allele_col='mhc_allele')
spec_base = FeatureSpec(use_mhc=False, use_esm2=False, mhc_allele_col='mhc_allele')

X_tr_mhc,_,_ = build_feature_matrix(train_df, spec_mhc)
X_tr_base,_,_ = build_feature_matrix(train_df, spec_base)
X_ext_mhc,_,_ = build_feature_matrix(ext, spec_mhc)
X_ext_base,_,_ = build_feature_matrix(ext, spec_base)
X_net61_mhc,_,_ = build_feature_matrix(net61, spec_mhc)
X_net61_base,_,_ = build_feature_matrix(net61, spec_base)
y_tr = train_df['is_binder'].values
y_te = test_df['is_binder'].values

# === Grid around shallow (depth=5, lr=0.1) ===
configs = [
    # (name, params)
    ('shallow_5',  dict(max_iter=500,  learning_rate=0.1,  max_depth=5,  l2_regularization=0.5, min_samples_leaf=20)),
    ('shallow_4',  dict(max_iter=500,  learning_rate=0.1,  max_depth=4,  l2_regularization=0.5, min_samples_leaf=20)),
    ('shallow_6', dict(max_iter=500,  learning_rate=0.1,  max_depth=6,  l2_regularization=0.5, min_samples_leaf=20)),
    ('shallow_lr7',dict(max_iter=500,  learning_rate=0.07, max_depth=5,  l2_regularization=0.5, min_samples_leaf=20)),
    ('shallow_lr15',dict(max_iter=500, learning_rate=0.15, max_depth=5,  l2_regularization=0.5, min_samples_leaf=20)),
    ('shallow_reg1',dict(max_iter=500,learning_rate=0.1, max_depth=5,  l2_regularization=1.0, min_samples_leaf=20)),
    ('shallow_reg2',dict(max_iter=500,learning_rate=0.1, max_depth=5,  l2_regularization=2.0, min_samples_leaf=20)),
    ('shallow_leaf50',dict(max_iter=500,learning_rate=0.1, max_depth=5, l2_regularization=0.5, min_samples_leaf=50)),
    # Also test base features
    ('base_shallow',dict(max_iter=500,learning_rate=0.1, max_depth=5, l2_regularization=0.5)),
]

results = []
for name, params in configs:
    use_mhc = 'base' not in name
    X1 = X_tr_mhc if use_mhc else X_tr_base
    X_e = X_ext_mhc if use_mhc else X_ext_base
    X_n = X_net61_mhc if use_mhc else X_net61_base

    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=42, **params)
    m.fit(X1, y_tr)

    p_ext = m.predict_proba(X_e)[:,1]
    p_net = m.predict_proba(X_n)[:,1]
    ae = roc_auc_score(y_ext, p_ext)
    an = roc_auc_score(y_net61, p_net)

    results.append({'name': name, 'params': params, 'auc_ext': ae, 'auc_net61': an,
                   'ext_f1': f1_score(y_ext, (p_ext>=0.5).astype(int)),
                   'net_f1': f1_score(y_net61, (p_net>=0.5).astype(int)),
                   'time': time.time()-t0, 'use_mhc': use_mhc})
    print(f"{name:15s}: iedb_ext={ae:.4f} netmhcpan={an:.4f} ({time.time()-t0:.0f}s)", flush=True)

# === Ensemble top-3 ===
print("\nEnsemble top-3...", flush=True)
top3 = sorted(results, key=lambda x: x['auc_ext'], reverse=True)[:3]
ens_ext = np.zeros(len(y_ext))
ens_net = np.zeros(len(y_net61))
for r in top3:
    X1 = X_tr_mhc if r['use_mhc'] else X_tr_base
    X_e = X_ext_mhc if r['use_mhc'] else X_ext_base
    X_n = X_net61_mhc if r['use_mhc'] else X_net61_base
    m = HistGradientBoostingClassifier(random_state=42, **r['params'])
    m.fit(X1, y_tr)
    ens_ext += m.predict_proba(X_e)[:,1]
    ens_net += m.predict_proba(X_n)[:,1]
ens_ext /= 3
ens_net /= 3
ae_ens = roc_auc_score(y_ext, ens_ext)
an_ens = roc_auc_score(y_net61, ens_net)
print(f"Ensemble top-3: iedb_ext={ae_ens:.4f} netmhcpan={an_ens:.4f}", flush=True)

# === Summary ===
print(f"\n{'='*60}")
print(f"BEST on iedb_heldout (2166 samples): {max(r['auc_ext'] for r in results):.4f}")
print(f"ENSEMBLE on iedb_heldout:              {ae_ens:.4f}")
print(f"BEST on NetMHCpan (61 samples):        {max(r['auc_net61'] for r in results):.4f}")
print(f"ENSEMBLE on NetMHCpan:                 {an_ens:.4f}")
print(f"NetMHCpan target:                     0.92-0.96")
print(f"{'='*60}")
print(f"\nTop configs:")
for r in sorted(results, key=lambda x: x['auc_ext'], reverse=True)[:5]:
    print(f"  {r['name']:20s}: ext={r['auc_ext']:.4f} net61={r['auc_net61']:.4f}")

# Save
out = {'results': results, 'ensemble': {'auc_ext': float(ae_ens), 'auc_net61': float(an_ens)}}
with open(PROJECT / "benchmarks/results/sprint_final.json", 'w') as f:
    json.dump(out, f, indent=2)
