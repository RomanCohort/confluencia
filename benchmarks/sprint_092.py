"""冲刺 0.92: 全量 binding 数据 + MHC + 调参"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "confluencia-2.0-epitope"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from core.features import FeatureSpec, build_feature_matrix

PROJECT = Path(__file__).resolve().parents[1]

bind = pd.read_csv(PROJECT / "confluencia-2.0-epitope/data/iedb_mhc_i_binding.csv")
train_df, test_df = train_test_split(bind, test_size=0.2, random_state=42, stratify=bind['is_binder'])

ext_test = pd.read_csv(PROJECT / "benchmarks/data/iedb_heldout_mhc.csv")
for col in ['dose','freq','treatment_time','circ_expr','ifn_score']:
    if col not in ext_test.columns: ext_test[col] = 0.0
y_ext = ext_test['is_binder'].astype(int).values

spec = FeatureSpec(use_mhc=True, use_esm2=False, mhc_allele_col='mhc_allele')

X_tr, _, _ = build_feature_matrix(train_df, spec)
X_te, _, _ = build_feature_matrix(test_df, spec)
X_ext, _, _ = build_feature_matrix(ext_test, spec)
y_tr = train_df['is_binder'].values
y_te = test_df['is_binder'].values

configs = [
    ('default',  dict(max_iter=500, learning_rate=0.05, max_depth=8,  l2_regularization=1.0)),
    ('deep',     dict(max_iter=1000,learning_rate=0.02, max_depth=10, l2_regularization=2.0)),
    ('wide',     dict(max_iter=800, learning_rate=0.03, max_depth=12, l2_regularization=1.5)),
    ('shallow',  dict(max_iter=500, learning_rate=0.1,  max_depth=5,  l2_regularization=0.5)),
]

models = {}
for name, params in configs:
    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=42, **params)
    m.fit(X_tr, y_tr)
    ai = roc_auc_score(y_te, m.predict_proba(X_te)[:,1])
    ae = roc_auc_score(y_ext, m.predict_proba(X_ext)[:,1])
    models[name] = (m, ae)
    sys.stdout.write(f"{name:10s}: internal={ai:.4f} external={ae:.4f} ({time.time()-t0:.0f}s)\n")
    sys.stdout.flush()

# Ensemble
top2 = sorted(models.items(), key=lambda x: x[1][1], reverse=True)[:2]
ens = np.mean([m.predict_proba(X_ext)[:,1] for m,_ in top2], axis=0)
ae_ens = roc_auc_score(y_ext, ens)
sys.stdout.write(f"\nEnsemble: {ae_ens:.4f}\nBest single: {max(v[1] for v in models.values()):.4f}\nTarget: 0.92\n")
sys.stdout.flush()
