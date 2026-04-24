"""
Mamba3Lite 注意力权重调优测试
测试不同 attention 残差权重对性能的影响
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "confluencia-2.0-epitope"))

from core.mamba3 import Mamba3LiteEncoder, Mamba3Config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import pandas as pd

# Load data
csv_path = Path(__file__).resolve().parents[1] / "data" / "example_epitope.csv"
df = pd.read_csv(csv_path)
for raw, internal in [("sequence","epitope_seq"),("concentration","dose"),("cell_density","circ_expr"),("incubation_hours","treatment_time")]:
    if raw in df.columns and internal not in df.columns:
        df[internal] = df[raw]
if "freq" not in df.columns: df["freq"] = 1.0
if "ifn_score" not in df.columns: df["ifn_score"] = 0.5

y = df["efficacy"].values.astype(np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

from core.features import _hash_kmer, _biochem_stats


def build_all_features(df, config):
    """Build complete feature matrix for given Mamba3Lite config."""
    encoder = Mamba3LiteEncoder(config)

    seqs = df["epitope_seq"].astype(str).tolist()
    n = len(seqs)

    # Mamba features
    mamba_feats = np.zeros((n, config.d_model * 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        feat = encoder.encode(seq)
        mamba_feats[i] = np.concatenate([feat["summary"], feat["local_pool"], feat["meso_pool"], feat["global_pool"]])

    # k-mer features
    kmer_feats = np.zeros((n, 128), dtype=np.float32)
    for i, seq in enumerate(seqs):
        kmer_feats[i, :64] = _hash_kmer(seq, k=2, dim=64)
        kmer_feats[i, 64:] = _hash_kmer(seq, k=3, dim=64)

    # Biochem
    bio_feats = np.stack([_biochem_stats(seq) for seq in seqs], axis=0)

    # Env
    env_cols = ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
    env_feats = df[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    return np.concatenate([mamba_feats, kmer_feats, bio_feats, env_feats], axis=1)


print("Testing attention residual weight effects...")
print("="*70)

results = {}

# Test 1: Baseline d=24 (no attention improvement expected)
print("\n[Test 1] Mamba3Lite baseline (d=24)")
X = build_all_features(df, Mamba3Config(d_model=24, seed=42))
print(f"  Features: {X.shape[1]}")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

mae_vals, r2_vals = [], []
for tr, va in kf.split(Xs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xs[tr], y[tr])
    p = m.predict(Xs[va])
    mae_vals.append(mean_absolute_error(y[va], p))
    ss_res = np.sum((y[va] - p) ** 2)
    ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
    r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

results["baseline_d24"] = {
    "mae": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "r2": float(np.mean(r2_vals)),
    "r2_std": float(np.std(r2_vals)),
    "config": "d=24, residual=0.1 (default)"
}
print(f"  MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}")

# Test 2: d=32 (intermediate)
print("\n[Test 2] Mamba3Lite d=32")
X = build_all_features(df, Mamba3Config(d_model=32, seed=42))
print(f"  Features: {X.shape[1]}")
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

mae_vals, r2_vals = [], []
for tr, va in kf.split(Xs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xs[tr], y[tr])
    p = m.predict(Xs[va])
    mae_vals.append(mean_absolute_error(y[va], p))
    ss_res = np.sum((y[va] - p) ** 2)
    ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
    r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

results["d32"] = {
    "mae": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "r2": float(np.mean(r2_vals)),
    "r2_std": float(np.std(r2_vals)),
    "config": "d=32, 224 Mamba features + 128 kmer + 16 bio + 5 env"
}
print(f"  MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}")

# Test 3: d=16 (smaller)
print("\n[Test 3] Mamba3Lite d=16")
X = build_all_features(df, Mamba3Config(d_model=16, seed=42))
print(f"  Features: {X.shape[1]}")
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

mae_vals, r2_vals = [], []
for tr, va in kf.split(Xs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xs[tr], y[tr])
    p = m.predict(Xs[va])
    mae_vals.append(mean_absolute_error(y[va], p))
    ss_res = np.sum((y[va] - p) ** 2)
    ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
    r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

results["d16"] = {
    "mae": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "r2": float(np.mean(r2_vals)),
    "r2_std": float(np.std(r2_vals)),
    "config": "d=16, 112 Mamba features + 128 kmer + 16 bio + 5 env"
}
print(f"  MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}")

# Test 4: d=64 (larger)
print("\n[Test 4] Mamba3Lite d=64")
X = build_all_features(df, Mamba3Config(d_model=64, seed=42))
print(f"  Features: {X.shape[1]}")
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

mae_vals, r2_vals = [], []
for tr, va in kf.split(Xs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    m.fit(Xs[tr], y[tr])
    p = m.predict(Xs[va])
    mae_vals.append(mean_absolute_error(y[va], p))
    ss_res = np.sum((y[va] - p) ** 2)
    ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
    r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

results["d64"] = {
    "mae": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "r2": float(np.mean(r2_vals)),
    "r2_std": float(np.std(r2_vals)),
    "config": "d=64, 448 Mamba features + 128 kmer + 16 bio + 5 env"
}
print(f"  MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}")

# Save
out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "mamba_attention_tuning.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Config':<20} {'MAE':>8} {'MAE_std':>8} {'R2':>8} {'R2_std':>8} {'Features':>10}")
print("-"*70)
for key, val in sorted(results.items(), key=lambda x: x[1]["mae"]):
    print(f"{key:<20} {val['mae']:.4f}   {val['mae_std']:.4f}   {val['r2']:.4f}   {val['r2_std']:.4f}   {val.get('config','').split(',')[0]}")
print(f"\nResults saved to {out_path}")