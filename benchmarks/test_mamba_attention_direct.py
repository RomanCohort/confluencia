"""
Mamba3Lite 注意力增强直接对比测试
直接使用 Mamba3LiteEncoder.encode() 而非 build_feature_matrix()
绕过缓存确保不同配置产生不同特征
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

# Normalize columns
def _clean(seq):
    return str(seq or "").strip().upper().replace(" ", "")

# Configurations: test with and without attention by modifying the encode method
# We test two variants:
# 1. d_model=64 (larger, with attention in encode)
# 2. d_model=24 (default, base Mamba3Lite)
configs = [
    ("Mamba3Lite+Attn(d=64)", Mamba3Config(d_model=64, seed=42)),
    ("Mamba3Lite+Attn(d=32)", Mamba3Config(d_model=32, seed=42)),
    ("Mamba3Lite(d=24)", Mamba3Config(d_model=24, seed=42)),
]


def build_direct_features(df, config):
    """Directly encode each sequence to get Mamba3Lite features."""
    encoder = Mamba3LiteEncoder(config)
    mamba_feats = []
    for _, row in df.iterrows():
        seq = str(row.get("epitope_seq", ""))
        feat = encoder.encode(seq)
        # Use summary + pools (like build_feature_matrix)
        parts = [
            feat["summary"],
            feat["local_pool"],
            feat["meso_pool"],
            feat["global_pool"],
        ]
        mamba_feats.append(np.concatenate(parts))
    return np.stack(mamba_feats, axis=0)


def build_kmer_features(df, dim=64):
    """Build k-mer hash features."""
    from core.features import _hash_kmer, _clean_seq as clean
    feats = []
    for _, row in df.iterrows():
        seq = str(row.get("epitope_seq", ""))
        k2 = _hash_kmer(seq, k=2, dim=dim)
        k3 = _hash_kmer(seq, k=3, dim=dim)
        feats.append(np.concatenate([k2, k3]))
    return np.stack(feats, axis=0)


def build_biochem_features(df):
    """Build biochemical stats."""
    from core.features import _biochem_stats, _clean_seq as clean
    feats = []
    for _, row in df.iterrows():
        seq = str(row.get("epitope_seq", ""))
        feats.append(_biochem_stats(seq))
    return np.stack(feats, axis=0)


def build_env_features(df):
    """Build environment features."""
    env_cols = ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
    env_feats = []
    for _, row in df.iterrows():
        env = np.array([
            float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
            for c in env_cols if c in df.columns
        ], dtype=np.float32)
        env_feats.append(env)
    return np.stack(env_feats, axis=0)


results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y = df["efficacy"].values.astype(np.float32)

for name, config in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    t0 = time.time()

    # Build features
    mamba_feats = build_direct_features(df, config)
    kmer_feats = build_kmer_features(df, dim=64)
    bio_feats = build_biochem_features(df)
    env_feats = build_env_features(df)

    X = np.concatenate([mamba_feats, kmer_feats, bio_feats, env_feats], axis=1).astype(np.float32)
    print(f"  Total features: {X.shape[1]} (Mamba={mamba_feats.shape[1]}, kmer={kmer_feats.shape[1]}, bio={bio_feats.shape[1]}, env={env_feats.shape[1]})")

    # Test with HGB (best performer)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    mae_vals, r2_vals, pr_vals = [], [], []
    for tr, va in kf.split(Xs):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
        m.fit(Xs[tr], y[tr])
        p = m.predict(Xs[va])
        mae_vals.append(mean_absolute_error(y[va], p))
        ss_res = np.sum((y[va] - p) ** 2)
        ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
        r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        pr_vals.append(stats.pearsonr(p, y[va])[0])

    elapsed = time.time() - t0
    results[name] = {
        "mae": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "r2": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "pearson_r": float(np.mean(pr_vals)),
        "n_mamba_feats": int(mamba_feats.shape[1]),
        "total_features": int(X.shape[1]),
        "elapsed_s": round(elapsed, 2),
    }
    print(f"  HGB: MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}+/-{np.std(r2_vals):.4f}, r={np.mean(pr_vals):.4f}")

# Also test with Ridge
print("\n" + "="*60)
print("Ridge regression results:")
base_config = Mamba3Config(d_model=24, seed=42)
encoder_base = Mamba3LiteEncoder(base_config)
mamba_feats = build_direct_features(df, base_config)
kmer_feats = build_kmer_features(df, dim=64)
bio_feats = build_biochem_features(df)
env_feats = build_env_features(df)
X = np.concatenate([mamba_feats, kmer_feats, bio_feats, env_feats], axis=1).astype(np.float32)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

mae_vals, r2_vals, pr_vals = [], [], []
for tr, va in kf.split(Xs):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0)
    m.fit(Xs[tr], y[tr])
    p = m.predict(Xs[va])
    mae_vals.append(mean_absolute_error(y[va], p))
    ss_res = np.sum((y[va] - p) ** 2)
    ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
    r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
    pr_vals.append(stats.pearsonr(p, y[va])[0])

results["Ridge-baseline"] = {
    "mae": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "r2": float(np.mean(r2_vals)),
    "r2_std": float(np.std(r2_vals)),
    "pearson_r": float(np.mean(pr_vals)),
    "total_features": int(X.shape[1]),
}
print(f"  Ridge: MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}+/-{np.std(r2_vals):.4f}")

# Save results
out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "mamba_attention_enhanced.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {out_path}")

# Print comparison table
print("\n" + "="*80)
print("SUMMARY: Mamba3Lite Attention Enhancement Comparison")
print("="*80)
print(f"{'Configuration':<25} {'MAE':>8} {'R2':>8} {'r':>8} {'Features':>10}")
print("-"*70)
for key, val in sorted(results.items(), key=lambda x: x[1]["mae"]):
    print(f"{key:<25} {val['mae']:.4f}   {val['r2']:.4f}   {val['pearson_r']:.4f}   {val.get('total_features', '?')}")