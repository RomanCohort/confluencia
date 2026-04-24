"""
Mamba3Lite 注意力消融实验
对比 WITH attention vs WITHOUT attention 在相同 d_model 下的表现
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "confluencia-2.0-epitope"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy import stats

from core.mamba3 import (
    Mamba3Config, Mamba3LiteEncoder,
    _sigmoid, _softmax, _scaled_dot_product_attention, _rolling_mean
)
from confluencia_shared.features import AA_ORDER, AA_TO_IDX
from core.features import _hash_kmer, _biochem_stats

# Load data
csv_path = Path(__file__).resolve().parents[1] / "data" / "example_epitope.csv"
df = pd.read_csv(csv_path)
for raw, internal in [("sequence","epitope_seq"),("concentration","dose"),("cell_density","circ_expr"),("incubation_hours","treatment_time")]:
    if raw in df.columns and internal not in df.columns:
        df[internal] = df[raw]
if "freq" not in df.columns: df["freq"] = 1.0
if "ifn_score" not in df.columns: df["ifn_score"] = 0.5
y = df["efficacy"].values.astype(np.float32)


class Mamba3LiteNoAttn(Mamba3LiteEncoder):
    """Mamba3Lite WITHOUT attention - pure SSM + pooling."""

    def encode(self, seq: str) -> dict:
        ids = self._tokenize(seq)
        d = int(self.config.d_model)
        if ids.size == 0:
            z = np.zeros((d,), dtype=np.float32)
            return {"summary": np.concatenate([z, z, z, z], axis=0), "local_pool": z, "meso_pool": z, "global_pool": z}

        x = self.embedding[ids]
        gates = _sigmoid(x @ self.gate_w + self.gate_b)

        s_fast = np.zeros((d,), dtype=np.float32)
        s_mid = np.zeros((d,), dtype=np.float32)
        s_slow = np.zeros((d,), dtype=np.float32)
        hidden = np.zeros_like(x, dtype=np.float32)

        for i in range(x.shape[0]):
            xi = x[i]
            g = gates[i]
            a_fast = self.config.decay_fast + self.config.gate_scale_fast * float(g[0])
            a_mid = self.config.decay_mid + self.config.gate_scale_mid * float(g[1])
            a_slow = self.config.decay_slow + self.config.gate_scale_slow * float(g[2])
            s_fast = a_fast * s_fast + (1.0 - a_fast) * xi
            s_mid = a_mid * s_mid + (1.0 - a_mid) * xi
            s_slow = a_slow * s_slow + (1.0 - a_slow) * xi
            hidden[i] = 0.5 * s_fast + 0.3 * s_mid + 0.2 * s_slow

        # NO attention - just use SSM hidden states directly
        local_hidden = _rolling_mean(hidden, self.config.local_window)
        meso_hidden = _rolling_mean(hidden, self.config.meso_window)
        global_hidden = _rolling_mean(hidden, self.config.global_window)

        local_pool = local_hidden.mean(axis=0).astype(np.float32)
        meso_pool = meso_hidden.mean(axis=0).astype(np.float32)
        global_pool = global_hidden.mean(axis=0).astype(np.float32)

        summary = np.concatenate([
            hidden.mean(axis=0), hidden.max(axis=0), hidden[-1],
            0.5 * local_pool + 0.3 * meso_pool + 0.2 * global_pool,
        ], axis=0).astype(np.float32)

        return {"summary": summary, "local_pool": local_pool, "meso_pool": meso_pool, "global_pool": global_pool}


def build_features(df, encoder):
    seqs = df["epitope_seq"].astype(str).tolist()
    n = len(seqs)

    # Build Mamba features dynamically (different d_model = different sizes)
    mamba_parts = []
    for seq in seqs:
        feat = encoder.encode(seq)
        mamba_parts.append(np.concatenate([feat["summary"], feat["local_pool"], feat["meso_pool"], feat["global_pool"]]))
    mamba_feats = np.stack(mamba_parts, axis=0).astype(np.float32)

    kmer_feats = np.stack([np.concatenate([_hash_kmer(seq, k=2, dim=64), _hash_kmer(seq, k=3, dim=64)]) for seq in seqs], axis=0).astype(np.float32)
    bio_feats = np.stack([_biochem_stats(seq) for seq in seqs], axis=0).astype(np.float32)
    env_cols = ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
    env_feats = df[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    return np.concatenate([mamba_feats, kmer_feats, bio_feats, env_feats], axis=1)


def eval_hgb(X, y, kf):
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
    return {
        "mae": float(np.mean(mae_vals)), "mae_std": float(np.std(mae_vals)),
        "r2": float(np.mean(r2_vals)), "r2_std": float(np.std(r2_vals)),
        "pearson_r": float(np.mean(pr_vals)),
    }


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Test for each d_model: WITH attention vs WITHOUT attention
for d in [16, 24, 32, 48, 64]:
    print(f"\n{'='*60}")
    print(f"d_model = {d}")

    # WITH attention
    enc_attn = Mamba3LiteEncoder(Mamba3Config(d_model=d, seed=42))
    X_attn = build_features(df, enc_attn)
    r_attn = eval_hgb(X_attn, y, kf)
    key_attn = f"SSM+Attn(d={d})"
    results[key_attn] = {**r_attn, "features": int(X_attn.shape[1]), "mamba_dims": d * 4}
    print(f"  {key_attn}: MAE={r_attn['mae']:.4f}+/-{r_attn['mae_std']:.4f}, R2={r_attn['r2']:.4f}")

    # WITHOUT attention
    enc_noattn = Mamba3LiteNoAttn(Mamba3Config(d_model=d, seed=42))
    X_noattn = build_features(df, enc_noattn)
    r_noattn = eval_hgb(X_noattn, y, kf)
    key_noattn = f"SSM-only(d={d})"
    results[key_noattn] = {**r_noattn, "features": int(X_noattn.shape[1]), "mamba_dims": d * 4}
    print(f"  {key_noattn}: MAE={r_noattn['mae']:.4f}+/-{r_noattn['mae_std']:.4f}, R2={r_noattn['r2']:.4f}")

    # Delta
    delta_mae = r_attn["mae"] - r_noattn["mae"]
    delta_r2 = r_attn["r2"] - r_noattn["r2"]
    print(f"  Attention effect: dMAE={delta_mae:+.4f}, dR2={delta_r2:+.4f}")

# Save
out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "mamba_attention_ablation.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {out_path}")

# Final table
print("\n" + "="*80)
print("ATTENTION ABLATION RESULTS (HGB, 5-fold CV)")
print("="*80)
print(f"{'Config':<20} {'MAE':>8} {'R2':>8} {'r':>8} {'Feats':>8}")
print("-"*60)
for key, val in sorted(results.items(), key=lambda x: x[1]["mae"]):
    print(f"{key:<20} {val['mae']:.4f}   {val['r2']:.4f}   {val['pearson_r']:.4f}   {val['features']:>6}")
