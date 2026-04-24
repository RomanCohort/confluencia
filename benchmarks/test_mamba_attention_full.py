"""
Mamba3Lite 注意力增强增强测试 - 完整版
运行多种 d_model 配置和注意力权重对比
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "confluencia-2.0-epitope"))

from core.mamba3 import Mamba3LiteEncoder, Mamba3Config
from core.features import build_feature_matrix, FeatureSpec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
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

# Configurations to test
configs = [
    # 注意力增强配置
    ("Mamba3Lite+Attn (d=64)", Mamba3Config(d_model=64, seed=42)),
    ("Mamba3Lite+Attn (d=32)", Mamba3Config(d_model=32, seed=42)),
    ("Mamba3Lite+Attn (d=48)", Mamba3Config(d_model=48, seed=42)),
    # 无注意力基线
    ("Mamba3Lite-NoAttn (d=24)", Mamba3Config(d_model=24, seed=42)),
]

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, config in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Config: d_model={config.d_model}")

    encoder = Mamba3LiteEncoder(config)
    X, _, _ = build_feature_matrix(df, FeatureSpec())
    print(f"Features: {X.shape[1]}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    for mode, desc in [("ridge", "Ridge"), ("hgb", "HGB")]:
        mae_vals, r2_vals, pr_vals = [], [], []
        for tr, va in kf.split(Xs):
            if mode == "ridge":
                from sklearn.linear_model import Ridge
                m = Ridge(alpha=1.0)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
            m.fit(Xs[tr], y[tr])
            p = m.predict(Xs[va])
            mae_vals.append(mean_absolute_error(y[va], p))
            ss_res = np.sum((y[va] - p) ** 2)
            ss_tot = np.sum((y[va] - y[va].mean()) ** 2)
            r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            pr_vals.append(stats.pearsonr(p, y[va])[0])

        key = f"{name} + {desc}"
        results[key] = {
            "mae": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "r2": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
            "pearson_r": float(np.mean(pr_vals)),
            "n_features": int(X.shape[1]),
            "config": {"d_model": config.d_model, "seed": config.seed}
        }
        print(f"  {desc}: MAE={np.mean(mae_vals):.4f}+/-{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}+/-{np.std(r2_vals):.4f}")

# Save results
out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "mamba_attention_test.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {out_path}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Mamba3Lite Attention Enhancement Results")
print("="*80)
print(f"{'Configuration':<30} {'MAE':>10} {'R2':>10} {'r':>8}")
print("-"*60)
for key, val in results.items():
    print(f"{key:<30} {val['mae']:.4f}     {val['r2']:.4f}     {val['pearson_r']:.4f}")
