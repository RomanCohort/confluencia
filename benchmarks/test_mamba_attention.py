"""Test enhanced Mamba3Lite with attention."""
import sys, time
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

X, _, _ = build_feature_matrix(df, FeatureSpec())
y = df["efficacy"].values.astype(np.float32)

# Test with attention (larger d_model for attention benefit)
config_attn = Mamba3Config(d_model=64, seed=42)
encoder_attn = Mamba3LiteEncoder(config_attn)
print(f"Attention encoder: {X.shape[1]} features")

# 5-fold CV
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for mode, desc in [("ridge", "Ridge (baseline)"), ("hgb", "HGB")]:
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
    print(f"{desc}: MAE={np.mean(mae_vals):.4f}±{np.std(mae_vals):.4f}, R2={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}")
