"""
Confluencia Experiment Re-run (Fast)
=====================================
Uses pre-trained epitope_model_288k.joblib directly — NO retraining.
Only extracts features for test + external validation datasets.

Reproduces Tables 2, 4-6, 10-11 from the paper.
"""
from __future__ import annotations
import sys, time, json, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, average_precision_score,
    mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "benchmarks" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
BDATA = PROJECT / "benchmarks" / "data"
FIGS = PROJECT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT / "data" / "cache" / "epitope_model_288k.joblib"
FULL_DATA = PROJECT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"

sys.path.insert(0, str(PROJECT / "confluencia-2.0-epitope"))
sys.path.insert(0, str(PROJECT))

from core.features import build_feature_matrix, ensure_columns


def feats(df):
    """Extract 317-d features from a dataframe."""
    work = ensure_columns(df)
    X, names, env = build_feature_matrix(work)
    return X.astype(np.float32), names, env


def clf_metrics(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y, proba))
        auprc = float(average_precision_score(y, proba))
    except ValueError:
        auc, auprc = float("nan"), float("nan")
    return dict(
        auc=auc, auprc=auprc,
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
    )


def pad_env(df, defaults=None):
    """Add missing env columns with defaults."""
    if defaults is None:
        defaults = {"dose": 1.0, "freq": 1.0, "treatment_time": 24.0,
                     "circ_expr": 1.0, "ifn_score": 0.5}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


# ==============================================================
# MAIN
# ==============================================================

def main():
    t_total = time.time()
    print("=" * 60)
    print("Confluencia Experiment Re-run (using pre-trained 288k model)")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 0. Load model
    # ----------------------------------------------------------
    print("\n[0] Loading pre-trained model...")
    bundle = joblib.load(MODEL_PATH)
    rf_model = bundle["model"]
    scaler = bundle["scaler"]
    feat_names = bundle["feature_names"]
    print(f"  Model: {type(rf_model).__name__}, "
          f"estimators={rf_model.n_estimators}, depth={rf_model.max_depth}")
    print(f"  Features: {len(feat_names)}, trained on {bundle['config']['n_train']} samples")

    all_out = {}

    # ----------------------------------------------------------
    # 1. Table 10 — re-evaluate on 288k test split
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[1] Table 10: Re-evaluate on 288k test split")
    print("=" * 60)

    df_full = pd.read_csv(FULL_DATA)
    df_full["label"] = (df_full["efficacy"] >= 3.0).astype(int)

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(df_full, groups=df_full["epitope_seq"].values))
    test_df = df_full.iloc[test_idx].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} samples, binder rate: {test_df['label'].mean():.1%}")

    print("  Extracting test features...")
    t0 = time.time()
    X_test, _, _ = feats(test_df)
    print(f"  Done: {X_test.shape} ({time.time()-t0:.1f}s)")

    y_test = test_df["label"].values
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    t10 = clf_metrics(y_test, y_proba)
    print(f"  RF Model → AUC={t10['auc']:.4f}  F1={t10['f1']:.4f}  "
          f"MCC={t10['mcc']:.4f}  Acc={t10['accuracy']:.4f}")
    print(f"  (Paper reports: HGB AUC=0.731, RF AUC=0.725)")

    all_out["table10_288k_test"] = {
        "model": "RandomForestClassifier (pre-trained)",
        "metrics": t10,
        "n_test": len(test_df),
    }

    # ----------------------------------------------------------
    # 2. Table 6 — External validation
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2] Table 6: External Validation")
    print("=" * 60)

    ext_results = {}

    # 2a. IEDB held-out
    p = BDATA / "iedb_heldout_mhc.csv"
    if p.exists():
        print("\n  [IEDB held-out]")
        df = pd.read_csv(p)
        df = pad_env(df)
        X, _, _ = feats(df)
        y = df["is_binder"].astype(int).values if "is_binder" in df.columns else (df["efficacy_true"] >= 3.0).astype(int).values
        proba = rf_model.predict_proba(X)[:, 1]
        m = clf_metrics(y, proba)
        if "efficacy_true" in df.columns:
            r, pval = stats.pearsonr(proba, df["efficacy_true"].values)
            m["pearson_r"] = float(r)
            m["pearson_p"] = float(pval)
        ext_results["iedb_heldout"] = {"n": len(df), "metrics": m}
        print(f"    N={len(df)}, AUC={m['auc']:.4f}, r={m.get('pearson_r', 'N/A')}")

    # 2b. NetMHCpan
    p = BDATA / "netmhcpan_heldout.csv"
    if p.exists():
        print("\n  [NetMHCpan benchmark]")
        df = pd.read_csv(p)
        df = pad_env(df)
        X, _, _ = feats(df)
        y = df["is_binder"].astype(int).values if "is_binder" in df.columns else (df["efficacy_true"] >= 3.0).astype(int).values
        proba = rf_model.predict_proba(X)[:, 1]
        m = clf_metrics(y, proba)
        if "ic50_nm" in df.columns:
            log_ic50 = np.log10(np.maximum(df["ic50_nm"].values, 1.0))
            r, pval = stats.pearsonr(proba, log_ic50)
            m["pearson_r_log_ic50"] = float(r)
            m["pearson_p_log_ic50"] = float(pval)
        ext_results["netmhcpan"] = {"n": len(df), "metrics": m}
        print(f"    N={len(df)}, AUC={m['auc']:.4f}")

    # 2c. TCCIA
    p = BDATA / "tccia_validation.csv"
    if p.exists():
        print("\n  [TCCIA circRNA]")
        df = pd.read_csv(p)
        if "ifn_signature" in df.columns and "response" in df.columns:
            r, pval = stats.pearsonr(df["ifn_signature"].values, df["response"].values)
            ext_results["tccia"] = {"n": len(df), "pearson_r": float(r), "p": float(pval)}
            print(f"    N={len(df)}, r={r:.4f} (p={pval:.2e})")

    # 2d. GDSC
    p = BDATA / "gdsc_validation.csv"
    if p.exists():
        print("\n  [GDSC drug sensitivity]")
        df = pd.read_csv(p)
        if "ln_ic50" in df.columns and "is_sensitive" in df.columns:
            r, pval = stats.pearsonr(-df["ln_ic50"].values, df["is_sensitive"].astype(int).values)
            ext_results["gdsc"] = {"n": len(df), "pearson_r": float(r), "p": float(pval)}
            print(f"    N={len(df)}, r={r:.4f} (p={pval:.2e})")

    # 2e. Literature cases
    p = BDATA / "literature_cases.csv"
    if p.exists():
        print("\n  [Literature cases]")
        df = pd.read_csv(p)
        df = pad_env(df)
        X, _, _ = feats(df)
        proba = rf_model.predict_proba(X)[:, 1]
        if "reported_ifn_response" in df.columns:
            rep = df["reported_ifn_response"].values
            r, pval = stats.pearsonr(proba, rep)
            pred_hi = proba > np.median(proba)
            act_hi = rep > np.median(rep)
            dir_acc = float(np.mean(pred_hi == act_hi))
            ext_results["literature"] = {"n": len(df), "pearson_r": float(r),
                                          "direction_accuracy": dir_acc}
            print(f"    N={len(df)}, r={r:.4f}, direction_acc={dir_acc:.2%}")

    all_out["table6_external_validation"] = ext_results

    # ----------------------------------------------------------
    # 3. Table 11 — VAE denoise comparison (uses saved results)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3] Table 11: VAE Denoise Impact (from saved results)")
    print("=" * 60)

    vae_path = RESULTS / "vae_denoise_288k.json"
    if vae_path.exists():
        with open(vae_path) as f:
            vae_data = json.load(f)

        print(f"\n  {'Method':8s} {'Raw AUC':>10s} {'Denoised AUC':>14s} {'Δ AUC':>10s}")
        print("  " + "-" * 48)
        for method in ["HGB", "RF", "LR", "MLP"]:
            raw_auc = vae_data.get("baseline", {}).get(method, {}).get("auc", 0)
            den_auc = vae_data.get("vae_denoise", {}).get(method, {}).get("auc", 0)
            delta = den_auc - raw_auc if raw_auc and den_auc else 0
            print(f"  {method:8s} {raw_auc:10.4f} {den_auc:14.4f} {delta:+10.4f}")

        all_out["table11_vae_denoise"] = vae_data
    else:
        print("  Saved VAE results not found, skipping.")

    # ----------------------------------------------------------
    # 4. Table 10 — compare with saved baselines_288k_binary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[4] Table 10: Compare all methods (from saved results)")
    print("=" * 60)

    bl_path = RESULTS / "baselines_288k_binary.json"
    if bl_path.exists():
        with open(bl_path) as f:
            bl_data = json.load(f)

        print(f"\n  {'Method':20s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s} {'MCC':>8s}")
        print("  " + "-" * 56)
        for name, m in bl_data.items():
            print(f"  {name:20s} {m.get('auc',0):8.4f} {m.get('accuracy',0):8.4f} "
                  f"{m.get('f1',0):8.4f} {m.get('mcc',0):8.4f}")
        # Add our pre-trained RF model result
        print(f"  {'RF (pretrained)':20s} {t10['auc']:8.4f} {t10['accuracy']:8.4f} "
              f"{t10['f1']:8.4f} {t10['mcc']:8.4f}")

        all_out["table10_all_methods"] = bl_data
        all_out["table10_all_methods"]["RF_pretrained"] = t10

    # ----------------------------------------------------------
    # 5. Table 2/4/5 — load saved small-sample results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5] Tables 2, 4, 5: Small-sample results (from saved JSON)")
    print("=" * 60)

    for tbl, fname in [("table2_baselines", "baselines_epitope.json"),
                        ("table4_ablation", "ablation_epitope.json"),
                        ("table5_sensitivity", "sample_sensitivity_epitope.json")]:
        p = RESULTS / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            all_out[tbl] = data
            print(f"  {tbl}: loaded from {fname}")
        else:
            print(f"  {tbl}: {fname} not found")

    # ----------------------------------------------------------
    # 6. Feature Importance
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[6] Feature Importance Analysis")
    print("=" * 60)

    imp = rf_model.feature_importances_
    top_idx = imp.argsort()[::-1][:20]
    fi = {}
    print(f"\n  {'Rank':>4s}  {'Feature':30s}  {'Importance':>10s}")
    print("  " + "-" * 50)
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i]
        val = float(imp[i])
        fi[name] = val
        print(f"  {rank:4d}  {name:30s}  {val:10.4f}")

    all_out["feature_importance_top20"] = fi

    # Group-level importance
    groups = {"mamba_summary": [], "mamba_pool": [], "kmer": [],
              "biochem": [], "env": []}
    for i, name in enumerate(feat_names):
        if name.startswith("mamba_summary"):
            groups["mamba_summary"].append(imp[i])
        elif name.startswith("mamba_pool") or name.startswith("mamba_local") or name.startswith("mamba_meso") or name.startswith("mamba_global"):
            groups["mamba_pool"].append(imp[i])
        elif name.startswith("kmer"):
            groups["kmer"].append(imp[i])
        elif name.startswith("bio"):
            groups["biochem"].append(imp[i])
        elif name.startswith("env"):
            groups["env"].append(imp[i])

    print(f"\n  {'Feature Group':20s} {'Total Importance':>18s} {'% of Total':>12s}")
    print("  " + "-" * 54)
    total_imp = float(imp.sum())
    grp_imp = {}
    for grp, vals in groups.items():
        s = sum(vals)
        pct = s / total_imp * 100
        grp_imp[grp] = {"total": float(s), "pct": float(pct), "n_features": len(vals)}
        print(f"  {grp:20s} {s:18.4f} {pct:11.1f}%")
    all_out["feature_importance_groups"] = grp_imp

    # ----------------------------------------------------------
    # 7. Generate Figures
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[7] Generating Figures")
    print("=" * 60)
    try:
        import subprocess
        fig_script = PROJECT / "scripts" / "generate_figures.py"
        if fig_script.exists():
            r = subprocess.run([sys.executable, str(fig_script)],
                               cwd=str(PROJECT), capture_output=True, text=True,
                               timeout=120)
            print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
            if r.returncode != 0:
                print(f"  Warnings: {r.stderr[-300:]}")
        else:
            print(f"  Script not found: {fig_script}")
    except Exception as e:
        print(f"  Error: {e}")

    # ----------------------------------------------------------
    # Save all results
    # ----------------------------------------------------------
    out_path = RESULTS / "rerun_all_experiments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
