"""
Confluencia Clinical Validation Experiments
============================================
Validates Confluencia predictions against held-out public database data.

Experiments:
  A. IEDB MHC-I Cross-Validation - sequence-aware held-out validation
  B. ChEMBL Drug Bioactivity Validation - drug-target binding prediction
  C. NetMHCpan Benchmark Concordance - correlation with known benchmark
  D. Literature Case Study Comparison - qualitative validation

Usage:
    python -m benchmarks.clinical_validation
    python -m benchmarks.clinical_validation --iedb
    python -m benchmarks.clinical_validation --chembl
    python -m benchmarks.clinical_validation --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"

# Full epitope training data (288k rows from IEDB + NetMHCpan + circRNA)
EPITOPE_FULL_TRAIN = PROJECT_ROOT / "confluencia-2.0-epitope" / "data" / "epitope_training_full.csv"
# Small training data (300 rows, used in benchmarks)
EPITOPE_SMALL_TRAIN = PROJECT_ROOT / "data" / "example_epitope.csv"
# Drug training data
DRUG_TRAIN = PROJECT_ROOT / "confluencia-2.0-drug" / "data" / "breast_cancer_drug_dataset.csv"


def _ensure_epitope_path():
    p = str(PROJECT_ROOT / "confluencia-2.0-epitope")
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_drug_path():
    p = str(PROJECT_ROOT / "confluencia-2.0-drug")
    # Insert drug path FIRST to avoid epitope module shadowing
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "n_samples": len(y_true),
    }


def compute_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "note": "only one class"}
    try:
        auc_val = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc_val = float("nan")
    return {
        "auc": auc_val,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, confidence=0.95, seed=42):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(float(statistic(sample)))
    boot_stats = np.array(boot_stats)
    alpha = (1 - confidence) / 2
    return {
        "lower": float(np.percentile(boot_stats, alpha * 100)),
        "upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "point_estimate": float(statistic(data)),
    }


# ---------------------------------------------------------------------------
# Sequence-aware data splitting
# ---------------------------------------------------------------------------

def sequence_aware_split(df, seq_col="epitope_seq", test_ratio=0.2, seed=42):
    """Split data so no sequence appears in both train and test."""
    unique_seqs = df[seq_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seqs)
    n_test = max(1, int(len(unique_seqs) * test_ratio))
    test_seqs = set(unique_seqs[:n_test])
    test_mask = df[seq_col].isin(test_seqs)
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Shared model training helper
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test, y_binary=None):
    """Train Ridge, HGB, RF, MOE and return metrics."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    experts = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "hgb": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42),
        "rf": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=1),
    }

    predictions = {}
    preds = {}
    for name, model in experts.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(np.float32)
        train_time = time.time() - t0
        preds[name] = pred

        reg = compute_regression_metrics(y_test, pred)
        reg["train_time"] = train_time
        predictions[name] = reg

        if y_binary is not None:
            pred_norm = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-8)
            cls = compute_classification_metrics(y_binary, pred_norm, threshold=0.5)
            predictions[name].update({
                "classification_auc": cls.get("auc", float("nan")),
                "classification_accuracy": cls.get("accuracy", float("nan")),
                "classification_f1": cls.get("f1", float("nan")),
            })

        print(f"  {name:10s}: Pearson r={reg['pearson_r']:.4f} "
              f"Spearman r={reg['spearman_r']:.4f} MAE={reg['mae']:.4f} R2={reg['r2']:.4f}")

    # MOE ensemble
    moe_pred = (preds["ridge"] + preds["hgb"] + preds["rf"]) / 3.0
    moe_reg = compute_regression_metrics(y_test, moe_pred)
    predictions["moe"] = moe_reg

    if y_binary is not None:
        moe_norm = (moe_pred - moe_pred.min()) / max(moe_pred.max() - moe_pred.min(), 1e-8)
        moe_cls = compute_classification_metrics(y_binary, moe_norm, threshold=0.5)
        predictions["moe"].update({
            "classification_auc": moe_cls.get("auc", float("nan")),
            "classification_accuracy": moe_cls.get("accuracy", float("nan")),
            "classification_f1": moe_cls.get("f1", float("nan")),
        })

    print(f"  {'moe':10s}: Pearson r={moe_reg['pearson_r']:.4f} "
          f"Spearman r={moe_reg['spearman_r']:.4f} MAE={moe_reg['mae']:.4f} R2={moe_reg['r2']:.4f}")

    return predictions, moe_pred


# ---------------------------------------------------------------------------
# Experiment A: IEDB MHC-I Cross-Validation
# ---------------------------------------------------------------------------

def validate_iedb_mhc() -> Dict[str, Any]:
    """
    Validate epitope predictions using sequence-aware cross-validation
    on the full IEDB/NetMHCpan/circRNA training data (288k rows).
    """
    print("\n" + "=" * 60)
    print("Experiment A: IEDB MHC-I Sequence-Aware Cross-Validation")
    print("=" * 60)

    if not EPITOPE_FULL_TRAIN.exists():
        return {"error": "epitope_training_full.csv not found"}

    df = pd.read_csv(EPITOPE_FULL_TRAIN)
    print(f"  Full training data: {len(df)} rows, {df['epitope_seq'].nunique()} unique sequences")
    if "data_source" in df.columns:
        print(f"  Sources: {df['data_source'].value_counts().to_dict()}")

    # Subsample for efficiency (too many rows otherwise)
    # Keep all unique sequences but limit to ~5000 samples
    rng = np.random.default_rng(42)
    unique_seqs = df["epitope_seq"].unique()
    if len(unique_seqs) > 5000:
        selected_seqs = rng.choice(unique_seqs, size=5000, replace=False)
        df = df[df["epitope_seq"].isin(selected_seqs)].reset_index(drop=True)
        print(f"  Subsampled to {len(df)} rows ({df['epitope_seq'].nunique()} unique sequences)")

    # Sequence-aware split (80/20)
    train_df, test_df = sequence_aware_split(df, "epitope_seq", test_ratio=0.2, seed=42)
    print(f"  Train: {len(train_df)} rows ({train_df['epitope_seq'].nunique()} seqs)")
    print(f"  Test:  {len(test_df)} rows ({test_df['epitope_seq'].nunique()} seqs)")

    # Verify no leakage
    train_seqs = set(train_df["epitope_seq"])
    test_seqs = set(test_df["epitope_seq"])
    overlap = train_seqs & test_seqs
    print(f"  Leakage check: {len(overlap)} overlapping sequences ({'PASS' if len(overlap)==0 else 'FAIL'})")

    # Build features
    _ensure_epitope_path()
    from core.features import FeatureSpec, build_feature_matrix, ensure_columns

    train_work = ensure_columns(train_df)
    X_train, feature_names, env_cols = build_feature_matrix(train_work, FeatureSpec())
    y_train = train_work["efficacy"].to_numpy(dtype=np.float32)

    test_work = ensure_columns(test_df)
    X_test, _, _ = build_feature_matrix(test_work, FeatureSpec())
    y_test = test_work["efficacy"].to_numpy(dtype=np.float32)

    print(f"  Features: {X_train.shape[1]} dimensions")
    print(f"  Train efficacy range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Test efficacy range:  [{y_test.min():.3f}, {y_test.max():.3f}]")

    # Train and evaluate
    predictions, moe_pred = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Bootstrap CI on Pearson r
    pearson_ci = bootstrap_ci(
        y_test,
        statistic=lambda x: stats.pearsonr(x, moe_pred)[0],
        n_bootstrap=1000,
    )

    return {
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_seqs": len(train_seqs),
        "n_test_seqs": len(test_seqs),
        "leakage_check": len(overlap) == 0,
        "models": predictions,
        "pearson_r_ci_95": pearson_ci,
        "efficacy_range_train": [float(y_train.min()), float(y_train.max())],
        "efficacy_range_test": [float(y_test.min()), float(y_test.max())],
        "data_source": "IEDB + NetMHCpan + circRNA literature (sequence-aware split)",
    }


# ---------------------------------------------------------------------------
# Experiment B: ChEMBL Drug Bioactivity Validation
# ---------------------------------------------------------------------------

def validate_chembl_drug() -> Dict[str, Any]:
    """
    Validate drug predictions against held-out ChEMBL bioactivity data.
    Train on breast_cancer_drug_dataset.csv, predict on held-out ChEMBL entries.
    """
    print("\n" + "=" * 60)
    print("Experiment B: ChEMBL Drug Bioactivity Validation")
    print("=" * 60)

    chembl_path = DATA_DIR / "chembl_heldout_bioactivity.csv"
    if not chembl_path.exists():
        return {"error": "chembl_data_not_found"}

    chembl_df = pd.read_csv(chembl_path)
    if len(chembl_df) == 0:
        return {"error": "chembl_data_empty"}

    print(f"  Held-out ChEMBL records: {len(chembl_df)}")
    print(f"  Active compounds (IC50<1000nM): {chembl_df['is_active'].sum()}")
    print(f"  Targets: {chembl_df['target_protein'].value_counts().to_dict()}")

    if not DRUG_TRAIN.exists():
        return {"error": "drug_training_data_not_found"}

    train_df = pd.read_csv(DRUG_TRAIN)
    print(f"  Training samples: {len(train_df)}")

    # Use RDKit directly (avoid module import issues)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        return {"error": "RDKit not available"}

    def compute_drug_features(smiles_list):
        features = []
        n_invalid = 0
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_invalid += 1
                    features.append(np.zeros(2056))
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_arr = np.array(fp, dtype=np.float32)
                desc = [
                    Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol),
                ]
                feat = np.concatenate([fp_arr, np.array(desc, dtype=np.float32)])
                # Check for NaN/Inf
                if not np.isfinite(feat).all():
                    feat = np.zeros(2056, dtype=np.float32)
                features.append(feat)
            except Exception:
                n_invalid += 1
                features.append(np.zeros(2056))
        if n_invalid > 0:
            print(f"    Warning: {n_invalid} invalid SMILES")
        arr = np.array(features, dtype=np.float32)
        # Replace any remaining NaN/Inf with zeros
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    print("  Computing training features...")
    X_train = compute_drug_features(train_df["smiles"].tolist())
    y_train = train_df["target_binding"].to_numpy(dtype=np.float32)

    # Handle NaN in target
    valid_mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    print(f"  After removing invalid: {len(y_train)} training samples")

    print("  Computing ChEMBL test features...")
    X_test = compute_drug_features(chembl_df["smiles"].tolist())
    y_test_binding = chembl_df["target_binding_true"].to_numpy(dtype=np.float32)

    # Handle NaN in test data
    valid_test_mask = np.isfinite(y_test_binding) & np.isfinite(X_test).all(axis=1)
    X_test = X_test[valid_test_mask]
    y_test_binding = y_test_binding[valid_test_mask]
    chembl_valid = chembl_df[valid_test_mask].reset_index(drop=True)
    print(f"  After removing invalid: {len(y_test_binding)} test samples")

    print(f"  Train features: {X_train.shape}, Test features: {X_test.shape}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "hgb": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42),
    }

    predictions = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(np.float32)
        train_time = time.time() - t0

        reg = compute_regression_metrics(y_test_binding, pred)
        reg["train_time"] = train_time
        predictions[name] = reg

        y_true_binary = chembl_valid["is_active"].to_numpy().astype(int)
        pred_norm = np.clip(pred, 0, 1)
        cls = compute_classification_metrics(y_true_binary, pred_norm, threshold=0.5)
        predictions[name].update({
            "classification_auc": cls.get("auc", float("nan")),
            "classification_accuracy": cls.get("accuracy", float("nan")),
            "classification_f1": cls.get("f1", float("nan")),
        })

        print(f"  {name:10s}: Pearson r={reg['pearson_r']:.4f} "
              f"AUC={cls.get('auc', 0):.4f} MAE={reg['mae']:.4f} R2={reg['r2']:.4f}")

    return {
        "n_heldout": len(chembl_df),
        "n_active": int(chembl_df["is_active"].sum()),
        "target_distribution": chembl_df["target_protein"].value_counts().to_dict(),
        "models": predictions,
        "data_source": "ChEMBL (held-out, IC50 measurements)",
    }


# ---------------------------------------------------------------------------
# Experiment C: NetMHCpan Benchmark Concordance
# ---------------------------------------------------------------------------

def validate_netmhcpan_concordance() -> Dict[str, Any]:
    """
    Test whether model trained on full data can correctly rank
    NetMHCpan benchmark peptides by binding affinity.
    """
    print("\n" + "=" * 60)
    print("Experiment C: NetMHCpan Benchmark Concordance")
    print("=" * 60)

    nmp_path = DATA_DIR / "netmhcpan_heldout.csv"
    if not nmp_path.exists():
        return {"error": "netmhcpan_data_not_found"}

    nmp_df = pd.read_csv(nmp_path)
    if len(nmp_df) == 0:
        return {"error": "netmhcpan_data_empty"}

    print(f"  NetMHCpan held-out peptides: {len(nmp_df)}")
    print(f"  Binders: {nmp_df['is_binder'].sum()}, Non-binders: {(~nmp_df['is_binder']).sum()}")

    # Load full training data, exclude NetMHCpan benchmark peptides from training
    if not EPITOPE_FULL_TRAIN.exists():
        return {"error": "epitope_training_full.csv not found"}

    full_df = pd.read_csv(EPITOPE_FULL_TRAIN)
    nmp_seqs = set(nmp_df["epitope_seq"].str.upper())
    train_df = full_df[~full_df["epitope_seq"].str.upper().isin(nmp_seqs)].reset_index(drop=True)

    print(f"  Full data: {len(full_df)} rows")
    print(f"  After excluding benchmark peptides: {len(train_df)} rows")

    # Subsample for efficiency
    rng = np.random.default_rng(42)
    unique_seqs = train_df["epitope_seq"].unique()
    if len(unique_seqs) > 5000:
        selected_seqs = rng.choice(unique_seqs, size=5000, replace=False)
        train_df = train_df[train_df["epitope_seq"].isin(selected_seqs)].reset_index(drop=True)

    # Build features
    _ensure_epitope_path()
    from core.features import FeatureSpec, build_feature_matrix, ensure_columns

    train_work = ensure_columns(train_df)
    X_train, _, _ = build_feature_matrix(train_work, FeatureSpec())
    y_train = train_work["efficacy"].to_numpy(dtype=np.float32)

    # Prepare test data
    nmp_work = nmp_df.copy()
    for col, default in [("dose", 1.0), ("freq", 1.0), ("treatment_time", 24.0),
                          ("circ_expr", 1.0), ("ifn_score", 0.5)]:
        if col not in nmp_work.columns:
            nmp_work[col] = default
    nmp_work = ensure_columns(nmp_work)
    X_test, _, _ = build_feature_matrix(nmp_work, FeatureSpec())
    y_test = nmp_df["efficacy_true"].to_numpy(dtype=np.float32)
    ic50_test = nmp_df["ic50_nm"].to_numpy(dtype=np.float32)

    # Handle dimension mismatch
    if X_train.shape[1] != X_test.shape[1]:
        min_dim = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_dim]
        X_test = X_test[:, :min_dim]

    print(f"  Features: {X_train.shape[1]} dimensions")

    # Train models
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "hgb": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(np.float32)

        reg = compute_regression_metrics(y_test, pred)
        log_ic50 = np.log10(np.maximum(ic50_test, 1.0))
        corr_with_ic50, corr_p = stats.pearsonr(pred, log_ic50)

        y_true_binary = nmp_df["is_binder"].to_numpy().astype(int)
        pred_norm = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-8)
        cls = compute_classification_metrics(y_true_binary, pred_norm, threshold=0.5)

        results[name] = {
            **reg,
            "correlation_with_log_ic50": float(corr_with_ic50),
            "correlation_with_log_ic50_p": float(corr_p),
            "classification_auc": cls.get("auc", float("nan")),
            "classification_accuracy": cls.get("accuracy", float("nan")),
            "classification_f1": cls.get("f1", float("nan")),
        }

        print(f"  {name:10s}: Pearson r={reg['pearson_r']:.4f} "
              f"corr(logIC50)={corr_with_ic50:.4f} AUC={cls.get('auc', 0):.4f}")

    # MOE
    ridge_pred = models["ridge"].predict(X_test)
    hgb_pred = models["hgb"].predict(X_test)
    moe_pred = (ridge_pred + hgb_pred) / 2.0

    moe_reg = compute_regression_metrics(y_test, moe_pred.astype(np.float32))
    log_ic50 = np.log10(np.maximum(ic50_test, 1.0))
    moe_corr, moe_corr_p = stats.pearsonr(moe_pred, log_ic50)
    y_true_binary = nmp_df["is_binder"].to_numpy().astype(int)
    moe_norm = (moe_pred - moe_pred.min()) / max(moe_pred.max() - moe_pred.min(), 1e-8)
    moe_cls = compute_classification_metrics(y_true_binary, moe_norm, threshold=0.5)

    results["moe"] = {
        **moe_reg,
        "correlation_with_log_ic50": float(moe_corr),
        "correlation_with_log_ic50_p": float(moe_corr_p),
        "classification_auc": moe_cls.get("auc", float("nan")),
        "classification_accuracy": moe_cls.get("accuracy", float("nan")),
        "classification_f1": moe_cls.get("f1", float("nan")),
    }

    print(f"  {'moe':10s}: Pearson r={moe_reg['pearson_r']:.4f} "
          f"corr(logIC50)={moe_corr:.4f} AUC={moe_cls.get('auc', 0):.4f}")

    pearson_ci = bootstrap_ci(
        y_test,
        statistic=lambda x: stats.pearsonr(x, moe_pred.astype(np.float32))[0],
        n_bootstrap=1000,
    )

    return {
        "n_heldout": len(nmp_df),
        "n_binders": int(nmp_df["is_binder"].sum()),
        "n_nonbinders": int((~nmp_df["is_binder"]).sum()),
        "models": results,
        "pearson_r_ci_95": pearson_ci,
        "note": "Negative corr(logIC50) indicates correct directionality (higher efficacy = lower IC50)",
        "data_source": "NetMHCpan-4.1 benchmark (external)",
    }


# ---------------------------------------------------------------------------
# Experiment D: Literature Case Study Comparison
# ---------------------------------------------------------------------------

def validate_literature_cases() -> Dict[str, Any]:
    """
    Compare Confluencia predictions against published circRNA vaccine data.
    Uses full training data for better accuracy.
    """
    print("\n" + "=" * 60)
    print("Experiment D: Literature Case Study Comparison")
    print("=" * 60)

    lit_path = DATA_DIR / "literature_cases.csv"
    if not lit_path.exists():
        return {"error": "literature_data_not_found"}

    lit_df = pd.read_csv(lit_path)
    print(f"  Literature cases: {len(lit_df)}")

    # Load full training data
    if EPITOPE_FULL_TRAIN.exists():
        full_df = pd.read_csv(EPITOPE_FULL_TRAIN)
        # Exclude literature test sequences from training
        lit_seqs = set(lit_df["epitope_seq"].str.upper())
        train_df = full_df[~full_df["epitope_seq"].str.upper().isin(lit_seqs)].reset_index(drop=True)
        # Subsample for efficiency
        rng = np.random.default_rng(42)
        unique_seqs = train_df["epitope_seq"].unique()
        if len(unique_seqs) > 5000:
            selected_seqs = rng.choice(unique_seqs, size=5000, replace=False)
            train_df = train_df[train_df["epitope_seq"].isin(selected_seqs)].reset_index(drop=True)
        print(f"  Training on {len(train_df)} rows (excluding literature sequences)")
    else:
        train_df = pd.read_csv(EPITOPE_SMALL_TRAIN)
        col_map = {"sequence": "epitope_seq", "concentration": "dose",
                   "cell_density": "circ_expr", "incubation_hours": "treatment_time"}
        for raw, internal in col_map.items():
            if raw in train_df.columns and internal not in train_df.columns:
                train_df[internal] = train_df[raw]
        print(f"  Training on {len(train_df)} rows (small dataset)")

    _ensure_epitope_path()
    from core.features import FeatureSpec, build_feature_matrix, ensure_columns

    train_work = ensure_columns(train_df)
    X_train, _, _ = build_feature_matrix(train_work, FeatureSpec())
    y_train = train_work["efficacy"].to_numpy(dtype=np.float32)

    lit_work = lit_df.copy()
    for col, default in [("freq", 1.0), ("treatment_time", 24.0),
                          ("circ_expr", 1.0), ("ifn_score", 0.5)]:
        if col not in lit_work.columns:
            lit_work[col] = default
    lit_work = ensure_columns(lit_work)
    X_test, _, _ = build_feature_matrix(lit_work, FeatureSpec())

    if X_train.shape[1] != X_test.shape[1]:
        min_dim = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_dim]
        X_test = X_test[:, :min_dim]

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    hgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    ridge.fit(X_train, y_train)
    hgb.fit(X_train, y_train)

    ridge_pred = ridge.predict(X_test)
    hgb_pred = hgb.predict(X_test)
    moe_pred = (ridge_pred + hgb_pred) / 2.0

    category_order = {"none": 0, "low": 1, "low-medium": 1.5, "medium": 2, "high": 3}
    reported_numeric = lit_df["efficacy_category"].map(category_order).to_numpy()
    reported_ifn = lit_df["reported_ifn_response"].to_numpy()

    pearson_r, pearson_p = stats.pearsonr(moe_pred, reported_ifn)
    spearman_r, spearman_p = stats.spearmanr(moe_pred, reported_numeric)

    n_agree = 0
    n_total = 0
    case_results = []

    for i, row in lit_df.iterrows():
        pred_val = float(moe_pred[i])
        reported_cat = row["efficacy_category"]
        reported_ifn_val = row["reported_ifn_response"]

        predicted_high = pred_val > np.median(moe_pred)
        actual_high = reported_cat in ("high", "medium")
        direction_agree = predicted_high == actual_high

        n_total += 1
        if direction_agree:
            n_agree += 1

        case_results.append({
            "epitope_seq": row["epitope_seq"],
            "dose": row["dose"],
            "predicted_efficacy": round(pred_val, 4),
            "reported_category": reported_cat,
            "reported_ifn_response": reported_ifn_val,
            "direction_agree": direction_agree,
            "citation": row["citation"],
        })

        print(f"  {row['epitope_seq']:15s} pred={pred_val:.3f} "
              f"reported={reported_cat:12s} IFN={reported_ifn_val:.1f} "
              f"{'OK' if direction_agree else 'MISMATCH'}")

    direction_accuracy = n_agree / max(n_total, 1)

    print(f"\n  Pearson r (pred vs IFN): {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman r (pred vs category): {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"  Direction accuracy: {direction_accuracy:.2%} ({n_agree}/{n_total})")

    return {
        "n_cases": len(lit_df),
        "pearson_r_with_ifn": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r_with_category": float(spearman_r),
        "spearman_p": float(spearman_p),
        "direction_accuracy": float(direction_accuracy),
        "n_direction_tests": n_total,
        "cases": case_results,
        "data_source": "Published circRNA vaccine literature",
        "references": [
            "Wesselhoeft et al. (2018) Nature Communications",
            "Chen et al. (2017) Cell Research",
            "Liu et al. (2019) Nature Communications",
            "Yang et al. (2017) Cell Research",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Confluencia Clinical Validation")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--iedb", action="store_true")
    parser.add_argument("--chembl", action="store_true")
    parser.add_argument("--netmhcpan", action="store_true")
    parser.add_argument("--literature", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.iedb, args.chembl, args.netmhcpan, args.literature]):
        args.all = True

    print("=" * 60)
    print("Confluencia Clinical Validation Experiments")
    print("=" * 60)

    all_results = {}

    if args.all or args.iedb:
        try:
            all_results["iedb_mhc_validation"] = validate_iedb_mhc()
        except Exception as e:
            print(f"  IEDB validation error: {e}")
            all_results["iedb_mhc_validation"] = {"error": str(e)}

    if args.all or args.chembl:
        try:
            all_results["chembl_drug_validation"] = validate_chembl_drug()
        except Exception as e:
            print(f"  ChEMBL validation error: {e}")
            all_results["chembl_drug_validation"] = {"error": str(e)}

    if args.all or args.netmhcpan:
        try:
            all_results["netmhcpan_concordance"] = validate_netmhcpan_concordance()
        except Exception as e:
            print(f"  NetMHCpan error: {e}")
            all_results["netmhcpan_concordance"] = {"error": str(e)}

    if args.all or args.literature:
        try:
            all_results["literature_cases"] = validate_literature_cases()
        except Exception as e:
            print(f"  Literature validation error: {e}")
            all_results["literature_cases"] = {"error": str(e)}

    out_path = RESULTS_DIR / "clinical_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("CLINICAL VALIDATION SUMMARY")
    print("=" * 60)
    for exp_name, exp_result in all_results.items():
        if "error" in exp_result:
            print(f"  {exp_name}: ERROR - {exp_result['error']}")
        else:
            n = exp_result.get("n_heldout", exp_result.get("n_test", exp_result.get("n_cases", "?")))
            print(f"  {exp_name}: {n} samples validated")

    return all_results


if __name__ == "__main__":
    main()
