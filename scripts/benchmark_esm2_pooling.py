#!/usr/bin/env python3
"""
ESM-2 Pooling Mode Comparison Benchmark
========================================
Reviewer 1 concern: Mean pooling loses position information for MHC binding prediction.

This script compares three pooling strategies:
  - "mean": Mean pooling over all tokens (default, backward compatible)
  - "cls": CLS token embedding (position-aware, global context)
  - "anchor": Anchor position embeddings (P2, P3, P5 for MHC-I 9mer)

Benchmark on IEDB MHC-I binding data to assess whether position-aware pooling
improves epitope prediction accuracy.

Usage:
  # Quick test on CPU with small model
  python benchmark_esm2_pooling.py

  # Full benchmark on AutoDL GPU with 650M model
  ESM2_MODEL_SIZE=650M ESM2_BENCHMARK_N=2000 python benchmark_esm2_pooling.py

  # Use HuggingFace mirror (AutoDL China)
  HF_ENDPOINT=https://hf-mirror.com python benchmark_esm2_pooling.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-epitope"))
from core.esm2_encoder import ESM2Encoder

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(os.environ.get(
    "ESM2_DATA_DIR",
    str(PROJECT_ROOT / "confluencia-2.0-epitope" / "data"),
))
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model size: 35M for quick CPU test, 650M for full benchmark (needs GPU)
MODEL_SIZE = os.environ.get("ESM2_MODEL_SIZE", "35M")
N_SAMPLES = int(os.environ.get("ESM2_BENCHMARK_N", "500"))
N_FOLDS = 5
RANDOM_SEED = 42

# AutoDL HuggingFace mirror
if "HF_ENDPOINT" not in os.environ:
    # AutoDL provides hf-mirror.com as default mirror
    default_mirror = "https://hf-mirror.com"
    print(f"[Config] HF_ENDPOINT not set, using default: {default_mirror}")
    print(f"[Config] Set HF_ENDPOINT= to override, or export HF_ENDPOINT={default_mirror}")

np.random.seed(RANDOM_SEED)


# ============================================================================
# Data Loading
# ============================================================================

def load_iedb_data(max_samples: int = None) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load IEDB MHC-I binding data.

    Returns:
        sequences: List of peptide sequences
        labels: Binding affinity values (IC50 nM)
        alleles: List of allele names
    """
    # Try common IEDB data locations
    data_files = [
        DATA_DIR / "iedb_mhc_i_binding.csv",
        DATA_DIR / "mhc_i_binding.csv",
        DATA_DIR / "iedb_benchmark.csv",
    ]

    for data_file in data_files:
        if data_file.exists():
            print(f"[Data] Loading: {data_file}")
            df = pd.read_csv(data_file)

            # Standardize column names
            col_mapping = {
                "peptide": "sequence",
                "peptide_seq": "sequence",
                "sequence": "sequence",
                "ic50": "ic50",
                "affinity": "ic50",
                "binding_affinity": "ic50",
                "allele": "allele",
                "mhc_allele": "allele",
            }
            df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

            # Filter to 9mers (standard MHC-I peptide length)
            if "sequence" in df.columns:
                df = df[df["sequence"].str.len() == 9]

            # Subsample if needed
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=RANDOM_SEED)

            sequences = df["sequence"].tolist()

            # Get labels (IC50 or binary)
            if "ic50" in df.columns:
                labels = df["ic50"].values.astype(float)
            elif "binder" in df.columns:
                binary_labels = df["binder"].values.astype(int)
                labels = binary_labels.astype(float) * 100  # Dummy IC50
            else:
                labels = np.random.uniform(1, 10000, len(sequences))

            alleles = df["allele"].tolist() if "allele" in df.columns else ["A*02:01"] * len(sequences)

            print(f"[Data] Loaded {len(sequences)} 9mer peptides")
            return sequences, labels, alleles

    # No data file found — generate synthetic peptides for testing
    print(f"[Data] No IEDB file found in {DATA_DIR}")
    print("[Data] Generating synthetic 9mer peptides for pipeline testing")
    print("[Data] NOTE: Results on synthetic data are NOT meaningful for publication")
    print("[Data] To use real data, download IEDB MHC-I binding data and place in:")
    print(f"[Data]   {DATA_DIR}/iedb_mhc_i_binding.csv")
    print("[Data] Required columns: sequence (9mer peptide), ic50 (nM), allele")

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequences = ["".join(np.random.choice(list(amino_acids), 9)) for _ in range(max_samples or 200)]
    labels = np.random.uniform(1, 10000, len(sequences))
    alleles = ["A*02:01"] * len(sequences)

    return sequences, labels, alleles


# ============================================================================
# Pooling Comparison
# ============================================================================

def encode_with_pooling(
    sequences: List[str],
    pooling: str,
    model_size: str = MODEL_SIZE,
) -> np.ndarray:
    """Encode sequences with specified pooling mode."""
    print(f"\n[ESM-2] Encoding with pooling='{pooling}', model={model_size}")

    encoder = ESM2Encoder(
        model_size=model_size,
        pooling=pooling,
        batch_size=32,
        max_length=15,  # 9mer + BOS/EOS padding
    )

    start_time = time.time()
    embeddings = encoder.encode(sequences)
    elapsed = time.time() - start_time

    print(f"[ESM-2] Shape: {embeddings.shape}, Time: {elapsed:.1f}s")

    # For anchor pooling, dimension is 3x embed_dim (P2, P3, P5 concatenated)
    expected_dim = {
        "mean": {"35M": 480, "150M": 640, "650M": 1280},
        "cls": {"35M": 480, "150M": 640, "650M": 1280},
        "anchor": {"35M": 480 * 3, "150M": 640 * 3, "650M": 1280 * 3},
    }

    if pooling in expected_dim and model_size in expected_dim[pooling]:
        expected = expected_dim[pooling][model_size]
        if embeddings.shape[1] != expected:
            print(f"[Warning] Expected dim {expected}, got {embeddings.shape[1]}")

    return embeddings


def evaluate_pooling(
    embeddings: np.ndarray,
    labels: np.ndarray,
    binary_labels: np.ndarray,
    pooling: str,
) -> Dict:
    """Evaluate embeddings with cross-validation."""
    print(f"\n[Evaluation] Pooling: {pooling}")

    results = {"pooling": pooling, "embed_dim": embeddings.shape[1]}

    # Regression (IC50 prediction)
    # Use log(IC50) for better distribution
    log_ic50 = np.log10(labels + 1)  # +1 to avoid log(0)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    r2_scores = cross_val_score(ridge, embeddings, log_ic50, cv=cv, scoring="r2")
    results["ridge_r2_mean"] = float(np.mean(r2_scores))
    results["ridge_r2_std"] = float(np.std(r2_scores))
    print(f"  Ridge R2: {results['ridge_r2_mean']:.3f} +/- {results['ridge_r2_std']:.3f}")

    # Classification (binder vs non-binder)
    lr = LogisticRegression(max_iter=1000, C=0.1)
    auc_scores = cross_val_score(lr, embeddings, binary_labels, cv=cv, scoring="roc_auc")
    results["lr_auc_mean"] = float(np.mean(auc_scores))
    results["lr_auc_std"] = float(np.std(auc_scores))
    print(f"  LR AUC: {results['lr_auc_mean']:.3f} +/- {results['lr_auc_std']:.3f}")

    return results


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark():
    """Run full pooling comparison benchmark."""
    print("=" * 60)
    print("ESM-2 Pooling Mode Comparison Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Folds: {N_FOLDS}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Load data
    sequences, labels, alleles = load_iedb_data(max_samples=N_SAMPLES)
    binary_labels = (labels < 500).astype(int)

    print(f"\n[Data] Binder ratio: {binary_labels.mean():.2%}")
    print(f"[Data] IC50 range: {labels.min():.0f} - {labels.max():.0f} nM")

    # Test all pooling modes
    pooling_modes = ["mean", "cls", "anchor"]
    all_results = []
    all_embeddings = {}

    for pooling in pooling_modes:
        try:
            embeddings = encode_with_pooling(sequences, pooling)
            all_embeddings[pooling] = embeddings

            results = evaluate_pooling(embeddings, labels, binary_labels, pooling)
            all_results.append(results)

        except Exception as e:
            print(f"[Error] Pooling '{pooling}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "pooling": pooling,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / f"esm2_pooling_benchmark_{MODEL_SIZE}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model_size": MODEL_SIZE,
                "n_samples": N_SAMPLES,
                "n_folds": N_FOLDS,
                "random_seed": RANDOM_SEED,
                "data_dir": str(DATA_DIR),
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n[Output] Saved to: {output_file}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    valid_results = [r for r in all_results if "error" not in r]
    if len(valid_results) >= 2:
        best_r2 = max(valid_results, key=lambda x: x.get("ridge_r2_mean", 0))
        best_auc = max(valid_results, key=lambda x: x.get("lr_auc_mean", 0))

        print(f"Best R2 (IC50 prediction): {best_r2['pooling']} = {best_r2['ridge_r2_mean']:.3f}")
        print(f"Best AUC (Binder classification): {best_auc['pooling']} = {best_auc['lr_auc_mean']:.3f}")

        # Check if anchor/cls improves over mean
        mean_result = next((r for r in valid_results if r["pooling"] == "mean"), None)
        if mean_result:
            for other in ["cls", "anchor"]:
                other_result = next((r for r in valid_results if r["pooling"] == other), None)
                if other_result:
                    r2_diff = other_result.get("ridge_r2_mean", 0) - mean_result.get("ridge_r2_mean", 0)
                    auc_diff = other_result.get("lr_auc_mean", 0) - mean_result.get("lr_auc_mean", 0)

                    if r2_diff > 0.01:
                        print(f"\n[+] {other.upper()} improves R2 by +{r2_diff:.3f} over mean pooling")
                    elif r2_diff < -0.01:
                        print(f"\n[-] {other.upper()} decreases R2 by {r2_diff:.3f} vs mean pooling")
                    else:
                        print(f"\n[=] {other.upper()} R2 similar to mean pooling (diff: {r2_diff:.3f})")

    return all_results


if __name__ == "__main__":
    run_benchmark()
