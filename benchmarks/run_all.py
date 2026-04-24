"""
Confluencia Full Benchmark Runner
===================================
Runs all experiments for Bioinformatics journal submission.

Usage:
    python -m benchmarks.run_all --data-epitope data/example_epitope.csv
    python -m benchmarks.run_all --data-drug data/example_drug.csv
    python -m benchmarks.run_all --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def run_all(
    data_epitope: str = "data/example_epitope.csv",
    data_drug: str = "data/example_drug.csv",
    output_dir: str = "benchmarks/results",
    seed: int = 42,
):
    """Run all benchmark experiments."""
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "experiments": {},
    }

    # ---- 1. Ablation Study ----
    print("=" * 60)
    print("EXPERIMENT 1: Ablation Study (Epitope)")
    print("=" * 60)
    try:
        from benchmarks.ablation import run_epitope_ablation
        path = run_epitope_ablation(data_epitope, output_dir=str(output), seed=seed)
        manifest["experiments"]["ablation_epitope"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["ablation_epitope"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("EXPERIMENT 1b: Ablation Study (Drug)")
    print("=" * 60)
    try:
        from benchmarks.ablation import run_drug_ablation
        path = run_drug_ablation(data_drug, output_dir=str(output), seed=seed)
        manifest["experiments"]["ablation_drug"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["ablation_drug"] = {"status": "failed", "error": str(e)}

    # ---- 2. Baseline Comparison ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Baseline Comparison (Epitope)")
    print("=" * 60)
    try:
        from benchmarks.baselines import run_baseline_comparison
        path = run_baseline_comparison("epitope", data_epitope, output_dir=str(output), seed=seed)
        manifest["experiments"]["baselines_epitope"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["baselines_epitope"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("EXPERIMENT 2b: Baseline Comparison (Drug)")
    print("=" * 60)
    try:
        path = run_baseline_comparison("drug", data_drug, output_dir=str(output), seed=seed)
        manifest["experiments"]["baselines_drug"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["baselines_drug"] = {"status": "failed", "error": str(e)}

    # ---- 3. Sample Size Sensitivity ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Sample Size Sensitivity (Epitope)")
    print("=" * 60)
    try:
        from benchmarks.sample_sensitivity import run_sensitivity_experiment
        path = run_sensitivity_experiment(
            "epitope", data_epitope, output_dir=str(output), seed=seed,
        )
        manifest["experiments"]["sensitivity_epitope"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["sensitivity_epitope"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("EXPERIMENT 3b: Sample Size Sensitivity (Drug)")
    print("=" * 60)
    try:
        path = run_sensitivity_experiment(
            "drug", data_drug, output_dir=str(output), seed=seed,
        )
        manifest["experiments"]["sensitivity_drug"] = {"status": "ok", "path": path}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["sensitivity_drug"] = {"status": "failed", "error": str(e)}

    # ---- 4. Torch-Mamba Corrected ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Torch-Mamba Corrected Config")
    print("=" * 60)
    try:
        from benchmarks.mamba_fix import run_mamba_experiment
        res = run_mamba_experiment(data_epitope, str(output), seed=seed)
        manifest["experiments"]["mamba_corrected"] = {"status": "ok", "results": res}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["mamba_corrected"] = {"status": "failed", "error": str(e)}

    # ---- 5. Data Leakage Check ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Data Leakage Verification")
    print("=" * 60)
    try:
        from benchmarks.sequence_split import verify_no_leakage, sequence_split
        df_epi = pd.read_csv(project_root / data_epitope)
        if "epitope_seq" in df_epi.columns:
            train, test = sequence_split(df_epi, "epitope_seq", seed=seed)
            ok = verify_no_leakage(train, test, "epitope_seq")
            manifest["experiments"]["leakage_check_epitope"] = {"status": "ok" if ok else "leakage_detected"}

        df_drug = pd.read_csv(project_root / data_drug)
        if "smiles" in df_drug.columns:
            train, test = sequence_split(df_drug, "smiles", seed=seed)
            ok = verify_no_leakage(train, test, "smiles")
            manifest["experiments"]["leakage_check_drug"] = {"status": "ok" if ok else "leakage_detected"}
    except Exception as e:
        print(f"FAILED: {e}")
        manifest["experiments"]["leakage_check"] = {"status": "failed", "error": str(e)}

    # Save manifest
    manifest["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path = output / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"All experiments completed. Manifest: {manifest_path}")
    print(f"{'=' * 60}")

    # Summary
    ok_count = sum(1 for v in manifest["experiments"].values() if v.get("status") == "ok")
    total = len(manifest["experiments"])
    print(f"Results: {ok_count}/{total} experiments succeeded")


def main():
    import pandas as pd  # noqa: ensure pandas is importable

    parser = argparse.ArgumentParser(description="Run all Confluencia benchmarks")
    parser.add_argument("--data-epitope", default="data/example_epitope.csv")
    parser.add_argument("--data-drug", default="data/example_drug.csv")
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all(args.data_epitope, args.data_drug, args.output, args.seed)


if __name__ == "__main__":
    main()
