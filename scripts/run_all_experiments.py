#!/usr/bin/env python3
"""
Confluencia Experiment Runner — One-click execution of all experiments

Runs experiments in priority order:
  D (case study) → A (adaptive vs fixed weights) → C (BioGatedMOE) → B (cross-module consistency)

Results are saved to benchmarks/results/ as JSON files.

Usage: python run_all_experiments.py
"""

import subprocess
import sys
import os
import json
import time
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))

EXPERIMENTS = [
    {
        "name": "D: End-to-End Case Study + A: Adaptive vs Fixed",
        "script": "experiment_D_case_study.py",
        "output": "experiment_D_case_study.json",
        "priority": "HIGH — core argument for paper",
    },
    {
        "name": "C: BioGatedMOE vs MOERegressor",
        "script": "experiment_C_bio_gated_moe.py",
        "output": "experiment_C_bio_gated_moe.json",
        "priority": "MEDIUM — bio-mimetic gating validation",
    },
    {
        "name": "B: Cross-Module Consistency",
        "script": "experiment_B_cross_module_consistency.py",
        "output": "experiment_B_cross_module_consistency.json",
        "priority": "MEDIUM — physiological plausibility validation",
    },
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")


def run_experiment(exp):
    """Run a single experiment script."""
    script_path = os.path.join(SCRIPTS_DIR, exp["script"])

    print(f"\n{'=' * 70}")
    print(f"Running: {exp['name']}")
    print(f"Priority: {exp['priority']}")
    print(f"Script: {script_path}")
    print(f"Expected output: {OUTPUT_DIR}/{exp['output']}")
    print(f"{'=' * 70}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout per experiment
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"SUCCESS — completed in {elapsed:.1f}s")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        print(f"FAILED — exit code {result.returncode}")
        print(f"STDERR: {result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr}")

    return {
        "experiment": exp["name"],
        "script": exp["script"],
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main():
    print("=" * 70)
    print("Confluencia Experiment Runner")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    succeeded = 0
    failed = 0

    for exp in EXPERIMENTS:
        exp_result = run_experiment(exp)
        results.append(exp_result)

        if exp_result["returncode"] == 0:
            succeeded += 1
        else:
            failed += 1

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Succeeded: {succeeded}/{len(EXPERIMENTS)}")
    print(f"  Failed: {failed}/{len(EXPERIMENTS)}")

    for r in results:
        status = "OK" if r["returncode"] == 0 else "FAILED"
        print(f"  [{status}] {r['experiment']} ({r['elapsed_s']:.1f}s)")

    # Save runner log
    runner_log = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
        "succeeded": succeeded,
        "failed": failed,
    }

    log_path = os.path.join(OUTPUT_DIR, "experiment_runner_log.json")
    with open(log_path, "w") as f:
        json.dump(runner_log, f, indent=2, default=str)

    print(f"\n  Runner log saved to: {log_path}")
    print("  All done!")

    # List output files
    print("\n  Generated output files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("experiment_") and f.endswith(".json"):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath)
            print(f"    {f} ({size} bytes)")


if __name__ == "__main__":
    main()