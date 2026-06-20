#!/usr/bin/env python3
"""
Parameter Sensitivity Experiments for TorusFold Proxy Experiment

Two experiments:
1. Harmonics sensitivity: Test H = 8, 16, 32, 64
2. BSJ window sensitivity: Test window = 10, 15, 20, 25

Runs on top of the main experiment's data generation.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

# Import shared functions from main experiment
sys.path.insert(0, str(Path(__file__).parent))
from torusfold_proxy_experiment import (
    generate_diverse_circrna_sequences,
    compute_pairing_probabilities_circ,
    compute_pairing_probabilities_synthetic,
    extract_bsj_targets,
    TorusPositionalEncoding,
    PairingPredictor,
    CircRNADataset,
    train_epoch,
    evaluate,
    check_dependencies,
)


def run_harmonics_sweep(
    sequences, targets, bsj_positions,
    harmonics_list=[8, 16, 32, 64],
    n_epochs=80,
    n_seeds=3,
    bsj_window=20,
    output_dir="results",
):
    """Test different harmonic counts for TPE."""
    print(f"\n{'='*60}")
    print("HARMONICS SENSITIVITY TEST")
    print(f"{'='*60}")
    print(f"Testing H = {harmonics_list}")
    print(f"Epochs: {n_epochs}, Seeds: {n_seeds}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = max(len(s) for _, s, _ in sequences)

    harmonics_results = {}

    for H in harmonics_list:
        print(f"\n--- H = {H} ---")
        d_model = 256  # Keep d_model fixed, projection handles mismatch

        all_mse_full = []
        all_mse_bsj = []

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 17
            torch.manual_seed(seed)
            np.random.seed(seed)

            dataset = CircRNADataset(sequences, targets, bsj_positions,
                                     max_len=max_len, bsj_window=bsj_window)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=True, drop_last=False)

            model = PairingPredictor(
                d_model=d_model,
                nhead=8,
                num_layers=6,
                use_tpe=True,
                n_harmonics=H,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            for epoch in range(n_epochs):
                train_epoch(model, dataloader, optimizer, criterion, device)

            mse_full, mse_bsj = evaluate(model, dataloader, criterion, device)
            all_mse_full.append(mse_full)
            all_mse_bsj.append(mse_bsj)

            print(f"  Seed {seed_idx+1}: BSJ MSE = {mse_bsj.mean():.6f}, Full MSE = {mse_full.mean():.6f}")

        harmonics_results[H] = {
            'mse_bsj_mean': float(np.concatenate(all_mse_bsj).mean()),
            'mse_bsj_std': float(np.concatenate(all_mse_bsj).std()),
            'mse_full_mean': float(np.concatenate(all_mse_full).mean()),
            'mse_full_std': float(np.concatenate(all_mse_full).std()),
        }

    # Print summary
    print(f"\n{'='*60}")
    print("HARMONICS SENSITIVITY RESULTS")
    print(f"{'='*60}")
    print(f"{'H':>5} | {'BSJ MSE':>12} | {'Full MSE':>12} | {'Notes':>20}")
    print("-" * 60)
    for H in harmonics_list:
        r = harmonics_results[H]
        note = "<-- default" if H == 16 else ""
        print(f"{H:>5} | {r['mse_bsj_mean']:.6f} +/- {r['mse_bsj_std']:.6f} | "
              f"{r['mse_full_mean']:.6f} +/- {r['mse_full_std']:.6f} | {note}")

    # Also run standard PE baseline for comparison
    print(f"\n--- Standard PE Baseline ---")
    all_mse_full = []
    all_mse_bsj = []
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 17
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = CircRNADataset(sequences, targets, bsj_positions,
                                 max_len=max_len, bsj_window=bsj_window)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, drop_last=False)

        model = PairingPredictor(
            d_model=256, nhead=8, num_layers=6, use_tpe=False,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            train_epoch(model, dataloader, optimizer, criterion, device)

        mse_full, mse_bsj = evaluate(model, dataloader, criterion, device)
        all_mse_full.append(mse_full)
        all_mse_bsj.append(mse_bsj)

    baseline_bsj = np.concatenate(all_mse_bsj).mean()
    baseline_full = np.concatenate(all_mse_full).mean()
    print(f"  Standard PE: BSJ MSE = {baseline_bsj:.6f}, Full MSE = {baseline_full:.6f}")

    harmonics_results['baseline_standard_pe'] = {
        'mse_bsj_mean': float(baseline_bsj),
        'mse_full_mean': float(baseline_full),
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'harmonics_sensitivity.json', 'w') as f:
        json.dump(harmonics_results, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        Hs = [h for h in harmonics_list]
        bsj_means = [harmonics_results[h]['mse_bsj_mean'] for h in Hs]
        bsj_stds = [harmonics_results[h]['mse_bsj_std'] for h in Hs]

        ax.errorbar(Hs, bsj_means, yerr=bsj_stds, marker='o', capsize=5,
                    label='TPE (varying H)', color='coral', linewidth=2)
        ax.axhline(y=baseline_bsj, color='steelblue', linestyle='--',
                   label='Standard PE baseline', linewidth=2)
        ax.set_xlabel('Number of Harmonics (H)')
        ax.set_ylabel('MSE (BSJ region)')
        ax.set_title('Harmonics Sensitivity: TPE vs Standard PE')
        ax.legend()
        ax.set_xticks(Hs)

        plt.tight_layout()
        plt.savefig(output_path / 'harmonics_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_path / 'harmonics_sensitivity.pdf', bbox_inches='tight')
        print(f"Plot saved to {output_path / 'harmonics_sensitivity.png'}")
    except ImportError:
        pass

    return harmonics_results


def run_bsj_window_sweep(
    sequences, bsj_positions_raw,
    windows=[10, 15, 20, 25],
    n_epochs=80,
    n_seeds=3,
    n_harmonics=16,
    use_viennarna=False,
    output_dir="results",
):
    """Test different BSJ window sizes."""
    print(f"\n{'='*60}")
    print("BSJ WINDOW SENSITIVITY TEST")
    print(f"{'='*60}")
    print(f"Testing windows = {windows}")
    print(f"Epochs: {n_epochs}, Seeds: {n_seeds}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = max(len(s) for _, s, _ in sequences)

    window_results = {}

    for window in windows:
        print(f"\n--- Window = +/-{window}nt ---")

        # Regenerate targets with this window size
        targets = []
        for circ_id, seq, bsj_pos in sequences:
            if use_viennarna:
                pairing_probs = compute_pairing_probabilities_circ(seq)
            else:
                pairing_probs = compute_pairing_probabilities_synthetic(seq, bsj_pos)
            target = extract_bsj_targets(pairing_probs, bsj_pos, window=window)
            targets.append(target)

        all_mse_full_tpe = []
        all_mse_bsj_tpe = []
        all_mse_full_std = []
        all_mse_bsj_std = []

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 17

            # TPE
            torch.manual_seed(seed)
            np.random.seed(seed)

            dataset = CircRNADataset(sequences, targets, bsj_positions_raw,
                                     max_len=max_len, bsj_window=window)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=True, drop_last=False)

            model = PairingPredictor(
                d_model=256, nhead=8, num_layers=6,
                use_tpe=True, n_harmonics=n_harmonics,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            for epoch in range(n_epochs):
                train_epoch(model, dataloader, optimizer, criterion, device)

            mse_full_tpe, mse_bsj_tpe = evaluate(model, dataloader, criterion, device)
            all_mse_full_tpe.append(mse_full_tpe)
            all_mse_bsj_tpe.append(mse_bsj_tpe)

            # Standard PE
            torch.manual_seed(seed)
            np.random.seed(seed)

            model_std = PairingPredictor(
                d_model=256, nhead=8, num_layers=6, use_tpe=False,
            ).to(device)

            optimizer_std = optim.Adam(model_std.parameters(), lr=1e-4)

            for epoch in range(n_epochs):
                train_epoch(model_std, dataloader, optimizer_std, criterion, device)

            mse_full_std, mse_bsj_std = evaluate(model_std, dataloader, criterion, device)
            all_mse_full_std.append(mse_full_std)
            all_mse_bsj_std.append(mse_bsj_std)

        tpe_bsj = np.concatenate(all_mse_bsj_tpe).mean()
        std_bsj = np.concatenate(all_mse_bsj_std).mean()
        delta = (tpe_bsj - std_bsj) / std_bsj * 100

        # Paired t-test
        tpe_per_seq = np.concatenate(all_mse_bsj_tpe)
        std_per_seq = np.concatenate(all_mse_bsj_std)
        if len(tpe_per_seq) > 1 and len(std_per_seq) > 1:
            min_len = min(len(tpe_per_seq), len(std_per_seq))
            t_stat, p_val = stats.ttest_rel(std_per_seq[:min_len], tpe_per_seq[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0

        window_results[window] = {
            'tpe_bsj_mse': float(tpe_bsj),
            'std_bsj_mse': float(std_bsj),
            'delta_percent': float(delta),
            'p_value': float(p_val),
            't_statistic': float(t_stat),
        }

        print(f"  TPE BSJ MSE: {tpe_bsj:.6f}")
        print(f"  Std BSJ MSE: {std_bsj:.6f}")
        print(f"  Delta: {delta:+.1f}%")
        print(f"  p-value: {p_val:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print("BSJ WINDOW SENSITIVITY RESULTS")
    print(f"{'='*60}")
    print(f"{'Window':>8} | {'TPE MSE':>10} | {'Std MSE':>10} | {'Delta':>8} | {'p-value':>8} | Notes")
    print("-" * 70)
    for w in windows:
        r = window_results[w]
        note = "<-- default" if w == 20 else ""
        print(f"+/-{w:>4}nt | {r['tpe_bsj_mse']:>10.6f} | {r['std_bsj_mse']:>10.6f} | "
              f"{r['delta_percent']:>+7.1f}% | {r['p_value']:>8.4f} | {note}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'bsj_window_sensitivity.json', 'w') as f:
        json.dump(window_results, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: MSE vs window
        ws = windows
        tpe_mses = [window_results[w]['tpe_bsj_mse'] for w in ws]
        std_mses = [window_results[w]['std_bsj_mse'] for w in ws]

        axes[0].plot(ws, tpe_mses, 'o-', label='TPE', color='coral', linewidth=2)
        axes[0].plot(ws, std_mses, 's--', label='Standard PE', color='steelblue', linewidth=2)
        axes[0].set_xlabel('BSJ Window (+/-nt)')
        axes[0].set_ylabel('MSE (BSJ region)')
        axes[0].set_title('MSE vs BSJ Window Size')
        axes[0].legend()
        axes[0].set_xticks(ws)

        # Plot 2: Delta% and p-value
        deltas = [window_results[w]['delta_percent'] for w in ws]
        p_vals = [window_results[w]['p_value'] for w in ws]

        ax2 = axes[1]
        ax2.bar([w - 1 for w in ws], deltas, width=2, label='Delta MSE (%)', color='coral', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('BSJ Window (+/-nt)')
        ax2.set_ylabel('Delta MSE (%)')
        ax2.set_title('TPE Improvement over Standard PE')

        # Mark significance
        for w in ws:
            p = window_results[w]['p_value']
            marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax2.annotate(marker, (w, window_results[w]['delta_percent']),
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path / 'bsj_window_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_path / 'bsj_window_sensitivity.pdf', bbox_inches='tight')
        print(f"Plot saved to {output_path / 'bsj_window_sensitivity.png'}")
    except ImportError:
        pass

    return window_results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TorusFold Parameter Sensitivity')
    parser.add_argument('--experiment', choices=['harmonics', 'window', 'both'],
                       default='both', help='Which experiment to run')
    parser.add_argument('--n-sequences', type=int, default=50)
    parser.add_argument('--n-epochs', type=int, default=80)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--viennarna', action='store_true')
    args = parser.parse_args()

    deps = check_dependencies()

    # Generate data once
    print("Generating circRNA sequences...")
    sequences = generate_diverse_circrna_sequences(args.n_sequences)
    print(f"  {len(sequences)} sequences, length range: "
          f"{min(len(s) for _, s, _ in sequences)}-{max(len(s) for _, s, _ in sequences)} nt")

    use_vr = args.viennarna and deps['viennarna']
    print(f"ViennaRNA: {'Yes' if use_vr else 'No (synthetic)'}")

    if args.experiment in ['harmonics', 'both']:
        # Generate targets with default window=20
        print("\nGenerating pairing probabilities (window=20)...")
        targets = []
        bsj_positions = []
        for circ_id, seq, bsj_pos in sequences:
            if use_vr:
                probs = compute_pairing_probabilities_circ(seq)
            else:
                probs = compute_pairing_probabilities_synthetic(seq, bsj_pos)
            target = extract_bsj_targets(probs, bsj_pos, window=20)
            targets.append(target)
            bsj_positions.append(bsj_pos)

        run_harmonics_sweep(
            sequences, targets, bsj_positions,
            harmonics_list=[8, 16, 32, 64],
            n_epochs=args.n_epochs,
            n_seeds=args.n_seeds,
            bsj_window=20,
            output_dir=args.output_dir,
        )

    if args.experiment in ['window', 'both']:
        run_bsj_window_sweep(
            sequences,
            [bp for _, _, bp in sequences],
            windows=[10, 15, 20, 25],
            n_epochs=args.n_epochs,
            n_seeds=args.n_seeds,
            n_harmonics=16,
            use_viennarna=use_vr,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*60}")
    print("ALL PARAMETER SENSITIVITY EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
