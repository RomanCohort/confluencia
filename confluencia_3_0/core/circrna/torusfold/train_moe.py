#!/usr/bin/env python3
"""
train_moe.py — Train TorusFold Mixture-of-Experts model.

Phase 2: Freeze pretrained experts, train gating network + fusion.
Phase 3 (optional): End-to-end fine-tuning with low LR.

Usage:
    # Phase 2: train gating + fusion (experts frozen)
    python train_moe.py --pretrained-dir models/torusfold --epochs 30 --phase 2

    # Phase 3: end-to-end fine-tuning
    python train_moe.py --pretrained-dir models/torusfold --epochs 10 --phase 3 --lr 1e-5

    # With medium-length data
    python train_moe.py --labels data/medium_length_3d --pretrained-dir models/torusfold_medium
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.torusfold_moe import (
    TorusFoldMOE, TorusFoldMOEConfig, train_torusfold_moe,
)
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn, kabsch_rmsd,
)


def discover_pretrained(pretrained_dir: str) -> dict:
    """Find pretrained scheme checkpoints in directory."""
    paths = {}
    if not os.path.isdir(pretrained_dir):
        return paths

    for fname in os.listdir(pretrained_dir):
        fpath = os.path.join(pretrained_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Match patterns: scheme1_best.pt, scheme3.pt, etc.
        for sid in range(1, 8):
            patterns = [
                f"scheme{sid}_best.pt",
                f"scheme{sid}.pt",
                f"scheme{sid}_best.pth",
                f"scheme{sid}.pth",
            ]
            if fname in patterns:
                paths[sid] = fpath
                break

    print(f"  Discovered pretrained experts:")
    for sid, path in sorted(paths.items()):
        size_mb = os.path.getsize(path) / 1e6
        print(f"    Scheme {sid}: {path} ({size_mb:.1f} MB)")

    missing = set(range(1, 8)) - set(paths.keys())
    if missing:
        print(f"    Missing schemes: {sorted(missing)} (will use helical fallback)")

    return paths


def build_phase2_args(args):
    """Override args for Phase 2 (gating + fusion training)."""
    args.lr = args.gate_lr
    args.epochs = args.phase2_epochs
    return args


def build_phase3_args(args):
    """Override args for Phase 3 (end-to-end fine-tuning)."""
    args.lr = args.finetune_lr
    args.epochs = args.phase3_epochs
    return args


def main():
    parser = argparse.ArgumentParser(description='Train TorusFold MOE model')

    # Data
    parser.add_argument('--labels', type=str, default='',
                        help='Path to pre-generated pseudo-labels directory')
    parser.add_argument('--n-train', type=int, default=2000,
                        help='Number of samples if no --labels')
    parser.add_argument('--min-len', type=int, default=30)
    parser.add_argument('--max-len', type=int, default=0,
                        help='Max sequence length (0=no limit)')

    # Pretrained experts
    parser.add_argument('--pretrained-dir', type=str, default='models/torusfold',
                        help='Directory containing scheme checkpoints')

    # MOE config
    parser.add_argument('--top-k', type=int, default=2,
                        help='Number of experts to select per sequence')
    parser.add_argument('--fusion-mode', type=str, default='confidence',
                        choices=['weighted_avg', 'confidence', 'stacked_refine'],
                        help='Expert fusion strategy')
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)

    # Training
    parser.add_argument('--phase', type=int, default=2,
                        choices=[2, 3],
                        help='Training phase: 2=gating+fusion, 3=end-to-end')
    parser.add_argument('--phase2-epochs', type=int, default=30)
    parser.add_argument('--phase3-epochs', type=int, default=10)
    parser.add_argument('--gate-lr', type=float, default=5e-4)
    parser.add_argument('--finetune-lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (overridden by phase)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs (overridden by phase)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--load-balance-weight', type=float, default=0.01)

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default='models/torusfold_moe')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # Apply phase-specific settings
    if args.phase == 2:
        args = build_phase2_args(args)
    elif args.phase == 3:
        args = build_phase3_args(args)

    print("=" * 60)
    print("  TorusFold MOE Training")
    print("=" * 60)
    print(f"  Phase: {args.phase}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Fusion: {args.fusion_mode}")
    print(f"  Device: {args.device}")

    # Load data
    if args.labels and os.path.exists(args.labels):
        print(f"  Loading from: {args.labels}")
        max_len_filter = args.max_len if args.max_len > 0 else None
        sequences, coords_labels, pair_labels, metadata = load_pseudo_labels(
            args.labels, max_len=max_len_filter)
    else:
        print(f"  Generating pseudo-labels (n={args.n_train})")
        from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
            generate_3d_pseudo_labels,
        )
        gen_max = args.max_len if args.max_len > 0 else 500
        sequences, coords_labels, pair_labels, metadata = generate_3d_pseudo_labels(
            n_seqs=args.n_train, min_len=args.min_len,
            max_len=gen_max, seed=args.seed,
        )

    if len(sequences) < 10:
        print("ERROR: Not enough data.")
        return

    print(f"  Training samples: {len(sequences)}")

    # Filter for MOE: use all schemes' max length
    max_scheme_len = 800  # conservative for most schemes
    if args.max_len > 0:
        max_scheme_len = args.max_len
    keep = [i for i, m in enumerate(metadata) if m['length'] <= max_scheme_len]
    if len(keep) < len(sequences):
        print(f"  Filtered: {len(sequences)} → {len(keep)} (max_len={max_scheme_len})")
    sequences = [sequences[i] for i in keep]
    coords_labels = [coords_labels[i] for i in keep]
    pair_labels = [pair_labels[i] for i in keep]

    # Split
    split = int(0.9 * len(sequences))
    train_ds = CircRNADataset(sequences[:split], coords_labels[:split], pair_labels[:split])
    val_ds = CircRNADataset(sequences[split:], coords_labels[split:], pair_labels[split:])

    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=(device.type == 'cuda'))

    # Discover pretrained experts
    pretrained_paths = discover_pretrained(args.pretrained_dir)

    # Save MOE config
    moe_config = {
        'n_experts': 7,
        'top_k': args.top_k,
        'fusion_mode': args.fusion_mode,
        'd_hidden': args.d_hidden,
        'n_layers': args.n_layers,
        'phase': args.phase,
        'gate_lr': args.gate_lr,
        'finetune_lr': args.finetune_lr,
        'load_balance_weight': args.load_balance_weight,
        'bond_length': 5.9,
        'pretrained_paths': {str(k): v for k, v in pretrained_paths.items()},
    }
    with open(f"{args.output}/moe_config.json", 'w') as f:
        json.dump(moe_config, f, indent=2)

    # Train
    t0 = time.time()
    best_val = train_torusfold_moe(
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        pretrained_paths=pretrained_paths,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  MOE Training Complete")
    print(f"  Phase: {args.phase}")
    print(f"  Best val RMSD: {best_val:.2f}Å")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Model: {args.output}/torusfold_moe_best.pt")
    print(f"  Config: {args.output}/moe_config.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
