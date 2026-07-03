#!/usr/bin/env python3
"""
load_pipeline_dataset.py — Load circRNA 3D structures exported by
deploy_package/circrna_3d_pipeline pipeline (three-layer sampling).

Reads the 5 .npy files exported by export_torusfold():
    coords.npy            (N, L, 3)  — C3' coordinates (padded to max L)
    confidences.npy       (N,)       — per-sample confidence
    sample_weights.npy    (N,)       — A-layer=1.0, B-layer=0.3
    sources.npy           (N,)       — 'real' / 'synthetic' / 'benchmark'
    hard_negatives.npy    (Nc, ...)  — C-layer, excluded from training

And produces the same 5-tuple format as load_pseudo_labels() so it can
drop into train_scheme6_fixed.py / train_all_schemes.py unchanged:
    (sequences, coords_labels, pair_labels, confidence_weights, metadata)

Scheme C split (training_strategy_v2):
    real      -> 85% train, 5% val, 10% test-B
    synthetic -> 100% train
    test-A    -> independent PDB benchmark (not in this file)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse the existing dataset/collate so behavior matches train_all_schemes.py
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    CircRNADataset, collate_fn,
)


def load_pipeline_export(export_dir, include_soft_noise=True, max_len=None):
    """Load pipeline-exported .npy files into the (seqs, coords, pairs, confs, meta) format.

    Args:
        export_dir: directory containing coords.npy / confidences.npy /
                    sample_weights.npy / sources.npy [/ hard_negatives.npy]
        include_soft_noise: if False, drop B-layer samples (weight==0.3)
        max_len: optional max sequence length filter

    Returns:
        (sequences, coords_labels, pair_labels, confidence_weights, metadata)
        — same shape as load_pseudo_labels()
    """
    coords = np.load(os.path.join(export_dir, 'coords.npy'), allow_pickle=True)
    confidences = np.load(os.path.join(export_dir, 'confidences.npy'), allow_pickle=True)
    weights = np.load(os.path.join(export_dir, 'sample_weights.npy'), allow_pickle=True)
    sources = np.load(os.path.join(export_dir, 'sources.npy'), allow_pickle=True)

    # Optional: sequences.json written alongside by export (if not present,
    # we cannot recover the RNA string from coords alone — pipeline must
    # also dump sequences.json). Fall back to empty strings if missing.
    seq_path = os.path.join(export_dir, 'sequences.json')
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            sequences = json.load(f)
    else:
        # coords (N, L, 3) -> assume L nucleotides; placeholder 'N' * L
        sequences = []
        for i in range(len(coords)):
            L = coords[i].shape[0] if coords[i].ndim == 3 else coords.shape[1]
            sequences.append('N' * L)

    sequences = list(sequences)
    n_total = len(sequences)
    if not (len(confidences) == n_total and len(weights) == n_total and len(sources) == n_total):
        raise ValueError(
            f"Length mismatch: seqs={n_total} conf={len(confidences)} "
            f"weights={len(weights)} sources={len(sources)}"
        )

    # Filter: drop B-layer if not wanted; C-layer is already excluded by
    # export_torusfold (only A+B land in coords.npy).
    keep = np.ones(n_total, dtype=bool)
    if not include_soft_noise:
        keep = weights == 1.0

    # Filter corrupt coords (NaN/Inf/zero)
    for i in range(n_total):
        c = coords[i] if coords.ndim == 1 else coords[i]
        if np.isnan(c).any() or np.isinf(c).any() or np.abs(c).max() == 0:
            keep[i] = False

    # Build per-sample lists
    sequences_f, coords_f, pairs_f, confs_f, meta_f = [], [], [], [], []
    for i in range(n_total):
        if not keep[i]:
            continue
        c = coords[i] if coords.ndim == 1 else coords[i]
        L = c.shape[0]
        seq = sequences[i]
        # Truncate seq to match coords length (pipeline may have padded)
        if len(seq) > L:
            seq = seq[:L]
        elif len(seq) < L:
            seq = seq + 'N' * (L - len(seq))

        if max_len is not None and L > max_len:
            continue

        sequences_f.append(seq)
        coords_f.append(np.asarray(c, dtype=np.float32))
        # Pair matrix: pipeline export does not carry pairs; build zero matrix.
        # train_all_schemes.CircRNADataset handles pair_labels=None gracefully.
        pairs_f.append(np.zeros((L, L), dtype=np.float32))
        # confidence_weights: combine A/B layer weight with per-sample confidence.
        # weight=1.0 (A) -> use confidence; weight=0.3 (B) -> scale down.
        confs_f.append(float(weights[i]) * float(confidences[i]))
        meta_f.append({
            'id': f'pipeline_{i:06d}',
            'length': L,
            'source': str(sources[i]),
            'confidence': confs_f[-1],
            'layer': 'A' if weights[i] == 1.0 else 'B',
        })

    print(f"  Loaded {len(sequences_f)} samples from {export_dir}")
    print(f"    A-layer: {sum(1 for m in meta_f if m['layer']=='A')} | "
          f"B-layer: {sum(1 for m in meta_f if m['layer']=='B')}")
    print(f"    real: {sum(1 for m in meta_f if m['source']=='real')} | "
          f"synthetic: {sum(1 for m in meta_f if m['source']=='synthetic')}")
    return sequences_f, coords_f, pairs_f, confs_f, meta_f


def scheme_c_split(sequences, coords, pairs, confs, meta,
                   val_ratio=0.05, test_b_ratio=0.10, seed=42):
    """Split data per training_strategy_v2 Scheme C.

    real      -> train (1 - val - test_b), val, test-B
    synthetic -> train only (100%)

    Returns:
        dict with 'train'/'val'/'test_b' keys, each a
        (sequences, coords, pairs, confs, meta) tuple
    """
    rng = np.random.RandomState(seed)
    n = len(sequences)

    real_idx = [i for i in range(n) if meta[i]['source'] == 'real']
    syn_idx = [i for i in range(n) if meta[i]['source'] == 'synthetic']
    rng.shuffle(real_idx)

    n_real = len(real_idx)
    n_val = int(n_real * val_ratio)
    n_test = int(n_real * test_b_ratio)
    val_idx = real_idx[:n_val]
    test_idx = real_idx[n_val:n_val + n_test]
    train_real_idx = real_idx[n_val + n_test:]
    train_idx = train_real_idx + syn_idx

    def _gather(idx_list):
        s = [sequences[i] for i in idx_list]
        c = [coords[i] for i in idx_list]
        p = [pairs[i] for i in idx_list]
        cf = [confs[i] for i in idx_list]
        m = [meta[i] for i in idx_list]
        return s, c, p, cf, m

    splits = {
        'train': _gather(train_idx),
        'val': _gather(val_idx),
        'test_b': _gather(test_idx),
    }
    print(f"  Scheme C split: train={len(train_idx)} "
          f"(real={len(train_real_idx)}, syn={len(syn_idx)}) | "
          f"val={len(val_idx)} | test-B={len(test_idx)}")
    return splits


def build_scheme_c_loaders(export_dir, batch_size=4, max_len=None,
                            val_ratio=0.05, test_b_ratio=0.10, seed=42,
                            include_soft_noise=True, num_workers=0):
    """One-call: load pipeline export + Scheme C split + DataLoaders.

    Returns:
        (train_loader, val_loader, test_b_loader, splits_meta)
    """
    seqs, coords, pairs, confs, meta = load_pipeline_export(
        export_dir, include_soft_noise=include_soft_noise, max_len=max_len
    )
    splits = scheme_c_split(seqs, coords, pairs, confs, meta,
                            val_ratio=val_ratio, test_b_ratio=test_b_ratio, seed=seed)

    def _make_loader(split, shuffle):
        s, c, p, cf, m = split
        ds = CircRNADataset(s, c, p, cf)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=num_workers)

    train_loader = _make_loader(splits['train'], shuffle=True)
    val_loader = _make_loader(splits['val'], shuffle=False)
    test_loader = _make_loader(splits['test_b'], shuffle=False)

    return train_loader, val_loader, test_loader, splits


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load pipeline export + Scheme C split')
    parser.add_argument('--export-dir', required=True,
                        help='Directory with coords.npy/confidences.npy/'
                             'sample_weights.npy/sources.npy from pipeline')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-len', type=int, default=None)
    parser.add_argument('--no-soft-noise', action='store_true',
                        help='Exclude B-layer soft noise samples')
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--test-b-ratio', type=float, default=0.10)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, splits = build_scheme_c_loaders(
        args.export_dir, batch_size=args.batch_size, max_len=args.max_len,
        val_ratio=args.val_ratio, test_b_ratio=args.test_b_ratio,
        include_soft_noise=not args.no_soft_noise,
    )
    print(f"\nLoaders ready:")
    print(f"  train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"  test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")
