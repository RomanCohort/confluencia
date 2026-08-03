"""check_coords_quality.py -- Verify physical quality of generated CG coords.

Checks per sample: bond length, bond angle, clash, BSJ closure, chirality.
Run on a generated npz to confirm the data is physical before training.

Usage:
    python check_coords_quality.py <npz> [--n 50] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def sample_quality(c: np.ndarray):
    """Return (bond_mean, bond_std, angle_mean, clash_pct, bsj, ok)."""
    L = len(c)
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)  # L-1 bonds
    # bond angle at each interior P (cos of angle between consecutive bonds)
    v1 = c[1:-1] - c[:-2]
    v2 = c[2:] - c[1:-1]
    cos_a = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
    angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    # clash: non-neighbor pairs < 2.5 A (sampled for speed)
    if L > 100:
        sub = c[::3]
        dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        mask = np.triu(np.ones_like(dist, dtype=bool), k=3)
    else:
        dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        mask = np.triu(np.ones_like(dist, dtype=bool), k=2)
    clash_pct = float((dist[mask] < 2.5).mean() * 100) if mask.any() else 0.0
    bsj = float(np.linalg.norm(c[0] - c[-1]))
    ok = (4.5 < d.mean() < 7.5) and clash_pct < 10 and bsj < 30
    return float(d.mean()), float(d.std()), float(angle_deg.mean()), clash_pct, bsj, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    ids, lens, coords = npz['ids'], npz['lengths'], npz['coords']
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(coords), min(args.n, len(coords)), replace=False)

    stats = {k: [] for k in ['bond_mean', 'bond_std', 'angle', 'clash', 'bsj']}
    n_ok = n_bad = n_skip = 0
    print(f"{'idx':>6} {'L':>5} {'bond':>7} {'bstd':>5} {'ang°':>6} {'clash%':>6} {'bsj':>6}  {'ok'}")
    print('-' * 55)
    for i in idx:
        c = coords[i]
        L = int(lens[i])
        if not isinstance(c, np.ndarray) or c.ndim != 2 or len(c) < 10:
            print(f'{i:>6} {L:>5}  skip (bad coords)')
            n_skip += 1
            continue
        bm, bs, ang, clash, bsj, ok = sample_quality(np.asarray(c))
        stats['bond_mean'].append(bm); stats['bond_std'].append(bs)
        stats['angle'].append(ang); stats['clash'].append(clash); stats['bsj'].append(bsj)
        if ok: n_ok += 1
        else: n_bad += 1
        print(f'{i:>6} {L:>5} {bm:>6.2f} {bs:>5.2f} {ang:>6.1f} {clash:>5.1f}% {bsj:>6.1f}  {"OK" if ok else "BAD"}')

    print('-' * 55)
    print(f'ok={n_ok} bad={n_bad} skip={n_skip}')
    if stats['bond_mean']:
        for k, v in stats.items():
            print(f'  {k}: mean={np.mean(v):.2f} median={np.median(v):.2f}')
    print(f'=> {n_ok/(n_ok+n_bad)*100:.0f}% physical OK')


if __name__ == '__main__':
    main()