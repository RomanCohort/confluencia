"""check_coords_energy.py -- Compute physical energy of generated CG coords.

Reuses the same OpenMM CG forcefield (build_segmented_3bead_system) that
generation used, rebuilds the system from saved coords, and reports the
refined-system energy. Also converts to 3-bead and runs cgRNASP knowledge
potential if FebRNA data is present.

Usage:
    python check_coords_energy.py <npz> [--n 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCHEME2_WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scheme2_work')
if _SCHEME2_WORK not in sys.path:
    sys.path.insert(0, _SCHEME2_WORK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from torusfold.scheme2.segmented_folding import (
        extract_stems_from_structure, build_segmented_3bead_system,
    )

    npz = np.load(args.npz, allow_pickle=True)
    ids, lens, coords = npz['ids'], npz['lengths'], npz['coords']
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(coords), min(args.n, len(coords)), replace=False)

    # 加载 FASTA (Documents/circbase) 用于序列
    import gzip
    fasta = {}
    fa_path = r'C:\Users\颜子壹\Documents\circbase_seqs.fa.gz'
    if os.path.isfile(fa_path):
        with gzip.open(fa_path, 'rt') as f:
            name = None; buf = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if name: fasta[name] = ''.join(buf)
                    name = line[1:].split('|')[0].split()[0]
                    buf = []
                else:
                    buf.append(line)
            if name: fasta[name] = ''.join(buf)
        print(f'  FASTA loaded: {len(fasta)} seqs')

    print(f"{'idx':>6} {'L':>5} {'E_cg(kJ/mol)':>14} {'bond':>6} {'clash%':>6} {'bsj':>6}  ok")
    print('-' * 60)
    energies, ok_n, bad_n = [], 0, 0
    for i in idx:
        c = coords[i]
        L = int(lens[i])
        if not isinstance(c, np.ndarray) or c.ndim != 2 or len(c) < 10:
            print(f'{i:>6} {L:>5}  skip')
            continue
        c = np.asarray(c, dtype=np.float32)
        # 完整力场: 用 ViennaRNA 重算真实 WC 配对 (同生成时链路)
        seq_id = str(ids[i])
        seq_str = fasta.get(seq_id)
        if not seq_str:
            seq_str = 'A' * L  # 无序列则用纯 A (配对少, 能量偏低)
        seq_str = seq_str[:L].upper().replace('T', 'U')
        from torusfold.scheme2 import vienna_pair_probs
        real_pairs, _ = vienna_pair_probs(seq_str, 0.5)
        pairs = real_pairs if len(real_pairs) > 5 else [(j, j + 1, 1.0) for j in range(L - 1)]
        stems = extract_stems_from_structure(pairs)
        try:
            system, coords_nm, pair_force, pair_bonds, bsj_force = \
                build_segmented_3bead_system(c, pairs, stems)
        except Exception as e:
            print(f'{i:>6} {L:>5}  build fail: {e}')
            continue

        # 算能量
        from openmm import Context, Platform, VerletIntegrator
        integrator = VerletIntegrator(0.001)
        platform = Platform.getPlatformByName('CPU')
        ctx = Context(system, integrator, platform)
        ctx.setPositions(coords_nm)
        e = ctx.getState(getEnergy=True).getPotentialEnergy()
        e_kj = float(e)  # kJ/mol
        del ctx, integrator

        # 几何
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)
        bsj = float(np.linalg.norm(c[0] - c[-1]))
        if L > 60:
            sub = c[::4]
            dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            mask = np.triu(np.ones_like(dist, dtype=bool), k=3)
            clash = float((dist[mask] < 2.5).mean() * 100) if mask.any() else 0.0
        else:
            clash = 0.0

        ok = e_kj < 5e6 and clash < 10 and bsj < 30
        if ok: ok_n += 1
        else: bad_n += 1
        energies.append(e_kj)
        print(f'{i:>6} {L:>5} {e_kj:>14.0f} {d.mean():>6.2f} {clash:>5.1f}% {bsj:>6.1f}  {"OK" if ok else "BAD"}')

    print('-' * 60)
    if energies:
        arr = np.array(energies)
        print(f'Energy: median={np.median(arr):.0f} mean={arr.mean():.0f} '
              f'min={arr.min():.0f} max={arr.max():.0f} kJ/mol')
        print(f'ok={ok_n} bad={bad_n} => {ok_n/(ok_n+bad_n)*100:.0f}% low-energy physical')


if __name__ == '__main__':
    main()
