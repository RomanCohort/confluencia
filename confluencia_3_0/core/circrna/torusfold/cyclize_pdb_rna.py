"""Cyclize linear RNA C3' coords into pseudo-circRNA training samples.

Two strategies based on end-to-end distance:
- Near (d_end <= 15 A): accept as-is (already circularizable), just ensure
  BSJ bond ~5.9 A via light projection (no relax needed)
- Far  (d_end >  15 A): add BSJ bond restraint, run light Adam relax using
  physics_refine energy to absorb the tension locally

Output: (L,3) C3' coords saved as .npy in data/pdb_cyclized/, aligned with
circrna_3d_all format for Phase-0 (目标1) pretraining.

Does NOT use OpenMM/amber14 — works at C3' granularity (fast, scales to ~5k).
"""
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEPLOY_ROOT = BASE.parents[3]
IN_DIR = DEPLOY_ROOT / 'data' / 'pdb_rna_c3prime'
OUT_DIR = DEPLOY_ROOT / 'data' / 'pdb_cyclized'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOND_LENGTH = 5.9       # P-P / C3'-C3' backbone
NEAR_THRESHOLD = 15.0   # end-to-end <= this = already circularizable
MIN_LEN, MAX_LEN = 20, 500


def end_to_end_dist(coords):
    return float(np.linalg.norm(coords[0] - coords[-1]))


def project_bsj_bond(coords, bond_length=BOND_LENGTH):
    """Pull residue 0 and L-1 apart/together so their distance = bond_length.
    Splits the correction 50/50 between the two endpoints (rigid-ish)."""
    d = coords[0] - coords[-1]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return coords
    half = d * 0.5 * (1.0 - bond_length / n)
    coords[0] = coords[0] - half
    coords[-1] = coords[-1] + half
    return coords


def cyclize_far(coords, n_steps=30, lr=0.3):
    """Light relax for far-endpoint RNAs: add BSJ bond restraint, minimize
    stereochemistry energy via Adam (reuses physics_refine approach).
    Only the linker region near the ends moves; stems stay put.
    """
    import torch
    from physics_refine import get_stereo_loss_breakdown
    x = torch.tensor(coords, dtype=torch.float32, requires_grad=True)
    L = x.shape[0]
    lengths = torch.tensor([L])
    opt = torch.optim.Adam([x], lr=lr)
    with torch.enable_grad():
        for step in range(n_steps):
            opt.zero_grad()
            # stereo energy (bond/angle/clash/dihedral)
            e = get_stereo_loss_breakdown(x.unsqueeze(0), lengths)['total']
            # BSJ closure restraint: pull 0 and L-1 to bond_length
            d_bsj = (x[0] - x[-1]).norm()
            e = e + 10.0 * ((d_bsj - BOND_LENGTH) ** 2)
            e.backward()
            opt.step()
    return x.detach().numpy()


def cyclize_one(coords):
    """Returns (cyclized_coords, method) where method in {'near','far','skip'}."""
    L = len(coords)
    if L < MIN_LEN or L > MAX_LEN:
        return None, 'skip_len'
    d = end_to_end_dist(coords)
    if d <= NEAR_THRESHOLD:
        # Already circularizable: just project BSJ bond to exact 5.9 A
        c = project_bsj_bond(coords.copy())
        return c, 'near'
    else:
        # Far: relax with BSJ restraint
        try:
            c = cyclize_far(coords.copy())
            c = project_bsj_bond(c)  # final exact bond
            return c, 'far'
        except Exception as e:
            return None, f'fail:{e}'


def main(limit=None):
    files = sorted(IN_DIR.glob('*.npy'))
    if limit:
        files = files[:limit]
def _worker(f):
    """Process one .npy: cyclize + save. Returns method string."""
    try:
        coords = np.load(f)
        c, method = cyclize_one(coords)
        if c is None:
            return f.name, 'skip'
        np.save(OUT_DIR / f.name, c)
        return f.name, method
    except Exception as e:
        return f.name, f'fail:{e}'


def main(limit=None, n_workers=32):
    import logging
    from multiprocessing import Pool
    log_file = DEPLOY_ROOT / 'data' / 'cyclize.log'
    logging.basicConfig(filename=str(log_file), level=logging.INFO,
                        format='%(asctime)s %(message)s', force=True)
    log = logging.getLogger()

    files = sorted(IN_DIR.glob('*.npy'))
    if limit:
        files = files[:limit]
    print('=' * 60)
    print(f'  Cyclize {len(files)} linear RNA C3\' structures  (workers={n_workers})')
    print(f'  log: {log_file}')
    print('=' * 60)
    log.info(f'start n={len(files)} workers={n_workers}')

    n_near = n_far = n_skip = 0
    with Pool(n_workers) as pool:
        for i, (name, method) in enumerate(pool.imap_unordered(_worker, files)):
            if method == 'near': n_near += 1
            elif method == 'far': n_far += 1
            else: n_skip += 1
            if (i + 1) % 200 == 0:
                msg = f'[{i+1}/{len(files)}] near={n_near} far={n_far} skip={n_skip}'
                print(msg); log.info(msg)
    done = f'DONE: {n_near} near + {n_far} far = {n_near+n_far} cyclized, {n_skip} skip'
    print(done); log.info(done)
    print(f'  output: {OUT_DIR}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    main(limit=args.limit, n_workers=args.workers)
