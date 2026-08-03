"""generate_training_data_scheme2.py -- Generate training data from circBase FASTA.

Pipeline: ViennaRNA pairing -> 3-bead segmented folding (init_from_secondary_structure +
  refine_segmented_3bead) -> RL far-pair optimization (optimize_far_pairs) -> npz output.

Input:  circBase FASTA (Documents/circbase_seqs.fa.gz)
Output: training npz with ids/lengths/coords (compatible with train_s10_curriculum)
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from multiprocessing import Process, Queue

import numpy as np

# [v5.2] 从本地 scheme2_work import (复制自 TorusFold-scheme2-rl, 独立运行)
_SCHEME2_WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scheme2_work')
if _SCHEME2_WORK not in sys.path:
    sys.path.insert(0, _SCHEME2_WORK)


def worker_fn(wid: int, in_q: Queue, out_q: Queue):
    from torusfold.scheme2 import vienna_pair_probs, build_full_pair_graph, extract_stem_blocks
    from torusfold.scheme2.rl_optimizer import optimize_far_pairs
    from torusfold.scheme2.segmented_folding import (
        init_from_secondary_structure, refine_segmented_3bead,
    )
    print(f'  worker {wid} ready', flush=True)
    while True:
        task = in_q.get()
        if task is None:
            break
        idx, seq_id, L, seq = task
        try:
            pairs, _ = vienna_pair_probs(seq, 0.5)
            p_init = init_from_secondary_structure(L, pairs)
            cg, e0, e1 = refine_segmented_3bead(p_init, pairs, 'CPU', n_anneal=60)
            _, scan_pairs, far_pairs = build_full_pair_graph(seq, pairs, do_scan=True)
            stem_blocks = extract_stem_blocks(pairs, scan_pairs)
            n_far = len(far_pairs)
            far_before = far_after = 0.0
            rl_info = None
            if far_pairs:
                opt_p, cg_orig, rl_info = optimize_far_pairs(
                    cg, seq, far_pairs, stem_blocks,
                    policy_path=None, n_simulations=50,
                    dpo_weight=5.0, dpo_simulate=True,
                )
                cg = opt_p
                # far_mean before/after (与 rl_optimizer 的 [pull] print 一致)
                try:
                    from torusfold.scheme2.cg_forcefield import p_coords_to_3bead
                    N_b = p_coords_to_3bead(cg_orig)[2::3]
                    N_a = p_coords_to_3bead(cg)[2::3]
                    db = np.array([np.linalg.norm(N_b[i]-N_b[j])
                                   for i, j in far_pairs if i < L and j < L])
                    da = np.array([np.linalg.norm(N_a[i]-N_a[j])
                                   for i, j in far_pairs if i < L and j < L])
                    if len(db): far_before = float(db.mean())
                    if len(da): far_after = float(da.mean())
                except Exception:
                    pass
            if np.isnan(cg).any() or np.isinf(cg).any():
                raise RuntimeError('NaN/Inf')
            out_q.put((idx, np.asarray(cg, dtype=np.float32),
                       n_far, far_before, far_after, e0, e1))
        except Exception as e:
            out_q.put((idx, None, str(e)[:100]))
            print(f'  worker {wid} err idx={idx} L={L}: {e}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta',
                    default=r'C:\Users\颜子壹\Documents\circbase_seqs.fa.gz')
    ap.add_argument('--max-len', type=int, default=5000)
    ap.add_argument('--n-workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out',
                    default=r'C:\Users\颜子壹\deploy\IGEM集成方案\data\circrna_training_scheme2.npz')
    args = ap.parse_args()

    # Read FASTA
    print(f'FASTA: {args.fasta}')
    all_ids, all_lens, all_seqs = [], [], []
    with gzip.open(args.fasta, 'rt') as f:
        name = None; buf = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    all_ids.append(name)
                    all_lens.append(len(''.join(buf)))
                    all_seqs.append(''.join(buf))
                name = line[1:].split('|')[0].split()[0]
                buf = []
            else:
                buf.append(line)
        if name:
            all_ids.append(name)
            all_lens.append(len(''.join(buf)))
            all_seqs.append(''.join(buf))
    print(f'  Total: {len(all_ids)} sequences')

    # Filter by max-len
    pass_idx = [i for i in range(len(all_ids)) if all_lens[i] <= args.max_len]
    tasks = [(ti, all_ids[i], all_lens[i], all_seqs[i])
             for ti, i in enumerate(pass_idx)]
    del all_seqs
    print(f'  <= {args.max_len}nt: {len(tasks)} sequences')
    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f'  Limit: {len(tasks)}')

    # Multi-process dispatch
    in_q, out_q = Queue(), Queue()
    n_workers = min(args.n_workers, len(tasks), 32)
    procs = [Process(target=worker_fn, args=(w, in_q, out_q), daemon=True)
             for w in range(n_workers)]
    for p in procs: p.start()
    for t in tasks: in_q.put(t)
    for _ in procs: in_q.put(None)

    # Collect results
    result_ids = [None] * len(tasks)
    result_coords = [None] * len(tasks)
    result_lens = [None] * len(tasks)
    result_meta = [None] * len(tasks)   # (n_far, far_before, far_after, e0, e1)
    done = 0; t0 = time.time()
    rl_hits = 0; rl_total_far = 0; far_improv_sum = 0.0
    e1_vals = []; e1_by_len = {}
    while done < len(tasks):
        res = out_q.get()
        if len(res) == 3:
            idx, _, err = res
            print(f'  [{done}/{len(tasks)}] err {idx}: {err}', flush=True)
        else:
            idx, c, n_far, far_b, far_a, e0, e1 = res
            orig = tasks[idx]
            result_ids[idx] = orig[1]
            result_lens[idx] = orig[2]
            result_coords[idx] = c
            result_meta[idx] = (n_far, far_b, far_a, e0, e1)
            if n_far > 0:
                rl_hits += 1
                rl_total_far += n_far
                if far_b > 0: far_improv_sum += max(0.0, far_b - far_a)
            if e1 is not None and not np.isnan(e1):
                e1_vals.append(e1)
                bkt = 'short' if orig[2] < 200 else ('mid' if orig[2] < 1000 else 'long')
                e1_by_len.setdefault(bkt, []).append(e1)
        done += 1
        if done % 2000 == 0:
            el = time.time() - t0
            print(f'  {done}/{len(tasks)} done, {el:.0f}s, '
                  f'ETA {(el/done)*(len(tasks)-done)/60:.0f}min', flush=True)
    for p in procs: p.join()

    # RL / energy summary
    print(f'\n=== RL far-pair stats ===')
    print(f'  {rl_hits}/{len(tasks)} 序列有远配被优化, 远配总数 {rl_total_far}, '
          f'平均 far_mean 拉拢 {far_improv_sum/max(rl_hits,1):.1f}Å')
    if e1_vals:
        print(f'=== CG energy (E1 kJ/mol) ===')
        print(f'  全体: n={len(e1_vals)} median={np.median(e1_vals):,.0f} '
              f'mean={np.mean(e1_vals):,.0f}')
        for bkt, vals in sorted(e1_by_len.items()):
            print(f'  {bkt}: n={len(vals)} median={np.median(vals):,.0f}')

    # Write npz
    valid = [(i, rid, rl, rc) for i, (rid, rl, rc) in
             enumerate(zip(result_ids, result_lens, result_coords)) if rc is not None]
    n_ok = len(valid)
    print(f'  Success: {n_ok}/{len(tasks)}')
    meta_out = [result_meta[i] if result_meta[i] is not None else (0, 0.0, 0.0, 0.0, 0.0)
                for i, _, _, _ in valid]
    np.savez(args.out,
             ids=np.array([v[1] for v in valid], dtype=object),
             lengths=np.array([v[2] for v in valid], dtype=np.int32),
             coords=np.array([v[3] for v in valid], dtype=object),
             meta=np.array(meta_out, dtype=object))
    sz = os.path.getsize(args.out) / 1e9
    print(f'  Written: {args.out} ({sz:.2f}GB, {n_ok} samples)')
    print(f'  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()