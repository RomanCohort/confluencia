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
from multiprocessing import Process, Queue

import numpy as np

SCHEME2_ROOT = r'C:\Users\颜子壹\TorusFold-scheme2-rl'
if SCHEME2_ROOT not in sys.path:
    sys.path.insert(0, SCHEME2_ROOT + r'\src')


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
            if far_pairs:
                opt_p, _, _ = optimize_far_pairs(
                    cg, seq, far_pairs, stem_blocks,
                    policy_path=None, n_simulations=50,
                    dpo_weight=5.0, dpo_simulate=True,
                )
                cg = opt_p
            if np.isnan(cg).any() or np.isinf(cg).any():
                raise RuntimeError('NaN/Inf')
            out_q.put((idx, np.asarray(cg, dtype=np.float32)))
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
    done = 0; t0 = time.time()
    while done < len(tasks):
        res = out_q.get()
        if len(res) == 3:
            idx, _, err = res
            print(f'  [{done}/{len(tasks)}] err {idx}: {err}', flush=True)
        else:
            idx, c = res
            orig = tasks[idx]
            result_ids[idx] = orig[1]
            result_lens[idx] = orig[2]
            result_coords[idx] = c
        done += 1
        if done % 2000 == 0:
            el = time.time() - t0
            print(f'  {done}/{len(tasks)} done, {el:.0f}s, '
                  f'ETA {(el/done)*(len(tasks)-done)/60:.0f}min', flush=True)
    for p in procs: p.join()

    # Write npz
    valid = [(i, rid, rl, rc) for i, (rid, rl, rc) in
             enumerate(zip(result_ids, result_lens, result_coords)) if rc is not None]
    n_ok = len(valid)
    print(f'  Success: {n_ok}/{len(tasks)}')
    np.savez(args.out,
             ids=np.array([v[1] for v in valid], dtype=object),
             lengths=np.array([v[2] for v in valid], dtype=np.int32),
             coords=np.array([v[3] for v in valid], dtype=object))
    sz = os.path.getsize(args.out) / 1e9
    print(f'  Written: {args.out} ({sz:.2f}GB, {n_ok} samples)')
    print(f'  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()