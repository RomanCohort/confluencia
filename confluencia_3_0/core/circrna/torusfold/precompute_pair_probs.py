"""precompute_pair_probs.py — ViennaRNA bpp, 32-worker parallel.

Design — each worker loads only 1/32 of data (no shared large arrays):
  Main: split coords (496MB) and FASTA (495MB) into 32 small chunks.
  Worker: loads own chunk files (~30MB total) + compute bp via ViennaRNA.

Memory per worker: < 200 MB. Total peak: ~6 GB + main process ~500 MB.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import gzip
from collections import OrderedDict
from multiprocessing import Pool, get_context
from pathlib import Path


def load_fasta_ids(fasta_path):
    """FASTA -> list of (id, sequence)."""
    entries = []
    cur_id, cur_seq = None, ""
    opener = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")
    with opener as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    entries.append((cur_id, cur_seq))
                cur_id = line.strip()[1:].split("|")[0]
                cur_seq = ""
            else:
                cur_seq += line.strip().upper().replace("T", "U")
        if cur_id is not None:
            entries.append((cur_id, cur_seq))
    return entries


def compute_bp_probs_vienna(sequence):
    import RNA
    md = RNA.md()
    md.circ = True
    fc = RNA.fold_compound(sequence, md)
    fc.bpp()
    return np.asarray(fc.bpPP(), dtype=np.float32)


def worker_chunk(args):
    """
    Worker loads only its own small chunk files.
    Args: (w_dir, w_id, id_list, lengths_list, out_path, max_len)
    """
    w_dir, w_id, id_list, lengths_list, out_path, max_len = args
    w_dir = Path(w_dir)

    # Load this worker's FASTA seq dict (small: ~15MB)
    fasta_json = w_dir / f"fasta_{w_id}.json"
    with open(fasta_json, 'r') as f:
        seq_map = json.load(f)

    # Load this worker's coords chunk (small: ~15MB)
    coords_npy = w_dir / f"coords_{w_id}.npy"
    coords_chunk = np.load(coords_npy, mmap_mode='r')

    n = len(id_list)
    ids_out, lengths_out, bp_out = [], [], []

    for i in range(n):
        nid = id_list[i]
        L = lengths_list[i]
        seq = seq_map.get(nid)
        coords = coords_chunk[i, :L]  # zero-copy slice

        if seq is None or L > max_len:
            bp = np.zeros((L, L), dtype=np.float32)
        else:
            try:
                bp = compute_bp_probs_vienna(seq)
            except Exception:
                bp = np.zeros((L, L), dtype=np.float32)

        ids_out.append(nid)
        lengths_out.append(L)
        bp_out.append(bp)

    np.savez(out_path,
             ids=np.array(ids_out, dtype=object),
             lengths=np.array(lengths_out, dtype=np.int32),
             bp_probs=np.array(bp_out, dtype=object))

    return (w_id, n, bp_out[0].shape if bp_out else (0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=32)
    args = parser.parse_args()

    # ── Step 1: load consolidated npz ──
    print(f"Loading npz: {args.npz}")
    t0 = time.time()
    data = np.load(args.npz, allow_pickle=True)
    all_ids = data['ids']
    all_lengths = data['lengths']
    all_coords = data['coords']  # (N, maxL, 3) float32
    n_total = len(all_ids)
    maxL = all_coords.shape[1]
    print(f"  {n_total} samples, maxL={maxL}, {time.time()-t0:.1f}s, "
          f"coords={all_coords.nbytes/1e6:.0f}MB")

    # ── Step 2: load FASTA as list ──
    print(f"Loading FASTA: {args.fasta}")
    fasta_entries = load_fasta_ids(args.fasta)
    seq_map = OrderedDict(fasta_entries)
    matched = sum(1 for nid in all_ids if str(nid) in seq_map)
    print(f"  {len(seq_map)} FASTA seqs, {matched}/{n_total} matched")

    # ── Step 3: split into worker chunks (small files) ──
    work_dir = Path(args.output).parent / ".precompute_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(args.n_workers, n_total)
    chunk_size = (n_total + n_workers - 1) // n_workers

    print(f"Splitting {n_total} samples into {n_workers} chunks...")
    worker_args = []
    for w in range(n_workers):
        s = w * chunk_size
        e = min((w + 1) * chunk_size, n_total)
        if s >= e:
            break

        # coords chunk: (chunk_size, maxL, 3) — small
        coords_chunk = all_coords[s:e, :].copy()
        np.save(work_dir / f"coords_{w}.npy", coords_chunk)

        # FASTA chunk: only seqs for this chunk's IDs
        chunk_ids = [str(all_ids[i]) for i in range(s, e)]
        chunk_lengths = [int(all_lengths[i]) for i in range(s, e)]
        chunk_seqs = {nid: seq_map[nid] for nid in chunk_ids}
        with open(work_dir / f"fasta_{w}.json", 'w') as f:
            json.dump(chunk_seqs, f)

        out_path = str(work_dir / f"bp_{w}.npz")
        worker_args.append((
            str(work_dir), w, chunk_ids, chunk_lengths, out_path, args.max_len
        ))

    # Verify file sizes
    total_size_mb = sum(os.path.getsize(p) for p in work_dir.glob("coords_*.npy") for p in [p]) / 1e6
    print(f"  chunk files: {total_size_mb:.0f}MB coords total, "
          f"each chunk ~{total_size_mb/n_workers:.1f}MB")

    # ── Step 4: parallel compute ──
    print(f"Starting {n_workers} workers (spawn)...")
    t0 = time.time()
    ctx = get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(worker_chunk, worker_args)

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Step 5: merge ──
    print("Merging...")
    merged_ids, merged_lengths, merged_bp = [], [], []
    for wid, count, shape in results:
        out_path = work_dir / f"bp_{wid}.npz"
        chunk = np.load(out_path, allow_pickle=True)
        merged_ids.extend(chunk['ids'])
        merged_lengths.extend(chunk['lengths'])
        merged_bp.extend(chunk['bp_probs'])
        os.remove(out_path)
        if (wid + 1) % 4 == 0:
            print(f"  {wid+1}/{n_workers}")

    merged_ids = np.array(merged_ids, dtype=object)
    merged_lengths = np.array(merged_lengths, dtype=np.int32)
    merged_bp = np.array(merged_bp, dtype=object)
    assert len(merged_bp) == n_total, f"Mismatch: {len(merged_bp)} vs {n_total}"

    # ── Step 6: save final ──
    print(f"Saving {args.output}...")
    np.savez(args.output, ids=merged_ids, lengths=merged_lengths, bp_probs=merged_bp)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB")

    verify = np.load(args.output, allow_pickle=True)
    print(f"Verify: {len(verify['bp_probs'])} entries, first shape={verify['bp_probs'][0].shape}")

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
