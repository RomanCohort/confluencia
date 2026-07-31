"""precompute_pair_probs.py — ViennaRNA bpp for circRNA (32-worker).

Usage (Windows, circrna3d conda env):
    python precompute_pair_probs.py \\
        --npz ../../data/circrna_3d_all_consolidated.npz \\
        --fasta ../../data/circrna/circbase_seqs.fa.gz \\
        --output ../../data/circrna_3d_all_pair_probs.npz \\
        --n-workers 32

Design:
  - Main process loads FASTA + consolidated npz, matches sequences to IDs.
  - Builds worker payload: list of (nid, L, sequence_or_None, coords_array).
  - Worker receives (nid, L, seq, coords) directly — no npz/pickle loading.
  - L>max_len -> zero matrix (skip ViennaRNA + geometric).
  - ViennaRNA unavailable / seq missing -> geometric fallback.
  - Incremental save every 500 samples per worker (crash recovery).
"""

import os
import sys
import argparse
import time
import numpy as np
import gzip
from collections import OrderedDict
from multiprocessing import Pool, get_context
from pathlib import Path


# ── Sequence loading ─────────────────────────────────────────────

def load_fasta_ids(fasta_path):
    """FASTA -> dict id -> sequence (ACGU)."""
    seq_map = OrderedDict()
    cur_id, cur_seq = None, ""
    opener = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")
    with opener as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    seq_map[cur_id] = cur_seq
                cur_id = line.strip()[1:].split("|")[0]
                cur_seq = ""
            else:
                cur_seq += line.strip().upper().replace("T", "U")
        if cur_id is not None:
            seq_map[cur_id] = cur_seq
    return seq_map


# ── Bpp computation ─────────────────────────────────────────────

def compute_bp_probs_vienna(sequence):
    """ViennaRNA bpp for circular RNA -> (L,L) float32."""
    import RNA
    L = len(sequence)
    md = RNA.md()
    md.circ = True
    fc = RNA.fold_compound(sequence, md)
    fc.bpp()
    return np.asarray(fc.bpPP(), dtype=np.float32)


def compute_bp_probs_geometric(coords):
    """Memory-efficient geometric fallback: compute row-by-row."""
    L = coords.shape[0]
    prob = np.zeros((L, L), dtype=np.float32)
    k = 3
    for i in range(L):
        di = np.linalg.norm(coords[i] - coords, axis=1)
        row_prob = np.exp(-((di - 10.6) ** 2) / (2 * 1.5 ** 2))
        mask = (di >= 8.0) & (di <= 13.0)
        mask[i] = False
        mask[(i - 1) % L] = False
        mask[(i + 1) % L] = False
        row_prob *= mask
        if L > k:
            top_idx = np.argpartition(row_prob, -k)[-k:]
            top_idx = top_idx[row_prob[top_idx] > 0.1]
            prob[i, top_idx] = row_prob[top_idx]
        else:
            prob[i] = row_prob
    return prob


# ── Worker ───────────────────────────────────────────────────────

def worker_chunk(args):
    """
    Worker receives pre-built payload directly:
        worker_id: int
        items: list of (nid, L, seq_or_None, coords_array)  # picklable (np.ndarray + str)
        out_path: str (where to save chunk results)
        use_geometric: bool
        max_len: int
    """
    worker_id, items, out_path, use_geometric, max_len = args

    ids_out, lengths_out, bp_out = [], [], []
    n = len(items)

    for i, (nid, L, seq, coords) in enumerate(items):
        if use_geometric or seq is None:
            try:
                bp = compute_bp_probs_geometric(coords)
            except Exception:
                bp = np.zeros((L, L), dtype=np.float32)
        elif L > max_len:
            bp = np.zeros((L, L), dtype=np.float32)
        else:
            try:
                bp = compute_bp_probs_vienna(seq)
            except Exception:
                try:
                    bp = compute_bp_probs_geometric(coords)
                except Exception:
                    bp = np.zeros((L, L), dtype=np.float32)

        ids_out.append(nid)
        lengths_out.append(L)
        bp_out.append(bp)

        if (i + 1) % 500 == 0:
            np.savez(out_path + ".partial",
                     ids=np.array(ids_out, dtype=object),
                     lengths=np.array(lengths_out, dtype=np.int32),
                     bp_probs=np.array(bp_out, dtype=object))

    np.savez(out_path,
             ids=np.array(ids_out, dtype=object),
             lengths=np.array(lengths_out, dtype=np.int32),
             bp_probs=np.array(bp_out, dtype=object))
    return (worker_id, n, out_path)


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--use-geometric-fallback", action="store_true")
    parser.add_argument("--n-workers", type=int, default=32)
    args = parser.parse_args()

    # Step 1: load consolidated npz
    print(f"Loading npz: {args.npz}")
    t0 = time.time()
    data = np.load(args.npz, allow_pickle=True)
    all_ids = data['ids']
    all_lengths = data['lengths']
    all_coords = data['coords']  # object array of (L,3) float32
    n_total = len(all_ids)
    print(f"  {n_total} samples, {time.time()-t0:.1f}s")

    # Step 2: load FASTA
    print(f"Loading FASTA: {args.fasta}")
    seq_map = load_fasta_ids(args.fasta)
    matched = sum(1 for nid in all_ids if str(nid) in seq_map)
    print(f"  {len(seq_map)} FASTA seqs, {matched}/{n_total} matched")

    # Step 3: build worker payloads (no npz/pickle — plain Python objects)
    print("Building worker payloads...")
    n_workers = min(args.n_workers, n_total)
    chunk_size = (n_total + n_workers - 1) // n_workers

    items = []
    for idx in range(n_total):
        nid = str(all_ids[idx])
        L = int(all_lengths[idx])
        seq = seq_map.get(nid)
        coords = np.asarray(all_coords[idx][:L], dtype=np.float32)
        items.append((nid, L, seq, coords))

    print(f"  {len(items)} items built ({sys.getsizeof(items) / 1e6:.1f} MB list)")

    # Step 4: split into worker chunks
    work_dir = Path(args.output).parent / ".precompute_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for w in range(n_workers):
        start = w * chunk_size
        end = min(start + chunk_size, n_total)
        if start >= end:
            break
        chunk = items[start:end]
        out_path = str(work_dir / f"bp_{w}.npz")
        worker_args.append((w, chunk, out_path, args.use_geometric_fallback, args.max_len))

    print(f"  {len(worker_args)} worker chunks")

    # Step 5: parallel compute
    print(f"Starting {n_workers} workers (spawn)...")
    t0 = time.time()
    ctx = get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(worker_chunk, worker_args)

    elapsed = time.time() - t0
    print(f"All workers done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Step 6: merge
    print("Merging...")
    merged_ids, merged_lengths, merged_bp = [], [], []
    for wid, count, out_path in results:
        chunk_data = np.load(out_path, allow_pickle=True)
        merged_ids.extend(chunk_data['ids'])
        merged_lengths.extend(chunk_data['lengths'])
        merged_bp.extend(chunk_data['bp_probs'])
        os.remove(out_path)
        # Also remove partial files
        partial = out_path + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        if (wid + 1) % 4 == 0:
            print(f"  {wid+1}/{n_workers}")

    merged_ids = np.array(merged_ids, dtype=object)
    merged_lengths = np.array(merged_lengths, dtype=np.int32)
    merged_bp = np.array(merged_bp, dtype=object)
    assert len(merged_bp) == n_total, f"Mismatch: {len(merged_bp)} vs {n_total}"

    # Step 7: save
    print(f"Saving {args.output}...")
    np.savez(args.output, ids=merged_ids, lengths=merged_lengths, bp_probs=merged_bp)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB")

    verify = np.load(args.output, allow_pickle=True)
    print(f"Verify: {len(verify['bp_probs'])} entries, first shape={verify['bp_probs'][0].shape}")

    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
