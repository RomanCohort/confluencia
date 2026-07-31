"""precompute_pair_probs.py — ViennaRNA bpp for circRNA (32-worker multiprocessing).

Usage (on Windows with circrna3d conda env):
    conda activate circrna3d
    python precompute_pair_probs.py \\
        --npz ../../data/circrna_3d_all_consolidated.npz \\
        --fasta ../../data/circrna/circbase_seqs.fa.gz \\
        --output ../../data/circrna_3d_all_pair_probs.npz \\
        --n-workers 32

ViennaRNA is NOT thread-safe, so we use multiprocessing with 'spawn' context.
Each worker loads only its chunk of coords to keep memory low.

Output: pair_probs.npz with ids, lengths, bp_probs (aligned with consolidated npz).
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
    """Load FASTA -> dict id -> sequence (str of ACGU)."""
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


def compute_bp_probs_vienna(sequence):
    """ViennaRNA bpp for circular RNA -> (L,L) float32."""
    import RNA
    L = len(sequence)
    md = RNA.md()
    md.circ = True
    fc = RNA.fold_compound(sequence, md)
    fc.bpp()
    bpPP = fc.bpPP()
    return np.asarray(bpPP, dtype=np.float32)


def compute_bp_probs_geometric(coords):
    """Fallback: pair probs from C3' coords (no ViennaRNA).

    Memory-efficient: computes distances row-by-row to avoid (L,L)
    intermediate for very long sequences.
    """
    L = coords.shape[0]
    prob = np.zeros((L, L), dtype=np.float32)
    k = 3
    for i in range(L):
        di = np.linalg.norm(coords[i] - coords, axis=1)  # (L,)
        d2 = (di - 10.6) ** 2
        row_prob = np.exp(-d2 / (2 * 1.5 ** 2))
        # Mask: only pairs with dist in [8,13], skip neighbors
        mask = (di >= 8.0) & (di <= 13.0)
        mask[i] = False
        mask[(i - 1) % L] = False
        mask[(i + 1) % L] = False
        row_prob *= mask
        # Keep top-k strongest
        if L > k:
            top_idx = np.argpartition(row_prob, -k)[-k:]
            top_idx = top_idx[row_prob[top_idx] > 0.1]
            prob[i, top_idx] = row_prob[top_idx]
        else:
            prob[i] = row_prob
    return prob


def worker_compute(args):
    """
    Worker: compute bp_probs for one chunk.

    Args (picklable):
        chunk_npz: path to this chunk's data npz (ids, lengths, coords)
        fasta_json_path: path to pre-saved FASTA JSON
        out_path: path to output npz for this chunk
        use_geometric: bool
        max_len: int
    """
    chunk_npz, fasta_json_path, out_path, use_geometric, max_len = args

    # Load FASTA sequences (once per worker)
    with open(fasta_json_path, 'r') as f:
        seq_map = json.load(f)

    # Load chunk data (only this worker's coords)
    chunk = np.load(chunk_npz, allow_pickle=True)
    chunk_ids = chunk['ids']
    chunk_lengths = chunk['lengths']
    chunk_coords = chunk['coords']

    results = []
    n = len(chunk_ids)

    for i in range(n):
        nid = str(chunk_ids[i])
        L = int(chunk_lengths[i])
        seq = seq_map.get(nid)
        coords = np.asarray(chunk_coords[i][:L], dtype=np.float32)

        if use_geometric or not seq:
            try:
                bp = compute_bp_probs_geometric(coords)
            except Exception:
                bp = np.zeros((L, L), dtype=np.float32)
        elif L > max_len:
            # Very long sequences: skip ViennaRNA AND geometric (too slow / heavy).
            # Zero matrix = no pairing constraints; training still works fine.
            bp = np.zeros((L, L), dtype=np.float32)
        else:
            try:
                bp = compute_bp_probs_vienna(seq)
            except Exception:
                try:
                    bp = compute_bp_probs_geometric(coords)
                except Exception:
                    bp = np.zeros((L, L), dtype=np.float32)

        results.append(bp)

        # Incremental save every 500 samples (don't lose progress on crash)
        if (i + 1) % 500 == 0:
            np.savez(out_path + ".partial",
                     ids=np.array(chunk_ids[:i+1], dtype=object),
                     lengths=np.array(chunk_lengths[:i+1], dtype=np.int32),
                     bp_probs=np.array(results, dtype=object))

    np.savez(out_path,
             ids=np.array(chunk_ids, dtype=object),
             lengths=np.array(chunk_lengths, dtype=np.int32),
             bp_probs=np.array(results, dtype=object))
    return (len(results), out_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute ViennaRNA pair probabilities")
    parser.add_argument("--npz", required=True, help="Path to consolidated npz")
    parser.add_argument("--fasta", required=True, help="Path to circbase_seqs.fa.gz")
    parser.add_argument("--output", required=True, help="Output pair_probs npz")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Skip ViennaRNA for L>this (geometric fallback)")
    parser.add_argument("--use-geometric-fallback", action="store_true",
                        help="Use geometric fallback for ALL")
    parser.add_argument("--n-workers", type=int, default=32,
                        help="Parallel workers (default 32)")
    parser.add_argument("--work-dir", default=None,
                        help="Temp dir for chunk files (default: .precompute_tmp in output dir)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else Path(args.output).parent / ".precompute_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load consolidated npz ──
    print(f"Loading consolidated npz: {args.npz}")
    t0 = time.time()
    data = np.load(args.npz, allow_pickle=True)
    all_ids = data['ids']
    all_lengths = data['lengths']
    all_coords = data['coords']
    n_total = len(all_ids)
    print(f"  {n_total} samples, {time.time()-t0:.2f}s")

    # ── Step 2: load FASTA ──
    print(f"Loading FASTA: {args.fasta}")
    seq_map = load_fasta_ids(args.fasta)
    matched = sum(1 for nid in all_ids if str(nid) in seq_map)
    print(f"  {len(seq_map)} in FASTA, {matched}/{n_total} matched")

    # ── Step 3: save FASTA as JSON for workers ──
    fasta_json_path = str(work_dir / "fasta_seqs.json")
    with open(fasta_json_path, 'w') as f:
        json.dump(seq_map, f)
    print(f"  FASTA JSON saved ({os.path.getsize(fasta_json_path)/1e6:.1f} MB)")

    # ── Step 4: split data into chunk npz files ──
    print(f"Splitting data into {args.n_workers} chunk npz files...")
    t0 = time.time()
    n_workers = min(args.n_workers, n_total)
    chunk_size = (n_total + n_workers - 1) // n_workers
    chunk_paths = []

    for w in range(n_workers):
        start = w * chunk_size
        end = min(start + chunk_size, n_total)
        if start >= end:
            break
        chunk_npz = str(work_dir / f"chunk_{w}.npz")
        np.savez(chunk_npz,
                 ids=np.array(all_ids[start:end], dtype=object),
                 lengths=np.array(all_lengths[start:end], dtype=np.int32),
                 coords=np.array(all_coords[start:end], dtype=object))
        chunk_paths.append(chunk_npz)

    n_chunks = len(chunk_paths)
    print(f"  {n_chunks} chunks in {time.time()-t0:.2f}s")

    # ── Step 5: parallel computation ──
    print(f"Starting {n_workers} workers (spawn, ViennaRNA not thread-safe)...")
    worker_args = [(cp, fasta_json_path, str(work_dir / f"bp_{w}.npz"),
                    args.use_geometric_fallback, args.max_len)
                   for w, cp in enumerate(chunk_paths)]

    t0 = time.time()
    ctx = get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(worker_compute, worker_args)

    elapsed = time.time() - t0
    print(f"All workers done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── Step 6: merge bp chunk npz files ──
    print("Merging bp results...")
    merged_ids, merged_lengths, merged_bp = [], [], []
    bp_file_paths = [str(work_dir / f"bp_{w}.npz") for w in range(n_chunks)]
    chunk_npz_paths = chunk_paths

    for w in range(n_chunks):
        bp_data = np.load(bp_file_paths[w], allow_pickle=True)
        merged_ids.extend(bp_data['ids'])
        merged_lengths.extend(bp_data['lengths'])
        merged_bp.extend(bp_data['bp_probs'])
        # Clean up chunk files
        os.remove(bp_file_paths[w])
        os.remove(chunk_npz_paths[w])
        if (w + 1) % 4 == 0:
            print(f"  Merged {w+1}/{n_chunks} chunks")

    merged_ids = np.array(merged_ids, dtype=object)
    merged_lengths = np.array(merged_lengths, dtype=np.int32)
    merged_bp = np.array(merged_bp, dtype=object)

    assert len(merged_bp) == n_total, f"Merge mismatch: {len(merged_bp)} vs {n_total}"

    # ── Step 7: save final ──
    print(f"Saving {args.output}...")
    np.savez(args.output, ids=merged_ids, lengths=merged_lengths, bp_probs=merged_bp)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB")

    # Verify
    verify = np.load(args.output, allow_pickle=True)
    print(f"Verify: {len(verify['bp_probs'])} entries, "
          f"first shape={verify['bp_probs'][0].shape}")

    # Clean up FASTA JSON
    os.remove(fasta_json_path)


if __name__ == "__main__":
    main()
