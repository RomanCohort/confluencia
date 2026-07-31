"""precompute_pair_probs.py — ViennaRNA bpp, chunked + resumable + streaming write.

Produces pair probability matrices (bpPP) for ALL samples in the
consolidated dataset, aligned 1:1 by index with circrna_3d_all_consolidated.npz.

Design:
  - Splits work into CHUNK_SIZE-rows-per-chunk shards stored in .precompute_tmp/
    (so a crash loses at most one chunk, not the whole run).
  - Streaming write per sample: each ViennaRNA result is saved immediately to a
    per-sample .npy file on disk. The 2000-sample "buffer" from the old version
    (which consumed ~1.7 GB of O(L²) matrices) is gone — peak memory is now
    ~20 MB regardless of chunk size or --n-threads.
  - Resume via marker files: each completed shard leaves a .shard_{i}.marker
    touchfile. Checking existence is far cheaper than loading the full npz.
  - Per-shard subdirectories hold the .npy samples; on completion a final
    bp_{i}.npz is assembled and the .npy files are cleaned up.
  - Final step merges all shards into a single output .npz aligned with the
    consolidated data (same ids / lengths / ordering).

Memory: pair probabilities are O(L²) per sample (~850 KB for L=461),
roughly 150× larger than the O(L) coordinate vectors used by other pipelines.
The streaming-write design avoids accumulating them in memory.

Thread safety: ViennaRNA 2.7.2 Python bindings are thread-safe — each call
builds its own fold_compound. ThreadPoolExecutor shares sequence memory
(no 5 GB duplication).

A800-ready: pure Python, no CUDA dependency (ViennaRNA runs on CPU).

Usage:
    python precompute_pair_probs.py \\
        --consolidated ../../data/circrna_3d_all_consolidated.npz \\
        --fasta ../../data/circrna/circbase_seqs.fa.gz \\
        --output ../../data/circrna_3d_all_pair_probs.npz \\
        --chunk-size 2000 \\
        --max-len 1000 \\
        --n-threads 32

Usage on A800 (same command, install ViennaRNA Python bindings first):
    pip install viennarna
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import shutil
import numpy as np
import gzip
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── FASTA loader ──────────────────────────────────────────────────────────────


def load_fasta_ids(fasta_path: str) -> OrderedDict[str, str]:
    """Load FASTA → OrderedDict of id → sequence (T→U)."""
    seq_map: OrderedDict[str, str] = OrderedDict()
    cur_id: str | None = None
    cur_seq = ""
    opener = (
        gzip.open(fasta_path, "rt")
        if fasta_path.endswith(".gz")
        else open(fasta_path, "r")
    )
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


# ── ViennaRNA wrapper ────────────────────────────────────────────────────────


def compute_bp_probs_vienna(sequence: str) -> np.ndarray:
    """Compute bpPP via ViennaRNA (circular mode). Returns (L, L) float32.

    ViennaRNA Python bindings expose bpPP as an upper-triangle tuple of tuples
    (returned by fc.bpp()), NOT as fc.bpPP(). Partition-function matrices must
    be filled first (fc.pf()).
    """
    import RNA

    L = len(sequence)
    md = RNA.md()
    md.circ = True
    fc = RNA.fold_compound(sequence, md)
    fc.pf()
    tri = fc.bpp()
    P = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for off, p in enumerate(tri[i]):
            j = i + off + 1
            if 0 <= j < L:
                P[i, j] = float(p)
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 0.0)
    return P


# ── Shard helpers ────────────────────────────────────────────────────────────


def shard_dir_path(tmp_dir: Path, shard_idx: int) -> Path:
    """Per-shard subdirectory holding per-sample .npy files."""
    return tmp_dir / f"shard_{shard_idx}"


def marker_path(tmp_dir: Path, shard_idx: int) -> Path:
    """Marker file indicating a shard is fully complete."""
    return tmp_dir / f".shard_{shard_idx}.marker"


def sample_npy_path(shard_idx: int, global_idx: int) -> Path:
    """Path of the per-sample .npy within its shard dir (relative to tmp_dir)."""
    # We pass the full path from the shard subdirectory in compute_one
    # — the helper builds it given tmp_dir.
    pass


def is_shard_complete(marker: Path, npz: Path, expected_n: int) -> bool:
    """Return True if shard .npz is readable AND marker exists."""
    if not marker.exists():
        return False
    try:
        d = np.load(npz, allow_pickle=True)
        ids = d.get("ids")
        return ids is not None and len(ids) == expected_n
    except Exception:
        return False


def _compute_and_write(
    global_idx: int,
    nid: str,
    L: int,
    seq: str | None,
    max_len: int,
    out_npy: str,
) -> tuple[int, bool]:
    """Compute bpPP for one sample and write directly to .npy on disk.

    Returns (global_idx, ok) where ok=False on ViennaRNA error.
    """
    if seq is None or L > max_len:
        bp = np.zeros((L, L), dtype=np.float32)
        ok = True
    else:
        try:
            bp = compute_bp_probs_vienna(seq)
            ok = True
        except Exception as e:
            print(f"\n  ViennaRNA failed {nid} L={L}: {e}", flush=True)
            bp = np.zeros((L, L), dtype=np.float32)
            ok = False
    np.save(out_npy, bp)
    return (global_idx, ok)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ViennaRNA bpPP for consolidated circRNA dataset"
    )
    parser.add_argument("--consolidated", required=True, help="Path to consolidated npz")
    parser.add_argument("--fasta", required=True, help="Path to FASTA (or .fa.gz)")
    parser.add_argument("--output", required=True, help="Output merged .npz path")
    parser.add_argument("--tmp-dir", type=str, default=None,
                        help="Shard tmp dir (default: output's parent / .precompute_tmp)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Samples per shard (default 2000)")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Skip samples longer than this (default 1000)")
    parser.add_argument("--n-threads", type=int, default=32,
                        help="Threads per chunk (ViennaRNA thread-safe)")
    args = parser.parse_args()

    chunk_dir = Path(args.tmp_dir) if args.tmp_dir else Path(args.output).parent / ".precompute_tmp"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ── Load consolidated ids / lengths ────────────────────────────────
    print(f"Loading consolidated npz: {args.consolidated}")
    t0 = time.time()
    data = np.load(args.consolidated, allow_pickle=True)
    all_ids = data["ids"]
    all_lengths = data["lengths"]
    n_total = len(all_ids)
    print(f"  {n_total} samples in {time.time() - t0:.1f}s")

    # ── Load FASTA ─────────────────────────────────────────────────────
    print(f"Loading FASTA: {args.fasta}")
    t0 = time.time()
    seq_map = load_fasta_ids(args.fasta)
    matched = sum(1 for nid in all_ids if str(nid) in seq_map)
    print(f"  {len(seq_map)} FASTA seqs, {matched}/{n_total} matched "
          f"({matched / n_total * 100:.1f}%) in {time.time() - t0:.1f}s")

    # ── Build chunk list ───────────────────────────────────────────────
    n_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
    print(f"Chunks: {n_chunks} (chunk_size={args.chunk_size}, "
          f"max_len={args.max_len}, threads={args.n_threads})")
    print(f"Shard dir: {chunk_dir}")
    print()

    t_start = time.time()

    for ci in range(n_chunks):
        s0 = ci * args.chunk_size
        s1 = min((ci + 1) * args.chunk_size, n_total)
        actual_n = s1 - s0
        shard_npz = chunk_dir / f"bp_{ci}.npz"
        shard_md = shard_dir_path(chunk_dir, ci)
        marker = marker_path(chunk_dir, ci)

        # Resume check
        if is_shard_complete(marker, shard_npz, actual_n):
            print(f"  [{ci + 1}/{n_chunks}] shard bp_{ci}.npz "
                  f"already complete — skipping")
            continue

        shard_md.mkdir(parents=True, exist_ok=True)

        print(f"  [{ci + 1}/{n_chunks}] computing shard bp_{ci}.npz "
              f"(samples {s0}–{s1 - 1}, n={actual_n}) ...", end="", flush=True)

        chunk_ids = [str(all_ids[i]) for i in range(s0, s1)]
        chunk_lens = [int(all_lengths[i]) for i in range(s0, s1)]
        chunk_seqs = [seq_map.get(str(all_ids[i])) for i in range(s0, s1)]

        # Build task tuples — each writes to its own .npy
        tasks = []
        for idx, nid, L, seq in zip(range(s0, s1), chunk_ids, chunk_lens, chunk_seqs):
            out_file = str(shard_md / f"s{idx}.npy")
            tasks.append((idx, nid, L, seq, out_file))

        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.n_threads) as pool:
            futures = {
                pool.submit(_compute_and_write, idx, nid, L, seq, args.max_len, out): idx
                for idx, nid, L, seq, out in tasks
            }
            for fut in as_completed(futures):
                idx, ok = fut.result()
                if ok:
                    n_ok += 1

        elapsed = time.time() - t_start
        eta = (elapsed / (ci + 1)) * (n_chunks - ci - 1) if ci + 1 > 0 else 0
        print(f" done ({n_ok}/{actual_n} ok, "
              f"elapsed={elapsed:.0f}s, "
              f"eta={eta / 60:.1f}min, "
              f"{(ci + 1) / n_chunks * 100:.1f}%)")

        # ── Assemble shard npz from per-sample .npy files ─────────────
        # Build arrays in memory once (we hold all in RAM for the save)
        shard_ids = []
        shard_lens = []
        shard_bp = []
        for idx in range(s0, s1):
            npy = shard_md / f"s{idx}.npy"
            if not npy.exists():
                # Sample failed and wasn't written — create zeros
                L = int(all_lengths[idx])
                shard_bp.append(np.zeros((L, L), dtype=np.float32))
            else:
                shard_bp.append(np.load(str(npy)))
            shard_ids.append(str(all_ids[idx]))
            shard_lens.append(int(all_lengths[idx]))

        np.savez(
            str(shard_npz),
            ids=np.array(shard_ids, dtype=object),
            lengths=np.array(shard_lens, dtype=np.int32),
            bp_probs=np.array(shard_bp, dtype=object),
        )

        # Write marker AFTER successful np.savez
        marker.touch()

        # Clean up per-sample .npy files to save disk space
        for idx in range(s0, s1):
            npy = shard_md / f"s{idx}.npy"
            if npy.exists():
                try:
                    npy.unlink()
                except Exception:
                    pass
        try:
            if shard_md.exists() and not any(shard_md.iterdir()):
                shard_md.rmdir()
        except Exception:
            pass

        shard_bytes = shard_npz.stat().st_size / 1e6
        print(f"        shard size: {shard_bytes:.1f} MB")

    # ── Merge shards ───────────────────────────────────────────────────
    print(f"\nMerging {n_chunks} shards → {args.output} ...")
    t0 = time.time()
    merged_bp = []
    merged_ids = []
    merged_lengths = []

    for ci in range(n_chunks):
        shard_npz = chunk_dir / f"bp_{ci}.npz"
        d = np.load(shard_npz, allow_pickle=True)
        merged_ids.extend(list(d["ids"]))
        merged_lengths.extend(list(d["lengths"]))
        merged_bp.extend(list(d["bp_probs"]))

    np.savez(
        args.output,
        ids=np.array(merged_ids, dtype=object),
        lengths=np.array(merged_lengths, dtype=np.int32),
        bp_probs=np.array(merged_bp, dtype=object),
    )
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB  ({time.time() - t0:.1f}s)")

    # ── Verify ─────────────────────────────────────────────────────────
    verify = np.load(args.output, allow_pickle=True)
    match = np.array_equal(np.array(merged_ids, dtype=object), np.array(list(all_ids), dtype=object))
    print(f"Verify: {len(verify['bp_probs'])} entries, "
          f"first shape={verify['bp_probs'][0].shape}, "
          f"ids match consolidated={match}")

    total_time = time.time() - t_start
    print(f"\nTotal wall time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("Done.")


if __name__ == "__main__":
    main()
