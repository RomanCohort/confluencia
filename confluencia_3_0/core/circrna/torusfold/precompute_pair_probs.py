"""precompute_pair_probs.py — ViennaRNA bpp pair probabilities for circRNA sequences.

Usage:
    # Install ViennaRNA first:
    pip install viennarna
    conda install -c bioconda viennarna

    # From consolidated npz (reads IDs directly; only needs FASTA for ViennaRNA mode):
    python precompute_pair_probs.py \\
        --npz ../../data/circrna_3d_all_consolidated.npz \\
        --fasta ../../data/circrna/circbase_seqs.fa.gz \\
        --output ../../data/circrna_3d_all_pair_probs.npz

Modes:
    ViennaRNA (default): real bpp from secondary structure prediction.
        L>1000 uses geometric fallback to keep runtime reasonable.
    --use-geometric-fallback: skip ViennaRNA entirely, infer from C3' coords.

Output:
    pair_probs.npz with ids, lengths, bp_probs (aligned with consolidated npz).
"""

import os
import sys
import argparse
import time
import numpy as np
import gzip
from collections import OrderedDict


def load_fasta_ids(fasta_path):
    """Load FASTA -> dict id -> sequence (str of ACGU)."""
    seq_map = OrderedDict()
    cur_id, cur_seq = None, ""
    with gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r") as f:
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
    """Compute ViennaRNA base-pair probabilities for a circular RNA.

    Returns (L,L) float32 symmetric matrix of pair probabilities.
    """
    import RNA
    L = len(sequence)
    md = RNA.md()
    md.circ = True
    fc = RNA.fold_compound(sequence, md)
    fc.bpp()
    bpPP = fc.bpPP()
    return np.asarray(bpPP, dtype=np.float32)


def compute_bp_probs_geometric(coords):
    """Fallback: infer pair probs from C3' coords.

    Heuristic: base pairs are ~10.6 A C3'-C3' apart.
    Returns (L,L) float32 matrix with 0-1 probabilities.
    """
    L = coords.shape[0]
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    mask = (dists >= 8.0) & (dists <= 13.0)
    np.fill_diagonal(mask, False)
    for i in range(L):
        mask[i, (i - 1) % L] = False
        mask[i, (i + 1) % L] = False
        mask[(i - 1) % L, i] = False
        mask[(i + 1) % L, i] = False
        mask[i, i] = False

    prob = np.exp(-((dists - 10.6) ** 2) / (2 * 1.5 ** 2)) * mask.astype(np.float32)

    out = np.zeros_like(prob)
    k = 3
    for i in range(L):
        row = prob[i]
        top_k_idx = np.argpartition(row, -k)[-k:] if k < L else range(L)
        top_k_idx = top_k_idx[row[top_k_idx] > 0.1]
        out[i, top_k_idx] = row[top_k_idx]

    np.fill_diagonal(out, 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Precompute ViennaRNA pair probabilities")
    parser.add_argument("--npz", required=True,
                        help="Path to consolidated npz (reads ids/lengths/coords from here)")
    parser.add_argument("--fasta", required=True, help="Path to circbase_seqs.fa.gz")
    parser.add_argument("--output", required=True, help="Output pair_probs npz path")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Skip ViennaRNA for sequences > this length (geometric fallback)")
    parser.add_argument("--use-geometric-fallback", action="store_true",
                        help="Use geometric fallback for ALL sequences (no ViennaRNA needed)")
    args = parser.parse_args()

    # Load FASTA sequences
    print(f"Loading FASTA: {args.fasta}")
    seq_map = load_fasta_ids(args.fasta)
    print(f"  {len(seq_map)} sequences in FASTA")

    # Load consolidated npz for IDs, lengths, and coords (for geometric fallback)
    print(f"Loading consolidated npz: {args.npz}")
    t0_npz = time.time()
    data = np.load(args.npz, allow_pickle=True)
    all_ids = data['ids']
    lengths_arr = data['lengths']
    # Coords for geometric fallback; may be stored as object array
    coords_arr = data['coords']
    print(f"  {len(all_ids)} IDs, {time.time()-t0_npz:.2f}s")

    has_vienna = False
    try:
        import RNA
        has_vienna = True
        print("ViennaRNA available")
    except ImportError:
        print("ViennaRNA NOT installed; using geometric fallback for all")

    bp_probs_list = []
    t0_all = time.time()
    skipped_long = 0
    matched = 0

    for idx in range(len(all_ids)):
        nid = str(all_ids[idx])
        L = int(lengths_arr[idx])
        seq = seq_map.get(nid)

        if seq is not None:
            matched += 1

        if idx % 2000 == 0:
            print(f"  [{idx}/{len(all_ids)}] {nid} L={L} elapsed={time.time()-t0_all:.1f}s")

        # Decide method
        if args.use_geometric_fallback or not has_vienna:
            bp = _bp_geometric(coords_arr, idx, L)
        elif L > args.max_len:
            skipped_long += 1
            bp = _bp_geometric(coords_arr, idx, L)
        else:
            try:
                bp = compute_bp_probs_vienna(seq)
            except Exception as e:
                print(f"  ViennaRNA failed {nid}: {e}; geometric fallback")
                bp = _bp_geometric(coords_arr, idx, L)

        bp_probs_list.append(bp)

    print(f"\nDone: {len(bp_probs_list)} computed, {matched} matched FASTA, "
          f"{skipped_long} long skipped (geometric fallback)")
    print(f"Total time: {time.time()-t0_all:.1f}s ({(time.time()-t0_all)/60:.1f}min)")

    print(f"Saving {args.output}...")
    np.savez(
        args.output,
        ids=np.array(all_ids, dtype=object),
        lengths=lengths_arr,
        bp_probs=np.array(bp_probs_list, dtype=object),
    )
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB")


def _bp_geometric(coords_arr, idx, L):
    """Load coords and compute geometric fallback bpp."""
    c = np.asarray(coords_arr[idx][:L], dtype=np.float32)
    return compute_bp_probs_geometric(c)


if __name__ == "__main__":
    main()
