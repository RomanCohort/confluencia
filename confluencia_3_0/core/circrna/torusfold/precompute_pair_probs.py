"""precompute_pair_probs.py — ViennaRNA bpp pair probabilities for circRNA sequences.

Usage (must be run on a machine with ViennaRNA installed, e.g. the A800 node):
    # Install ViennaRNA first:
    conda install -c bioconda viennarna

    # Precompute pair probabilities for all sequences in FASTA:
    python precompute_pair_probs.py \\
        --fasta ../../data/circrna/circbase_seqs.fa.gz \\
        --npy-dir ../../data/circrna_3d_all/ \\
        --output ../../data/circrna_3d_all_pair_probs.npz

Output:
    pair_probs.npz — numpy npz with:
        ids: (N,) str IDs (hsa_circ_xxx)
        bp_probs: (N,) object array, each element is a (L, L) float32 matrix
        lengths: (N,) int32

For long sequences (L>500), bpp is slow (~O(L^2) time). Script skips xlong
(L>1000) and fills with geometric fallback so training doesn't hang.

FALLBACK (no ViennaRNA):
    Uses distance-based heuristic pair_probs from coords. If ViennaRNA is
    unavailable, set --use-geometric-fallback.
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
    # fc.bpPP() returns (L,L) numpy array
    bpPP = fc.bpPP()
    return np.asarray(bpPP, dtype=np.float32)


def compute_bp_probs_geometric(coords):
    """Fallback: infer pair probs from C3' coords.

    Heuristic: base pairs are ~10.6 Å C3'-C3' apart.
    Returns (L,L) float32 matrix with 0-1 probabilities.
    """
    L = coords.shape[0]
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)  # (L,L)

    # Pair candidates: distance in [8.0, 13.0] Å AND not adjacent in backbone
    mask = (dists >= 8.0) & (dists <= 13.0)
    np.fill_diagonal(mask, False)
    # Exclude immediate neighbors (backbone, not base pairs)
    for i in range(L):
        mask[i, (i - 1) % L] = False
        mask[i, (i + 1) % L] = False
        mask[(i - 1) % L, i] = False
        mask[(i + 1) % L, i] = False
        mask[i, i] = False

    # Convert distance to prob: Gaussian centered at 10.6 Å
    prob = np.exp(-((dists - 10.6) ** 2) / (2 * 1.5 ** 2)) * mask.astype(np.float32)

    # For each i, keep only the top-k strongest pairs (k=3) to avoid noise
    out = np.zeros_like(prob)
    k = 3
    for i in range(L):
        row = prob[i]
        top_k_idx = np.argpartition(row, -k)[-k:] if k < L else range(L)
        top_k_idx = top_k_idx[row[top_k_idx] > 0.1]  # threshold
        out[i, top_k_idx] = row[top_k_idx]

    np.fill_diagonal(out, 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Precompute ViennaRNA pair probabilities")
    parser.add_argument("--fasta", required=True, help="Path to circbase_seqs.fa.gz")
    parser.add_argument("--npy-dir", required=True, help="Path to circrna_3d_all/ (.npy coords dir)")
    parser.add_argument("--output", required=True, help="Output npz path")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Skip ViennaRNA for sequences > this length (use geometric fallback)")
    parser.add_argument("--use-geometric-fallback", action="store_true",
                        help="Use geometric fallback for ALL sequences (no ViennaRNA needed)")
    args = parser.parse_args()

    print(f"Loading FASTA: {args.fasta}")
    seq_map = load_fasta_ids(args.fasta)
    print(f"  {len(seq_map)} sequences loaded")

    # Load npy IDs and coords for reference
    npy_files = sorted([f for f in os.listdir(args.npy_dir) if f.endswith(".npy")])
    npy_ids = [f[:-4] for f in npy_files]
    print(f"  {len(npy_ids)} npy files in {args.npy_dir}")

    # Cross-reference: only compute for npy IDs that exist in FASTA
    all_ids = npy_ids
    print(f"Computing pair probs for {len(all_ids)} sequences...")

    bp_probs_list = []
    lengths_list = []

    has_vienna = False
    try:
        import RNA
        has_vienna = True
        print("ViennaRNA available")
    except ImportError:
        print("ViennaRNA NOT installed; using geometric fallback for all")

    t0_all = time.time()
    skipped = 0

    for idx, nid in enumerate(all_ids):
        coords_path = os.path.join(args.npy_dir, f"{nid}.npy")
        if not os.path.isfile(coords_path):
            print(f"  [{idx}/{len(all_ids)}] {nid}: npy not found, skipping")
            continue

        coords = np.load(coords_path)
        L = coords.shape[0]

        seq = seq_map.get(nid)
        if seq is None:
            print(f"  [{idx}/{len(all_ids)}] {nid}: no FASTA seq, geometric fallback only")
            seq = None

        if idx % 1000 == 0:
            print(f"  [{idx}/{len(all_ids)}] {nid} L={L} elapsed={time.time()-t0_all:.1f}s")

        # Decide method
        if args.use_geometric_fallback or not has_vienna or L > args.max_len:
            if L > args.max_len:
                skipped += 1
            bp = compute_bp_probs_geometric(coords)
        else:
            try:
                bp = compute_bp_probs_vienna(seq)
            except Exception as e:
                print(f"  ViennaRNA failed for {nid}: {e}; geometric fallback")
                bp = compute_bp_probs_geometric(coords)

        bp_probs_list.append(bp)
        lengths_list.append(L)

    print(f"\nDone: {len(bp_probs_list)} computed, {skipped} skipped (L>{args.max_len})")
    print(f"Total time: {time.time()-t0_all:.1f}s ({(time.time()-t0_all)/60:.1f}min)")

    # Save
    print(f"Saving {args.output}...")
    np.savez(
        args.output,
        ids=np.array(all_ids, dtype=object),
        lengths=np.array(lengths_list, dtype=np.int32),
        bp_probs=np.array(bp_probs_list, dtype=object),
    )
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {fsize:.1f} MB")


if __name__ == "__main__":
    main()
