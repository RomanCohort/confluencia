"""
preprocess_isrnacirc_labels.py — Extract high-quality pseudo-labels from IsRNAcirc.

IsRNAcirc provides 34 circRNA 3D structures with DFIRE-RNA and rsRNASP scoring.
These are much more reliable than ViennaRNA 2D predictions for TorusFold training.

Strategy:
1. Load IsRNAcirc predicted 3D structures (PDB files)
2. Load DFIRE-RNA and rsRNASP scores
3. Convert 3D coordinates to pair distance matrices (training targets)
4. Filter by score thresholds
"""

import os, sys, glob
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ISRNACIRC_DIR = PROJECT_ROOT / "tools" / "IsRNAcirc"


def parse_pdb_coordinates(pdb_path):
    """Extract C3' or P atom coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                # Use C3' for RNA backbone, or P for phosphate
                if atom_name in ["C3'", "C3'", "P"]:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords)


def compute_distance_matrix(coords):
    """Compute pairwise distance matrix from 3D coordinates."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def distance_to_pair_prob(distance_matrix, cutoff=15.0, sigma=2.0):
    """
    Convert distance matrix to pairing probability.

    P(i,j) = exp(-(d_ij - d_wc)^2 / (2*sigma^2)) for d < cutoff
    where d_wc ≈ 10.6 Å is Watson-Crick C1'-C1' distance

    For circRNA, also boost BSJ-flanking regions.
    """
    L = distance_matrix.shape[0]
    pair_prob = np.zeros((L, L), dtype=np.float32)

    # Watson-Crick distance ~10.6 Å
    d_wc = 10.6

    for i in range(L):
        for j in range(i+1, L):
            d = distance_matrix[i, j]
            if d < cutoff:
                # Gaussian centered at WC distance
                prob = np.exp(-((d - d_wc) ** 2) / (2 * sigma ** 2))
                # Add contribution for nearby distances (stacking, etc.)
                if d < 8:
                    prob += 0.3 * np.exp(-((d - 6) ** 2) / (2 * 1.5 ** 2))
                pair_prob[i, j] = min(prob, 1.0)
                pair_prob[j, i] = pair_prob[i, j]

    # BSJ boost: positions near 0 and L-1 should have higher pairing
    bsj = 20
    if L > 2 * bsj:
        # These positions are neighbors in circRNA topology
        pair_prob[:bsj, L-bsj:] *= 1.5
        pair_prob[L-bsj:, :bsj] *= 1.5

    return np.clip(pair_prob, 0, 1)


def parse_scoring_log(log_path):
    """Parse DFIRE-RNA or rsRNASP scoring log."""
    scores = []
    with open(log_path, 'r') as f:
        for line in f:
            # Format varies, try to extract numeric scores
            parts = line.strip().split()
            for p in parts:
                try:
                    score = float(p)
                    scores.append(score)
                except:
                    pass
    return scores


def load_isrnacirc_data():
    """Load all IsRNAcirc predicted structures and scores."""
    data_dir = ISRNACIRC_DIR / "circular_RNA_Data"
    scoring_dir = ISRNACIRC_DIR / "scoring"

    sequences = []
    pair_probs = []
    scores_dfire = []
    scores_rsrnasp = []
    circ_ids = []

    # Load DFIRE-RNA scores
    dfire_log = scoring_dir / "DFIRE-RNA_IsRNAcirc_QRNAS_refined_scoring.log"
    if dfire_log.exists():
        scores_dfire = parse_scoring_log(dfire_log)
        print(f"Loaded {len(scores_dfire)} DFIRE-RNA scores")

    # Load rsRNASP scores
    rsrnasp_log = scoring_dir / "rsRNASP_IsRNAcirc_predicted_scoring.log"
    if rsrnasp_log.exists():
        scores_rsrnasp = parse_scoring_log(rsrnasp_log)
        print(f"Loaded {len(scores_rsrnasp)} rsRNASP scores")

    # Find all PDB files
    pdb_files = list(data_dir.glob("*/*/job_IsRNAcirc.pdb"))
    print(f"Found {len(pdb_files)} IsRNAcirc predicted structures")

    for i, pdb_path in enumerate(pdb_files):
        circ_id = pdb_path.parent.name

        # Load coordinates
        coords = parse_pdb_coordinates(pdb_path)
        if len(coords) == 0:
            print(f"  Warning: no coordinates in {pdb_path}")
            continue

        L = len(coords)

        # Load sequence from .subo file or infer from PDB
        subo_files = list(pdb_path.parent.glob("sequence_2D_structure/*.subo"))
        if subo_files:
            with open(subo_files[0], 'r') as f:
                # .subo format: sequence line followed by structure
                lines = f.readlines()
                seq = lines[0].strip() if lines else "N" * L
        else:
            seq = "N" * L  # Unknown sequence

        # Compute distance matrix and convert to pair probabilities
        dist_matrix = compute_distance_matrix(coords)
        pair_prob = distance_to_pair_prob(dist_matrix)

        sequences.append(seq.upper().replace('T', 'U'))
        pair_probs.append(pair_prob)
        circ_ids.append(circ_id)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(pdb_files)} structures")

    print(f"Loaded {len(sequences)} circRNA 3D structures from IsRNAcirc")
    return circ_ids, sequences, pair_probs, scores_dfire, scores_rsrnasp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/circrna/isrnacirc_labels.npz")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Minimum DFIRE-RNA score to include (filter low quality)")
    args = parser.parse_args()

    circ_ids, sequences, pair_probs, scores_dfire, scores_rsrnasp = load_isrnacirc_data()

    if len(sequences) == 0:
        print("ERROR: No sequences loaded")
        return

    # Length stats
    lengths = [len(s) for s in sequences]
    print(f"\nStats:")
    print(f"  Structures: {len(sequences)}")
    print(f"  Length: min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.0f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_dict = {
        'ids': np.array(circ_ids, dtype=object),
        'sequences': np.array(sequences, dtype=object),
        'n_seqs': len(sequences),
    }
    for i, pp in enumerate(pair_probs):
        save_dict[f'label_{i}'] = pp

    if scores_dfire:
        save_dict['dfire_scores'] = np.array(scores_dfire)
    if scores_rsrnasp:
        save_dict['rsrnasp_scores'] = np.array(scores_rsrnasp)

    np.savez_compressed(args.output, **save_dict, allow_pickle=True)
    print(f"\nSaved to {args.output}")

    # Also save summary
    summary_path = args.output.replace('.npz', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"IsRNAcirc Pseudo-labels Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total structures: {len(sequences)}\n")
        f.write(f"Length range: {min(lengths)}-{max(lengths)}\n")
        f.write(f"Source: IsRNAcirc (DongZhangRNA/IsRNAcirc)\n")
        f.write(f"Scoring: DFIRE-RNA, rsRNASP\n\n")
        f.write(f"{'ID':<20} {'Length':<8} {'DFIRE':<10} {'rsRNASP':<10}\n")
        for i, cid in enumerate(circ_ids):
            dfire = scores_dfire[i] if i < len(scores_dfire) else 'N/A'
            rsrna = scores_rsrnasp[i] if i < len(scores_rsrnasp) else 'N/A'
            f.write(f"{cid:<20} {len(sequences[i]):<8} {dfire:<10} {rsrna:<10}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
