"""
preprocess_circbase_labels.py — Pre-generate ViennaRNA pseudo-labels for circBase.

Run this on CPU overnight. Saves labels as .npz so training script can load instantly.

Usage:
    python preprocess_circbase_labels.py --max-seqs 5000 --output data/circrna/circbase_labels_5k.npz
    python preprocess_circbase_labels.py --max-seqs 2000 --output data/circrna/circbase_labels_2k.npz
"""

import os, sys, gzip, pickle, argparse
from pathlib import Path
import subprocess
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_circbase_fasta(path, min_len=50, max_len=500, max_seqs=5000):
    """Load circBase sequences from gzipped FASTA."""
    sequences = []
    ids = []
    opn = gzip.open if path.endswith('.gz') else open

    with opn(path, 'rt') as f:
        current_id = ""
        current_seq = ""
        for line in f:
            if line.startswith('>'):
                if current_seq and min_len <= len(current_seq) <= max_len:
                    sequences.append(current_seq.upper().replace('T', 'U'))
                    ids.append(current_id)
                    if len(sequences) >= max_seqs:
                        break
                current_id = line.strip()[1:].split('|')[0]
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq and min_len <= len(current_seq) <= max_len and len(sequences) < max_seqs:
            sequences.append(current_seq.upper().replace('T', 'U'))
            ids.append(current_id)

    return ids, sequences


def run_viennarna(sequence):
    """Run ViennaRNA RNAfold on a single sequence."""
    L = len(sequence)
    try:
        result = subprocess.run(
            f"echo '{sequence}' | RNAfold --MEA --noPS",
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return parse_dot_bracket(sequence, result.stdout)
    except:
        pass
    return heuristic_pairs(sequence)


def parse_dot_bracket(sequence, output):
    """Parse ViennaRNA dot-bracket to pair probability matrix."""
    L = len(sequence)
    pair_probs = np.zeros((L, L), dtype=np.float32)

    for line in output.strip().split('\n'):
        if '.' in line and '(' in line:
            # Parse paired positions
            stack = []
            for j, char in enumerate(line[:L]):
                if char == '(':
                    stack.append(j)
                elif char == ')' and stack:
                    i = stack.pop()
                    pair_probs[i, j] = 0.85
                    pair_probs[j, i] = 0.85

            # Add weak background for complementary bases
            complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
            for i in range(L):
                for j in range(i+1, min(i+15, L)):
                    if pair_probs[i, j] == 0 and complement.get(sequence[i]) == sequence[j]:
                        pair_probs[i, j] = pair_probs[j, i] = 0.15

            # BSJ boost
            bsj = 20
            if L > 2 * bsj:
                pair_probs[:bsj, L-bsj:] = np.clip(pair_probs[:bsj, L-bsj:] * 1.5, 0, 1)
                pair_probs[L-bsj:, :bsj] = np.clip(pair_probs[L-bsj:, :bsj] * 1.5, 0, 1)
            return pair_probs

    return pair_probs


def heuristic_pairs(sequence):
    """Fallback heuristic pairing."""
    L = len(sequence)
    pair_probs = np.zeros((L, L), dtype=np.float32)
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    for i in range(L):
        for j in range(i+1, min(i+20, L)):
            if complement.get(sequence[i]) == sequence[j]:
                pair_probs[i, j] = pair_probs[j, i] = 0.7
            elif (sequence[i] == 'G' and sequence[j] == 'U') or \
                 (sequence[i] == 'U' and sequence[j] == 'G'):
                pair_probs[i, j] = pair_probs[j, i] = 0.3
    return pair_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/circrna/circbase_seqs.fa.gz")
    parser.add_argument("--max-seqs", type=int, default=5000)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--output", default="data/circrna/circbase_labels_5k.npz")
    args = parser.parse_args()

    # Load sequences
    ids, sequences = load_circbase_fasta(
        args.fasta, min_len=args.min_len, max_len=args.max_len, max_seqs=args.max_seqs
    )
    print(f"Loaded {len(sequences)} sequences from circBase ({args.min_len}-{args.max_len}nt)")

    # Generate ViennaRNA labels
    print(f"Running ViennaRNA RNAfold on {len(sequences)} sequences...")
    labels = []
    viennarna_ok = 0
    heuristic_fallback = 0

    for i, seq in enumerate(sequences):
        pair_probs = run_viennarna(seq)
        labels.append(pair_probs)

        # Track which method was used
        if pair_probs.sum() > 0 and np.any(pair_probs > 0.5):
            viennarna_ok += 1
        else:
            heuristic_fallback += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(sequences)} (ViennaRNA: {viennarna_ok}, heuristic: {heuristic_fallback})")

    print(f"Done. ViennaRNA: {viennarna_ok}, heuristic fallback: {heuristic_fallback}")

    # Save as compressed npz
    # Store as dict of arrays keyed by index
    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    save_dict = {
        'ids': np.array(ids, dtype=object),
        'sequences': np.array(sequences, dtype=object),
        'n_seqs': len(sequences),
        'viennarna_ok': viennarna_ok,
        'heuristic_fallback': heuristic_fallback,
    }

    # Save pair labels as separate arrays (npz can handle variable-size with allow_pickle)
    for i, label in enumerate(labels):
        save_dict[f'label_{i}'] = label

    np.savez_compressed(args.output, **save_dict, allow_pickle=True)
    print(f"Saved {len(sequences)} labeled sequences to {args.output}")

    # Also save a simpler format: sequences + lengths
    meta_path = args.output.replace('.npz', '_meta.csv')
    with open(meta_path, 'w') as f:
        f.write("idx,circrna_id,length,method\n")
        for i in range(len(sequences)):
            method = "viennarna" if i < viennarna_ok else "heuristic"
            f.write(f"{i},{ids[i]},{len(sequences[i])},{method}\n")
    print(f"Metadata saved to {meta_path}")

    # Quick stats
    lengths = [len(s) for s in sequences]
    print(f"\nStats:")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Length: median={np.median(lengths):.0f}, mean={np.mean(lengths):.0f}")
    print(f"  GC content: mean={np.mean([s.count('G')+s.count('C') for s in sequences])/np.mean(lengths):.2f}")
    print(f"  ViennaRNA success rate: {viennarna_ok/len(sequences)*100:.1f}%")


if __name__ == "__main__":
    main()
