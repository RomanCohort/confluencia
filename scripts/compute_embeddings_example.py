"""Example script: compute embeddings for example drug / epitope sequences.

This script uses `SequenceVectorizer` (random projection) to produce dense vectors
without requiring PyTorch. It serves as a quick smoke test and demonstration.
"""
from pathlib import Path
import numpy as np
from src.representations.sequence_vectorizer import SequenceVectorizer


def load_example_sequences():
    root = Path(__file__).resolve().parents[1]
    drug_csv = root / 'data' / 'example_drug.csv'
    epi_csv = root / 'data' / 'example_epitope.csv'
    seqs = {"drug": ["CCO", "NCC(=O)O"], "epitope": ["ACDEFGHIK", "GGGASD"]}
    try:
        import pandas as pd
        if drug_csv.exists():
            df = pd.read_csv(drug_csv)
            if 'smiles' in df.columns:
                seqs['drug'] = df['smiles'].astype(str).dropna().tolist()[:50]
        if epi_csv.exists():
            df2 = pd.read_csv(epi_csv)
            if 'sequence' in df2.columns:
                seqs['epitope'] = df2['sequence'].astype(str).dropna().tolist()[:50]
    except Exception:
        pass
    return seqs


def main():
    seqs = load_example_sequences()
    combined = seqs['drug'] + seqs['epitope']
    sv = SequenceVectorizer(max_len=128, emb_dim=128)
    sv.fit(combined)
    emb_drug = sv.embed_random(seqs['drug'])
    emb_epi = sv.embed_random(seqs['epitope'])
    out_dir = Path('build')
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / 'emb_drug.npy', emb_drug)
    np.save(out_dir / 'emb_epitope.npy', emb_epi)
    print('drug embeddings:', emb_drug.shape, 'saved to', out_dir / 'emb_drug.npy')
    print('epitope embeddings:', emb_epi.shape, 'saved to', out_dir / 'emb_epitope.npy')


if __name__ == '__main__':
    main()
