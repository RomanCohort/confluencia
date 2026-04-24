"""Full example: read CSV (if available), train with validation and early stopping, save model and print metrics.

Run with:
    $env:PYTHONPATH='.'; python .\scripts\train_transformer_full_example.py
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.train_transformer import train_transformer, load_transformer_bundle, predict_one


def main():
    root = Path(__file__).resolve().parents[1]
    csv = root / 'data' / 'example_drug.csv'
    if csv.exists():
        df = pd.read_csv(csv)
        if 'smiles' in df.columns and 'efficacy' in df.columns:
            seqs = df['smiles'].astype(str).tolist()
            targets = pd.to_numeric(df['efficacy'], errors='coerce').fillna(0.0).tolist()
        else:
            print('example_drug.csv lacks smiles/efficacy columns; using synthetic data')
            seqs = ['CCO', 'NCCO', 'CCCC', 'C1=CC=CC=C1', 'N[N+](=O)[O-]']
            targets = [0.1, 0.2, 0.05, 0.9, 0.5]
    else:
        seqs = ['CCO', 'NCCO', 'CCCC', 'C1=CC=CC=C1', 'N[N+](=O)[O-]']
        targets = [0.1, 0.2, 0.05, 0.9, 0.5]

    out = train_transformer(seqs, targets, max_len=64, emb_dim=64, nhead=2, num_layers=1, ff_dim=128, batch_size=2, lr=1e-3, epochs=50, use_cuda=False, save_path='build/transformer_full.pt', test_size=0.2, patience=5)
    print('history keys:', out['history'].keys())
    print('last train loss:', out['history']['train_loss'][-1])
    if out['history']['val_loss']:
        print('last val loss:', out['history']['val_loss'][-1])

    bundle = load_transformer_bundle('build/transformer_full.pt') if Path('build/transformer_full.pt').exists() else None
    if bundle:
        print('Saved bundle loaded; making a test prediction:')
        print(predict_one(bundle, 'CCO'))


if __name__ == '__main__':
    main()
