"""Small example: train transformer on tiny synthetic dataset to verify end-to-end flow."""
from pathlib import Path
from src.models.train_transformer import train_transformer, load_transformer_bundle, predict_one


def main():
    seqs = ["CCO", "NCCO", "CCCC", "C1=CC=CC=C1", "N[N+](=O)[O-]"]
    # synthetic targets
    targets = [0.1, 0.2, 0.05, 0.9, 0.5]
    out = train_transformer(seqs, targets, max_len=32, emb_dim=64, nhead=2, num_layers=1, ff_dim=128, batch_size=2, lr=1e-3, epochs=3, use_cuda=False, save_path='build/transformer_small.pt')
    print('history', out['history'])
    bundle = load_transformer_bundle('build/transformer_small.pt')
    pred = predict_one(bundle, 'CCO')
    print('prediction for CCO:', pred)


if __name__ == '__main__':
    main()
