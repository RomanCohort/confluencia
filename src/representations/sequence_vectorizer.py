import json
from typing import List, Dict, Tuple
import numpy as np


def build_char_vocab(sequences: List[str]) -> Dict[str, int]:
    chars = set()
    for s in sequences:
        chars.update(list(s))
    chars = sorted(chars)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, c in enumerate(chars, start=2):
        vocab[c] = i
    return vocab


def encode_sequence(seq: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        arr[i] = vocab.get(ch, vocab.get("<UNK>", 1))
    return arr


def batch_encode(seqs: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i] = encode_sequence(s, vocab, max_len)
    return out


class SequenceVectorizer:
    """Simple char-level vectorizer with random projection fallback.

    - `fit` builds a char vocab
    - `transform` returns integer indices array
    - `embed_random` projects indices -> dense embedding via random matrix + mean pooling
    """

    def __init__(self, max_len: int = 128, emb_dim: int = 128, seed: int = 42):
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.vocab: Dict[str, int] = {}
        self.seed = seed
        self._proj = None

    def fit(self, sequences: List[str]):
        self.vocab = build_char_vocab(sequences)
        # initialize random projection matrix (vocab_size x emb_dim)
        rng = np.random.RandomState(self.seed)
        self._proj = rng.normal(scale=0.1, size=(len(self.vocab), self.emb_dim)).astype(np.float32)

    def transform(self, sequences: List[str]) -> np.ndarray:
        if not self.vocab:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return batch_encode(sequences, self.vocab, self.max_len)

    def embed_random(self, sequences: List[str]) -> np.ndarray:
        """Return dense embeddings (N x emb_dim) by mean-pooling token projections."""
        indices = self.transform(sequences)
        if self._proj is None:
            rng = np.random.RandomState(self.seed)
            self._proj = rng.normal(scale=0.1, size=(len(self.vocab), self.emb_dim)).astype(np.float32)
        # indices shape (N, L)
        token_emb = self._proj[indices]  # (N, L, emb_dim)
        # mask PAD tokens (index 0)
        mask = (indices != 0).astype(np.float32)[..., None]
        summed = (token_emb * mask).sum(axis=1)
        denom = mask.sum(axis=1)
        denom[denom == 0] = 1.0
        return (summed / denom).astype(np.float32)

    def save(self, path: str):
        obj = {"max_len": self.max_len, "emb_dim": self.emb_dim, "vocab": self.vocab, "seed": self.seed}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SequenceVectorizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sv = cls(max_len=obj["max_len"], emb_dim=obj["emb_dim"], seed=obj.get("seed", 42))
        sv.vocab = obj["vocab"]
        rng = np.random.RandomState(sv.seed)
        sv._proj = rng.normal(scale=0.1, size=(len(sv.vocab), sv.emb_dim)).astype(np.float32)
        return sv


if __name__ == "__main__":
    # quick smoke test
    seqs = ["CCO", "N[C@@H](C)C(=O)O", "C1=CC=CC=C1"]
    sv = SequenceVectorizer(max_len=20, emb_dim=32)
    sv.fit(seqs)
    idx = sv.transform(seqs)
    emb = sv.embed_random(seqs)
    print("indices.shape", idx.shape)
    print("emb.shape", emb.shape)
