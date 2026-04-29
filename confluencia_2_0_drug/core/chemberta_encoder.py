"""
ChemBERTa SMILES language model encoder.
Extracts [CLS] token embedding (768-dim) from a pretrained ChemBERTa model.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

_EMBED_DIM = 768
_DEFAULT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaEncoder:
    """Lazy-loading ChemBERTa encoder.

    Extracts [CLS] token as 768-dim molecular embedding.
    Caches embeddings to disk per unique SMILES set.

    Online mode: downloads model from HuggingFace hub on first use.
    Offline mode: returns zero vectors (model not available locally).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_dir: str = "./.cache",
        device: Optional[str] = None,
        batch_size: int = 64,
        online: bool = False,  # new: attempt online download
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._device = device or ("cuda" if _torch_cuda() else "cpu")
        self._online = online
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return

        # online=True: allow download from hub; online=False: cache only
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=not self._online,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, local_files_only=not self._online,
            )
        except Exception:
            return

        try:
            import torch
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
        except Exception:
            pass
        self._model.eval()

    def _cache_path(self, unique_smiles: List[str]) -> str:
        digest = hashlib.md5("|".join(unique_smiles).encode()).hexdigest()
        safe = self.model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"chemberta_{safe}_{digest}.pkl")

    def encode(self, smiles_list: List[str], use_cache: bool = True) -> np.ndarray:
        """Encode SMILES list into (n, 768) matrix.

        Args:
            smiles_list: SMILES strings.
            use_cache: Read/write disk cache.

        Returns:
            np.ndarray (n, 768), dtype float32.
        """
        if not smiles_list:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        # Deduplicate
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for s in smiles_list:
            key = str(s).strip()
            if key not in seen:
                seen[key] = len(unique)
                unique.append(key)

        # Check cache
        if use_cache:
            cp = self._cache_path(unique)
            if os.path.exists(cp):
                with open(cp, "rb") as f:
                    cached = pickle.load(f)
                return self._expand(cached, seen, smiles_list)

        self.load()
        if self._model is None:
            # Model unavailable → return zeros
            return np.zeros((len(smiles_list), _EMBED_DIM), dtype=np.float32)

        import torch

        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(unique), self.batch_size):
                chunk = unique[start : start + self.batch_size]
                inputs = self._tokenizer(
                    chunk, padding=True, truncation=True, max_length=128, return_tensors="pt",
                )
                if self._device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self._model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
                for emb in cls_emb.cpu().numpy():
                    embeddings.append(emb.astype(np.float32))

        emb_arr = np.array(embeddings, dtype=np.float32)

        # Save cache
        if use_cache and unique:
            os.makedirs(self.cache_dir, exist_ok=True)
            cp = self._cache_path(unique)
            with open(cp, "wb") as f:
                pickle.dump({i: emb_arr[i] for i in range(len(unique))}, f)

        return self._expand({i: emb_arr[i] for i in range(len(unique))}, seen, smiles_list)

    def _expand(
        self, cached: Dict[int, np.ndarray], seen: Dict[str, int], smiles_list: List[str]
    ) -> np.ndarray:
        result = np.zeros((len(smiles_list), _EMBED_DIM), dtype=np.float32)
        for orig_idx, s in enumerate(smiles_list):
            key = str(s).strip()
            uid = seen.get(key)
            if uid is not None and uid in cached:
                result[orig_idx] = cached[uid]
        return result


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
