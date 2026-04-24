from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Reduce noisy RDKit parsing logs (invalid SMILES etc.).
try:  # pragma: no cover
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.error")
except Exception:  # pragma: no cover
    pass


@dataclass(frozen=True)
class MoleculeFeatures:
    """RDKit-based featurizer for SMILES.

    version=1:
      Morgan fingerprint only.

    version=2:
      Morgan fingerprint + a small set of RDKit descriptors.

    Notes
    -----
    - Returns float32 vectors.
    - Invalid SMILES return all-zeros.
    """

    version: int = 2
    radius: int = 2
    n_bits: int = 2048

    def feature_names(self) -> List[str]:
        names = [f"morgan_{i}" for i in range(int(self.n_bits))]
        if int(self.version) >= 2:
            names += [
                "desc_MolWt",
                "desc_MolLogP",
                "desc_TPSA",
                "desc_NumHDonors",
                "desc_NumHAcceptors",
                "desc_NumRotatableBonds",
                "desc_RingCount",
                "desc_FractionCSP3",
            ]
        return names

    def _fingerprint(self, mol) -> np.ndarray:
        # Imported lazily so the module can be imported even when RDKit isn't installed.
        from rdkit.Chem import AllChem  # type: ignore

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(self.radius), nBits=int(self.n_bits))
        arr = np.zeros((int(self.n_bits),), dtype=np.int8)
        # RDKit writes into a numpy array via ConvertToNumpyArray
        from rdkit import DataStructs  # type: ignore

        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)

    def _descriptors(self, mol) -> np.ndarray:
        from rdkit.Chem import Descriptors  # type: ignore
        from rdkit.Chem import rdMolDescriptors  # type: ignore

        vals = np.array(
            [
                float(Descriptors.MolWt(mol)),
                float(Descriptors.MolLogP(mol)),
                float(rdMolDescriptors.CalcTPSA(mol)),
                float(rdMolDescriptors.CalcNumHBD(mol)),
                float(rdMolDescriptors.CalcNumHBA(mol)),
                float(Descriptors.NumRotatableBonds(mol)),
                float(rdMolDescriptors.CalcNumRings(mol)),
                float(rdMolDescriptors.CalcFractionCSP3(mol)),
            ],
            dtype=np.float32,
        )
        return vals

    def transform_one(self, smiles: str) -> Tuple[np.ndarray, bool]:
        """Returns (vector, is_valid)."""
        from rdkit import Chem  # type: ignore

        s = "" if smiles is None else str(smiles).strip()
        if not s:
            return np.zeros((self.dim(),), dtype=np.float32), False

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros((self.dim(),), dtype=np.float32), False

        fp = self._fingerprint(mol)
        if int(self.version) < 2:
            return fp, True

        desc = self._descriptors(mol)
        return np.concatenate([fp, desc], axis=0).astype(np.float32), True

    def transform_many(self, smiles_list: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not smiles_list:
            empty_x = np.zeros((0, self.dim()), dtype=np.float32)
            empty_v = np.zeros((0,), dtype=bool)
            return empty_x, empty_v

        # Deduplicate to avoid repeated RDKit work for identical SMILES
        cache: dict[str, Tuple[np.ndarray, bool]] = {}
        xs = []
        valids = []
        for s in smiles_list:
            key = "" if s is None else str(s).strip()
            if key in cache:
                x, ok = cache[key]
            else:
                x, ok = self.transform_one(key)
                cache[key] = (x, ok)
            xs.append(x)
            valids.append(ok)
        return np.stack(xs, axis=0).astype(np.float32), np.array(valids, dtype=bool)

    def dim(self) -> int:
        return int(self.n_bits) + (8 if int(self.version) >= 2 else 0)
