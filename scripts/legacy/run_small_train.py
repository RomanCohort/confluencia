import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np 
print('Python executable:', sys.executable)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

print('Python executable:', sys.executable)

try:
    import torch
    print('torch', torch.__version__)
except Exception as e:
    print('torch import error', e)

try:
    from rdkit import Chem
    print('rdkit OK')
except Exception as e:
    print('rdkit import error', e)

from src.drug.torch_predictor import train_torch_bundle, dump_torch_bundle

# If RDKit is not available in this environment, provide a lightweight fallback
# featurizer that produces a deterministic hash-based fingerprint so training
# can proceed without RDKit.
try:
    import src.drug.featurizer as _featurizer_mod
except Exception:
    _featurizer_mod = None

if _featurizer_mod is not None:
    try:
        # quick check whether RDKit-backed MoleculeFeatures will import descriptors
        _featurizer_mod.MoleculeFeatures()
    except Exception:
        _featurizer_mod = _featurizer_mod

if _featurizer_mod is not None:
    try:
        from rdkit import Chem  # type: ignore
        _HAS_RDKit = True
    except Exception:
        _HAS_RDKit = False
else:
    _HAS_RDKit = False

if not _HAS_RDKit:
    import hashlib

    class MoleculeFeatures:
        """Fallback featurizer that hashes SMILES into a fixed-length bit vector.

        This mirrors the external API used by the training code so it can run
        even when RDKit is not installed. Invalid/empty SMILES return zeros.
        """

        def __init__(self, version: int = 2, radius: int = 2, n_bits: int = 2048):
            self.version = int(version)
            self.radius = int(radius)
            self.n_bits = int(n_bits)

        def feature_names(self):
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

        def _hash_bits(self, s: str):
            b = hashlib.sha256(s.encode("utf-8")).digest()
            need = (self.n_bits + 7) // 8
            rep = (b * ((need // len(b)) + 1))[:need]
            arr = np.unpackbits(np.frombuffer(rep, dtype=np.uint8))
            return arr[: self.n_bits].astype(np.float32)

        def transform_one(self, smiles: str):
            s = "" if smiles is None else str(smiles).strip()
            if not s:
                return np.zeros((self.dim(),), dtype=np.float32), False
            fp = self._hash_bits(s)
            if int(self.version) < 2:
                return fp, True
            desc = np.zeros((8,), dtype=np.float32)
            return np.concatenate([fp, desc], axis=0).astype(np.float32), True

        def transform_many(self, smiles_list):
            xs = []
            valids = []
            for s in smiles_list:
                x, ok = self.transform_one(s)
                xs.append(x)
                valids.append(ok)
            return np.stack(xs, axis=0).astype(np.float32), np.array(valids, dtype=bool)

        def dim(self):
            return int(self.n_bits) + (8 if int(self.version) >= 2 else 0)

    # replace module classes so training code uses fallback
    if _featurizer_mod is not None:
        _featurizer_mod.MoleculeFeatures = MoleculeFeatures
    try:
        import src.drug.torch_predictor as _tp

        _tp.MoleculeFeatures = MoleculeFeatures
    except Exception:
        pass

# load example data
csv_path = Path('data') / 'example_drug.csv'
print('loading', csv_path)
df = pd.read_csv(csv_path)
print('n rows', len(df))

# small training
bundle, metrics = train_torch_bundle(df, epochs=5, batch_size=32, use_cuda=False)
print('training completed')
print(metrics)

out_dir = Path('models') / 'pretrained'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'test_drug_torch_pretrained.pt'
dump_torch_bundle(bundle, str(out_path))
print('saved bundle to', out_path)
