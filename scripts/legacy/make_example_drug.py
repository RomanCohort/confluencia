from __future__ import annotations

# pyright: reportAttributeAccessIssue=false

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from outside the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors  # type: ignore

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    # Small, diverse-ish SMILES pool (public domain examples)
    smiles_pool = [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
        "C1=CC=CC=C1",  # benzene
        "CCN(CC)CC",  # triethylamine-like
        "O=C(O)C(O)(CO)CO",  # citric acid fragment-ish
        "CCOC(=O)C1=CC=CC=C1",  # ethyl benzoate
        "CC(C)NCC(O)CO",  # isopropanolamine-like
    ]

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/example_drug.csv")
    args = ap.parse_args()

    rows = []
    for i in range(int(args.n)):
        # Inject some invalid SMILES to test robustness
        if i % 37 == 0:
            smi = "INVALID_SMILES"
            mol = None
        else:
            smi = rng.choice(smiles_pool)
            mol = Chem.MolFromSmiles(smi)

        dose = float(rng.choice([1, 3, 10, 30, 100]))
        freq = float(rng.choice([1, 2, 3]))

        if mol is None:
            mol_wt = 0.0
            logp = 0.0
            tpsa = 0.0
        else:
            mol_wt = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))

        # Synthetic target learnable from descriptors + env
        y = (
            0.004 * mol_wt
            - 0.25 * logp
            - 0.001 * tpsa
            + 0.02 * np.log1p(dose)
            + 0.03 * freq
            + float(np_rng.normal(0.0, 0.05))
        )

        rows.append({"smiles": smi, "dose": dose, "freq": freq, "efficacy": y})

    df = pd.DataFrame(rows)
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
