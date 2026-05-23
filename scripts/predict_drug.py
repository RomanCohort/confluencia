"""
predict_drug.py — Standalone drug efficacy prediction.

Zero dependency on drug_cli.py / src.common — only needs:
  joblib, rdkit, sklearn, numpy, pandas, confluencia_shared

Usage (AutoDL):
  cd /root/autodl-tmp/confluencia
  PYTHONPATH=/root/autodl-tmp/confluencia:$PYTHONPATH \
  python scripts/predict_drug.py \
      --model data/cache/drug_model.joblib \
      --smiles "CC(=O)Oc1ccccc1C(=O)O"

  # Batch screening:
  python scripts/predict_drug.py \
      --model data/cache/drug_model.joblib \
      --candidates data/candidates.csv \
      --smiles-col smiles \
      --out output/drug_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

# Auto-detect project root so confluencia_shared is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Stub streamlit/plotly so confluencia_shared imports succeed ---
import types


def _decorator(*a, **kw):
    if len(a) == 1 and callable(a[0]) and not kw:
        return a[0]
    return lambda f: f


def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__getattr__ = lambda attr: _decorator
    return mod


for _mn in ["streamlit", "plotly", "plotly.graph_objects", "plotly.express"]:
    sys.modules.setdefault(_mn, _make_stub(_mn))

# ---------------------------------------------------------------------------
# Inlined MoleculeFeatures — avoids importing core.predictor which has
# heavy dependencies on src.common.plotting, confluencia_shared.training, etc.
# Only the featurizer + bundle.load + model.predict path is needed.
# ---------------------------------------------------------------------------

try:
    from rdkit import RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.error")
except Exception:
    pass


class _MoleculeFeatures:
    """Minimal MoleculeFeatures (Morgan fingerprint + RDKit descriptors)."""

    def __init__(self, version: int = 2, radius: int = 2, n_bits: int = 2048):
        self.version = version
        self.radius = radius
        self.n_bits = n_bits

    def dim(self) -> int:
        return self.n_bits + (8 if self.version >= 2 else 0)

    def transform_one(self, smiles: str):
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import DataStructs  # type: ignore

        s = "" if smiles is None else str(smiles).strip()
        if not s:
            return np.zeros((self.dim(),), dtype=np.float32), False

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros((self.dim(),), dtype=np.float32), False

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        mol_x = arr.astype(np.float32)

        if self.version >= 2:
            from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore
            desc = np.array([
                float(Descriptors.MolWt(mol)),
                float(Descriptors.MolLogP(mol)),
                float(rdMolDescriptors.CalcTPSA(mol)),
                float(rdMolDescriptors.CalcNumHBD(mol)),
                float(rdMolDescriptors.CalcNumHBA(mol)),
                float(Descriptors.NumRotatableBonds(mol)),
                float(rdMolDescriptors.CalcNumRings(mol)),
                float(rdMolDescriptors.CalcFractionCSP3(mol)),
            ], dtype=np.float32)
            mol_x = np.concatenate([mol_x, desc], axis=0)

        return mol_x.astype(np.float32), True

    def transform_many(self, smiles_list):
        xs = []
        valids = []
        cache = {}
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


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


def predict_single(bundle, smiles: str, env_params: Optional[Dict[str, float]] = None) -> float:
    """Predict efficacy for a single SMILES string."""
    featurizer = _MoleculeFeatures(
        version=int(bundle.featurizer_version),
        radius=int(bundle.radius),
        n_bits=int(bundle.n_bits),
    )
    mol_x, ok = featurizer.transform_one(smiles)

    env_params = env_params or {}
    env_vec = []
    for c in bundle.env_cols:
        if c in env_params:
            env_vec.append(float(env_params[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.array(env_vec, dtype=np.float32) if bundle.env_cols else np.zeros((0,), dtype=np.float32)
    x = np.concatenate([mol_x, env_x], axis=0).reshape(1, -1)

    y_pred = bundle.model.predict(x)
    return float(np.asarray(y_pred).reshape(-1)[0])


def predict_batch(bundle, df: pd.DataFrame, smiles_col: str = "smiles") -> np.ndarray:
    """Predict efficacy for all rows in a DataFrame."""
    featurizer = _MoleculeFeatures(
        version=int(bundle.featurizer_version),
        radius=int(bundle.radius),
        n_bits=int(bundle.n_bits),
    )

    smiles_list = df[smiles_col].astype(str).tolist()
    mol_x, valids = featurizer.transform_many(smiles_list)

    env_x = np.zeros((len(df), len(bundle.env_cols)), dtype=np.float32)
    for j, c in enumerate(bundle.env_cols):
        if c in df.columns:
            env_x[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(bundle.env_medians.get(c, 0.0)).astype(np.float32).values
        else:
            env_x[:, j] = float(bundle.env_medians.get(c, 0.0))

    x = np.concatenate([mol_x, env_x], axis=1)

    preds = np.empty((x.shape[0],), dtype=np.float32)
    chunk = 5000
    for start in range(0, x.shape[0], chunk):
        end = min(x.shape[0], start + chunk)
        preds[start:end] = np.asarray(bundle.model.predict(x[start:end]), dtype=np.float32).reshape(-1)

    # Mark invalid SMILES as NaN
    if valids is not None:
        preds = preds.astype(float)
        preds[~valids.astype(bool)] = float("nan")

    return preds


def main():
    parser = argparse.ArgumentParser(description="Drug efficacy predictor (standalone)")
    parser.add_argument("--model", required=True, help="Model bundle path (.joblib)")
    parser.add_argument("--smiles", type=str, default=None, help="Single SMILES to predict")
    parser.add_argument("--param", action="append", default=None,
                        help="Env parameter key=value (repeatable)")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Candidates CSV for batch screening")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column in candidates CSV")
    parser.add_argument("--out", default="drug_predictions.csv", help="Output CSV for batch screening")
    parser.add_argument("--out-col", default="pred_efficacy", help="Prediction column name in output")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    bundle = joblib.load(model_path)
    print(f"  target_col: {bundle.target_col}")
    print(f"  env_cols: {bundle.env_cols}")
    print(f"  featurizer: v{bundle.featurizer_version}, r={bundle.radius}, nbits={bundle.n_bits}")

    if args.smiles:
        # --- Single prediction ---
        env_params = {}
        if args.param:
            for p in args.param:
                if "=" not in p:
                    print(f"ERROR: invalid param '{p}', expected key=value")
                    sys.exit(1)
                k, v = p.split("=", 1)
                env_params[k.strip()] = float(v.strip())

        y = predict_single(bundle, args.smiles, env_params)
        print("\n== Prediction ==")
        print(f"smiles: {args.smiles}")
        print(f"pred:   {y:.6g}")
        if bundle.env_cols:
            resolved = {c: float(env_params.get(c, bundle.env_medians.get(c, 0.0))) for c in bundle.env_cols}
            print(f"env:    {resolved}")

    elif args.candidates:
        # --- Batch screening ---
        df = pd.read_csv(args.candidates)
        if args.smiles_col not in df.columns:
            print(f"ERROR: column '{args.smiles_col}' not in {args.candidates}")
            sys.exit(1)

        print(f"Batch screening: {len(df)} candidates")
        preds = predict_batch(bundle, df, smiles_col=args.smiles_col)

        out = df.copy()
        out[args.out_col] = preds

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

        print(f"\n== Screening done ==")
        print(f"candidates: {args.candidates}")
        print(f"out:        {out_path}")
        print(f"valid:      {int(np.isfinite(preds).sum())}/{len(preds)}")
        print(f"mean pred:  {np.nanmean(preds):.4g}")
    else:
        print("ERROR: provide --smiles for single prediction or --candidates for batch screening")
        sys.exit(1)


if __name__ == "__main__":
    main()