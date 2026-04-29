from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.legacy_algorithms import LegacyAlgorithmConfig
from core.training import train_and_predict_drug

try:
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass


def _make_df(n: int = 36, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles = ["CCO", "CCN(CC)CC", "C1=CC=CC=C1", "CC(=O)O"]
    seq = ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV", "LLFGYPVYV"]
    df = pd.DataFrame(
        {
            "smiles": [smiles[i % len(smiles)] for i in range(n)],
            "epitope_seq": [seq[i % len(seq)] for i in range(n)],
            "dose": rng.uniform(0.2, 5.0, size=n),
            "freq": rng.uniform(0.5, 2.5, size=n),
            "treatment_time": rng.uniform(0, 72, size=n),
            "group_id": rng.choice(["G1", "G2", "G3"], size=n),
        }
    )
    df["efficacy"] = 0.45 * df["dose"] + 0.2 * df["freq"] + rng.normal(0, 0.2, size=n)
    return df


def _assert_run_ok(df: pd.DataFrame, cfg: LegacyAlgorithmConfig) -> pd.DataFrame:
    result_df, curve_df, artifacts, report = train_and_predict_drug(
        df,
        compute_mode="low",
        model_backend="ridge",
        legacy_cfg=cfg,
    )

    assert len(result_df) == len(df), "row count mismatch"
    assert "efficacy_pred" in result_df.columns, "missing efficacy prediction"
    assert int(curve_df.shape[0]) > 0, "ctm curve should not be empty"
    assert artifacts.model_backend == "ridge", "backend mismatch"
    assert report.sample_count == len(df), "report count mismatch"

    pred = pd.Series(pd.to_numeric(result_df["efficacy_pred"], errors="coerce"), dtype="float64")
    assert bool(pred.notna().all()), "prediction contains NaN"

    if "cross_group_impact" in result_df.columns:
        cgi = pd.Series(pd.to_numeric(result_df["cross_group_impact"], errors="coerce"), dtype="float64").fillna(0.0)
        assert np.isfinite(cgi.to_numpy(dtype=np.float32)).all(), "cross_group_impact has non-finite values"

    return result_df


def main() -> None:
    cfg = LegacyAlgorithmConfig(epochs=5, batch_size=16)

    base_df = _make_df(36, seed=19)
    _ = _assert_run_ok(base_df, cfg)

    miss_df = base_df.copy()
    miss_df.loc[miss_df.index[::5], "dose"] = np.nan
    miss_df.loc[miss_df.index[::7], "freq"] = np.nan
    _ = _assert_run_ok(miss_df, cfg)

    out_df = base_df.copy()
    out_df.loc[out_df.index[::6], "dose"] = out_df["dose"].max() * 12.0
    out_df.loc[out_df.index[::8], "freq"] = out_df["freq"].max() * 10.0
    _ = _assert_run_ok(out_df, cfg)

    small_df = _make_df(14, seed=23)
    _ = _assert_run_ok(small_df, cfg)

    print("drug robustness regression test passed")


if __name__ == "__main__":
    main()
