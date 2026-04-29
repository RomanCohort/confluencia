from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.legacy_algorithms import LegacyAlgorithmConfig
from core.training import train_and_predict_drug


def _make_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    smiles = ["CCO", "CCN(CC)CC", "C1=CC=CC=C1"]
    seq = ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV"]
    df = pd.DataFrame(
        {
            "smiles": [smiles[i % len(smiles)] for i in range(n)],
            "epitope_seq": [seq[i % len(seq)] for i in range(n)],
            "dose": rng.uniform(0.2, 5.0, size=n),
            "freq": rng.uniform(0.5, 2.5, size=n),
            "treatment_time": rng.uniform(0, 72, size=n),
            "group_id": rng.choice(["G1", "G2"], size=n),
        }
    )
    df["efficacy"] = 0.4 * df["dose"] + 0.25 * df["freq"] + rng.normal(0, 0.2, size=n)
    return df


def main() -> None:
    df = _make_df(40)
    cfg = LegacyAlgorithmConfig(epochs=5, batch_size=16)
    result_df, curve_df, artifacts, report = train_and_predict_drug(df, compute_mode="low", model_backend="hgb", legacy_cfg=cfg)

    assert len(result_df) == len(df), "result rows mismatch"
    assert "efficacy_pred" in result_df.columns, "missing efficacy_pred"
    assert "ctm_auc_efficacy" in result_df.columns, "missing ctm summary"
    assert "immune_response_auc" in result_df.columns, "missing immune ABM summary"
    assert len(curve_df) > 0, "missing ctm curve"
    assert artifacts.model_backend == "hgb", "backend mismatch"
    assert report.sample_count == len(df), "report sample mismatch"
    print("drug smoke test passed")


if __name__ == "__main__":
    main()
