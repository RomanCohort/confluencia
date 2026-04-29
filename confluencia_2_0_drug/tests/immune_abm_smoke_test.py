from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.immune_abm import build_epitope_triggers, simulate_immune_response, summarize_immune_curve


def main() -> None:
    df = pd.DataFrame(
        {
            "epitope_seq": ["SLYNTVATL", "GILGFVFTL"],
            "dose": [2.0, 1.5],
            "treatment_time": [0.0, 12.0],
        }
    )
    triggers = build_epitope_triggers(df)
    assert len(triggers) == 2, "trigger count mismatch"

    curve = simulate_immune_response(triggers)
    summary = summarize_immune_curve(curve)

    assert len(curve) > 0, "empty immune curve"
    assert summary["immune_peak_antibody"] > 0, "antibody peak should be positive"
    assert summary["immune_response_auc"] > 0, "immune AUC should be positive"
    print("immune abm smoke test passed")


if __name__ == "__main__":
    main()
