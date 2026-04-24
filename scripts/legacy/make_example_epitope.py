from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from outside the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.epitope.featurizer import AA_ORDER, SequenceFeatures


def _rand_seq(rng: random.Random, min_len: int = 8, max_len: int = 12) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(AA_ORDER) for _ in range(n))


def main() -> None:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    rows = []
    featurizer = SequenceFeatures(version=2)

    for _ in range(300):
        seq = _rand_seq(rng)
        concentration = float(10 ** rng.uniform(-2, 2))  # 0.01..100
        cell_density = float(rng.choice([2e5, 5e5, 1e6, 2e6]))
        incubation_hours = float(rng.choice([6, 12, 24, 48]))

        x = featurizer.transform_one(seq)
        # Use a simple synthetic ground truth that's learnable from features + env
        # y = a * hydropathy_mean + b * net_charge + c * log10(conc) + d * incubation + noise
        hydropathy_mean = float(x[featurizer.feature_names().index("hydropathy_mean")])
        net_charge = float(x[featurizer.feature_names().index("net_charge_est")])
        y = (
            0.8 * hydropathy_mean
            + 0.6 * net_charge
            + 0.4 * np.log10(concentration + 1e-9)
            + 0.01 * incubation_hours
            + 0.0000001 * cell_density
            + float(np_rng.normal(0.0, 0.15))
        )

        rows.append(
            {
                "sequence": seq,
                "concentration": concentration,
                "cell_density": cell_density,
                "incubation_hours": incubation_hours,
                "efficacy": y,
            }
        )

    df = pd.DataFrame(rows)

    out_path = Path("data/example_epitope.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
