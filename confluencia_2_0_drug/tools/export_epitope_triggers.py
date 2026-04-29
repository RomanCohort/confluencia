from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.immune_abm import export_netlogo_trigger_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predicted epitopes to NetLogo trigger CSV")
    parser.add_argument("--input", required=True, help="Input CSV with epitope_seq, dose, treatment_time columns")
    parser.add_argument("--output", required=True, help="Output trigger CSV for NetLogo")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    trig = export_netlogo_trigger_csv(df, str(out))
    print(f"exported {len(trig)} trigger events to {out}")


if __name__ == "__main__":
    main()
