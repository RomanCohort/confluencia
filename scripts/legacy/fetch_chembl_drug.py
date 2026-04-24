"""
Fetch drug bioactivity data from ChEMBL REST API.

Target: 新建文件夹/data/drug_chembl.csv

Queries ChEMBL for cancer-related compounds with IC50/Ki data,
extracts SMILES and pChEMBL values, and maps to 1.0 drug format.

Output columns: smiles, dose, freq, efficacy
  - smiles: from ChEMBL canonical SMILES
  - dose/freq: filled with default values (ChEMBL doesn't provide dosing)
  - efficacy: derived from pChEMBL value (already -log10 normalized)
"""

import json
import urllib.request
import urllib.parse
import time
from pathlib import Path
import pandas as pd
import numpy as np

DST = Path(r"D:\IGEM集成方案\新建文件夹\data\drug_chembl.csv")

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Cancer-related target ChEMBL IDs for querying
# Focus on breast cancer and common cancer targets
TARGET_IDS = [
    "CHEMBL203",   # EGFR (Epidermal Growth Factor Receptor)
    "CHEMBL240",   # HER2
    "CHEMBL279",   # VEGFR2
    "CHEMBL4016",  # CDK2
    "CHEMBL301",   # Estrogen receptor alpha
    "CHEMBL206",   # BRCA1 (limited data)
]

# Standard types to query
STANDARD_TYPES = ["IC50", "Ki"]

DEFAULT_DOSE = 10.0    # mg placeholder
DEFAULT_FREQ = 1.0     # times/day placeholder


def fetch_chembl_activities(target_chembl_id: str, standard_type: str = "IC50",
                            limit: int = 5000) -> list[dict]:
    """Fetch activity data from ChEMBL API for a given target."""
    all_records = []
    offset = 0
    page_size = 1000

    while offset < limit:
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": standard_type,
            "has_smiles": "true",
            "format": "json",
            "limit": page_size,
            "offset": offset,
        }
        url = f"{CHEMBL_API}/activity.json?{urllib.parse.urlencode(params)}"

        print(f"  Fetching {target_chembl_id} {standard_type} offset={offset}...")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            if act.get("canonical_smiles") and act.get("pchembl_value"):
                try:
                    pchembl = float(act["pchembl_value"])
                    all_records.append({
                        "smiles": act["canonical_smiles"],
                        "pchembl_value": pchembl,
                        "standard_type": act.get("standard_type", ""),
                        "target_chembl_id": target_chembl_id,
                    })
                except (ValueError, TypeError):
                    continue

        offset += page_size
        if len(activities) < page_size:
            break
        time.sleep(0.5)  # rate limiting

    return all_records


def main():
    print("Fetching drug bioactivity data from ChEMBL API...")
    print(f"  Targets: {TARGET_IDS}")
    print(f"  Standard types: {STANDARD_TYPES}")

    all_records = []
    for target_id in TARGET_IDS:
        for stype in STANDARD_TYPES:
            records = fetch_chembl_activities(target_id, stype, limit=5000)
            all_records.extend(records)
            print(f"    Got {len(records)} records for {target_id} {stype}")

    if not all_records:
        print("\nNo records fetched from ChEMBL. Check network connectivity.")
        print("Creating empty placeholder file.")
        DST.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["smiles", "dose", "freq", "efficacy"]).to_csv(DST, index=False)
        return

    df = pd.DataFrame(all_records)
    print(f"\nTotal records fetched: {len(df)}")

    # Deduplicate by SMILES (keep highest pChEMBL value)
    df = df.sort_values("pchembl_value", ascending=False)
    df = df.drop_duplicates(subset=["smiles"], keep="first")
    print(f"After SMILES dedup: {len(df)}")

    # Map to 1.0 format
    # pChEMBL is -log10(IC50/Ki/Kd in M), typically range 4-10
    # Higher pChEMBL = stronger binding = higher efficacy
    # Normalize to [0, 1] range using common pChEMBL range
    pchembl_min = df["pchembl_value"].min()
    pchembl_max = df["pchembl_value"].max()
    print(f"pChEMBL range: [{pchembl_min:.2f}, {pchembl_max:.2f}]")

    # Use min-max normalization
    df["efficacy"] = (df["pchembl_value"] - pchembl_min) / max(pchembl_max - pchembl_min, 1e-6)

    out = pd.DataFrame({
        "smiles": df["smiles"],
        "dose": DEFAULT_DOSE,
        "freq": DEFAULT_FREQ,
        "efficacy": df["efficacy"].round(6),
    })

    # Filter out invalid SMILES
    out = out.dropna(subset=["smiles", "efficacy"])
    out = out[out["smiles"].str.len() > 0]

    # Reset index
    out = out.reset_index(drop=True)

    # Save
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"\nSaved {len(out)} rows to {DST}")
    print(f"  Columns: {list(out.columns)}")
    print(f"  Efficacy range: [{out['efficacy'].min():.4f}, {out['efficacy'].max():.4f}]")
    print(f"  Unique SMILES: {out['smiles'].nunique()}")


if __name__ == "__main__":
    main()
