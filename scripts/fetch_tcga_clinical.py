"""
fetch_tcga_clinical.py — Fetch TCGA-BRCA patient-level clinical features.

The current cache only has sample-level molecular data. TCGA stores
ER_STATUS, HER2_STATUS, PR_STATUS, GRADE at the patient level.

This script fetches patient-level clinical attributes and saves them
to data/gene_signature/cache/TCGA-BRCA_patient_clinical.json

If the API is unreachable, the script exits gracefully.
"""

import json
import sys
from pathlib import Path

import requests

CACHE_DIR = Path("data/gene_signature/cache")

CBIO_BASE = "https://www.cbioportal.org/api"

# Clinical attributes we want from TCGA
WANTED_ATTRS = {
    "ER_STATUS_BY_IHC", "HER2_STATUS_BY_IHC", "PR_STATUS_BY_IHC",
    "GRADE", "HISTOLOGICAL_TYPE", "PATHOLOGY_T_STAGE",
    "AJCC_TUMOR_PATHOLOGIC_PT", "TUMOR_STAGE",
}


def fetch_patient_clinical() -> list:
    """Fetch patient-level clinical data from cBioPortal."""
    # Try patient clinical data endpoint
    url = f"{CBIO_BASE}/studies/brca_tcga/clinical-data"
    params = {
        "clinicalDataType": "PATIENT",
        "projection": "DETAILED",
        "pageSize": 50000,
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("CONNECTION ERROR: Cannot reach cBioPortal API")
        return None
    except Exception as e:
        print(f"Error fetching TCGA clinical data: {e}")
        return []


def main():
    print("=" * 60)
    print("Fetching TCGA-BRCA patient-level clinical features")
    print("=" * 60)

    out_path = CACHE_DIR / "TCGA-BRCA_patient_clinical.json"
    if out_path.exists():
        print(f"\nAlready cached: {out_path}")
        with open(out_path) as f:
            data = json.load(f)
        print(f"  {len(data)} records loaded from cache")
    else:
        print("\nFetching from cBioPortal API...")
        data = fetch_patient_clinical()

        if data is None:
            print("\nCannot connect to cBioPortal. Skipping TCGA clinical fetch.")
            print("TCGA samples will use default clinical values in training.")
            return

        if not data:
            print("\nNo data returned from API.")
            return

        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"  Saved {len(data)} records to {out_path}")

    # Parse and summarize
    print("\n--- Parsing clinical attributes ---")
    attr_counts = {}
    for item in data:
        attr = item.get("clinicalAttributeId", "")
        if attr in WANTED_ATTRS or "ER" in attr.upper() or "HER2" in attr.upper() or "PR" in attr.upper() or "GRADE" in attr.upper():
            attr_counts[attr] = attr_counts.get(attr, 0) + 1

    print("Available clinical attributes:")
    for attr, count in sorted(attr_counts.items(), key=lambda x: -x[1]):
        print(f"  {attr}: {count} patients")

    # Also check what sample-level clinical we already have
    sample_clin_path = CACHE_DIR / "TCGA-BRCA_clinical.json"
    if sample_clin_path.exists():
        with open(sample_clin_path) as f:
            sample_data = json.load(f)
        sample_attrs = set()
        for item in sample_data:
            sample_attrs.add(item.get("clinicalAttributeId", ""))
        print(f"\nSample-level clinical already has: {sorted(sample_attrs)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
