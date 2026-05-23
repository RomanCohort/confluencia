"""
fetch_extra_genes.py — Fetch extra 14 gene expressions from cBioPortal API.

Downloads expression data for the 14 genes in the 19-gene Stepwise Cox model
that are NOT in the current 5-gene ADC target set.

Saves JSON caches to data/gene_signature/cache/{study}_expr_{SYMBOL}.json

If the API is unreachable (e.g., on AutoDL without internet), the script
exits gracefully — the training pipeline will use only the 5 cached genes.
"""

import json
import sys
import time
from pathlib import Path

import requests

CACHE_DIR = Path("data/gene_signature/cache")

# 14 extra genes (excludes TMEM65 which overlaps with the 5-gene set)
EXTRA_GENES = {
    "ERBB2": 2064, "PIK3CA": 5290, "GATA3": 2625, "MKI67": 4288,
    "BRCA1": 672, "LARP6": 55323, "ESR1": 2099, "NR1H3": 10060,
    "BAX": 581, "PSMD2": 5721, "CDH1": 999, "MTDH": 92115,
    "CASP3": 836, "AKT1": 207, "BCL2": 596, "MYC": 4609,
    "TP53": 7157, "PTEN": 5728,
}

# cBioPortal API base
CBIO_BASE = "https://www.cbioportal.org/api"

# Study molecular profile IDs
PROFILES = {
    "TCGA-BRCA": "brca_tcga_rna_seq_v2_mrna",
    "METABRIC": "brca_metabric_mrna",
}

SAMPLE_LISTS = {
    "TCGA-BRCA": "brca_tcga_all",
    "METABRIC": "brca_metabric_all",
}


def fetch_gene_expression(study: str, gene_symbol: str, entrez_id: int) -> list:
    """Fetch molecular data for one gene from cBioPortal."""
    profile_id = PROFILES.get(study)
    sample_list_id = SAMPLE_LISTS.get(study)

    if not profile_id or not sample_list_id:
        print(f"  Unknown study: {study}")
        return []

    url = f"{CBIO_BASE}/molecular-profiles/{profile_id}/molecular-data"
    params = {
        "sampleListId": sample_list_id,
        "entrezGeneId": entrez_id,
        "projection": "SUMMARY",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"  CONNECTION ERROR: Cannot reach cBioPortal API")
        print(f"  This is expected if you're offline (e.g., on AutoDL)")
        return None  # Signal to stop trying
    except requests.exceptions.HTTPError as e:
        print(f"  HTTP error for {study}_{gene_symbol}: {e}")
        return []
    except Exception as e:
        print(f"  Error fetching {study}_{gene_symbol}: {e}")
        return []


def main():
    print("=" * 60)
    print("Fetching extra gene expressions from cBioPortal")
    print(f"Genes to fetch: {len(EXTRA_GENES)}")
    print("=" * 60)

    # Check which files already exist
    already_cached = set()
    for symbol in EXTRA_GENES:
        tcga_path = CACHE_DIR / f"TCGA-BRCA_expr_{symbol}.json"
        metabric_path = CACHE_DIR / f"METABRIC_expr_{symbol}.json"
        if tcga_path.exists() and metabric_path.exists():
            already_cached.add(symbol)

    if already_cached:
        print(f"\nAlready cached: {sorted(already_cached)}")
        to_fetch = {k: v for k, v in EXTRA_GENES.items() if k not in already_cached}
    else:
        to_fetch = EXTRA_GENES

    if not to_fetch:
        print("\nAll extra genes already cached!")
        return

    print(f"Need to fetch: {sorted(to_fetch.keys())}")

    # Try a test connection first
    print("\nTesting cBioPortal API connection...")
    try:
        resp = requests.get(f"{CBIO_BASE}/studies", timeout=10)
        resp.raise_for_status()
        print("  Connection OK!")
    except Exception as e:
        print(f"  Cannot connect to cBioPortal: {e}")
        print("\nSkipping extra gene fetch. Training will use only 5 cached genes.")
        print("You can re-run this script when internet is available.")
        return

    # Fetch each gene for each study
    new_count = 0
    api_unreachable = False

    for symbol, entrez_id in sorted(to_fetch.items()):
        if api_unreachable:
            break

        for study in ["TCGA-BRCA", "METABRIC"]:
            out_path = CACHE_DIR / f"{study}_expr_{symbol}.json"
            if out_path.exists():
                continue

            print(f"\n  Fetching {study}_{symbol} (entrez={entrez_id})...")
            data = fetch_gene_expression(study, symbol, entrez_id)

            if data is None:
                # Connection failed — stop all fetching
                api_unreachable = True
                break

            if data:
                with open(out_path, "w") as f:
                    json.dump(data, f)
                print(f"    Saved {len(data)} samples to {out_path}")
                new_count += 1
            else:
                print(f"    No data returned")

            # Rate limit: be nice to the API
            time.sleep(0.5)

    print(f"\n--- Summary ---")
    print(f"New files fetched: {new_count}")

    # Check what we have now
    available = []
    for symbol in EXTRA_GENES:
        tcga = (CACHE_DIR / f"TCGA-BRCA_expr_{symbol}.json").exists()
        metabric = (CACHE_DIR / f"METABRIC_expr_{symbol}.json").exists()
        if tcga and metabric:
            available.append(symbol)

    print(f"Extra genes with both cohorts: {len(available)}/{len(EXTRA_GENES)}")
    if available:
        print(f"  Available: {sorted(available)}")

    missing = [s for s in EXTRA_GENES if s not in available]
    if missing:
        print(f"  Missing: {sorted(missing)}")
        print("  Training will proceed with available genes only.")

    print("\nDone!")


if __name__ == "__main__":
    main()
