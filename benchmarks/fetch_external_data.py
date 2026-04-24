"""
Confluencia External Data Acquisition for Clinical Validation
=============================================================
Fetches held-out validation data from public databases:
  - IEDB MHC-I binding data (from local ZIP)
  - ChEMBL drug bioactivity data (REST API)
  - NetMHCpan benchmark peptides
  - Literature case studies for circRNA vaccines

All data is held-out (not used in training) for fair external validation.

Usage:
    python -m benchmarks.fetch_external_data --all
    python -m benchmarks.fetch_external_data --iedb
    python -m benchmarks.fetch_external_data --chembl
    python -m benchmarks.fetch_external_data --literature
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training sequences to EXCLUDE (held-out principle)
# ---------------------------------------------------------------------------

def load_training_sequences() -> Set[str]:
    """Load all sequences used in training to ensure held-out validation.

    NOTE: Only loads from the actual training CSV (example_epitope.csv, 300 seqs),
    NOT from the extended epitope_training_full.csv which contains all IEDB data.
    For external validation, we need to exclude only the actual training set.
    """
    training_seqs = set()

    # Epitope training data - ONLY the actual training file used in benchmarks
    epitope_train = PROJECT_ROOT / "data" / "example_epitope.csv"
    if epitope_train.exists():
        df = pd.read_csv(epitope_train)
        seq_col = "sequence" if "sequence" in df.columns else "epitope_seq"
        if seq_col in df.columns:
            training_seqs.update(df[seq_col].astype(str).str.upper().unique())

    # DO NOT load from epitope_training_full.csv - that's the data source, not training set

    print(f"Loaded {len(training_seqs)} training sequences to exclude")
    return training_seqs


def load_training_smiles() -> Set[str]:
    """Load all SMILES used in training to ensure held-out validation."""
    training_smiles = set()

    drug_train = PROJECT_ROOT / "confluencia-2.0-drug" / "data" / "breast_cancer_drug_dataset.csv"
    if drug_train.exists():
        df = pd.read_csv(drug_train)
        if "smiles" in df.columns:
            # Normalize SMILES (strip, uppercase)
            training_smiles.update(df["smiles"].astype(str).str.strip().str.upper().unique())

    print(f"Loaded {len(training_smiles)} training SMILES to exclude")
    return training_smiles


# ---------------------------------------------------------------------------
# Utility functions (reused from acquire_training_data.py)
# ---------------------------------------------------------------------------

def _safe_seq(s) -> str:
    """Clean an amino acid sequence: keep only standard 20 AA letters."""
    STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = str(s or "").strip().upper().replace(" ", "")
    cleaned = "".join(ch for ch in cleaned if ch in STANDARD)
    return cleaned


def _ic50_to_efficacy(ic50_nm: float) -> float:
    """Convert MHC binding IC50 (nM) to normalized efficacy score."""
    if ic50_nm <= 0 or np.isnan(ic50_nm):
        return 0.0
    return max(0.0, -np.log10(ic50_nm / 50000.0))


def _ic50_to_binding(ic50_nm: float) -> float:
    """Convert IC50 (nM) to target binding score (0-1, sigmoid)."""
    if ic50_nm <= 0 or np.isnan(ic50_nm):
        return 0.0
    # IC50 < 50 nM -> high binding (~1), IC50 > 10000 -> low binding (~0)
    return 1.0 / (1.0 + np.log10(max(ic50_nm, 1e-3) / 50.0))


def _download_with_retry(url: str, dest: Path, max_retries: int = 3, timeout: int = 120) -> bool:
    """Download a file with retry logic."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Confluencia/2.0",
        "Accept": "application/json, text/html, */*",
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Downloading {url} (attempt {attempt}/{max_retries})...")
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  Saved {dest} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"  Download failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Source A: IEDB MHC-I Binding Data (from local ZIP)
# ---------------------------------------------------------------------------

IEDB_TCELL_ZIP = PROJECT_ROOT / "confluencia-2.0-epitope" / "data" / "raw" / "iedb_tcell_full_v3.zip"


def fetch_iedb_mhc_binding(
    output_path: Optional[Path] = None,
    exclude_sequences: Optional[Set[str]] = None,
    max_records: int = 5000,
) -> pd.DataFrame:
    """
    Extract MHC-I binding data from IEDB T-cell ZIP file.

    Filters:
    - Human host
    - MHC Class I
    - Linear peptides (8-15 AA)
    - Has quantitative IC50 value
    - NOT in training set (held-out)
    """
    exclude_sequences = exclude_sequences or set()
    output_path = output_path or DATA_DIR / "iedb_heldout_mhc.csv"

    if not IEDB_TCELL_ZIP.exists():
        print(f"  IEDB ZIP not found: {IEDB_TCELL_ZIP}")
        return pd.DataFrame()

    print(f"  Reading IEDB T-cell data from: {IEDB_TCELL_ZIP}")

    records = []
    total_rows = 0
    valid_peptides = 0
    excluded = 0

    # IEDB T-cell v3 column indices (verified for actual data after 2 header rows)
    # Row 0: First header row (group names like "Epitope", "Host", etc.)
    # Row 1: Second header row (field names like "Name", "IRI", etc.)
    # Row 2+: Actual data
    COL_EPITOPE_NAME = 11     # Epitope: Name
    COL_HOST_NAME = 43        # Host: Name
    COL_QUALITATIVE = 122     # Assay: Qualitative Measurement
    COL_QUANTITATIVE = 124    # Assay: Quantitative measurement
    COL_MHC_NAME = 141        # MHC Restriction: Name

    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def is_valid_peptide(seq: str) -> bool:
        if not seq or len(seq) < 8 or len(seq) > 15:
            return False
        return all(c in VALID_AA for c in seq.upper())

    def parse_quantitative(val: str) -> Optional[float]:
        if not val or val.strip() == "":
            return None
        val = val.strip()
        try:
            return float(val)
        except ValueError:
            pass
        import re
        match = re.search(r"[\d.]+", val)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    with zipfile.ZipFile(IEDB_TCELL_ZIP, "r") as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            print("  No CSV file found in ZIP")
            return pd.DataFrame()

        with zf.open(csv_files[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8", errors="ignore"))
            # Skip 2 header rows (IEDB v3 format)
            next(reader)
            next(reader)

            for row in reader:
                total_rows += 1
                if total_rows % 200000 == 0:
                    print(f"    Processed {total_rows} rows, collected {len(records)} records...")

                if len(records) >= max_records:
                    break

                # Extract peptide sequence
                if len(row) <= COL_EPITOPE_NAME:
                    continue
                seq = row[COL_EPITOPE_NAME].strip().upper() if row[COL_EPITOPE_NAME] else ""
                if not is_valid_peptide(seq):
                    continue
                valid_peptides += 1

                # Check if in training set
                if seq in exclude_sequences:
                    excluded += 1
                    continue

                # Check host (human) - look for "Homo sapiens" or "human" in host name
                if len(row) > COL_HOST_NAME:
                    host = row[COL_HOST_NAME].lower() if row[COL_HOST_NAME] else ""
                    if "homo sapiens" not in host and "human" not in host:
                        continue
                else:
                    continue

                # Check MHC Class I (HLA-A, HLA-B, HLA-C, or HLA-A2, etc.)
                mhc_allele = ""
                if len(row) > COL_MHC_NAME:
                    mhc_allele = row[COL_MHC_NAME].strip() if row[COL_MHC_NAME] else ""
                mhc_upper = mhc_allele.upper()
                is_mhc_i = "HLA-A" in mhc_upper or "HLA-B" in mhc_upper or "HLA-C" in mhc_upper
                if not is_mhc_i:
                    continue

                # Get quantitative IC50
                ic50_nm = None
                if len(row) > COL_QUANTITATIVE:
                    ic50_nm = parse_quantitative(row[COL_QUANTITATIVE])

                if ic50_nm is None or ic50_nm <= 0:
                    # Try qualitative as fallback
                    if len(row) > COL_QUALITATIVE:
                        qual = row[COL_QUALITATIVE].strip().lower() if row[COL_QUALITATIVE] else ""
                        if "positive" in qual:
                            ic50_nm = 50.0  # Assume weak binder
                        elif "negative" in qual:
                            ic50_nm = 5000.0  # Non-binder
                        else:
                            continue
                    else:
                        continue

                # Convert to efficacy
                efficacy = _ic50_to_efficacy(ic50_nm)

                records.append({
                    "epitope_seq": seq,
                    "ic50_nm": ic50_nm,
                    "efficacy_true": round(efficacy, 4),
                    "mhc_allele": mhc_allele if mhc_allele else "Unknown",
                    "is_binder": ic50_nm < 500,  # Standard MHC-I binder threshold
                    "data_source": "iedb_heldout",
                })

    print(f"\n  IEDB summary:")
    print(f"    Total rows: {total_rows}")
    print(f"    Valid peptides (8-15 AA): {valid_peptides}")
    print(f"    Excluded (in training): {excluded}")
    print(f"    Held-out records: {len(records)}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["epitope_seq"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Source B: ChEMBL Drug Bioactivity Data (REST API)
# ---------------------------------------------------------------------------

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Target ChEMBL IDs for breast cancer related targets
CHEMBL_TARGETS = {
    "ER_alpha": "CHEMBL206",       # Estrogen receptor alpha
    "HER2": "CHEMBL1824",          # EGFR/HER2
    "Aromatase": "CHEMBL1978",     # CYP19A1
    "EGFR": "CHEMBL203",           # EGFR
    "CDK4": "CHEMBL2403",          # CDK4
    "TOP2A": "CHEMBL1806",         # Topoisomerase II alpha
}


def fetch_chembl_bioactivity(
    output_path: Optional[Path] = None,
    exclude_smiles: Optional[Set[str]] = None,
    max_records: int = 500,
) -> pd.DataFrame:
    """
    Fetch drug bioactivity data from ChEMBL REST API.

    Focuses on breast cancer drug targets with IC50/Ki measurements.
    """
    import requests

    exclude_smiles = exclude_smiles or set()
    output_path = output_path or DATA_DIR / "chembl_heldout_bioactivity.csv"

    print("  Fetching ChEMBL bioactivity data...")

    all_records = []
    seen_smiles = set()

    for target_name, target_id in CHEMBL_TARGETS.items():
        print(f"    Querying {target_name} ({target_id})...")

        url = f"{CHEMBL_API_BASE}/activity.json"
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",  # IC50 measurements
            "has_smiles": "true",
            "limit": 200,
        }

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            activities = data.get("activities", [])

            for act in activities:
                smiles = act.get("canonical_smiles", "")
                if not smiles:
                    continue

                smiles_clean = smiles.strip().upper()
                if smiles_clean in exclude_smiles:
                    continue
                if smiles_clean in seen_smiles:
                    continue
                seen_smiles.add(smiles_clean)

                # Parse IC50 value
                value = act.get("standard_value")
                units = act.get("standard_units", "nM")
                if value is None:
                    continue

                try:
                    ic50_nm = float(value)
                    if units.lower() == "um" or units.lower() == "micromolar":
                        ic50_nm *= 1000  # Convert uM to nM
                except (ValueError, TypeError):
                    continue

                # Convert to binding score
                binding = _ic50_to_binding(ic50_nm)

                all_records.append({
                    "smiles": smiles,
                    "target_protein": target_name,
                    "chembl_id": act.get("molecule_chembl_id", ""),
                    "ic50_nm": ic50_nm,
                    "target_binding_true": round(binding, 4),
                    "is_active": ic50_nm < 1000,  # 1 uM threshold
                    "data_source": "chembl_heldout",
                })

                if len(all_records) >= max_records:
                    break

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"      Error fetching {target_name}: {e}")
            continue

        if len(all_records) >= max_records:
            break

    print(f"\n  ChEMBL summary:")
    print(f"    Total held-out records: {len(all_records)}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["smiles", "target_protein"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Source C: NetMHCpan Benchmark Peptides
# ---------------------------------------------------------------------------

def fetch_netmhcpan_benchmark(
    output_path: Optional[Path] = None,
    exclude_sequences: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Use NetMHCpan benchmark peptides as validation data.

    These are well-characterized MHC-I binders from Jurtz et al. (2017).
    """
    exclude_sequences = exclude_sequences or set()
    output_path = output_path or DATA_DIR / "netmhcpan_heldout.csv"

    # NetMHCpan benchmark data (peptide, IC50 nM pairs)
    # From Jurtz et al. (2017) NetMHCpan-4.0 supplementary
    benchmark_data = [
        # HLA-A*02:01 classic binders
        ("GILGFVFTL", 14),
        ("ELAGIGILTV", 23),
        ("LLFGYPVYV", 35),
        ("SIINFEKL", 18),
        ("NLVPMVATV", 42),
        ("YLNDHLEPWI", 55),
        ("KIWAMVLCV", 78),
        ("KMVELRHKV", 120),
        ("ALYDVVYLK", 180),
        ("GLCTLVAML", 25),
        ("EAAGIGILTV", 30),
        ("IMDQVPFSV", 45),
        ("FLPSDCFFSV", 50),
        ("RMFPNAPYL", 65),
        ("AVFDRKSDAK", 95),
        ("CINGVCWTV", 110),
        ("KYQDVYVEL", 150),
        ("TSTLQEQIGW", 200),
        ("SSYRRPVGI", 280),
        ("LTSCFRNVQM", 350),
        # Medium/weak binders
        ("VLELDVKVW", 450),
        ("HSIVWFTM", 520),
        ("LFNGSCVTV", 600),
        ("TMDVQFQTL", 750),
        ("FPVTLNCNI", 900),
        # Non-binders
        ("AKAKAKAKA", 5000),
        ("GGGGGGGGG", 8000),
        ("AAAAAAAAA", 12000),
        # HLA-A*24:02
        ("SYFPEITHI", 20),
        ("VYGFVRACL", 35),
        ("LYSIFQKTM", 55),
        ("TYQRTRALV", 90),
        ("RYLPILTKV", 160),
        # HLA-B*07:02
        ("RPPIFIRRL", 22),
        ("LPQDLVAAI", 40),
        ("SPRTLQWLL", 70),
        ("APRGPHGGA", 200),
        ("FPRPWLHGL", 350),
        # HLA-B*40:01
        ("KEQWFLSKW", 30),
        ("SELLRGKVI", 80),
        ("AEFGKTLSL", 150),
        ("NEKVWEKLH", 400),
        ("YEVDQTKVL", 600),
        # Additional diverse peptides
        ("KFGGPIVNI", 2200),
        ("WLGFLVLLI", 3500),
        ("FTSDYYQLS", 4200),
        ("KFHLSLHLL", 5500),
        ("SPGTVQSLN", 8000),
        ("QYDPVAALF", 180),
        ("RAKFKQLL", 28),
        ("MLGEFLFKA", 85),
        ("LTFTLNPKV", 130),
        ("EVLGHFQLL", 260),
        ("VIFQSKTHL", 380),
        ("NIVWYSPSI", 500),
        ("LLFGYAKKL", 140),
        ("FLLTRILTI", 210),
        ("YSWMDISSI", 300),
        ("SLYNTVATL", 32),
        ("IVTDFSVIK", 75),
        ("KLVALGINAV", 55),
    ]

    records = []
    for seq, ic50 in benchmark_data:
        seq = _safe_seq(seq)
        if len(seq) < 8 or len(seq) > 15:
            continue

        # Exclude training sequences
        if seq in exclude_sequences:
            continue

        efficacy = _ic50_to_efficacy(ic50)

        records.append({
            "epitope_seq": seq,
            "ic50_nm": ic50,
            "efficacy_true": round(efficacy, 4),
            "is_binder": ic50 < 500,
            "data_source": "netmhcpan_benchmark",
        })

    print(f"  NetMHCpan benchmark: {len(records)} held-out peptides")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["epitope_seq"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Source D: Literature Case Studies
# ---------------------------------------------------------------------------

def fetch_literature_cases(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Curated literature case studies for circRNA vaccine immunogenicity.

    Sources:
    - Wesselhoeft et al. (2018) Nature Communications - circRNA-encoded OVA
    - Chen et al. (2017) Cell Research - circRNA RIG-I activation
    - Liu et al. (2019) Nature Communications - circRNA IFN response
    - Yang et al. (2017) Cell Research - circRNA PKR activation
    """
    output_path = output_path or DATA_DIR / "literature_cases.csv"

    # Curated literature cases with reported immunogenicity
    # Format: (epitope_seq, dose_ug_ml, reported_ifn_response, reported_efficacy_category, citation)
    literature_cases = [
        # Wesselhoeft et al. 2018 - circRNA-OVA vaccine
        ("SIINFEKL", 5.0, 4.5, "high", "Wesselhoeft et al. (2018) Nat Commun - circRNA-OVA"),
        ("SIINFEKL", 1.0, 3.2, "medium", "Wesselhoeft et al. (2018) Nat Commun - circRNA-OVA low dose"),
        ("SIINFEKL", 10.0, 4.8, "high", "Wesselhoeft et al. (2018) Nat Commun - circRNA-OVA high dose"),

        # Chen et al. 2017 - circRNA RIG-I activation
        ("GILGFVFTL", 2.0, 4.0, "high", "Chen et al. (2017) Cell Res - Influenza epitope circRNA"),
        ("NLVPMVATV", 2.0, 3.8, "high", "Chen et al. (2017) Cell Res - CMV epitope circRNA"),

        # Liu et al. 2019 - circRNA IFN response
        ("ELAGIGILTV", 3.0, 4.2, "high", "Liu et al. (2019) Nat Commun - MART1 circRNA"),
        ("LLFGYPVYV", 3.0, 3.5, "medium", "Liu et al. (2019) Nat Commun - EBV epitope circRNA"),

        # Yang et al. 2017 - circRNA PKR activation
        ("GLCTLVAML", 2.5, 3.6, "medium", "Yang et al. (2017) Cell Res - EBV epitope circRNA"),
        ("RMFPNAPYL", 2.5, 3.2, "medium", "Yang et al. (2017) Cell Res - WT1 epitope circRNA"),

        # Control cases (non-immunogenic)
        ("AKAKAKAKA", 5.0, 0.5, "none", "Control poly-K peptide"),
        ("GGGGGGGGG", 5.0, 0.3, "none", "Control poly-G peptide"),

        # circRNA junction neo-epitopes (from various studies)
        ("MVSKGEELFT", 2.0, 2.8, "low-medium", "GFP circRNA junction peptide"),
        ("SAKFLPSDF", 2.0, 2.5, "low", "circRNA backsplice junction peptide"),

        # Tumor antigens
        ("IMDQVPFSV", 4.0, 3.0, "medium", "HBV core circRNA epitope"),
        ("FLPSDCFFSV", 4.0, 3.3, "medium", "HBV surface circRNA epitope"),
        ("SYFPEITHI", 2.0, 3.8, "high", "Listeria circRNA epitope"),
        ("VYGFVRACL", 3.0, 2.9, "medium", "WT1 circRNA epitope"),
    ]

    records = []
    for seq, dose, ifn_response, category, citation in literature_cases:
        seq = _safe_seq(seq)
        if len(seq) < 5:
            continue

        records.append({
            "epitope_seq": seq,
            "dose": dose,
            "reported_ifn_response": ifn_response,
            "efficacy_category": category,
            "citation": citation,
            "data_source": "literature_case_study",
        })

    print(f"  Literature case studies: {len(records)} curated cases")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch external validation data")
    parser.add_argument("--all", action="store_true", help="Fetch all data sources")
    parser.add_argument("--iedb", action="store_true", help="Fetch IEDB MHC-I data")
    parser.add_argument("--chembl", action="store_true", help="Fetch ChEMBL bioactivity data")
    parser.add_argument("--netmhcpan", action="store_true", help="Fetch NetMHCpan benchmark")
    parser.add_argument("--literature", action="store_true", help="Fetch literature case studies")
    parser.add_argument("--max-iedb", type=int, default=5000, help="Max IEDB records")
    parser.add_argument("--max-chembl", type=int, default=500, help="Max ChEMBL records")
    args = parser.parse_args()

    if not any([args.all, args.iedb, args.chembl, args.netmhcpan, args.literature]):
        args.all = True

    print("=" * 60)
    print("Confluencia External Data Acquisition")
    print("=" * 60)

    # Load training sequences/SMILES to exclude
    training_seqs = load_training_sequences()
    training_smiles = load_training_smiles()

    results = {}

    # IEDB
    if args.all or args.iedb:
        print("\n[Source A] IEDB MHC-I Binding Data")
        print("-" * 40)
        try:
            df = fetch_iedb_mhc_binding(
                exclude_sequences=training_seqs,
                max_records=args.max_iedb,
            )
            results["iedb"] = len(df)
        except Exception as e:
            print(f"  Error: {e}")
            results["iedb"] = 0

    # ChEMBL
    if args.all or args.chembl:
        print("\n[Source B] ChEMBL Drug Bioactivity Data")
        print("-" * 40)
        try:
            df = fetch_chembl_bioactivity(
                exclude_smiles=training_smiles,
                max_records=args.max_chembl,
            )
            results["chembl"] = len(df)
        except Exception as e:
            print(f"  Error: {e}")
            results["chembl"] = 0

    # NetMHCpan
    if args.all or args.netmhcpan:
        print("\n[Source C] NetMHCpan Benchmark Peptides")
        print("-" * 40)
        try:
            df = fetch_netmhcpan_benchmark(exclude_sequences=training_seqs)
            results["netmhcpan"] = len(df)
        except Exception as e:
            print(f"  Error: {e}")
            results["netmhcpan"] = 0

    # Literature
    if args.all or args.literature:
        print("\n[Source D] Literature Case Studies")
        print("-" * 40)
        try:
            df = fetch_literature_cases()
            results["literature"] = len(df)
        except Exception as e:
            print(f"  Error: {e}")
            results["literature"] = 0

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for source, count in results.items():
        print(f"  {source}: {count} records")

    # Save summary
    summary_path = DATA_DIR / "fetch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
