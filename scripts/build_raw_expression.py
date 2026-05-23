"""
build_raw_expression.py — Build raw (non-normalized) gene expression CSVs
from cached cBioPortal JSON data.

Outputs:
  data/gene_signature/cache/tcga_raw_expr.csv
  data/gene_signature/cache/metabric_raw_expr.csv
  data/gene_signature/cache/combined_raw_with_survival.csv

Key difference from combined_five_gene.csv: values are log2-transformed
raw expression (NOT min-max normalized to 0-1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_DIR = Path("data/gene_signature/cache")

# Gene symbols → JSON filename mapping
GENE_SYMBOLS = ["TACSTD2", "PVRL4", "SLC39A8", "VTCN1", "TMEM65"]
GENE_DISPLAY = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]

# 14 extra genes from the 19-gene Stepwise Cox model
EXTRA_GENES = {
    "ERBB2": 2064, "PIK3CA": 5290, "GATA3": 2625, "MKI67": 4288,
    "BRCA1": 672, "LARP6": 55323, "ESR1": 2099, "NR1H3": 10060,
    "BAX": 581, "PSMD2": 5721, "CDH1": 999, "MTDH": 92115,
    "CASP3": 836, "AKT1": 207, "BCL2": 596, "MYC": 4609,
    "TP53": 7157, "PTEN": 5728,
}

# Clinical attributes to extract from METABRIC JSON
CLINICAL_ATTRS = ["ER_STATUS", "HER2_STATUS", "PR_STATUS", "GRADE",
                  "TUMOR_STAGE", "TUMOR_SIZE"]


def load_expr_json(study: str, gene: str) -> dict:
    """Load expression JSON, return {sample_id: raw_value}."""
    path = CACHE_DIR / f"{study}_expr_{gene}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {d["sampleId"]: d["value"] for d in data if "value" in d and d["value"] is not None}


def build_expr_df(study: str, gene_symbols: list, gene_display: list,
                  transform: str = "log2") -> pd.DataFrame:
    """Build expression DataFrame from JSON caches.

    Parameters
    ----------
    study : str
        e.g., "TCGA-BRCA" or "METABRIC"
    gene_symbols : list
        Gene symbols matching JSON filenames (e.g., "TACSTD2")
    gene_display : list
        Display names for columns (e.g., "TROP2")
    transform : str
        "log2" for TCGA (RSEM), "none" for METABRIC (already log2)
    """
    all_samples = None
    gene_data = {}

    for symbol, display in zip(gene_symbols, gene_display):
        vals = load_expr_json(study, symbol)
        if not vals:
            print(f"  WARNING: No data for {study}_{symbol}")
            continue
        gene_data[display] = vals
        if all_samples is None:
            all_samples = set(vals.keys())
        else:
            all_samples &= set(vals.keys())

    if not gene_data or all_samples is None:
        return pd.DataFrame()

    # Build DataFrame
    rows = []
    for sid in sorted(all_samples):
        row = {"sample_id": sid}
        for display, vals in gene_data.items():
            v = vals.get(sid, np.nan)
            if transform == "log2" and not np.isnan(v):
                v = np.log2(v + 1)
            row[display] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {study}: {len(df)} samples, {len(gene_data)} genes")
    return df


def extract_clinical_from_json(study: str) -> pd.DataFrame:
    """Extract clinical attributes from cached JSON."""
    path = CACHE_DIR / f"{study}_clinical.json"
    if not path.exists():
        print(f"  WARNING: No clinical JSON for {study}")
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    # Pivot: one row per sample, one column per attribute
    records = {}
    for item in data:
        sid = item.get("sampleId", "")
        attr = item.get("clinicalAttributeId", "")
        val = item.get("value", None)
        if sid and attr in CLINICAL_ATTRS:
            if sid not in records:
                records[sid] = {"sample_id": sid}
            records[sid][attr] = val

    df = pd.DataFrame.from_dict(records, orient="index")
    if "sample_id" not in df.columns:
        return pd.DataFrame()

    # Convert to numeric/binary
    if "GRADE" in df.columns:
        df["grade"] = pd.to_numeric(df["GRADE"], errors="coerce")
    if "ER_STATUS" in df.columns:
        df["ER_positive"] = (df["ER_STATUS"] == "Positive").astype(float)
        df.loc[df["ER_STATUS"].isna(), "ER_positive"] = np.nan
    if "HER2_STATUS" in df.columns:
        df["HER2_positive"] = (df["HER2_STATUS"] == "Positive").astype(float)
        df.loc[df["HER2_STATUS"].isna(), "HER2_positive"] = np.nan
    if "PR_STATUS" in df.columns:
        df["PR_positive"] = (df["PR_STATUS"] == "Positive").astype(float)
        df.loc[df["PR_STATUS"].isna(), "PR_positive"] = np.nan
    if "TUMOR_STAGE" in df.columns:
        df["tumor_stage"] = pd.to_numeric(df["TUMOR_STAGE"], errors="coerce")
    if "TUMOR_SIZE" in df.columns:
        df["tumor_size"] = pd.to_numeric(df["TUMOR_SIZE"], errors="coerce")

    # Keep only processed columns
    keep = ["sample_id", "grade", "ER_positive", "HER2_positive",
            "PR_positive", "tumor_stage", "tumor_size"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)

    print(f"  {study} clinical (sample-level): {len(df)} samples, {len(keep)-1} features")
    return df


def extract_tcga_patient_clinical() -> pd.DataFrame:
    """Extract TCGA patient-level clinical from fetched JSON.

    TCGA stores ER/HER2/PR/GRADE at the patient level (not sample level).
    We merge via patient_id (first 12 chars of TCGA sample barcodes).
    """
    path = CACHE_DIR / "TCGA-BRCA_patient_clinical.json"
    if not path.exists():
        print("  WARNING: No TCGA patient clinical JSON")
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    # Relevant patient-level attributes
    tcga_attrs = {
        "ER_STATUS_BY_IHC": "ER_STATUS",
        "PR_STATUS_BY_IHC": "PR_STATUS",
        "IHC_HER2": "HER2_STATUS",
        "GRADE": "GRADE",
        "AJCC_TUMOR_PATHOLOGIC_PT": "TUMOR_STAGE",
    }

    # Pivot: one row per patient
    records = {}
    for item in data:
        pid = item.get("patientId", "")
        attr = item.get("clinicalAttributeId", "")
        val = item.get("value", None)
        mapped = tcga_attrs.get(attr)
        if pid and mapped and val is not None:
            if pid not in records:
                records[pid] = {"patient_id": pid}
            records[pid][mapped] = val

    df = pd.DataFrame.from_dict(records, orient="index")
    if "patient_id" not in df.columns:
        return pd.DataFrame()

    # Convert
    if "GRADE" in df.columns:
        df["grade"] = pd.to_numeric(df["GRADE"], errors="coerce")
    if "ER_STATUS" in df.columns:
        df["ER_positive"] = (df["ER_STATUS"] == "Positive").astype(float)
        df.loc[df["ER_STATUS"].isna(), "ER_positive"] = np.nan
    if "HER2_STATUS" in df.columns:
        df["HER2_positive"] = (df["HER2_STATUS"] == "Positive").astype(float)
        df.loc[df["HER2_STATUS"].isna(), "HER2_positive"] = np.nan
    if "PR_STATUS" in df.columns:
        df["PR_positive"] = (df["PR_STATUS"] == "Positive").astype(float)
        df.loc[df["PR_STATUS"].isna(), "PR_positive"] = np.nan
    if "TUMOR_STAGE" in df.columns:
        # Parse AJCC stage strings like "T2", "T3c", "T1a"
        import re
        def parse_ajcc(s):
            if pd.isna(s):
                return np.nan
            m = re.match(r"T(\d)", str(s))
            return float(m.group(1)) if m else np.nan
        df["tumor_stage"] = df["TUMOR_STAGE"].apply(parse_ajcc)

    # IHC_HER2 may have different value encoding
    if "IHC_HER2" in df.columns and "HER2_positive" not in df.columns:
        # IHC_HER2 values: "Positive", "Negative", "Equivocal"
        df["HER2_positive"] = (df["IHC_HER2"] == "Positive").astype(float)
        df.loc[df["IHC_HER2"].isna(), "HER2_positive"] = np.nan

    keep = ["patient_id", "grade", "ER_positive", "HER2_positive",
            "PR_positive", "tumor_stage"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)

    print(f"  TCGA patient clinical: {len(df)} patients, {len(keep)-1} features")
    return df


def main():
    print("=" * 60)
    print("Building raw expression matrix from JSON caches")
    print("=" * 60)

    # --- Build expression DataFrames ---
    print("\n--- TCGA-BRCA (log2 RSEM + 1) ---")
    tcga_expr = build_expr_df("TCGA-BRCA", GENE_SYMBOLS, GENE_DISPLAY, transform="log2")

    print("\n--- METABRIC (already log2 microarray) ---")
    metabric_expr = build_expr_df("METABRIC", GENE_SYMBOLS, GENE_DISPLAY, transform="none")

    # --- Also check for extra gene JSONs ---
    extra_symbols = list(EXTRA_GENES.keys())
    extra_display = extra_symbols  # Use gene symbol as column name
    extra_found_tcga = []
    extra_found_metabric = []

    for sym in extra_symbols:
        tcga_path = CACHE_DIR / f"TCGA-BRCA_expr_{sym}.json"
        metabric_path = CACHE_DIR / f"METABRIC_expr_{sym}.json"
        if tcga_path.exists():
            extra_found_tcga.append(sym)
        if metabric_path.exists():
            extra_found_metabric.append(sym)

    if extra_found_tcga:
        print(f"\n--- Extra genes found for TCGA: {extra_found_tcga} ---")
        tcga_extra = build_expr_df("TCGA-BRCA", extra_found_tcga, extra_found_tcga, transform="log2")
        tcga_expr = tcga_expr.merge(tcga_extra, on="sample_id", how="inner")

    if extra_found_metabric:
        print(f"\n--- Extra genes found for METABRIC: {extra_found_metabric} ---")
        metabric_extra = build_expr_df("METABRIC", extra_found_metabric, extra_found_metabric, transform="none")
        metabric_expr = metabric_expr.merge(metabric_extra, on="sample_id", how="inner")

    # --- Add clinical features ---
    print("\n--- Clinical features ---")
    tcga_clin = extract_clinical_from_json("TCGA-BRCA")
    metabric_clin = extract_clinical_from_json("METABRIC")

    # TCGA patient-level clinical (ER/HER2/PR/GRADE only available at patient level)
    tcga_patient_clin = extract_tcga_patient_clinical()

    # --- Add survival data ---
    tcga_surv = pd.read_csv(CACHE_DIR / "tcga_survival.csv")
    metabric_surv = pd.read_csv(CACHE_DIR / "metabric_survival.csv")

    # Merge TCGA
    tcga = tcga_expr.merge(tcga_surv, on="sample_id", how="inner")
    tcga["patient_id"] = tcga["sample_id"].str[:12]

    # Merge TCGA clinical: try patient-level first, then sample-level fallback
    if not tcga_patient_clin.empty:
        tcga = tcga.merge(tcga_patient_clin, on="patient_id", how="left")
        print(f"  TCGA patient clinical merged: {tcga['ER_positive'].notna().sum()} have ER status")
    if not tcga_clin.empty:
        # Fill missing from patient-level with sample-level data
        tcga_clin["patient_id"] = tcga_clin["sample_id"].str[:12]
        for col in ["grade", "ER_positive", "HER2_positive", "PR_positive", "tumor_stage"]:
            if col in tcga_clin.columns and col in tcga.columns:
                tcga[col] = tcga[col].fillna(
                    tcga.merge(tcga_clin[["patient_id", col]], on="patient_id", how="left")[f"{col}_y"]
                )
    tcga["source"] = "TCGA-BRCA"

    # Merge METABRIC
    metabric = metabric_expr.merge(metabric_surv, on="sample_id", how="inner")
    if not metabric_clin.empty:
        metabric = metabric.merge(metabric_clin, on="sample_id", how="left")
    metabric["source"] = "METABRIC"
    metabric["patient_id"] = metabric["sample_id"]

    # --- Combine ---
    # Ensure same columns
    all_cols = set(tcga.columns) | set(metabric.columns)
    for col in all_cols:
        if col not in tcga.columns:
            tcga[col] = np.nan
        if col not in metabric.columns:
            metabric[col] = np.nan

    combined = pd.concat([tcga, metabric], ignore_index=True)

    # Column order
    gene_cols = [c for c in combined.columns if c in GENE_DISPLAY + extra_symbols]
    clin_cols = ["grade", "ER_positive", "HER2_positive", "PR_positive",
                 "tumor_stage", "tumor_size"]
    clin_cols = [c for c in clin_cols if c in combined.columns]
    meta_cols = ["sample_id", "patient_id", "source", "OS_months", "OS_status"]

    ordered = meta_cols + gene_cols + clin_cols
    ordered = [c for c in ordered if c in combined.columns]
    combined = combined[ordered]

    # --- Save ---
    tcga_out = CACHE_DIR / "tcga_raw_expr.csv"
    metabric_out = CACHE_DIR / "metabric_raw_expr.csv"
    combined_out = CACHE_DIR / "combined_raw_with_survival.csv"

    tcga.to_csv(tcga_out, index=False)
    metabric.to_csv(metabric_out, index=False)
    combined.to_csv(combined_out, index=False)

    print(f"\n--- Output ---")
    print(f"TCGA: {tcga_out} ({len(tcga)} rows, {len(tcga.columns)} cols)")
    print(f"METABRIC: {metabric_out} ({len(metabric)} rows, {len(metabric.columns)} cols)")
    print(f"Combined: {combined_out} ({len(combined)} rows, {len(combined.columns)} cols)")

    # --- Summary stats ---
    print(f"\n--- Gene value ranges (raw log2) ---")
    for g in gene_cols:
        if g in combined.columns:
            vals = combined[g].dropna()
            if len(vals) > 0:
                print(f"  {g}: min={vals.min():.2f}, max={vals.max():.2f}, "
                      f"mean={vals.mean():.2f}, std={vals.std():.2f}")

    print(f"\n--- Clinical coverage ---")
    for c in clin_cols:
        if c in combined.columns:
            n_valid = combined[c].notna().sum()
            print(f"  {c}: {n_valid}/{len(combined)} ({100*n_valid/len(combined):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()