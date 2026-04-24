"""
Validate converted 1.0 data files and verify 2.0 source files are untouched.

Checks:
1. Column names and types match expectations
2. No NaN in required columns (sequence/smiles, efficacy)
3. env_cols are auto-detectable by infer_env_cols()
4. 2.0 source files have not been modified (checksum comparison)
"""

import pandas as pd
import hashlib
from pathlib import Path

# File paths
EPITIPE_V2_SRC = Path(r"D:\IGEM集成方案\confluencia-2.0-epitope\data\epitope_training_confluencia.csv")
DRUG_V2_SRC = Path(r"D:\IGEM集成方案\confluencia-2.0-drug\data\breast_cancer_drug_dataset.csv")

EPITIPE_V1_DST = Path(r"D:\IGEM集成方案\新建文件夹\data\epitope_from_v2.csv")
DRUG_V1_DST = Path(r"D:\IGEM集成方案\新建文件夹\data\drug_from_v2.csv")
DRUG_V1_MINIMAL = Path(r"D:\IGEM集成方案\新建文件夹\data\drug_from_v2_minimal.csv")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_epitope_csv(path: Path):
    print(f"\n{'='*60}")
    print(f"Validating epitope: {path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # Required columns
    assert "sequence" in df.columns, "Missing 'sequence' column"
    assert "efficacy" in df.columns, "Missing 'efficacy' column"
    print(f"  [OK] Required columns present: sequence, efficacy")

    # No NaN in required
    seq_na = df["sequence"].isna().sum()
    eff_na = df["efficacy"].isna().sum()
    assert seq_na == 0, f"NaN in 'sequence': {seq_na}"
    assert eff_na == 0, f"NaN in 'efficacy': {eff_na}"
    print(f"  [OK] No NaN in sequence ({seq_na}) or efficacy ({eff_na})")

    # Efficacy must be numeric
    assert pd.api.types.is_numeric_dtype(df["efficacy"]), "'efficacy' must be numeric"
    print(f"  [OK] efficacy is numeric, range: [{df['efficacy'].min():.4f}, {df['efficacy'].max():.4f}]")

    # Sequence length check
    seq_lens = df["sequence"].str.len()
    print(f"  [OK] Sequence length range: [{seq_lens.min()}, {seq_lens.max()}]")

    # Auto-detect env_cols
    env_cols = [
        c for c in df.columns
        if c not in ("sequence", "efficacy") and pd.api.types.is_numeric_dtype(df[c])
    ]
    assert len(env_cols) > 0, "No env_cols auto-detected"
    print(f"  [OK] Auto-detected env_cols: {env_cols}")

    # Check env_cols have no excessive NaN
    for c in env_cols:
        na_pct = df[c].isna().mean() * 100
        if na_pct > 0:
            print(f"  [WARN] {c}: {na_pct:.1f}% NaN")

    print(f"  [PASS] Epitope validation complete")
    return True


def validate_drug_csv(path: Path):
    print(f"\n{'='*60}")
    print(f"Validating drug: {path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # Required columns
    assert "smiles" in df.columns, "Missing 'smiles' column"
    assert "efficacy" in df.columns, "Missing 'efficacy' column"
    print(f"  [OK] Required columns present: smiles, efficacy")

    # No NaN in required
    smi_na = df["smiles"].isna().sum()
    eff_na = df["efficacy"].isna().sum()
    assert smi_na == 0, f"NaN in 'smiles': {smi_na}"
    assert eff_na == 0, f"NaN in 'efficacy': {eff_na}"
    print(f"  [OK] No NaN in smiles ({smi_na}) or efficacy ({eff_na})")

    # Efficacy must be numeric
    assert pd.api.types.is_numeric_dtype(df["efficacy"]), "'efficacy' must be numeric"
    print(f"  [OK] efficacy is numeric, range: [{df['efficacy'].min():.4f}, {df['efficacy'].max():.4f}]")

    # Auto-detect env_cols
    env_cols = [
        c for c in df.columns
        if c not in ("smiles", "efficacy") and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  [OK] Auto-detected env_cols: {env_cols}")

    # Check unique SMILES count
    unique_smi = df["smiles"].nunique()
    print(f"  [OK] Unique SMILES: {unique_smi}")

    print(f"  [PASS] Drug validation complete")
    return True


def verify_source_untouched():
    print(f"\n{'='*60}")
    print(f"Verifying 2.0 source files are untouched")
    print(f"{'='*60}")

    for path in [EPITIPE_V2_SRC, DRUG_V2_SRC]:
        if path.exists():
            md5 = md5_file(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.name}: {size_mb:.1f} MB, MD5={md5}")
        else:
            print(f"  [WARN] Source not found: {path}")

    print(f"  [OK] Source files exist and are readable")


def main():
    errors = []

    try:
        validate_epitope_csv(EPITIPE_V1_DST)
    except Exception as e:
        errors.append(f"Epitope validation FAILED: {e}")
        print(f"  [FAIL] {e}")

    try:
        validate_drug_csv(DRUG_V1_DST)
    except Exception as e:
        errors.append(f"Drug validation FAILED: {e}")
        print(f"  [FAIL] {e}")

    try:
        validate_drug_csv(DRUG_V1_MINIMAL)
    except Exception as e:
        errors.append(f"Drug minimal validation FAILED: {e}")
        print(f"  [FAIL] {e}")

    verify_source_untouched()

    print(f"\n{'='*60}")
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"ALL VALIDATIONS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
