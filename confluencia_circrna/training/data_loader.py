"""
data_loader.py — Standardized data loading for circRNA training.

Ensures:
1. Gene expression values are consistently normalized
2. Reference ranges are based on literature/TCGA data
3. Cross-sample comparability
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Reference ranges from TCGA-BRCA (loaded from config)
_REFERENCE_FILE = Path(__file__).parent.parent / "data" / "reference" / "scoring_weights_literature.json"

# Default TCGA-based reference ranges
GENE_REFERENCE_RANGES = {
    "TROP2": {"low": 2.0, "high": 12.0, "median": 7.0},
    "NECTIN4": {"low": 2.0, "high": 10.0, "median": 5.0},
    "LIV-1": {"low": 1.0, "high": 8.0, "median": 3.5},
    "B7-H4": {"low": 2.0, "high": 10.0, "median": 6.0},
    "MKI67": {"low": 4.0, "high": 14.0, "median": 8.0},
    "MYC": {"low": 3.0, "high": 12.0, "median": 4.5},
}


@dataclass
class GeneNormalizationConfig:
    """Gene expression normalization configuration."""
    method: str = "minmax"  # minmax, zscore, quantile
    reference_ranges: Dict[str, Dict[str, float]] = None
    clip_range: tuple = (0.0, 1.0)

    def __post_init__(self):
        if self.reference_ranges is None:
            self.reference_ranges = self._load_reference_ranges()

    def _load_reference_ranges(self) -> Dict[str, Dict[str, float]]:
        """Load reference ranges from config file."""
        if _REFERENCE_FILE.exists():
            try:
                with open(_REFERENCE_FILE) as f:
                    config = json.load(f)
                return config.get("gene_normalization_ranges", GENE_REFERENCE_RANGES)
            except Exception:
                pass
        return GENE_REFERENCE_RANGES


def normalize_gene_expression(
    gene_expr: Dict[str, float],
    config: Optional[GeneNormalizationConfig] = None,
) -> Dict[str, float]:
    """
    Normalize gene expression to [0, 1] range.

    Uses literature-based reference ranges for consistent normalization
    across different datasets (TCGA, METABRIC, GEO).

    Args:
        gene_expr: Raw gene expression values {gene: value}
        config: Normalization configuration

    Returns:
        Normalized gene expression {gene: normalized_value}
    """
    if config is None:
        config = GeneNormalizationConfig()

    normalized = {}
    for gene, value in gene_expr.items():
        # Handle gene name variations (B7-H4 -> B7H4)
        gene_key = gene.replace("-", "").replace(" ", "")
        alt_key = gene  # Try original name too

        ref = None
        for key in [gene, gene_key, alt_key]:
            if key in config.reference_ranges:
                ref = config.reference_ranges[key]
                break

        if ref:
            low = ref["low"]
            high = ref["high"]

            if config.method == "minmax":
                # Linear normalization to [0, 1]
                if high > low:
                    norm_val = (value - low) / (high - low)
                else:
                    norm_val = 0.5
                norm_val = np.clip(norm_val, config.clip_range[0], config.clip_range[1])

            elif config.method == "zscore":
                # Z-score normalization using median and estimated std
                median = ref.get("median", (low + high) / 2)
                std = (high - low) / 4.0  # Approximate std from range
                z = (value - median) / std if std > 0 else 0
                # Convert z-score to [0, 1] using sigmoid-like mapping
                norm_val = 0.5 + 0.25 * z  # Maps -2 to 0, +2 to 1
                norm_val = np.clip(norm_val, config.clip_range[0], config.clip_range[1])

            else:
                norm_val = value

            normalized[gene] = float(norm_val)
        else:
            # Unknown gene: use conservative default
            normalized[gene] = float(np.clip(value / 10.0, config.clip_range[0], config.clip_range[1]))

    return normalized


def normalize_gene_batch(
    gene_exprs: List[Dict[str, float]],
    config: Optional[GeneNormalizationConfig] = None,
) -> List[Dict[str, float]]:
    """Normalize a batch of gene expression dicts."""
    return [normalize_gene_expression(g, config) for g in gene_exprs]


def build_gene_expression_dict(
    df: pd.DataFrame,
    gene_cols: List[str],
    sample_idx: int,
    config: Optional[GeneNormalizationConfig] = None,
) -> Dict[str, float]:
    """
    Build normalized gene expression dict from DataFrame row.

    Args:
        df: DataFrame with gene expression columns
        gene_cols: List of gene column names
        sample_idx: Row index
        config: Normalization config

    Returns:
        Dict {gene: normalized_value}
    """
    row = df.iloc[sample_idx]
    raw_expr = {}
    for gene in gene_cols:
        # Try various column name formats
        col_names = [
            gene,
            f"gene_{gene}",
            gene.replace("-", ""),
            gene.replace(" ", "_"),
        ]
        for col in col_names:
            if col in df.columns:
                raw_expr[gene] = float(row[col])
                break
        else:
            raw_expr[gene] = 0.5  # Default if column not found

    return normalize_gene_expression(raw_expr, config)


def resolve_label(
    df: pd.DataFrame,
    label_col: str,
    sample_idx: int,
    default: float = 0.5,
) -> Optional[float]:
    """
    Resolve label value from DataFrame, handling missing values.

    Args:
        df: DataFrame
        label_col: Column name for label
        sample_idx: Row index
        default: Default value if missing

    Returns:
        Label value or None if not found
    """
    if label_col in df.columns:
        val = df.iloc[sample_idx][label_col]
        if pd.notna(val):
            return float(val)
    return None


def validate_gene_expression_range(
    gene_expr: Dict[str, float],
    config: Optional[GeneNormalizationConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Validate gene expression values against reference ranges.

    Returns:
        Dict with validation results per gene: {gene: {status, value, reference}}
    """
    if config is None:
        config = GeneNormalizationConfig()

    results = {}
    for gene, value in gene_expr.items():
        ref = config.reference_ranges.get(gene)
        if ref:
            low = ref["low"]
            high = ref["high"]
            median = ref.get("median", (low + high) / 2)

            status = "normal"
            if value < low:
                status = "below_range"
            elif value > high:
                status = "above_range"

            percentile = (value - low) / (high - low) * 100

            results[gene] = {
                "status": status,
                "raw_value": value,
                "normalized_value": normalize_gene_expression({gene: value}, config)[gene],
                "reference_low": low,
                "reference_high": high,
                "reference_median": median,
                "estimated_percentile": percentile,
            }
        else:
            results[gene] = {
                "status": "unknown_gene",
                "raw_value": value,
                "normalized_value": value / 10.0,
            }

    return results


if __name__ == "__main__":
    # Demo
    test_expr = {
        "TROP2": 9.5,
        "NECTIN4": 3.2,
        "B7-H4": 15.0,  # Above range
        "MKI67": 6.0,
        "MYC": 2.0,     # Below range
        "UNKNOWN_GENE": 5.0,
    }

    print("Gene Expression Normalization Demo")
    print("=" * 60)

    config = GeneNormalizationConfig()
    normalized = normalize_gene_expression(test_expr, config)

    print("\nRaw values:")
    for gene, val in test_expr.items():
        print(f"  {gene}: {val:.2f}")

    print("\nNormalized values:")
    for gene, val in normalized.items():
        print(f"  {gene}: {val:.3f}")

    print("\nValidation:")
    validation = validate_gene_expression_range(test_expr, config)
    for gene, info in validation.items():
        print(f"  {gene}: status={info['status']}, percentile={info.get('estimated_percentile', 'N/A')}")