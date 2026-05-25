"""
features.py — Feature specification for circRNA analysis.

Dataclass pattern for defining feature extraction configuration,
mirroring the FeatureSpec pattern from confluencia-2.0-epitope.

Reference: D:/IGEM集成方案/confluencia-2.0-epitope/core/features.py
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json

FEATURE_SCHEMA_VERSION = "circrna-feature-schema-v3"
KMER_HASH_VERSION = "circrna-kmer-hash-v2"


@dataclass(frozen=True)
class CircRNAFeatureSpec:
    """
    Feature extraction specification for circRNA.

    Immutable configuration that determines:
    - Sequence encoding parameters
    - Structure prediction settings
    - Gene expression columns
    - Scoring weight configuration

    Pattern mirrors FeatureSpec from epitope module for consistency.
    """

    # Sequence encoding
    max_seq_length: int = 3000
    kmer_sizes: Tuple[int, ...] = (2, 3, 4)
    kmer_hash_dim: int = 64

    # Structure prediction
    enable_structure_prediction: bool = True
    min_dsrna_length: int = 33  # PKR threshold per Nallagatla et al., 2007

    # Gene expression
    gene_cols: Tuple[str, ...] = (
        "TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"
    )
    gene_normalization_method: str = "minmax"

    # Immune sensing config
    rig_i_motifs: Tuple[str, ...] = ("CCUCC", "UCUCC", "ACUCC", "GCUCC")
    tlr_motifs: Tuple[str, ...] = ("GUUG", "UUGU", "UGUU", "GUUU", "GUU")
    blunt_end_window: int = 20

    # Scoring weights source
    use_literature_weights: bool = True
    weights_config_path: str = ""

    # Encoder settings
    use_rna_fm: bool = True
    rna_fm_model: str = "RNA-FM-small"
    freeze_backbone: bool = True

    # MLP head
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.2
    gene_proj_dim: int = 32

    def schema_id(self) -> str:
        """Generate unique schema identifier for version control."""
        spec_dict = asdict(self)
        # Remove path for stable hash
        spec_dict["weights_config_path"] = ""
        spec_str = json.dumps(spec_dict, sort_keys=True)
        hash_val = hashlib.sha256(spec_str.encode()).hexdigest()[:8]
        return f"{FEATURE_SCHEMA_VERSION};hash={hash_val}"

    def get_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = 0

        # Sequence features
        if self.use_rna_fm:
            # RNA-FM embedding dimension (480 for small)
            from confluencia_circrna.encoder.config import RNA_FM_MODELS
            dim += RNA_FM_MODELS.get(self.rna_fm_model, {}).get("embed_dim", 480)

        # K-mer features
        for k in self.kmer_sizes:
            dim += self.kmer_hash_dim

        # Gene expression features
        dim += len(self.gene_cols)

        # Structure features (if enabled)
        if self.enable_structure_prediction:
            dim += 5  # mfe, dsrna_fraction, stability, hairpin_count, stem_count

        return dim


@dataclass
class CircRNAModelBundle:
    """
    Trained model bundle for circRNA encoder.

    Mirrors EpitopeModelBundle pattern for consistency.

    Contains:
    - Model object
    - Feature specification used during training
    - Training metrics
    - Integration metadata
    """

    model_backend: str
    model: Any
    feature_spec: CircRNAFeatureSpec
    feature_names: List[str]
    feature_dim: int

    # Training metrics
    y_std: float = 1.0
    train_r2: float = 0.0
    val_r2: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0

    # Metadata
    schema_version: str = FEATURE_SCHEMA_VERSION
    kmer_hash_version: str = KMER_HASH_VERSION
    trained_at: str = ""
    n_samples: int = 0

    # MOE weights (if ensemble)
    moe_weights: Dict[str, float] = field(default_factory=dict)
    moe_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bundle to dict (excluding model object)."""
        d = asdict(self)
        d["model"] = None  # Cannot serialize model object
        d["feature_spec"] = asdict(self.feature_spec)
        return d

    def schema_id(self) -> str:
        """Get schema ID from feature spec."""
        return self.feature_spec.schema_id()


def load_feature_spec_from_config(config_path: str) -> CircRNAFeatureSpec:
    """Load FeatureSpec from JSON config file."""
    path = config_path
    if not path:
        # Default path
        from pathlib import Path
        path = Path(__file__).parent.parent / "data" / "reference" / "feature_spec_default.json"

    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        with open(path) as f:
            config = json.load(f)
        return CircRNAFeatureSpec(**config)

    return CircRNAFeatureSpec()


def save_feature_spec_to_config(spec: CircRNAFeatureSpec, config_path: str) -> None:
    """Save FeatureSpec to JSON config file."""
    with open(config_path, "w") as f:
        json.dump(asdict(spec), f, indent=2)


# Default feature spec
DEFAULT_FEATURE_SPEC = CircRNAFeatureSpec()


if __name__ == "__main__":
    # Demo
    spec = CircRNAFeatureSpec()
    print("CircRNA FeatureSpec Demo")
    print("=" * 60)
    print(f"Schema ID: {spec.schema_id()}")
    print(f"Feature dim: {spec.get_feature_dim()}")
    print(f"Gene cols: {spec.gene_cols}")
    print(f"Max seq length: {spec.max_seq_length}")
    print(f"Enable structure: {spec.enable_structure_prediction}")

    # Save default config
    from pathlib import Path
    default_path = Path(__file__).parent.parent / "data" / "reference" / "feature_spec_default.json"
    save_feature_spec_to_config(spec, str(default_path))
    print(f"\nSaved default config to: {default_path}")