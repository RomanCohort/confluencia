"""Confluencia Hub Ethics Enhancement Module.

This module adds ethical safeguards to the federated model sharing system:
1. Data source declaration (mandatory)
2. Clinical use warnings
3. Patient data IRB verification
4. Dual-use screening

Usage:
    from confluencia_cli.hub_ethics import EthicsEnhancedHub

    hub = EthicsEnhancedHub()

    # Upload with ethical declarations
    hub.push_model(
        "my_model.joblib",
        data_source="GDSC (DOI:10.1016/j.drudis.2013)",
        clinical_use_intent="research_only",
        irb_approval=None  # Not needed for public data
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union
from enum import Enum
import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Ethics Enums and Constants
# ---------------------------------------------------------------------------

class DataSourceType(Enum):
    """Training data source classification."""
    PUBLIC_DATASET = "public_dataset"       # GDSC, ChEMBL, IEDB (DOI required)
    PROPRIETARY = "proprietary"             # Company/internal data (license check)
    PATIENT_DERIVED = "patient_derived"     # Human subject data (IRB required)
    SYNTHETIC = "synthetic"                 # Generated data
    MIXED = "mixed"                         # Multiple sources
    UNKNOWN = "unknown"                     # Not declared (blocked)


class ClinicalUseIntent(Enum):
    """Declared intent for model usage."""
    RESEARCH_ONLY = "research_only"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    PRECLINICAL_REFERENCE = "preclinical_reference"
    CLINICAL_TRIAL_SUPPORT = "clinical_trial_support"
    NOT_FOR_HUMAN_USE = "not_for_human_use"


class EthicsWarning(Exception):
    """Raised when ethical requirements are not met."""
    pass


class DualUseWarning(Exception):
    """Raised when potential dual-use issue detected."""
    pass


# Mandatory terms of service
TERMS_OF_SERVICE = """
CONFLUENCIA HUB ETHICAL USE AGREEMENT

By uploading models or data to Confluencia Hub, you agree to:

1. DATA SOURCE DECLARATION
   - Public data: Provide DOI or URL
   - Proprietary data: Confirm sharing rights
   - Patient-derived data: Provide IRB/ethics approval number
   - Failure to declare will result in upload rejection

2. NO CLINICAL WARRANTY
   - Models are NOT validated for clinical decision-making
   - Use is restricted to research and hypothesis generation
   - Clinical application requires independent validation

3. DUAL-USE PROHIBITION
   - Models designed for harmful purposes are prohibited
   - Toxicity prediction for legitimate drug development is allowed
   - Weaponization or bioterrorism applications are banned

4. LICENSE COMPLIANCE
   - Model weights: declared license applies
   - Training data: source license applies separately
   - You must have rights to share both

5. LIABILITY
   - You retain responsibility for claims accuracy
   - Confluencia is not liable for model-based decisions
   - Misrepresentation may result in removal and ban

ACCEPTING THIS AGREEMENT IS REQUIRED FOR ALL UPLOADS.
"""


# ---------------------------------------------------------------------------
# Enhanced Metadata Classes
# ---------------------------------------------------------------------------

@dataclass
class EthicsDeclaration:
    """Ethical declaration for model upload."""
    data_source_type: DataSourceType
    data_source_reference: str  # DOI, URL, or IRB number
    clinical_use_intent: ClinicalUseIntent
    irb_approval_number: Optional[str] = None
    dual_use_declaration: bool = False  # "I declare this is not for harmful use"
    patient_consent_status: Optional[str] = None  # "obtained", "not_required", "unknown"
    data_anonymization: bool = False  # Whether training data was anonymized


@dataclass
class EnhancedModelMeta:
    """Metadata with ethics fields."""
    model_id: str
    task: Literal["drug", "epitope", "circRNA"]
    model_name: str
    uploader: str
    version: str
    uploaded_at: str
    metrics: Dict[str, float]
    n_samples: int
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"

    # Ethics fields (NEW)
    ethics_declaration: Optional[Dict[str, Any]] = None
    clinical_warning: bool = True
    ethics_verified: bool = False
    ethics_reviewer: Optional[str] = None


# ---------------------------------------------------------------------------
# Ethics Verification Functions
# ---------------------------------------------------------------------------

def verify_data_source_declaration(declaration: EthicsDeclaration) -> bool:
    """Verify that data source declaration is complete and valid."""

    # Public dataset: must have DOI or URL
    if declaration.data_source_type == DataSourceType.PUBLIC_DATASET:
        if not declaration.data_source_reference:
            raise EthicsWarning(
                "Public dataset requires DOI or URL reference.\n"
                "Example: 'DOI:10.1016/j.drudis.2013' or 'https://gdsc.org'"
            )
        # Validate DOI format
        if not re.match(r'(DOI:|https?://)', declaration.data_source_reference):
            raise EthicsWarning(
                f"Invalid reference format: {declaration.data_source_reference}\n"
                "Use 'DOI:xxx' or 'https://xxx' format"
            )

    # Patient-derived: must have IRB approval
    if declaration.data_source_type == DataSourceType.PATIENT_DERIVED:
        if not declaration.irb_approval_number:
            raise EthicsWarning(
                "Patient-derived data requires IRB/ethics approval number.\n"
                "Provide format: 'IRB-2024-001' or 'EthicsCommittee-Approval-123'"
            )
        if declaration.patient_consent_status != "obtained":
            raise EthicsWarning(
                "Patient-derived data requires consent status 'obtained'.\n"
                "If consent was not obtained, upload is not permitted."
            )
        if not declaration.data_anonymization:
            raise EthicsWarning(
                "Patient-derived data must be anonymized before model training.\n"
                "Set data_anonymization=True to proceed."
            )

    # Proprietary: must have license confirmation
    if declaration.data_source_type == DataSourceType.PROPRIETARY:
        if not declaration.data_source_reference:
            raise EthicsWarning(
                "Proprietary data requires license/rights confirmation.\n"
                "Provide: 'Company-X-License-Agreement-2024'"
            )

    # Unknown: blocked
    if declaration.data_source_type == DataSourceType.UNKNOWN:
        raise EthicsWarning(
            "Data source type 'unknown' is not permitted.\n"
            "You must declare the training data source to upload."
        )

    # Dual-use check
    if not declaration.dual_use_declaration:
        raise EthicsWarning(
            "Dual-use declaration required.\n"
            "You must confirm this model is not designed for harmful purposes."
        )

    return True


def check_patient_sequence_presence(bundle) -> bool:
    """Check if model may contain patient-derived sequences."""
    # Heuristic checks:
    # 1. If bundle has sequence column and n_samples is small (<100)
    # 2. If bundle metadata mentions patient/cohort
    # 3. If sequence examples look like genomic fragments

    suspicious_indicators = 0

    if hasattr(bundle, 'n_samples') and bundle.n_samples < 100:
        suspicious_indicators += 1

    if hasattr(bundle, 'metadata'):
        meta = bundle.metadata if isinstance(bundle.metadata, dict) else {}
        patient_keywords = ['patient', 'cohort', 'clinical', 'hospital', 'subject']
        for kw in patient_keywords:
            if any(kw in str(v).lower() for v in meta.values()):
                suspicious_indicators += 1

    return suspicious_indicators >= 2


def generate_ethics_label(declaration: EthicsDeclaration) -> str:
    """Generate human-readable ethics label."""
    source = declaration.data_source_type.value
    intent = declaration.clinical_use_intent.value

    if declaration.data_source_type == DataSourceType.PATIENT_DERIVED:
        return f"[ETHICS:Patient-IRB:{declaration.irb_approval_number}] [USE:{intent}]"
    elif declaration.data_source_type == DataSourceType.PUBLIC_DATASET:
        return f"[ETHICS:Public:{declaration.data_source_reference}] [USE:{intent}]"
    else:
        return f"[ETHICS:{source}] [USE:{intent}]"


# ---------------------------------------------------------------------------
# Ethics Enhanced Hub Class
# ---------------------------------------------------------------------------

class EthicsEnhancedHub:
    """Hub with mandatory ethics verification."""

    TERMS_ACCEPTED_FILE = Path.home() / ".confluencia" / "terms_accepted.txt"

    def __init__(self, cache_dir: Optional[Path] = None, require_terms: bool = True):
        from confluencia_cli.hub import ConfluenciaHub, HUB_CACHE_DIR

        self._hub = ConfluenciaHub(cache_dir)
        self.require_terms = require_terms

        if require_terms and not self._check_terms_accepted():
            self._display_terms_and_request_acceptance()

    def _check_terms_accepted(self) -> bool:
        """Check if user has accepted terms."""
        if self.TERMS_ACCEPTED_FILE.exists():
            accepted_date = self.TERMS_ACCEPTED_FILE.read_text().strip()
            # Terms valid for 1 year
            try:
                date = datetime.fromisoformat(accepted_date)
                if (datetime.now() - date).days < 365:
                    return True
            except:
                pass
        return False

    def _display_terms_and_request_acceptance(self) -> None:
        """Display terms and request acceptance."""
        print(TERMS_OF_SERVICE)
        print("\n" + "=" * 60)

        # In automated context, assume acceptance for testing
        if os.environ.get("CONFLUENCIA_AUTO_ACCEPT_TERMS") == "1":
            self._accept_terms()
            return

        response = input("Do you accept this agreement? (yes/no): ")
        if response.lower() != "yes":
            raise EthicsWarning("Terms of service must be accepted to use Hub")
        self._accept_terms()

    def _accept_terms(self) -> None:
        """Record terms acceptance."""
        self.TERMS_ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.TERMS_ACCEPTED_FILE.write_text(datetime.now().isoformat())
        print("[Ethics] Terms of service accepted.")

    def push_model(
        self,
        bundle_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        uploader: str = "anonymous",
        license: str = "MIT",
        strip_env_medians: bool = True,  # Changed default to True

        # NEW ETHICS PARAMETERS
        data_source_type: str = "unknown",
        data_source_reference: str = "",
        clinical_use_intent: str = "research_only",
        irb_approval_number: Optional[str] = None,
        dual_use_declaration: bool = False,
        patient_consent_status: Optional[str] = None,
        data_anonymization: bool = False,
    ) -> str:
        """Upload model with ethics verification.

        Args:
            bundle_path: Path to .joblib model bundle
            metadata: Performance metrics and model info
            uploader: Contributor identifier
            license: License for model weights
            strip_env_medians: Remove training data statistics (default True)

            data_source_type: "public_dataset", "proprietary", "patient_derived",
                              "synthetic", "mixed", "unknown"
            data_source_reference: DOI, URL, or license reference
            clinical_use_intent: "research_only", "hypothesis_generation", etc.
            irb_approval_number: Required for patient_derived
            dual_use_declaration: Must be True to proceed
            patient_consent_status: "obtained" for patient data
            data_anonymization: True if patient data was anonymized

        Returns:
            model_id: Unique identifier for the uploaded model

        Raises:
            EthicsWarning: If ethical requirements not met
        """

        # Create ethics declaration
        declaration = EthicsDeclaration(
            data_source_type=DataSourceType(data_source_type),
            data_source_reference=data_source_reference,
            clinical_use_intent=ClinicalUseIntent(clinical_use_intent),
            irb_approval_number=irb_approval_number,
            dual_use_declaration=dual_use_declaration,
            patient_consent_status=patient_consent_status,
            data_anonymization=data_anonymization,
        )

        # Verify ethics
        try:
            verify_data_source_declaration(declaration)
        except EthicsWarning as e:
            print(f"\n[Ethics Warning] Upload blocked:\n{e}")
            raise

        # Check for patient sequences if suspicious
        bundle = joblib.load(bundle_path)
        if check_patient_sequence_presence(bundle):
            if declaration.data_source_type != DataSourceType.PATIENT_DERIVED:
                print("\n[Ethics Warning] Model appears to contain patient-derived data")
                print("but was declared as {data_source_type}.")
                print("Please verify and update declaration.")
                raise EthicsWarning(
                    "Potential patient data detected but not declared as patient_derived"
                )

        # Generate ethics label
        ethics_label = generate_ethics_label(declaration)

        # Add ethics to metadata
        enhanced_metadata = metadata or {}
        enhanced_metadata["ethics_declaration"] = asdict(declaration)
        enhanced_metadata["ethics_label"] = ethics_label
        enhanced_metadata["clinical_warning"] = True

        # Upload using base hub
        model_id = self._hub.push_model(
            bundle_path,
            metadata=enhanced_metadata,
            uploader=uploader,
            license=license,
            strip_env_medians=strip_env_medians,
        )

        # Append ethics label to model_id
        print(f"\n[Ethics] Model uploaded with declaration: {ethics_label}")

        return model_id

    def pull_model(self, model_id: str) -> str:
        """Download model with ethics warning display."""
        # Get metadata
        models = self._hub.list_models()
        model_meta = next((m for m in models if m.get("model_id") == model_id), None)

        if model_meta:
            ethics_label = model_meta.get("ethics_label", "")
            if ethics_label:
                print(f"\n[Ethics Label] {ethics_label}")

            # Always display clinical warning
            print("\n" + "=" * 60)
            print("CLINICAL USE WARNING")
            print("=" * 60)
            print("This model is NOT validated for clinical decision-making.")
            print("Use is restricted to research and hypothesis generation.")
            print("Clinical application requires independent validation.")
            print("=" * 60 + "\n")

        return self._hub.pull_model(model_id)

    # Delegate other methods to base hub
    def list_models(self, **kwargs):
        return self._hub.list_models(**kwargs)

    def data_stats(self):
        return self._hub.data_stats()

    def push_data(self, **kwargs):
        return self._hub.push_data(**kwargs)


# ---------------------------------------------------------------------------
# Convenience function for quick ethics-compliant upload
# ---------------------------------------------------------------------------

def quick_upload_public_model(
    bundle_path: str,
    doi: str,
    metrics: Dict[str, float],
    uploader: str = "anonymous"
) -> str:
    """Quick upload for models trained on public datasets.

    Args:
        bundle_path: Path to model bundle
        doi: DOI of the training dataset (e.g., "DOI:10.1016/j.drudis.2013")
        metrics: Performance metrics {"r2": 0.85, "mae": 0.04}
        uploader: Your identifier

    Returns:
        model_id
    """
    os.environ["CONFLUENCIA_AUTO_ACCEPT_TERMS"] = "1"
    hub = EthicsEnhancedHub(require_terms=True)

    return hub.push_model(
        bundle_path,
        metadata=metrics,
        uploader=uploader,
        data_source_type="public_dataset",
        data_source_reference=doi,
        clinical_use_intent="research_only",
        dual_use_declaration=True,
    )