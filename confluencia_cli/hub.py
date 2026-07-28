"""Confluencia Hub — community model and data sharing.

Design principles:
1. Federated model sharing: users upload .joblib bundles (no raw data exposed)
2. Optional data pool: users can contribute CSVs to community datasets
3. Privacy-first: no SMILES/sequences are logged; only aggregated metrics
4. Versioning: each upload gets a version hash for reproducibility
5. Integration: hub models can be loaded via cf_drug_predict(bundle_id="hub:xxx")
6. Attribution: ORCID-bound uploads → Zenodo DOI → citation-ready
7. Impact tracking: download count + citation badges per contributor
8. Quality tiers: unverified / reproducible / verified / benchmark_top
9. Circ-CASP linkage: competition models auto-enter hub with metrics

Usage:
    from confluencia_cli.hub import ConfluenciaHub

    hub = ConfluenciaHub()

    # Upload a trained model (federated, ORCID-bound, DOI-minted)
    hub.push_model(
        "my_drug_model.joblib",
        metadata={"model_name": "ridge", "r2": 0.91},
        uploader_orcid="0000-0002-1825-0097",
    )

    # List available community models (sorted by tier + downloads)
    models = hub.list_models(task="drug")
    print(models)

    # Download and use a community model
    bundle_path = hub.pull_model("hub:drug:0000-0002-1825-0097:abc123def456")

    # Optionally contribute data to community pool
    hub.push_data("my_drug_data.csv", license="CC-BY-4.0", anonymous=True)

    # Query community data statistics (no raw data exposed)
    stats = hub.data_stats()
    print(stats)  # {"drug": {"n_samples": 15000, "n_contributors": 12}, ...}

    # Contributor impact report (downloads, citations, badges)
    report = hub.get_contributor_stats("0000-0002-1825-0097")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union
import urllib.request
import urllib.error
import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HUB_URL = os.environ.get("CONFLUENCIA_HUB_URL", "https://hub.confluencia.org/api/v1")
HUB_CACHE_DIR = Path(os.environ.get("CONFLUENCIA_HUB_CACHE", Path.home() / ".confluencia" / "hub"))

# Backend tokens (optional — graceful degradation when absent)
HF_TOKEN = os.environ.get("CONFLUENCIA_HF_TOKEN") or os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("CONFLUENCIA_HF_REPO", "confluencia/hub")
ZENODO_TOKEN = os.environ.get("CONFLUENCIA_ZENODO_TOKEN")
ZENODO_API = os.environ.get("CONFLUENCIA_ZENODO_API", "https://zenodo.org/api/deposit/depositions")

# Verification tiers (ordered low → high)
VERIFICATION_LEVELS = ("unverified", "reproducible", "verified", "benchmark_top")


# ---------------------------------------------------------------------------
# ORCID validation
# ---------------------------------------------------------------------------

ORCID_PATTERN = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def validate_orcid(orcid: str) -> str:
    """Validate ORCID iD format and return the canonical form.

    Raises ValueError if format is invalid. The check digit is verified per
    the ISO 7064 11-2 algorithm.
    """
    if not orcid or not ORCID_PATTERN.match(orcid):
        raise ValueError(
            f"Invalid ORCID format: {orcid!r}. Expected 16-digit form XXXX-XXXX-XXXX-XXXX."
        )
    # Verify check digit (ISO 7064 11-2)
    digits = orcid.replace("-", "")[:-1]
    check = orcid[-1].upper()
    total = 0
    for d in digits:
        total = (total + int(d)) * 2
    remainder = total % 11
    expected = "X" if remainder == 1 else str((11 - remainder) % 11)
    if expected != check:
        raise ValueError(f"Invalid ORCID check digit: {orcid}")
    return orcid


def orcid_short(orcid: str) -> str:
    """Compact ORCID form for use in model_id (last 4 digits + check)."""
    return orcid.replace("-", "")[-8:]


# ---------------------------------------------------------------------------
# Metadata classes
# ---------------------------------------------------------------------------

@dataclass
class ModelMeta:
    """Metadata for a shared model (ORCID-bound, citation-ready)."""
    model_id: str  # hub:{task}:{orcid_short}:{file_hash}
    task: Literal["drug", "epitope", "circRNA"]
    model_name: str  # ridge, hgb, etc.
    uploader: str  # display name (kept for back-compat)
    version: str  # git-style hash
    uploaded_at: str

    # Attribution (NEW)
    uploader_orcid: str = ""  # canonical ORCID iD
    contributors: List[str] = field(default_factory=list)  # additional ORCIDs
    zenodo_doi: str = ""  # e.g., "10.5281/zenodo.1234567"
    license: str = "MIT"

    # Provenance
    metrics: Dict[str, float] = field(default_factory=dict)  # {"r2": 0.91, "mae": 0.03}
    n_samples: int = 0  # training set size
    tags: List[str] = field(default_factory=list)

    # Quality (NEW)
    verification_level: str = "unverified"  # see VERIFICATION_LEVELS
    reproducibility_url: str = ""  # training code repo
    circ_casp_metrics: Dict[str, float] = field(default_factory=dict)  # competition results

    # Impact (NEW — populated lazily by get_contributor_stats)
    download_count: int = 0
    citation_count: int = 0


@dataclass
class DataMeta:
    """Metadata for a shared dataset (stats only, no raw data)."""
    dataset_id: str
    task: Literal["drug", "epitope", "circRNA"]
    uploader: str
    uploaded_at: str
    n_samples: int
    n_features: int
    columns: List[str]  # column names only
    license: str
    anonymous: bool = True
    # Attribution (NEW)
    uploader_orcid: str = ""
    zenodo_doi: str = ""


# ---------------------------------------------------------------------------
# ConfluenciaHub class
# ---------------------------------------------------------------------------

class ConfluenciaHub:
    """Community model and data sharing hub.

    Supports two sharing modes:
    1. Federated: upload trained models (.joblib), no raw data exposed
    2. Data pool: optionally upload CSVs to community dataset (with license)
    """

    def __init__(self, cache_dir: Optional[Path] = None, hub_url: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else HUB_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hub_url = hub_url or HUB_URL
        self._offline = os.environ.get("CONFLUENCIA_HUB_OFFLINE", "").lower() in ("1", "true")

    # ---- Model Operations (Federated) ----

    def push_model(self, bundle_path: str, metadata: Optional[Dict[str, Any]] = None,
                   uploader: str = "anonymous", license: str = "MIT",
                   strip_env_medians: bool = False,
                   uploader_orcid: str = "",
                   contributors: Optional[List[str]] = None,
                   reproducibility_url: str = "",
                   mint_doi: bool = True) -> str:
        """Upload a trained model bundle to the hub.

        Returns the model_id (e.g., "hub:drug:0000-0002-1825-0097:abc123def456").
        The raw training data is NOT uploaded — only the model weights.

        Args:
            strip_env_medians: If True, remove env_medians from the bundle
                before sharing. This eliminates any statistical traces of the
                training data, at the cost of needing users to provide their
                own env values at prediction time.
            uploader_orcid: Canonical ORCID iD (e.g., "0000-0002-1825-0097").
                Strongly recommended — binds the upload to a citable identity
                and enables Zenodo DOI minting. If empty, upload proceeds but
                with no citation handle.
            contributors: Additional ORCID iDs of co-contributors.
            reproducibility_url: URL to training code repo (raises tier to
                "reproducible" if provided and valid).
            mint_doi: If True (default) and ZENODO_TOKEN is set, mint a DOI
                for this upload so it can be cited in papers.
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        # Validate ORCID if provided
        if uploader_orcid:
            uploader_orcid = validate_orcid(uploader_orcid)
        if contributors:
            contributors = [validate_orcid(c) for c in contributors]

        # Compute version hash from file content
        file_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:12]

        # Detect task type from bundle
        bundle = joblib.load(bundle_path)
        task = self._detect_task(bundle)

        # Optionally strip statistical traces of training data
        if strip_env_medians and hasattr(bundle, "env_medians"):
            bundle.env_medians = {}
            # Re-save stripped bundle
            stripped_path = self.cache_dir / "staging" / f"{file_hash}_stripped.joblib"
            stripped_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, stripped_path)
            upload_path = stripped_path
        else:
            upload_path = bundle_path

        # Build model_id: prefer ORCID-short, fall back to uploader name
        id_authority = orcid_short(uploader_orcid) if uploader_orcid else uploader
        model_id = f"hub:{task}:{id_authority}:{file_hash}"

        # Determine initial verification level
        if reproducibility_url and reproducibility_url.startswith(("http://", "https://")):
            verification_level = "reproducible"
        else:
            verification_level = "unverified"

        # Mint Zenodo DOI (best-effort)
        zenodo_doi = ""
        if mint_doi and uploader_orcid and ZENODO_TOKEN:
            zenodo_doi = self._mint_zenodo_doi(
                upload_path, model_id, uploader_orcid, metadata or {}
            )

        meta = ModelMeta(
            model_id=model_id,
            task=task,
            model_name=metadata.get("model_name", "unknown") if metadata else "unknown",
            uploader=uploader,
            version=file_hash,
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
            uploader_orcid=uploader_orcid,
            contributors=contributors or [],
            zenodo_doi=zenodo_doi,
            license=license,
            metrics=metadata.get("metrics", {}) if metadata else {},
            n_samples=metadata.get("n_samples", 0) if metadata else 0,
            tags=metadata.get("tags", []) if metadata else [],
            verification_level=verification_level,
            reproducibility_url=reproducibility_url,
            circ_casp_metrics=metadata.get("circ_casp_metrics", {}) if metadata else {},
        )

        if self._offline:
            # Save locally only
            self._save_model_local(upload_path, meta)
            return model_id

        # Upload to hub server (HuggingFace Hub)
        self._upload_model(upload_path, meta)
        return model_id

    def pull_model(self, model_id: str) -> str:
        """Download a model from the hub. Returns local path to .joblib."""
        if model_id.startswith("hub:"):
            # Parse model_id
            parts = model_id.split(":")
            if len(parts) != 4:
                raise ValueError(f"Invalid model_id: {model_id}")
            _, task, uploader, version = parts
        else:
            # Treat as local path
            return model_id

        local_path = self.cache_dir / "models" / task / uploader / f"{version}.joblib"
        if local_path.exists():
            return str(local_path)

        if self._offline:
            raise RuntimeError(f"Offline mode: cannot download {model_id}")

        # Download from hub
        self._download_model(model_id, local_path)
        return str(local_path)

    def list_models(self, task: Optional[str] = None, uploader: Optional[str] = None,
                    limit: int = 50) -> List[Dict[str, Any]]:
        """List available community models."""
        if self._offline:
            return self._list_models_local(task, uploader, limit)

        # Query hub server
        return self._query_models(task, uploader, limit)

    # ---- Data Operations (Optional Pool) ----

    def push_data(self, csv_path: str, license: str = "CC-BY-4.0",
                  anonymous: bool = True, uploader: str = "anonymous") -> str:
        """Contribute a dataset to the community pool.

        The CSV is uploaded to the hub server. By default, uploader is anonymous.
        License must be specified (CC-BY-4.0, MIT, or proprietary).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        task = self._detect_task_from_csv(df)

        file_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]
        # Always anonymize uploader in dataset_id when anonymous=True
        effective_uploader = "anonymous" if anonymous else uploader
        dataset_id = f"data:{task}:{effective_uploader}:{file_hash}"

        meta = DataMeta(
            dataset_id=dataset_id,
            task=task,
            uploader=effective_uploader,
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
            n_samples=len(df),
            n_features=len(df.columns),
            columns=list(df.columns),
            license=license,
            anonymous=anonymous,
        )

        if self._offline:
            self._save_data_local(csv_path, meta)
            return dataset_id

        self._upload_data(csv_path, meta)
        return dataset_id

    def data_stats(self) -> Dict[str, Any]:
        """Get statistics about community datasets (no raw data exposed)."""
        if self._offline:
            return self._data_stats_local()

        return self._query_data_stats()

    def pull_data(self, dataset_id: str, output_path: Optional[str] = None) -> str:
        """Download a community dataset. Returns local path to CSV."""
        parts = dataset_id.split(":")
        if len(parts) != 4 or parts[0] != "data":
            raise ValueError(f"Invalid dataset_id: {dataset_id}")
        _, task, uploader, version = parts

        local_path = Path(output_path) if output_path else (
            self.cache_dir / "data" / task / uploader / f"{version}.csv"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            return str(local_path)

        if self._offline:
            raise RuntimeError(f"Offline mode: cannot download {dataset_id}")

        self._download_data(dataset_id, local_path)
        return str(local_path)

    # ---- Private Methods ----

    def _detect_task(self, bundle) -> str:
        """Detect drug vs epitope vs circRNA from bundle object."""
        # DrugModelBundle has smiles_col, EpitopeModelBundle has sequence_col,
        # circRNA bundle has circ_id / backsplice / sequence_col with circ flag.
        if hasattr(bundle, "smiles_col"):
            return "drug"
        elif getattr(bundle, "is_circular", False) or hasattr(bundle, "backsplice_idx"):
            return "circRNA"
        elif hasattr(bundle, "sequence_col"):
            return "epitope"
        else:
            raise ValueError("Unknown bundle type")

    def _detect_task_from_csv(self, df: pd.DataFrame) -> str:
        """Detect task from CSV columns."""
        cols = set(df.columns.str.lower())
        if "smiles" in cols:
            return "drug"
        elif "sequence" in cols:
            return "epitope"
        else:
            raise ValueError("Cannot detect task from CSV columns")

    def _save_model_local(self, bundle_path: Path, meta: ModelMeta) -> None:
        """Save model to local cache (offline mode)."""
        target = self.cache_dir / "models" / meta.task / meta.uploader / f"{meta.version}.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bundle_path.read_bytes())

        meta_path = target.with_suffix(".json")
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

    def _save_data_local(self, csv_path: Path, meta: DataMeta) -> None:
        """Save data to local cache (offline mode)."""
        version_hash = meta.dataset_id.split(":")[-1]
        target = self.cache_dir / "data" / meta.task / meta.uploader / f"{version_hash}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(csv_path.read_bytes())

        meta_path = target.with_suffix(".json")
        meta_path.write_text(json.dumps(asdict(meta), indent=2))

    def _list_models_local(self, task: Optional[str], uploader: Optional[str],
                           limit: int) -> List[Dict[str, Any]]:
        """List models from local cache."""
        models_dir = self.cache_dir / "models"
        if not models_dir.exists():
            return []

        results = []
        for task_dir in models_dir.iterdir():
            if task and task_dir.name != task:
                continue
            for uploader_dir in task_dir.iterdir():
                if uploader and uploader_dir.name != uploader:
                    continue
                for model_file in uploader_dir.glob("*.joblib"):
                    meta_file = model_file.with_suffix(".json")
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        results.append(meta)
                        if len(results) >= limit:
                            return results
        return results

    def _data_stats_local(self) -> Dict[str, Any]:
        """Get data stats from local cache."""
        data_dir = self.cache_dir / "data"
        if not data_dir.exists():
            return {"drug": {"n_samples": 0, "n_contributors": 0},
                    "epitope": {"n_samples": 0, "n_contributors": 0}}

        stats = {"drug": {"n_samples": 0, "n_contributors": set()},
                 "epitope": {"n_samples": 0, "n_contributors": set()}}

        for task_dir in data_dir.iterdir():
            if task_dir.name not in stats:
                continue
            for uploader_dir in task_dir.iterdir():
                stats[task_dir.name]["n_contributors"].add(uploader_dir.name)
                for csv_file in uploader_dir.glob("*.csv"):
                    meta_file = csv_file.with_suffix(".json")
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        stats[task_dir.name]["n_samples"] += meta.get("n_samples", 0)

        # Convert sets to counts
        for task in stats:
            stats[task]["n_contributors"] = len(stats[task]["n_contributors"])
        return stats

    # ---- Zenodo DOI minting ----

    def _mint_zenodo_doi(self, bundle_path: Path, model_id: str,
                         uploader_orcid: str, metadata: Dict[str, Any]) -> str:
        """Mint a Zenodo DOI for a model upload. Best-effort: returns "" on failure.

        Requires CONFLUENCIA_ZENODO_TOKEN. The model file is uploaded as the
        deposition payload, and metadata (title, author ORCID, license) is set
        so the resulting DOI is citation-ready.
        """
        if not ZENODO_TOKEN:
            return ""
        try:
            import json as _json
            # 1. Create empty deposition
            req = urllib.request.Request(
                ZENODO_API,
                method="POST",
                headers={
                    "Authorization": f"Bearer {ZENODO_TOKEN}",
                    "Content-Type": "application/json",
                },
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                dep = _json.loads(resp.read().decode())
            dep_id = dep["id"]
            # 2. Upload the model file
            upload_url = f"{ZENODO_API}/{dep_id}/files"
            filename = bundle_path.name
            # Build multipart form-data
            boundary = "----confluencia" + hashlib.sha256(filename.encode()).hexdigest()[:16]
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode() + bundle_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                upload_url,
                method="POST",
                headers={
                    "Authorization": f"Bearer {ZENODO_TOKEN}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                data=body,
            )
            urllib.request.urlopen(req, timeout=120).read()
            # 3. Set deposition metadata (title, author, license)
            title = f"Confluencia Hub model: {metadata.get('model_name', model_id)}"
            meta_payload = {
                "metadata": {
                    "title": title,
                    "upload_type": "software",
                    "description": (
                        f"Model uploaded to Confluencia Hub ({model_id}). "
                        f"ORCID: {uploader_orcid}. "
                        f"License: {metadata.get('license', 'MIT')}."
                    ),
                    "creators": [{"name": uploader_orcid, "affiliation": "Confluencia Hub"}],
                    "license": metadata.get("license", "MIT"),
                    "keywords": ["confluencia", "model", model_id.split(":")[1]],
                }
            }
            req = urllib.request.Request(
                f"{ZENODO_API}/{dep_id}",
                method="PUT",
                headers={
                    "Authorization": f"Bearer {ZENODO_TOKEN}",
                    "Content-Type": "application/json",
                },
                data=_json.dumps(meta_payload).encode(),
            )
            urllib.request.urlopen(req, timeout=30).read()
            # 4. Publish
            req = urllib.request.Request(
                f"{ZENODO_API}/{dep_id}/actions/publish",
                method="POST",
                headers={"Authorization": f"Bearer {ZENODO_TOKEN}"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                pub = _json.loads(resp.read().decode())
            doi = pub.get("doi", "")
            if doi:
                print(f"[Hub] DOI minted: {doi}")
            return doi
        except Exception as e:
            print(f"[Hub] Zenodo DOI minting failed: {e}")
            return ""

    # ---- HTTP Operations (HuggingFace Hub backend) ----

    def _hf_api(self):
        """Lazily import huggingface_hub. Returns the module or None."""
        try:
            from huggingface_hub import HfApi, hf_hub_download, upload_file
            return HfApi, hf_hub_download, upload_file
        except ImportError:
            return None

    def _hf_path_for(self, meta_or_id) -> str:
        """Compute the HF repo path for a model: {task}/{authority}/{hash}.joblib"""
        if isinstance(meta_or_id, ModelMeta):
            task = meta_or_id.task
            parts = meta_or_id.model_id.split(":")
        else:
            parts = meta_or_id.split(":")
            task = parts[1] if len(parts) >= 4 else "unknown"
        authority = parts[2] if len(parts) >= 4 else "anon"
        version = parts[3] if len(parts) >= 4 else "unknown"
        return f"{task}/{authority}/{version}.joblib"

    def _upload_model(self, bundle_path: Path, meta: ModelMeta) -> None:
        """Upload model to HuggingFace Hub. Falls back to local save if HF unavailable."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            # Fallback: local cache + instructions
            self._save_model_local(bundle_path, meta)
            print(f"[Hub] HF backend unavailable. Model saved locally. To share:")
            print(f"      {self.hub_url}/models/{meta.model_id}")
            print(f"      Set CONFLUENCIA_HF_TOKEN to enable HuggingFace Hub upload.")
            return

        HfApi, _, upload_file = hf
        repo_path_in_repo = self._hf_path_for(meta)
        try:
            upload_file(
                path_or_fileobj=str(bundle_path),
                path_in_repo=repo_path_in_repo,
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
            # Upload metadata as sidecar JSON
            meta_path = self.cache_dir / "staging" / f"{meta.version}_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(asdict(meta), indent=2))
            upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo=repo_path_in_repo.replace(".joblib", ".json"),
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
            # Also write a model card README at repo root if missing
            self._ensure_model_card(meta)
            print(f"[Hub] Uploaded to HF: {HF_REPO_ID}/{repo_path_in_repo}")
        except Exception as e:
            print(f"[Hub] HF upload failed ({e}), saved locally instead.")
            self._save_model_local(bundle_path, meta)

    def _ensure_model_card(self, meta: ModelMeta) -> None:
        """Write/update a README.md model card in the HF repo for this model."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            return
        HfApi, _, upload_file = hf
        readme = self._render_model_card(meta)
        tmp = self.cache_dir / "staging" / f"{meta.version}_README.md"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(readme, encoding="utf-8")
        try:
            upload_file(
                path_or_fileobj=str(tmp),
                path_in_repo=self._hf_path_for(meta).replace(".joblib", ".md"),
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
        except Exception:
            pass  # best-effort

    def _render_model_card(self, meta: ModelMeta) -> str:
        """Render a citation-ready model card (markdown)."""
        lines = [
            f"# {meta.model_name}",
            "",
            f"- **Model ID:** `{meta.model_id}`",
            f"- **Task:** {meta.task}",
            f"- **License:** {meta.license}",
            f"- **Uploaded:** {meta.uploaded_at}",
            f"- **Verification tier:** {meta.verification_level}",
        ]
        if meta.uploader_orcid:
            lines.append(f"- **Uploader ORCID:** [{meta.uploader_orcid}](https://orcid.org/{meta.uploader_orcid})")
        if meta.contributors:
            contribs = ", ".join(f"[{c}](https://orcid.org/{c})" for c in meta.contributors)
            lines.append(f"- **Contributors:** {contribs}")
        if meta.zenodo_doi:
            lines.append(f"- **DOI:** [10.5281/zenodo → {meta.zenodo_doi}](https://doi.org/{meta.zenodo_doi})")
        if meta.reproducibility_url:
            lines.append(f"- **Reproducibility:** {meta.reproducibility_url}")
        if meta.metrics:
            lines.append("")
            lines.append("## Metrics")
            for k, v in meta.metrics.items():
                lines.append(f"- {k}: {v}")
        if meta.circ_casp_metrics:
            lines.append("")
            lines.append("## Circ-CASP Competition Results")
            for k, v in meta.circ_casp_metrics.items():
                lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Citation")
        if meta.zenodo_doi:
            bib = (
                "@misc {{confluencia_hub_" + meta.version + "},\n"
                "  title  = {" + meta.model_name + "},\n"
                "  author = {" + (meta.uploader_orcid or meta.uploader) + "},\n"
                "  year   = {2026},\n"
                "  doi    = {" + meta.zenodo_doi + "},\n"
                "  note   = {Confluencia Hub model " + meta.model_id + "}\n"
                "}"
            )
            lines.append("```")
            lines.append(bib)
            lines.append("```")
        else:
            lines.append(f"Cite as: Confluencia Hub model `{meta.model_id}`.")
        return "\n".join(lines) + "\n"

    def _download_model(self, model_id: str, local_path: Path) -> None:
        """Download model from HuggingFace Hub."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            raise NotImplementedError(
                f"HuggingFace backend unavailable (huggingface_hub not installed or no token). "
                f"Model {model_id} not downloadable. "
                f"Set CONFLUENCIA_HUB_OFFLINE=1 to use local cache only."
            )
        _, hf_hub_download, _ = hf
        repo_path = self._hf_path_for(model_id)
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_path,
                repo_type="model",
                token=HF_TOKEN,
                cache_dir=str(self.cache_dir / "hf"),
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(Path(downloaded).read_bytes())
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_id} from HF: {e}") from e

    def _query_models(self, task: Optional[str], uploader: Optional[str],
                      limit: int) -> List[Dict[str, Any]]:
        """Query HuggingFace Hub for models. Falls back to local cache."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            return self._list_models_local(task, uploader, limit)
        HfApi, _, _ = hf
        try:
            api = HfApi(token=HF_TOKEN)
            results = []
            files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model")
            for f in files:
                if not f.endswith(".json"):
                    continue
                # Filter by task/uploader from path segments
                segs = f.split("/")
                if task and len(segs) > 0 and segs[0] != task:
                    continue
                if uploader and len(segs) > 2 and not segs[1].startswith(uploader[:4]):
                    continue
                try:
                    from huggingface_hub import hf_hub_download as _dl
                    meta_file = _dl(
                        repo_id=HF_REPO_ID, filename=f, repo_type="model",
                        token=HF_TOKEN, cache_dir=str(self.cache_dir / "hf"),
                    )
                    results.append(json.loads(Path(meta_file).read_text()))
                    if len(results) >= limit:
                        break
                except Exception:
                    continue
            # Sort by verification tier (desc) then download_count (desc)
            tier_order = {v: i for i, v in enumerate(VERIFICATION_LEVELS)}
            results.sort(
                key=lambda m: (-tier_order.get(m.get("verification_level", "unverified"), 0),
                               -m.get("download_count", 0))
            )
            return results
        except Exception:
            return self._list_models_local(task, uploader, limit)

    def _upload_data(self, csv_path: Path, meta: DataMeta) -> None:
        """Upload dataset to HuggingFace Hub. Falls back to local save."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            self._save_data_local(csv_path, meta)
            print(f"[Hub] HF backend unavailable. Data saved locally. To share:")
            print(f"      {self.hub_url}/data/{meta.dataset_id}")
            return
        _, _, upload_file = hf
        version = meta.dataset_id.split(":")[-1]
        repo_path = f"data/{meta.task}/{meta.uploader}/{version}.csv"
        try:
            upload_file(
                path_or_fileobj=str(csv_path),
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            # Sidecar metadata
            meta_tmp = self.cache_dir / "staging" / f"{version}_data_meta.json"
            meta_tmp.parent.mkdir(parents=True, exist_ok=True)
            meta_tmp.write_text(json.dumps(asdict(meta), indent=2))
            upload_file(
                path_or_fileobj=str(meta_tmp),
                path_in_repo=repo_path.replace(".csv", ".json"),
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print(f"[Hub] Data uploaded to HF: {HF_REPO_ID}/{repo_path}")
        except Exception as e:
            print(f"[Hub] HF data upload failed ({e}), saved locally.")
            self._save_data_local(csv_path, meta)

    def _download_data(self, dataset_id: str, local_path: Path) -> None:
        """Download dataset from HuggingFace Hub."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            raise NotImplementedError(
                f"HuggingFace backend unavailable. Dataset {dataset_id} not downloadable. "
                f"Set CONFLUENCIA_HUB_OFFLINE=1 to use local cache only."
            )
        _, hf_hub_download, _ = hf
        parts = dataset_id.split(":")
        task = parts[1] if len(parts) >= 4 else "unknown"
        uploader = parts[2] if len(parts) >= 4 else "anon"
        version = parts[3] if len(parts) >= 4 else "unknown"
        repo_path = f"data/{task}/{uploader}/{version}.csv"
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID, filename=repo_path, repo_type="dataset",
                token=HF_TOKEN, cache_dir=str(self.cache_dir / "hf"),
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(Path(downloaded).read_bytes())
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset_id} from HF: {e}") from e

    def _query_data_stats(self) -> Dict[str, Any]:
        """Query HuggingFace Hub for dataset stats. Falls back to local cache."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            return self._data_stats_local()
        HfApi, _, _ = hf
        try:
            api = HfApi(token=HF_TOKEN)
            files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
            stats: Dict[str, Any] = {
                "drug": {"n_samples": 0, "n_contributors": set()},
                "epitope": {"n_samples": 0, "n_contributors": set()},
                "circRNA": {"n_samples": 0, "n_contributors": set()},
            }
            for f in files:
                if not f.endswith(".json") or not f.startswith("data/"):
                    continue
                segs = f.split("/")
                task = segs[1] if len(segs) > 1 else ""
                uploader = segs[2] if len(segs) > 2 else ""
                if task not in stats:
                    continue
                try:
                    from huggingface_hub import hf_hub_download as _dl
                    meta_file = _dl(
                        repo_id=HF_REPO_ID, filename=f, repo_type="dataset",
                        token=HF_TOKEN, cache_dir=str(self.cache_dir / "hf"),
                    )
                    meta = json.loads(Path(meta_file).read_text())
                    stats[task]["n_samples"] += meta.get("n_samples", 0)
                    stats[task]["n_contributors"].add(uploader)
                except Exception:
                    continue
            for t in stats:
                stats[t]["n_contributors"] = len(stats[t]["n_contributors"])
            return stats
        except Exception:
            return self._data_stats_local()

    # ---- Impact tracking (downloads, citations, badges) ----

    def _fetch_download_count(self, meta: Dict[str, Any]) -> int:
        """Fetch the download count for a model from HuggingFace Hub."""
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            return meta.get("download_count", 0)
        HfApi, _, _ = hf
        try:
            api = HfApi(token=HF_TOKEN)
            repo_path = self._hf_path_for(meta.get("model_id", ""))
            info = api.model_info(repo_id=HF_REPO_ID)
            # Sum downloads across siblings matching this model file
            count = 0
            for s in (info.siblings or []):
                if s.rfilename == repo_path:
                    # HF doesn't expose per-file downloads directly; use repo-level
                    count = getattr(info, "downloads", 0) or 0
                    break
            return count
        except Exception:
            return meta.get("download_count", 0)

    @staticmethod
    def impact_badge(download_count: int, citation_count: int = 0) -> str:
        """Return an impact badge based on download/citation thresholds.

        🥇 gold   : downloads >= 1000 OR citations >= 10
        🥈 silver : downloads >= 500  OR citations >= 5
        🥉 bronze : downloads >= 100  OR citations >= 1
        (none)    : below bronze
        """
        if download_count >= 1000 or citation_count >= 10:
            return "🥇"
        if download_count >= 500 or citation_count >= 5:
            return "🥈"
        if download_count >= 100 or citation_count >= 1:
            return "🥉"
        return ""

    def get_contributor_stats(self, orcid: str) -> Dict[str, Any]:
        """Aggregate impact report for a contributor (by ORCID).

        Returns total downloads, citations, badge, and per-model breakdown.
        Useful for annual reports / CV entries.
        """
        orcid = validate_orcid(orcid)
        authority = orcid_short(orcid)
        all_models = self.list_models(limit=1000)
        mine = [m for m in all_models if m.get("uploader_orcid") == orcid
                or orcid in (m.get("contributors") or [])
                or m.get("model_id", "").split(":")[2] == authority]

        total_dl = 0
        total_cite = 0
        breakdown = []
        for m in mine:
            dl = self._fetch_download_count(m)
            cite = m.get("citation_count", 0)
            total_dl += dl
            total_cite += cite
            breakdown.append({
                "model_id": m.get("model_id"),
                "model_name": m.get("model_name"),
                "verification_level": m.get("verification_level", "unverified"),
                "downloads": dl,
                "citations": cite,
                "zenodo_doi": m.get("zenodo_doi", ""),
                "badge": self.impact_badge(dl, cite),
            })
        return {
            "orcid": orcid,
            "n_models": len(mine),
            "total_downloads": total_dl,
            "total_citations": total_cite,
            "badge": self.impact_badge(total_dl, total_cite),
            "models": breakdown,
        }

    # ---- Quality verification tiers ----

    def verify_model(self, model_id: str, level: str,
                     reviewer: str = "",
                     evidence_url: str = "",
                     circ_casp_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Promote a model's verification tier.

        Tiers (low → high): unverified < reproducible < verified < benchmark_top.
        A tier can only be raised, never lowered, by this method. Downgrades
        require manual repo admin action.

        Args:
            model_id: The hub model ID to verify.
            level: Target tier. Must be one of VERIFICATION_LEVELS.
            reviewer: ORCID or name of the verifier (for audit trail).
            evidence_url: URL supporting the claim (e.g., Circ-CASP results page,
                reproducible-training repo, benchmark report).
            circ_casp_metrics: If level == "benchmark_top" or "verified",
                attach competition metrics (e.g., {"rmsd": 8.2, "total": 71.5}).

        Returns the updated metadata dict.
        """
        if level not in VERIFICATION_LEVELS:
            raise ValueError(f"Invalid level {level!r}; must be one of {VERIFICATION_LEVELS}")

        # Load existing meta (local cache or HF)
        models = self.list_models(limit=10000)
        meta = next((m for m in models if m.get("model_id") == model_id), None)
        if meta is None:
            raise KeyError(f"Model {model_id} not found in hub")

        current_level = meta.get("verification_level", "unverified")
        current_idx = VERIFICATION_LEVELS.index(current_level)
        target_idx = VERIFICATION_LEVELS.index(level)
        if target_idx <= current_idx:
            return {"model_id": model_id, "verification_level": current_level,
                    "message": f"Already at {current_level} (>= {level}); no change."}

        # Tier-specific evidence requirements
        if level == "reproducible" and not (evidence_url or meta.get("reproducibility_url")):
            raise ValueError("Tier 'reproducible' requires evidence_url (training code repo).")
        if level in ("verified", "benchmark_top") and not (evidence_url or circ_casp_metrics):
            raise ValueError(
                f"Tier {level!r} requires evidence_url or circ_casp_metrics "
                f"(Circ-CASP blind-test results)."
            )

        meta["verification_level"] = level
        if evidence_url:
            meta["reproducibility_url"] = evidence_url
        if circ_casp_metrics:
            existing = meta.get("circ_casp_metrics", {}) or {}
            existing.update(circ_casp_metrics)
            meta["circ_casp_metrics"] = existing
        meta["ethics_reviewer"] = reviewer  # audit trail

        # Persist back to local cache + HF (best-effort)
        self._update_model_meta(meta)
        return {"model_id": model_id, "verification_level": level,
                "reviewer": reviewer, "evidence_url": evidence_url}

    def _update_model_meta(self, meta: Dict[str, Any]) -> None:
        """Persist updated metadata to local cache and (best-effort) HF."""
        # Local cache update
        try:
            parts = meta["model_id"].split(":")
            task, authority, version = parts[1], parts[2], parts[3]
            local_meta = (self.cache_dir / "models" / task / authority
                          / f"{version}.json")
            local_meta.parent.mkdir(parents=True, exist_ok=True)
            local_meta.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[Hub] Local meta update failed: {e}")

        # HF update (best-effort)
        hf = self._hf_api()
        if hf is None or not HF_TOKEN:
            return
        _, _, upload_file = hf
        try:
            tmp = self.cache_dir / "staging" / f"{meta.get('version','x')}_meta.json"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(meta, indent=2))
            repo_path = self._hf_path_for(meta["model_id"]).replace(".joblib", ".json")
            upload_file(
                path_or_fileobj=str(tmp),
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"[Hub] HF meta update failed: {e}")

    # ---- Circ-CASP competition linkage ----

    def push_circ_casp_submission(self, bundle_path: str,
                                  uploader_orcid: str,
                                  team_name: str,
                                  circ_casp_metrics: Dict[str, float],
                                  reproducibility_url: str = "",
                                  contributors: Optional[List[str]] = None,
                                  license: str = "MIT") -> str:
        """Upload a Circ-CASP competition model with auto-bound results.

        This wraps push_model for the competition flow:
        - task is forced to "circRNA"
        - circ_casp_metrics (rmsd, t1-t5, total) are bound to metadata
        - verification_level starts at "reproducible" if code repo provided,
          and can be promoted to "verified"/"benchmark_top" via verify_model()
          once blind-test results are published.

        Args:
            bundle_path: Path to the .joblib model bundle.
            uploader_orcid: Team lead ORCID (required for DOI minting).
            team_name: Circ-CASP team name (stored in tags for filtering).
            circ_casp_metrics: Competition results, e.g.:
                {"rmsd": 8.2, "t1": 80, "t2": 100, "t3": 75, "t4": 60,
                 "t5": 50, "total": 71.5, "rank": 3}
            reproducibility_url: Inference code repo (raises tier to reproducible).
            contributors: Additional team member ORCIDs.
            license: Model weight license (default MIT).

        Returns:
            model_id of the uploaded competition model.
        """
        metadata = {
            "model_name": f"Circ-CASP-{team_name}",
            "metrics": circ_casp_metrics,
            "circ_casp_metrics": circ_casp_metrics,
            "tags": ["circ-casp", "circRNA", team_name, "competition"],
            "n_samples": circ_casp_metrics.get("n_samples", 0),
        }
        model_id = self.push_model(
            bundle_path=bundle_path,
            metadata=metadata,
            uploader=team_name,
            uploader_orcid=uploader_orcid,
            contributors=contributors,
            reproducibility_url=reproducibility_url,
            license=license,
            mint_doi=True,
        )
        print(f"[Hub] Circ-CASP submission uploaded: {model_id}")
        print(f"      Metrics bound: {circ_casp_metrics}")
        if reproducibility_url:
            print(f"      Tier: reproducible (promote via verify_model after blind-test)")
        else:
            print(f"      Tier: unverified (provide reproducibility_url to upgrade)")
        return model_id

    def list_circ_casp_baselines(self, min_tier: str = "verified",
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """List circRNA models eligible as Circ-CASP baselines for the next cycle.

        A model is baseline-eligible if its verification_level >= min_tier
        (default "verified"). Benchmark-top models are surfaced first.
        """
        if min_tier not in VERIFICATION_LEVELS:
            raise ValueError(f"Invalid min_tier {min_tier!r}")
        min_idx = VERIFICATION_LEVELS.index(min_tier)
        all_circ = self.list_models(task="circRNA", limit=1000)
        eligible = [m for m in all_circ
                    if VERIFICATION_LEVELS.index(
                        m.get("verification_level", "unverified")) >= min_idx]
        tier_order = {v: i for i, v in enumerate(VERIFICATION_LEVELS)}
        eligible.sort(key=lambda m: (-tier_order.get(m.get("verification_level", ""), 0),
                                     -m.get("download_count", 0)))
        return eligible[:limit]

    def promote_baselines_to_next_circ_casp(self) -> List[str]:
        """Auto-promote benchmark_top circRNA models as next-cycle baselines.

        Returns the list of model_ids promoted. This is the mechanism by which
        Circ-CASP winners become permanent baselines — accumulating citations
        across competition cycles.
        """
        promoted = []
        for m in self.list_circ_casp_baselines(min_tier="benchmark_top", limit=10):
            tags = m.get("tags", []) or []
            if "circ-casp-baseline" not in tags:
                tags.append("circ-casp-baseline")
                m["tags"] = tags
                self._update_model_meta(m)
                promoted.append(m["model_id"])
                print(f"[Hub] Promoted to baseline: {m['model_id']}")
        return promoted



