"""Confluencia Hub — community model and data sharing.

Design principles:
1. Federated model sharing: users upload .joblib bundles (no raw data exposed)
2. Optional data pool: users can contribute CSVs to community datasets
3. Privacy-first: no SMILES/sequences are logged; only aggregated metrics
4. Versioning: each upload gets a version hash for reproducibility
5. Integration: hub models can be loaded via cf_drug_predict(bundle_id="hub:xxx")

Usage:
    from confluencia_cli.hub import ConfluenciaHub

    hub = ConfluenciaHub()

    # Upload a trained model (federated)
    hub.push_model("my_drug_model.joblib", metadata={"model_name": "ridge", "r2": 0.91})

    # List available community models
    models = hub.list_models(task="drug")
    print(models)

    # Download and use a community model
    bundle_path = hub.pull_model("hub:drug:user123:v1")
    pred = bridge.drug_predict(bundle_path, "CC(=O)Oc1ccccc1C(=O)O")

    # Optionally contribute data to community pool
    hub.push_data("my_drug_data.csv", license="CC-BY-4.0", anonymous=True)

    # Query community data statistics (no raw data exposed)
    stats = hub.data_stats()
    print(stats)  # {"drug": {"n_samples": 15000, "n_contributors": 12}, ...}
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import urllib.request
import urllib.error
import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HUB_URL = os.environ.get("CONFLUENCIA_HUB_URL", "https://hub.confluencia.org/api/v1")
HUB_CACHE_DIR = Path(os.environ.get("CONFLUENCIA_HUB_CACHE", Path.home() / ".confluencia" / "hub"))


@dataclass
class ModelMeta:
    """Metadata for a shared model."""
    model_id: str  # hub:drug:user123:v1
    task: Literal["drug", "epitope"]
    model_name: str  # ridge, hgb, etc.
    uploader: str  # anonymous or user ID
    version: str  # git-style hash
    uploaded_at: str
    metrics: Dict[str, float]  # {"r2": 0.91, "mae": 0.03}
    n_samples: int  # training set size
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"


@dataclass
class DataMeta:
    """Metadata for a shared dataset (stats only, no raw data)."""
    dataset_id: str
    task: Literal["drug", "epitope"]
    uploader: str
    uploaded_at: str
    n_samples: int
    n_features: int
    columns: List[str]  # column names only
    license: str
    anonymous: bool = True


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
                   strip_env_medians: bool = False) -> str:
        """Upload a trained model bundle to the hub.

        Returns the model_id (e.g., "hub:drug:user123:v1").
        The raw training data is NOT uploaded — only the model weights.

        Args:
            strip_env_medians: If True, remove env_medians from the bundle
                before sharing. This eliminates any statistical traces of the
                training data, at the cost of needing users to provide their
                own env values at prediction time.
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

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

        model_id = f"hub:{task}:{uploader}:{file_hash}"

        meta = ModelMeta(
            model_id=model_id,
            task=task,
            model_name=metadata.get("model_name", "unknown") if metadata else "unknown",
            uploader=uploader,
            version=file_hash,
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
            metrics=metadata.get("metrics", {}) if metadata else {},
            n_samples=metadata.get("n_samples", 0) if metadata else 0,
            tags=metadata.get("tags", []) if metadata else [],
            license=license,
        )

        if self._offline:
            # Save locally only
            self._save_model_local(upload_path, meta)
            return model_id

        # Upload to hub server
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
        """Detect drug vs epitope from bundle object."""
        # DrugModelBundle has smiles_col, EpitopeModelBundle has sequence_col
        if hasattr(bundle, "smiles_col"):
            return "drug"
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

    # ---- HTTP Operations (stub implementations) ----

    def _upload_model(self, bundle_path: Path, meta: ModelMeta) -> None:
        """Upload model to hub server. Stub — requires server implementation."""
        # In production, this would POST to {HUB_URL}/models/
        # For now, save locally and print instructions
        self._save_model_local(bundle_path, meta)
        print(f"[Hub] Model saved locally. To share with community, upload to:")
        print(f"      {self.hub_url}/models/{meta.model_id}")

    def _download_model(self, model_id: str, local_path: Path) -> None:
        """Download model from hub server. Stub — requires server implementation."""
        raise NotImplementedError(
            f"Hub server not yet deployed. Model {model_id} not available.\n"
            f"Set CONFLUENCIA_HUB_OFFLINE=1 to use local cache only."
        )

    def _upload_data(self, csv_path: Path, meta: DataMeta) -> None:
        """Upload data to hub server. Stub — requires server implementation."""
        self._save_data_local(csv_path, meta)
        print(f"[Hub] Data saved locally. To share with community, upload to:")
        print(f"      {self.hub_url}/data/{meta.dataset_id}")

    def _download_data(self, dataset_id: str, local_path: Path) -> None:
        """Download data from hub server. Stub — requires server implementation."""
        raise NotImplementedError(
            f"Hub server not yet deployed. Dataset {dataset_id} not available.\n"
            f"Set CONFLUENCIA_HUB_OFFLINE=1 to use local cache only."
        )

    def _query_models(self, task: Optional[str], uploader: Optional[str],
                      limit: int) -> List[Dict[str, Any]]:
        """Query hub server for models. Stub."""
        return self._list_models_local(task, uploader, limit)

    def _query_data_stats(self) -> Dict[str, Any]:
        """Query hub server for data stats. Stub."""
        return self._data_stats_local()
