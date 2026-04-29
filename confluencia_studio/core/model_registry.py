"""Model Version Registry for ConfluenciaStudio.

Manages model versions with metadata, metrics, and lineage tracking.
Supports model promotion, comparison, and retrieval.
"""

from __future__ import annotations

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ModelVersion:
    """Represents a versioned model."""
    id: str
    name: str
    version: str  # Semver (e.g., "1.0.0")
    path: str  # Model file path
    model_type: str  # rf, xgb, moe, etc.
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_hash: Optional[str] = None
    parent_version: Optional[str] = None  # Previous version ID
    tags: List[str] = field(default_factory=list)
    description: str = ""
    is_production: bool = False
    experiment_id: Optional[str] = None  # Link to experiment tracker

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelVersion':
        return cls(**d)


class ModelRegistry:
    """Manages model versioning and metadata.

    Features:
    - Register models with versioning
    - Track metrics and parameters
    - Promote models to production
    - Query by name, version, tags
    - Model lineage tracking

    Usage:
        registry = ModelRegistry()
        model_id = registry.register_model(
            path="models/drug_moe.pkl",
            name="drug-efficacy",
            version="1.0.0",
            model_type="moe",
            metrics={"r2": 0.72, "rmse": 0.15},
            parameters={"n_estimators": 100}
        )
        registry.set_production("drug-efficacy", "1.0.0")
    """

    def __init__(self, storage_dir: str = "~/.confluencia/models"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_dir / "models.json"
        self.models: Dict[str, ModelVersion] = {}  # id -> ModelVersion
        self.name_index: Dict[str, List[str]] = {}  # name -> [model_ids]

        self._load_db()

    def _load_db(self):
        """Load models from storage."""
        if not self.db_path.exists():
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.models = {k: ModelVersion.from_dict(v) for k, v in data.get("models", {}).items()}
            self.name_index = data.get("name_index", {})

        except Exception:
            self.models = {}
            self.name_index = {}

    def _save_db(self):
        """Save models to storage."""
        data = {
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "name_index": self.name_index,
        }

        temp_path = self.db_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(self.db_path)

    def _compute_file_hash(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()[:16]
        except Exception:
            return ""

    def _parse_version(self, version: str) -> tuple:
        """Parse semver string to tuple."""
        try:
            parts = version.split('.')
            return tuple(int(p) for p in parts[:3])
        except Exception:
            return (0, 0, 0)

    def register_model(
        self,
        path: str,
        name: str,
        version: str,
        model_type: str,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        parent_version: Optional[str] = None,
        experiment_id: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ) -> str:
        """Register a new model version.

        Args:
            path: Path to the model file
            name: Model name (e.g., "drug-efficacy")
            version: Version string (e.g., "1.0.0")
            model_type: Model type (rf, xgb, moe, etc.)
            metrics: Performance metrics
            parameters: Model hyperparameters
            tags: Tags for filtering
            description: Model description
            parent_version: Previous version ID for lineage
            experiment_id: Link to experiment
            dataset_path: Path to training dataset (for hashing)

        Returns:
            model_id: Unique identifier
        """
        model_id = str(uuid.uuid4())[:8]

        # Compute dataset hash if provided
        dataset_hash = None
        if dataset_path:
            dataset_hash = self._compute_file_hash(dataset_path)

        model = ModelVersion(
            id=model_id,
            name=name,
            version=version,
            path=path,
            model_type=model_type,
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or [],
            description=description,
            parent_version=parent_version,
            experiment_id=experiment_id,
            dataset_hash=dataset_hash,
        )

        self.models[model_id] = model

        # Update name index
        if name not in self.name_index:
            self.name_index[name] = []
        self.name_index[name].append(model_id)

        self._save_db()

        return model_id

    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Get a model by name and optionally version.

        Args:
            name: Model name
            version: Version string (latest if None)

        Returns:
            ModelVersion or None
        """
        if name not in self.name_index:
            return None

        model_ids = self.name_index[name]
        candidates = [self.models[mid] for mid in model_ids if mid in self.models]

        if not candidates:
            return None

        if version:
            for m in candidates:
                if m.version == version:
                    return m
            return None

        # Return latest version
        candidates.sort(key=lambda m: self._parse_version(m.version), reverse=True)
        return candidates[0]

    def get_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Get a model by ID."""
        return self.models.get(model_id)

    def list_models(self, name_filter: Optional[str] = None) -> List[ModelVersion]:
        """List all models, optionally filtered by name.

        Args:
            name_filter: Filter by name prefix

        Returns:
            List of ModelVersion
        """
        results = []

        for name, model_ids in self.name_index.items():
            if name_filter and not name.startswith(name_filter):
                continue

            for mid in model_ids:
                if mid in self.models:
                    results.append(self.models[mid])

        # Sort by creation time
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model.

        Args:
            name: Model name

        Returns:
            List of ModelVersion sorted by version number
        """
        if name not in self.name_index:
            return []

        model_ids = self.name_index[name]
        versions = [self.models[mid] for mid in model_ids if mid in self.models]

        # Sort by version number descending
        versions.sort(key=lambda m: self._parse_version(m.version), reverse=True)
        return versions

    def set_production(self, name: str, version: str) -> bool:
        """Promote a model version to production.

        Args:
            name: Model name
            version: Version to promote

        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        if not model:
            return False

        # Demote current production
        for mid in self.name_index.get(name, []):
            if mid in self.models and self.models[mid].is_production:
                self.models[mid].is_production = False

        model.is_production = True
        self._save_db()

        return True

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Get the production version of a model.

        Args:
            name: Model name

        Returns:
            Production ModelVersion or None
        """
        if name not in self.name_index:
            return None

        for mid in self.name_index[name]:
            if mid in self.models and self.models[mid].is_production:
                return self.models[mid]

        return None

    def compare_versions(self, name: str, versions: List[str]) -> Dict[str, Any]:
        """Compare multiple versions of a model.

        Args:
            name: Model name
            versions: List of version strings

        Returns:
            Comparison dict
        """
        model_versions = []
        for v in versions:
            m = self.get_model(name, v)
            if m:
                model_versions.append(m)

        if not model_versions:
            return {"error": "No models found"}

        # Gather all metric and parameter keys
        all_metrics = set()
        all_params = set()
        for m in model_versions:
            all_metrics.update(m.metrics.keys())
            all_params.update(m.parameters.keys())

        # Build comparison
        metrics_comparison = {}
        for key in sorted(all_metrics):
            metrics_comparison[key] = {m.version: m.metrics.get(key) for m in model_versions}

        params_comparison = {}
        for key in sorted(all_params):
            params_comparison[key] = {m.version: m.parameters.get(key) for m in model_versions}

        return {
            "name": name,
            "metrics": metrics_comparison,
            "parameters": params_comparison,
            "versions": [
                {
                    "version": m.version,
                    "created_at": m.created_at,
                    "is_production": m.is_production,
                }
                for m in model_versions
            ]
        }

    def delete_version(self, name: str, version: str) -> bool:
        """Delete a specific model version.

        Args:
            name: Model name
            version: Version to delete

        Returns:
            True if deleted
        """
        model = self.get_model(name, version)
        if not model:
            return False

        if model.is_production:
            return False  # Can't delete production model

        model_id = model.id
        del self.models[model_id]

        if name in self.name_index:
            self.name_index[name] = [mid for mid in self.name_index[name] if mid != model_id]

        self._save_db()
        return True

    def get_lineage(self, model_id: str) -> List[ModelVersion]:
        """Get the lineage (ancestors) of a model.

        Args:
            model_id: Starting model ID

        Returns:
            List of ModelVersion from oldest to newest
        """
        lineage = []
        current = self.models.get(model_id)

        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.models.get(current.parent_version)
            else:
                break

        return list(reversed(lineage))

    def export_registry(self, output_path: str):
        """Export the entire registry to JSON."""
        data = {
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "name_index": self.name_index,
            "exported_at": datetime.now().isoformat(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def import_registry(self, input_path: str, merge: bool = True) -> int:
        """Import models from a registry export.

        Args:
            input_path: Input file path
            merge: If True, merge with existing; if False, replace

        Returns:
            Number of models imported
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not merge:
            self.models = {}
            self.name_index = {}

        count = 0
        for model_id, model_data in data.get("models", {}).items():
            if model_id not in self.models:
                model = ModelVersion.from_dict(model_data)
                self.models[model_id] = model

                if model.name not in self.name_index:
                    self.name_index[model.name] = []
                self.name_index[model.name].append(model_id)
                count += 1

        self._save_db()
        return count


# Global singleton
_model_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
