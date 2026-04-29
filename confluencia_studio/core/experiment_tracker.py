"""Experiment Tracker for ConfluenciaStudio.

Tracks experiments with parameters, metrics, and artifacts.
Provides comparison and query capabilities for model development workflows.
"""

from __future__ import annotations

import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExperimentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Experiment:
    """Represents a tracked experiment."""
    id: str
    name: str
    module: str  # "drug", "epitope", "circrna", "joint", "bench"
    parameters: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # File paths
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: str = "pending"
    notes: str = ""
    parent_id: Optional[str] = None  # For experiment lineage

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Experiment':
        return cls(**d)


class ExperimentTracker:
    """Manages experiment tracking and comparison.

    Features:
    - Start/finish experiment lifecycle
    - Log metrics and artifacts
    - Query by module, status, tags
    - Compare multiple experiments
    - Persist to JSON storage

    Usage:
        tracker = ExperimentTracker()
        exp_id = tracker.start_experiment("Drug RF Model", "drug", {"n_estimators": 100})
        tracker.log_metric(exp_id, "r2", 0.72)
        tracker.log_metric(exp_id, "rmse", 0.15)
        tracker.log_artifact(exp_id, "models/drug_rf.pkl")
        tracker.finish_experiment(exp_id)
    """

    def __init__(self, storage_dir: str = "~/.confluencia/experiments"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_dir / "experiments.json"
        self.experiments: Dict[str, Experiment] = {}

        self._load_db()

    def _load_db(self):
        """Load experiments from storage."""
        if not self.db_path.exists():
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.experiments = {k: Experiment.from_dict(v) for k, v in data.items()}
        except Exception:
            self.experiments = {}

    def _save_db(self):
        """Save experiments to storage."""
        data = {k: v.to_dict() for k, v in self.experiments.items()}
        temp_path = self.db_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(self.db_path)

    def start_experiment(
        self,
        name: str,
        module: str,
        parameters: Dict[str, Any],
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """Start a new experiment.

        Args:
            name: Human-readable experiment name
            module: Module name (drug, epitope, etc.)
            parameters: Hyperparameters and configuration
            tags: Optional tags for filtering
            parent_id: Optional parent experiment ID for lineage
            notes: Optional notes

        Returns:
            experiment_id: Unique identifier
        """
        exp_id = str(uuid.uuid4())[:8]

        experiment = Experiment(
            id=exp_id,
            name=name,
            module=module,
            parameters=parameters,
            tags=tags or [],
            parent_id=parent_id,
            notes=notes,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        self.experiments[exp_id] = experiment
        self._save_db()

        return exp_id

    def log_metric(self, exp_id: str, key: str, value: Any):
        """Log a metric for an experiment.

        Args:
            exp_id: Experiment ID
            key: Metric name
            value: Metric value (number, string, or list)
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        self.experiments[exp_id].metrics[key] = value
        self._save_db()

    def log_metrics(self, exp_id: str, metrics: Dict[str, Any]):
        """Log multiple metrics at once.

        Args:
            exp_id: Experiment ID
            metrics: Dict of metric name -> value
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        self.experiments[exp_id].metrics.update(metrics)
        self._save_db()

    def log_artifact(self, exp_id: str, file_path: str, description: str = ""):
        """Log an artifact file for an experiment.

        Args:
            exp_id: Experiment ID
            file_path: Path to the artifact file
            description: Optional description
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        artifact_entry = file_path
        if description:
            artifact_entry = f"{file_path} ({description})"

        self.experiments[exp_id].artifacts.append(artifact_entry)
        self._save_db()

    def log_tag(self, exp_id: str, tag: str):
        """Add a tag to an experiment."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        if tag not in self.experiments[exp_id].tags:
            self.experiments[exp_id].tags.append(tag)
            self._save_db()

    def finish_experiment(self, exp_id: str, status: str = "completed", notes: str = ""):
        """Mark an experiment as finished.

        Args:
            exp_id: Experiment ID
            status: Final status (completed, failed, cancelled)
            notes: Optional final notes
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        exp = self.experiments[exp_id]
        exp.status = status
        exp.finished_at = datetime.now().isoformat()
        if notes:
            exp.notes = notes

        self._save_db()

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(exp_id)

    def list_experiments(
        self,
        module: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Experiment]:
        """List experiments with optional filtering.

        Args:
            module: Filter by module name
            status: Filter by status
            tags: Filter by tags (AND logic)
            limit: Maximum number to return

        Returns:
            List of experiments sorted by creation time (newest first)
        """
        results = []

        for exp in self.experiments.values():
            # Apply filters
            if module and exp.module != module:
                continue
            if status and exp.status != status:
                continue
            if tags:
                if not all(t in exp.tags for t in tags):
                    continue

            results.append(exp)

        # Sort by creation time descending
        results.sort(key=lambda e: e.created_at, reverse=True)

        return results[:limit]

    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments side-by-side.

        Args:
            exp_ids: List of experiment IDs to compare

        Returns:
            Dict with parameters comparison, metrics comparison, and summary
        """
        experiments = [self.experiments.get(eid) for eid in exp_ids]
        experiments = [e for e in experiments if e is not None]

        if not experiments:
            return {"error": "No experiments found"}

        # Gather all parameter and metric keys
        all_params = set()
        all_metrics = set()
        for exp in experiments:
            all_params.update(exp.parameters.keys())
            all_metrics.update(exp.metrics.keys())

        # Build comparison tables
        params_comparison = {}
        for key in sorted(all_params):
            params_comparison[key] = {exp.id: exp.parameters.get(key) for exp in experiments}

        metrics_comparison = {}
        for key in sorted(all_metrics):
            metrics_comparison[key] = {exp.id: exp.metrics.get(key) for exp in experiments}

        # Summary
        summary = {
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "module": exp.module,
                    "status": exp.status,
                    "created_at": exp.created_at,
                }
                for exp in experiments
            ],
            "parameter_count": len(all_params),
            "metric_count": len(all_metrics),
        }

        return {
            "parameters": params_comparison,
            "metrics": metrics_comparison,
            "summary": summary,
        }

    def delete_experiment(self, exp_id: str) -> bool:
        """Delete an experiment record.

        Args:
            exp_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        if exp_id in self.experiments:
            del self.experiments[exp_id]
            self._save_db()
            return True
        return False

    def get_best_experiment(
        self,
        module: str,
        metric: str,
        mode: str = "max"
    ) -> Optional[Experiment]:
        """Find the best experiment by a metric.

        Args:
            module: Module to search
            metric: Metric name to compare
            mode: "max" or "min"

        Returns:
            Best experiment or None
        """
        experiments = self.list_experiments(module=module, status="completed")

        valid_exps = [
            e for e in experiments
            if metric in e.metrics and isinstance(e.metrics[metric], (int, float))
        ]

        if not valid_exps:
            return None

        if mode == "max":
            return max(valid_exps, key=lambda e: e.metrics[metric])
        else:
            return min(valid_exps, key=lambda e: e.metrics[metric])

    def export_experiments(self, output_path: str, exp_ids: Optional[List[str]] = None):
        """Export experiments to JSON file.

        Args:
            output_path: Output file path
            exp_ids: Specific experiment IDs to export (all if None)
        """
        if exp_ids:
            experiments = {k: v for k, v in self.experiments.items() if k in exp_ids}
        else:
            experiments = self.experiments

        data = {k: v.to_dict() for k, v in experiments.items()}

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def import_experiments(self, input_path: str) -> int:
        """Import experiments from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Number of experiments imported
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for exp_id, exp_data in data.items():
            if exp_id not in self.experiments:
                self.experiments[exp_id] = Experiment.from_dict(exp_data)
                count += 1

        self._save_db()
        return count


# Global singleton
_experiment_tracker: Optional[ExperimentTracker] = None

def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker instance."""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker()
    return _experiment_tracker
