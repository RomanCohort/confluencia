"""Frontend client for Confluencia 2.0 Drug cloud/local mode.

Provides a unified interface for the Streamlit frontend to call
computation functions either locally or through the remote API server.

Usage in app.py:
    from api.frontend_client import CloudClient

    client = CloudClient(mode="remote", server_url="http://your-server:8000")
    # or
    client = CloudClient(mode="local")

    # Drop-in replacement for train_and_predict_drug()
    result_df, curve_df, artifacts, report = client.train_and_predict(df, ...)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from api.serialization import csv_str_to_df, df_to_csv_str


@dataclass
class CloudConnectionStatus:
    """Status of the cloud server connection."""
    connected: bool = False
    url: str = ""
    error: str = ""
    latency_ms: float = 0.0


class CloudClient:
    """Unified frontend client that routes computation to local or remote backend.

    In local mode, calls core functions directly (identical to existing behavior).
    In remote mode, calls the cloud API server via HTTP.

    The remote mode stores model_id references instead of full model objects.
    """

    def __init__(self, mode: str = "local", server_url: str = "http://localhost:8000"):
        self.mode = mode
        self.server_url = server_url.rstrip("/")
        self._remote_model_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def check_connection(self) -> CloudConnectionStatus:
        """Check if the remote server is reachable."""
        if self.mode == "local":
            return CloudConnectionStatus(connected=True, url="local")
        try:
            import httpx
            import time
            t0 = time.time()
            resp = httpx.get(f"{self.server_url}/api/health", timeout=5.0)
            latency = (time.time() - t0) * 1000
            if resp.status_code == 200:
                return CloudConnectionStatus(
                    connected=True,
                    url=self.server_url,
                    latency_ms=round(latency, 1),
                )
            return CloudConnectionStatus(
                connected=False,
                url=self.server_url,
                error=f"HTTP {resp.status_code}",
            )
        except Exception as e:
            return CloudConnectionStatus(
                connected=False,
                url=self.server_url,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Drug training & prediction
    # ------------------------------------------------------------------

    def train_and_predict(
        self,
        df: pd.DataFrame,
        compute_mode: str = "auto",
        model_backend: str = "moe",
        dynamics_model: str = "ctm",
        legacy_cfg: Any = None,
        adaptive_enabled: bool = False,
        adaptive_strength: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
        """Drop-in replacement for train_and_predict_drug()."""
        if self.mode == "local":
            from core.training import train_and_predict_drug
            return train_and_predict_drug(
                df=df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                dynamics_model=dynamics_model,
                legacy_cfg=legacy_cfg,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )

        return self._remote_train_and_predict(
            df, compute_mode, model_backend, dynamics_model,
            adaptive_enabled, adaptive_strength,
        )

    def train(
        self,
        df: pd.DataFrame,
        compute_mode: str = "auto",
        model_backend: str = "moe",
        dynamics_model: str = "ctm",
        legacy_cfg: Any = None,
        adaptive_enabled: bool = False,
        adaptive_strength: float = 0.2,
    ) -> Any:
        """Drop-in replacement for train_drug_model().

        Returns (trained_model_or_model_id, is_remote).
        In local mode, returns the DrugTrainedModel directly.
        In remote mode, returns a dict with model_id and metadata.
        """
        if self.mode == "local":
            from core.training import train_drug_model
            return train_drug_model(
                df=df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                dynamics_model=dynamics_model,
                legacy_cfg=legacy_cfg,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )

        return self._remote_train(
            df, compute_mode, model_backend, dynamics_model,
            adaptive_enabled, adaptive_strength,
        )

    def predict(
        self,
        df: pd.DataFrame,
        trained_model: Any,
        adaptive_enabled: bool = False,
        adaptive_strength: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
        """Drop-in replacement for predict_drug_with_model()."""
        if self.mode == "local":
            from core.training import predict_drug_with_model
            return predict_drug_with_model(
                df=df,
                trained_model=trained_model,
                adaptive_enabled=adaptive_enabled,
                adaptive_strength=adaptive_strength,
            )

        model_id = self._resolve_model_id(trained_model)
        return self._remote_predict(df, model_id, adaptive_enabled, adaptive_strength)

    def export_model(self, trained_model: Any) -> bytes:
        """Export model to bytes. Works in both modes."""
        if self.mode == "local":
            from core.training import export_drug_model_bytes
            return export_drug_model_bytes(trained_model)

        model_id = self._resolve_model_id(trained_model)
        return self._remote_export_model(model_id)

    def import_model(self, payload: bytes) -> Any:
        """Import model from bytes. Works in both modes."""
        if self.mode == "local":
            from core.training import import_drug_model_bytes
            return import_drug_model_bytes(payload, allow_unsafe_deserialization=True)

        return self._remote_import_model(payload)

    def get_model_metadata(self, trained_model: Any) -> Dict[str, str]:
        """Get model metadata."""
        if self.mode == "local":
            from core.training import get_drug_model_metadata
            return get_drug_model_metadata(trained_model)

        if isinstance(trained_model, dict):
            return trained_model.get("metadata", {})
        return {}

    # ------------------------------------------------------------------
    # Molecular evolution
    # ------------------------------------------------------------------

    def evolve_molecules(
        self,
        seed_smiles: List[str],
        cfg: Any,
        ed2mol_repo_dir: str,
        ed2mol_config_path: str,
        ed2mol_python_cmd: str = "python",
    ) -> Tuple[pd.DataFrame, Any]:
        """Drop-in replacement for evolve_molecules_with_reflection()."""
        if self.mode == "local":
            from core.evolution import evolve_molecules_with_reflection
            return evolve_molecules_with_reflection(
                seed_smiles=seed_smiles,
                cfg=cfg,
                ed2mol_repo_dir=ed2mol_repo_dir,
                ed2mol_config_path=ed2mol_config_path,
                ed2mol_python_cmd=ed2mol_python_cmd,
            )

        return self._remote_evolve_molecules(
            seed_smiles, cfg, ed2mol_repo_dir,
            ed2mol_config_path, ed2mol_python_cmd,
        )

    def evolve_cirrna(self, cfg: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drop-in replacement for evolve_cirrna_sequences()."""
        if self.mode == "local":
            from core.evolution import evolve_cirrna_sequences
            return evolve_cirrna_sequences(cfg=cfg)

        return self._remote_evolve_cirrna(cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_id(self, trained_model: Any) -> str:
        """Extract model_id from a trained model reference."""
        if isinstance(trained_model, dict) and "model_id" in trained_model:
            return trained_model["model_id"]
        raise ValueError("In remote mode, trained_model must be a dict with 'model_id'")

    def _post_json(self, path: str, payload: dict, timeout: float = 300.0) -> dict:
        """POST JSON to remote server and return response."""
        import httpx
        resp = httpx.post(f"{self.server_url}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_bytes(self, path: str, payload: bytes, timeout: float = 60.0) -> bytes:
        """POST raw bytes to remote server and return response bytes."""
        import httpx
        resp = httpx.post(f"{self.server_url}{path}", content=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Remote implementations
    # ------------------------------------------------------------------

    def _remote_train_and_predict(self, df, compute_mode, model_backend,
                                  dynamics_model, adaptive_enabled, adaptive_strength):
        from api.schemas import ArtifactsSchema, TrainingReportSchema
        payload = {
            "data_csv": df_to_csv_str(df),
            "compute_mode": compute_mode,
            "model_backend": model_backend,
            "dynamics_model": dynamics_model,
            "adaptive_enabled": adaptive_enabled,
            "adaptive_strength": adaptive_strength,
        }
        data = self._post_json("/api/drug/train-and-predict", payload)
        result_df = csv_str_to_df(data["result_csv"])
        curve_df = csv_str_to_df(data["curve_csv"])
        artifacts_data = data.get("artifacts", {})
        report_data = data.get("report", {})
        return result_df, curve_df, artifacts_data, report_data

    def _remote_train(self, df, compute_mode, model_backend,
                      dynamics_model, adaptive_enabled, adaptive_strength):
        payload = {
            "data_csv": df_to_csv_str(df),
            "compute_mode": compute_mode,
            "model_backend": model_backend,
            "dynamics_model": dynamics_model,
            "adaptive_enabled": adaptive_enabled,
            "adaptive_strength": adaptive_strength,
        }
        data = self._post_json("/api/drug/train", payload)
        # Store the model_id for later use
        self._remote_model_id = data["model_id"]
        return {"model_id": data["model_id"], "metadata": data.get("metadata", {})}

    def _remote_predict(self, df, model_id, adaptive_enabled, adaptive_strength):
        payload = {
            "model_id": model_id,
            "data_csv": df_to_csv_str(df),
            "adaptive_enabled": adaptive_enabled,
            "adaptive_strength": adaptive_strength,
        }
        data = self._post_json("/api/drug/predict", payload)
        result_df = csv_str_to_df(data["result_csv"])
        curve_df = csv_str_to_df(data["curve_csv"])
        return result_df, curve_df, data.get("artifacts", {}), data.get("report", {})

    def _remote_export_model(self, model_id: str) -> bytes:
        import httpx
        payload = {"model_id": model_id}
        resp = httpx.post(
            f"{self.server_url}/api/model/export",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.content

    def _remote_import_model(self, payload: bytes) -> Any:
        import httpx
        resp = httpx.post(
            f"{self.server_url}/api/model/import",
            content=payload,
            timeout=60.0,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        data = resp.json()
        self._remote_model_id = data["model_id"]
        return {"model_id": data["model_id"], "metadata": {}}

    def _remote_evolve_molecules(self, seed_smiles, cfg, ed2mol_repo_dir,
                                 ed2mol_config_path, ed2mol_python_cmd):
        from dataclasses import fields
        from api.schemas import EvolutionConfigSchema
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        payload = {
            "seed_smiles": seed_smiles,
            "config": cfg_dict,
            "ed2mol_repo_dir": ed2mol_repo_dir,
            "ed2mol_config_path": ed2mol_config_path,
            "ed2mol_python_cmd": ed2mol_python_cmd,
        }
        data = self._post_json("/api/evolution/molecules", payload, timeout=600.0)
        result_df = csv_str_to_df(data["results_csv"])
        return result_df, data.get("artifacts", {})

    def _remote_evolve_cirrna(self, cfg):
        from dataclasses import fields
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        payload = {"config": cfg_dict}
        data = self._post_json("/api/evolution/cirrna", payload, timeout=600.0)
        result_df = csv_str_to_df(data["results_csv"])
        return result_df, data.get("artifacts", {})


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_cloud_client(
    mode: str = "local",
    server_url: str = "http://localhost:8000",
) -> CloudClient:
    """Create a CloudClient instance.

    Parameters
    ----------
    mode : str
        "local" for direct computation, "remote" for cloud API.
    server_url : str
        Base URL of the cloud server.
    """
    return CloudClient(mode=mode, server_url=server_url)
