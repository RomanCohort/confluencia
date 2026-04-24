"""Cloud client for Confluencia 2.0 Epitope.

Provides a RESTful API client that delegates training, prediction, and model
management to a remote cloud server.  The client is designed as a **slot-in
replacement** for local training/prediction functions so that the Streamlit
frontends can switch between local and cloud mode with minimal code changes.

Usage::

    from core.cloud_client import CloudEpitopeClient

    client = CloudEpitopeClient.from_config()          # reads cloud_config.yaml
    if client.is_available():
        report = client.train(df, backend="hgb")
        result_df = client.predict(model_id, df)
"""

from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cloud_config import CloudConfig, load_cloud_config

# ---------------------------------------------------------------------------
# Minimal HTTP helper (uses urllib from stdlib, no extra dependency required)
# ---------------------------------------------------------------------------

try:
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error

    _HAS_URLLIB = True
except Exception:
    _HAS_URLLIB = False


def _urlopen_with_retry(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    timeout: int = 300,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[int, bytes]:
    """Open *url* with simple retry logic.  Returns (status_code, response_body)."""
    if not _HAS_URLLIB:
        raise RuntimeError("urllib is not available in this environment.")

    last_exc: Optional[Exception] = None
    for attempt in range(max(max_retries, 1)):
        try:
            req = _urllib_request.Request(url, data=body, method=method)
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            with _urllib_request.urlopen(req, timeout=timeout) as resp:
                return int(resp.status), resp.read()
        except _urllib_error.HTTPError as exc:
            body_text = b""
            try:
                body_text = exc.read()
            except Exception:
                pass
            if exc.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                last_exc = exc
                time.sleep(retry_delay)
                continue
            return int(exc.code), body_text
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue

    raise ConnectionError(f"Request to {url} failed after {max_retries} retries") from last_exc


# ---------------------------------------------------------------------------
# Data classes for cloud results
# ---------------------------------------------------------------------------

@dataclass
class CloudTrainResult:
    """Result returned by a cloud training request."""
    task_id: str
    model_id: str
    status: str  # "completed" | "failed" | "pending"
    metrics: Dict[str, float] = field(default_factory=dict)
    model_backend: str = ""
    sample_count: int = 0
    error_message: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudPredictResult:
    """Result returned by a cloud prediction request."""
    task_id: str
    status: str
    predictions: Optional[np.ndarray] = None
    sensitivity: Optional[Dict[str, Any]] = None
    error_message: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudHealthStatus:
    """Health check result."""
    available: bool
    version: str = ""
    gpu_available: bool = False
    message: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class CloudEpitopeClient:
    """RESTful API client for Confluencia cloud server."""

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self._session_id = uuid.uuid4().hex[:12]

    # -- Factory helpers -----------------------------------------------------

    @classmethod
    def from_config(cls, path: Optional[str] = None) -> "CloudEpitopeClient":
        cfg = load_cloud_config(path)
        return cls(cfg)

    @classmethod
    def from_params(
        cls,
        server_url: str,
        token: str,
        *,
        api_prefix: str = "/api/v1",
        timeout: int = 300,
        mode: str = "cloud",
    ) -> "CloudEpitopeClient":
        cfg = CloudConfig(
            enabled=True,
            server=CloudServerConfig.__new__(CloudServerConfig),
            auth=CloudAuthConfig.__new__(CloudAuthConfig),
            mode=CloudModeConfig.__new__(CloudModeConfig),
            storage=CloudStorageConfig.__new__(CloudStorageConfig),
        )
        # Manually init dataclasses (avoid __init__ defaults fight)
        cfg.server = CloudServerConfig(url=server_url, api_prefix=api_prefix, timeout=timeout)
        cfg.auth = CloudAuthConfig(token=token)
        cfg.mode = CloudModeConfig(default=mode)
        cfg.storage = CloudStorageConfig()
        return cls(cfg)

    # -- Availability --------------------------------------------------------

    def is_available(self) -> bool:
        return self.config.enabled and bool(self.config.server.url)

    # -- HTTP helpers --------------------------------------------------------

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            self.config.auth.header_name: self.config.auth.token,
            "X-Client-Session": self._session_id,
        }
        if extra:
            h.update(extra)
        return h

    def _url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        url = self._url(path)
        raw_body: Optional[bytes] = None
        if body is not None:
            raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")

        status, resp_bytes = _urlopen_with_retry(
            url,
            method=method,
            headers=self._headers(extra_headers),
            body=raw_body,
            timeout=self.config.server.timeout,
            max_retries=self.config.server.max_retries,
            retry_delay=self.config.server.retry_delay,
        )

        try:
            resp_data = json.loads(resp_bytes.decode("utf-8"))
        except Exception:
            resp_data = {"raw": resp_bytes.decode("utf-8", errors="replace")}

        return status, resp_data

    def _request_binary(
        self,
        path: str,
        *,
        method: str = "GET",
        extra_headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> Tuple[int, bytes]:
        url = self._url(path)
        hdrs = self._headers(extra_headers)
        if body is not None:
            hdrs["Content-Type"] = "application/octet-stream"
        return _urlopen_with_retry(
            url,
            method=method,
            headers=hdrs,
            body=body,
            timeout=self.config.server.timeout,
            max_retries=self.config.server.max_retries,
            retry_delay=self.config.server.retry_delay,
        )

    # -- Public API: Health --------------------------------------------------

    def health_check(self) -> CloudHealthStatus:
        """Check whether the cloud server is alive and ready."""
        if not self.is_available():
            return CloudHealthStatus(available=False, message="Cloud client not configured")

        t0 = time.monotonic()
        try:
            status, data = self._request("/health")
            latency = (time.monotonic() - t0) * 1000.0
            if status == 200:
                return CloudHealthStatus(
                    available=True,
                    version=str(data.get("version", "")),
                    gpu_available=bool(data.get("gpu_available", False)),
                    message=str(data.get("message", "OK")),
                    latency_ms=latency,
                )
            return CloudHealthStatus(
                available=False,
                message=f"HTTP {status}: {data}",
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000.0
            return CloudHealthStatus(
                available=False,
                message=str(exc),
                latency_ms=latency,
            )

    # -- Public API: Training ------------------------------------------------

    def submit_train(
        self,
        df: pd.DataFrame,
        *,
        model_backend: str = "hgb",
        compute_mode: str = "auto",
        torch_cfg: Optional[Dict[str, Any]] = None,
    ) -> CloudTrainResult:
        """Submit a training job to the cloud server.

        The DataFrame is serialized to CSV and sent in the request body.
        Returns a CloudTrainResult with the task_id for status polling.
        """
        if not self.is_available():
            return CloudTrainResult(
                task_id="", model_id="", status="failed",
                error_message="Cloud client not configured",
            )

        # Truncate if configured
        max_rows = self.config.storage.max_upload_rows
        if max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)

        payload: Dict[str, Any] = {
            "data_csv": csv_buf.getvalue(),
            "model_backend": model_backend,
            "compute_mode": compute_mode,
        }
        if torch_cfg:
            payload["torch_cfg"] = torch_cfg

        try:
            status, data = self._request("/train", method="POST", body=payload)
            if status in (200, 201, 202):
                return CloudTrainResult(
                    task_id=str(data.get("task_id", "")),
                    model_id=str(data.get("model_id", "")),
                    status=str(data.get("status", "pending")),
                    metrics=dict(data.get("metrics", {})),
                    model_backend=str(data.get("model_backend", model_backend)),
                    sample_count=int(data.get("sample_count", len(df))),
                    raw_response=data,
                )
            return CloudTrainResult(
                task_id="", model_id="", status="failed",
                error_message=f"HTTP {status}: {json.dumps(data, ensure_ascii=False)[:500]}",
                raw_response=data,
            )
        except Exception as exc:
            return CloudTrainResult(
                task_id="", model_id="", status="failed",
                error_message=str(exc),
            )

    def poll_train_status(self, task_id: str) -> CloudTrainResult:
        """Poll the status of a previously submitted training task."""
        if not self.is_available():
            return CloudTrainResult(
                task_id=task_id, model_id="", status="failed",
                error_message="Cloud client not configured",
            )

        try:
            status, data = self._request(f"/train/{task_id}/status")
            return CloudTrainResult(
                task_id=task_id,
                model_id=str(data.get("model_id", "")),
                status=str(data.get("status", "unknown")),
                metrics=dict(data.get("metrics", {})),
                model_backend=str(data.get("model_backend", "")),
                sample_count=int(data.get("sample_count", 0)),
                error_message=str(data.get("error_message", "")),
                raw_response=data,
            )
        except Exception as exc:
            return CloudTrainResult(
                task_id=task_id, model_id="", status="failed",
                error_message=str(exc),
            )

    def wait_for_train(
        self,
        task_id: str,
        *,
        poll_interval: float = 3.0,
        max_wait: float = 600.0,
    ) -> CloudTrainResult:
        """Block until the training task completes or times out."""
        t0 = time.monotonic()
        while True:
            result = self.poll_train_status(task_id)
            if result.status in ("completed", "failed"):
                return result
            elapsed = time.monotonic() - t0
            if elapsed > max_wait:
                result.status = "failed"
                result.error_message = f"Timed out after {max_wait}s"
                return result
            time.sleep(poll_interval)

    # -- Public API: Prediction ----------------------------------------------

    def submit_predict(
        self,
        model_id: str,
        df: pd.DataFrame,
        *,
        sensitivity_sample_idx: int = 0,
    ) -> CloudPredictResult:
        """Submit a prediction job to the cloud server."""
        if not self.is_available():
            return CloudPredictResult(
                task_id="", status="failed",
                error_message="Cloud client not configured",
            )

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)

        payload: Dict[str, Any] = {
            "model_id": model_id,
            "data_csv": csv_buf.getvalue(),
            "sensitivity_sample_idx": sensitivity_sample_idx,
        }

        try:
            status, data = self._request("/predict", method="POST", body=payload)
            if status in (200, 201, 202):
                preds = None
                if "predictions" in data:
                    preds = np.asarray(data["predictions"], dtype=np.float32).reshape(-1)

                return CloudPredictResult(
                    task_id=str(data.get("task_id", "")),
                    status=str(data.get("status", "pending")),
                    predictions=preds,
                    sensitivity=data.get("sensitivity"),
                    raw_response=data,
                )
            return CloudPredictResult(
                task_id="", status="failed",
                error_message=f"HTTP {status}: {json.dumps(data, ensure_ascii=False)[:500]}",
                raw_response=data,
            )
        except Exception as exc:
            return CloudPredictResult(
                task_id="", status="failed",
                error_message=str(exc),
            )

    def poll_predict_status(self, task_id: str) -> CloudPredictResult:
        """Poll the status of a previously submitted prediction task."""
        if not self.is_available():
            return CloudPredictResult(
                task_id=task_id, status="failed",
                error_message="Cloud client not configured",
            )

        try:
            status, data = self._request(f"/predict/{task_id}/status")
            preds = None
            if "predictions" in data:
                preds = np.asarray(data["predictions"], dtype=np.float32).reshape(-1)

            return CloudPredictResult(
                task_id=task_id,
                status=str(data.get("status", "unknown")),
                predictions=preds,
                sensitivity=data.get("sensitivity"),
                error_message=str(data.get("error_message", "")),
                raw_response=data,
            )
        except Exception as exc:
            return CloudPredictResult(
                task_id=task_id, status="failed",
                error_message=str(exc),
            )

    def wait_for_predict(
        self,
        task_id: str,
        *,
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> CloudPredictResult:
        """Block until the prediction task completes or times out."""
        t0 = time.monotonic()
        while True:
            result = self.poll_predict_status(task_id)
            if result.status in ("completed", "failed"):
                return result
            elapsed = time.monotonic() - t0
            if elapsed > max_wait:
                result.status = "failed"
                result.error_message = f"Timed out after {max_wait}s"
                return result
            time.sleep(poll_interval)

    # -- Public API: Model management ----------------------------------------

    def upload_model(self, model_bytes: bytes, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Upload a serialized model to the cloud server."""
        if not self.is_available():
            return {"status": "failed", "error": "Cloud client not configured"}

        mid = model_id or f"upload_{uuid.uuid4().hex[:8]}"
        try:
            status, data = self._request_binary(
                f"/model/{mid}/upload",
                method="POST",
                body=model_bytes,
            )
            resp = json.loads(data.decode("utf-8")) if data else {}
            resp["http_status"] = status
            return resp
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    def download_model(self, model_id: str) -> Optional[bytes]:
        """Download a model from the cloud server. Returns raw bytes or None."""
        if not self.is_available():
            return None

        try:
            status, data = self._request_binary(f"/model/{model_id}/download")
            if status == 200:
                return data
            return None
        except Exception:
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models on the cloud server."""
        if not self.is_available():
            return []

        try:
            status, data = self._request("/models")
            if status == 200:
                return list(data.get("models", []))
            return []
        except Exception:
            return []

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the cloud server."""
        if not self.is_available():
            return False

        try:
            status, _ = self._request(f"/model/{model_id}", method="DELETE")
            return status in (200, 204)
        except Exception:
            return False

    # -- Public API: Convenience wrappers ------------------------------------

    def train_sync(
        self,
        df: pd.DataFrame,
        *,
        model_backend: str = "hgb",
        compute_mode: str = "auto",
        torch_cfg: Optional[Dict[str, Any]] = None,
        poll_interval: float = 3.0,
        max_wait: float = 600.0,
    ) -> CloudTrainResult:
        """Submit a training job and wait for completion."""
        result = self.submit_train(df, model_backend=model_backend, compute_mode=compute_mode, torch_cfg=torch_cfg)
        if result.status == "failed":
            return result
        return self.wait_for_train(result.task_id, poll_interval=poll_interval, max_wait=max_wait)

    def predict_sync(
        self,
        model_id: str,
        df: pd.DataFrame,
        *,
        sensitivity_sample_idx: int = 0,
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> CloudPredictResult:
        """Submit a prediction job and wait for completion."""
        result = self.submit_predict(model_id, df, sensitivity_sample_idx=sensitivity_sample_idx)
        if result.status == "failed":
            return result
        return self.wait_for_predict(result.task_id, poll_interval=poll_interval, max_wait=max_wait)
