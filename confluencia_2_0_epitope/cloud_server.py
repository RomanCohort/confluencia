"""
Confluencia 2.0 Epitope - Cloud Server

A FastAPI-based cloud server that exposes training, prediction, and model
management APIs. Deploy this on a cloud server with GPU support for remote
execution of Confluencia workloads.

Usage::

    uvicorn cloud_server:app --host 0.0.0.0 --port 8000

Environment variables::

    API_TOKEN         Required. Token for API authentication.
    DATA_DIR          Optional. Directory for persistent storage (default: ./cloud_data)
    MAX_UPLOAD_SIZE   Optional. Max upload size in MB (default: 100)

Endpoints::

    GET  /health                  Health check
    POST /train                   Submit training job
    GET  /train/{task_id}/status  Poll training status
    POST /predict                 Submit prediction job
    GET  /predict/{task_id}/status Poll prediction status
    GET  /models                  List available models
    POST /model/{model_id}/upload Upload model
    GET  /model/{model_id}/download Download model
    DELETE /model/{model_id}      Delete model
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional FastAPI import with graceful fallback
try:
    from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None
    HTTPException = None
    Header = None
    Request = None
    BackgroundTasks = None
    JSONResponse = None
    Response = None
    BaseModel = object

import numpy as np
import pandas as pd

from core.training import (
    EpitopeTrainingReport,
    build_artifacts_from_model,
    export_epitope_model_bytes,
    train_epitope_model,
    predict_epitope_model,
)
from core.torch_mamba import TorchMambaConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_TOKEN = os.environ.get("API_TOKEN", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "./cloud_data"))
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", "100")) * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "models").mkdir(exist_ok=True)
(DATA_DIR / "tasks").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# In-memory task store (for demo; replace with Redis/DB in production)
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    task_id: str
    task_type: str  # "train" or "predict"
    status: str  # "pending" | "running" | "completed" | "failed"
    created_at: str
    updated_at: str
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


_tasks: Dict[str, TaskInfo] = {}


# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------

if _HAS_FASTAPI:
    class TrainRequest(BaseModel):
        data_csv: str
        model_backend: str = "hgb"
        compute_mode: str = "auto"
        torch_cfg: Optional[Dict[str, Any]] = None

    class PredictRequest(BaseModel):
        model_id: str
        data_csv: str
        sensitivity_sample_idx: int = 0

    class TrainResponse(BaseModel):
        task_id: str
        model_id: str
        status: str
        sample_count: int
        model_backend: str
        metrics: Dict[str, float]

    class PredictResponse(BaseModel):
        task_id: str
        status: str
        predictions: Optional[List[float]] = None
        sensitivity: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Background task handlers
# ---------------------------------------------------------------------------

def _run_training_task(task_id: str, model_id: str, request: "TrainRequest"):
    """Background handler for training jobs."""
    task = _tasks.get(task_id)
    if not task:
        return

    task.status = "running"
    task.updated_at = datetime.now().isoformat()

    try:
        # Parse CSV
        df = pd.read_csv(io.StringIO(request.data_csv))

        # Build torch config if provided
        torch_cfg = None
        if request.torch_cfg:
            torch_cfg = TorchMambaConfig(**request.torch_cfg)

        # Run training
        model_bundle, report = train_epitope_model(
            df,
            compute_mode=request.compute_mode,
            model_backend=request.model_backend,
            torch_cfg=torch_cfg,
        )

        # Save model
        model_path = DATA_DIR / "models" / f"{model_id}.zip"
        model_bytes = export_epitope_model_bytes(model_bundle)
        model_path.write_bytes(model_bytes)

        # Update task
        task.status = "completed"
        task.result = {
            "model_id": model_id,
            "model_backend": model_bundle.model_backend,
            "sample_count": report.sample_count,
            "metrics": {
                "mae": float(report.metrics.get("mae", 0.0)),
                "rmse": float(report.metrics.get("rmse", 0.0)),
                "r2": float(report.metrics.get("r2", 0.0)),
            },
        }
        task.updated_at = datetime.now().isoformat()

    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        task.updated_at = datetime.now().isoformat()


def _run_prediction_task(task_id: str, request: "PredictRequest"):
    """Background handler for prediction jobs."""
    task = _tasks.get(task_id)
    if not task:
        return

    task.status = "running"
    task.updated_at = datetime.now().isoformat()

    try:
        # Load model
        model_path = DATA_DIR / "models" / f"{request.model_id}.zip"
        if not model_path.exists():
            raise ValueError(f"Model not found: {request.model_id}")

        from core.training import import_epitope_model_bytes
        model_bundle = import_epitope_model_bytes(model_path.read_bytes(), allow_unsafe=True)

        # Parse CSV
        df = pd.read_csv(io.StringIO(request.data_csv))

        # Run prediction
        result_df, sens = predict_epitope_model(
            model_bundle,
            df,
            sensitivity_sample_idx=request.sensitivity_sample_idx,
        )

        # Build response
        predictions = result_df["efficacy_pred"].tolist() if "efficacy_pred" in result_df.columns else []
        sensitivity_data = {
            "sample_index": sens.sample_index,
            "prediction": float(sens.prediction),
            "top_rows": sens.top_rows.head(20).to_dict(orient="records") if not sens.top_rows.empty else [],
            "neighborhood_rows": sens.neighborhood_rows.to_dict(orient="records") if not sens.neighborhood_rows.empty else [],
        }

        task.status = "completed"
        task.result = {
            "predictions": predictions,
            "sensitivity": sensitivity_data,
        }
        task.updated_at = datetime.now().isoformat()

    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        task.updated_at = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

import io

def create_app() -> "FastAPI":
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI is not installed. Run: pip install fastapi uvicorn")

    @asynccontextmanager
    async def lifespan(app):
        # Startup
        print(f"Confluencia Cloud Server starting...")
        print(f"Data directory: {DATA_DIR}")
        print(f"Max upload size: {MAX_UPLOAD_SIZE // (1024*1024)} MB")
        yield
        # Shutdown
        print("Confluencia Cloud Server shutting down...")

    app = FastAPI(
        title="Confluencia 2.0 Epitope Cloud API",
        description="Cloud API for training and prediction of epitope efficacy models",
        version="2.0.0",
        lifespan=lifespan,
    )

    # -- Auth middleware -----------------------------------------------------

    def verify_token(x_api_token: str = Header(default="", alias="X-API-Token")):
        if not API_TOKEN:
            return True  # No token configured, allow all
        if x_api_token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing API token")
        return True

    # -- Health check --------------------------------------------------------

    @app.get("/health")
    async def health():
        import platform
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except Exception:
            pass

        return {
            "status": "ok",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": gpu_available,
            "platform": platform.platform(),
            "message": "Confluencia Cloud Server is running",
        }

    # -- Training ------------------------------------------------------------

    @app.post("/train")
    async def submit_train(
        request: TrainRequest,
        background_tasks: BackgroundTasks,
        _: bool = Header(default="", alias="X-API-Token", include_in_schema=False),
    ):
        verify_token(_)
        task_id = uuid.uuid4().hex
        model_id = f"model_{task_id[:8]}"

        # Create task record
        _tasks[task_id] = TaskInfo(
            task_id=task_id,
            task_type="train",
            status="pending",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # Queue background job
        background_tasks.add_task(_run_training_task, task_id, model_id, request)

        # Quick parse for sample count
        try:
            df = pd.read_csv(io.StringIO(request.data_csv))
            sample_count = len(df)
        except Exception:
            sample_count = 0

        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "model_id": model_id,
                "status": "pending",
                "sample_count": sample_count,
                "model_backend": request.model_backend,
                "metrics": {},
            },
        )

    @app.get("/train/{task_id}/status")
    async def get_train_status(task_id: str, _: bool = Header(default="", alias="X-API-Token", include_in_schema=False)):
        verify_token(_)
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "task_id": task_id,
            "status": task.status,
            "error_message": task.error,
            **task.result,
        }

    # -- Prediction ----------------------------------------------------------

    @app.post("/predict")
    async def submit_predict(
        request: PredictRequest,
        background_tasks: BackgroundTasks,
        _: bool = Header(default="", alias="X-API-Token", include_in_schema=False),
    ):
        verify_token(_)
        task_id = uuid.uuid4().hex

        _tasks[task_id] = TaskInfo(
            task_id=task_id,
            task_type="predict",
            status="pending",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        background_tasks.add_task(_run_prediction_task, task_id, request)

        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "status": "pending",
            },
        )

    @app.get("/predict/{task_id}/status")
    async def get_predict_status(task_id: str, _: bool = Header(default="", alias="X-API-Token", include_in_schema=False)):
        verify_token(_)
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "task_id": task_id,
            "status": task.status,
            "error_message": task.error,
            **task.result,
        }

    # -- Model management ----------------------------------------------------

    @app.get("/models")
    async def list_models(_: bool = Header(default="", alias="X-API-Token", include_in_schema=False)):
        verify_token(_)
        models = []
        for p in (DATA_DIR / "models").glob("*.zip"):
            stat = p.stat()
            models.append({
                "model_id": p.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return {"models": models}

    @app.post("/model/{model_id}/upload")
    async def upload_model(
        model_id: str,
        request: Request,
        _: bool = Header(default="", alias="X-API-Token", include_in_schema=False),
    ):
        verify_token(_)
        body = await request.body()
        if len(body) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        model_path = DATA_DIR / "models" / f"{model_id}.zip"
        model_path.write_bytes(body)

        return {"status": "ok", "model_id": model_id, "size_bytes": len(body)}

    @app.get("/model/{model_id}/download")
    async def download_model(model_id: str, _: bool = Header(default="", alias="X-API-Token", include_in_schema=False)):
        verify_token(_)
        model_path = DATA_DIR / "models" / f"{model_id}.zip"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        return Response(
            content=model_path.read_bytes(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{model_id}.zip"'},
        )

    @app.delete("/model/{model_id}")
    async def delete_model(model_id: str, _: bool = Header(default="", alias="X-API-Token", include_in_schema=False)):
        verify_token(_)
        model_path = DATA_DIR / "models" / f"{model_id}.zip"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        model_path.unlink()
        return {"status": "ok", "model_id": model_id}

    return app


# Create app instance for uvicorn
app = create_app() if _HAS_FASTAPI else None


if __name__ == "__main__":
    import uvicorn
    if not _HAS_FASTAPI:
        print("Error: FastAPI is not installed. Run: pip install fastapi uvicorn")
        exit(1)

    if not API_TOKEN:
        print("Warning: API_TOKEN environment variable is not set. Authentication is disabled.")

    uvicorn.run(
        "cloud_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
