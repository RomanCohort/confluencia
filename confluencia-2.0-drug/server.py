"""Confluencia 2.0 Drug - Cloud API Server

Starts a FastAPI server exposing the core drug computation,
molecular evolution, and clinical trial simulation as REST API endpoints.

Usage:
    python server.py                          # default host=0.0.0.0, port=8000
    python server.py --host 127.0.0.1 --port 8080
    python server.py --reload                 # development mode with auto-reload
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Any, Dict

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import drug, evolution, trial, model
from api.schemas import HealthResponse

# ---------------------------------------------------------------------------
# Shared in-memory model store: model_id -> DrugTrainedModel
# ---------------------------------------------------------------------------
MODEL_STORE: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Confluencia 2.0 Drug API",
    version="2.0",
    description=(
        "Cloud server interface for drug efficacy prediction, "
        "molecular evolution, and clinical trial simulation. "
        "Part of the Confluencia 2.0 IGEM integration platform."
    ),
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(drug.router,      prefix="/api/drug",      tags=["Drug Training & Prediction"])
app.include_router(evolution.router,  prefix="/api/evolution", tags=["Molecular Evolution"])
app.include_router(trial.router,      prefix="/api/trial",     tags=["Clinical Trial Simulation"])
app.include_router(model.router,      prefix="/api/model",     tags=["Model Management"])


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="2.0")


@app.get("/api/info")
def server_info():
    """Return server information including available models."""
    return {
        "name": "Confluencia 2.0 Drug API",
        "version": "2.0",
        "stored_models": len(MODEL_STORE),
        "model_ids": list(MODEL_STORE.keys()),
        "registered_functions": [
            "default_dlt_prob",
            "default_efficacy_fn",
            "default_survival_fn",
        ],
        "endpoints": {
            "drug": ["/api/drug/train-and-predict", "/api/drug/train", "/api/drug/predict"],
            "model": ["/api/model/export", "/api/model/import", "/api/model/list", "/api/model/{model_id}/metadata", "/api/model/{model_id}"],
            "evolution": ["/api/evolution/molecules", "/api/evolution/cirrna"],
            "trial": ["/api/trial/cohort", "/api/trial/phase-i", "/api/trial/phase-ii", "/api/trial/phase-iii", "/api/trial/full-pipeline", "/api/trial/report"],
        },
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Confluencia 2.0 Drug Cloud API Server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Confluencia 2.0 Drug API Server")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Swagger docs:  http://{args.host}:{args.port}/docs")
    print(f"  Health check:  http://{args.host}:{args.port}/api/health")
    print()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
