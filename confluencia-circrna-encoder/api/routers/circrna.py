"""
circRNA prediction API router.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional

router = APIRouter(prefix="/circrna", tags=["circrna"])


class PredictRequest(BaseModel):
    sequence: str
    gene_expr: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    immunotherapy_score: float
    tumor_killing_index: float
    overall_immunogenicity: float
    ips: float
    predicted_response: str
    composite_score: float


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict circRNA immunogenicity."""
    # Mock prediction for now
    return PredictResponse(
        immunotherapy_score=0.5,
        tumor_killing_index=0.4,
        overall_immunogenicity=0.45,
        ips=4.5,
        predicted_response="intermediate",
        composite_score=0.42,
    )


@router.post("/batch")
async def batch_predict(sequences: List[str]):
    """Batch prediction."""
    return {"results": [{"sequence_id": i, "score": 0.5} for i, s in enumerate(sequences)]}