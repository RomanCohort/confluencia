"""Drug training/prediction API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    DrugPredictRequest,
    DrugPredictResponse,
    DrugTrainPredictRequest,
    DrugTrainPredictResponse,
    DrugTrainRequest,
    DrugTrainResponse,
)

router = APIRouter()


def _get_store():
    """Lazy import of the shared model store from server.py."""
    from server import MODEL_STORE
    return MODEL_STORE


def _get_drug_slot():
    """Create a LocalDrugSlot using the shared model store."""
    from api.slots import LocalDrugSlot
    return LocalDrugSlot(model_store=_get_store())


@router.post("/train-and-predict", response_model=DrugTrainPredictResponse)
def drug_train_and_predict(request: DrugTrainPredictRequest):
    """Train model and predict in one pass."""
    slot = _get_drug_slot()
    try:
        return slot.train_and_predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=DrugTrainResponse)
def drug_train(request: DrugTrainRequest):
    """Train model and store in server memory."""
    slot = _get_drug_slot()
    try:
        return slot.train(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=DrugPredictResponse)
def drug_predict(request: DrugPredictRequest):
    """Predict using a stored model."""
    slot = _get_drug_slot()
    try:
        return slot.predict(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
