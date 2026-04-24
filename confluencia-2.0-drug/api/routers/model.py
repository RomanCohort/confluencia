"""Model management API router (export/import/list/delete)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.schemas import (
    ModelExportRequest,
    ModelImportResponse,
    ModelListResponse,
    ModelMetadataResponse,
)

router = APIRouter()


def _get_store():
    from server import MODEL_STORE
    return MODEL_STORE


@router.post("/export")
def model_export(request: ModelExportRequest):
    """Export a stored model as gzipped pickle bytes."""
    from core.training import export_drug_model_bytes
    store = _get_store()
    trained = store.get(request.model_id)
    if trained is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")
    try:
        payload = export_drug_model_bytes(trained)
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=model_{request.model_id[:8]}.cf2drug"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import", response_model=ModelImportResponse)
def model_import(payload: bytes):
    """Import a model from gzipped pickle bytes."""
    import uuid
    from core.training import import_drug_model_bytes
    store = _get_store()
    try:
        trained = import_drug_model_bytes(payload, allow_unsafe_deserialization=True)
        model_id = str(uuid.uuid4())
        store[model_id] = trained
        return ModelImportResponse(model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list", response_model=ModelListResponse)
def model_list():
    """List all stored models."""
    from core.training import get_drug_model_metadata
    store = _get_store()
    models = []
    for model_id, trained in store.items():
        meta = get_drug_model_metadata(trained)
        models.append({"model_id": model_id, "metadata": meta})
    return ModelListResponse(models=models)


@router.get("/{model_id}/metadata", response_model=ModelMetadataResponse)
def model_metadata(model_id: str):
    """Get metadata for a stored model."""
    from core.training import get_drug_model_metadata
    store = _get_store()
    trained = store.get(model_id)
    if trained is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    meta = get_drug_model_metadata(trained)
    return ModelMetadataResponse(model_id=model_id, metadata=meta)


@router.delete("/{model_id}")
def model_delete(model_id: str):
    """Delete a stored model."""
    store = _get_store()
    if model_id not in store:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    del store[model_id]
    return {"ok": True}
