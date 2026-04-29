"""Molecular evolution API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    EvolveCirrnaRequest,
    EvolveCirrnaResponse,
    EvolveMoleculesRequest,
    EvolveMoleculesResponse,
)

router = APIRouter()


@router.post("/molecules", response_model=EvolveMoleculesResponse)
def evolve_molecules(request: EvolveMoleculesRequest):
    """Run small-molecule evolution with RL + ED2Mol."""
    from api.slots import LocalEvolutionSlot
    slot = LocalEvolutionSlot()
    try:
        return slot.evolve_molecules(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cirrna", response_model=EvolveCirrnaResponse)
def evolve_cirrna(request: EvolveCirrnaRequest):
    """Run circRNA sequence evolution."""
    from api.slots import LocalEvolutionSlot
    slot = LocalEvolutionSlot()
    try:
        return slot.evolve_cirrna(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
