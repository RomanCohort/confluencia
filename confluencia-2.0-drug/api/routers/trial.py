"""Clinical trial simulation API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    CohortRequest,
    CohortResponse,
    FullTrialRequest,
    FullTrialResponse,
    PhaseIRequest,
    PhaseIResponse,
    PhaseIIRequest,
    PhaseIIResponse,
    PhaseIIIRequest,
    PhaseIIIResponse,
    TrialReportRequest,
    TrialReportResponse,
)

router = APIRouter()


def _get_trial_slot():
    from api.slots import LocalTrialSlot
    return LocalTrialSlot()


@router.post("/cohort", response_model=CohortResponse)
def generate_cohort(request: CohortRequest):
    """Generate a virtual patient cohort."""
    slot = _get_trial_slot()
    try:
        return slot.generate_cohort(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phase-i", response_model=PhaseIResponse)
def simulate_phase_i(request: PhaseIRequest):
    """Simulate Phase I dose escalation trial."""
    slot = _get_trial_slot()
    try:
        return slot.simulate_phase_i(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phase-ii", response_model=PhaseIIResponse)
def simulate_phase_ii(request: PhaseIIRequest):
    """Simulate Phase II efficacy trial."""
    slot = _get_trial_slot()
    try:
        return slot.simulate_phase_ii(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phase-iii", response_model=PhaseIIIResponse)
def simulate_phase_iii(request: PhaseIIIRequest):
    """Simulate Phase III comparative trial."""
    slot = _get_trial_slot()
    try:
        return slot.simulate_phase_iii(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full-pipeline", response_model=FullTrialResponse)
def full_trial_pipeline(request: FullTrialRequest):
    """Run complete Phase I/II/III trial pipeline."""
    slot = _get_trial_slot()
    try:
        return slot.full_pipeline(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report", response_model=TrialReportResponse)
def generate_report(request: TrialReportRequest):
    """Generate a CSR-style trial report."""
    slot = _get_trial_slot()
    try:
        return slot.generate_report(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
