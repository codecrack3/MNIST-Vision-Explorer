"""Model status and training route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import ModelStatusResponse, TrainRequest, TrainResponse

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse)
def model_status(request: Request) -> ModelStatusResponse:
    model_state = request.app.state.model_state
    return ModelStatusResponse(**model_state.get_status())


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
def train_model_endpoint(payload: TrainRequest, request: Request) -> TrainResponse:
    model_state = request.app.state.model_state
    result = model_state.start_training(
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        lr=payload.lr,
        force_restart=payload.force_restart,
    )

    if not result.accepted:
        raise HTTPException(status_code=409, detail=result.reason)

    if result.job_id is None or result.state is None:
        raise HTTPException(status_code=500, detail="Training job accepted but no job metadata was set.")

    return TrainResponse(accepted=True, job_id=result.job_id, state=result.state)
