"""Prediction route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import ImagePayload, PredictionResponse
from src.preprocessing.transforms import image_list_to_array

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict_digit(payload: ImagePayload, request: Request) -> PredictionResponse:
    model_state = request.app.state.model_state
    image_array = image_list_to_array(payload.image)

    try:
        response = model_state.predict(image_array)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(**response)
