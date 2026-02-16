"""Visualization route handlers."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import ImagePayload
from src.preprocessing.transforms import image_list_to_array

router = APIRouter(prefix="/api", tags=["visualization"])


@router.post("/activations")
def activations(payload: ImagePayload, request: Request) -> dict[str, Any]:
    model_state = request.app.state.model_state
    image_array = image_list_to_array(payload.image)

    try:
        return model_state.activations(image_array)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/filters")
def filters(request: Request) -> dict[str, Any]:
    model_state = request.app.state.model_state

    try:
        return model_state.filters()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/weights")
def weights(request: Request) -> dict[str, Any]:
    model_state = request.app.state.model_state

    try:
        return model_state.weights()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
