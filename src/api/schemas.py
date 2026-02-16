"""Pydantic schema definitions for API routes."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class ImagePayload(BaseModel):
    image: list[list[float]]

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 28:
            raise ValueError("Image must contain 28 rows.")

        for row_index, row in enumerate(value):
            if len(row) != 28:
                raise ValueError(f"Row {row_index} must contain 28 values.")
            for pixel in row:
                if not math.isfinite(pixel):
                    raise ValueError("Image contains non-finite pixel values.")
                if pixel < 0.0 or pixel > 1.0:
                    raise ValueError("Pixel values must be in [0, 1].")

        return value


class PredictionResponse(BaseModel):
    predicted_class: int = Field(ge=0, le=9)
    probabilities: list[float] = Field(min_length=10, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_version: str


class TrainRequest(BaseModel):
    epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=128, ge=16, le=512)
    lr: float = Field(default=1e-3, gt=0.0, le=1.0)
    force_restart: bool = False


class TrainResponse(BaseModel):
    accepted: bool
    job_id: str
    state: str


class ModelStatusResponse(BaseModel):
    state: str
    epoch: int
    total_epochs: int
    train_loss: float | None
    val_accuracy: float | None
    last_error: str | None
    weights_path: str
    updated_at: str
    model_version: str
