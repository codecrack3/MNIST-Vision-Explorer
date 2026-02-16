"""Prediction and activation extraction helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.model.network import ConvNet
from src.preprocessing.transforms import preprocess_image


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    probabilities: list[float]
    confidence: float
    latency_ms: float
    model_version: str


@dataclass(frozen=True)
class ActivationResult:
    conv1: dict[str, Any]
    conv2: dict[str, Any]
    fc1: dict[str, Any]
    output_probs: list[float]


def _tensor_payload(tensor: torch.Tensor, *, squeeze_batch: bool = True) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy()
    if squeeze_batch and array.shape[0] == 1:
        array = array[0]

    array = array.astype(np.float32)
    rounded = np.round(array, 5)
    return {
        "shape": list(array.shape),
        "data": rounded.tolist(),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def predict(model: ConvNet, image_28x28: np.ndarray, *, model_version: str = "cnn-v1") -> PredictionResult:
    """Run inference for a single normalized 28x28 image."""
    model.eval()
    image_tensor = preprocess_image(image_28x28)

    started_at = time.perf_counter()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return PredictionResult(
        predicted_class=predicted_class,
        probabilities=[float(value) for value in probabilities.tolist()],
        confidence=confidence,
        latency_ms=float(latency_ms),
        model_version=model_version,
    )


def extract_activations(model: ConvNet, image_28x28: np.ndarray) -> ActivationResult:
    """Return intermediate activations and output probabilities."""
    model.eval()
    image_tensor = preprocess_image(image_28x28)

    with torch.no_grad():
        logits, activations = model.forward_with_activations(image_tensor)
        output_probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)

    return ActivationResult(
        conv1=_tensor_payload(activations["conv1"]),
        conv2=_tensor_payload(activations["conv2"]),
        fc1=_tensor_payload(activations["fc1"]),
        output_probs=[float(value) for value in output_probs.tolist()],
    )
