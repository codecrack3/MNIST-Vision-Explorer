"""Image validation and preprocessing for MNIST inputs."""

from __future__ import annotations

import numpy as np
import torch

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def validate_image_array(image: np.ndarray) -> np.ndarray:
    """Validate that an image is 28x28 grayscale in the [0, 1] range."""
    array = np.asarray(image, dtype=np.float32)

    if array.shape != (28, 28):
        raise ValueError("Expected a 28x28 image array.")

    if not np.isfinite(array).all():
        raise ValueError("Image array contains non-finite values.")

    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("Image array values must be in the [0, 1] range.")

    return array


def image_list_to_array(image: list[list[float]]) -> np.ndarray:
    """Convert a JSON-friendly image list to a validated numpy array."""
    return validate_image_array(np.asarray(image, dtype=np.float32))


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Normalize and convert a 28x28 image into a model-ready tensor."""
    array = validate_image_array(image)
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return (tensor - MNIST_MEAN) / MNIST_STD
