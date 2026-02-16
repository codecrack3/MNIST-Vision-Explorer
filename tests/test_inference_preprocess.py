import numpy as np
import pytest

from src.preprocessing.transforms import preprocess_image


def test_preprocess_image_shape_and_dtype() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    tensor = preprocess_image(image)

    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert str(tensor.dtype) == "torch.float32"


def test_preprocess_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="28x28"):
        preprocess_image(np.zeros((27, 28), dtype=np.float32))


def test_preprocess_rejects_values_outside_range() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[0, 0] = 1.2

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        preprocess_image(image)
