"""Filter visualization payload builders."""

from __future__ import annotations

import numpy as np

from src.model.network import ConvNet


def _array_payload(array: np.ndarray) -> dict[str, object]:
    array = array.astype(np.float32)
    rounded = np.round(array, 5)
    return {
        "shape": list(array.shape),
        "data": rounded.tolist(),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def get_filter_payload(model: ConvNet) -> dict[str, object]:
    """Build filter payloads for conv layers."""
    conv1 = model.conv1.weight.detach().cpu().numpy().astype(np.float32)
    conv2 = model.conv2.weight.detach().cpu().numpy().astype(np.float32)
    conv2_summary = np.mean(np.abs(conv2), axis=1)

    return {
        "conv1": _array_payload(conv1),
        "conv2_summary": {
            "shape": list(conv2.shape),
            "reduced_shape": list(conv2_summary.shape),
            "channel_reduction": "mean_abs",
            "data": np.round(conv2_summary, 5).tolist(),
            "min": float(conv2_summary.min()),
            "max": float(conv2_summary.max()),
        },
    }
