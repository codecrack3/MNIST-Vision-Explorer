"""Weight summary and histogram generators."""

from __future__ import annotations

import numpy as np

from src.model.network import ConvNet


def get_weight_payload(model: ConvNet, *, bins: int = 30) -> dict[str, object]:
    """Return per-layer weight statistics and histogram data."""
    layers: list[dict[str, object]] = []

    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue

        values = param.detach().cpu().numpy().astype(np.float32).reshape(-1)
        counts, edges = np.histogram(values, bins=bins)

        layers.append(
            {
                "name": name,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "hist": {
                    "bins": np.round(edges, 5).tolist(),
                    "counts": counts.astype(int).tolist(),
                },
            }
        )

    return {"layers": layers}
