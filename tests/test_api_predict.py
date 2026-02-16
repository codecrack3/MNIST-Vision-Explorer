from pathlib import Path

import torch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.model.network import ConvNet


def _blank_image() -> list[list[float]]:
    return [[0.0 for _ in range(28)] for _ in range(28)]


def _make_app_with_weights(tmp_path: Path):
    weights_path = tmp_path / "mnist_cnn.pt"
    model = ConvNet()
    torch.save(model.state_dict(), weights_path)
    return create_app(weights_path=weights_path)


def test_predict_returns_probabilities(tmp_path: Path) -> None:
    app = _make_app_with_weights(tmp_path)
    client = TestClient(app)

    response = client.post("/api/predict", json={"image": _blank_image()})
    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["predicted_class"] <= 9
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-5


def test_predict_rejects_bad_shape(tmp_path: Path) -> None:
    app = _make_app_with_weights(tmp_path)
    client = TestClient(app)

    bad_image = [[0.0 for _ in range(27)] for _ in range(28)]
    response = client.post("/api/predict", json={"image": bad_image})
    assert response.status_code == 422


def test_predict_returns_503_when_untrained(tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    client = TestClient(app)

    response = client.post("/api/predict", json={"image": _blank_image()})
    assert response.status_code == 503
