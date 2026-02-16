from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.state import StartTrainingResult


def test_train_endpoint_accepts_job(monkeypatch, tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    client = TestClient(app)

    def fake_start_training(*, epochs: int, batch_size: int, lr: float, force_restart: bool):
        assert epochs == 2
        assert batch_size == 64
        assert lr == 0.001
        assert force_restart is False
        return StartTrainingResult(accepted=True, job_id="local-123", state="training")

    monkeypatch.setattr(app.state.model_state, "start_training", fake_start_training)

    response = client.post(
        "/api/model/train",
        json={"epochs": 2, "batch_size": 64, "lr": 0.001, "force_restart": False},
    )

    assert response.status_code == 202
    assert response.json()["accepted"] is True


def test_train_endpoint_conflict_when_active(monkeypatch, tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    client = TestClient(app)

    def fake_start_training(*, epochs: int, batch_size: int, lr: float, force_restart: bool):
        return StartTrainingResult(accepted=False, reason="Training is already in progress.")

    monkeypatch.setattr(app.state.model_state, "start_training", fake_start_training)

    response = client.post(
        "/api/model/train",
        json={"epochs": 2, "batch_size": 64, "lr": 0.001, "force_restart": False},
    )

    assert response.status_code == 409


def test_train_endpoint_force_restart_path(monkeypatch, tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    client = TestClient(app)

    def fake_start_training(*, epochs: int, batch_size: int, lr: float, force_restart: bool):
        assert force_restart is True
        return StartTrainingResult(accepted=True, job_id="local-456", state="training")

    monkeypatch.setattr(app.state.model_state, "start_training", fake_start_training)

    response = client.post(
        "/api/model/train",
        json={"epochs": 3, "batch_size": 128, "lr": 0.001, "force_restart": True},
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == "local-456"
