from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_model_status_untrained(tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    client = TestClient(app)

    response = client.get("/api/model/status")
    assert response.status_code == 200
    assert response.json()["state"] == "untrained"


def test_model_status_training_and_failed(tmp_path: Path) -> None:
    app = create_app(weights_path=tmp_path / "missing.pt")
    state = app.state.model_state
    client = TestClient(app)

    state.set_status_for_testing(
        state="training",
        epoch=1,
        total_epochs=3,
        train_loss=0.42,
        val_accuracy=0.86,
        last_error=None,
    )
    response_training = client.get("/api/model/status")
    assert response_training.status_code == 200
    assert response_training.json()["state"] == "training"

    state.set_status_for_testing(
        state="failed",
        epoch=0,
        total_epochs=3,
        train_loss=None,
        val_accuracy=None,
        last_error="disk full",
    )
    response_failed = client.get("/api/model/status")
    assert response_failed.status_code == 200
    assert response_failed.json()["state"] == "failed"
    assert response_failed.json()["last_error"] == "disk full"
